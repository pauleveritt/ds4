# SwiftStar: a Swift desktop application over the DwarfStar engine

This note records research performed in July 2026 into Apple's third-generation
foundation models, Core AI, and how both relate to DwarfStar, with particular attention
to small context windows and local inference.

## Current direction

The note began as research into driving Apple's models from the Pi coding agent. That
is no longer the goal. **SwiftStar is now a macOS desktop application that treats
`ds4_agent` as a library**: the DS4 engine embedded in-process, the terminal UI replaced
by a native GUI, and additional capability added on the Swift side. The GUI replaces the
TUI rather than shipping alongside it, so `ds4-agent` stops being a supported client and
the refactor can be aggressive. A second commitment sits alongside the macOS one:
SwiftStar is part of a Python-community effort to build AI tooling on Python's own
terms, so agent behavior lives in Python while the machinery stays native — see "Swift
body, Python brain".

The Pi and Foundation Models material below is retained because two parts of it still
apply directly. The small-context discipline — bounded tool observations, a replaceable
working set, a durable ledger — is a property of running a local model, not of Pi. And
the deterministic-controller and specialist-agent designs are the intellectual basis for
the subagent model described later in this note. The specific claim that Pi should reach
Swift over localhost HTTP is now moot.

The short version is:

- Apple's Foundation Models framework is a high-level Swift API for Apple's own
  on-device and Private Cloud Compute models. It is not an HTTP service.
- Core AI is a lower-level compiler and inference runtime for models supplied by an
  application. A compatible PyTorch model can be converted to `.aimodel`, specialized
  for the current Apple device, and run through Core AI.
- A converted model uses Apple's Core AI compiler/runtime, but it does not become an
  Apple Foundation Model and does not acquire AFM 3's training, system integration, or
  flash-backed sparse-expert architecture.
- `.aimodel` can describe a model with a larger context window. The format does not
  impose the Foundation Models system model's context limit. The usable window still
  depends on the original model, exported graph, positional encoding, KV cache,
  memory, and attention cost.
- A desktop application should embed the DS4 engine directly. The engine is already a
  library in all but build-target name.
- Multiple concurrent DS4 sessions are cheap. Model weights live in the engine and are
  shared; a bounded-context session costs a few hundred MiB of KV cache.
- The throughput win from parallel subagents comes from overlapping non-GPU work —
  containers, test runs, builds — with decode, not from batched GPU decode.
- Apple silicon has three compute engines; SwiftStar routes harness roles across all of
  them by joules and latency tolerance: AFM and embeddings on the ANE, a specialized
  implementer model and Laguna on the GPU. The router is deterministic — rules first,
  model judgment last. PCC is demoted to an experiment.
- The engine's public logits API makes grammar-constrained tool calls possible for
  Laguna: malformed tool calls become impossible by construction, implemented in Swift.
- APFS copy-on-write clones plus KV snapshots make agent turns transactional: the
  durable agent world forks in constant time at turn boundaries.
- Energy is a design axis. Pace-to-read decoding, watts-aware routing, and deferred
  prefill keep the fan silent while the user reads.
- The harness is a policy/mechanism split: the Swift body owns everything per-token and
  hardware-facing; the Python brain — a hot-reloadable, uv-managed peer process — owns
  agent logic, tools, and policy, and is testable against a fake body without a Mac.
- An 8K model can still be useful for coding with retrieval, bounded tool results,
  deterministic orchestration, and ephemeral specialist agents instead of a single
  accumulating conversation.

## The Apple AI layers

Apple now exposes two related but distinct developer stacks.

### Foundation Models

`FoundationModels` supplies `LanguageModelSession`, tools, guided generation, and
access to Apple's system models. The current developer preview describes:

- `SystemLanguageModel`: Apple's on-device model.
- `PrivateCloudComputeLanguageModel`: the stronger server model with a 32K-token
  context window and reasoning levels.

Switching a session to PCC is intentionally small at the API level:

```swift
let session = LanguageModelSession(
    model: PrivateCloudComputeLanguageModel()
)
```

PCC is not a public HTTP inference endpoint. It is reached through the Apple
framework. It requires a managed entitlement, is conditional on device and regional
availability, and has per-user daily quotas. A product that uses it needs a local
fallback and should not make PCC availability part of its essential control plane.

Foundation Models supplies much more than tensor execution: Apple owns the model,
tokenizer, prompting behavior, tool protocol, guided generation, session semantics,
availability checks, and PCC privacy boundary.

### Core AI

Core AI is the lower-level bring-your-own-model path. Its Python conversion package,
`coreai-torch`, consumes a `torch.export.ExportedProgram`, traverses its FX graph, and
maps supported ATen operations into Core AI operations. Unsupported operations need
decomposition, a custom lowering, or a custom Metal kernel.

Core AI has first-class composite operations relevant to language models, including:

- scaled dot-product attention, including MHA, GQA, and MQA;
- causal and sliding-window attention;
- RoPE with position IDs and decode offsets;
- RMSNorm;
- gather-matmul for mixture-of-experts models;
- mutable model state, including exported KV-cache buffers.

This makes conversion of a modern transformer plausible, but not necessarily easy.
The ability to convert a small example does not establish that a very large DeepSeek,
GLM, or Laguna graph can be converted, compiled, and run efficiently.

## What `.aimodel` means

An `.aimodel` is an unspecialized Core AI asset. It contains one or more inference
functions, their graphs, weights, state, tensor signatures, and metadata. Dynamic
tensor dimensions appear as runtime-supplied dimensions in its function signatures.

Loading the asset creates an `AIModel`. Core AI specializes the model for the current
device, chooses suitable CPU, GPU, and Neural Engine execution, and caches the
specialized result. Specialization may be expensive for a large model.

Core AI also has ahead-of-time compilation:

```sh
xcrun coreai-build compile MyModel.aimodel \
  --platform macOS \
  --min-deployment-version 27.0 \
  --output compiled/
```

This creates architecture-specific `.aimodelc` assets. It moves the most expensive
compiler work to the build machine, although some device specialization still occurs
when the asset is installed and loaded.

The benefits of the format and runtime are therefore:

- a first-party graph compiler for Apple silicon;
- hardware-specific specialization and reusable caches;
- optimized transformer composite operations;
- explicit mutable state suitable for a KV cache;
- dynamic input dimensions;
- separation between storage and compute representations;
- Xcode model inspection and Core AI debugging/profiling;
- bundling or downloading model assets without tying the application to PyTorch.

Conversion does not automatically quantize a model, make it sparse, extend its
context, or make unsupported operations disappear. Those remain model preparation and
bring-up work.

## Running and serving a converted model

Core AI is an in-process inference API, not a model server. A native host loads the
asset, loads its inference functions, constructs `NDArray` inputs, and runs them:

```swift
let model = try await AIModel(contentsOf: modelURL)
let prefill = try model.loadFunction(named: "prefill")
let decode = try model.loadFunction(named: "decode")

let outputs = try await decode.run(inputs: [
    "token": token,
    "kv_cache": cache,
    "position": position,
])
```

For an LLM, an application must still supply:

- tokenizer and chat template;
- prefill and decode scheduling;
- KV-cache allocation and lifetime;
- token sampling and stopping rules;
- streaming and cancellation;
- tool-call syntax and parsing;
- session isolation and admission control;
- an HTTP, Unix-socket, or XPC interface for non-Swift clients.

The natural Pi integration is consequently:

```text
Pi/Bun -> localhost HTTP -> Swift service -> Foundation Models or Core AI
```

The Swift service can present an OpenAI-compatible endpoint, but a smaller private
protocol is sufficient if Pi is its only client. Keeping the protocol provider-neutral
would allow the same Pi harness to compare AFM, a converted Core AI model, MLX, and
`ds4-server`.

Core AI also has a Python runtime for loading and executing `.aimodel` assets. Swift is
still the conservative production choice for a macOS service and for Foundation Models
or PCC access.

## What Core AI does and does not inherit from AFM 3

A converted model does use Apple's built-in Core AI inference stack. It can benefit
from the Core AI compiler, hardware placement, specialized operations, and caching.
It does not use the Apple Intelligence system-model service and does not inherit:

- AFM 3 weights, instruction tuning, reasoning, or coding ability;
- Apple's tokenizer, chat protocol, guided generation, or tool behavior;
- system adapters, safety behavior, or OS integration;
- automatic routing to Private Cloud Compute;
- the AFM 3 Core Advanced sparse-expert loading strategy.

AFM 3 Core Advanced is an architecture/runtime co-design. Most expert weights remain
in NAND. A small dense component makes a routing decision for a prompt, selected
experts are loaded and patched with always-resident shared experts, and routing is
updated only periodically during generation. Avoiding per-token expert swaps is what
makes flash bandwidth usable. Apple can also vary the active parameter count by use
case.

Nothing in the public `.aimodel` documentation promises that arbitrary converted
weights receive this treatment. `coreai-torch` uses the word *externalization* for
preserving a submodule as a named composite operation. It should not be confused with
externalizing its weights to SSD.

A sparse model might benefit from Core AI's gather-matmul and other compiler-recognized
operations. Reproducing AFM 3's prompt-level expert selection and NAND-to-DRAM policy
would still require the right model architecture and a matching exposed runtime. It is
not a file-format feature.

## Context windows with `.aimodel`

Core AI is a general tensor runtime and does not document one universal language-model
context limit. A custom model can therefore expose a larger context window than the
Apple system-model API, provided that the model and graph actually support it.

The practical limit is determined by:

- the context length used in training and evaluation;
- positional encoding and any validated RoPE scaling;
- exported static, bounded-dynamic, or dynamic sequence dimensions;
- KV-cache layout and capacity;
- unified-memory availability;
- prefill latency and attention complexity;
- the operations and shapes accepted by the Core AI compiler.

Converting an 8K model with a 32K input dimension does not make it a reliable 32K
model. A better experiment is to convert a model already trained for 32K or longer.

Long context primarily consumes KV-cache memory and prefill compute. Flash-backed
model weights do not solve either problem directly. Reducing resident weight memory
can leave more DRAM for KV state, but context still needs an appropriate model and
attention implementation. GQA/MQA, compressed KV, sliding-window attention, and
careful prefill scheduling are more directly relevant.

This distinction is especially important for DwarfStar. DS4 already has explicit
mmap-backed loading, compressed long-context state, disk KV checkpoints, and routed
expert SSD streaming. A generic `.aimodel` conversion should not be presumed to retain
those properties.

## Why a Swift/JavaScript bridge is not the main path

*Historical: this section addressed embedding Pi inside a Swift process. SwiftStar is a
native application with no JavaScript runtime, so the question no longer arises. The
conclusion is retained because it is the reason the note stopped pursuing that direction.*

JavaScriptCore can embed JavaScript in a Swift process, but it does not provide Bun or
Node compatibility. An unmodified Pi process expects modern modules and runtime APIs
such as filesystem access, subprocesses, environment variables, package resolution,
and Bun/Node behavior. Recreating that surface around JavaScriptCore would amount to
building and maintaining another JavaScript runtime.

Swift/JavaScript and WebAssembly interop projects solve useful but different problems;
they do not provide a mature way to embed an arbitrary Bun application inside a Swift
process. Bun itself does not expose a stable embedding API comparable to a library ABI.

A separate Swift inference service is consequently the simpler boundary. It isolates
developer-preview frameworks, can be restarted independently, allows Pi to remain a
normal Bun application, and makes alternative inference backends easy to benchmark.
The localhost transport cost is negligible compared with LLM inference.

## Pi with a small local context

Pi's default four-tool prompt is not itself prohibitively large. Measurement of Pi's
normal system prompt and `read`, `bash`, `edit`, and `write` schemas was approximately
1,309 tokens. The larger problems for an 8K model are accumulated history and tool
observations.

The current Pi defaults are inappropriate for an 8K model:

- compaction reserves 16,384 tokens;
- compaction tries to retain 20,000 recent tokens;
- branch summaries reserve 16,384 tokens;
- tool output may contain 2,000 lines or 50 KiB.

The compaction trigger is based on `contextWindow - reserveTokens`, so those defaults
make an 8K session attempt compaction immediately. A single permitted tool result can
also exceed the entire context.

A useful 8K packet is closer to:

| Component | Approximate tokens |
| --- | ---: |
| System prompt and enabled tools | 1,300 |
| User task and constraints | 400 |
| Ranked repository map | 700 |
| Durable task ledger | 500 |
| Active source excerpts | 1,600 |
| Latest bounded observation | 800 |
| Most recent exchange | 500 |
| Reserved generation | 2,000 |

This budget requires the harness to treat context as a replaceable working set rather
than a transcript. Search should precede reads, file reads should request narrow line
ranges, test output should retain a concise error summary and tail, and only one or two
files should normally be promoted into full context.

Relevant prior art supports this approach:

- Aider uses a dependency-ranked repository map with a default budget around 1K
  tokens and promotes only active edit files.
- SWE-agent shows that a purpose-built agent-computer interface materially changes
  coding performance.
- Agentless demonstrates the strength of a simple localization, repair, and
  validation pipeline.
- mini-SWE-agent uses a very small control surface and bounded observations, although
  its 10,000-character default observation remains too large for a strict 8K packet.

## Deterministic controller

A deterministic controller is a Bun/TypeScript state machine that owns the workflow.
The model still provides semantic judgments and code, but it does not freely decide
which phase, tool, or unbounded command comes next.

The controller performs a loop such as:

1. **Localize**: search an external repository index and ask the model to rank a small
   candidate set.
2. **Read**: validate the returned paths and load only the requested line ranges.
3. **Repair**: ask for structured edit operations or a patch.
4. **Apply**: validate paths and edit sizes, then apply the patch.
5. **Validate**: run a focused test chosen by policy, not arbitrary model exploration.
6. **Diagnose**: return only bounded failure evidence for at most a configured number
   of correction attempts.
7. **Finish**: verify the diff and tests and produce the final report.

Each model call can be a fresh session constructed from the current task packet. The
durable state lives in a compact ledger containing the objective, decisions, relevant
files, applied edits, test evidence, and next action. Invalid structured output is
retried against the same schema instead of becoming another long conversational turn.

This design is less autonomous but much easier to keep inside a hard context budget.
A stronger — or merely *different* — model can be invoked for ambiguous decomposition or
final review. In the current direction that reviewer is local: seed-diverse Laguna passes
or the AFM tier, not PCC (see "PCC is demoted" below).

## Ephemeral Pi specialists

Pi's example subagent extension is a useful starting point. It starts each specialist
as a separate `pi --no-session` subprocess, permits per-agent model and tool choices,
streams progress, and reports context usage. It supports chains and parallel tasks.

For an on-device model, the production design should differ in several ways:

- Specialists receive immutable, bounded task packets.
- Results use a small structured capsule: status, file/line evidence, recommendation,
  risks, and required next inputs.
- A chain must not interpolate the previous agent's full prose result.
- Read-only scouts can work concurrently, but repository writers are serialized or
  use isolated Git worktrees.
- A single inference semaphore should normally allow only one local AFM generation at
  a time. Multiple Pi processes do not create additional Neural Engine throughput,
  and multiple live KV caches consume memory.
- The durable orchestrator verifies cited code, patches, and tests instead of trusting
  specialist summaries.

A good hierarchy is:

```text
                         optional PCC 32K planner/reviewer
                                      |
user task -> deterministic Bun controller and artifact ledger
              |          |           |
          local scout  local worker  local test diagnostician
              \          |           /
                    single integrator
                           |
                     tests and final review
```

This hierarchy predates the current direction, which removes PCC from the picture
entirely (see "PCC is demoted" below). Everything else carries over to SwiftStar's
subagents intact: deterministic controller, bounded scouts and workers, a single
integrator, verification at the top.

## Implications for DwarfStar

Core AI is interesting to DS4 as an experiment and comparison, not an obvious
replacement.

DS4's central advantages are precisely the areas a generic conversion does not
promise:

- model-specific graph and memory policy;
- explicit routed-expert SSD streaming with overlapped I/O;
- mmap-backed model loading;
- aggressive asymmetric routed-expert quantization;
- compressed long-context caches;
- disk KV checkpoints and exact prompt/tool replay;
- integrated OpenAI, Anthropic, and Responses-compatible serving;
- deliberate support for a very small set of known model layouts.

Core AI could offer better compiler scheduling or Apple-silicon operation selection
for a model that converts cleanly. It might also offer a simpler first-party runtime
for a smaller dense or GQA coding model. It should be evaluated by measurement rather
than assumed to preserve DS4's large-model behavior.

The most useful SwiftStar experiments would be:

1. **Foundation Models spike**: wrap `SystemLanguageModel` behind the same internal
   Swift protocol SwiftStar uses for the embedded DS4 engine, so harness roles can route
   to either. PCC itself is demoted to an experiment (see "PCC is demoted"); the
   protocol matters because the AFM tier runs harness roles, not because of PCC.
2. **Core AI conversion spike**: convert the candidate implementer model (see "A
   specialized implementer model") — small, dense or simple-GQA, already trained for a
   32K context — with distinct prefill/decode functions and an explicit mutable KV
   cache, and confirm per-op ANE/GPU placement with Xcode's performance reports rather
   than assuming it.
3. **Runtime comparison**: compare Core AI, MLX, and an appropriate DS4 baseline for
   time to first token, decode rate, prefill rate, resident memory, model-load time,
   and correctness at 8K, 16K, and 32K.
4. **Storage tracing**: measure resident memory and filesystem reads during prefill and
   decode. Do not describe Core AI as SSD-streaming without evidence of weight
   residency and I/O behavior.
5. **Small-context harness**: evaluate deterministic localization/repair/validation
   and ephemeral specialists on representative one-file, two-file, test-failure, and
   repository-navigation tasks. Applies to SwiftStar's own orchestrator, not only to a
   third-party harness.
6. **GPU duty-cycle measurement**: instrument a single-agent session to record what
   fraction of wall-clock the GPU is idle awaiting tools, split by tool type. This is the
   number that determines whether subagents are worth building, and it can be measured on
   `ds4-agent` today, before any Swift work.
7. **Constrained-decoding spike**: prototype grammar-masked tool-call sampling over
   `ds4_session_copy_logits` / `ds4_session_set_logits` for the Laguna dialect and
   measure the per-token mask cost against decode throughput.
8. **Tier energy measurement**: run the same condensation/triage role on AFM (ANE) and
   Laguna (GPU) and measure joules per token and latency for each. The planning
   assumptions — roughly an order of magnitude better energy on the ANE at perhaps half
   the small-model speed — should be replaced with these numbers.
9. **Token-volume-by-role measurement**: from existing `ds4-agent` session transcripts,
   measure what fraction of prefilled tokens are tool results, the distribution of
   result sizes by tool, and how many of those tokens a conservative condensation would
   have kept out of the main context. Together with experiment 6 this sizes the payoff
   of the AFM tier before any Swift work.
10. **Fake-body spike**: define the brain-body protocol, implement the fake body in
    pure Python, and port one tainie-class tool onto it. This validates the
    community-development story — agent logic tested on Linux CI — before the Swift
    body exists.

An important strategic separation is:

- use Foundation Models when the goal is access to Apple's AFM behavior;
- use Core AI when the goal is Apple's native runtime for a model we choose;
- use DS4 when the goal is model-specific large-MoE inference, long context, explicit
  storage control, and a complete local server/agent stack.

## Embedding DS4 directly in Swift: engine architecture and concurrency

SwiftStar embeds the DS4 engine directly rather than wrapping `ds4-server` over HTTP.
The engine is already a library in all but build-target name; the agent and server are
consumers of the same public C API.

### Engine-as-library

The `ds4.h` header defines a deliberate public boundary. The opening comment reads
"Keep this header narrow so HTTP/CLI code does not depend on tensor internals."
Both `ds4_agent.c` (11,235-line coding agent) and `ds4_server.c` (17,898-line HTTP
server) have their own `main()` and call the same API:

```c
ds4_engine_open(&engine, &opts);           // load model into GPU memory
ds4_session_create(&session, engine, ctx); // allocate KV cache for one timeline
ds4_session_sync(session, &prompt, ...);   // bring KV state to a token prefix
ds4_session_eval(session, token, ...);     // advance one decode token
ds4_session_sample(session, temp, ...);    // sample from logits with temperature
```

A Swift app calls these directly through a bridging header (`#include "ds4.h"`) —
no FFI wrappers, no serialization, no subprocess. The C code is self-contained;
it never touches Swift-allocated memory.

### Agent logic: where to cut

You would not embed `ds4_agent.c` as-is. It has a hard-coded console UI (linenoise
editor, signal handlers, raw terminal I/O). But "rewrite the agent in Swift" is also
wrong, because it discards the parts that are hardest to get right.

The useful boundary does not run between C and Swift by layer. It runs between
**parsing and executing**, because those have different couplings:

- **Parsing is model-coupled, so it stays in C.** The three tool-call dialects (DSML
  for DeepSeek, `<tool_call>` for GLM, Laguna's variant), their streaming state
  machines, the guards that reject tool calls inside `<think></think>`
  (`ds4_agent.c:3498`, `:3894`, `:8699`), and the per-dialect prompt builders
  (`agent_build_tools_prompt`, `ds4_agent.c:1137`) change when a model changes.
  Reimplementing them in Swift is re-earning subtle bugs for no gain.
- **Executing is OS-coupled, so it moves to Swift.** `agent_tool_read`, `write`,
  `edit`, `list`, `search`, and the bash runner become `FileManager`, `Process`, and
  `NSRegularExpression`, with better error surfaces, native `async`, and access to
  container isolation.

The seam already exists. `agent_execute_tool_call(w, call) -> char *`
(`ds4_agent.c:7960`) takes one parsed call and returns a result string. Replacing its
body with a callback into Swift moves every long-running operation off the C worker
thread. That single refactor is the precondition for concurrent subagents.

What SwiftStar discards from `ds4_agent.c`: linenoise, the raw-mode line editor, signal
handling, the terminal visualizer, and the reserved-row bookkeeping. A GUI wants a
structured event stream instead, and `agent_tool_visualizer` (`ds4_agent.c:299`) already
decomposes rendering into approximately the right events — tool started, argument delta,
diff old/new, complete — so the event shape can be extracted rather than invented.
Per-session context gauges come free from `agent_status.ctx_used` and `ctx_size`
(`ds4_agent.c:99`).

Because the GUI replaces the TUI, the extracted parser library has exactly one
consumer. There is no need to preserve a callback API general enough for both.

### Concurrency: one GPU, many sessions

The engine is GPU-bound and session-stateful, but it is not limited to one session.

- **One session owns one KV cache**, backing Metal allocations for every layer.
- **The same session cannot be evaled concurrently.** The Metal graph mutates
  internal state on each `ds4_session_eval()` call.
- **Multiple live sessions are cheap.** Model weights live in the `ds4_engine` and are
  shared by every session over it. Per session you pay KV cache plus a small fixed
  scratch allocation — not another copy of the weights. An earlier version of this note
  claimed two sessions would exceed 128 GiB with a 45 GiB Laguna model; that
  double-counted the weights and was wrong. See the sizing table below.
- **`ds4_sessions_eval_batch()`** (`ds4.h:398`) advances independent sessions by one
  token each. It requires distinct sessions on the same engine and rejects duplicates
  (`ds4.c:62778`).
- **`ds4_sessions_eval_batch_with_prefill()`** advances one resumed prefill suffix and a
  decode batch as a single scheduling step, so a newly spawned session can prefill while
  its siblings decode.
- **The server's practical solution**: serialized bounded prefill, batched
  independent-session decode, job queue with pthread mutexes guarding inference,
  tool memory, and trace state. It also sets `share_session_prefill_workspace` whenever
  batching is on (`ds4_server.c:13143`), which routes every session through one shared
  prefill workspace on the engine (`ds4.c:35783`) instead of allocating per session.

#### Batched Metal decode is currently unavailable for Laguna

`ds4_sessions_eval_batch_metal_supported()` returns false when the model family is
Laguna (`ds4.c:61991`), and for any SSD-streaming session (`ds4.c:62011`). The call still
works — it falls back to a correctness-first sequential loop (`ds4.c:62809`) — but on
Laguna, concurrent sessions timeshare the GPU at roughly aggregate single-session
throughput. This is a real constraint on the subagent design, and the reason the
throughput argument in the next section rests on overlapping non-GPU work rather than on
batched decode.

The fallback also has a blast radius worth designing around: on an eval failure it calls
`ds4_session_invalidate()` on **every** batch member (`ds4.c:62813`), so one session's
failure forces all its siblings to rebuild KV state. Genuine subagent isolation needs
either batches of one on the fallback path, or an accepted sibling-rebuild cost.

### Safe embedding pattern

The natural Swift design serializes GPU work onto a single dispatch queue and
protects engine state behind an actor:

```swift
final class InferenceHost {
    private var engine: OpaquePointer?    // ds4_engine *
    private let queue = DispatchQueue(label: "inference")   // serial: one GPU

    func load(modelPath: String, contextSize: Int = 80000) throws {
        var opts = ds4_engine_options()
        opts.model_path = strdup(modelPath)   // engine must outlive this
        opts.backend = DS4_BACKEND_METAL
        opts.context_size = Int32(contextSize)
        guard ds4_engine_open(&engine, &opts) == 0 else {
            throw ModelError.engineOpenFailed
        }
    }

    func generate(session: OpaquePointer,
                  prompt: [Int32]) -> AsyncThrowingStream<Int32, Error> {
        AsyncThrowingStream { continuation in
            queue.async {
                var err = [CChar](repeating: 0, count: 256)
                var toks = prompt
                var tok = ds4_tokens(v: &toks, len: Int32(toks.count),
                                     cap: Int32(toks.count))

                // sync() is tri-state: 0 ok, DS4_SESSION_SYNC_INTERRUPTED (2)
                // when the cancel callback fired, anything else is an error.
                switch ds4_session_sync(session, &tok, &err, err.count) {
                case 0: break
                case 2: continuation.finish(); return       // user cancelled
                default: continuation.finish(throwing: ModelError.syncFailed); return
                }

                while true {
                    let token = ds4_session_argmax(session)
                    continuation.yield(token)
                    if ds4_token_is_stop(self.engine, token) { break }
                    guard ds4_session_eval(session, token, &err, err.count) == 0 else {
                        continuation.finish(throwing: ModelError.evalFailed)
                        return
                    }
                }
                continuation.finish()
            }
        }
    }
}
```

Why this is safe:

- The serial `queue` prevents concurrent GPU mutations — no race conditions. It is the
  single admission point for *all* sessions, which is what makes the subagent scheduler
  below tractable.
- `ds4_session_eval()` blocks the calling thread during Metal graph execution.
  Running it inside `queue.async` keeps the main thread responsive for UI.
- The engine's memory management is `xmalloc`/`free` on its own heap. No Swift
  ARC references cross into C, and no C pointers escape the host.
- Metal's own command queue already serializes GPU work internally; even without
  the dispatch queue the hardware would not see conflicting commands, only
  logically stale session state.

Note the tri-state return. `ds4_session_sync()` returns
`DS4_SESSION_SYNC_INTERRUPTED` (2) when the cooperative cancel callback stops it at a
safe boundary (`ds4.h:361`). Treating that as failure would turn every user-initiated
cancel into a spurious error, which matters constantly at an 80K context where prefill is
long enough to want cancelling.

### Practical concerns for a Mac app

**Load time**: `ds4_engine_open()` loads the full 45 GiB GGUF and warms the Metal
graph. This is tens of seconds and needs a splash screen with a progress callback. The
figure should be measured on the target machine rather than assumed; the same standard
this note applies to Core AI applies to DS4.

There are **two** progress callbacks, and the distinction matters. `ds4_session_set_progress`
reports durable checkpoint boundaries. `ds4_session_set_display_progress` may report
fine-grained progress *inside* a prefill chunk, and the header explicitly warns that
callers "must not treat it as a durable KV checkpoint boundary" (`ds4.h:338`). A smooth
GUI progress bar wants the display variant; anything that persists state wants the other.
`ds4_agent.c` sets both to the same function in three places.

`ds4_session_progress_fn` is a bare C function pointer
(`void (*)(void *ud, const char *event, int current, int total)`), so the Swift closure
must capture nothing and the controller has to arrive through `ud`:

```swift
ds4_session_set_display_progress(session, { ud, event, current, total in
    guard total > 0, let ud else { return }          // total can be 0
    let host = Unmanaged<ModelController>.fromOpaque(ud).takeUnretainedValue()
    let fraction = Double(current) / Double(total)
    Task { @MainActor in host.progress = fraction }
}, Unmanaged.passUnretained(self).toOpaque())
```

Both guards are load-bearing. A capturing closure will not convert to a C function
pointer, and an indeterminate event with `total == 0` produces a NaN that propagates into
SwiftUI layout.

**Memory residency**: Laguna S 2.1 Q2/Q3 at 80K context consumes roughly

| Component | Approximate |
|---|---:|
| Model weights (Q2_K/Q3_K GGUF), shared by all sessions | 45 GiB |
| KV cache, 12 full-attention layers at 80K | 3.7 GiB |
| KV cache, 36 SWA layers (512-window) | 0.07 GiB |
| Scratch and prefill workspace | 5–6 GiB, needs measuring |
| **Total** | **~54 GiB** |

On a 128 GiB Mac this leaves generous headroom. On a 96 GiB machine the smaller
Q2/Q3 Laguna still fits (~54 GiB used), but the Q4_K_M version (~68 GiB file,
~73 GiB resident) would be tight. The app should measure committed memory and
report free headroom before accepting a model.

The scratch-and-prefill row deserves scrutiny. Laguna's context estimator sets
`prefill_cap = 1` (`ds4.c:35277`), unlike the DeepSeek path, and its `scratch_bytes` term
(`ds4.c:35294`) has no context-size factor at all — it is a fixed handful of vectors,
single-digit MB. The large buffer is the prefill workspace, which is sized by prefill
chunk and is the thing `share_session_prefill_workspace` shares. Whether that recovers a
fixed 5 GiB on Laguna specifically has not been measured.

#### Per-session cost

This is the number that decides how many subagents are affordable. Laguna's KV formula
is explicit at `ds4.c:35274`. With the shape at `ds4.c:675` — 48 layers, 8 KV heads,
128 head dim, SWA window 512 — a KV row is `2 × 8 × 128 × 2 = 4 KiB` per token per layer.
Sliding-window layers cap at 512 tokens regardless of context; only full-attention layers
scale. Taking the 12-full / 36-SWA split, which follows from per-layer head counts in the
GGUF (`ds4_laguna_layer_is_swa`, `ds4.c:1141`):

| Session context | Full-attention KV | SWA KV (constant) | Total |
|---|---:|---:|---:|
| 4K | 192 MiB | 72 MiB | **264 MiB** |
| 8K | 384 MiB | 72 MiB | **456 MiB** |
| 16K | 768 MiB | 72 MiB | **840 MiB** |
| 32K | 1536 MiB | 72 MiB | **1.57 GiB** |
| 80K (main session) | 3.66 GiB | 72 MiB | **3.73 GiB** |

The 80K row reproduces the 3.7 GiB figure in the table above, so the formula and the
residency estimate agree. The architectural point is that three quarters of Laguna's
layers have a per-session KV cost that is **constant**, not proportional to context. Six
bounded 8K sessions cost about 2.7 GiB on top of the main session. KV cache is not what
limits parallelism.

**Fast task switching**: A full new `ds4_session_sync` from a cold prompt requires
bounded prefill (16K tokens at a time for Laguna). Tearing down and recreating
sessions is cheap compared with model load, so the app can cleanly switch between
chat threads, tool-calling loops, or agent tasks without reloading the engine.

### Comparison with the HTTP server path

| | Direct embedding | HTTP to `ds4-server` |
|---|---|---|
| GPU memory | Shared: one process | Separate: server process |
| Latency | Zero serialization | ~1ms localhost overhead |
| Swift-C boundary | Bridging header, direct calls | JSON encode/decode |
| Tool memory replay | Must implement in Swift | Server's `tool_memory` handles it |
| KV disk cache | Link `ds4_kvstore.c` | Server's `--kv-disk-dir` handles it |
| Session isolation | App's responsibility | Server's resident-slot model |
| Cancellation | `ds4_session_set_cancel()` callback | HTTP disconnect |
| Build system | `make ds4` + Swift Package Manager | Separate processes |

Direct embedding is the right choice for SwiftStar: it avoids a separate daemon, shares
GPU memory efficiently, and eliminates protocol overhead.

The cost is smaller than an earlier version of this note assumed. Disk KV checkpointing
does **not** need reimplementing — `ds4_kvstore.c` is already a standalone library with
its own header, which `ds4-agent` links against for exactly this, and the payload
primitives (`ds4_session_stage_payload`, `ds4_session_save_payload`,
`ds4_session_load_payload`) are public in `ds4.h`. SwiftStar links the existing C library.
Tool-memory replay does live inside `ds4_server.c` and would genuinely need
reimplementing, but only if SwiftStar wants the server's replay-verification semantics.

The HTTP path remains the better choice for any client that already speaks HTTP to
providers, since `ds4-server` handles batching, tool-memory verification, disk caching,
and OpenAI/Anthropic protocol compatibility. That is no longer SwiftStar's situation.

## Parallelism

This section records what parallelism in SwiftStar can and cannot buy, verified against
the engine source rather than inferred from the API surface. The short version: **the GPU
is serialized and will stay serialized, so the wins that matter are not throughput wins.**
Designs sold as "more tokens per second" are dead on arrival; designs that consume idle
time or shrink the critical path are not.

### The safety model: one serial queue, no exceptions

The constraint is stronger than "a session cannot be evaled concurrently with itself."
Sessions share *engine-level mutable GPU scratch*. When `share_session_prefill_workspace`
is set, every session's Metal graph is allocated against one `e->shared_prefill_workspace`
(`ds4.c:58858`):

```c
const ds4_gpu_graph *shared_prefill_workspace =
    e->share_session_prefill_workspace && e->shared_prefill_workspace_ready
        ? &e->shared_prefill_workspace : NULL;
```

Two threads prefilling different sessions concurrently would scribble on the same
workspace. This is why `ds4_server.c` guards every inference entry point with a single
`inference_mu` (`ds4_server.c:8473`) across roughly fourteen coarse lock sites, and why
the Laguna fallback in `ds4_sessions_eval_batch` is a plain sequential loop over
`ds4_session_eval` (`ds4.c:62806`).

So the Swift design follows: **N threads for tool execution and orchestration, one serial
dispatch queue as the sole admission point for eval.** Concurrency lives entirely outside
the engine. The serial queue in "Safe embedding pattern" above is not a convenience; it is
the correctness boundary.

The fallback path carries one further hazard: a single eval failure calls
`ds4_session_invalidate()` on **every** batch member (`ds4.c:62813`). Genuine subagent
isolation therefore wants batches of one, or an accepted sibling-rebuild cost.

### Snapshots are writable clones, not read-only transactions

It is tempting to read `ds4_session_save_snapshot` as an MVCC-style read-only view — a
snapshot of state that a worker can compute against without allocating KV. It is not.

`ds4_session_save_snapshot` (`ds4.c:52477`) calls `ds4_session_save_payload` into an
`fmemopen` buffer: it **serializes live KV out to host RAM**. `load_snapshot` deserializes
it back into the *target* session's own GPU buffers and calls `ds4_gpu_synchronize()`.
Consequences:

- The copy is **eager and full**. No pages are shared with the parent, and there is no
  copy-on-write anywhere in the path.
- The child owns a **writable** KV cache and appends to it normally. There is no
  compute-only mode; generation inherently writes KV rows.
- Nothing in the API exposes "share the parent's KV prefix read-only, append into a
  private overlay." Cross-session prefix sharing does not exist in DS4 — each session owns
  its allocation.

The correct analogy is `git clone` at a tag, or a restore-into-a-new-database — not
`BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ`.

The one genuinely cheap property is that payload size tracks **live checkpoint length**,
not `ctx_size`: `rows = min(cache_cap[layer], checkpoint_len)` (`ds4.c:49838`), at
`2 × 8 × 128 × 2 = 4 KiB` per row per layer. A 1,300-token template snapshot is therefore

| | rows | bytes |
|---|---:|---:|
| 12 full-attention layers | 1,300 | 61 MiB |
| 36 SWA layers (capped at 512) | 512 | 72 MiB |
| **template snapshot, host RAM** | | **~133 MiB** |

held once and reused for every spawn. Note it moves through stdio in 8 MiB chunks
(`DS4_SESSION_IO_CHUNK`, `ds4.c:49635`), so it is not a raw memcpy; treat load cost as
unmeasured rather than free.

### Threads do not duplicate KV; sessions do

Worth stating precisely, because the answers differ:

- **A thread** costs a stack. Ten threads over one session add zero KV.
- **A session** costs its own KV cache — 264 MiB at 4K, 456 MiB at 8K per the sizing table
  above. Each subagent is a session, so each subagent *does* allocate fresh KV.
- **Weights are never duplicated.** 45 GiB lives in the `ds4_engine` and is shared by every
  session. This is what makes the whole design viable.
- **The snapshot** is written *into* the child's already-allocated KV. It does not add a
  per-child copy beyond the one host-RAM template.

Six bounded 4K scouts cost roughly 1.6 GiB plus the 133 MiB template — noise against
45 GiB of weights. Memory is not the constraint on parallelism; per "Containers invert the
per-subagent memory budget" below, container memory is.

### Where the wins actually come from

Four candidate sources, only two of which are worth building against:

| Source | Real? | Ceiling |
|---|---|---|
| Parallel decode | **No** | Zero. GPU serialized; batched Metal decode off for Laguna (`ds4.c:61991`). |
| Filling GPU idle during long tools | Yes, bounded | Capped by idle fraction. A 70% decode / 30% tool loop caps near 1.4× regardless of subagent count. |
| Filling GPU idle during *user* time | Yes, large | The GPU is ~100% idle while the user reads and types. Unclaimed real estate. |
| **Reducing tokens on the critical path** | **Yes, unbounded** | Does not fight serialization — sidesteps it. |

The first three divide a fixed pie. The fourth shrinks it, and it is the one this note has
previously undersold.

### Reducing the critical path: context as a ratchet

`ds4_session_sync` evaluates only the *suffix* when the current checkpoint is a prefix of
the new prompt; otherwise "the backend state is refilled from scratch" (`ds4.h:357`). A
conversation that only ever **appends** is cheap forever. Anything that **rewrites
history** costs a full re-prefill.

That makes context a ratchet: tokens enter easily and leave only by paying. The economic
claim is therefore not that reading a large file is free — the scout still spends GPU
reading it — but that the bill is paid **O(N) once instead of O(N) every subsequent
turn**.

How this manifests from the user's chair:

**The log-dump cliff.** A test run emits 40K tokens. In a single-session agent those
tokens join the timeline permanently — not because anyone judged them important, but
because they were on screen when they arrived. Every later turn carries them. With a
bounded scout reading the log and returning a 200-token distillate, the user sees what
they actually wanted (*"3 failures, all in `test_router.py`, same assertion"*) and the raw
tokens never enter the main KV.

**The compaction stall — the sharpest one.** `/compact` (`ds4_agent.c:10989`) is
double-priced: the model first *generates* a summary, decoding over the entire bloated
context, and the agent then "rebuild[s] the live DS4 session" from it
(`ds4_agent.c:8144`) — a from-scratch re-prefill, because the prefix is gone. The user
experiences a multi-second-to-minute freeze, arriving unpredictably, usually mid-thought,
and pays fidelity for it: the agent returns having paraphrased its own memory. Firewalling
does not make compaction faster; it makes the session hover at 20K used instead of
climbing to 78K, so compaction stops being a recurring event. That is the difference a
user would actually notice.

**Sluggish decode — real but damped.** Longer KV means more attention work per token, so
the agent visibly types slower late in a session. Calibrate honestly: only 12 of Laguna's
48 layers are full-attention, and the other 36 are sliding-window capped at 512 rows with
*constant* cost regardless of context. Laguna degrades more gracefully here than a dense
model. Second-order next to the compaction stall.

**Rewinding stops being expensive.** "Ignore that last approach" currently means living
with the dead branch or triggering a rebuild (`DS4_SESSION_REWRITE_REBUILD_NEEDED`,
`ds4.h:352`). If the exploration happened in a subagent, undo is `ds4_session_free` and the
parent is bit-identical. Cheap undo is what makes a user willing to let the agent *try*
things, because a failed attempt no longer taxes the rest of the hour.

**What the GUI renders.** The context gauge (`ctx_used` / `ctx_size`,
`ds4_agent.c:99`) stops being a doom clock — today it climbs monotonically and the user
learns to read it as time-until-freeze. Tool results become collapsed chips holding the
distillate, expandable to raw output, where expanding is a UI action against a file on
disk rather than a context action. **Looking at something stops costing anything** — a
distinction a TUI cannot make, because in a terminal showing and remembering are the same
event.

**The honest cost.** The distillate is lossy and the parent never saw the original. If the
summary drops the decisive line, the parent cannot introspect its way back to it. The
design therefore needs cheap re-query (keep the scout warm briefly, or re-spawn — spawn is
cheap, which is the premise) and `file:line` provenance so the orchestrator verifies
against disk rather than trusting prose. The failure mode to test for before shipping is
an agent reasoning confidently over a summary that quietly omitted what mattered; that is
strictly worse than today's bloated-but-complete context.

### Ranked uses

Ordered by expected value, given that the GPU is fixed:

1. **Context firewalling.** Bounded scouts read large artifacts and return distillates.
   Needs no concurrency, no scheduler, no OS primitives — only template-spawn. The only
   item here that compounds. Build this first, alone if necessary.
2. **Speculative warming on user-idle time.** On save, prefill a session over the changed
   file and its dependency neighbourhood so time-to-first-token collapses when the question
   arrives (see "FSEvents" below). Cheap spawn is what makes being *wrong* acceptable: a
   wasted speculative session costs 264 MiB and cycles nobody was using.
3. **Disposable exploration.** Failed branches die with the session, leaving the parent
   untouched. This is the valuable half of "transactional" and it follows from session
   lifetime alone — no `clonefile()`, no fork taxonomy.
4. **Overlapping genuinely long tool calls.** Real but narrow: pays only when tool duration
   greatly exceeds decode, so builds, test suites, and containers — not `read`/`search`.
   One or two long-running validators, not N uniform workers.

Not worth building: N-best ensembles (N× serialized GPU for a quality guess), and anything
whose pitch is aggregate throughput.

### The lightweight design: template-spawn only

Items 1–3 above need nothing beyond a template snapshot and session lifetime. That is a
much smaller thing to build than "Transactional forks" below implies:

```
Orchestrator (Swift actor)
├── template: ds4_session synced to system prompt + tool schemas, snapshotted once
├── snapshot: ds4_session_snapshot (~133 MiB, host RAM, immutable)
├── queue:    DispatchQueue(label: "inference")   // serial — sole GPU admission
└── pool:     [Subagent]

spawn(role):
    s = ds4_session_create(engine, role.ctxSize)   // 4K scout, 32K validator
    ds4_session_load_snapshot(s, &snapshot)        // on `queue`
    ds4_session_sync(s, taskPacket)                // short, immutable, bounded
    → no inheritance, no fork, no filesystem clone

finish(subagent):
    ds4_session_free(s)                            // no merge, no diff
    → structured result; orchestrator verifies file:line against disk
```

What this drops relative to the fuller design: `clonefile()`, mid-task forking, the
forkable/leased/engine state taxonomy, turn-boundary fork barriers, and workspace
merge-by-diff. Isolation without APFS is a plain scratch directory per writer with scouts
held read-only: you lose O(1) rollback and gain not depending on an OS primitive for a
nice-to-have.

### Measure before building the scheduler

The duty-cycle scheduler in "Where the parallelism win actually is" is justified by a
number nobody has measured. The `os_signpost` instrumentation described under
"Instrumentation" below — intervals around prefill, decode, and tool spans — produces it.
**If GPU duty cycle is already above roughly 85%, the scheduler is worth about 1.15× and
should not be written.** Context firewalling needs no such justification, which is a
further argument for sequencing it first.

## Subagents in SwiftStar

`ds4_agent` has no subagent concept today. Its worker owns exactly one session and one
transcript. Adding subagents is therefore new design, not a port, and it is a
**nice-to-have for the initial GUI** rather than part of the first milestone. This section
records the shape it should take so that earlier decisions do not foreclose it.

### A subagent is a session

The natural mapping is **one subagent per `ds4_session`**. A single `ds4_engine` holds the
weights; each subagent is an independent KV timeline over those same weights
(`ds4.h:332`). Nothing in the engine assumes one session per process — `ds4_server.c`
already runs many. Three public primitives follow:

**Spawn from a template, not from cold.** `ds4_session_save_snapshot` and
`load_snapshot` (`ds4.h:383`) are in-memory KV copies sized by checkpoint length
(`ds4.c:52487`). Keep one template session prefilled with just the system prompt and tool
schemas — measured at roughly 1,300 tokens for Pi's four-tool prompt, and the same order
of magnitude for the DS4 agent's eleven — snapshot it once, and clone it into every
subagent. Startup becomes a memcpy plus a short task-packet prefill instead of
re-prefilling the preamble every time. Prefill dominates spawn cost, so this matters more
than it first appears. Note that `ds4_session_payload_bytes` returning 0 means there is no
valid checkpoint (`ds4.c:52488`), so the template must be synced before first use.

**Fork mid-task.** The same primitive against a live parent gives the child a KV cache
that already contains the conversation. Note that this is an *eager full copy*, not
copy-on-write — see "Snapshots are writable clones" above. Cheap at 8K and costly at 80K
(about 3.73 GiB per fork, paid twice: once serialized to host RAM and once restored into
the child), so it is a deliberate knob rather than a default — which is itself an argument
for immutable bounded task packets over inheritance.
Forking the *full* agent world — filesystem and harness state included — has additional
constraints; see "Transactional forks" under "The OS as harness substrate".

**Bounded context per role.** `ds4_session_create(&s, engine, ctx_size)` takes a
per-session context size. A read-only scout does not need the main session's 80K. This is
where the sizing table above pays off.

The orchestrator is then a Swift actor owning the template, a session pool, and the
durable ledger described earlier — including the verification role that design assigns it:
checking cited `file:line` evidence against disk rather than trusting a subagent's prose
summary.

### Where the parallelism win actually is

This subsection covers the *duty-cycle* win specifically; see "Parallelism" above for the
full accounting, including the larger critical-path win it does not address, and for the
measurement that should gate building the scheduler described here.

It is not batched GPU decode. On Laguna that path is disabled (`ds4.c:61991`), so
concurrent sessions timeshare one GPU.

The win is that **a serialized GPU does not serialize a turn.** A subagent's turn is
roughly `decode → tool exec → decode → …`, and during tool execution it holds no GPU at
all. The scheduler should therefore be **work-conserving over sessions in a decode-ready
state**, with sessions awaiting a tool simply absent from the batch:

| Session state | Holds GPU | In scheduler batch |
|---|---|---|
| Prefilling | yes | via `eval_batch_with_prefill` |
| Decode-ready | yes | yes |
| Awaiting tool | **no** | no |
| Done / failed | no | no |

The metric to optimise is **GPU duty cycle**, not aggregate tokens per second. A
single-agent loop leaves the GPU idle through every tool call; N subagents fill those gaps
with a sibling's decode. This requires no batched Metal decode and works on the
sequential fallback path, which is what makes it viable on Laguna today.

The size of the win tracks the tool-to-decode ratio, and that varies by an order of
magnitude across the tool set:

- `read`, `list`, `search`, `more` — sub-second against tens of tokens per second of
  decode. Overlap buys almost nothing.
- `bash` running a build, a test suite, or a container — seconds to minutes. Overlap buys
  nearly the entire tool duration.

So the payoff concentrates in bash-heavy roles (test diagnostician, validator) and is
close to nil for read-only scouts. That argues for an asymmetric pool — one or two
long-running validators whose tool time is effectively free, plus scouts that are
sequential anyway — rather than N uniform subagents. The heterogeneous-compute section
adds the orthogonal axis: scouts gain little from parallelism but the most from tier
routing, since their model work is trivial (see "What routes where"). The two
optimizations split the same pool the same way and compose cleanly: scouts route down a
tier, validators overlap their tool time.

**This ratio moves decisively in favour of parallelism as the deterministic side grows.**
Once SwiftStar adds container-isolated execution, a Python toolchain, and real test
running, the non-GPU portion of a turn stops being a `grep` and becomes a container start
plus a dependency resolution plus a test run. Every second of that is GPU idle time in a
single-agent design and hideable in a concurrent one. The deterministic-controller work
described earlier is not merely a context-budget discipline; it is what creates the
overlap that makes subagents worth building.

The non-blocking machinery already exists in C: `agent_bash_job` (`ds4_agent.c:7466`)
holds `pid`, `pipe_fd`, `running`, and `timeout_sec`, with poll and stop tools. What is
missing is a scheduler that does something with the idle GPU while a job runs.

### Intra-turn tool parallelism

A separate and much cheaper win, independent of subagents:
`agent_execute_tool_calls` loops **serially** over every call in one assistant turn
(`ds4_agent.c:8025`). When a model emits three reads in one turn they run one after
another. Executing independent calls in a turn concurrently needs no engine change, no
session pool, and no subagent concept — and once tool execution moves to Swift it is a
`TaskGroup`. This is probably the largest speedup available for the least work, and it
should land before any subagent work.

Calls within a turn are not always independent — a `write` followed by a `bash` that runs
the written file must stay ordered — so this needs a dependency rule, likely conservative:
parallelise read-only tools, serialise anything that mutates the filesystem.

## Container isolation

Apple's Containerization framework runs Linux containers on Apple silicon from Swift on
top of Virtualization.framework, with each container in its own lightweight VM. Relevant
properties for SwiftStar:

- **Sub-second start times**, via an optimised Linux kernel configuration and a minimal
  root filesystem with a lightweight init system.
- **`vminitd`**, a small init inside the VM exposing a gRPC API over vsock, which
  configures the runtime environment, launches containerised processes, and handles I/O,
  signals, and events. This maps closely onto what `agent_bash_job` already does over a
  pipe.
- **Explicit per-container resource allocation.** Apple's own examples pass
  `--cpus 8 --memory 8g`.
- **Rosetta 2** for `linux/amd64` images.
- An example named `sandboxy` whose stated purpose is running coding agents inside
  sandboxed containers, with options for workspace, network restrictions, mounts, and
  session management. This is close enough to SwiftStar's needs to be read carefully
  before designing anything.

The premise for planning is that this ships as Apple Containers 1.1 in macOS 27 in
September 2026. Version specifics, and the exact API surface, must be rechecked against
the installed SDK.

### Containers invert the per-subagent memory budget

Combining container isolation with the sizing table above produces the most important
practical constraint on parallel subagents, and it is the opposite of the intuitive one.

| Per-subagent cost | 8K session |
|---|---:|
| KV cache | 456 MiB |
| Fixed scratch | single-digit MB |
| Container VM (allocated) | 1–8 GiB |

The container dominates the KV cache by roughly 2× to 18×. **The binding constraint on
subagent count is container memory, not the model's KV cache.** Any admission-control
policy should be written against the container budget, with KV as a secondary term.

Two consequences:

- Not every subagent needs a container. Read-only scouts doing `read`, `search`, and
  `list` can run host-side against the workspace and cost only their KV cache. Reserve
  containers for subagents that execute code. This reinforces the asymmetric pool above:
  cheap scouts, expensive validators.
- Container start latency, though sub-second, is itself hideable by the same
  work-conserving scheduler. A subagent waiting on container boot holds no GPU.

Isolation is also the honest answer to a gap in the current agent. `ds4-agent` prompts for
confirmation only before starting Chrome for web tools (`ds4_agent.c:4476`); `bash`,
`write`, and `edit` execute unprompted. A GUI that fans work out to several subagents
makes that considerably more consequential than it is in a single-threaded TUI where a
human watches every call scroll past. Containers, plus a per-subagent workspace mount, are
a better answer than adding confirmation prompts to a parallel system.

### What runs where: whose code executes

The container boundary is not about language; it is about **whose code executes**. A
container protects against code nobody reviewed: the model's shell commands, its Python,
a test suite, a dependency install running arbitrary post-install hooks. A Swift `read`
implementation is first-party code — written, reviewed, compiled — and was never the
threat. What *is* untrusted about a first-party tool is its **arguments**: `write`
pointed at `~/.ssh/authorized_keys` is a trusted implementation with a malicious target.
That is a path-validation problem, not a containment problem, and a Linux VM is a wildly
indirect way to solve it.

| Runs host-side | Runs in the container |
|---|---|
| `read`, `more`, `list`, `search` | `bash` |
| `write`, `edit` (after path validation) | `python` (the kernel) |
| Trusted brain tools (see "Swift body, Python brain") | Test runs, builds, dependency installs |

Containerizing the host-side tools would be actively costly, not merely neutral:

- It buys no security, per the principle above.
- It requires a second build target: Containerization runs Linux guests, so Swift tools
  would need cross-compilation and a duplicate artifact of the same code.
- It discards the reason execution moved to Swift at all: FSEvents, security-scoped
  bookmarks, Spotlight metadata, `NLContextualEmbedding`, and unified logging do not
  exist inside a Linux guest.
- It flattens structured results into bytes over vsock, when the router, the condenser,
  and the GUI all want typed values.

What actually protects host-side tools is native and layered: path validation against
the workspace root (the real fix for the untrusted-argument problem); security-scoped
bookmarks, so a validation bug cannot reach beyond what the user granted; and, for file
mutations, an XPC helper whose sandbox profile is narrower than the main app's — a
distinct use of XPC from the engine-helper option in "Smaller native affordances" — so
even a validation bug cannot touch `~/.ssh` or `~/Library`.

Two coherence consequences. Host-side `edit` and container-side `bash` must see the same
bytes, so the workspace (or its APFS clone) mounts read-write into the container — which
means the container can mutate files outside the validated-path discipline. That is
precisely why FSEvents-as-evidence matters: the journal reports what the container
actually touched, rather than what it claims. And stateful tools stay host-side by the
fork taxonomy: the `read`/`more` pagination cursor is a forkable value, and keeping it in
Swift keeps containers stateless — freely discardable leases.

One edge case: `google_search` and `visit_page` do execute untrusted code — JavaScript,
in Chrome. Chrome brings its own sandbox and its own process; the untrusted *content* it
returns is a prompt-injection concern for the harness, not a code-execution concern for
the sandbox.

## Heterogeneous compute: route roles across the silicon

Everything above uses one of Apple silicon's three compute engines. Laguna saturates the
GPU; the ANE and the efficiency cores sit idle for the whole session. The harness should
treat this as a routing problem: every model-shaped role in the system gets a tier,
chosen by joules and latency tolerance, not only by capability.

| Tier | Silicon | Roles |
|---|---|---|
| AFM (`SystemLanguageModel`) | ANE | Tool-output condensation, triage, commit messages, ledger summaries, dissenting review opinions |
| Implementer model (next section) | GPU | Spec'd diffs, test writing, mechanical edits |
| Laguna via DS4 | GPU | Main loop, planning, review, integration |

Two clarifications keep this honest:

- **The ANE runs model work, not tool work.** Reading files, git operations, and search
  are ordinary CPU work and never touch the ANE. What routes to the ANE is the
  small-model inference *about* those results — condensing a long test log, classifying
  a search hit, embedding a file.
- **The ANE trades peak speed for joules.** Planning assumptions, to be measured:
  roughly an order of magnitude better energy per operation than the GPU, at perhaps
  half the like-for-like small-model speed. The routing policy makes the speed penalty
  invisible: ANE-tier roles are latency-tolerant, generate few tokens, and run
  concurrently with GPU decode on separate silicon, so in a work-conserving scheduler
  their wall-clock cost is approximately zero.

### What routes where: turns and results, not calls

A common observation about coding agents is that most tool calls are boring and do not
need a frontier model. The observation is right, but the unit is wrong, and the
correction shapes the design. Emitting a tool call is roughly 30 tokens of decode —
cheap even on Laguna — and it happens **inside Laguna's KV timeline**: whatever the main
model conditions on next must be in its cache, so a "boring" call cannot be handed to
AFM mid-turn without breaking context continuity. Call-level routing to the ANE is
mostly incoherent. What is actually boring, and actually expensive, is:

- **Consuming results.** At local speeds this is the real cost. A few hundred tokens
  per second of prefill means a 5,000-token test log dumped into Laguna's context costs
  seconds and joules immediately, adds attention cost to every subsequent token, and
  burns budget toward an eventual compaction — itself a re-prefill.
- **Whole navigation turns.** The bulk of a session is read → search → look → decide
  loops where the decision is trivial. The frontier model earns its keep at sparse
  points: diagnosis, design, hard edits, integration. This skew is measurable on
  `ds4-agent` today from session transcripts (experiment 9).

The dispatchable unit is therefore the **role**, not the call:

| Route to AFM/ANE | Keep on Laguna | Why the line is there |
|---|---|---|
| Result condensation (the #1 candidate) | Consuming the condensed result | Highest volume, read-only; the payoff is Laguna prefill avoided |
| Search and triage ranking | Deciding what the ranking means | Classification; schema-guaranteed via guided generation |
| Whole scout turns (list/read/search → capsule) | Diagnosis from the capsule | The loop is boring; the interpretation is not |
| Commit messages, titles, ledger summaries | — | Pure prose, zero risk |
| — | Edits, even mechanical ones | A wrong edit is a side effect; mechanical edits are the implementer's job, under a spec |
| — | Anything gating a side effect | "Boring" is not "safe to be wrong about" |
| — | Tool-call emission mid-turn | KV continuity, and it is cheap anyway |

**Condensation is a cache, not a one-way door.** The failure mode is the small model
dropping the one line that mattered from a test log, and Laguna diagnosing from bad
evidence. Mitigations: condense conservatively — error lines and tail verbatim, the same
rule the small-context packet already applies to test output — and keep the raw result
addressable so Laguna can pull the full text on demand. This is exactly the existing
bounded-`read`-plus-`more` shape (`agent_tool_more`, `ds4_agent.c:6217`): the condensed
result is the default view, `more` is the escape hatch. Nothing new to invent.

### The router is deterministic

"Is this boring?" is itself a triage question, and the deterministic-controller
principle applies to the harness's own plumbing: prefer rules where rules work, and
spend model judgment only where rules cannot decide. The router is ordinary Swift code:

- tool result larger than N tokens → condense before it enters Laguna's context;
- task matches an SDD template → implementer;
- read-only reconnaissance with a capsule contract → AFM scout;
- everything else → Laguna.

Rules make routing predictable, auditable, and free. A model-based router is a last
resort for genuinely ambiguous dispatch — and if one is ever needed, it is itself an
AFM-tier classification with a guided-generation schema, never a Laguna call.

### PCC is demoted

Earlier sections treated Private Cloud Compute as an optional 32K planner and reviewer.
The current direction drops that role entirely. With Laguna at 80K locally, PCC's
capability edge is gone. With spec-driven development (SDD) — roadmap phases specified
as repo artifacts before implementation — planning is a document produced and reviewed,
not a runtime service. And PCC's quotas and availability conditions already disqualified
it from the control plane. The residual value, judgment diversity for review, is
available locally for free: re-run the review role at different seeds and temperatures
(sampling is Swift-side via `ds4_sample_logits`), or use the AFM tier as a cheap
dissenting opinion. PCC remains listed as an experiment, nothing more.

### Guided generation guarantees the harness's capsules

Foundation Models' `@Generable` guided generation is constrained decoding: the framework
masks logits during generation so output physically cannot violate the Swift type's
schema. Every structured artifact a harness role produces on the AFM tier — verdicts,
routing decisions, triage classifications — is schema-guaranteed by construction, which
deletes the parse-and-retry loop for those roles.

### Constrained decoding for Laguna via the logits API

The same guarantee is available for the big model, because the DS4 engine's public API
already exposes the sampling seam: `ds4_session_copy_logits`, `ds4_session_set_logits`,
and `ds4_sample_logits` are all in `ds4.h`. SwiftStar can therefore implement
grammar-constrained sampling for Laguna's tool calls:

1. The dialect parser's streaming state machine already tracks the grammar state
   ("inside `<tool_call>`, the next tokens must continue a valid tool name").
2. At each decode step in a constrained region, copy the logits out, mask every token
   that cannot continue the grammar, write them back, sample.

Malformed tool calls become impossible by construction — the property the KV design
claims for cache state, extended to the tool protocol — and the retry-on-malformed
machinery disappears for Laguna, not just for AFM roles. The cost is a vocabulary-sized
mask per token in constrained regions only; whether that is measurable against decode at
tens of tokens per second is experiment 7.

### Embeddings and retrieval on the ANE

The repository map and retrieval layer should never cost a decode cycle. A small
embedding model — `NLContextualEmbedding`, or a purpose-chosen model converted through
Core AI — runs on the ANE and keeps semantic code search entirely off the GPU.

### An ANE draft model for speculative decoding

The most speculative idea in this note, recorded with its preconditions. The engine
already exposes speculative machinery: `ds4_session_eval_speculative_argmax` and
`ds4_engine_has_mtp` are public. A small vocab-compatible draft model converted to
`.aimodel` and resident on the ANE could draft tokens while the GPU verifies batches —
drafting on silicon Laguna does not use. Fewer GPU passes per accepted token is an
energy win as well as a latency one. Preconditions: a distilled draft sharing Laguna's
tokenizer; a verify entry point accepting externally supplied drafts (the current path is
MTP self-drafting); and, decisively, verified ANE placement. Core AI chooses CPU, GPU, or
ANE per operation — placement is a compiler outcome, not a promise. Design the draft to
be ANE-friendly (small, fp16, GQA, bounded shapes) and confirm residency with Xcode's
performance reports before believing any of this.

### ANE concurrency from Swift

ANE access is mediated by the framework and a system daemon; calls are `async` and safe
from any task or actor, and the OS schedules requests across all clients. Two properties
to design for: the ANE is shared — embeddings, AFM roles, and system Apple Intelligence
features all timeshare it, so ANE-tier work must tolerate variable latency — and each
model should still sit behind one actor, not for memory safety but for admission control
and batching. The property that matters most is negative: ANE queue depth never affects
the Metal queue. Different silicon, different driver queue.

## A specialized implementer model

A 12B-class, Python-tuned implementer is the natural middle tier, and SDD is what makes
it viable: a specialized small model is weak on open-ended tasks and strong on narrow,
well-specified ones, and SDD makes tasks narrow and well-specified by construction. The
division of labour: Laguna plans, reviews, and integrates (the judgment roles); the
implementer executes spec'd diffs and writes tests (the mechanical roles); AFM condenses
and triages (the trivial roles).

Three practical points. A 12B at Q4 is roughly 7 GiB of weights — trivially co-resident
beside Laguna's 45 GiB on a 128 GiB machine. Twelve billion parameters is the scale
where fine-tuning on a Mac is realistic (LoRA via MLX), so "tuned for this project's
workflows" is an achievable iteration loop rather than a wish. And the implementer
almost certainly runs on the GPU — AFM-class ANE models are around 3B, and 12B likely
exceeds comfortable ANE throughput — so its decode contends with Laguna and must be
admitted through the same work-conserving scheduler as any other session.

### What converts to `.aimodel`, and what does not

The conversion question has a sharp answer once the artifact types are kept straight.
`coreai-torch` consumes a `torch.export.ExportedProgram` — a PyTorch model. **DS4 cannot
be converted**: it is an inference engine, not a model; there is no graph to export.
**Laguna could be converted but should not be**: its Q2_K/Q3_K quantizations are
GGUF-specific block formats Core AI does not ingest, so conversion means requantizing —
losing the asymmetric routed-expert scheme and likely growing past 45 GiB — and DS4's
value (SSD streaming, compressed KV, mmap loading, disk checkpoints) is runtime policy
that no graph conversion carries over, as the Core AI sections above already establish.
**The implementer is where `.aimodel` earns its keep**: small, dense or simple-GQA,
quantizes conventionally, benefits from the Core AI compiler and possible ANE placement,
and needs no new DS4 model-family work — which matters because DS4 is deliberately
narrow. This gives the Core AI conversion spike its concrete target.

## The OS as harness substrate

macOS is not just a place to host the agent; several OS primitives map one-to-one onto
harness concepts this note already defines.

### Transactional forks: APFS clones plus KV snapshots

**Status: optional extension, not a prerequisite.** Per "The lightweight design" under
"Parallelism", the first three subagent uses need only template-spawn and session
lifetime. Everything in this subsection is what you add if O(1) workspace rollback turns
out to be worth an OS-primitive dependency — and note that only the filesystem half is
copy-on-write; the KV half is an eager copy.

APFS `clonefile()` produces O(1) copy-on-write clones of directories: no data is copied
until divergence. Paired with `ds4_session_save_snapshot`, the harness can fork the
durable agent world — KV state and filesystem together — in constant time: snapshot the
session, clone the workspace, proceed; on failure, discard both. Subagent workspace
isolation stops needing git worktrees: each writer gets a clone, and merging is a diff
against the parent clone.

The design rule that makes this coherent is a state taxonomy, because the C agent's
worker state (`ds4_agent.c:107`) mixes things that fork with things that cannot:

- **Forkable values**: transcript tokens, ledger, configuration, session title. In Swift
  these are `Codable` value types; forking is a copy.
- **Leased resources**: bash jobs (live PIDs), containers, Python kernels, the Chrome
  connection. Never forked. A child re-acquires its own leases; a fresh container over
  the cloned workspace starts in under a second.
- **Engine state**: KV, captured by `ds4_session_save_snapshot`, with the matching token
  prefix from `ds4_session_tokens()` so the pair is taken consistently on the inference
  queue.

Forks happen only at turn boundaries with no leases outstanding. The `more` pagination
cursor is a cache and is dropped. That this reduced state suffices to reconstruct a
session is not conjecture — it is what the existing `/strip` semantics prove: the
kvstore already rebuilds a full session from text alone.

### FSEvents: evidence, ambient work, and speculative warming

Three uses, in increasing order of ambition.

**Evidence.** The orchestrator verifies subagent claims instead of trusting summaries.
FSEvents provides the authoritative journal of what a subagent's workspace actually
touched, so verification starts from the real change list rather than from the files the
subagent chose to mention.

**Ambient work (push mode).** A debounced FSEvents trigger on the user's save — from any
editor — can wake background roles without a prompt: re-embed changed files, run
affected tests, pre-compute a review. Guardrails are part of the design, not an
afterthought: unprompted work is read-only; write actions remain user-initiated; ambient
jobs are admitted only with AC power and thermal headroom; and they run as
lowest-priority filler in the work-conserving scheduler's idle GPU gaps.

**Speculative context warming.** The DS4-specific version: on save, tokenize the changed
file and its dependency neighbourhood and prefill a warm session in the background. When
the user actually asks a question, the KV prefix is already resident and
time-to-first-token collapses. Prefill is the dominant interactive latency in this
note's own analysis; this hides it before the question is asked.

### Memory pressure and thermals as harness inputs

`DispatchSource` memory-pressure events map directly onto the kvstore: at warning
pressure, strip idle subagent sessions to disk using the existing payload APIs and
rebuild them by prefill when next needed — the app cooperates with the memory system
instead of being its largest casualty. `ProcessInfo.thermalState` and power-source
changes drive `ds4_engine_set_power`: the agent throttles itself on a hot or unplugged
laptop.

### Instrumentation

The GPU duty-cycle measurement in the experiments list should be `os_signpost`
intervals around prefill, decode, and tool spans, not a bespoke tracer. Instruments then
shows GPU idle-awaiting-tools per tool type on a timeline, and the number that justifies
subagents falls out of a screenshot.

### Smaller native affordances

- **`~Copyable` session ownership.** Model the `ds4_session` handle as a move-only
  Swift type. "The same session cannot be evaled concurrently" stops being a comment and
  becomes a compile error.
- **Security-scoped bookmarks.** The GUI's workspace access is user-granted per folder —
  a real permission boundary the TUI never had, and the honest complement to container
  isolation for host-side tools.
- **App Intents.** Agent tasks exposed to Shortcuts and Spotlight ("run the test-fix
  loop on this folder"), which also gives automation users a scriptable surface for
  free.
- **An XPC helper as an option.** Running the engine in a separate helper process keeps
  45 GiB of warm weights alive across app relaunches and UI crashes. It trades away the
  in-process directness this note argues for, so it is recorded as an option, not the
  plan; KV lives in Metal buffers and does not move cheaply between processes.

## Python integration

The kernel described here is containerized because it runs **model-authored code** — not
because it is Python. Trusted, human-authored Python is a different habitat entirely; see
"Swift body, Python brain" below. Do not embed this interpreter in-process:
PythonKit-style embedding entangles the GIL with Swift concurrency and shares one crash
blast radius. The native design is a **persistent Python kernel inside the
Containerization VM**, uv-managed, speaking a Jupyter-shaped protocol over vminitd's
vsock gRPC.

Interpreter state is a feature for iteration and a hazard for evidence, so the design
splits the modes:

- **Exploratory mode** uses the stateful kernel: REPL iteration, plotting, data
  inspection. Each tool call executes in a fresh namespace over a shared import cache,
  which bounds leak damage while keeping imports warm. Rich outputs — images,
  dataframes — stream back as data the GUI renders natively, a tool-result class the TUI
  could never display.
- **Evidence mode** is hermetic: anything the orchestrator records as verification (test
  results, benchmark numbers) runs as a fresh `uv run` process. Evidence never comes
  from a stateful kernel.

Kernels are leased resources per the fork taxonomy: per-subagent, never forked, and
restart is tens of milliseconds in a warm container. Python state checkpointing
(dill/CRIU-style) is explicitly out of scope.

## Swift body, Python brain

SwiftStar has a second allegiance alongside macOS: Python. The project sits inside a
Python-community effort to build AI tooling on Python's own terms, which means Python
stays first-class — not a scripting afterthought, but the place where agent behavior
lives. The traditional pattern — skills as prompt files that shell out to CLIs —
underuses the language. The bolder architecture is a **policy/mechanism split**, the
shape Emacs chose: a native core owns the machinery, a dynamic language owns the
behavior, and the system's character comes from the dynamic side.

- **The body (Swift, C, Metal, ANE)** owns everything per-token and everything
  hardware-facing: the DS4 engine, tier dispatch, constrained-decoding mask execution,
  the work-conserving scheduler, APFS forks, FSEvents, the power governor, containers,
  and the GUI.
- **The brain (Python)** owns everything per-turn: agent loops, SDD workflows, routing
  policy, tool implementations, prompt construction, capsule schemas.

### The layering rule: per-token is body, per-turn is brain

The boundary is a latency budget, not an aesthetic. Brain decisions happen tens to
hundreds of times per session; body operations happen tens of times per second. A
Unix-socket round-trip of roughly 100µs is invisible per-turn and fatal per-token.
Anything that must touch every token — sampling, grammar masks, KV management,
duty-cycle control — is compiled into the body. Anything that decides *what happens
next* is brain.

### Python declares, Swift executes

The deterministic router stays deterministic and stays in Swift — but its *rules* are
authored in Python and shipped to the body as data, the way eBPF programs are authored
in one place and executed in another. The same declare/execute split covers:

- **Tool schemas from type hints.** A tool is a typed Python function. Its signature
  generates three artifacts from one source of truth: the schema text in the prompt,
  the constrained-decoding grammar the body enforces per-token, and the GUI's rendering
  of the call. PEP 750 t-strings make the prompt fragments typed templates rather than
  string concatenation.
- **Routing tables.** Size thresholds, SDD-template matches, tier assignments: Python
  data structures, validated once and executed by the body.
- **Capsule schemas.** Declared in Python, enforced by the body — guided generation on
  the AFM tier, grammar masks on Laguna.

### The brain is a peer process with hot reload

The brain runs as a separate host-side process — uv-managed, spawned and supervised by
the body — speaking a small typed protocol over a Unix socket. Peer, not embedded, for
three reasons:

- **Hot reload.** Edit agent logic; the brain restarts in milliseconds; the body keeps
  45 GiB of weights warm and the KV timeline intact. Iterating on agent behavior stops
  costing a model load. This inverts the engine-XPC idea: the restartable part is the
  policy, not the engine.
- **Crash isolation without a VM.** The brain is trusted, human-authored code — by the
  whose-code-executes principle it needs no container — but a peer process keeps a
  brain bug from taking down the engine.
- **The brain is a normal Python package.** Installed with uv, tested with pytest
  against a **fake body** — a pure-Python implementation of the body protocol. Agent
  logic becomes developable and CI-testable on Linux, by contributors with no Mac and
  no GPU. The Mac-specific part of SwiftStar is the body; the community-developable
  part is the brain.

Embedded free-threaded CPython (3.14+) remains the upgrade path if the per-turn
boundary ever proves too slow; by the layering rule, it should not.

### Three Python habitats

The whose-code-executes principle now yields a trust gradient with three habitats:

| Habitat | Trust | Runs |
|---|---|---|
| Brain process (host) | Human-authored, reviewed | Policy, first-party tools, tainie-class tools |
| Kernel (container) | Model-authored | Exploratory REPL work |
| Hermetic runs (container) | Model-authored | Evidence: tests, benchmarks |

Community skills default to the container and are promotable to the brain when trusted —
the same judgment as adopting any dependency, made explicit.

### Tainie as the exemplar tool class

[Tainie](https://github.com/pauleveritt/tainie) — type-resolved refactoring with LLM
judgment — is the existence proof for what brain tools should look like, and it
converges with this note's designs to a degree worth listing:

- Its three-parameter API (`workspace`, `symbol`, `instruction`) **is a task packet**,
  and its `Worklist` with first-class `unresolvable` output **is a capsule** with an
  honest failure surface — the discipline the subagent section demands, arrived at
  independently.
- Its **overlay-first atomicity** — speculate all edits in memory, verify the whole
  candidate, write only if it passes — is the transactional-fork idea at file
  granularity. On SwiftStar, a tainie-class tool gets an APFS-cloned workspace as its
  overlay substrate for free.
- Its **negative oracle** — the type checker proves edits don't break, without
  demanding they be perfect — generalizes the evidence-mode rule: verification sets a
  safety floor under model creativity rather than replacing it.
- Its per-site micro-agent maps directly onto the tier router: deterministic discovery
  (pyrefly type facts) costs no model at all; per-site mechanical edits under an
  instruction are precisely the **implementer tier's** job; the oracle runs as a
  hermetic evidence check. Tainie already runs against `ds4-server`, so driving
  DwarfStar from Python tooling has precedent today.

The generalization: brain tools are not prompt files wrapping CLIs. They are typed
Python packages that combine deterministic discovery, bounded model judgment requested
*from the body's router*, and oracle verification — with tiers, forks, and evidence
machinery available as ambient services from the body.

## Energy as a design axis

Local inference makes energy a user-facing property — battery drain and fan noise — and
the harness has more leverage over it than the engine does.

- **Pace-to-read decoding.** When output streams to a human, generating faster than
  reading speed is wasted energy. The duty-cycle knob is public
  (`ds4_engine_set_power`): throttle decode toward reading pace during interactive
  display and open the throttle only when output feeds a parser, a tool, or a queued
  turn. The user-visible result: the fan does not spin while you read.
- **Watts-aware routing.** The tier table above is an energy policy as much as a latency
  policy: every token moved from the GPU to the ANE is battery. The router's cost
  function should include joules.
- **A power governor as a component.** Inputs: thermal state, power source, Low Power
  Mode. Outputs: engine power percentage, ambient-work admission, and tool-work QoS
  class (`.utility` QoS steers CPU-side work to efficiency cores — a direct fan-noise
  win).
- **Nap-mode prefill.** Deferred, plugged-in warming via `NSBackgroundActivityScheduler`,
  which already defers to system energy policy: overnight, pre-warm template sessions
  and repo-map context for active repositories, so morning sessions start with zero
  prefill.
- **Speculative decoding and pre-tokenization as joules-per-token wins.** Batch-verified
  drafts mean fewer GPU passes per accepted token, and FSEvents-driven warming means
  prefill work happens once, on AC, instead of interactively on battery.

## Peer-to-peer co-development

The subagent design requires immutable bounded task packets and capsule results verified
against local evidence — constraints adopted for context-budget reasons. Those same
constraints make a work unit **network-serializable**. That convergence is the whole
idea:

- A peer SwiftStar instance with an idle GPU advertises capacity (Bonjour on the LAN;
  CloudKit presence beyond it). The orchestrator dispatches a task packet to the peer
  instead of spawning a local session; the capsule and diff come back and are verified
  locally against the APFS-cloned workspace, exactly like a local subagent's output.
  Unlike local parallelism, this genuinely multiplies decode throughput: two machines,
  two GPUs.
- **The trust model is the existing verification stance.** Capsules are never trusted;
  diffs are verified locally; only stripped text and ledgers cross the wire, never KV
  payloads, which are quant- and backend-specific and would not transfer meaningfully.
- **Handoff between one user's Macs** is the small version: `NSUserActivity` Continuity
  carries the stripped session — the desktop runs heavy overnight phases, the laptop
  picks up the ledger elsewhere, KV rebuilt by local prefill.
- **Pair-agenting** is the two-person version: with SDD, the spec is the shared
  artifact; CloudKit shared zones (`CKShare`) provide sync and access control; one
  machine runs the implementer role while the peer runs the reviewer over the synced
  ledger. Merging stays git-shaped: peers work in their own clones and exchange diffs.

One layer distinction to keep sharp: DS4 already has Mac-to-Mac *engine-level*
distribution (`ds4_distributed.c`, wired into the CLI tools) — splitting one model's
inference across machines. The peer-to-peer described here is *task-level* and
complementary; the two should not be conflated.

## Suggested phasing

Ordered so that each step is useful alone and none forecloses the next.

1. **Extract the parse/execute seam.** Turn `agent_execute_tool_call`
   (`ds4_agent.c:7960`) into a callback, and expose the streaming events already implicit
   in `agent_tool_visualizer`. Keep the dialect parsers and prompt builders in C.
2. **Build the GUI against one session.** Native transcript, tool-call rendering,
   context gauge from `agent_status`, cancellation via `ds4_session_set_cancel` with the
   tri-state `sync` handling. Retire the TUI.
3. **Move tool execution to Swift**, then parallelise independent intra-turn calls.
   Standalone throughput win, no new concepts.
4. **Add template-spawn subagents for context firewalling and speculative warming.**
   Per "Parallelism" above, this is the highest-value use and needs only
   `ds4_session_save_snapshot`/`load_snapshot`, bounded per-role context, and
   `ds4_session_free` on completion — no containers, no scheduler, no APFS. Moved ahead
   of container work because it is the one win that compounds (shrinks the critical path)
   rather than merely hiding idle GPU time, and because nothing later in this list is a
   prerequisite for it.
5. **Add container-backed execution** for `bash` and any code-running tool, with per-tool
   policy for host-side versus containerised.
6. **Add the work-conserving scheduler and disposable-exploration panes**: a scheduler
   over the serial inference queue that overlaps long tool calls with sibling decode, plus
   user-visible subagent panes for exploration that discard cleanly via `ds4_session_free`.
   Gate this step on the duty-cycle measurement in "Measure before building the
   scheduler" — if idle GPU is already thin, skip straight to step 7. Treat
   "Transactional forks" (APFS clones, mid-task KV forking) as an optional extension of
   this step, not a requirement of it.
7. **Optionally investigate the Laguna batch-decode exclusion** (`ds4.c:61991`). Highest
   ceiling, but it is engine work in the hot path and per `AGENT.md` would require
   regression checks across Metal, SSD streaming, and CUDA.

Steps 1 through 3 deliver a better single-agent application. Step 4 delivers the largest
context-budget win and depends only on step 1's session/template machinery — it does not
need the GUI shell, containers, or a scheduler. Step 6 only pays off once step 5 has made
tool durations long enough to be worth hiding, and only if step 6's own gating measurement
says the idle GPU is there to fill.

Several tracks are orthogonal to this order and can land independently of it:

- **Constrained decoding** over the logits API — after step 1, since it needs the
  parser's grammar state exposed.
- **The power governor and pace-to-read** — after step 2; they only need the duty-cycle
  knob and a streaming flag.
- **The AFM tier and ANE embeddings** — any time; no engine coupling at all.
- **The Python kernel** — with step 5, since it lives in the container.
- **App Intents and security-scoped bookmarks** — with step 2, as part of the app shell.
- **FSEvents ambient work and speculative warming** — after step 4, since it reuses the
  same template-spawn machinery on a debounced save trigger; needs neither containers nor
  the scheduler.
- **Peer dispatch** — after step 6, since it reuses the task-packet interface unchanged.
- **The implementer model** — after step 6 gives it a scheduler to live in; its
  Core AI conversion spike can start any time.
- **The brain protocol and Python tools** — the fake body can start immediately, since
  it needs no Swift at all; the real body wires in after step 3, once tool execution
  has a seam.

## Sources

- [Core AI](https://developer.apple.com/documentation/coreai)
- [Integrating on-device models with Core AI](https://developer.apple.com/documentation/coreai/integrating-on-device-ai-models-in-your-app-with-core-ai)
- [Compiling Core AI models ahead of time](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time)
- [Core AI PyTorch Extensions](https://apple.github.io/coreai-torch/main/)
- [Foundation Models](https://developer.apple.com/documentation/foundationmodels)
- [Adding server-side intelligence with Private Cloud Compute](https://developer.apple.com/documentation/foundationmodels/adding-server-side-intelligence-with-private-cloud-compute)
- [Introducing the Third Generation of Apple's Foundation Models](https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models)
- [Aider repository map](https://github.com/aider-ai/aider/blob/v0.86.2/aider/website/docs/repomap.md)
- [SWE-agent](https://arxiv.org/abs/2405.15793)
- [Agentless](https://arxiv.org/abs/2407.01489)
- [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)
- [Apple Containerization](https://github.com/apple/containerization)
- [Containerization `sandboxy` example](https://github.com/apple/containerization/blob/main/examples/sandboxy/README.md)
- [MLX](https://github.com/ml-explore/mlx)
- [Tainie](https://github.com/pauleveritt/tainie)
- [PEP 750 — Template Strings](https://peps.python.org/pep-0750/)

All Apple APIs described here are from developer-preview documentation and should be
rechecked against the installed macOS/Xcode SDK before implementation. The macOS 27 and
Apple Containers 1.1 timing is a planning premise, not a verified release commitment.

DS4 line references were checked against branch `laguna-s2.1` at commit `3b0ec5f`. The
per-session KV figures are derived from `ds4_context_memory_estimate_with_prefill_mode`
(`ds4.c:35274`) and the Laguna shape at `ds4.c:675`; they are arithmetic from the engine's
own formula, not measurements of a running process. The logits, snapshot, speculative,
power, and token-prefix APIs cited throughout are all present in the same `ds4.h`. Load
time, prefill throughput, the prefill-workspace row, the ANE energy and speed ratios
(experiment 8), and ANE placement of any converted model remain unmeasured.

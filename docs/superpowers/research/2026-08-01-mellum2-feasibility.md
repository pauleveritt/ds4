# Mellum 2 feasibility for DwarfStar

Research begun 2026-08-01 on branch `mellum-2.1`, based on the Laguna XS 2.1
SSD-streaming line at `bcf1b1c`.

The working conversation calls the target "Mellum 2.1". JetBrains' public
artifacts and technical report call the architecture **Mellum 2**. This note
uses *Mellum* for the architecture, *public Thinking release* for the released
JetBrains checkpoint, and *local GRPO checkpoint* for the exact downloaded
weights we ultimately want to run.

## Decision summary

Mellum is a credible DwarfStar target. Resident Metal inference should be a
bounded new-family port: the model uses operations ds4 already has in some
form, and both a canonical llama.cpp implementation and official GGUF files
exist.

Do not implement Mellum as a Laguna variant. The two families share useful
primitives, but differ in layer schedule, router semantics, MLP topology, and
published quantization layout. Mellum should have its own family, shape,
metadata validator, tensor binding, and graph entry points.

Bring up resident inference before SSD streaming. The public Q8_0 GGUF is the
best first correctness artifact because ds4 already supports Q8_0. The public
Q4_K_M file contains Q5_0 expert-down tensors, which ds4 can parse as GGUF but
cannot currently execute on Metal. SSD streaming can reuse the cache lifecycle,
but not the existing Laguna XS streamed expert kernels unchanged.

Keep MTP out of the initial scope. The report describes a one-head MTP draft
model, but neither the local GRPO checkpoint nor the public Thinking release
contains MTP tensors.

## Pinned evidence

| Source | Revision or identity | What it establishes |
| --- | --- | --- |
| [Mellum 2 Technical Report](https://arxiv.org/abs/2605.31268) | arXiv `2605.31268v1`, 2026-05-29 | Intended architecture, training, layer-selective YaRN, MTP design |
| Local GRPO checkpoint | HF cache snapshot `572fe3b9aa78f1476b68eb6017043fb0688bb278` | Exact local weights under investigation |
| [Public Thinking release](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking) | `ba4838faad89e968c36f39e76e95319d756714fe` | Canonical Mellum config and tokenizer |
| [Public Q4_K_M GGUF](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M) | repo `71a489e7b95efacf89feaaa6fe3b2995f3542409`; file SHA-256 `489cf0d7ca86ef4683e34e2efe8a46a9b52573bfe994a6ad9c86bf57c7173ccb` | Canonical GGUF metadata, tensor names, and published mixed quantization |
| Local llama.cpp | `0e4a0362239713ea95a6864a17a8de4b0ad90d62` | Executable Mellum graph and GGUF reference |
| DwarfStar XS base | `bcf1b1c2773a7cbaddd54416482a8317d2a0a8bc` | Existing Laguna XS resident and SSD-streaming machinery |

The local GRPO snapshot contains five safetensor shards and 24,299,846,144
bytes of tensor payload. Its index has 5,631 tensor entries. The public
Thinking release has the same tensor names, shard assignment, shard sizes, and
total payload size, but different shard hashes. The public release is therefore
an architecture and packaging oracle, not a behavior oracle for the local
weights.

## Canonical architecture

The released config identifies `MellumForCausalLM` with model type `mellum`.
The local GRPO snapshot instead identifies `Qwen3MoeForCausalLM` with model type
`qwen3_moe`. That local config is an older compatibility export and is not a
complete statement of Mellum inference semantics.

The fixed shape is:

| Property | Value |
| --- | ---: |
| Layers | 28 |
| Hidden width | 2,304 |
| Vocabulary | 98,304 |
| Query heads | 32 |
| KV heads | 4 |
| Head dimension | 128 |
| Query projection width | 4,096 |
| Expert count | 64 per layer |
| Active experts | 8 per token |
| Expert hidden width | 896 |
| Routed expert projection tensors | 5,376 (`28 x 64 x 3`) |
| Sliding window | 1,024 tokens |
| Maximum context | 131,072 tokens |

Every layer is sparse. There is no leading dense layer and no shared expert.
Each layer has a standard pre-attention RMS norm, Q/K per-head RMS norms,
GQA, an attention residual, a pre-MoE RMS norm, routed SwiGLU experts, and an
MoE residual.

### Attention and RoPE

The layer pattern is:

```text
sliding, sliding, sliding, full
```

repeated seven times. Full-attention layers are therefore `3, 7, 11, 15, 19,
23, 27`. This is the opposite phase from Laguna XS, whose four-layer groups
begin with the global layer.

The two attention types also use different rotary behavior:

- Full-attention layers use YaRN with theta 500,000, factor 16, original
  context 8,192, beta-fast 32, beta-slow 1, and attention factor
  1.2772588722239782.
- Sliding-attention layers use ordinary, unscaled RoPE with theta 500,000.

The report is explicit that YaRN is applied only to full-attention layers.
The canonical Transformers implementation constructs separate masks and
rotary embeddings for the two layer types. The older `qwen3_moe` config in the
local snapshot flattens the RoPE settings and cannot express this distinction
faithfully under stock Qwen3-MoE inference.

### Router

For every token and layer:

1. Compute 64 bias-free router logits.
2. Apply softmax across all 64 logits in FP32.
3. Select the top eight probabilities.
4. Renormalize those eight selected probabilities to sum to one.
5. Apply the selected weights to the eight SwiGLU expert outputs and sum them.

There is no expert bias, shared expert, or Laguna-style expert weight scale.
The current Laguna graph calls a router helper that requires a bias tensor and
passes an expert scale. A Mellum path needs a bias-free form with scale 1.

### MTP

The technical report says pre-training and post-training retain a single MTP
head. The released and local weight indexes both contain zero tensors named for
MTP, next-token prediction, or a draft module. The released runtime graph also
has no MTP block. Treat this as a report-versus-release packaging distinction;
do not infer missing weights or include speculation in the first port.

## GGUF findings

JetBrains publishes Mellum Thinking GGUFs in BF16, Q8_0, Q6_K, Q4_K_M, and
MXFP4_MOE. This removes the earlier assumption that a new converter is needed
before loader bring-up.

The public Q4_K_M file is 8,071,295,040 bytes and contains 339 tensors. Its
relevant metadata uses the `mellum.*` namespace and records the full shape,
sliding-window pattern, YaRN parameters, and separate SWA RoPE base.

The expert layout is not uniform:

- All expert gate and up tensors are Q4_K.
- Fourteen expert down tensors are Q8_0.
- Fourteen expert down tensors are Q5_0.

The 896-element expert width is not divisible by the 256-element K-quant block,
so the expert-down projections cannot all use Q4_K/Q6_K. This invalidates the
initial rough estimate based on an ideal four-bit payload and prevents direct
reuse of the XS Q4/Q6 streamed address-table kernels.

The GGUF parser knows Q5_0's block geometry, but `DS4_TENSOR_Q5_0` and Metal
Q5_0 matmul dispatch are absent. The official Q4_K_M artifact is therefore not
the smallest-risk first inference target. Start with the official Q8_0 GGUF;
add mixed Q4/Q5/Q8 support only after architecture correctness is established.

### Token termination discrepancy

The tokenizer defines:

| Token | ID |
| --- | ---: |
| `<|endoftext|>` | 0 |
| `<|im_start|>` | 27 |
| `<|im_end|>` | 28 |

The HF model config says `eos_token_id = 0`, while the tokenizer config names
`<|im_end|>` as EOS and the official GGUF records EOS ID 28. DwarfStar should
follow the GGUF/tokenizer contract for the GGUF runtime, but this disagreement
must be captured in the golden generation fixture rather than decided by
assumption.

## Step 1 reference artifact

Completed 2026-08-01. The initial correctness oracle is the pinned public
Thinking Q8_0 GGUF, kept in the Hugging Face cache rather than copied into the
worktree:

```text
/Users/pauleveritt/.cache/huggingface/hub/models--JetBrains--Mellum2-12B-A2.5B-Thinking-GGUF-Q8_0/snapshots/6f5b0031c9ea37740f630362d3c06c54933fc2f4/Mellum2-12B-A2.5B-Thinking-Q8_0.gguf
```

It was fetched with `hf download` at revision
`6f5b0031c9ea37740f630362d3c06c54933fc2f4`. The 12,925,417,536-byte file has
SHA-256 `d4049c2599796d18245523818c0534e8f8605166fc58c60dd0753547facdb3a2`.
The cache snapshot is a symlink to the content-addressed blob carrying that
same digest.

### GGUF contract

The artifact identifies itself as `Mellum2 12B A2.5B Thinking`, architecture
`mellum`, GGUF quantization version 2, and file type 7 (`Q8_0`). Its 339
tensors are exactly 198 `Q8_0` tensors and 141 `F32` tensors: there are no
Q5, K-quant, or other execution types in this bring-up artifact.

The parsed `mellum.*` metadata confirms the fixed contract described above:
28 blocks; context 131,072; embedding 2,304; routed feed-forward width 7,168
(eight active 896-wide experts); 32 query and four KV heads; 128-element K/V
heads; 64 total and eight selected experts; and a 1,024-token sliding window.
It also records YaRN factor 16, original context 8,192, attention factor
1.2772589, beta-fast 32, beta-slow 1, and both ordinary and SWA RoPE bases at
500,000.

The tokenizer contract is `gpt2` with pre-tokenizer `mellum2`, BOS ID 0, and
EOS ID 28. The embedded chat template renders a single user message with
thinking disabled as:

```text
<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

For `Answer with one word: the capital of France is`, this is the 24-token
sequence:

```text
[27, 1397, 233, 5698, 434, 768, 1574, 60, 302, 5782, 332, 8438,
 359, 28, 233, 27, 8091, 233, 23, 233, 233, 24, 233, 233]
```

This tokenization was cross-checked with the local snapshot's matching
`mellum2` tokenizer. It must be rechecked directly against DwarfStar's GGUF
tokenizer implementation when loader support lands; it is a tokenizer fixture,
not evidence that the local GRPO weights have public-release behavior.

### Greedy generation baseline

Reference runtime: local llama.cpp `0e4a0362239713ea95a6864a17a8de4b0ad90d62`
(`b10154-0e4a03622`) on the Apple M5 Max, all layers GPU-offloaded. Each run
used the embedded chat template, `--single-turn --reasoning off`, context 512,
`--temp 0`, seed 123, and `--predict 16`; no sampling penalty or stochastic
sampler was enabled. The observed continuations were:

| User message | First greedy token | 16-token continuation |
| --- | --- | --- |
| `Complete this Python function: def add(a, b):` | `def` (ID 910) | `def add(a, b):\n    """Return the sum of a` |
| `Write only the JavaScript body of function square(x).` | three backticks (ID 889) | `&#96;&#96;&#96;javascript\nreturn x * x;\n&#96;&#96;&#96;` |
| `Answer with one word: the capital of France is` | `Paris` (ID 50195) | `Paris` |

One-token reruns matched those three first-token values. The local CLI build
does not expose a top-logit/probability dump or a standalone server binary, so
this step records exact first-token IDs and continuations but **not** numeric
logits. Add a logits-capable llama.cpp build or a minimal reference evaluator
before using a logit tolerance as a DwarfStar gate.

llama.cpp terminates its generated chat response on the GGUF EOS ID 28. It
does not configure ID 0 as EOS. Keep both IDs in DwarfStar's generation test
matrix: 28 is the public GGUF runtime terminator; 0 remains the conflicting HF
model-config value and must not silently become the GGUF stop condition.

## Reuse assessment

### Useful existing pieces

- GGUF mmap loading and metadata parsing.
- Q8_0, Q4_K, and dense F32/F16 tensor operations. In particular, the generic
  Q8_0 matvec accepts input widths divisible by 32, so both 2,304 and 896 are
  valid dense projection widths.
- Laguna's GQA attention structure, per-head Q/K RMS norm, NeoX RoPE, mixed
  full/SWA scheduling, and layer-specific KV capacity.
- Routed top-k MoE buffer management and general Metal command orchestration.
- The generic streamed expert-cache lifecycle: lookup, load, eviction,
  address buffers, and accounting.
- Laguna XS resident-versus-streamed A/B methodology and quality fixtures as a
  process template.

### Pieces that cannot be reused unchanged

- Laguna model-family detection, fixed shape, metadata names, and tensor
  binding.
- Laguna's `[full, sliding, sliding, sliding]` layer phase.
- Its router helper, which expects a per-expert bias tensor and a scale.
- Its leading-dense and shared-expert branches.
- Laguna's fused attention store, which consumes an attention-gate tensor that
  Mellum does not have. The Q/K RMS-normalization-and-RoPE primitive is useful,
  but attention storage/compute needs an ungated entry point.
- The fused routed/shared MoE kernel. It requires a shared expert, restricts
  the streamed down type to Q4_K/Q6_K, and rejects an expert width such as 896.
- The generic GLM routed-MoE entry point. Its resident kernels accept K-quant
  gate/up and down tensors, not Q8_0, and its host validation requires the
  expert intermediate width to be divisible by 256. Mellum's width is 896.
- The current XS streamed expert address-table kernels, which do not support
  the published Mellum Q5_0/Q8_0 down-projection mix.

The likely clean design is a separate Mellum graph that reuses low-level
attention, norm, matmul, and cache primitives. It should not accumulate
family conditionals inside `laguna_graph_forward_*`.

## Memory and SSD relevance

The public Q4_K_M model is about 7.52 GiB. The public Q8_0 model is reported as
12.9 GB. A Q4-class file may fit model-only on a 16 GB machine at modest
context, but a quality-oriented resident deployment needs workspace and macOS
headroom; the first supported target should be 32 GB unified memory. Q8 is
intended only as the first correctness artifact on the 128 GB development
machine.

### Proposed agentic-planning target

For a quality-oriented DwarfStar release, do not apply a uniform low-bit
quantization to Mellum. The useful MoE strategy is selective: the 1,792 routed
experts occupy about 10.98 GiB of the public Q8 file, while embeddings,
attention, output, routing, norms, and other non-expert tensors occupy only
about 1.05 GiB. Keep the latter at their existing Q8/F32 precision and
quantize only routed-expert gate/up tensors according to activation importance
measured on a planning-and-tool-use calibration corpus.

The initial target should be named along these lines:

| Tensor group | Target precision |
| --- | --- |
| Norms and router | F32 |
| Token embedding, attention, and output | Q8_0 |
| Every routed expert down projection | Q8_0 |
| Routed expert gate/up, layers 0–21 | imatrix-calibrated Q4_K |
| Routed expert gate/up, final layers 22–27 | Q8_0 |

This preserves the control path and the final six MoE layers while spending
compression where Mellum has sparse redundancy. It also deliberately avoids
Q5_0: the public Q4_K_M artifact uses Q5_0 in some down projections, but ds4
does not execute Q5_0 today. Mellum's 896-wide down input is not divisible by
the 256-element K-quant block, so retaining its down projections at Q8_0 is
both the quality-first and implementation-friendly choice.

The sizing estimate uses the exact GGUF block geometries: one Mellum Q8 expert
projection is 2,193,408 bytes; a Q4_K gate/up projection is 1,161,216 bytes.
The proposed Q4_K/Q8 target is therefore approximately **9.33 GiB** on disk.
It is a deployment target, not yet an executable ds4 artifact.

| Routed-expert recipe | Estimated GGUF size | Use |
| --- | ---: | --- |
| Public all-Q8 reference | 12.04 GiB | Correctness oracle on high-memory development hardware |
| Q4_K gate/up + Q8_0 down | 8.59 GiB | Baseline quality-oriented compression |
| Proposed Q4_K gate/up + Q8_0 final six layers/down | 9.33 GiB | First agentic-planning release target |
| Q3_K gate/up + Q8_0 down | 7.68 GiB | Later measured compromise |
| Q2_K gate/up + Q8_0 down | 6.98 GiB | Experimental; do not make the first planning target |

These estimates exclude a small GGUF header/alignment difference and should be
treated as within a few tens of MiB until a real artifact is written.

For the proposed target at a 32,768-token context, BF16 K/V storage costs about
0.49 GiB: the seven full layers retain all 32K rows, while 21 SWA layers retain
only 1,024 rows. At 131,072 tokens this becomes about 1.79 GiB. Budget a further
1–2 GiB for graph scratch and activations until the Mellum graph is measured.
That puts the 32K resident working set near 11–12 GiB before macOS and other
applications. Recommend **32 GB unified memory** for the first quality-oriented
resident deployment; 16 GB leaves too little headroom for normal desktop use.

SSD streaming may later make a 16 GB deployment plausible, but it is a capacity
mode rather than a quality mechanism. The proposed artifact's routed expert is
about 4.31 MiB, so a 512-expert cache would consume about 2.15 GiB. Such a path
requires Mellum-specific Q4_K gate/up plus Q8_0 down selected-expert and
address-table kernels, which do not exist yet. Do not advertise it as supported
until resident-versus-streamed parity and real smaller-memory measurements are
green.

With an optimized per-layer KV policy, 128K context is less expensive than a
naive all-layer allocation suggests. A BF16 K+V row is 2,048 bytes per layer
and token. Seven full layers at 131,072 tokens consume about 1.75 GiB; the 21
sliding layers need only about 42 MiB for 1,024-token rings. This excludes graph
scratch and other runtime state, but it makes resident Q4 at long context
plausible enough that SSD streaming should be justified by measurement rather
than assumed necessary.

SSD streaming remains interesting for extra memory headroom, smaller machines,
or larger caches and application state. Its feasibility is conditional:

- Positive: only 1,792 routed experts exist, and the cache supports 64 total /
  top-8 routing comfortably.
- Negative: the published Q4 expert-down layout needs Q5_0 and Q8_0 compute
  from streamed addresses, which the XS cache kernels do not implement.
- Unknown: the expert locality and cache size needed to beat mmap fallback have
  not been measured for Mellum.

## Design checkpoint after loader bring-up

The loader commit establishes that the Q8 artifact is structurally compatible
with ds4's mmap-backed model representation. It does **not** yet establish that
the tokenizer, chat template, router, MoE compute, or attention graph is
compatible. Treating `--inspect` success as the end of all loader-adjacent work
would move several cheap contract failures into the much harder full-graph
debugging phase.

Three findings change the order of work:

1. DwarfStar initially could not tokenize the Mellum artifact: it fell through
   to DeepSeek vocabulary initialization and looked for
   `<｜begin▁of▁sentence｜>`. Mellum now has a family-specific vocabulary branch,
   the `mellum2` pre-tokenizer, and `<|im_start|>` / `<|im_end|>` chat framing.
   The canonical `mellum2` split is GPT-2/StarCoder-like, with digits split one
   at a time; it is not ds4's JoyAI default.
2. Existing routed-MoE execution is not a drop-in Q8 bring-up path. The GLM
   entry point rejects Q8 tensors and requires an intermediate width divisible
   by 256, while Mellum uses 896. The generic Q8 matvec itself supports 896, so
   a small Mellum-specific selected-expert path is feasible, but it must be
   implemented and tested explicitly.
3. Final-token comparison alone is too coarse for the first full graph. A
   pinned llama.cpp server fixture now captures first-token top-10
   log-probabilities for three prompts. Small primitive tests should still
   precede graph assembly so router, RoPE, attention, and MoE errors can be
   localized; per-layer checkpoints remain a useful later diagnostic.

The Step 2 implementation also has a test-coverage distinction worth making
explicit: strict malformed-metadata rejection logic exists, and the authentic
artifact exercises the valid path, but focused negative fixtures have not yet
been added. The original Step 2 gate is therefore only partially complete.

All Mellum commands that can create a session or touch persistent KV state must
run with an isolated home, for example `HOME=/tmp/mellum2 ./ds4 ...`. DwarfStar
stores KV data in a shared location rather than partitioning it by model.
`--inspect` and the tokenizer-only `--dump-tokens` path do not create a session,
but graph, generation, prefill, and continuation tests do. This isolation is an
execution rule for every later gate, not an optional cleanup step.

## Revised first implementation sequence

### Step 1: Freeze the target contract and reference outputs

Use the public Q8_0 GGUF for initial bring-up while retaining the local GRPO
checkpoint as the final weight target.

Capture:

- complete GGUF metadata and tensor-type inventory;
- tokenizer IDs and exact chat-template rendering;
- deterministic llama.cpp tokenization and short greedy continuations;
- a small first-token/logit reference set;
- the expected behavior for both EOS IDs seen in the artifacts.

**Gate:** one pinned reference manifest and a compact golden fixture that can
be run before ds4 produces tokens.

**Status:** artifact identity, tensor inventory, rendered chat text, token IDs,
EOS behavior, three deterministic continuations, and a pinned top-10
first-token log-probability fixture are recorded.

### Step 2: Add loader-only Mellum support

Add a Mellum family and fixed shape, then implement:

- `mellum.*` metadata validation;
- the 339-tensor GGUF binding contract;
- tensor layout validation for the Q8_0 bring-up artifact;
- `--inspect` support without constructing a Metal graph.

Keep loading mmap-backed and do not alter the Laguna, GLM, or DeepSeek family
paths.

**Gate:** `ds4 --inspect` accepts the pinned Mellum Q8_0 GGUF and focused tests
reject incorrect layer count, expert shape, head shape, SWA pattern, and RoPE
metadata.

**Status:** implementation and authentic-artifact validation are complete.
Focused malformed-model coverage is not; do not describe this gate as fully
closed until those negative cases exist.

### Step 2a: Complete the tokenizer and chat contract

Before creating a graph:

- add Mellum vocabulary initialization using GGUF BOS/EOS metadata;
- implement the `mellum2` pre-tokenizer rather than falling through to JoyAI;
- recognize `<|im_start|>`, `<|im_end|>`, `<think>`, and `</think>` as literal
  special tokens;
- render the official enabled/disabled-thinking chat prefixes without adding
  an implicit BOS token; and
- keep tokenizer-only operation free of session/KV-cache creation.

**Gate:** `--dump-tokens` exactly reproduces the pinned 24-token fixture and a
small set covering whitespace, punctuation, contractions, and single-digit
splitting. Chat rendering agrees with llama.cpp for thinking enabled and
disabled. Add the outstanding negative loader coverage through table-driven
tests of extracted metadata/schedule validation helpers or a compact synthetic
GGUF; do not duplicate or rewrite the 12.9 GB artifact merely to corrupt a few
metadata fields.

**Status:** complete for the canonical no-thinking fixture and the thinking
and no-thinking assistant prefixes. `--dump-chat-tokens` provides a
tokenizer-only diagnostic, and `test-mellum-tokenizer` verifies both rendered
and native-chat forms against the pinned IDs. Broader punctuation and
contraction coverage remains desirable before release.

### Step 2b: Strengthen the numerical oracle

Build or expose a logits-capable llama.cpp reference and capture, for the same
pinned prompts:

- first-token top-k logits or log-probabilities;
- the exact input token sequence used by the evaluator; and
- if practical, one diagnostic prompt's per-layer checkpoints at Q/K norm,
  post-RoPE Q/K, attention output, router selection/weights, and MoE output.

The per-layer dump is diagnostic scaffolding, not a permanent product
dependency. It is valuable because 28 layers can turn a small early mismatch
into an opaque final-logit failure.

**Gate:** numeric references are reproducible from a pinned command/build and
stored compactly enough for repeated local comparison.

**Status:** complete for first-token top-10 log-probabilities. The fixture is
`tests/test-vectors/mellum-llama-cpp/first-token-logprobs.json`, captured from
the pinned llama.cpp revision's newly built `llama-server` with the exact
24/25/26-token rendered prompts. Per-layer checkpoints remain deferred
diagnostic work.

### Step 3: Prove the Mellum-specific primitives

Add focused CPU-scalar or small Metal tests before assembling the whole model:

- bias-free FP32 softmax over 64 logits, deterministic top-8 selection, selected
  weight renormalization with the reference denominator floor, and scale 1;
- the exact `[sliding, sliding, sliding, full] x 7` layer helper;
- sliding versus full RoPE parameter selection, including ordinary unscaled
  RoPE for sliding layers and YaRN only for full layers;
- an ungated four-KV-head GQA attention operation; and
- a resident selected-expert path matching the proposed Q4_K gate/up + Q8_0
  down target for `2304 -> 896 -> 2304`, eight of 64 experts, using SwiGLU and
  weighted summation.

For the initial target-quant MoE implementation, prefer a
correctness-oriented Mellum-specific path. Do not generalize the K-quant GLM
interface by weakening its `% 256` checks: those checks describe its kernels
correctly and Mellum needs different kernels.

**Gate:** each primitive agrees with a small independent scalar reference,
including adversarial router logits and the non-256-divisible expert width.

**Status:** the resident selected-expert MoE decode primitive is complete and
numerically tested. `ds4_gpu_mellum_routed_moe_one_tensor` reuses the proven
scalar Q4_K gate/up SwiGLU kernel and adds a Mellum-specific Q8_0 down kernel.
It validates explicit expert and row byte strides, so its resident layout can
later be replaced with selected-expert address-table bindings without changing
the graph-facing contract. The focused Metal test uses `512 -> 288 -> 128`,
three noncontiguous selected experts, and independent Q4_K/Q8_0 scalar
references; 288 intentionally exercises the Q8-compatible, non-256-divisible
intermediate case that Mellum's 896 width requires. On Apple M5 Max it measured
maximum absolute error `1.91e-6` for the weighted SwiGLU intermediate and
`1.91e-5` for the summed output. It is resident-only: SSD address-table support
is deliberately deferred rather than implied by the stride-aware interface.

The router and layer-policy parts of this step are now also complete.
`ds4_gpu_mellum_router_select_tensor` implements Mellum's bias-free softmax,
deterministic lower-ID tie-break, top-8 selection, and selected-probability
renormalization (scale one). Its Metal regression independently checks 64
experts across ordinary, tied, and extreme logits. The shared layer-policy
helpers prove all 28 slots follow `[sliding, sliding, sliding, full] x 7`:
21 ordinary-RoPE sliding layers and seven YaRN full-attention layers. The
Ungated four-KV-head GQA decode is now also proven: the resident F16-KV
primitive supports absolute-position ring addressing for both SWA and full
layers, and its 32-query-head/four-KV-head wrapped-cache test agrees with the
scalar softmax reference to `4.47e-8` maximum absolute error. It is
correctness-first, one SIMD group per query head; only a later measurement
should justify split-K specialization.

**Status update:** the graph-facing descriptor now exists and is constructed
during Mellum loading. Every one of the 28 entries binds the canonical Q/K/V,
norm, output, router, and three expert tensors; it records Q4/Q8-independent
expert byte spans and the `1024`-row SWA versus session-context full-KV policy.
The pinned Q8 GGUF constructs all descriptors successfully through
`--inspect`. No execution path consumes them yet, so the explicit
inspect-only inference guard remains in place.

**Status update:** the descriptor-defined K/V cache write/read contract is
now closed. Mellum stores the current four-head K/V projection as F16 at the
absolute-position ring row, then its ungated GQA primitive reads that same row
alongside prior wrapped history. The GQA regression deliberately leaves the
current GPU cache row stale, writes it through the new store primitive, and
compares the resulting 32-head attention output with a scalar reference. The
next seam is dense Q8 projection plus RMSNorm/RoPE—not cache semantics.

#### Deep-review checkpoint and next tasks (2026-08-01)

The primitive tests are internally green, but they now serve two different
artifacts and must not be conflated:

- the pinned numerical oracle is all Q8_0; and
- the completed selected-expert primitive is Q4_K gate/up plus Q8_0 down,
  matching the proposed smaller deployment artifact.

Consequently the current MoE primitive cannot complete a layer from the pinned
Q8 model. This does not invalidate it, but it changes the integration order.
The next steps are:

1. Assemble and independently test the attention prelude: hidden-state
   RMSNorm, Q8 Q/K/V projections, per-head Q/K RMSNorm, descriptor-selected
   ordinary RoPE versus YaRN, F16 KV store, ungated GQA, Q8 output projection,
   and residual. Stop at the post-attention hidden state.
2. Add a correctness-first Q8_0 gate/up selected-expert path, preferably behind
   the same graph-facing MoE contract, so the pinned all-Q8 model can provide an
   end-to-end numerical gate. Do not make the mixed-quant kernel pretend to
   support Q8 inputs.
3. Only then assemble one complete Q8 layer and compare a real-model
   intermediate. Keep the inspect-only guard until an incremental authentic
   model comparison is green.

Review corrections also made the descriptor slice-aware, replaced the
`UINT32_MAX` full-cache sentinel with `kv_window == 0` meaning session context,
and removed redundant per-entry layer/YaRN state. The F32-to-F16 KV store is
shared with Laguna under neutral internal names. GQA now rejects nonpositive
scales and uses widened ring-position addition. Split-K optimization, prefill,
and SSD address-table work remain deliberately deferred.

**Status update:** step 1 is now implemented as the tested
`ds4_gpu_mellum_attention_decode_tensor` composition. It performs weighted
hidden-state RMSNorm; Q8_0 Q/K/V projections; per-head Q/K RMSNorm plus
NeoX RoPE using the supplied ordinary-or-YaRN parameters; F16 KV storage;
ungated GQA; Q8_0 attention output projection; and the residual add. Its
two-query-head/one-KV-head wrapped-cache scalar test covers a nonzero ordinary
RoPE position and reports maximum absolute errors of `2.15e-6` (Q),
`2.03e-6` (K), `9.54e-7` (V), `4.77e-7` (attention), and `2.86e-6` (residual
output). The API is intentionally not yet attached to `ds4_engine`, so this is
an integration seam rather than permission to remove the inspect-only guard.

**Status update:** step 2 is now implemented for the numerical oracle as the
separate `ds4_gpu_mellum_q8_0_routed_moe_one_tensor` path. It deliberately
does not widen the Q4_K/Q8_0 deployment primitive: the two gate/up layouts are
not interchangeable. The new selected-expert Q8_0 gate/up Metal kernel shares
the existing Q8_0 down projection and is checked by an independent scalar
`512 -> 288 -> 128`, 4-expert/3-selected, noncontiguous-expert regression.
Maximum absolute drift was `2.86e-6` in the SwiGLU intermediate and `5.72e-6`
after the down projection. This gives the pinned all-Q8 GGUF a real complete
MoE numerical seam, but it is still not connected to `ds4_engine`.

### Step 4: Bring up resident one-token Metal decode incrementally

Implement a separate Mellum graph and compare after each layer where diagnostic
references are available. The graph needs:

- Q/K per-head RMS normalization before RoPE;
- correct per-layer full/SWA choice and dual RoPE parameters;
- four-KV-head GQA;
- bias-free softmax/top-8/renormalized routing;
- routed SwiGLU with no shared expert;
- final norm and output projection.

Use the pinned Q8_0 artifact first so architecture work is not mixed with a new
quantization. Its Q8_0 selected-expert path is now scalar-tested; the existing
mixed Q4_K/Q8_0 MoE primitive remains for the deployment target, not the
oracle. The immediate task is therefore to compose one full Q8 layer from the
attention prelude, router, and Q8 MoE, then compare an authentic intermediate
before attaching it to the engine.

**Status update:** the Q8 layer composition now exists as
`ds4_gpu_mellum_q8_0_layer_decode_tensor`. It preserves Mellum's pre-norm
residual order: attention+residual, FFN RMSNorm, F32 router/top-k, Q8 routed
SwiGLU, then the FFN residual. A small Q8-experts/F32-router scalar regression reports zero
attention-residual drift (the test deliberately uses a zero attention output to
focus this wiring), `1.19e-7` FFN-norm drift, `9.54e-7` router-logit drift,
`1.14e-5` MoE-intermediate drift, and `3.05e-5` final-layer drift. This closes
the synthetic full-layer seam. It is not an authentic-model comparison: the
existing llama.cpp fixture contains final first-token log probabilities, but no
captured `ffn_out`/layer checkpoint. Capture one such checkpoint next, then
run the all-Q8 layer against the actual GGUF before considering engine
attachment.

**Reference-capture update:** an isolated build of the pinned llama.cpp source
revision (`0e4a03622`) now provides that first checkpoint without modifying the
reference source. Its stock `llama-eval-callback` example runs the exact
26-token `python_add` ChatML fixture and exposes the named post-layer tensor as
`l_out-0` (rather than `ffn_out-0`): it is the output after the second residual
and before layer 1's RMSNorm. With all layers on Metal, the layer-0 prefill
tensor has aggregate sum `-296.095093`; its final-token sampled values are
`[-0.0022, -0.0350, 0.0154, ..., 0.0193, 0.0265, -0.0070]`. The same graph
also exposes `ffn_moe_topk-0` and `ffn_moe_weights_norm-0`; their selected
weights sum to one for every one of the 26 positions (aggregate `25.999994`).

This changes the immediate implementation task slightly: add a **diagnostic,
non-engine** layer-0 probe that consumes the known 26 token IDs sequentially,
uses the actual GGUF tensor offsets and F16 KV ring, and emits `l_out` and
router checksums/samples for comparison. A batched ds4 prefill graph is not
needed for this first gate, and generation remains prohibited. The stock
callback displays samples and aggregate sums rather than a raw tensor dump, so
its initial acceptance criterion should be exact token count/router IDs plus
bounded agreement on the documented aggregate/sample statistics; a compact raw
checkpoint dumper can strengthen this after the tokenwise probe works.

**Authentic-probe update:** `./ds4 --mellum-layer0-probe --model MODEL` now
opens Mellum in inspect mode, creates no session, and replays those 26 IDs
through layer 0 one token at a time. It deliberately maps the Q8 GGUF only for
this diagnostic and keeps the normal Mellum execution guard in place. The probe
uses the actual F32 router (a correction to the earlier assumption that every
projection was Q8_0), and produces the final router exactly:
`18:0.3846770,58:0.1485689,46:0.1022641,47:0.0976073,55:0.0808460,6:0.0646314,52:0.0609993,27:0.0604062`.
Its final hidden sample is
`[-0.0021954, -0.0350052, 0.0154191, ..., 0.0192883, 0.0265708, -0.0069613]`,
which agrees with the rounded llama.cpp display. Its F32 aggregate is
`-296.220612` versus llama.cpp's `-296.095093`: a `0.125519` difference over
59,904 values, or about `4.2e-4` relative to the checkpoint magnitude. This is
a successful structural/intermediate seam, but not yet the strict numerical
gate: obtain a raw llama.cpp tensor dump next, compare max/RMS error rather
than a rounded checksum, and only then consider engine attachment.

**Raw layer-0 gate:** the llama.cpp callback capture is now retained as
`tests/test-vectors/mellum-llama-cpp/l-out-0.f32`, a 239,616-byte (26 x 2,304
little-endian F32) checkpoint with SHA-256
`5a99d699c83a1a5417f9175f46e30c343026767372546001a112aef34ce49243`.  It was
captured at llama.cpp `0e4a0362239713ea95a6864a17a8de4b0ad90d62` from the
exact Q8 GGUF and `python_add` fixture. The inspect-only ds4 probe can write
its corresponding output with `--mellum-layer0-probe-out FILE`; the option
stages a sibling temporary file and atomically replaces `FILE` only after all
26 outputs are closed successfully. `tests/check_mellum_layer0_oracle.py`
checks the fixture hash, exact byte count, and acceptance envelope. On the
same fixture, the two files compare as follows:

| Measure | Result |
| --- | ---: |
| Maximum absolute error | `0.00493813` (token 15, channel 1236; ds4 `-8.95174026`, llama.cpp `-8.94680214`) |
| RMS error | `9.80462e-5` |
| Mean absolute error | `3.69404e-5` |
| Token 0 maximum / RMS | `1.40302e-4` / `3.80239e-5` |
| Final token maximum / RMS | `0.00353670` / `1.03951e-4` |

The **final-token** router has the same selected-expert order and agrees at
the probe's displayed seven decimal places for every normalized weight.
Routing for the preceding 25 positions has not yet been captured, so this is
not a bitwise or all-token router claim. The small output difference is
consistent with comparing ds4's sequential decode and F16 KV ring against
llama.cpp's batched prefill/reduction schedule, but the raw output envelope is
the evidence—not an assertion that every possible source of drift is already
eliminated. Record it as the **layer-0 acceptance envelope** for this exact
reference platform: RMS at most `1.25e-4` and maximum absolute error at most
`6e-3`. Do not relax that envelope silently when adding layers.

**Replan:** first capture and pin `l_out-27` from the same llama.cpp callback,
then build a second diagnostic-only, session-free probe spanning all 28 layers.
Its descriptor constructor must choose ordinary RoPE and a 1,024-token ring
for layers 2/26, then YaRN (`freq_scale=1/16`, extension factor 1, attention
factor 1.2772589) and a full-context ring for layers 3/27. Test both
transitions explicitly. Only then add the final RMSNorm and Q8 output head
and compare final logits. Generation, ordinary sessions, and SSD streaming
remain out of scope until this all-layer numerical gate and a short greedy
continuation fixture are green.

**All-layer diagnostic result:** the `l_out-27` final-token callback capture
is pinned at `tests/test-vectors/mellum-llama-cpp/l-out-27-last.f32` (2,304
little-endian F32 values; SHA-256
`050436ca257f8ebe36656f32785b9649ddf8b8ad75a08ec47f005ea588b72ebb`). The
new `--mellum-all-layers-probe[-out FILE]` replays the same 26 IDs through all
28 descriptors without creating a session. It allocates an independent F16 KV
store per layer, uses a 1,024-token ring and ordinary RoPE for sliding layers,
and uses the full current context plus YaRN for layers 3, 7, ..., 27. The
policy tests explicitly cover the 2-to-3 and 26-to-27 transitions.

The first full comparison was **red**: ds4's final `l_out-27` had sum
`-1923.217410`, versus llama.cpp's `-1622.638965`; maximum absolute error was
`643.940552` (channel 368), RMS `27.803057`, and MAE `18.852392`.

**Localization and resolution:** layer 1 remained within `9.31e-5` RMS, while
the first full/YaRN layer (layer 3) initially showed `0.02249` RMS. Its ds4
post-RoPE Q/K vectors were directionally identical to llama.cpp but larger by
exactly `1.2772589`, Mellum's configured YaRN attention factor. The shared
Metal YaRN helper already multiplies by its internal `1 + 0.1 log(16)` mscale;
passing Mellum's effective attention factor into that helper applied it twice.
The full-layer descriptor now passes a neutral external factor and retains
the model factor as the helper's effective internal scale. Layer 3 then agrees
at `1.23e-4` RMS (post-attention `1.03e-4` RMS), confirming the correction.

The original layer-27 callback checkpoint was made by batched llama.cpp
prefill, while ds4's diagnostic intentionally decodes one token at a time.
Those two schedules differ at layer 27 by `2.18424` RMS after the YaRN repair,
despite their early-layer agreement. A callback variant that decodes the same
26 tokens one at a time provides the matching-schedule checkpoint now pinned
as `tests/test-vectors/mellum-llama-cpp/l-out-27-tokenwise.f32` (SHA-256
`4ae7a46e409d6e3bf760ef8f7c671f32ff16d0798b8cd3dd25fe5fcbb3f086fe`). Against
that oracle, ds4 measures maximum absolute error `1.18909`, RMS `0.0437417`,
and MAE `0.0276634`. This clears the **tokenwise all-layer acceptance gate**
(max `<= 1.5`, RMS `<= 0.06`) for this platform and fixture. The comparator
retains both layer-27 checkpoints so the schedule distinction remains visible.

**Stopping boundary:** resident output-head integration, ordinary sessions,
greedy generation, and SSD streaming are deliberately not started. The next
authorized work after this research boundary is ordinary-session design only;
it does not authorize implementation. Greedy generation, resident prefill, and
SSD streaming remain out of scope.

**Engine-graph promotion:** the accepted 28-layer tokenwise composition is now
held in a reusable private `ds4_engine` Mellum decode state. It owns the Q8
layer descriptors, shared activation/router scratch, F16 KV tensors for every
layer, and a resettable token position; sliding layers retain their 1,024-row
rings while full-attention layers use the requested diagnostic context. The
inspect-only whole-model probe resets and reuses that state rather than
reallocating its graph per invocation. Re-running the pinned 26-token fixture
through this engine-owned state preserves the tokenwise layer-27 gate exactly:
maximum absolute error `1.18908691`, RMS `0.0437416537`. This state is not a
`ds4_session`, does not share the global KV-cache directory/store, and remains
reachable only through the inspect probe. At this stage it had no final
RMSNorm/output projection; token emission, greedy generation, resident prefill,
and SSD streaming remain excluded.

**Output-head promotion:** the same inspect-only engine state now owns F32
output-norm and raw-logit tensors. The new `--mellum-logits-probe[-out FILE]`
replays the pinned fixture through all 28 layers, final RMSNorm, and the Q8_0
output matrix, then writes or displays raw logits only. It neither computes an
argmax for an execution path nor emits a token. A matching tokenwise llama.cpp `result_output`
checkpoint is pinned as `result-output-tokenwise.f32` (98,304 F32 values;
SHA-256 `4ec7f9c838058fc267ec0a2f117aa4db523950df1621f61d25afcf63e3be2b1c`).
The initial ds4 comparison is maximum absolute error `0.0175094604`, RMS
`0.009415312`, and MAE `0.00922990179`; the explicit output-head gate is
maximum `<= 0.025` and RMS `<= 0.012`. The error is expected to include the
accepted tokenwise layer-stack drift; no token-quality inference or generation
claim follows from this raw-logit gate.

**Inspect-only ranking:** `--mellum-logits-probe-top-k N` ranks the raw logits
from that same fixed probe and reports the IDs and values without passing them
to a sampler, session, or token emitter. On the pinned fixture, the top 16 IDs
exactly preserve the llama.cpp order: `910, 2116, 889, 629, 3756, 45742, 2591,
2190, 1017, 433, 4697, 75, 3969, 20320, 21177, 7846`. This establishes only a
diagnostic ranking seam; it does not authorize generation.

**Next-phase session design (not implemented):** the first session-capable
Mellum change should be a separate state type, rather than widening the fixed
probe state. It needs one F16 key/value pair per layer, with a 1,024-row ring
for each sliding layer and a caller-chosen full-context capacity for layers 3,
7, 11, 15, 19, 23, and 27. The private probe's hard-coded fixture position and
its CPU activation round-trips must not become session behavior. Before any
decode or prefill is enabled, the design review must specify: (1) allocation
and teardown ownership under `ds4_session`; (2) reset, context-limit, and
sliding-ring semantics; (3) an explicit no-output/no-emission seam; and (4)
the persistent-cache isolation rule: run anything that creates KV state as
`HOME=/tmp/mellum2 ./ds4 ...`, because ds4's KV cache directory is shared
across models. The first implementation gate is session create/close and KV
layout inspection only—no prompt evaluation, prefill, token selection, or
generation.

**KV-layout allocation gate:** `--mellum-kv-layout-probe --ctx N` now
allocates and releases the intended per-layer F16 key/value tensors under
inspect mode, but does not create a `ds4_session` or evaluate a token. At
`--ctx 4096`, it validates 21 sliding layers at 1,024 rows and seven full
layers at 4,096 rows, with a KV dimension of 512 and a total allocation of
102,760,448 bytes (98 MiB). This establishes the storage contract without
reusing the generic session graph, which is structurally wrong for Mellum.

**Layout-only session lifecycle gate:**
`--mellum-session-lifecycle-probe --ctx N` now creates and releases a
dedicated `ds4_session` carrying that Mellum KV layout. The generic Metal graph
is deliberately not marked ready or allocated for this family. Every public
token-execution entry point used by the gate rejects the session before any
kernel work, and the probe verifies that rejection before teardown. At
`--ctx 4096` the lifecycle completes with 21 sliding and seven full layers.
This is a creation/teardown contract only: no prompt, prefill, logits, token
selection, or generation has been enabled.

**Boundary hardening review:** layout-only sessions intentionally allocate no
generic graph, output buffer, or sampling workspace. The review found that a
public argmax helper could otherwise dereference that absent logits buffer.
It now returns the normal invalid-token result (`-1`), alongside the other
selection and log-probability APIs. Payload, snapshot, and staged-payload APIs
also reject this session state before they can touch an unrelated graph layout.
The lifecycle probe now exercises token evaluation, eval-and-argmax, argmax,
sampling, top-logprobs, and token-logprob rejection before releasing its KV
tensors. This remains a safety boundary, not generation work.

**Decode-session contract:** the next inspect-only gate is one-token decode,
not a promotion of the reusable fixed-fixture probe. A decode session owns its
own activation/router scratch, F16 K/V tensors, absolute position, final-norm
scratch, and raw-logit tensor. It uses the same 21 sliding 1,024-row rings and
seven full-context tensors as the layout contract. Its position must equal the
session checkpoint length before each decode; a successful decode appends the
input token and produces only the resulting raw logits. Invalidating this
inspect session resets its checkpoint and decode position to zero; partial
rewind is deliberately reduced to that reset rather than exposing replay
semantics. The engine-owned probe state is deliberately not shared across
sessions.

The allowed entry point is an inspect-only fixture gate,
`--mellum-session-decode-probe[-out FILE]`. It drives the public one-token
session-evaluation API over the pinned 26 tokens and can write the final raw
logit row for the existing oracle checker. `ds4_session_sync` and all prefill,
batch, layer-slice, snapshot, payload, argmax, sampling, and log-probability
paths continue to reject Mellum sessions. The ordinary Mellum loader remains
inspect-only, and no CLI prompt path reaches this state. This makes the raw
logit oracle the acceptance criterion without authorizing token selection,
emission, generation, or SSD streaming. At `--ctx 4096`, the session gate's
26-token final row matches the pinned oracle with maximum absolute error
`0.0175094604` and RMS `0.009415312`, identical to the engine-owned diagnostic.

**Session-isolation gate:** `--mellum-session-isolation-probe --ctx 4096`
first records independent baselines for two fixture streams that differ at
their first token, then interleaves those streams in two sessions on one engine.
Each final 98,304-logit row must be finite and F32-bit-identical to its own
baseline, while both sessions continue to reject token selection. The divergent
schedule catches aliased KV rows or token positions; it does not claim to test
concurrent scratch safety or multi-session memory efficiency. It does not enable
batching, prefill, generation, or SSD streaming.

**First interactive resident path:** only `ds4-agent` can request the explicit
engine capability that creates an interactive Mellum session backed by the
independently owned, validated tokenwise decode state. Ordinary `ds4`, server,
benchmark, evaluation, and generic inspect opens remain outside that boundary;
an inspect-opened engine still creates only a layout session whose execution
and selection APIs reject. `ds4_session_sync()` is intentionally sequential:
it extends a matching live checkpoint token by token, or invalidates and
replays a changed prompt from token zero. It reports normal sync progress and
honours cancellation, but it neither allocates a generic batch graph nor
claims prefill performance. Interactive sessions may use argmax, sampling,
and log-probability helpers; the fixture-only decode sessions retain their
strict no-selection boundary. Rewind remains a full reset, so arbitrary
transcript rewrites replay safely rather than attempting partial KV surgery.

The loader explicitly refuses CPU, layer slicing, distributed or tensor
parallel placement, multi-GPU placement, and `--ssd-streaming` for this path.
Mellum KV payload/snapshot persistence, mixed/batched prefill, session batching,
layer-slice APIs, and speculative decode remain unavailable. The agent now
saves Mellum sessions in its existing zero-payload format: rendered transcript
and metadata are durable, and restore replays the transcript sequentially to
rebuild resident KV. This keeps `/save`, `/switch`, exit-save, and the system
prompt cache usable without pretending the Mellum tensors implement the shared
payload ABI. The first agent run must still use an isolated home because ds4's
cache namespace is not model-specific:

```sh
HOME=/tmp/mellum2 ./ds4-agent --metal -m MODEL --non-interactive \
  --raw-prompt --nothink --temp 0 -n 32 \
  -p 'Count upward from one, placing a space after every integer.'
```

On the M5 Max (128 GB) with the pinned Q8 artifact, the final run completed a
13-token sequential sync in 1,110 ms (**11.7 t/s**) and emitted 32 capped
tokens in 2,164 ms (**14.8 t/s**, including the first immediately sampled
token). The same model mapping reported 12.03 GiB resident model and 0.07 GiB
KV at the agent's default 100k context, or 12.10 GiB planned total. A separate
smoke prompt, `Reply with only: hello`, produced `hello world`; this proves the
agent sampling/eval loop works, not instruction quality. Initial process setup
had already paged the model in, so the capped trace's ~157 ms residency request
is not a cold-start claim.

This is not a like-for-like Laguna benchmark: it is short, one-token-at-a-time
prefill on a 128 GB M5 Max. Still, it gives the correct direction. The Laguna
XS record on the same class of machine reports 379.90 t/s resident prefill and
84.59 t/s resident generation; its later streamed-Q3 matched run reports
406.56 t/s prefill and 39.82 t/s generation (M1 Pro 32 GB: 67.86 / 14.87
t/s). Mellum's **14.8 t/s** decode is therefore far below resident Laguna XS
and roughly the 32 GB streamed-Laguna decode threshold, before Mellum has any
SSD path or batched prefill. The next performance gate is a reproducible
long-prompt/capped-decode harness, followed by batched resident prefill and
only then selected-expert SSD streaming.

The inspect-only `--mellum-interactive-session-probe` is the acceptance gate
for this promotion. On the pinned Q8 model it verifies that a generic inspect
session remains layout-only, then privately exercises fresh sequential sync,
suffix extension, divergent-prompt reset/replay, cancellation after a valid
two-token checkpoint and exact resume, full-reset rewind, argmax/sampling and
log-probability selection, and transcript-only persistence. The rebuilt and
resumed final logits are F32-bit-identical to independent baselines. Ordinary
`ds4` opening of the same model is separately rejected before it can enter the
incompatible generic graph.

**GPU-resident decode experiment:** the first 14.8 t/s agent trace was not a
model limit. Its normal decode loop submitted the composed Metal helpers as
many individually owned command buffers and copied the 2,304-F32 hidden state
from GPU to CPU and back between every Mellum layer. The improved normal path
keeps two activation buffers on GPU, ping-pongs them through the 28 layers,
and surrounds the complete layer stack plus output head with one existing
Metal command batch. The layer-trace/oracle mode deliberately retains its
synchronous readbacks, so diagnostic semantics are unchanged.

On the identical 13-token raw prompt and 32-token greedy cap, the trace from
the first generated token to the 32nd covers 31 completed decode evaluations.
It fell from 2.164 s (**14.3 t/s**) to 278 ms (**111.5 t/s**): a **7.8x**
steady-decode improvement. The initial 13-token sync remained about 1.2 s,
which is expected to include first-use pipeline work and is not yet a prefill
result. The pre- and post-change session raw-logit files compare byte-for-byte
identically; the session-logit oracle remains at `0.0175094604` maximum
absolute / `0.009415312` RMS drift, and the all-layer tokenwise oracle remains
at `1.18908691` maximum absolute / `0.0437416537` RMS drift. Interactive and
two-session isolation probes still pass.

The next measurement should use three to five warmed repetitions and record
wall time, sync time, and generation time separately; a Metal System Trace (or
the existing GPU-busy profiler) should confirm the command-buffer collapse.
Only then should work move to GPU token embedding or true resident batched
prefill. SSD work remains out of scope.

**Post-batch review and speed plan:** Sol's review found no correctness defect
in the GPU-resident path: the ping-pong buffers remain distinct, the final
output head consumes the current activation rather than relying on layer-count
parity, every Mellum primitive joins the existing command batch, and the
diagnostic path stays separate. The review did add one ownership regression:
an interactive Mellum evaluation attempted while a caller owns an empty Metal
batch must fail, preserve that batch, reset the session safely, and replay
exactly after the caller ends it. The live interactive probe now verifies this.

The next steps are deliberately ordered:

1. Run three to five warmed, fixed-prompt/capped-decode repetitions; record
   sync, generation, and wall time separately, and confirm the command-buffer
   collapse with Metal System Trace or the GPU-busy profiler.
2. Move the Q8 token embedding into the same owned Metal batch, removing the
   remaining CPU embedding and 9 KiB upload; accept only byte-identical raw
   logits and the existing oracle gates.
3. Build true resident batched prefill while preserving the 21 sliding / 7
   full KV policy. This is the main remaining agent-latency bottleneck.
4. Use the trace to target the dominant Q8 decode kernel or output head; do
   not speculate about fusion before that evidence exists.
5. Revisit the validated mixed Q4 gate/up plus Q8-down deployment artifact and
   remeasure quality/bandwidth. SSD streaming remains after resident parity,
   not before it.

**Review hardening:** Sol's post-commit review found that the initial oracle
checker could accidentally accept a NaN because comparisons with NaN are false,
and its fixture paths depended on the caller's current directory. The checker
now rejects non-finite reference, actual, and delta values before accumulation,
resolves fixtures from its own location, catches fixture-read failures, and is
regression-tested for valid, NaN, and infinity output from outside the repo.
The documented command explicitly uses `python3`, avoiding an executable-bit
assumption. The exact callback patch, Metal build configuration, full fixed
prompt/token sequence, batch-size-one tokenwise schedule, model identity, and
expected SHA-256 now live alongside the F32 fixtures in
`tests/test-vectors/mellum-llama-cpp/capture-layer-output.patch` and its README
section. This closes fixture provenance without expanding runtime scope.

**Gate:** the first greedy token matches llama.cpp on several prompts, and any
logit or intermediate drift is measured and explained rather than hidden by a
sampling comparison.

### Step 5: Add resident prefill and KV correctness

Add the batch graph and layer-specific KV policy. Validate both sides of the
1,024-token SWA boundary and ensure only full layers retain the entire context.

**Gate:** short continuation parity, a prompt crossing 1,024 tokens, and a
longer chunked-prefill run all agree with the reference within an explicit
numerical tolerance.

### Later: choose the release quant and SSD path

After resident Q8 is green, make a measured choice:

1. Add resident Q5_0 and support the official Q4_K_M artifact as published; or
2. Produce a more uniform Mellum GGUF intended for ds4 and validate its quality.

Only then extend SSD streaming for the chosen down-projection type and run the
same resident-versus-streamed A/B gate used for Laguna XS.

## Open questions

1. Is the local GRPO checkpoint the final intended product target, or should
   the public Thinking release become the shipped default after architecture
   bring-up?
2. What generated the local checkpoint's flattened `qwen3_moe` config, and is
   applying the canonical Mellum config to those weights the intended runtime
   contract?
3. Should the first release artifact preserve the official Q4_K_M mixture, or
   favor a uniform layout designed for DwarfStar streaming?
4. Which EOS behavior matches JetBrains' evaluation and serving setup: ID 0,
   ID 28, or both?
5. Does resident Q4 plus layer-specific KV already satisfy the desired 16 GB
   target, making SSD streaming optional rather than required?
6. What hit rate and cache footprint does Mellum's 8-of-64 routing produce on
   coding-agent workloads?

## Research log

### 2026-08-01

- Created the `mellum-2.1` worktree from `laguna-xs2.1`.
- Located and inspected the local GRPO safetensor checkpoint.
- Compared its config and tensor index with the public Thinking release.
- Verified the canonical per-layer attention and RoPE behavior in the Mellum
  Transformers implementation and the technical report.
- Verified local llama.cpp has a dedicated Mellum model graph.
- Inspected the published Q4_K_M GGUF header and tensor types without
  downloading the full file.
- Revised the feasibility assessment: resident support remains favorable;
  SSD streaming needs new mixed-quant expert compute support.
- Downloaded, checksummed, and inspected the pinned public Q8_0 GGUF.
- Captured its metadata, chat-template tokenization, deterministic greedy
  continuations, first-token IDs, and explicit EOS behavior.
- Added Mellum loader-only support: `--inspect` validates the public Q8_0
  metadata, 28-layer SWA schedule, and all 339 tensor bindings without
  building a Metal graph. Ordinary execution remains explicitly rejected until
  the Mellum decode graph lands.
- Audited the canonical llama.cpp graph against ds4's Laguna and GLM paths.
  Confirmed the exact bias-free normalized-softmax router, layer-selective
  YaRN, and absence of attention gating/shared experts.
- Tried ds4's tokenizer-only path against the pinned rendered prompt. It failed
  before tokenization because Mellum currently falls through to DeepSeek
  vocabulary initialization. Moved Mellum tokenizer/chat support ahead of the
  graph.
- Confirmed the existing GLM routed-MoE Metal interface cannot execute the Q8
  bring-up artifact and rejects Mellum's 896-wide expert intermediate. Revised
  the plan to prove a Mellum-specific selected-expert primitive before the full
  graph; the first implementation targeted Q4_K gate/up plus Q8_0 down, leaving
  the pinned all-Q8 oracle path as an explicit follow-up.
- Added native Mellum tokenizer and ChatML support: GGUF BOS/EOS metadata,
  `mellum2` digit-aware BPE splitting, literal ChatML/think tokens, and no
  implicit BOS in the chat prefix. The 24-token no-thinking fixture and the
  thinking prefix now pass a dedicated tokenizer-only regression test.
- Built local llama.cpp `llama-server` and captured a pinned top-10
  first-token log-probability fixture for the Python, JavaScript, and France
  prompts. It uses the exact rendered ChatML prompts and Q8 GGUF revision.
- Recorded a 32 GB quality-oriented Mellum deployment target: imatrix Q4_K
  gate/up experts except for the final six Q8 layers, with all down experts and
  control-path tensors preserved at Q8/F32. Estimated file size is 9.33 GiB;
  a 16 GB path remains conditional on future SSD streaming support.
- Implemented and passed a resident Mellum Q4_K gate/up + Q8_0 down
  selected-expert Metal primitive. The synthetic `512 -> 288 -> 128` test uses
  noncontiguous experts and agrees with its independent scalar reference to
  below `2e-5` maximum absolute output error after expanding the fixture to
  cross both kernels' 256-thread stride. The interface carries explicit
  projection offsets/strides to make the next address-table streaming step
  mechanically local, but it does not yet stream expert weights.
- Implemented the bias-free Mellum softmax/top-8 router and the exact 28-layer
  SWA/full policy helpers. The isolated Metal suite passes the router's
  ordinary, tied, and extreme-logit scalar comparisons alongside the Q4_K/Q8_0
  routed-MoE regression.
- Added and passed an ungated four-KV-head GQA decode primitive over a wrapped
  F16 KV cache. This is separate from Laguna's gated attention and establishes
  the resident attention contract needed for the first Mellum decode layer.
- Added a checked, per-layer Mellum decode descriptor during engine loading and
  verified it against the pinned public Q8 GGUF with `--inspect`. This is the
  seam between validated model binding and the still-future resident graph.
- Added the Mellum F32-to-F16 KV ring-store primitive and extended the wrapped
  GQA regression to prove its write/read integration before dense projection
  assembly.
- Deep-reviewed the descriptor and primitive boundary. Made sliced Mellum loads
  descriptor-safe, replaced the full-cache maximum-integer sentinel with an
  explicit zero-window policy, generalized the shared KV-store internals, and
  tightened GQA scale/ring arithmetic validation. Replanned the next gate around
  attention-only assembly followed by a Q8 gate/up oracle path; the completed
  mixed Q4_K/Q8_0 MoE primitive alone cannot execute the pinned all-Q8 model.
- Added a Mellum-specific one-token attention composition around the existing
  Q8, per-head norm/RoPE, KV, GQA, and residual primitives. Its independent
  wrapped-cache scalar regression is green; engine attachment and authentic
  model intermediates remain deliberately deferred until the Q8 gate/up MoE
  oracle path exists.
- Added the all-Q8_0 selected-expert SwiGLU oracle path required by the pinned
  GGUF, rather than incorrectly feeding Q8 gate/up rows to the Q4_K kernel.
  Its noncontiguous-expert scalar regression reports `2.86e-6` maximum
  intermediate drift and `5.72e-6` output drift. The next numerical gate is a
  complete real Q8 layer and an authentic-model intermediate, not generation,
  engine attachment, or SSD streaming.
- Added the all-Q8_0 complete-layer diagnostic composition and a scalar
  regression covering its post-attention wiring. It reports `3.05e-5` maximum
  final-layer drift. The next task narrows to capturing a llama.cpp layer
  checkpoint and comparing the same real Q8 layer; inspect-only remains in
  force until that gate is green.
- Built the pinned llama.cpp evaluation-callback example in an isolated
  temporary directory and captured the first authentic checkpoint: layer 0's
  `l_out-0` on the exact 26-token `python_add` fixture. It establishes the
  reference tensor name, aggregate sum, sampled final-token values, and the
  accompanying router state needed for a tokenwise ds4 diagnostic probe.
- Corrected the oracle contract after inspecting the actual GGUF: Mellum's
  dense/expert projections are Q8_0, but its router is F32. The full-layer
  composition and scalar test now exercise that F32 router path. Added the
  inspect-only `--mellum-layer0-probe`; it reproduces the authentic final
  router selection/weights and rounded `l_out-0` sample on the 26-token
  fixture. A raw reference tensor is the remaining requirement for a strict
  max/RMS comparison.
- Added raw F32 output support to that diagnostic and compared all 59,904
  layer-0 values with the llama.cpp callback checkpoint. The result is
  `9.80462e-5` RMS and `0.00493813` maximum absolute error, while the final
  router remains exact. This clears the first authentic numeric gate and
  narrows the next step to a session-free all-layer diagnostic, with a later
  checkpoint or final logits as its reference.
- Sol review corrected the raw-gate design before the all-layer work: committed
  the 239,616-byte llama.cpp layer-0 fixture with its SHA-256, added a
  dependency-free comparator enforcing the documented envelope, made probe
  output atomic, and limited the router claim to the displayed final-token
  values. Reordered the next work so a pinned layer-27 checkpoint and
  layer-aware RoPE/KV constructor precede the all-layer diagnostic.
- Pinned a final-token layer-27 llama.cpp checkpoint and added the
  session-free `--mellum-all-layers-probe`. It exercises 28 independent
  layer KV states with the real sliding/full RoPE policy, but fails its first
  authentic all-layer comparison badly (`27.803057` RMS). Generation remains
  gated; next capture/diagnostic work must locate the first divergent layer.
- Localized that discrepancy to a doubled YaRN mscale at layer 3, corrected
  the full-layer descriptor, and added final-token layer, attention, and Q/K
  diagnostic traces. The remaining batched-prefill comparison drift was shown
  to be schedule-specific: the pinned matching tokenwise layer-27 reference
  passes at `0.0437417` RMS and `1.18909` maximum absolute error. Stop here
  before output-head, generation, or SSD-streaming work.
- Moved Q8 token embedding into the resident Metal command batch. It removes
  the final CPU embedding/readback handoff and is byte-identical to the prior
  resident decode output; steady decode remains about 111 t/s, confirming that
  the layer compute rather than the 9 KiB embedding upload dominates.
- Added bounded (32-token) command-batched sequential session sync. This is
  deliberately not described as true multi-token prefill: it preserves one
  autoregressive decode graph per token and merely amortizes command submission.
  The existing interactive/session-isolation checks and pinned logits/all-layer
  oracles remain unchanged. The dedicated 1,030-token probe now compares this
  schedule bit-for-bit with per-token decode across the 1,024-token SWA ring
  boundary (21 sliding layers and 7 full layers). This clears the long-context
  correctness gate before a layer-major prefill graph is attempted.
- Ran the first normal agent-path measurement on an Apple M5 Max, with the
  pinned Q8_0 model resident, `ctx=4096`, an 11-token raw prompt, and a
  96-token continuation. Prefill was 254.886 ms (cold/short and therefore not
  a useful throughput claim); generated tokens were timestamped at roughly
  10 ms each, or **about 100 t/s**. Metal's accumulated GPU time was 638.1 ms
  for the first 64 command buffers, also about 10 ms/token. Thus the current
  resident decode is GPU-compute/bandwidth bound rather than host submission
  bound; the earlier 31-token microprobe's 111.5 t/s is a compatible,
  shorter-run result. Moving the embedding fully to GPU was exact but did not
  materially change either number.
- Replanned resident optimization accordingly: retain the small 32-token
  submission batch as a correctness-preserving interim prefill path; do not
  spend effort on another host-side micro-optimization. The next performance
  work is a layer-major multi-token graph, which requires batched Mellum
  attention, Q8 projection, router, and sparse-MoE composition rather than a
  larger command batch. Before implementation, profile kernel-level time or
  add a targeted no-output-head comparison so that the expensive primitive is
  identified. The mixed-Q4 deployment assessment remains a separate artifact
  gate: the official Q4_K_M file is only 7.52 GiB but has Q5_0 down experts,
  so it cannot yet exercise the resident Q8-only layer graph.
- Added a warmed in-process resident microprofile to separate final-head cost
  from the decode stack. On the same M5 Max, three 64-token runs measured
  **8.504 ms/token (117.6 t/s)** without output logits and **8.963 ms/token
  (111.6 t/s)** with logits; the final RMSNorm plus 98,304-vocabulary Q8 head
  costs only **0.459 ms/token** (~5% of the full token). This validates the
  replan: optimize the 28-layer attention/routed-MoE stack, not the output
  head. The inspect-only `--mellum-resident-profile` command keeps this result
  reproducible without widening generation scope.
- Began genuine layer-major prefill with the smallest safe primitive: ungated
  Mellum GQA over a token batch. It stages each chunk's F16 K/V, evaluates each
  query causally against the staged rows or pre-existing ring, then commits the
  ring after all queries. The isolated Metal regression crosses a wrapped
  17-row cache and agrees with an independent scalar reference at
  `4.47e-08` maximum absolute error. This is a foundation only: batched Q8
  projections, Q/K norm/RoPE, routing, and sparse-MoE remain before it can
  replace sequential session sync.
- Composed the layer-major Mellum attention prelude from existing batched
  weighted RMSNorm, Q8_0 projections, Q/K norm+RoPE, the new staged GQA, Q8
  output projection, and residual add. Its one-row result agrees with the
  independent attention reference within `2.87e-06`; the mult-row causal/ring
  behavior remains separately locked by the four-token GQA regression. The
  remaining large work item is a batched sparse-MoE path (router and selected
  experts) plus a real-model prefill-versus-decode oracle before session sync
  can opt into this graph.
- Sol review found and corrected a prefill ring safety issue: a batch longer
  than the cache capacity would commit two staged tokens to the same modulo
  F16 KV row concurrently. The primitive now rejects `n_tokens > cache_cap`;
  the numeric regression runs at the exact 17-token boundary and separately
  proves the 18-token call is rejected. Replanning order is now explicit:
  first establish multi-row attention parity (ordinary, sliding-wrap, and
  YaRN/full cases) and batch-ownership behavior; then add the bias-free
  batched router and Q8 selected-expert MoE; only then compare a real layer
  and logits against sequential decode. Session sync remains sequential until
  those gates pass, and its eventual chunk cap is 1,024 to fit every Mellum
  sliding ring.
- Strengthened the attention parity gate after a second Sol review. The
  composed API now rejects an oversized batch before queueing RMSNorm/QKV/RoPE
  work, matching its GQA primitive; the direct API rejection and active
  caller-owned-command-batch result are both covered. The three-row YaRN
  wrapped-ring comparison now starts from a deterministic seed, has measured
  margins (`2e-5` output and `5e-5` F16-KV drift), and verifies the caller-owned
  result/cache against a standalone submission. It measures `3.8147e-6`
  output drift and two one-ULP F16 cache values (`7.62939e-6` max). Remaining
  attention coverage before declaring the gate complete: multi-row ordinary
  pre-wrap and non-wrapped full-cache YaRN cases. These are correctness tests,
  not a reason to delay the next implementation dependency: the bias-free
  batched router.
- Added that token-major batched router. It launches one 256-thread group per
  token and intentionally preserves the one-token implementation's softmax
  reduction, deterministic lower-ID tie break, bitonic top-k, and selected
  weight summation order. The three-row ordinary/tied/extreme regression is
  bitwise-identical to invoking the established one-token router on tensor
  views. This unblocks the next component: a token-major Q8 selected-expert
  SwiGLU/down path and batch layer composition; it is not yet connected to
  session sync.
- Added the companion token-major all-Q8_0 selected-expert MoE primitive. Its
  SwiGLU and down-projection kernels retain the decode reduction order while
  indexing activations, selections, weights, intermediate rows, and output by
  token. A three-token regression with distinct noncontiguous expert choices
  and weights is bitwise-identical to repeating the proven one-token Q8 path.
  Activation-size and stride overflow validation is explicit. The remaining
  implementation step is a complete batch-layer composition that joins
  attention, F32 router projection/selection, this MoE, and the final residual;
  then use a real model to compare that layer and logits against sequential
  decode.
- Composed the Mellum Q8 batch-layer API: staged causal attention, batched F32
  router projection and bias-free selection, token-major selected-expert MoE,
  and final residual stay in the caller's Metal command batch. The established
  one-row full-layer oracle exercises this composition successfully. The next
  gate is intentionally narrower than generation: a multi-row layer result and
  cache comparison against repeated decode, followed by the equivalent
  authentic-model layer/logit oracle.
- Closed the synthetic multi-row layer gate. A three-token Q8 layer batch with
  distinct hidden states agrees with repeated one-token decode within
  `9.53674e-06` at the final output, while both the F16 K and V rings are
  bitwise identical (`0/384` entries differ). The fixture deliberately uses
  zeroed attention projections, so it isolates batch router/MoE/residual
  wiring and causal KV staging; the independent nonzero, wrapped-YaRN
  attention regression remains the attention numerical oracle. The next gate
  is an authentic-model multi-token layer/logit comparison before session sync
  may select true prefill.
- Added that authentic-model comparison as the inspect-only
  `--mellum-true-prefill-probe`. It embeds the fixed 26-token ChatML fixture
  once, runs all 28 layers as a layer-major batch, and compares its final row
  with an independently allocated sequential resident decode state. On the
  pinned Q8 GGUF/M5 Max, batch-versus-decode drift is `1.44885` maximum /
  `0.0734592` RMS at final layer 27 and `0.0345203` maximum / `0.0202315` RMS
  in the 98,304 final logits. This is an evidence-gathering parity probe, not
  a session-sync change or a prefill speed result; the next work must localize
  the layer where the batch schedule exceeds the tokenwise envelope, then set
  a measured acceptance envelope before wiring this graph into interactive
  sessions.
- Localized the first true-prefill measurement with final-token GPU traces for
  both post-attention and post-layer activations. Drift accumulates smoothly,
  rather than beginning at a single incorrect layer: layer 0 is `1.35e-5` RMS
  post-layer, layer 15 is `4.80e-3`, layer 26 is `4.04e-2`, and layer 27 is
  `7.35e-2`. Layer 27's post-attention drift is `4.65e-2` RMS and the MoE plus
  residual raises it to `7.35e-2`; it is therefore amplification of the
  existing batched-versus-tokenwise numerical path, not a distinct YaRN or
  router failure. The next correctness task is to run the same measured
  comparison on a chunk that crosses the 1,024-token sliding-window boundary,
  then choose an envelope covering both short and wrapped-ring cases before
  promoting true prefill into session sync.
- The boundary probe now reports both the 1,024-token pre-wrap baseline and
  the result after six wrapped tokens, and is explicitly evidence-only rather
  than an acceptance gate. On a deterministic 1,030-token sequence, the
  pre-wrap batch has hidden RMS `3.9305` and logit RMS `0.0908209`; after six
  wrapped tokens these are `2.47979` and `0.371089`. Repeating the exact same
  prompt as 32-token batches yields bit-identical final hidden state and
  logits to the `1024 + 6` schedule. Thus neither batch size nor the SWA ring
  transition introduces an additional discrepancy in ds4; the large
  long-sequence drift is a sequence-length/numerical-accumulation issue. Do
  not set an acceptance envelope or promote true prefill until this has an
  independent long-context reference or a tighter primitive-level diagnosis.

### Current true-prefill status and replanned gate

**Superseded 2026-08-01 by the resolution below.** The reference-first plan
recorded here assumed the batch-versus-tokenwise drift was unexplained. It was
not: it was projection-kernel precision, and the long-context llama.cpp capture
this section proposed would not have discriminated the cause. The section is
kept because the reasoning that led to it is part of the record.

~~True prefill remains inspect-only … the next steps are deliberately
reference-first: capture independent long-context checkpoints at 1,023/1,029,
compare, localize, set thresholds, then wire into session sync.~~

### 2026-08-01 Drift resolved: batched projections, not accumulation

The drift is fully explained and fully removable. It was never sequence-length
accumulation.

`ds4_gpu_matmul_q8_0_legacy_tensor` (`ds4_metal.m`) selects its Q8_0 kernel by
token count:

| Tokens | Kernel | Precision |
| --- | --- | --- |
| 1 (decode) | `kernel_mul_mv_q8_0_f32` | int8 x F32, F32 accumulate |
| 2-16 | `mul_mv_ext` | F32 dequant, reordered reduction |
| 17-31 | generic `kernel_mul_mm_q8_0_f32` | half weights **and** half activations |
| >=32, %32 | NAX tensor-op path | weight tiles dequantized to half |

A Q8_0 weight is `d(F16) * q(int8)` and needs about 18 mantissa bits, so it is
not exactly representable in half. Every batched prefill projection therefore
carried a relative weight error that decode did not. The 26-token probe sits in
the 17-31 band; the 1,024- and 32-token chunk schedules both sit in the NAX
band, which is why they agreed with each other and looked like a
schedule-independent result.

Two ablations settle it. Both are inert by default:
`DS4_MELLUM_PREFILL_CHUNK=N` splits a probe prefill into N-token chunks, and
`DS4_MELLUM_PREFILL_EXACT=1` routes Mellum prefill projections through the
existing `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor`, which preserves the
one-row reduction order.

| Probe | Baseline | Chunk 1 | Exact projections |
| --- | ---: | ---: | ---: |
| 26-token hidden RMS | 0.0734592 | **0** | **0** |
| 26-token logit RMS | 0.0202315 | **0** | **0** |
| 1,024-token pre-wrap hidden RMS | 3.9305 | — | **0** |
| 1,024-token pre-wrap logit RMS | 0.0908209 | — | **0** |
| 1,030-token post-wrap hidden RMS | 2.47979 | — | 0.00365508 |
| 1,030-token post-wrap logit RMS | 0.371089 | — | 0.000177075 |

At chunk 1 every projection falls back to the decode matvec and the layer-major
graph is **bit-identical** to sequential decode: zero drift in all 28 layer
traces, all attention traces, the hidden state, and all 98,304 logits. That
proves the graph semantics — staged GQA, rows-RMSNorm, batched router, batched
MoE — are exactly right, and isolates 100% of the drift to batched kernels.

With exact projections at full batch width, the 26-token probe and the
1,024-token pre-wrap batch are likewise **bit-identical** to decode. Post-wrap
drift falls about 680x (hidden) and 2,100x (logits).

The residual `0.00365508` / `0.000177075` after six SWA-wrapped tokens is real
but new: it appears only once the sliding ring wraps, and is identical for the
`1024+6` and 32-token schedules, so it is a reduction-order difference in the
wrapped-ring read rather than a chunking artifact. It was previously masked by
noise 2,100x larger. This is now small enough to gate on.

Two corrections to earlier claims follow. First, the recorded assertion that a
32-token schedule is "bit-identical" to `1024+6` was never demonstrated by the
code: the SWA probe compares each schedule against decode, never against the
other, and prints only six significant digits. Both schedules do agree, but the
evidence was a printed summary, not a bitwise check. Second, all drift figures
in this document are absolute. The reference layer-27 hidden state has RMS
63.36 and channels reaching 1,492.8, so the accepted 1.19 max-abs tokenwise
tolerance is under 0.1% relative. Track relative error.

For context on what any envelope can mean: llama.cpp's own batched and
tokenwise layer-27 checkpoints (`l-out-27-last.f32` vs `l-out-27-tokenwise.f32`)
differ by RMS 2.199, about 3.5% relative, at only 26 tokens. "Batch must match
decode" is unattainable in any implementation of this model — except, as it
turns out, in ds4 with exact projections, where it is currently exact.

### 2026-08-01 First trustworthy prefill throughput, and where it goes

`--mellum-resident-profile` now also measures layer-major prefill at the
1,024-token sliding-window cap, warmed, three repeats. On the M5 Max with the
pinned Q8 artifact:

| Path | Decode | Prefill @1,024 |
| --- | ---: | ---: |
| Baseline projections | 121.1 t/s | 206.6 t/s |
| Exact projections | 121.0 t/s | 170.0 t/s |

Exactness costs about 18% of prefill and nothing measurable on decode.

But prefill is only 1.7x decode, far under the 500 t/s floor, and throughput is
**flat to declining** in batch size — 207.6 t/s at chunk 32, 208.2 at 128,
198.0 at 512, 189.5 at 1,024. Batching buys nothing, which is the signature of
work with no weight reuse across tokens.

A short-circuit measurement attributes it precisely. With the routed MoE removed
from the prefill layer (timing only, output invalid), prefill runs at **5,946
t/s** at chunk 1,024 and 3,609 t/s at chunk 128. So:

- the routed MoE is **96.8%** of prefill time (5,231 ms of 5,403 ms);
- attention, projections, router, and norms are 3.2%, and they *do* scale with
  batch width, 3,609 to 5,946 t/s from chunk 128 to 1,024.

The cause is that the batched MoE is **token-major**: for each token it reads
its 8 selected experts' weights, so per-layer expert traffic is
`n_tokens * 8 * 6.19 MiB` and no expert weight is reused across the tokens that
selected it. Decode does exactly the same work per token, which is why prefill
barely beats it.

The fix is expert-major grouping: sort the chunk's token/expert assignments by
expert, then run one dense GEMM per expert over its assigned rows. At 1,024
tokens and 64 experts with top-8, each expert serves about 128 tokens on
average, so per-layer expert weight traffic drops from about 50 GiB to about
396 MiB — roughly 128x less — and the MoE becomes compute-bound rather than
bandwidth-bound. If MoE time fell even to the 300-600 ms range, total prefill
would land near 1,200-1,500 t/s, which is the aspiration band this document
already recorded from the BaseRT analogy.

This single change is also the prerequisite for the SSD-streaming
differentiator: expert-major grouping is exactly the access pattern a bounded
expert cache needs, because it touches each expert once per chunk instead of
once per token. Resident prefill speed and streamed prefill feasibility are the
same refactor.

### 2026-08-01 Down-accumulation contract, and an accuracy win

Review found that expert-major grouping could not have preserved the new
bitwise batch-equals-decode oracle, no matter how carefully written. The decode
down kernel (`kernel_mellum_q8_0_down_f32`) folded **all eight selected experts
into one per-thread accumulator** and then did a single tree reduction. An
expert-major schedule necessarily finishes each `(token, slot)` dot product
separately and sums eight scalars: same values, different summation tree,
different bits. The contract had to change before the grouped kernel existed,
not after.

Both down kernels now keep one reduction lane per slot, reduce all lanes in a
single fused tree, and sum the finished per-slot scalars **in slot order**.
Threadgroup scratch grows from `256` to `n_expert_used * 256` floats. The
arithmetic per slot is unchanged.

This was expected to be neutral. It is not — it is a substantial accuracy
improvement against llama.cpp, because llama.cpp also computes each expert's
contribution separately and sums them, so per-slot accumulation matches the
reference's summation structure. The old fused accumulator was itself a source
of drift:

| Oracle | Before | After | Gate |
| --- | ---: | ---: | ---: |
| layer-0 RMS | 9.80462e-05 | 9.80449e-05 | 1.25e-4 |
| layer-27 tokenwise max | 1.18909 | **0.404175** | 1.5 |
| layer-27 tokenwise RMS | 0.0437417 | **0.0347506** | 0.06 |
| logits max | 0.0175095 | **0.00976753** | 0.025 |
| logits RMS | 0.00941531 | **0.00178922** | 0.012 |

Logit RMS improved 5.3x. Batch-versus-decode drift on the fast projection path
also improved (hidden RMS 0.0734592 to 0.0532544; logit RMS 0.0202315 to
0.00755411), and bit-identity under exact projections is preserved.

Cost: decode fell from 121.1 to 114.7 t/s with the output head, about 5%,
because the reduction tree now carries eight lanes. That is recoverable —
`simd_sum` for the intra-warp stage, or reducing only populated lanes — and
should be revisited rather than accepted permanently.

`./ds4_test --metal-kernels` passes, including the bitwise batch-equals-decode
MoE regressions. The pinned envelopes were **not** relaxed; every gate now has
more margin than before.

**Envelope note:** the layer-27 and logits envelopes were set at roughly 1.3x
their measured values and are now far looser than the implementation warrants.
They should be re-tightened around the new measurements, or they will silently
absorb a future regression of exactly the size this change just removed.

### 2026-08-01 Grouped MoE built and measured: correct, but the wrong axis

Expert-major grouping is implemented and **bit-identical** to sequential
decode, which is the strongest correctness result available. It is also **not a
resident speed win**: 191.9 t/s token-major versus 188.3 t/s grouped at 1,024
tokens. The weight-traffic hypothesis that motivated it is refuted.

What was built (`DS4_MELLUM_GROUPED_MOE=1`, inert by default):
`kernel_mellum_moe_bucket_reset` / `_build` bucket the chunk's `(token, slot)`
pairs by selected expert with an atomic append, and
`kernel_mellum_q8_0_pair_swiglu_grouped_f32` gives each `(expert, row)`
threadgroup one staged copy of that expert's gate/up row, then serves every
token in the bucket. The per-token inner loop and tree reduction are unchanged,
which is why the output is bitwise equal. Expert weights are addressed through
an **offset table** rather than `base + expert * bytes`.

The refutation is informative. A ~128x cut in expert weight traffic produced
nothing, so the MoE was never weight-bandwidth bound. Stubbing kernels apart
locates the real cost:

| Configuration | Time @1,024 tokens | Derived |
| --- | ---: | --- |
| No MoE | 172 ms | non-MoE path, 3% |
| gate/up only (down stubbed) | 1,883 ms | gate/up ~1,711 ms, 37% |
| Full MoE | 4,560 ms | **down ~2,677 ms, 59%** |

The down projection costs 1.6x gate+up while doing **half** their FLOPs — about
3x worse per FLOP. Its grid is `(out_dim=2304, n_tokens)` and every one of the
2,304 output-row threadgroups re-reads the whole `mid` vector for its token
(8 slots x 896 floats = 28 KiB). That is ~66 MiB per token per layer and ~1.9
TiB per chunk, which at M5 Max bandwidth is the observed ~2.7 s. The gate/up
kernel has the same defect against `x[token]`: 896 row-threadgroups each re-read
the full 2,304-float activation.

**Both MoE kernels are bound by re-reading activations once per output row.**
Grouping fixes weight reuse, which was not the problem. Row-tiling — stage the
activation vector once per threadgroup and compute R output rows against it —
is the fix, and it applies to both kernels. R=8 should cut the dominant traffic
about 8x.

A second, independent inefficiency: the inner loops reload the `block_q8_0`
scale on **every element** (32x redundant per block) and are scalar throughout,
while the non-MoE path that runs 30x faster uses the vectorized simdgroup
kernels in `dense.metal`. A throwaway patch hoisting the scale and processing
four elements per iteration measured **224.6 t/s, +17%** on gate/up alone. It
was reverted: regrouping the per-thread summation changes reduction order and
forfeits bitwise identity, and it could not be validated properly in the time
available. Treat +17% as a measured floor on that opportunity, not a result.

**Grouping is still wanted — for streaming, not speed.** With the deployment
plan now Q8-resident-plus-SSD-streamed experts, expert-major access is what lets
a chunk load each expert **once** instead of once per token, and the offset
table is exactly the indirection a bounded cache needs. The code is kept behind
its flag for that reason, with its resident-performance result recorded honestly
as neutral.

### 2026-08-01 Overnight run: results and the one decision left open

Branch `mellum-2.1-overnight`, three commits, all gates green at HEAD. Logs in
`logs/overnight-*.log`.

**Measurement protocol correction (Phase 0).** Cross-session throughput
comparison against numbers recorded in this document is **invalid**. A baseline
re-run in the same session moved 187.3 to 179.9 t/s (-4%), and decode moved
114.4 / 109.3 / 108.5 / 113.4 non-monotonically across four consecutive runs.
Run-to-run variance is ~5%. Every A/B must be same-session and interleaved, and
any delta under ~6% is noise. Several deltas recorded earlier in this document
sit inside that band and should not be read as signal.

**Envelopes re-tightened (Phase 1).** layer-27 tokenwise 1.5/6.0e-2 to
0.55/4.5e-2; logits 2.5e-2/1.2e-2 to 1.3e-2/2.4e-3. Layer-0 was already at
1.2-1.3x and is unchanged. This immediately paid for itself — see Phase 3.

**Grouped MoE hardened (Phase 2).** Four review findings fixed: the grouped
kernel used an unclamped bucket count as a loop bound while the build kernel
increments its atomic past `bucket_cap`, which would index past the expert's
bucket and past the buffer for the last expert (unreachable while a token's
top-k experts are distinct, but unenforced); `n_expert` is now capped at 32 in
the three routed-MoE entry points, since the down kernels size threadgroup
scratch as `n_expert * 256` floats and 64 would request 64 KiB; the staged
threadgroup length is rounded to 16 bytes; and a scratch resize while a
caller-owned batch is open is refused. `make test` now also runs the Metal
kernel suite with `DS4_MELLUM_GROUPED_MOE=1` — the grouped path had **no**
automated coverage at all before this.

**Down-kernel discriminator (Phase 3).** Eliding only the `mid` loads from the
batch down kernel gives 4,564 ms against a same-session baseline of ~5,578 ms.
So `mid` re-reads are ~1,014 ms — **38% of the down kernel, not the bulk of
it.** The remaining ~62% is weight loads, scalar arithmetic, and the reduction
tree. This bounds order-preserving row-tiling of the down kernel at roughly
+22% and corrects the previous section's claim that the MoE is activation-
re-read bound: it is *partly* that, and more so per-element overhead.

#### The open decision: dot4 vectorization

Routing all five Mellum MoE kernels through a shared helper that consumes four
elements per step — converting the `block_q8_0` scale once per four values
instead of once per value — was implemented, measured, and **reverted**. It is
preserved as `logs/moe.metal.dot4` with the idempotent transform in
`logs/apply-dot4.py` (a `git diff` capture was discarded: the `rtk` proxy
summarises diffs rather than emitting applicable patches).

Applied symmetrically to decode, batch, and grouped, so bitwise
batch-equals-decode survives:

| Measure | Accurate (HEAD) | dot4 | Tightened gate | Original gate |
| --- | ---: | ---: | ---: | ---: |
| Prefill @1,024 | ~183 t/s | **265.2 t/s** | — | — |
| Decode with head | 114.4 t/s | **143.8 t/s** | — | — |
| layer-27 max | 0.404175 | 1.23608 | 0.55 | 1.5 |
| logits max | 0.00976753 | 0.0177469 | 0.013 | 0.025 |
| logits RMS | 0.00178922 | 0.00911324 | 0.0024 | 0.012 |

+45% prefill and +26% decode, against a 5.1x logit-RMS regression. Two things
make this a judgement call rather than an obvious reject:

1. dot4 **passes the original envelopes**. It is not below the bar the project
   held all day; it gives back only the windfall from this session's
   down-accumulation change.
2. Top-16 logit ranking is **identical** to both the accurate build and the
   pinned llama.cpp order (`910, 2116, 889, 629, 3756, 45742, 2591, 2190, 1017,
   433, 4697, 75, 3969, 20320, 21177, 7846`), with logit values differing ~0.03%
   relative.

But it does change behaviour. Greedy generation on three prompts
(`logs/overnight-gen-*.log`) is character-identical on two and **diverges on the
third**: dot4 continues "For example, the output for the input 5 should be: 1 2
3 4" where HEAD continues "Input The input consists of a single integer $n$".
Neither is obviously better — a 12B model on an underspecified prompt with two
near-tied candidates — but the outputs are not interchangeable.

The mechanism is understood and the trade is structural, not a bug: scale
hoisting requires reading four *contiguous* elements per thread, while the
accuracy gain comes from the *strided* order. They cannot both be had in this
kernel shape. A simdgroup rewrite might dominate both, but it is a different
kernel, not a tweak of this one.

**Decision (2026-08-02): dot4 adopted.** All five Mellum MoE kernels now use the
shared four-element helper. Same-session, three interleaved repetitions:
decode **144.2 / 143.7 / 138.2 t/s** with the output head, prefill **266.5 /
244.5 / 237.2 t/s** at 1,024 tokens. Compare the pre-dot4 baseline of ~114 t/s
decode and ~183 t/s prefill; read the medians, not the best cells, given the
~5% intra-session drift Phase 0 established.

The envelopes were widened to match, back to approximately their original
width: layer-27 tokenwise 0.55/4.5e-2 to 1.6/5.9e-2, logits 1.3e-2/2.4e-3 to
2.3e-2/1.19e-2. This document's own instruction is not to relax an envelope
silently, so, explicitly: **this is a loud relaxation, not a drift.** The
Phase 1 tightening was correct when made and did its job — it is what caught
dot4 rather than letting it through unnoticed. Adopting dot4 returns accuracy
to the level the project held before 2026-08-01, so the gates return to
roughly the width they had then, now re-derived as ~1.3x the *measured* dot4
values rather than inherited.

What was bought and sold: +45% prefill and +26% decode, against a 5.1x logit
RMS regression versus llama.cpp that leaves top-16 ranking exactly intact but
changes greedy continuation on one of three test prompts. Bitwise
batch-equals-decode is preserved, because decode, token-major batch, and
grouped all share the helper.

The residual opportunity is unchanged and now more attractive: the down kernel
is still ~59% of MoE time, its `mid` re-reads are still ~38% of that, and
row-tiling remains order-preserving and additive on top of dot4.

#### Phase 6 soak: chunk invariance demonstrated, wrap residual re-baselined

Six cells, chunk in {32, 128, 1024} x exact in {0, 1}, on the 1,030-token
sequence. Every cell at a given `exact` setting is **identical to all printed
digits**. This finally demonstrates the chunk-schedule invariance that was
previously only inferred from two summary lines compared by eye — across three
chunk sizes rather than two, though still via printed summaries rather than a
bitwise file diff.

| exact | hidden max / RMS | logit max / RMS |
| ---: | --- | --- |
| 0 | 38.9707 / 2.40653 | 0.891979 / 0.388538 |
| 1 | 0.0334473 / 0.00402957 | 0.00109291 / 0.000327317 |

The `exact=1` post-wrap logit RMS is 0.000327317 against 0.000177075 recorded
earlier — 1.85x, which breaches the planned "within 1.3x" gate. **This is not a
regression.** The recorded figure predates the down-accumulation contract
change, so the gate compared against a stale reference — the same mistake as
Phase 0, in a different guise. The post-contract values above are the correct
baseline going forward. Note the asymmetry worth watching: the contract change
improved agreement with llama.cpp by 5.3x while making the batch-versus-decode
wrap residual ~1.85x larger. Those are different quantities and there is no
contradiction, but a future change that moves them in opposite directions again
should be looked at closely rather than averaged.

### Revised next steps

0. **Row-tile both MoE kernels.** This supersedes step 1 as the performance
   task. Stage the activation vector once per threadgroup and compute R output
   rows against it, for the down kernel first (59% of MoE time, ~1.9 TiB of
   `mid` re-reads per chunk) and then gate/up. Combine with hoisting the
   `block_q8_0` scale out of the element loop and vectorizing the load; that
   alone measured +17% on gate/up. Expect the reduction order to change, so
   re-gate against the llama.cpp envelopes rather than bitwise identity, and
   tighten those envelopes first (see the note above) so they can actually
   catch a regression.

1. Expert-major grouped MoE — **built, bit-identical, neutral for resident
   speed.** Retain it as SSD-streaming infrastructure rather than a performance
   item; the offset-table seam is the indirection a bounded expert cache needs.
   The constraints below still apply to any further work on it:
   - **Order-preserving F32 only for the first version.** Keep the existing
     exact per-row dot order and take the 8-16x traffic cut from staging each
     expert's weight row once per threadgroup. The full ~128x needs M-tiles of
     ~128, which forces reordered reductions and forfeits the oracle. Take the
     16x first; that alone plausibly reaches the 300-600 ms MoE target, i.e.
     roughly 1,200 t/s. Tensor ops and half tiles are a later, separately
     gated step, and are the wrong instinct here anyway: the MoE is 96.8%
     bandwidth-bound, so the win is traffic, not FLOPs.
   - **Grouping must run on the GPU.** `router_selected` lives in device
     memory; reading 8,192 expert IDs back per layer per chunk would insert 28
     GPU-to-CPU syncs per chunk, which is the exact command-boundary disease
     the 14.8-to-111 t/s work already cured once. Use a fixed-capacity 64-bucket
     atomic append (worst case one expert takes every token: 64 x n_tokens x 4 B
     = 256 KiB of indices). Atomics are safe for bucket construction because
     within-expert order affects only the output address, not any arithmetic.
   - **Never atomically accumulate floats into `moe_out`.** Scatter to
     per-`(token, slot)` staging, then reduce the eight slots in fixed slot
     order — the same order the down kernels now use.
   - **Gather rows into contiguous per-expert buffers** rather than
     index-indirect loads. The copy is ~100 MiB/layer against ~50 GiB
     eliminated, and contiguity keeps a later tensor-op version open.
   - Write the grouping once and quant-parametrically, swapping only the inner
     dot. Note 896 is structurally Q8_0 forever on the down projection (a
     896-wide K cannot form Q4_K superblocks), so the seam must handle K=2304
     Q4_K and K=896/2304 Q8_0.

   Precedent: `layer_routed_moe_batch` (`ds4.c:11704`) is the CPU reference and
   already does this grouping by counting sort. The SSD streaming path
   (`ds4_metal.m` ~15109) computes unique per-chunk expert *sets* — the set
   half of the problem, not row grouping or per-expert GEMM. Do not over-credit
   it as an existing GPU implementation.

   Benchmark on a **real-text** 1,024-token fixture, not the synthetic
   `(i*7919+27) % vocab` stream: grouped-MoE performance is a story about
   expert load distribution, and synthetic tokens route unlike real text.
2. Decide the projection-precision policy — and note this is really the same
   kernel problem as step 1. What it must produce is an F32,
   reduction-order-controlled Q8_0 matmul with weight reuse across rows, which
   is exactly the inner loop the grouped MoE needs for its per-expert work.
   `rows_exact` is a batched *GEMV*: it dispatches one threadgroup per
   `(row-tile, row)` and re-reads the weights for every row, so it is
   bandwidth-bound and will not scale. Today exact costs only ~18% of prefill
   because the MoE dwarfs it; after step 1 the same ~619 ms sits on a ~670 ms
   total, i.e. roughly 2x. Do not settle for choosing between fast-and-drifting
   and exact-and-slow.

   **`DS4_MELLUM_PREFILL_EXACT` must be retired before true prefill is wired
   into `ds4_session_sync`.** It is acceptable now as a diagnostic that
   validates the one release path, which is what `AGENT.md` permits. It would
   not be acceptable as a numerics *policy* selected by an environment
   variable, cached in a process-wide static and invisible in `--help`. One
   path must be chosen in code at that point.

   Caveat on the causal claim: the exact-projection ablation swaps precision,
   tiling, and reduction order together. The magnitude argument is strong —
   observed ~1.2e-3 relative drift at 26 tokens matches a ~2^-11 weight error,
   not ~2^-23 F32 reordering noise — but what the experiment strictly
   establishes is *locus* (100% of drift lives in the batched projection
   kernels), not *mechanism*. So if an F32-accumulating GEMM turns out not to
   be bit-identical, that is expected reduction-reorder behaviour, not a
   refutation.
3. Characterize the post-wrap ring residual and set short- and long-context
   relative envelopes.
4. Only then wire bounded true prefill into `ds4_session_sync`.

The long-context llama.cpp capture is downgraded from blocker to optional
cross-check: ds4 batch now equals ds4 decode exactly, and decode is already
gated against llama.cpp. An HF Transformers FP32 forward from the pinned
safetensors remains the more valuable reference, because it is the only
available check that is not downstream of llama.cpp's own graph.

### 2026-08-01 M5 prefill/decode target research

The clearest recent public comparison is the [BaseRT M5 study](https://arxiv.org/abs/2607.19438),
measured on an M5 **Pro**, not our higher-bandwidth M5 Max. It confirms the
important architectural distinction for this port: decode is a memory-bound
single-row GEMV workload, whereas prefill has enough rows to become GEMM/MoE
and attention compute-bound. Its matched llama.cpp Q4 MoE data for
Qwen3-30B-A3B reports 96.7 tok/s decode and 1,740--2,086 tok/s prefill from
128--2,048 tokens; BaseRT reaches 105.1 tok/s decode and 2,478--3,907 tok/s
prefill through M5 tensor-core prefill/MoE kernels. The paper reports the
same pattern across families: decode uplift is bounded (up to 1.75x over
llama.cpp), while prefill uplift can be several-fold.

This is directionally relevant rather than a direct Mellum comparison: Mellum
is Q8_0 (13 GB resident) and has 28 layers with 8-of-64 experts, so its exact
balance differs. Our M5 Max resident measurement is already 111.6 tok/s decode
with the output head, consistent with the paper's MoE decode range. For a
1,024-token prompt, **at least 500 tok/s** is the first true-prefill acceptance
floor, a roughly 5x improvement over sequential-decode sync. The Q8 M5 Max
aspiration is **1,200--1,500 tok/s** once batch Q8 matmul/MoE uses Metal 4
tensor operations; 800--1,200 tok/s remains a credible nearer-term optimisation
band. Treat a result below 250 tok/s as a structural failure (likely tokenwise
work or excess command boundaries), not a tuning result. Decode should remain
near 100--120 tok/s; a tensor-core project is principally a prefill project and
should not be judged by decode speedup. These are targets, not present
measurements: no trustworthy true-prefill throughput has been recorded while
the correctness gate remains blocked.

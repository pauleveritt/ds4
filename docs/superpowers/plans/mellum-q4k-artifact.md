# Mellum Q4_K artifact: status and remaining work

**Goal.** `ds4-agent` loads the 9.33 GiB selective artifact — Q4_K expert
gate/up on layers 0–21, Q8_0 everywhere else — and generates correctly at a
speed worth having.

**Why it matters.** The stated goal for this line of work is *fast on 16 GB
machines*. Every optimization so far was measured on a 128 GB box against a
12.04 GiB Q8_0 model that does not usefully fit in 16 GB. This artifact is
9.33 GiB and is the only piece that addresses the actual goal.

Read `docs/superpowers/MELLUM.md` first — current state, how to run things, and
the traps. This document assumes it.

## Status

Pieces (1) and (2) are **landed and gated**. Piece (3) is not started.

| Piece | State |
| --- | --- |
| 1. Mixed-layout validation | Done |
| 2. Q4_K expert gate/up in decode | Done — but wired to the *scalar* kernel |
| 2b. Mellum-native Q4_K decode kernel | Done — decode 0.75x to 0.94x |
| 3. Q4_K expert-major prefill | Not started; tokenwise fallback in place |

Measured, interleaved, paired: decode **0.94x** Q8_0 (~117 vs ~123 t/s, after
2b; it was 0.75x before), session prefill **0.21x** (~73 vs ~341 t/s), planned
resident 9.40 vs 12.10 GiB. Decode is close to parity; prefill is the gap.

## Piece 2b: select the SIMD Q4_K kernel

**This was not in the original brief and displaces piece (3) in priority.**

Piece (2) wired decode to `kernel_glm_q4_K_pair_swiglu_f32`
(`metal/moe.metal:509`), which is the naive kernel in that file: one strided
thread per element, a per-element nibble extraction and scale lookup through
`ds4_glm_q4_K_value`, then a threadgroup tree reduction. That is why ds4's
Q4_K decode is ~25% *slower* than Q8_0 while llama.cpp measures the same file
6.6% *faster*. The gap is ds4's kernel selection, not the quantization.

`glm_q4_K_pair_swiglu_simd_f32_impl` (`metal/moe.metal:1306`) already exists
and is what the GLM paths dispatch (`:1481`, `:1512`, `:1520`, `:1587`). This
is the same shape of work piece (2) turned out to be — selection, not
invention — and it targets decode, which dominates interactive agent latency.
Do this before (3).

Confirm the Q8_0 comparison arm is itself SIMD before drawing conclusions
about the ratio.

The artifact is at `~/models/mellum-thinking-TARGET.gguf` (it was previously
only in a session scratchpad). `tools/mellum/build-selective-artifact.sh`
regenerates it from an HF snapshot in ~15 minutes;
`tools/mellum/RESULTS-selective-artifact.md` has the llama.cpp-side
measurements.

### What (1) and (2) actually needed

**There were two Q8_0 gates, not one.** Beyond
`weights_validate_mellum_layout`, the decode-contract desc builder
(`ds4.c:36943`) independently required Q8_0 experts. Relaxing validation alone
still refused the file. Anyone adding a further quant format must expect the
same pattern: the format is asserted in more than one place.

**Piece (2) was wiring, not implementation.**
`ds4_gpu_mellum_routed_moe_one_tensor` (`ds4_metal_mellum.m:862` — note, no
`q8_0` in the name) already dispatched `kernel_glm_q4_K_pair_swiglu_f32` for
gate/up and `kernel_mellum_q8_0_down_f32` for down, which is exactly this
artifact's recipe. It had no engine caller; only
`test_metal_mellum_q4_q8_routed_moe` (`tests/ds4_test.c:2195`) used it. The
engine now selects it by the layer's actual type.

The type travels validation → `ds4_gpu_mellum_q8_0_layer_desc.pair_type` →
dispatch, following Laguna's idiom at `ds4.c:49845`, which sets
`.gate_type = l->ffn_gate_exps->type` straight off the tensor rather than
re-deriving at the dispatch site.

### Verified

Against the artifact:

- Loads: `mapped 9554.64 MiB`, mixed layout accepted.
- `--mellum-diag layer0` exits 0 with a plausible router distribution.
- `ds4-agent --non-interactive -p ...` completes a turn at exit 0 with coherent
  output. It answered in prose instead of calling the write tool — **the Q8_0
  model does the same thing on the same prompt**, so that is prompt/agent
  behaviour, not a Q4_K regression. Always run the Q8_0 control before
  attributing a behaviour change to quantization.

Against Q8_0, unchanged: all four baseline diagnostics pass, and `ds4_test`'s
batch-versus-decode is byte-identical to its pre-change values
(`3.8147e-06`), with the oracle arm still bitwise `0`.

### Not yet done

Measurement (done #3), the 16 GiB question (done #4), and the `MELLUM.md`
update (done #5).

## Piece 3: Q4_K expert-major prefill

`ds4_gpu_mellum_q8_0_routed_moe_batch_tensor` (`ds4_metal_mellum.m:1325`) and
`kernel_mellum_q8_0_pair_swiglu_gemm_f32` (`metal/moe.metal:3187`).

**The threadgroup budget is not the risk.** An earlier draft of this brief said
to check it first, because the Q8 pair kernel uses 18,432 B of a 32 KiB limit.
That number cannot move: `ds4_mellum_stage_q8_rows` (`metal/moe.metal:3028`)
stages *dequantized `float`* into `threadgroup float *dst`, so the tile is
`rows * dim` floats — 2 × 2304 = 4,608 floats = 18,432 B — regardless of what
it was dequantized from. A Q4_K staging routine writes the same tile.

What piece (3) actually reduces to:

1. `ds4_mellum_stage_q4_k_rows` — `stage_q8_rows` with the inner expression
   swapped for `ds4_glm_q4_K_value(...)`, which already exists at
   `metal/moe.metal:449`.
2. A second pipeline variant of the pair GEMM, selected by type.
3. Dispatch selection, mirroring the decode branch already in place.

**The real risk is ALU, not memory.** `ds4_glm_q4_K_value` does nibble
extraction plus a scale lookup per element against Q8's single multiply, and
staging is on the critical path. Whether cheaper weight reads (0.5625 vs
1.0625 B/elem) pay for it is empirical. llama.cpp measures this artifact's
prefill as unchanged versus Q8_0, which suggests it roughly breaks even.

Preserve F32 accumulation initially; tensor-core parity is not the goal.

**The fallback, if (3) is deferred.** Models the batch kernel cannot stage take
the tokenwise sync path (`e->mellum_batched_prefill_unsupported`, set in
`ds4_engine_bind_mellum_decode_contract`). Correct but slower: `MELLUM.md`
records tokenwise at ~120 t/s against ~355 for expert-major. Shipping (1) and
(2) alone is a legitimate stopping point; say so rather than blocking.

## Gates

```bash
./ds4_test --metal-kernels
DS4_MELLUM_MOE_GEMM=0 ./ds4_test --metal-kernels
```

**`make test` cannot pass in this worktree.** Bare `./ds4_test` defaults to the
`--long-context` case, which opens `DS4_TEST_MODEL` — defaulted to
`ds4flash.gguf` at `Makefile:20` — a DeepSeek fixture that is not on this
machine. It dies at model-open, before any Mellum code runs. Everything else in
the target builds and passes. Do not read that failure as a Mellum regression,
and do not treat `make test` as a green/red signal until the fixture exists.

Then, against the Q8_0 model:

```bash
./ds4 --mellum-diag swa-boundary --ctx 4096 --model <q8.gguf>
./ds4 --mellum-diag true-prefill --ctx 4096 --model <q8.gguf>
./ds4 --mellum-diag layer0 --mellum-diag-out /tmp/l0.f32 --model <q8.gguf>
python3 tests/check_mellum_layer0_oracle.py /tmp/l0.f32
```

### What those diagnostics do on a Q4_K artifact

- `true-prefill` **fails, by design** (`batch layer stack failed`). The guard in
  `ds4_gpu_mellum_q8_0_layer_prefill_tensor` refuses Q4_K rather than letting
  the batch kernel read K-quant bytes as Q8_0 blocks. This is the correct
  result until (3) lands.
- `swa-boundary` **passes vacuously**. It compares a tokenwise decode fixture
  against `ds4_session_sync`; the fallback routes Q4_K sync to the tokenwise
  path, so both arms are the same computation and the deviation is exactly
  `0.000000`. It reports OK while validating nothing about prefill. Q8_0 still
  measures `0.101`, which is what makes the contrast legible.
- The **layer-0 oracle fails and should not be "fixed" by widening it**:
  `max_abs=0.0298` against a `0.006` limit, `rms=0.0016` against `0.000125`.
  That gate was calibrated for Q8_0. A Q4_K limit has to come from an
  independent Q4_K reference — llama.cpp's layer-0 output on this same
  artifact would do it — not from raising the threshold until it goes green.

### Still missing: a layer-level gate

`test_metal_mellum_q4_q8_routed_moe` covers the Q4_K MoE *kernel* with a
synthetic Q4_K expert. Nothing covers the layer-level `pair_type` selection
added for piece (2) — that `ds4_gpu_mellum_q8_0_layer_decode_tensor` routes to
the Q4_K MoE and produces correct results. Build it on the pattern of the
existing batch-versus-decode row: bitwise where the arithmetic should be
identical, a numeric bound where it should not.

Note `tests/ds4_test.c` has neither `DS4_TENSOR_*` nor `DS4_METAL_TENSOR_*` in
scope, so a test that sets `pair_type` explicitly needs a shared enum or a
documented literal.

## What "done" means

1. `ds4-agent` loads the artifact and completes a real coding turn. **Met.**
2. All Q8_0 gates unchanged. **Met.**
3. Decode and prefill measured. **Met** — see `MELLUM.md`. Note the
   measurement route: `resident-profile` cannot prime a Q4_K cache and
   `ds4-bench` refuses Mellum, so this goes through `ds4-server` with
   `DS4_MELLUM_SYNC_TRACE=1`, whose trace also names the sync path taken.
4. **Memory demonstrated, not asserted. Blocked — cannot be done here.**
   `--simulate-used-memory N` locks N GiB, so simulating a 16 GiB machine on a
   128 GiB box means locking **112**, not 16. (An earlier draft of this brief
   said `--simulate-used-memory 16GB`, which locks 16 and leaves 112 free —
   it would have passed for the wrong reason.) macOS refuses the 112 GiB lock,
   failing at `104.00/112.00 GiB`, so the tightest reachable simulation leaves
   ~24 GiB. Both models load and generate there, and the planner reports
   9.40 GiB against Q8_0's 12.10 GiB at ctx 16384 — but that is planning
   arithmetic, not memory pressure. **Only real 16 GiB hardware closes this.**
5. `MELLUM.md` updated with the artifact's numbers and how to build it.

## Traps that have already cost time here

- **`make -j8` does not build `ds4_test`**, and `make cpu` overwrites `ds4`,
  `ds4-server`, `ds4-bench`, `ds4-eval` and `ds4-agent` with CPU builds.
- **A header change can silently corrupt the test binary.** `Makefile:255`
  omitted `ds4_gpu.h` from `ds4_test.o`'s dependencies. Adding a field to the
  layer desc rebuilt `ds4_metal.o` but not `ds4_test.o`, and the mismatched
  struct layouts produced a test that compiled, linked, ran, returned
  *success*, and reported seven wrong values. Fixed — but this is the trap
  family's worst form, because the failure looks like a numerics bug in your
  own change.
- **Four diagnostics can pass without testing what you changed** — `layer0`,
  `all-layers` and `logits` run decode, not prefill; and `swa-boundary` goes
  vacuous on a Q4_K artifact (above).
- **Interleave measurement arms and read ratios.** The same baseline has
  measured 128 ms and 323 ms depending on machine load.
- **Context setting is not context used.** Decode figures need
  `DS4_MELLUM_PROFILE_DECODE_DEPTH`.
- **The eval prompts are inside the imatrix calibration set** for this
  artifact. Its quality scores are a regression check, not a benchmark. An
  out-of-sample suite — coding correctness, tool selection, long-context
  recall, patch success — is separate work, compared against ds4 Q8_0 rather
  than bitwise equality.
- **`timeout` is not on macOS.** Use the harness, not `timeout(1)`.

## Naming fossils to clean up after (3)

- `ds4_gpu_mellum_q8_0_layer_desc` now carries Q4_K. It should become a
  quant-neutral Mellum layer descriptor.
- The same applies to `ds4_gpu_mellum_q8_0_routed_moe_one_tensor` and
  `..._batch_tensor`, whose names now describe one of two supported formats.

## Finding: generic retry cannot fix fabricated validation (baseline for P9/P10)

Overnight tool-calling work (nudge, malformed-retry cap, recovery framing —
see `MELLUM.md`) got Mellum reliably *emitting* well-formed native tool calls.
It did not fix a separate failure mode: Mellum produces good implementation
content but does not reliably distinguish "printed code" from "mutated
workspace," and it fabricates validation (claims tests passed without a
recorded execution). This surfaced across five smoke-test rounds; tuning the
generic `DS4_AGENT_TOOL_NUDGE` retry further does not correct it, because the
nudge only prompts for *a* tool call, not the *right* one, and does nothing
about untrusted self-reported results. **Do not spend more rounds tuning the
generic nudge against this failure mode — that thread is closed.**

The fix is not `tool_choice=required` globally — conversational turns
legitimately need no tool, and forcing one there is the wrong trade. Instead,
a later phase (P9/P10 — not yet broken out in this doc) should add an
explicit host-controlled action mode:

- A handoff packet declares required files and a validation command.
- The host reports objective state back to the model — e.g. "0/4 files
  exist; pytest was not run" — computed by the host, not asserted by the
  model.
- A turn cannot complete while declared objectives fail.
- The retry round for a failed turn requires one registered tool call;
  prose-only output is rejected in that round.
- Claimed test results are never trusted without a host-recorded command
  execution backing them.

This run is the baseline that motivates that machinery — keep it referenced
from whatever P9/P10 doc eventually specs the action-mode design. If forced
execution is tested later, label those results separately from native-agent
competence numbers; they measure different things.

## Context: what this does *not* fix

Cross-session decode batching is separately excluded for Mellum at
`ds4_sessions_eval_batch_metal_supported` (`ds4.c:65079`) — "No batched Mellum
decode graph exists; it takes the ordered path" — alongside Laguna. That is a
different mechanism from prefill token batching, and neither piece (3) nor
anything in this document changes it. Sessions interleave serially on the GPU
regardless.

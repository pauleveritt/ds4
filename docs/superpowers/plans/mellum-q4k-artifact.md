# Brief: make ds4-agent run the 9.33 GiB Mellum artifact

**Goal.** `ds4-agent` loads `mellum-selective.gguf` — Q4_K expert gate/up on
layers 0–21, Q8_0 everywhere else — and generates correctly at a speed worth
having. Today ds4 refuses the file before it finishes loading.

**Why it matters.** The stated goal for this whole line of work is *fast on
16 GB machines*. Every optimization so far was measured on a 128 GB box against
a 12.04 GiB Q8_0 model that does not usefully fit in 16 GB. This artifact is
9.33 GiB. It is the only remaining piece that addresses the actual goal.

Read `docs/superpowers/MELLUM.md` first — current state, how to run things, and
the traps. This brief assumes it.

## What already exists

**The artifact is built and validated.** `tools/mellum/build-selective-artifact.sh`
reproduces it from an HF snapshot; `tools/mellum/RESULTS-selective-artifact.md`
has the measurements. Against Q8_0, through llama.cpp: 2.71 GiB smaller,
prefill unchanged, **decode 6.6% faster**, no measurable quality loss at n=3.
So the file is known-good before you touch ds4 — any failure you see is ds4's.

**Most of the kernel machinery exists**, which is the part to internalise
before estimating:

- `kernel_glm_q4_K_pair_swiglu_f32` (`metal/moe.metal:509`) takes
  **`ds4_metal_glm_routed_moe_args`** — the *same* args struct Mellum's MoE
  dispatch already builds (`ds4_metal_mellum.m:958`, `:1107`, `:1438`). Its
  parameter list is otherwise identical in shape to what Mellum needs: args,
  gate, up, x, selected, weights, mid, threadgroup scratch.
- That struct **already carries `gate_type`, `up_type` and `down_type`**
  (`ds4_gpu.h:2270`, `:2615`). Mellum sets only `down_type`, hard-coded to
  `DS4_METAL_TENSOR_Q8_0`, and leaves the other two at zero. Check what zero
  means for that enum before assuming it is harmless.
- A wider `q4_k_pair_swiglu` family exists for the GLM/DeepSeek paths —
  `id`, `group`, `group6`, `group8`, `slots6`, `table`, `addr` variants — so
  there is precedent for whichever shape the expert-major path wants.

This is adaptation, not invention. Do not start by writing a new Q4_K kernel.

## The three pieces

### 1. Mixed-layout validation

`weights_validate_mellum_layout` in `ds4.c` calls `tensor_expect_layout(...,
DS4_TENSOR_Q8_0, ...)` for `ffn_gate_exps`, `ffn_up_exps` and `ffn_down_exps`.
The first two must accept Q4_K **or** Q8_0; the third must stay Q8_0-only.

Down is not an oversight and must not be "fixed": ds4 does not execute Q5_0,
and **Mellum's 896-wide down input is not divisible by the 256-element K-quant
block**, so a K-quant there is unrepresentable rather than merely worse. Record
that where the check lives, or someone will relax it.

Accept the mix *per tensor*, not per model. Layers 0–21 are Q4_K and 22–27 are
Q8_0 in the same file, so a single model-wide "is this the Q4 build" flag is
the wrong shape and will not survive the next recipe.

### 2. Q4_K expert gate/up in decode

`ds4_gpu_mellum_q8_0_routed_moe_one_tensor` (`ds4_metal_mellum.m:1007`) is the
decode MoE. It currently always dispatches
`kernel_mellum_q8_0_pair_swiglu_f32`. It needs to pick by the tensor's actual
type, which the layer descriptor should carry down from validation rather than
re-deriving.

The down projection is unchanged — it stays Q8_0 in this artifact, so
`kernel_mellum_q8_0_down_f32` still applies. Only gate/up varies.

**This piece alone makes the artifact loadable and generatable**, and gives the
first real ds4-side numbers. Land it before touching prefill.

### 3. Q4_K expert-major prefill

`ds4_gpu_mellum_q8_0_routed_moe_batch_tensor` (`:1325`) and
`kernel_mellum_q8_0_pair_swiglu_gemm_f32` (`metal/moe.metal:3187`). The
expert-major path stages dequantized weight rows into threadgroup memory via
`ds4_mellum_stage_q8_rows`; a Q4_K equivalent has to stage from K-quant blocks,
and **the threadgroup budget is the thing to check first** — the Q8 pair kernel
already uses 18,432 B of a 32 KiB limit.

If this turns out awkward, shipping (1) and (2) alone is a legitimate stopping
point: prefill would fall back to the tokenwise path for the Q4 artifact, which
is slower but correct. Say so rather than blocking on it.

## Gates — run these, in this order

Nothing here needs the ds4 lock except the last two.

```bash
make test                                   # NOT `make -j8` — see traps
./ds4_test --metal-kernels
DS4_MELLUM_MOE_GEMM=0 ./ds4_test --metal-kernels
```

Then, against the Q8_0 model, prove you broke nothing:

```bash
./ds4 --mellum-diag swa-boundary --ctx 4096 --model <q8.gguf>
./ds4 --mellum-diag true-prefill --ctx 4096 --model <q8.gguf>
./ds4 --mellum-diag layer0 --mellum-diag-out /tmp/l0.f32 --model <q8.gguf>
python3 tests/check_mellum_layer0_oracle.py /tmp/l0.f32
```

Then, against the new artifact, the same diagnostics plus:

```bash
tests/orchestrator-eval/run_ds4_quality.sh 1 q4k
# score it against the committed q8 numbers in RESULTS-ds4-kernels.md
```

**Add a kernel-level gate for the new path.** The existing
batch-versus-decode check in `tests/ds4_test.c` is the model to copy: it
asserts bitwise when the arithmetic should be identical and a numeric bound
when it should not. A Q4_K gate/up path needs its own row there, with a
synthetic Q4_K expert, or it ships untested — which has already happened once
in this codebase and is documented in the journal.

## What "done" means

1. `ds4-agent` loads the artifact and completes a real coding turn.
2. All Q8_0 gates unchanged — this must not regress the existing model.
3. Decode and prefill measured on the artifact and compared against the Q8_0
   numbers in `MELLUM.md`, with the same interleaved discipline.
4. **Memory demonstrated, not asserted.** Run under
   `--simulate-used-memory 16GB` and report what actually happens at a useful
   context length. Real 16 GB hardware is better; ask.
5. `MELLUM.md` updated with the artifact's numbers and how to build it.

## Traps that have already cost time here

- **`make -j8` does not build `ds4_test`**, and `make cpu` overwrites `ds4`,
  `ds4-server`, `ds4-bench`, `ds4-eval` and `ds4-agent` with CPU builds. Both
  have produced believed-passing tests that were never compiled, and a server
  that refused Mellum with "requires --metal". Use `make test`, and rebuild
  everything after any `make cpu`.
- **Three diagnostics run decode, not prefill** — `layer0`, `all-layers`,
  `logits`. A prefill-only change will show a clean pass in all three.
- **Interleave measurement arms and read ratios.** The same baseline has
  measured 128 ms and 323 ms depending on machine load.
- **Context setting is not context used.** Decode figures need
  `DS4_MELLUM_PROFILE_DECODE_DEPTH`.
- **The eval prompts are inside the imatrix calibration set** for this
  artifact. Its quality scores are a regression check, not a benchmark. If you
  need an out-of-sample number, rebuild the imatrix without them.

## Sequencing

Do (1) and (2), measure, and report before starting (3). The first two are
worth shipping on their own; the third is an optimization on top and has a
defensible fallback. Estimating all three as one unit is how this gets
mis-scoped.

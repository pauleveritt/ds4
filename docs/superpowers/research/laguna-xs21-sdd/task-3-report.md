# Task 3 report: XS 2.1 layout validation + resident bring-up

## Status: BLOCKED (layout validation complete; generation blocked by an out-of-scope kernel-dispatch gap)

## What changed

File: `ds4.c`, inside `weights_validate_laguna_layout` (now ~5015-5185).

Confirmed per correction #2 that the legacy-layout marker detection
(`w->token_embd && type==Q4_K`) already classifies XS 2.1's official
Q4_K_M file as legacy — no change needed there.

Two real gaps existed in the legacy-layout branch, both from being written
only against S 2.1's F16-attention legacy recipe:

1. **`attn_type` hardcoded to F16.** Made it variant-aware:
   `signal_q8 ? Q8_0 : (DS4_MODEL_VARIANT == DS4_VARIANT_LAGUNA_XS21 ? Q4_K : F16)`.
   XS 2.1's official Q4_K_M file has attn_q/attn_k/attn_gate/attn_output as
   Q4_K (confirmed in xs2-facts.md section 4), not F16.

2. **`attn_v` needed per-layer Q4_K/Q6_K tolerance.** XS 2.1's attn_v type
   varies per layer (moves together with `ffn_down_exps`/`ffn_down_shexp`,
   uncorrelated with the SWA/global attention pattern — xs2-facts.md
   section 5). Added a `attn_v_tolerant` branch, mirroring the existing
   `ffn_down_exps`/`ffn_down_shexp` per-layer-type-tolerance pattern
   (ds4.c:5129-5162): accept either Q4_K or Q6_K, die with a clear message
   on anything else, then validate the tensor against its own observed type.

Verified per correction #3c that `ffn_gate_exps`/`ffn_up_exps`/
`ffn_down_exps`/`ffn_gate_shexp`/`ffn_up_shexp` and dense-layer FFN tensors
already matched XS 2.1's actual types without changes — confirmed against
xs2-facts.md sections 4/5.

Confirmed the config-expectation loader (`config_expect_u32` calls,
ds4.c:5807-5887+) already reads from `DS4_N_*` macros that resolve through
`g_ds4_shape`, which Task 2 already wired to select
`DS4_SHAPE_LAGUNA_XS21` by block_count (ds4.c:5997-5999). No hardcoded
literal config expectations remained to fix — Task 2's variant selection
already made these generic.

## Build

```
$ make -j8
```
Clean build, no warnings, all five binaries (ds4, ds4-server, ds4-bench,
ds4-eval, ds4-agent) linked successfully.

## Generation test — Q4_K_M (the real deliverable)

```
$ DS4_LOCK_FILE=/tmp/ds4-task3-<random>.lock ./ds4 -m gguf/Laguna-XS-2.1-Q4_K_M.gguf -c 4096 -p "def fizzbuzz(n):" -n 64 --nothink
```

Output:
```
ds4: Metal device Apple M5 Max, 128.00 GiB RAM
ds4: Metal 4 tensor API enabled for Tensor kernels
ds4: Metal model views created in 0.450 ms, residency requested in 3530.518 ms, warmup 2.957 ms (mapped 19331.55 MiB from offset 3.55 MiB)
ds4: Metal mapped mmaped model as 1 overlapping shared buffers
ds4: memory: KV 0.21 GiB + buffers 0.00 GiB + resident model 18.88 GiB = 19.09 GiB planned
ds4: Laguna Metal graph: ctx=4096, prefill=4096, KV 0.21 GiB, scratch 1017.66 MiB
processing 46 input tokens: 46/46 (100.0%)
To|UNK||UNK||UNK||UNK|... (repeats UNK 63 times)
ds4: prefill: 290.61 t/s, generation: 93.89 t/s
```

The model **loads** correctly (layout validation passes, no die, residency
maps the full 19.3 GB file on Metal, prefill of 46 tokens succeeds at
290 t/s). But single-token decode is broken: after one plausible-looking
token ("To"), every subsequent generated token is `〈|UNK|〉` (vocab id 0,
confirmed via `gguf.GGUFReader` dump of `tokenizer.ggml.tokens[0]` —
literal UNK token). This is not "no NaNs, no die, plausible tokens" — it's
a degenerate/garbage-logits failure mode (repeatedly landing on the same
special token is the classic signature of NaN-poisoned logits collapsing
argmax to index 0).

## Root cause (found, NOT fixed — out of scope per task instructions)

Traced the bug to `laguna_graph_forward_token` (ds4.c:47506+), the
single-token **decode** path used during generation (as opposed to
`laguna_graph_forward_batch`, used for prefill). At ds4.c:47557-47601:

```c
if (l->attn_q->type == DS4_TENSOR_F16) {
    ok = ds4_gpu_laguna_qkvg_f16_tensor(...);   /* fused F16 QKVG kernel */
} else {
    ok = ds4_gpu_matmul_q8_0_pair_tensor(...)   /* Q8_0-specific kernel */
      && ds4_gpu_matmul_q8_0_pair_tensor(...);
}
```

This binary branch assumes attn_q/k/v/gate is *either* F16 (legacy S 2.1)
*or* Q8_0 (signal recipe) — the two cases that existed before XS 2.1. For
XS 2.1's Q4_K_M file, `l->attn_q->type == DS4_TENSOR_Q4_K`, so it falls
into the `else` branch and calls a **Q8_0-specific** fused kernel
(`ds4_gpu_matmul_q8_0_pair_tensor`) on **Q4_K-encoded bytes** — silently
computing garbage (wrong dequant), which propagates through Q/K/attention
and collapses generation to the UNK token.

By contrast, the **prefill** path (`laguna_graph_forward_batch`, used via
`laguna_graph_matmul`, ds4.c:47468-47503) is fully type-generic — it
dispatches on `weight->type` through F16 / Q6_K / a fully generic
`ds4_gpu_matmul_quant_tensor` fallback that presumably handles Q4_K
correctly. That's why prefill of the 46-token prompt succeeded cleanly
while decode did not. `attn_output`'s dispatch (ds4.c:47647) and the FFN
routed/shared dispatch (ds4.c:47786) both already fall back to the
generic, type-safe path for non-F16 types, so they are NOT affected — the
bug is isolated to the fused QKVG projection in the decode path.

Per the task's explicit ambiguity-resolution rule ("if the failure is
happening past layout validation ... in the actual matmul/generation
code, ... stop and report rather than guessing at a fix" and "do not
touch ... Metal kernel selection/dispatch code"), I stopped here rather
than attempting a fix. A real fix would need either a new fused Q4_K QKVG
decode kernel, or falling back to the generic per-tensor
`laguna_graph_matmul` for Q/K/V/gate in the decode path when the type
isn't F16/Q8_0 (mirroring how `attn_output` and the FFN paths already
behave) — either way this is kernel-dispatch work requiring a scope
conversation, not a layout-validation fix.

## BF16 file — `--inspect` sanity check (per corrected scope, one attempt only)

```
$ DS4_LOCK_FILE=/tmp/ds4-task3-bf16-<random>.lock ./ds4 -m gguf/Laguna-XS-2.1-BF16.gguf --inspect
ds4: unsupported Laguna quantization layout marker bf16; expected legacy Q4_K/F16 or Q8_0 signal weights
$ echo $?
1
```

Dies cleanly with a clear "unsupported type" message, exit code 1, no
crash, no ambiguous behavior. No action needed — matches the expected
"clean rejection" outcome from the scope decision. Did not attempt to
make bf16 inference work (confirmed out of scope).

## Full test suite

```
$ make ds4_test && DS4_LOCK_FILE=/tmp/ds4-task3-test-<random>.lock ./ds4_test
```
All suites passed, ending `ds4 tests: ok` (exit 0). Includes
`metal-kernels: OK`, `metal-tensor-equivalence: OK`,
`streaming-decode-prefill-correctness: OK` (skipped, needs env var),
`local-golden-vectors: OK`, `metal-short-prefill: OK`, `server: OK`, and
all Laguna-specific numeric-exactness cases (GQA3 decode, paired Q/K
norm/RoPE) unchanged from before this change. No regressions from the
layout-validator edit.

## Commit

Committed the layout-validation fix only (Step 1 of the brief) — it is
correct, tested, causes no regressions, and is a genuine partial
deliverable. Did NOT commit any change to the decode-path kernel dispatch
since none was made (that's the blocked item).

## Self-review

- The layout-validation changes are narrow, mirror the existing
  `ffn_down_exps`/`ffn_down_shexp` per-layer-tolerance pattern exactly as
  instructed, and are variant-gated (`DS4_MODEL_VARIANT ==
  DS4_VARIANT_LAGUNA_XS21`) so S 2.1's legacy F16 path is untouched (S 2.1
  tests all still pass unchanged).
- I did not touch `laguna_graph_forward_token`, Metal kernel
  selection/dispatch, or attention math, per the explicit boundary in the
  task instructions.
- The core deliverable ("XS 2.1 loads and generates correctly on Metal")
  is NOT achieved — loading works, generation does not. This needs a
  scope conversation: either extend this task or open a follow-up task to
  add Q4_K-aware decode-path QKVG dispatch.
- I did not spend extra attempts on the bf16 file beyond the one
  `--inspect` call, per instructions.

## Follow-up: decode-path QKVG dispatch fix (this session)

### Status: DONE

The root cause identified above was confirmed correct and fixed. The
QKVG dispatch in `laguna_graph_forward_token` (ds4.c, per-layer loop
around line 47557) assumed non-F16 attention weights were always Q8_0
and called `ds4_gpu_matmul_q8_0_pair_tensor` unconditionally in the
`else` branch. XS 2.1's legacy attention weights are Q4_K
(attn_q/k/gate always Q4_K, attn_v Q4_K-or-Q6_K per layer), which fell
into that branch and got dequantized as if they were Q8_0 bytes,
producing garbage logits after the first decoded token.

### Fix

Changed the binary `if (F16) ... else ...` dispatch to a three-way
branch:

1. `l->attn_q->type == DS4_TENSOR_F16` — unchanged, still calls the
   fused `ds4_gpu_laguna_qkvg_f16_tensor` kernel (S 2.1 legacy recipe).
2. `l->attn_q->type == DS4_TENSOR_Q8_0` — unchanged, still calls the
   paired `ds4_gpu_matmul_q8_0_pair_tensor` kernels (S 2.1 signal
   recipe / any future Q8_0 XS 2.1 recipe).
3. New `else` branch — calls the existing generic, type-dispatching
   `laguna_graph_matmul` helper (ds4.c:47468-47504, already used
   elsewhere in this same function for FFN weights, and by
   `laguna_graph_forward_batch`'s prefill path for these exact same
   Q/K/V/gate projections at ds4.c:48012-48037) once each for
   `g->q`/`l->attn_q`, `g->k`/`l->attn_k`, `g->v`/`l->attn_v`, and
   `g->gate`/`l->attn_gate`, each with `n_tokens=1`. This covers
   XS 2.1's Q4_K attention weights today, and any other type
   `laguna_graph_matmul` supports in the future, without special-casing
   Q4_K explicitly.

No RoPE, attention math, KV cache, or other subsystem code was touched.

### Verification

1. `make -j8` — clean build, no warnings, all five binaries linked.
2. Acceptance run:
   ```
   DS4_LOCK_FILE=/tmp/ds4-task3-fix-<pid>.lock ./ds4 -m gguf/Laguna-XS-2.1-Q4_K_M.gguf -c 4096 -p "def fizzbuzz(n):" -n 64 --nothink
   ```
   Produced 64 tokens of coherent, plausible Python explanation/solution
   text (starts "The FizzBuzz problem requires generating a sequence
   where numbers divisible by 3 are replaced with..."), no NaNs, no
   collapse to `〈|UNK|〉` or any repeated token, no die. prefill 466.68 t/s,
   generation 99.93 t/s.
3. `make ds4_test && ./ds4_test` — full suite green, ends `ds4 tests: ok`,
   including `server: OK` and all prior Laguna/DeepSeek/GLM numeric-
   exactness cases. No regressions.
4. Extra cross-check (not required, done anyway): re-ran with `-n 96` —
   output stayed coherent through the full extended continuation with no
   sign of degradation or collapse at any point, consistent with the
   64-token run.

### Commit

`ds4.c` change committed as "Fix Laguna decode QKVG dispatch for Q4_K
attention weights".

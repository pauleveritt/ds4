# Task 5 Report: Variant-gate the streaming guard

## Change

In `ds4.c`, inside the `DS4_MODEL_FAMILY_LAGUNA` block of `ds4_engine_open_internal`
(guard located via `grep -n 'ssd-streaming is not implemented for Laguna' ds4.c`,
found at line 57547; block spans roughly 57545-57551), replaced the unconditional
streaming rejection with a variant-gated one:

```c
if (e->ssd_streaming && DS4_MODEL_VARIANT != DS4_VARIANT_LAGUNA_XS21) {
    fprintf(stderr,
            "ds4: --ssd-streaming for Laguna is only supported "
            "for Laguna XS 2.1\n");
    ds4_engine_close(e);
    *out = NULL;
    return 1;
}
```

All other Laguna restrictions in that block (Metal-only requirement, layer
slicing/distributed/TP/multi-GPU restriction, directional steering /
power-percent / prefill-chunk / MTP / dspark / glm-mtp / first-token-test
restriction) were left untouched.

## Manual verification

**S 2.1 still refuses** (`gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf --ssd-streaming`):

```
$ ./ds4 -m gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf --ssd-streaming -p hi -n 1 2>&1 | tail -1
ds4: --ssd-streaming for Laguna is only supported for Laguna XS 2.1
exit=1
```

Matches the expected message and exit code from the brief.

**XS 2.1 gets past the guard** (`gguf/Laguna-XS-2.1-Q4_K_M.gguf --ssd-streaming`):

```
$ ./ds4 -m gguf/Laguna-XS-2.1-Q4_K_M.gguf --ssd-streaming -p hi -n 1 2>&1 | tail -20
ds4: Metal device Apple M5 Max, 128.00 GiB RAM
ds4: Metal 4 tensor API enabled for Tensor kernels
ds4: drift-patch flags hc_stable=on norm_unify=on kv_raw_f32=off rope_exp2_log2=off math_safe=off tensor_matmul=on
ds4: Metal SSD streaming mode enabled; full model residency and warmup are skipped
ds4: SSD streaming auto cache budget
ds4:   metal recommends 107.52 GiB working set
ds4:   using 80% total for model + cached experts: 86.02 GiB
ds4:   non-routed weights: 9.64 GiB
ds4:   routed expert size: 1.95 MiB
ds4:   expert budget before prefill reserve: 10240 (19.45 GiB)
ds4: metal SSD streaming total expert budget 19.45 GiB = 0.97 GiB prefill headroom + 18.48 GiB dynamic cache (9728 experts, 1.95 MiB each)
ds4: SSD streaming mixed-precision model: 20/39 routed layers off the slab size class will bypass the expert cache and read experts via mapped model views
ds4: WARNING: the majority of routed layers (20/39) are off the slab size class (is the FIRST routed layer itself boosted?); expert-cache hit rate will be catastrophic
ds4: SSD streaming initial metal model map restricted to token embedding (1 spans, 0.11 GiB tensor span)
ds4: metal backend initialized for graph diagnostics
ds4: memory: KV 1.31 GiB ... = 20.87 GiB planned
ds4: Laguna Metal graph: ctx=32768, prefill=16384, KV 1.31 GiB, scratch 4069.45 MiB
ds4: Metal model range 0.27..0.27 GiB is not covered by mapped model views
ds4: Laguna batch prefill failed in attention norm after 0/40 layers
ds4: prompt processing failed: metal Laguna prefill failed at token 0
```

Confirmed: the guard's refusal message (`--ssd-streaming for Laguna is only
supported for Laguna XS 2.1`) does NOT appear. Execution proceeds well past
the guard into SSD streaming initialization and fails much further downstream
("Laguna batch prefill failed in attention norm after 0/40 layers") — this is
the expected, not-yet-wired-up streaming path that Task 6 will address. No
attempt was made to investigate or fix this downstream failure, per
instructions.

## Test suite

No new automated test was added. `test_laguna_variant_shapes` (the only
existing Laguna-variant test, near line 3903 of `tests/ds4_test.c`) is a pure
struct/config field assertion test against `DS4_SHAPE_LAGUNA_S21` /
`DS4_SHAPE_LAGUNA_XS21`, not an engine-open/CLI refusal test. The only
existing engine-open test helpers (`test_open_engine` / `test_get_engine`,
around line 90-120 and 6900-6930) hard-code a single fixed test model path
(the DeepSeek `ds4flash.gguf`) and aren't set up to swap in alternate
Laguna GGUFs. There is no clean existing pattern for testing this kind of
CLI-level engine-open refusal, so per the brief's ambiguity-resolution
guidance, this task relies on the manual verification above rather than
inventing a new test harness for a two-line gate change.

`make ds4_test && ./ds4_test`: **all green** — final line: `ds4 tests: ok`.
(Full build of `ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, `ds4-agent`, and
`ds4_test` all succeeded with no warnings/errors introduced by this change.)

## Self-review

- Change is minimal and scoped exactly to the guard condition, as instructed
  — no other logic in the Laguna block was touched.
- `DS4_MODEL_VARIANT` / `DS4_VARIANT_LAGUNA_XS21` are existing macros/enum
  values already used elsewhere in `ds4.c` (e.g. lines 5083, 5092, 651 in
  `ds4.h`), confirmed via grep before use — no new symbols introduced.
- Non-Metal + Laguna guard (the preceding `if (e->backend != DS4_BACKEND_METAL)`
  check) is untouched and still fires before the streaming check, so
  "Non-Metal + Laguna still refuses" as required by the brief's interface
  contract.
- Verified via grep that no other place in `ds4.c` prints or depends on the
  old message string ("--ssd-streaming is not implemented for Laguna S 2.1
  yet") before finalizing.

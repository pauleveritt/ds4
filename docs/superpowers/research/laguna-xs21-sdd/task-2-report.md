# Task 2 Report: Laguna XS 2.1 shape variant, detection, and head-table generalization

## What changed

### tests/ds4_test.c
Added `test_laguna_variant_shapes()` (after the existing Metal Laguna QK-norm/RoPE
test, ~line 3903), asserting both `DS4_SHAPE_LAGUNA_S21` (unchanged: n_layer=48,
n_head=72, n_head_global=48, n_head_swa=72) and the new
`DS4_SHAPE_LAGUNA_XS21` (n_layer=40, n_embd=2048, n_vocab=100352, n_head=64,
n_head_global=48, n_head_swa=64, n_expert=256, n_expert_used=8, n_ff_exp=512,
n_ff_dense=8192, n_leading_dense=1, n_rot=64, n_rot_swa=128,
rope_scale_factor=32.0, rope_orig_ctx=8192, rope_yarn_attn_factor=1.0) against
the GGUF-verified constants from `docs/superpowers/plans/xs2-facts.md`.
Registered the call inside `test_server_unit_group()`.

### ds4.c
- Removed the private `ds4_model_family` / `ds4_variant` / `ds4_shape`
  type definitions (previously ~line 480-540, static to the translation
  unit) — they moved to `ds4.h` (see below).
- Added `DS4_VARIANT_LAGUNA_XS21 = 4` to `ds4_variant` (now in ds4.h).
- Added `n_head_global` / `n_head_swa` fields to `ds4_shape` (now in
  ds4.h); populated on `DS4_SHAPE_LAGUNA_S21` (48/72) and the new
  `DS4_SHAPE_LAGUNA_XS21` (48/64). Left zero (unused) on the
  DeepSeek/GLM shapes.
- `DS4_SHAPE_LAGUNA_S21` and the new `DS4_SHAPE_LAGUNA_XS21` are now
  `const` (non-static) globals so the extern declarations in ds4.h can
  reach them from the test binary.
- Added `DS4_SHAPE_LAGUNA_XS21` shape struct (after `DS4_SHAPE_LAGUNA_S21`)
  with the values from the task brief, straight from the GGUF-verified
  facts file.
- `config_validate_laguna_model()`: moved the `laguna.block_count` read
  to the top of the function and used it to select the variant:
  48 → `DS4_SHAPE_LAGUNA_S21`, 40 → `DS4_SHAPE_LAGUNA_XS21`, else
  `fprintf(stderr, ...); exit(1);` with an "unsupported Laguna
  block_count %u" message (matches the `ds4_die`-style fatal pattern
  used elsewhere in this function).
- Per-layer head-count check (~line 6038): replaced the hardcoded
  `(il % 4u) == 0 ? 48u : 72u` with
  `(il % 4u) == 0 ? g_ds4_shape.n_head_global : g_ds4_shape.n_head_swa`,
  so the check is correct for whichever Laguna variant is loaded.

### ds4.h
- Added the `ds4_model_family` and `ds4_variant` enum typedefs and the
  `ds4_shape` struct typedef (moved from ds4.c, plus the two new
  `n_head_global`/`n_head_swa` fields).
- Added `extern const ds4_shape DS4_SHAPE_LAGUNA_S21;` and
  `extern const ds4_shape DS4_SHAPE_LAGUNA_XS21;` declarations.

## Why the shape struct/type moved to ds4.h

The new test needs to assert directly on `DS4_SHAPE_LAGUNA_S21.n_head_global`
etc. `tests/ds4_test.c` builds as its own translation unit and links
against `ds4.o`; it does not `#include "ds4.c"`. Since `ds4_shape` and the
two Laguna constant structs were `static`/file-local to ds4.c, the test
file had no way to name the type or reach the constants — the previous
plan's approach (hardcoding a duplicate expectation) was exactly what
this task is generalizing away from. Moving the type definitions to
ds4.h (which both ds4.c and ds4_test.c already include) and promoting
`DS4_SHAPE_LAGUNA_S21` / `DS4_SHAPE_LAGUNA_XS21` from `static const` to
plain `const` (file-scope, external linkage) with matching `extern`
declarations in the header is the standard C pattern for exposing
specific constants to a test binary without making the whole shape
table public. Everything else in the shape table (`DS4_SHAPE_FLASH`,
`DS4_SHAPE_PRO`, `DS4_SHAPE_GLM52`, `g_ds4_shape`, etc.) stays private
to ds4.c — only the two Laguna shapes the test needs were exposed.

`ds4_laguna_layer_is_swa()` (ds4.c:1120) was checked and left
unmodified: it keys on `ds4_layer_head_count(il) == DS4_N_HEAD`
(`DS4_N_HEAD` expands to `g_ds4_shape.n_head`). For XS 2.1,
`n_head_global=48` and `n_head_swa=n_head=64`, so on `il % 4 == 0`
layers the head count (48) differs from `n_head` (64) → correctly
classified non-SWA; on other layers the head count (64) equals `n_head`
(64) → correctly classified SWA. Same logic holds for S 2.1
(global=48 vs n_head=72). No change needed there, confirmed by the
full test suite (see below) rather than by inspection alone.

## Build

```
rm -f ds4_test ds4.o && make ds4_test
```

Compiled cleanly, no warnings, no errors:

```
cc -O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99 -c -o ds4.o ds4.c
cc -O3 -ffast-math -g -mcpu=native -Wall -Wextra -std=c99 -o ds4_test ds4_test.o ds4_help.o ds4_kvstore.o rax.o ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_metal.o ds4_layer_pack.o -lm -pthread -framework Foundation -framework Metal
```

## Test run

```
DS4_LOCK_FILE=/tmp/ds4-task2-finish.lock ./ds4_test
```

Final line: `ds4 tests: ok` (0 failures), confirmed stable across two
consecutive full runs after rebuilding from a clean object-file state.
`test_laguna_variant_shapes` (registered inside `test_server_unit_group`)
passed — no assertion failures anywhere near tests/ds4_test.c:3903-3925 in
either run's log.

Note on an initial red herring: the first two runs against this same
code (before the clean rebuild used for the final report) showed 43-48
failures concentrated in `long-context`, `tool-call-quality`, and
`think-tool-recovery` (all real-decode tests against the large DeepSeek
`ds4flash.gguf` model, unrelated to the Laguna code paths touched here).
To rule out a regression I stashed the changes, rebuilt, and reran: the
unmodified baseline also intermittently failed the same way under
repeated back-to-back full-suite runs loading multi-GB GGUF files
(memory/thermal pressure from repeated large mmap + Metal residency
setup, not a correctness bug). After a clean rebuild with the changes
restored, two consecutive full runs both completed with `ds4 tests: ok`,
including `metal-tensor-equivalence: OK`. The Laguna shape/variant
changes do not touch the DeepSeek decode path, and the flakiness was
reproducible on the pre-change baseline too, so it is an unrelated,
environment-driven flake and not attributable to this change.

## Self-review

- Matches the task brief's interfaces exactly:
  `DS4_VARIANT_LAGUNA_XS21` enum value, `DS4_SHAPE_LAGUNA_XS21` struct,
  `n_head_global`/`n_head_swa` shape fields, variant selection by
  `laguna.block_count` (48 → S21, 40 → XS21, else fatal).
- `DS4_SHAPE_LAGUNA_XS21` values copied verbatim from the brief, which
  states they are GGUF-verified against
  `docs/superpowers/plans/xs2-facts.md` — not re-derived or guessed.
- DeepSeek/GLM shapes (`DS4_SHAPE_FLASH`, `DS4_SHAPE_PRO`,
  `DS4_SHAPE_GLM52`) were not touched beyond the header move; their
  `n_head_global`/`n_head_swa` fields are zero-initialized (unused by
  those families) as the brief specifies.
- Scope check: `git status` before commit showed only `ds4.c`, `ds4.h`,
  `tests/ds4_test.c` modified plus an untracked `gguf/` directory
  (pre-existing model files, not part of this change) — nothing else
  touched or staged.

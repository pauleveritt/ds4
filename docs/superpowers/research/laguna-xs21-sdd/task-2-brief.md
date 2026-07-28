### Task 2: XS 2.1 shape variant, detection, and head-table generalization

> **Values below are GGUF-verified** from `docs/superpowers/plans/xs2-facts.md`
> (Task 1). They are NOT placeholders — use them exactly. The model that
> actually exists upstream is **Laguna XS 2.1**
> (`poolside/Laguna-XS-2.1-GGUF`); this plan was originally written against
> an older, differently-shaped model, and those constants were wrong. The
> values here replace them.

**Files:**
- Modify: `ds4.c` (~483 variant enum, ~671 shape structs, ~6054 head check, laguna config-load block ~5997-6100)
- Modify: `ds4.h` (only if a public predicate is needed; expect no change)
- Test: `tests/ds4_test.c`

**Interfaces:**
- Consumes: `docs/superpowers/plans/xs2-facts.md` (authoritative constants).
- Produces: `DS4_VARIANT_LAGUNA_XS21` enum value; `DS4_SHAPE_LAGUNA_XS21` struct; shape fields `n_head_global`/`n_head_swa` replacing the hardcoded 48/72 expectation; variant selection by `laguna.block_count`. Later tasks reference the variant constant by exactly this name.

- [ ] **Step 1: Write the failing test** — add to `tests/ds4_test.c` next to the existing Laguna tests:

```c
static void test_laguna_variant_shapes(void) {
    /* S 2.1 stays intact. */
    TEST_ASSERT(DS4_SHAPE_LAGUNA_S21.n_layer == 48);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_S21.n_head == 72);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_S21.n_head_global == 48);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_S21.n_head_swa == 72);
    /* XS 2.1, GGUF-verified (xs2-facts.md). */
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.n_layer == 40);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.n_embd == 2048);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.n_vocab == 100352);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.n_head == 64);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.n_head_global == 48);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.n_head_swa == 64);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.n_expert == 256);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.n_expert_used == 8);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.n_ff_exp == 512);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.n_ff_dense == 8192);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.n_leading_dense == 1);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.n_rot == 64);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.n_rot_swa == 128);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.rope_scale_factor == 32.0f);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.rope_orig_ctx == 8192);
    TEST_ASSERT(DS4_SHAPE_LAGUNA_XS21.rope_yarn_attn_factor == 1.0f);
}
```

Register it in the test main next to the other Laguna calls.

- [ ] **Step 2: Run to verify failure**

```bash
make ds4_test 2>&1 | tail -5
```

Expected: compile error `DS4_SHAPE_LAGUNA_XS21 undeclared` (and `n_head_global` not a member).

- [ ] **Step 3: Implement.** In `ds4.c`:

(a) Variant enum (~line 490): add `DS4_VARIANT_LAGUNA_XS21 = 4,` (use the next free value; check the enum).

(b) Add two fields to `ds4_shape`: `uint32_t n_head_global; uint32_t n_head_swa;`. Set them in `DS4_SHAPE_LAGUNA_S21` (global 48, swa 72). Leave zero for DeepSeek/GLM shapes (unused there).

(c) New shape struct after `DS4_SHAPE_LAGUNA_S21`:

```c
static const ds4_shape DS4_SHAPE_LAGUNA_XS21 = {
    .name = "Laguna XS 2.1",
    .family = DS4_MODEL_FAMILY_LAGUNA,
    .variant = DS4_VARIANT_LAGUNA_XS21,
    .n_layer = 40,
    .n_embd = 2048,
    .n_vocab = 100352,
    .n_head = 64,            /* majority/SWA value, mirrors S21's n_head=72 */
    .n_head_global = 48,     /* head_count array [48,64,64,64] x10 */
    .n_head_swa = 64,
    .n_head_kv = 8,
    .n_head_dim = 128,
    .n_value_dim = 128,
    .n_rot = 64,
    .n_expert = 256,
    .n_expert_used = 8,
    .n_expert_shared = 1,
    .n_ff_exp = 512,
    .n_ff_shared = 512,
    .n_ff_dense = 8192,
    .n_swa = 512,
    .n_leading_dense = 1,    /* 39 sparse layers, 9984 routed experts */
    .n_rot_swa = 128,
    .rms_eps = 1.0e-6f,
    .expert_weight_scale = 2.5f,
    .rope_freq_base = 500000.0f,
    .rope_scale_factor = 32.0f,
    .rope_yarn_beta_fast = 64.0f,
    .rope_yarn_beta_slow = 1.0f,
    .rope_yarn_attn_factor = 1.0f,
    .rope_freq_base_swa = 10000.0f,
    .context_length = 262144,
    .rope_orig_ctx = 8192,
};
```

(d) Variant selection: in the Laguna config-load path (where `DS4_SHAPE_LAGUNA_S21` is installed into `g_ds4_shape`), read `laguna.block_count` FIRST and select: 48 → S21, 40 → XS21, else `ds4_die("unsupported Laguna block_count %u")`.

(e) Head check at ~6054: replace the literal expectation with:

```c
const uint32_t expected = (il % 4u) == 0
        ? g_ds4_shape.n_head_global
        : g_ds4_shape.n_head_swa;
```

`ds4_laguna_layer_is_swa` (ds4.c:1141) keys on `head_count == DS4_N_HEAD`. XS 2.1's values are global 48 / SWA 64 and `n_head` is 64, so the existing predicate still discriminates correctly for both variants — no change needed there. Verify this holds rather than assuming it: if for any reason the predicate misclassifies, switch it to `(il % 4u) != 0` for the whole family (both variants are every-4th-global, so it is equivalent), and confirm the S21 assertions still pass.

- [ ] **Step 4: Run tests**

```bash
make ds4_test && ./ds4_test 2>&1 | tail -3
```

Expected: PASS including existing S 2.1 Laguna tests (proves no regression).

- [ ] **Step 5: Commit**

```bash
git add ds4.c tests/ds4_test.c
git commit -m "Add Laguna XS 2.1 shape variant and per-variant head tables"
```


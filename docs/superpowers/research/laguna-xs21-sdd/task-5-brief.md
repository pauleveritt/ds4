### Task 5: Variant-gate the streaming guard

**Files:**
- Modify: `ds4.c:57517-57551` (guard block)
- Test: `tests/ds4_test.c` + manual refusal checks

**Interfaces:**
- Produces: `--ssd-streaming` proceeds for XS 2.1, still refuses for S21 with the existing message. Non-Metal + Laguna still refuses.

- [ ] **Step 1: Change the guard.** Inside the `DS4_MODEL_FAMILY_LAGUNA` block replace the unconditional streaming rejection with:

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

Keep every other Laguna restriction in that block untouched.

- [ ] **Step 2: Verify refusals manually**

```bash
./ds4 -m gguf/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf --ssd-streaming -p hi -n 1 2>&1 | tail -1
```

Expected: `ds4: --ssd-streaming for Laguna is only supported for Laguna XS 2.1`, exit 1.

XS 2.1 + streaming will fail deeper in (not wired yet) — that's Task 6's job; confirm it gets PAST this guard.

- [ ] **Step 3: `make ds4_test && ./ds4_test` green; commit**

```bash
git add ds4.c
git commit -m "Gate Laguna SSD streaming to the XS 2.1 variant"
```


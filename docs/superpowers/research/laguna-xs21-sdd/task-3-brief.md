### Task 3: XS 2.1 layout validation + resident bring-up on the laptop

**Files:**
- Modify: `ds4.c` (`weights_validate_laguna_layout` ~5036, layout markers ~5055, config_expect values in loader)
- Test: manual generation run + `./ds4_test`

**Interfaces:**
- Consumes: Task 2 variant; xs2-facts.md item 6 (Q4_K_M/BF16 tensor type maps).
- Produces: XS 2.1 loads resident on Metal; layout markers `xs21-official-q4km` and `xs21-official-bf16` accepted (exact marker mechanism per existing S21 markers at ds4.c:5055 — follow the same detection style: identify by tensor-type pattern, not filename).

- [ ] **Step 1: Extend validation.** In `weights_validate_laguna_layout`, add recognition for the two official XS 2.1 layouts from xs2-facts.md item 6 (per-layer: routed gate/up/down types, shared types, attention types). Mirror the S21 recipe checks structurally — same loop, same die-with-marker-name on mismatch.

- [ ] **Step 2: Fix any remaining hardcoded S21 config expectations.** Build and run against the Q4_K_M file; every `config_expect_u32` that dies tells you which expectation needs to come from the shape struct rather than a literal. Iterate until load completes:

```bash
make -j8 && ./ds4 -m gguf/Laguna-XS-2.1-Q4_K_M.gguf -c 4096 -p "def fizzbuzz(n):" -n 64 --nothink
```

Expected: 64 tokens of plausible Python continuation, no NaNs, no die.

- [ ] **Step 3: Sanity the second file**

```bash
./ds4 -m gguf/Laguna-XS-2.1-BF16.gguf -c 4096 -p "def fizzbuzz(n):" -n 64 --nothink
```

Expected: similar continuation, near-greedy agreement with Q4 on the first ~10 tokens.

- [ ] **Step 4: Run the full test suite** (`make ds4_test && ./ds4_test`) — green.

- [ ] **Step 5: Commit**

```bash
git add ds4.c
git commit -m "Accept official Laguna XS 2.1 Q4_K_M and BF16 layouts"
```


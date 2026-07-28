### Task 6: Wire Laguna MoE dispatch through the streaming expert cache

This is the load-bearing task. Budget the most time here.

**Files:**
- Modify: `ds4.c` (Laguna Metal graph build/decode path — locate via `grep -n 'laguna_routed_shared_moe' ds4.c`), possibly `ds4_metal.m` + `metal/laguna.metal`/`metal/moe.metal` if a cache-reading kernel variant is needed
- Test: Task 7's A/B harness is the true test; unit-level via `./ds4_test`

**Interfaces:**
- Consumes: `ds4_gpu_stream_expert_table` (ds4_gpu.h:152), `ds4_gpu_stream_expert_cache_seed_selected` / `..._begin_selected_load` / `..._budget_for_expert_size`, GLM per-layer integration as template.
- Produces: XS 2.1 generates correct tokens with `--ssd-streaming` on Metal; cache hit/miss counters advance (`g_stream_expert_cache_hits/misses` visible in the existing streaming stats output).

- [ ] **Step 1: Study the GLM integration (read-only).** Map every call site:

```bash
grep -n 'stream_expert_cache' ds4.c | grep -in 'glm' 
grep -n 'glm_stream_expert_cache_begin_selected_load_tensor' ds4.c ds4_metal.m
grep -n 'stream_expert_table' ds4.c | head -20
```

Write a short call-flow note (which function builds the table per layer, where selected-expert ids come from, where the cache is consulted during decode vs prefill, how misses fall back to mmap reads) into the task journal/commit message. Do not skip this step — the integration points below must match what you find, and the GLM flow is authoritative over this plan's sketch.

- [ ] **Step 2: Build the per-layer stream table for Laguna.** Where the Laguna graph currently passes `model_map` offsets into `ds4_gpu_laguna_routed_shared_moe_one_tensor` via `ds4_gpu_laguna_moe_desc`, populate a `ds4_gpu_stream_expert_table` per sparse layer when `e->ssd_streaming` (fields map 1:1: gate/up/down offsets and expert strides are already in the desc). Follow the GLM site for lifecycle (built once per layer at graph setup, reused per token).

- [ ] **Step 3: Route decode through the cache.** At the Laguna decode dispatch, when streaming: call `ds4_gpu_stream_expert_cache_begin_selected_load(table, selected_ids, n_selected)` (or the `_tensor` variant if selection lives on-GPU, matching how Laguna computes routing — check whether Laguna's top-k selection result is host-visible like GLM's; use the same variant GLM uses if not). Misses must fall back to the existing mmap path so output is ALWAYS correct regardless of cache state — copy the GLM fallback structure exactly.

- [ ] **Step 4: Prefill.** Reuse the streaming prefill approach GLM uses (full-layer resident window; `ds4_gpu_set_glm_streaming_prefill_full_layer` naming suggests a GLM-specific toggle — if it is GLM-hardwired, add the Laguna case following the same shape rather than renaming shared plumbing).

- [ ] **Step 5: Budget wiring.** Confirm `ds4.c:30025` (`budget_for_expert_size`) picks up XS 2.1's `gate_expert_bytes`/`down_expert_bytes` from the first sparse layer's tensors — it reads `layer->ffn_gate_exps->dim[1] * gate_row_bytes`, which is shape-agnostic; verify by log line at startup showing a sane expert count (expect roughly 5 GiB / 1.8 MiB ≈ 2800 experts for Q4_K_M with the automatic budget on the laptop constrained run in Task 7).

- [ ] **Step 6: Smoke run**

```bash
./ds4 -m gguf/Laguna-XS-2.1-Q4_K_M.gguf --ssd-streaming -c 4096 -p "def fizzbuzz(n):" -n 64 --nothink
```

Expected: same class of output as resident; streaming stats line reports nonzero hits+misses.

- [ ] **Step 7: `make ds4_test && ./ds4_test` green; commit**

```bash
git add ds4.c ds4_metal.m metal/
git commit -m "Stream Laguna XS 2.1 routed experts through the Metal expert cache"
```

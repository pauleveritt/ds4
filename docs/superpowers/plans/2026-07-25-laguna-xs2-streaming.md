# Laguna XS 2.1 + Streaming + Biased Quant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ds4-agent running on a 16 GB Mac Mini at 32k context, backed by Laguna XS 2.1 streamed from SSD, with quant precision and expert-cache residency biased toward HTML/CSS/JS/TS/Python.

**Architecture:** Add an XS 2.1 variant to ds4's existing Laguna family support (shape struct + layout validation), route Laguna's Metal MoE dispatch through the existing shared streaming expert cache (GLM integration is the template), then bias two data artifacts: a llama.cpp-built RoutedQ3_K quant using a web/Python imatrix corpus, and an expert hotlist profiled on web/Python agent workloads.

**Tech Stack:** C (ds4.c engine), Objective-C/Metal (ds4_metal.m), Python 3 (corpus tooling), llama.cpp (imatrix collection + quantize, out-of-tree), zsh/Make.

**Spec:** `docs/superpowers/specs/2026-07-25-laguna-xs2-streaming-design.md`

## Global Constraints

- Personal branch: no CUDA/ROCm work, no upstream QA gates. Metal only.
- Deployment: 16 GB Mac Mini, ~10.5–11 GiB default wired budget, 32k context.
- Build machine: M5 Max 128 GB (this laptop). All heavy runs happen here.
- S 2.1 + `--ssd-streaming` must KEEP refusing. Only XS 2.1 streams.
- Corpus bias: ~55% HTML/CSS/JS/TS/Python + agent transcripts, ~30% prose/reasoning, ~15% shell/C. No Java/C#/Kotlin/PHP/Go/Rust.
- Die-loudly: unknown layouts/configs exit with a named reason, never limp.
- Test cycle: `make ds4_test && ./ds4_test` must stay green after every task.
- Commit after every task (small, single-purpose commits).

## Key repo anchors (verified 2026-07-25)

| What | Where |
|---|---|
| Laguna shape structs | `ds4.c:671` (`DS4_SHAPE_LAGUNA_S21`), variants enum `ds4.c:483-490` |
| Per-layer head check (hardcoded 48/72) | `ds4.c:6054` in the Laguna config loader |
| Laguna feature guard (streaming rejection) | `ds4.c:57517-57551` |
| Streaming cache API | `ds4_gpu.h:142-205` (`ds4_gpu_stream_expert_table`, seed/begin_selected_load) |
| Laguna MoE dispatch | `ds4_gpu_laguna_routed_shared_moe_one_tensor`, `ds4_gpu_laguna_moe_desc` (`ds4_gpu.h:2270-2292`) |
| GLM streaming template | `ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor` (`ds4_gpu.h:178`) |
| Hotlist writer (built-in) | `DS4_EXPERT_HOTLIST` env → `ds4_expert_profile_write_hotlist_file` (`ds4.c:1607`) |
| Hotlist runtime loader | `DS4_METAL_STREAMING_EXPERT_HOTLIST` env (`ds4.c:30038`) |
| Compiled-in hotlist defaults | variant switch `ds4.c:20841`, `.inc` includes `ds4.c:1360` |
| Cache budget calc | `ds4.c:30025` (`..._budget_for_expert_size`) |
| Layout validation | `weights_validate_laguna_layout` (`ds4.c:5036`) |
| Quant layout markers | `ds4.c:5055-5063` (marker string check) |
| Download targets | `download_model.sh:6-25` (repo/revision pins), case entries ~line 114 |
| Tests | `tests/ds4_test.c` (Laguna pattern at `test_metal_laguna_gqa3_decode_numeric`, `ds4.c` run via `make ds4_test && ./ds4_test`) |
| Quality fixtures | `gguf-tools/quality-testing/collect_official.py` + `data/` |
| Imatrix corpus builder | `gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py` |
| Mixed-quant splicer | `gguf-tools/mixed/splice_mixed_expert_layers_gguf.py` |

---

## Phase P0 — Facts and resident bring-up

### Task 1: Download official XS 2.1 GGUFs and record ground-truth facts

**Files:**
- Create: `docs/superpowers/plans/xs2-facts.md`
- Download to: `gguf/` (Q8_0 and Q4_K_M official files)

**Interfaces:**
- Produces: `xs2-facts.md` — the authoritative values for Tasks 2–4: per-layer head array, leading-dense count, rot/rot_swa, tensor names/types, exact file names. Every later task that says "per xs2-facts.md" reads this file.

- [ ] **Step 1: List the official GGUF repo to learn exact filenames**

```bash
curl -s "https://huggingface.co/api/models/poolside/Laguna-XS-2.1-GGUF/tree/main" | python3 -c "import json,sys; [print(f['path'], f.get('size',0)//2**20,'MiB') for f in json.load(sys.stdin)]"
```

Expected: a file list including a Q8_0 (~35 GB) and a Q4_K_M (~19 GB) GGUF. If the repo name 404s, find the exact name via `curl -s "https://huggingface.co/api/models?author=poolside&search=XS"` and use that repo for every later step; record the correction in xs2-facts.md.

- [ ] **Step 2: Download Q8_0 and Q4_K_M** (large; run in background, verify sizes after)

```bash
cd /Users/pauleveritt/projects/ds4/gguf
curl -L -O "https://huggingface.co/poolside/Laguna-XS-2.1-GGUF/resolve/main/Laguna-XS-2.1-BF16.gguf"
curl -L -O "https://huggingface.co/poolside/Laguna-XS-2.1-GGUF/resolve/main/Laguna-XS-2.1-Q4_K_M.gguf"
ls -lh laguna-xs*
```

Expected: both files present with sizes matching the tree listing (±1%).

- [ ] **Step 3: Inspect metadata with the existing inspector**

```bash
cd /Users/pauleveritt/projects/ds4
./ds4 -m gguf/Laguna-XS-2.1-Q4_K_M.gguf --inspect 2>&1 | head -80
```

Expected: this PRINTS METADATA BUT LIKELY DIES on the config check (block_count 40 ≠ 48) — that is fine; we only need the dump above the failure. If the inspect path dies before dumping, use the fallback:

```bash
python3 - <<'EOF'
from pathlib import Path
import json, struct
# Minimal GGUF KV reader: prints general.*, laguna.* keys and tensor list.
# gguf package is available via: pip3 install gguf
from gguf import GGUFReader
r = GGUFReader("gguf/Laguna-XS-2.1-Q4_K_M.gguf")
for f in r.fields.values():
    print(f.name, f.contents() if f.data else "")
for t in r.tensors[:20]:
    print(t.name, t.tensor_type, t.shape)
print("n_tensors:", len(r.tensors))
EOF
```

- [ ] **Step 4: Write xs2-facts.md**

Record, with the exact printed values:
1. `laguna.attention.head_count` per-layer array — flat 48, or reduced-on-global pattern (spec unknown #1). Derive `N_HEAD_GLOBAL` and `N_HEAD_SWA`.
2. `laguna.leading_dense_block_count` (1 or 2) and therefore sparse-layer count (39 or 38) and routed expert total (39×256=9984 or 38×256=9728) (spec unknown #2).
3. `laguna.rope.dimension_count` and `laguna.rope.dimension_count_swa` (spec unknown #3).
4. First-layer tensor names for gate/up/down routed + shared tensors and their GGUF types in both files (spec unknown #4).
5. Exact filenames + HF revision hash (`curl -s .../api/models/<repo>` → `sha`).
6. Q4_K_M per-tensor type map summary (which tensors are Q4_K vs Q6_K vs Q8_0) — this defines the Task 3 validation entry.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/xs2-facts.md
git commit -m "Record Laguna XS 2.1 GGUF ground-truth facts"
```

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

### Task 4: Quality fixtures for XS 2.1

**Files:**
- Create: `gguf-tools/quality-testing/data/laguna-xs21/` (continuations)
- Modify: `gguf-tools/quality-testing/collect_official.py` (model entry)

**Interfaces:**
- Consumes: working resident XS 2.1 (Task 3).
- Produces: fixture continuations used as the quality baseline in Tasks 10 and 13. Reference source: OpenRouter XS 2.1 if listed (spec unknown #5); otherwise laptop BF16 output is the reference — record which in the fixture README.

- [ ] **Step 1: Check OpenRouter availability**

```bash
curl -s https://openrouter.ai/api/v1/models | python3 -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['data'] if 'laguna' in m['id'].lower()])"
```

- [ ] **Step 2: Collect ~30 fixture cases** following the existing S 2.1 fixture pattern in `gguf-tools/quality-testing/` (the S 2.1 commit added `continuations/case_*.txt` + README). Prompt mix: 15 web/Python (HTML/CSS/JS/TS/Python), 10 general prose/reasoning, 5 tool-call format. Use `collect_official.py` with the OpenRouter id if available, else run the BF16 locally with the same prompts.

- [ ] **Step 3: Verify the local Q4_K_M against fixtures** using the existing quality-testing procedure (its README documents the comparison run). Record the agreement score in the fixture README as the P0 baseline.

- [ ] **Step 4: Commit**

```bash
git add gguf-tools/quality-testing/
git commit -m "Add Laguna XS 2.1 quality fixtures and P0 baseline"
```

---

## Phase P1 — Streaming port

### Task 5: Variant-gate the streaming guard

**Files:**
- Modify: `ds4.c:57517-57551` (guard block)
- Test: `tests/ds4_test.c` + manual refusal checks

**Interfaces:**
- Produces: `--ssd-streaming` proceeds for XS2, still refuses for S21 with the existing message. Non-Metal + Laguna still refuses.

- [ ] **Step 1: Change the guard.** Inside the `DS4_MODEL_FAMILY_LAGUNA` block replace the unconditional streaming rejection with:

```c
if (e->ssd_streaming && DS4_MODEL_VARIANT != DS4_VARIANT_LAGUNA_XS2) {
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

### Task 7: Streamed-vs-resident correctness gate

**Files:**
- Create: `tests/xs21_stream_ab.sh`

**Interfaces:**
- Consumes: Tasks 3 (resident) and 6 (streamed).
- Produces: the repeatable A/B gate used after every subsequent engine change.

- [ ] **Step 1: Write the harness**

```bash
#!/bin/zsh
# A/B: resident vs streamed logits/tokens must agree (greedy, fixed seed).
set -e
MODEL=${1:-gguf/Laguna-XS-2.1-Q4_K_M.gguf}
PROMPTS=("def fizzbuzz(n):" "<html><head><title>" "Explain HTTP caching briefly." "import asyncio")
for p in $PROMPTS; do
  A=$(./ds4 -m $MODEL -c 8192 -p "$p" -n 128 --nothink --temp 0)
  B=$(./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts 800 \
        -c 8192 -p "$p" -n 128 --nothink --temp 0)
  if [[ "$A" != "$B" ]]; then
    echo "MISMATCH on prompt: $p"; exit 1
  fi
done
echo "xs21 stream A/B: OK"
```

The 800-expert cache (< one layer's worth × several layers) forces heavy eviction — the stress case. Check `--temp 0` is the repo's greedy flag (`./ds4 --help | grep -i temp`); use the documented greedy mechanism if named differently, and per CONTRIBUTING.md use its output-agreement procedure if bytewise equality is not the repo's standard (adjust the comparison accordingly — the CONTRIBUTING procedure is authoritative).

- [ ] **Step 2: Run it**

```bash
chmod +x tests/xs21_stream_ab.sh && ./tests/xs21_stream_ab.sh
```

Expected: `xs21 stream A/B: OK`. Any mismatch is a Task 6 bug — fix there, re-run, only then proceed.

- [ ] **Step 3: Commit**

```bash
git add tests/xs21_stream_ab.sh
git commit -m "Add XS 2.1 streamed-vs-resident A/B correctness gate"
```

---

## Phase P2 — Mini baseline (first usable milestone)

### Task 8: Download target + Mini deployment run

**Files:**
- Modify: `download_model.sh` (add `xs21-q4` target following the `laguna-q4` pattern at lines 6-25/114: repo `poolside/Laguna-XS-2.1-GGUF`, pinned revision from xs2-facts.md item 5)
- Create: `docs/superpowers/plans/mini-notes.md` (deployment log)

**Interfaces:**
- Consumes: everything through Task 7.
- Produces: the P2 baseline measurements later tasks are compared against.

- [ ] **Step 1: Add and test the download target** (`./download_model.sh xs21-q4` on the laptop; expect skip/verify since the file exists).

- [ ] **Step 2: Deploy to the Mini** (build from this branch on the Mini, copy or re-download the Q4_K_M):

```bash
./ds4-agent -m gguf/Laguna-XS-2.1-Q4_K_M.gguf --ssd-streaming --ctx 32768
```

- [ ] **Step 3: Record in mini-notes.md**: startup expert-cache budget chosen by the automatic sizing, warm-cache decode t/s, cache hit rate after a 10-minute agent session, memory pressure observations, whether `iogpu.wired_limit_mb` was needed. Acceptance for P2: agent completes a simple multi-tool task (e.g., "read this file and summarize, then write a test") without OOM.

- [ ] **Step 4: Commit**

```bash
git add download_model.sh docs/superpowers/plans/mini-notes.md
git commit -m "Add xs2-q4 download target and Mini baseline notes"
```

---

## Phase P3 — Biased quant

### Task 9: Laguna-format biased corpus builder

**Files:**
- Create: `gguf-tools/imatrix/dataset/build_laguna_xs21_imatrix_dataset.py`
- Create: `gguf-tools/imatrix/dataset/laguna_xs21_manifest.json` (generated)

**Interfaces:**
- Consumes: Laguna chat/tool rendering (crib the exact template strings from the S 2.1 server/agent code — `grep -n 'laguna' ds4_server.c` for the template constants).
- Produces: `laguna_xs21_rendered_prompts.txt` (~3M tokens) consumed by Task 10. Composition contract: 55% web/Python, 30% prose/reasoning, 15% shell/C; zero Java/C#/Kotlin/PHP/Go/Rust.

- [ ] **Step 1: Fork the builder.** Start from `build_ds4_imatrix_dataset.py`; replace the DS4 prompt renderer with the Laguna template; replace the source-material mix:
  - web/Python: HTML/CSS/JS/TS/Python files sampled from well-known permissive repos you have locally or vendored snippets generated in-script (the DS corpus generates its material in-script — follow that pattern; no network fetches at build time), plus Laguna-tagged agent/tool transcripts (tool schema + call + result turns).
  - prose/reasoning: reuse the existing categories from the DS builder verbatim.
  - shell/C: reuse the existing Bash + C/Metal review categories, downweighted to 15%.
  - Both thinking and non-thinking assistant prefixes, matching the DS builder's split.

- [ ] **Step 2: Build and validate composition**

```bash
python3 gguf-tools/imatrix/dataset/build_laguna_xs21_imatrix_dataset.py
python3 -c "import json;m=json.load(open('gguf-tools/imatrix/dataset/laguna_xs21_manifest.json'));print(m)"
```

Expected: manifest reports category token shares within ±3 points of 55/30/15 and ~3M tokens total. Add the share computation to the manifest if the forked builder doesn't emit it.

- [ ] **Step 3: Commit** (script + manifest; the rendered .txt goes in only if the DS corpus tracks its rendered file — match existing .gitignore behavior).

```bash
git add gguf-tools/imatrix/dataset/
git commit -m "Add Python/web-biased Laguna XS 2.1 imatrix corpus builder"
```

### Task 10: Collect biased imatrix and build the RoutedQ3_K artifact (llama.cpp, out-of-tree)

**Files:**
- Create: `gguf-tools/imatrix/laguna-xs21-README.md` (procedure + exact commands used)
- Produces artifact: `gguf/laguna-xs-2.1-RoutedQ3_K-biased.gguf` (~13 GiB, not committed)

**Interfaces:**
- Consumes: Task 9 corpus; the official BF16 GGUF as quantize input (upstream ships no Q8_0 — user decision: quantize from BF16 directly).
- Produces: the biased GGUF; its exact per-tensor type map recorded in laguna-xs21-README.md for Task 11's validation entry.

- [ ] **Step 1: Set up llama.cpp** (outside the repo):

```bash
cd ~/src && git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_METAL=ON && cmake --build build -j8 --target llama-imatrix llama-quantize llama-cli
```

Verify Laguna support: `./build/bin/llama-cli -m ~/projects/ds4/gguf/Laguna-XS-2.1-Q4_K_M.gguf -p "hi" -n 8` produces tokens.

- [ ] **Step 2: Collect the imatrix against BF16 with the biased corpus**

```bash
./build/bin/llama-imatrix -m ~/projects/ds4/gguf/Laguna-XS-2.1-BF16.gguf \
  -f ~/projects/ds4/gguf-tools/imatrix/dataset/laguna_xs21_rendered_prompts.txt \
  -o ~/projects/ds4/gguf-tools/imatrix/laguna-xs21-biased.imatrix \
  -c 8192 --chunks 400
```

Expected: several hours on the M5 Max; output file ~tens of MB.

- [ ] **Step 3: Quantize — routed Q3_K, signal path Q8_0.** Use tensor-type overrides so ONLY `ffn_*_exps` go low (exact tensor names from xs2-facts.md item 4):

```bash
./build/bin/llama-quantize \
  --imatrix ~/projects/ds4/gguf-tools/imatrix/laguna-xs21-biased.imatrix \
  --tensor-type "ffn_gate_exps=q3_k" --tensor-type "ffn_up_exps=q3_k" \
  --tensor-type "ffn_down_exps=q3_k" \
  ~/projects/ds4/gguf/Laguna-XS-2.1-BF16.gguf \
  ~/projects/ds4/gguf/laguna-xs-2.1-RoutedQ3_K-biased.gguf q8_0
```

(Baseline type q8_0 keeps the signal path at Q8_0; the overrides pull routed experts to Q3_K. Verify flag syntax against `llama-quantize --help` — it has changed across versions; the invariant contract is: routed expert tensors Q3_K with imatrix, everything else Q8_0.)

Expected: file ≈13 GiB (±1.5). Record the actual per-tensor map:

```bash
./build/bin/llama-quantize --dry-run ... 2>/dev/null || python3 -c "
from gguf import GGUFReader
r=GGUFReader('$HOME/projects/ds4/gguf/laguna-xs-2.1-RoutedQ3_K-biased.gguf')
from collections import Counter
print(Counter(str(t.tensor_type) for t in r.tensors))"
```

- [ ] **Step 4: Write laguna-xs21-README.md** with the exact commands, llama.cpp commit hash, and the tensor map. Commit it.

```bash
git add gguf-tools/imatrix/laguna-xs21-README.md
git commit -m "Document biased XS 2.1 imatrix collection and Q3_K quantize procedure"
```

### Task 11: Accept the biased layout in ds4 + quality A/B

**Files:**
- Modify: `ds4.c` (`weights_validate_laguna_layout`: add `xs21-routedq3k-q8` recipe per Task 10's tensor map)
- Modify: `gguf-tools/quality-testing/data/laguna-xs21/README.md` (results table)

**Interfaces:**
- Consumes: Task 10 artifact + tensor map; Task 4 fixtures; Task 7 A/B gate.
- Produces: the go/no-go quality verdict for the biased artifact.

- [ ] **Step 1: Add the validation entry** (same structure as Task 3's entries; routed gate/up/down must be Q3_K, shared + attention + dense Q8_0, else die naming the offending tensor).

- [ ] **Step 2: Resident smoke + streaming A/B on the biased file**

```bash
make -j8 && ./ds4 -m gguf/laguna-xs-2.1-RoutedQ3_K-biased.gguf -c 4096 -p "def fizzbuzz(n):" -n 64 --nothink
./tests/xs21_stream_ab.sh gguf/laguna-xs-2.1-RoutedQ3_K-biased.gguf
```

Expected: sane output; `xs21 stream A/B: OK`.

- [ ] **Step 3: Quality A/B vs fixtures.** Run the Task 4 fixture comparison for (a) official Q4_K_M, (b) biased Q3_K. Record both, split by fixture category (web/Python vs general vs tool-call), in the results table. Decision gate: biased-Q3 must be ≥ official-Q4 on web/Python fixtures and not catastrophically below on general (>0.85× the Q4 agreement score — judgment call, record the numbers either way). If it fails, the fallback is routed Q4_K with the biased imatrix (rerun Task 10 step 3 with q4_k overrides) — smaller cache win, same quality logic.

- [ ] **Step 4: `make ds4_test && ./ds4_test` green; commit**

```bash
git add ds4.c gguf-tools/quality-testing/
git commit -m "Accept biased XS 2.1 RoutedQ3_K layout with quality A/B results"
```

---

## Phase P4 — Biased hotlist

### Task 12: Profile web/Python workloads and produce the hotlist

**Files:**
- Create: `gguf-tools/imatrix/laguna-xs21-hotlist-workload.md` (the scripted workload list)
- Produces: `gguf-tools/imatrix/laguna-xs21-biased.hotlist` (text, committed — it is small)

**Interfaces:**
- Consumes: resident biased artifact (fast profiling); built-in writer via `DS4_EXPERT_HOTLIST`.
- Produces: hotlist text file consumed by Task 13 (runtime env) and Task 14 (baked default).

- [ ] **Step 1: Script the workload.** ~10 ds4-agent sessions against real Python/web repos (e.g., a Flask app, a vite/TS frontend — local checkouts) with tasks like "add a route/test/component", plus ~20 raw completions over corpus web/Python slices. Record the exact list in laguna-xs21-hotlist-workload.md.

- [ ] **Step 2: Run with the profiler writer active**

```bash
DS4_EXPERT_HOTLIST=gguf-tools/imatrix/laguna-xs21-biased.hotlist \
  ./ds4-agent -m gguf/laguna-xs-2.1-RoutedQ3_K-biased.gguf -c 32768
```

(Repeat/append across sessions — check whether the writer appends or overwrites at ds4.c:1607: it overwrites on process exit, so either run all workloads in one long session per file and merge, or write per-session files and merge by summing hits per {layer,expert}; a 10-line Python merge is acceptable, include it in the workload doc.)

Expected file head: `# ds4 expert hotlist v1`, `# model Laguna XS 2.1`, sorted `layer expert hits weight` lines.

- [ ] **Step 3: Validate coverage.** Sanity: top-1000 entries should span most sparse layers, not clump in one layer (that would indicate a profiling bug). Quick check:

```bash
awk '!/^#/{print $1}' gguf-tools/imatrix/laguna-xs21-biased.hotlist | head -1000 | sort -u | wc -l
```

Expected: ≥ 30 distinct layers.

- [ ] **Step 4: Measure the effect** on the laptop with a constrained cache, warm-start comparison:

```bash
./ds4 -m gguf/laguna-xs-2.1-RoutedQ3_K-biased.gguf --ssd-streaming \
  --ssd-streaming-cache-experts 3000 -c 8192 -p "<web/python prompt>" -n 256 --nothink
DS4_METAL_STREAMING_EXPERT_HOTLIST=gguf-tools/imatrix/laguna-xs21-biased.hotlist \
  ./ds4 -m ... (same)
```

Expected: hotlist run shows higher cache hit rate in the streaming stats on first-prompt decode. Record both numbers.

- [ ] **Step 5: Commit**

```bash
git add gguf-tools/imatrix/laguna-xs21-biased.hotlist gguf-tools/imatrix/laguna-xs21-hotlist-workload.md
git commit -m "Profile and record Python/web-biased XS 2.1 expert hotlist"
```

### Task 13: Bake the hotlist as the XS 2.1 compiled-in default

**Files:**
- Create: `ds4_streaming_hotlist_laguna_xs21.inc`
- Create: `gguf-tools/imatrix/hotlist_to_inc.py`
- Modify: `ds4.c` (include at ~1360; variant case at ~20841)

**Interfaces:**
- Consumes: Task 12 hotlist text.
- Produces: `ds4_default_streaming_hotlist_laguna_xs2[][2]` + `_count`, selected automatically for XS2 when no env override.

- [ ] **Step 1: Write the converter**

```python
#!/usr/bin/env python3
"""hotlist text (layer expert hits weight) -> .inc uint16 pair array."""
import sys
name = sys.argv[2] if len(sys.argv) > 2 else "laguna_xs2"
rows = []
for line in open(sys.argv[1]):
    if line.startswith("#"): continue
    l, e, hits, w = line.split()
    if int(hits) == 0: continue
    rows.append((int(l), int(e)))
print("/* Generated from ds4 expert hotlist profiles; sorted by hits/weight. */")
print(f"/* Laguna XS 2.1 default streaming expert hotlist. */")
print(f"static const uint16_t ds4_default_streaming_hotlist_{name}[][2] = {{")
for l, e in rows[:8192]:
    print(f"    {{{l}, {e}}},")
print("};")
print(f"static const uint32_t ds4_default_streaming_hotlist_{name}_count = "
      f"sizeof(ds4_default_streaming_hotlist_{name}) / sizeof(ds4_default_streaming_hotlist_{name}[0]);")
```

(Cap 8192 entries; match the exact `_count` declaration style used in the existing `.inc` files — check `tail -3 ds4_streaming_hotlist_glm52.inc` and mirror it.)

- [ ] **Step 2: Generate, include, and select**

```bash
python3 gguf-tools/imatrix/hotlist_to_inc.py gguf-tools/imatrix/laguna-xs21-biased.hotlist > ds4_streaming_hotlist_laguna_xs21.inc
```

In `ds4.c`: add `#include "ds4_streaming_hotlist_laguna_xs21.inc"` at ~1361 and a variant case at ~20841:

```c
} else if (g_ds4_shape.variant == DS4_VARIANT_LAGUNA_XS2) {
    hotlist = ds4_default_streaming_hotlist_laguna_xs2;
    hotlist_count = ds4_default_streaming_hotlist_laguna_xs2_count;
```

- [ ] **Step 3: Verify selection** — streaming startup without the env var logs a nonzero preload; `make ds4_test && ./ds4_test` green.

- [ ] **Step 4: Commit**

```bash
git add ds4_streaming_hotlist_laguna_xs21.inc gguf-tools/imatrix/hotlist_to_inc.py ds4.c
git commit -m "Bake Python/web-biased hotlist as the XS 2.1 streaming default"
```

---

## Phase P5 — Final validation

### Task 14: Mini acceptance run and closing notes

**Files:**
- Modify: `docs/superpowers/plans/mini-notes.md`

**Interfaces:**
- Consumes: biased artifact + baked hotlist on the Mini.
- Produces: final acceptance record against the spec's criteria.

- [ ] **Step 1: Deploy** biased GGUF (scp) + rebuilt binaries to the Mini.

- [ ] **Step 2: Acceptance session**: `./ds4-agent -m gguf/laguna-xs-2.1-RoutedQ3_K-biased.gguf --ssd-streaming --ctx 32768`; run a scripted multi-tool web/Python task to completion.

- [ ] **Step 3: Record against spec acceptance**: warm-cache decode t/s (target ≥ ~10), cache hit rate vs the P2 baseline (Task 8 notes), memory headroom, any wired-limit adjustment. Also record the honest deltas: biased-Q3 vs official-Q4 quality table (from Task 11) and hotlist hit-rate gain (from Task 12).

- [ ] **Step 4: Final commit**

```bash
git add docs/superpowers/plans/mini-notes.md
git commit -m "Record XS 2.1 Mini acceptance results"
```

---

## Self-review notes (resolved during plan writing)

- Spec's "small new converter tool" for hotlists: superseded — the engine already writes hotlist text (`DS4_EXPERT_HOTLIST`) and loads it at runtime (`DS4_METAL_STREAMING_EXPERT_HOTLIST`); only the text→`.inc` bake (Task 13) is new, and it is ~20 lines.
- Spec's "imatrix collector compatibility with Laguna" work item: resolved by using llama.cpp `llama-imatrix`/`llama-quantize` out-of-tree (the repo has no Laguna quantizer; personal branch makes this acceptable). ds4-side work reduces to layout acceptance (Task 11).
- Spec unknowns #1–#4 are Task 1 outputs; every dependent constant in Tasks 2–3 is marked `FACTS:`.
- Task 6 deliberately defers exact call-site code to its Step-1 GLM study; its correctness is enforced by Task 7's A/B gate, which is the spec's load-bearing test.

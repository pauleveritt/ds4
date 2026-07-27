# Laguna XS 2.1 on ds4 — consolidated state and Phase 2 kickoff

**Status as of 2026-07-27.** Branch `laguna-xs21-footprint`, forked from
`laguna-xs2.1` at `05965c9`.

Phase 1 (streaming bring-up) is done and green: Laguna XS 2.1 loads, generates
correctly, and streams its routed experts through the Metal expert cache with a
passing streamed-vs-resident correctness gate. Phase 2 is a **footprint**
project, and late Phase 1 measurements reframed what that means — see §5, which
is the most important section in this document.

---

## 1. Where the research lives

| What | Where |
|---|---|
| Phase 1 branch (8 commits, all the code) | `laguna-xs2.1` |
| Phase 1 worktree | `.claude/worktrees/laguna-xs2.1` |
| **This** branch / worktree | `laguna-xs21-footprint` / `.claude/worktrees/laguna-xs21-footprint` |
| Original 14-task plan | `docs/superpowers/plans/2026-07-25-laguna-xs2-streaming.md` |
| Design spec | `docs/superpowers/specs/2026-07-25-laguna-xs2-streaming-design.md` |
| GGUF ground truth (verified against the file, not config.json) | `docs/superpowers/plans/xs2-facts.md` |
| 32 GB sizing measurements | `docs/superpowers/plans/mini-notes.md` |
| **Preserved SDD research** | `docs/superpowers/research/laguna-xs21-sdd/` |
| Quality fixtures + README | `gguf-tools/quality-testing/data/laguna-xs21/` |
| Python/web fixture prompts | `gguf-tools/quality-testing/prompts_laguna_xs21_webpy.jsonl` |
| A/B correctness gate | `tests/xs21_stream_ab.sh` |
| Models (shared symlink) | `gguf/Laguna-XS-2.1-Q4_K_M.gguf` (18.88 GiB), `gguf/Laguna-XS-2.1-BF16.gguf` (63.8 GiB) |

### Important: the SDD research was gitignored

Phase 1 ran under an SDD flow that wrote its ledger and per-task reports to
`.superpowers/sdd/`, which contains a `.gitignore` of `*` — so **none of it was
ever committed**, and it exists only as untracked files in the `laguna-xs2.1`
worktree. It would vanish if that worktree were removed.

It has been copied into `docs/superpowers/research/laguna-xs21-sdd/` on this
branch and is now tracked:

- `progress.md` — the canonical task-by-task ledger. Decisions, ground truth,
  every finding. **Read this before touching Phase 1 code.**
- `task-{2..7}-brief.md` — per-task briefs.
- `task-{2..6}-report.md` — per-task implementer reports with verification
  transcripts. `task-6-report.md` is the substantial one (22 KB).

The five `review-*.diff` files were **not** copied — they are raw diffs of
commits that still exist on `laguna-xs2.1`, so `git diff <range>` reconstructs
them. Every finding they produced is written up in the reports and ledger.

---

## 2. Model ground truth

Full detail in `xs2-facts.md`. The essentials:

- The model is **Laguna XS 2.1** (`poolside/Laguna-XS-2.1-GGUF`, revision
  `1a37c0a5fb8c7a18e6106decb6be6327d1b63fa6`), **not** the "Laguna XS.2" the
  original plan was drafted against — that repo has no GGUF release. Task 1
  corrected this; several constants differ.
- 40 layers, embd 2048, vocab 100352, 256 experts / 8 used, ff_exp 512,
  ff_shared 512, ff_dense 8192, swa 512, kv heads 8, head_dim 128.
- `head_count` array `[48,64,64,64]`×10 → n_head_global 48, n_head_swa 64
  (the `n_head` field, S21-style majority value, must be 64).
- `leading_dense 1` → 39 sparse layers → **9,984 routed experts**.
- rot 64 / rot_swa 128; yarn freq_base 500000, swa 10000, factor 32.0,
  orig_ctx 8192, beta_fast 64, beta_slow 1, attn_factor 1.0, rms_eps 1e-6.
- Tensor names are identical to ds4's existing `weights_bind_laguna_layer`.
- **No Q8_0 exists upstream.** BF16 (63.8 GiB) was downloaded as the quantize
  source instead. Every "Q8_0 file" reference in the original plan means the
  BF16 file. ds4 does **not** run BF16 resident — that is deliberately out of
  scope; BF16 is only a source for out-of-tree llama.cpp quantization.
- Q4_K_M has per-layer Q4_K/Q6_K variation on `attn_v`, `ffn_down_exps`, and
  `ffn_down_shexp` (a llama.cpp heuristic, uncorrelated with the SWA pattern).
  This detail turns out to matter enormously — see §5.

---

## 3. What Phase 1 built

Eight commits on `laguna-xs2.1`, `026e90a`..`05965c9` plus earlier:

| Task | Outcome |
|---|---|
| T1–T2 | Ground truth established; `DS4_VARIANT_LAGUNA_XS21` / `DS4_SHAPE_LAGUNA_XS21` in `ds4.h` |
| T3 | Layout validation + resident bring-up. **Found a real decode bug** |
| T4 | 120 quality fixtures (100 general, 20 web/Python/tool-call). **Found a sampling bug** |
| T5 | `--ssd-streaming` variant-gated to XS 2.1; S 2.1 still refuses |
| T6 | **Load-bearing.** MoE dispatch wired through the Metal expert cache. **Found a real kernel bug** |
| T7 | Streamed-vs-resident A/B gate — **passes** |
| T8 | `xs21-q4` download target + 32 GB sizing notes. Hardware acceptance run still open |

Three of these tasks found real bugs beyond their literal scope. Two shared a
signature worth internalizing: **this codebase fails plausibly, not loudly.**

- **T3**: `laguna_graph_forward_token`'s QKVG dispatch assumed any non-F16
  attention weight was Q8_0, silently corrupting XS 2.1's Q4_K attention
  tensors. Decode collapsed to `<|UNK|>` after token 1; prefill was unaffected.
- **T6**: the fused kernel's down-pipeline always selected the Q4_K
  address-table kernel when streaming was eligible, even though XS 2.1's file
  pins its slab class to Q6_K — decoding Q6_K blocks with a Q4_K layout. Same
  `<|UNK|>` collapse.

Because both produced coherent-looking wrong output rather than a crash,
"it generates something reasonable" is not evidence of correctness. That is
why `tests/xs21_stream_ab.sh` exists — run it after any engine change:

```bash
./tests/xs21_stream_ab.sh
```

It takes ~24 s (mmap is lazy, so its 8 model loads are nearly free), compares
resident vs streamed greedy output bytewise across 4 prompts, and pins the
cache to 800 experts to force continuous eviction. Last run: all 4 matched,
with 6,669 evictions at a 0.613 hit rate.

### Post-review fix to T6

Two independent reviewers examined T6. Both found that the report and commit
message described a `laguna_stream_decode_experts_addr_table_eligible` helper
**that never existed in the committed code**. One additionally found an
un-enforced invariant: host-side `weights_streaming_layer_experts_uniform`
pins the slab class by *byte size* and never inspects quant type, while the
Metal kernel only serves Q4_K/Q6_K down layers — so a layer could be dropped
from the resident span set yet be unservable from cache. Neither mapped nor
cached: T3's silent-corruption shape again.

A first fix attempt was **caught and rejected before it was built**. It added a
family-gated `exit(1)` premised on "validation guarantees down type is Q4_K or
Q6_K" — false. `ds4.c:5150-5163` accepts `down == layer_routed_type`, where
that type may be Q4_K, **Q3_K or Q2_K**. The guard also had no streaming
condition despite sitting in a path that runs for resident Laguna too. It
would have killed the existing `laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf` at
startup, and would reject Phase 2's own RoutedQ3_K artifact by construction.

The shipped fix (`3351a4e`) makes the gap structurally impossible instead:
`laguna_decode_experts_cache_servable` reports a non-Q4_K/Q6_K routed down
layer as not-servable, so it stays in the resident span set and takes the
mapped-model fallback. Correct on every model, no barrier to Q2_K/Q3_K work.

---

## 4. Measured performance and memory

All measured on the dev laptop (M5 Max, **128 GB**) at `05965c9`.

### Resident vs streamed, same prompt, back to back

| | prefill | generation |
|---|---:|---:|
| resident | 379.90 t/s | 84.59 t/s |
| streamed (800-expert cache, heavy eviction) | 197.93 t/s | 48.85 t/s |

≈42% decode and ≈48% prefill penalty — but under worst-case cache pressure, so
treat it as a floor rather than the expected steady state.

### Expert-cache sweep — ctx 16384, greedy, 256 tokens, Python prompt

| experts | hit rate | SSD read | footprint | gen t/s |
|---:|---:|---:|---:|---:|
| 800 | 0.593 | 30.00 GiB | 6.34 GiB | 46.4 |
| 1,600 | 0.769 | 17.04 GiB | 7.86 GiB | 48.7 |
| 3,200 | 0.902 | 7.20 GiB | 10.92 GiB | 52.1 |
| 4,800 | 0.907 | 6.87 GiB | 12.81 GiB | 51.9 |
| 6,400 | 0.907 | 6.87 GiB | 12.81 GiB | 52.3 |

Saturates at **~3,600 experts** — the 4,800 and 6,400 rows are byte-identical
(same 3,615 misses), so the whole Python working set is cached and further
budget is dead weight. Below 1,600, SSD traffic grows 4.4×. The residual
6.87 GiB is compulsory cold-start: 3,615 × 1.95 MiB, each expert fetched once.

Context is nearly free once saturated: at 4,000 experts, ctx 16384 = 12.44 GiB
and ctx 32768 = 13.07 GiB, at identical throughput. **Take the larger context.**

### These throughput numbers are optimistic

Everything above ran where the entire 18.88 GiB model sits in OS page cache.
The `pread` rate proves it — 14.19 GiB in 368 ms is **38.6 GiB/s**, RAM
bandwidth, not storage. On a memory-constrained machine those become real SSD
reads (~3–5 GB/s on Apple NVMe). **Memory and cache-budget figures should
transfer; t/s will not.** Do not quote 52 t/s as a small-machine expectation.

---

## 5. The reframe — read this before planning Phase 2

Every run prints this warning, and it is the central fact of the project:

```
SSD streaming mixed-precision model: 20/39 routed layers off the slab size
class will bypass the expert cache and read experts via mapped model views
WARNING: the majority of routed layers (20/39) are off the slab size class
```

**Only 19 of 39 sparse layers use the expert cache at all.** The other 20 read
through mmap and never appear in the hit/miss counters — so the 0.907 hit rate
above describes roughly half the model. (Sanity check: 63 decode steps × 19
layers × 8 experts = 9,576 = hits + misses exactly.)

### The tensor decomposition

`./ds4 -m gguf/Laguna-XS-2.1-Q4_K_M.gguf --inspect` reports 678 tensors,
33.44 B logical params, 18.88 GiB: `f32` 239 tensors / 0.08 GiB, `q4_k` 379 /
14.69 GiB, `q6_k` 60 / 4.11 GiB. Reconstructing:

| | size |
|---|---:|
| true non-routed (attn, embeddings, shared experts, dense layer, router) | **~1.2 GiB** |
| routed experts — 19 cache-eligible layers | ~9.3 GiB |
| routed experts — 20 bypass layers (Q6_K down) | ~8.4 GiB |

The 60 `q6_k` tensors are exactly 20 layers × {`attn_v`, `ffn_down_exps`,
`ffn_down_shexp`}, and 20 × 210 MiB = 4.11 GiB ✓.

### Why this changes the goal

The startup log's "non-routed weights: 9.64 GiB" is **not** what it sounds
like. It is computed as the decode static span set
(`weights_streaming_non_routed_bytes`, `ds4.c:6874`), which necessarily
includes the routed experts of the 20 bypass layers. True non-routed weight is
only **~1.2 GiB**.

So ~8.4 GiB of apparently mandatory-resident weight is not structural — it is
those 20 layers being off the slab class. **Uniformity is the whole ballgame,
not aggressiveness.** Getting all 39 routed layers onto one slab class is worth
~8.4 GiB; going Q4_K→Q2_K on top of that is worth maybe 1–2 GiB more.

### Projected footprint under uniformity

Assuming a uniform routed quant makes all 39 layers cache-eligible, at Q2_K
(~0.98 MiB/expert):

| component | ctx 8192 | ctx 16384 | ctx 32768 |
|---|---:|---:|---:|
| true non-routed | 1.2 | 1.2 | 1.2 |
| KV cache | 0.37 | 0.66 | 1.31 |
| graph scratch | 1.99 | 3.97 | 3.97 |
| expert cache (2,000) | 1.9 | 1.9 | 1.9 |
| **total** | **5.5 GiB** | **7.7 GiB** | **8.4 GiB** |

Uniform *Q4_K* at ctx 16384 lands ~9.1 GiB, so Q2_K may not even be needed to
get under 10 GiB. **These are projections, not measurements.**

### Two consequences for planning

1. **The expert cache is not the binding constraint.** It is the smallest term
   in every column. A perfect Python hotlist saves ~1 GiB of footprint — its
   real value is SSD traffic and throughput, not memory.
2. **Graph scratch becomes the second-largest item** — 3.97 GiB at prefill
   16384 (it scales with prefill width: 4096→0.99, 8192→1.99, 16384→3.97,
   where prefill caps). Laguna refuses `--prefill-chunk`, so there is no lever
   today. **This is not in the original plan and probably should be.**

---

## 6. Current sizing recommendation

Target was moved from the plan's 16 GB Mac Mini to **32 GB** (user decision,
2026-07-27) because 16 GB does not fit today: at ctx 32768 the decode static
span set (9.64 GiB) + KV (1.31) + scratch (3.97) = 14.92 GiB before a single
cached expert.

```bash
./ds4-agent -m gguf/Laguna-XS-2.1-Q4_K_M.gguf \
    --ssd-streaming --ssd-streaming-cache-experts 4000 -c 32768
```

≈13.1 GiB task footprint, ≈22.7 GiB including hot mmap pages, ≈9.3 GiB left for
OS and other apps. A 3,200-expert variant gives ~20.5 GiB total for a 0.907 →
0.902 hit-rate cost.

**16 GB becomes viable if Phase 2's uniformity work lands** — that is the §5
projection. It is a "support it" target, not a "design for it" target.

---

## 7. Open items inherited from Phase 1

**Minor findings deferred to a final whole-branch review:**

1. T1: missing inline citation for S 2.1 `n_head=72` in `xs2-facts.md`.
2. T2: the family-level streaming refusal message (`ds4.c:57652`) says
   "Laguna S 2.1" — correct behavior, stale wording. (The ledger records
   this as `ds4.c:57509`; line numbers shifted after `3351a4e`.)
3. T2: `test_laguna_variant_shapes` covers only static constants; the
   `block_count` variant-selection and die-on-unsupported paths are untested.
4. T3: the `attn_v` layout-validation check is slightly redundant with the
   `tensor_expect_layout` call right after it — harmless.
5. T6: no unit-test coverage for the streaming decode/prefill path; it rests
   entirely on `tests/xs21_stream_ab.sh`.

**Other known issues:**

- **The original plan text still says "Mini" and 16 GB** in Tasks 8 and 14.
  Read as 32 GB. Not yet edited.
- **`download_model.sh` relinks `ds4flash.gguf` on every target**, and that is
  `ds4_test`'s default model. Running `xs21-q4` silently repoints the suite
  away from the DeepSeek IQ2XXS model it expects. Restore with:
  ```sh
  ln -sf "$PWD/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf" ds4flash.gguf
  ```
  A guard or warning in the script would be worth doing.
- **T8's hardware acceptance run never happened.** Checklist in
  `mini-notes.md` §7.
- **A ~71–80% self-consistency floor** in the fixture scoring tool was
  investigated in T4 and confirmed pre-existing (it reproduces on untouched
  Laguna S 2.1), **not** an XS 2.1 regression. Leading unverified cause:
  `score_official.c` retokenizes printed continuation text rather than reusing
  sampled token IDs. Documented in the fixture README. Do not re-investigate
  from scratch.

---

## 8. Proposed Phase 2 kickoff plan

Intended for Superpowers. This is a **proposal to brainstorm against, not a
committed plan** — §5 changed the goal late, and the task order below reflects
that reframe rather than the original plan's ordering.

The original plan's Tasks 9–14 remain the substrate; find them at
`docs/superpowers/plans/2026-07-25-laguna-xs2-streaming.md:483` onward.

### Phase 2 goal

Reduce Laguna XS 2.1's streaming footprint far enough that 16 GB is usable and
32 GB is comfortable, without losing the correctness Phase 1 established.

### Proposed ordering

**P2.0 — Validate the uniformity premise first (new, blocking).**
Everything below assumes a uniform routed quant is achievable. That is
unproven: llama.cpp's per-layer boost heuristic is what created the 20/19
split. Before investing in corpus and imatrix work, confirm that quantizing
from BF16 with explicit per-tensor overrides produces an artifact where all 39
routed layers share one byte-size class. A single small experiment answers it.
**If this fails, the whole footprint thesis needs rethinking** — so do it
first, cheaply.

**P2.1 — Graph scratch investigation (new).**
3.97 GiB at prefill 16384 is the second-largest consumer and has no lever
today. Understand what scratch is used for, whether Laguna's `--prefill-chunk`
refusal is fundamental or incidental, and whether a smaller prefill window is
viable. Potentially the cheapest large win available, and it is entirely
absent from the original plan.

**P2.2 — Biased corpus builder** (original Task 9, `plan.md:483`).
Unblocked, no GPU, independently startable. 55% web/Python, 30%
prose/reasoning, 15% shell/C; zero Java/C#/Kotlin/PHP/Go/Rust. Fork
`build_ds4_imatrix_dataset.py`, swap in the Laguna chat/tool template.

**P2.3 — Collect imatrix, build the artifact** (original Task 10, `:515`).
Out-of-tree llama.cpp. Quantize **from BF16** (no Q8_0 exists). Carry P2.0's
uniformity overrides into this step.

**P2.4 — Accept the layout in ds4 + quality A/B** (original Task 11, `:575`).
Compare against T4's P0 baseline. Note the pre-existing self-consistency floor
(§7) — do not mistake it for a regression.

**P2.5 — Re-measure footprint.** Re-run the §4 sweep and `mini-notes.md`
numbers against the new artifact. This is where the §5 projections get tested.

**P2.6 — Python expert hotlist** (original Tasks 12–13, `:609`, `:658`).
Deliberately *after* re-measurement, because §5 shows it is a
throughput/SSD-traffic lever rather than a footprint one — its priority
depends on what P2.5 finds.

**P2.7 — Acceptance run** (original Task 14, `:721`, plus T8's open Step 2).
On real constrained hardware. `mini-notes.md` §7 is the checklist.

### Standing rules for Phase 2

- **Run `./tests/xs21_stream_ab.sh` after every engine change.** ~24 s. This
  codebase produces plausible wrong output; that gate is the only thing that
  reliably catches it.
- **Re-measure rather than trusting §4/§5 numbers** once the artifact changes.
  The projections in §5 are arithmetic, not measurement.
- **Any t/s figure measured on the 128 GB laptop is optimistic** for the
  reasons in §4.

### Operational lessons from Phase 1

- **Never run two model-loading jobs at once.** A full `./ds4_test` maps
  ~82 GiB (its default `ds4flash.gguf` is the DeepSeek model, not Laguna).
  Two concurrent runs drove the 128 GB host into swap and produced a real GPU
  OOM. `tests/xs21_stream_ab.sh` now refuses to start if another `ds4` is live;
  the same guard is worth applying elsewhere.
- **A dispatched implementer agent silently sub-dispatched its own agent** and
  returned a false "done, nothing changed" result immediately, without waiting.
  A second dispatch was launched before this was noticed, and the two collided
  in one worktree for ~55 minutes. Explicitly forbid the Agent tool in
  implementer prompts, and verify `git log`/`git status` before believing any
  completion claim.
- **Reports drift from code.** T6's report described a function that did not
  exist; two reviewers caught it independently. Verify claims against the diff.
- **Cheap analysis beats expensive iteration.** The rejected `exit(1)` guard
  and the entire §5 reframe both came from greps and one `--inspect` call, with
  no build or model run. Do that work first.

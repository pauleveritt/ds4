# Laguna XS 2.1 + SSD Streaming + Python/Web-Biased Quantization

**Date**: 2026-07-25
**Status**: Approved design, pre-implementation
**Branch**: personal (based on `laguna-s2.1`)

## Goal

Run `ds4-agent` usefully on a 16 GB Mac Mini at 32k context, backed by
Laguna XS 2.1 (33B total / ~3B active MoE) streamed from SSD, with both
quantization precision and expert-cache residency biased toward
HTML/CSS/JS/TS/Python workloads.

## Context and constraints

- **Build machine**: M5 Max MacBook, 128 GB. All bring-up, imatrix
  collection, quant generation, and profiling happen here. The Mini is
  validation/deployment only.
- **Deployment target**: 16 GB Mac Mini. Metal default wired budget is
  roughly 10.5–11 GiB (raisable via `sysctl iogpu.wired_limit_mb`).
- **Scope**: personal branch. No upstream QA gates, no CUDA/ROCm work,
  minimal docs.
- **Agent context**: 32k tokens.
- **Corpus bias**: "biased + ballast" — ~55% HTML/CSS/JS/TS/Python and
  agent tool-call transcripts, ~30% general prose/reasoning, ~15%
  shell/C. Java/C#/Kotlin/PHP/Go/Rust deliberately absent.

### Why XS 2.1 and why streaming (Recipe B)

XS 2.1 is the same `LagunaForCausalLM` architecture as the already-ported
S 2.1, shrunk. ds4 currently supports nothing that fits a 16 GB
machine, even streamed. XS 2.1's routed experts are ~30.5B of its 33B
params (38 sparse layers × 256 experts × 3 mats × 2048×512); the signal
path is ~2.5B. Fully-resident low-bit (Recipe A) lands at 11–13 GiB —
too tight. Streaming Q3/Q4 experts with a 5–6 GiB cache fits the
default wired budget with better quality.

### XS 2.1 vs S 2.1 shape

| Parameter | S 2.1 (in ds4) | XS 2.1 |
|---|---|---|
| Layers | 48 | 40 |
| Hidden size | 3072 | 2048 |
| Heads / KV / head_dim | 72 (48 global) / 8 / 128 | 64 (48 global) / 8 / 128 |
| Experts total/used/shared | 256 / 10 / 1 | 256 / 8 / 1 |
| Expert FFN width | 1024 | 512 |
| Dense FFN | 12288 | 8192 |
| Attention pattern | global every 4th layer, SWA 512 | same (10 global / 30 SWA) |
| Leading dense / sparse layers | — | 1 / 39 (9,984 routed experts) |
| Vocab | 100352 | 100352 |
| YaRN | scale 32, orig 8192, attn_factor 1.0 | scale 32, orig 8192, attn_factor 1.0 |
| Router | sigmoid gating, scale 2.5 | same |

All XS 2.1 values above were read directly from the official GGUF at P0 and
are recorded in `docs/superpowers/plans/xs2-facts.md`. An earlier draft of
this spec carried values taken from the older Laguna XS.2 model's
`config.json`; those were wrong for the model that actually ships and have
been replaced.

## Approach

**Approach A — working first, tuned second.** Get XS 2.1 running
end-to-end on official Poolside GGUFs before any custom quantization
exists; domain tuning lands as a second wave on proven infrastructure.
Each phase uses the previous as its correctness baseline: one variable
at a time. Rejected alternatives: full custom pipeline up front (stacks
three unknowns before anything runs on target) and resident-first
Recipe A detour (spends imatrix effort on a throwaway ~2 bpw recipe).

## Components

### C1 — XS 2.1 shape variant

- New `DS4_VARIANT_LAGUNA_XS21` and `DS4_SHAPE_LAGUNA_XS21` in `ds4.c`
  (alongside `DS4_SHAPE_LAGUNA_S21`, ds4.c:671): 40 layers, embd 2048,
  vocab 100352, heads 64 (48 on global layers) / 8 KV / 128 dim,
  experts 256/8/1, ff_exp 512, ff_shared 512, ff_dense 8192, swa 512,
  leading dense 1, rot 64 / rot_swa 128, YaRN freq_base 500000 /
  scale 32 / orig 8192 / beta_fast 64 / beta_slow 1 / attn_factor 1.0,
  swa freq_base 10000.
- Detection: arch string stays `laguna`; discriminate variants by
  `block_count` (40 vs 48).
- Generalize the hardcoded per-layer head expectation
  `(il%4)==0 ? 48 : 72` (ds4.c:6054) into shape fields
  `n_head_global` / `n_head_swa`, values per variant.
- Extend `weights_validate_laguna_layout` with XS 2.1 entries: official
  BF16 and Q4_K_M initially; the biased custom mix later.
- Metal kernels: dims flow from the shape struct. A tile-size perf pass
  for 512-wide experts is flagged and deferred; correctness first.

### C2 — Laguna streaming port

- Replace the family-level rejection at ds4.c:57521 with a variant
  gate: XS 2.1 may stream; S 2.1 keeps the explicit refusal (untested ≠
  supported).
- Route Laguna's Metal MoE dispatch through the existing shared
  streaming machinery (`g_stream_expert_cache_*` in `ds4_metal.m`),
  using the GLM 5.2 integration as the template.
- Cache entry = `{layer, expert}` = gate+up+down slices: ~1.8 MiB at
  Q4_K, ~1.4 MiB at Q3_K. ~9.7k routed experts total (38×256 or
  39×256, pending known-unknown #2).
- New `ds4_streaming_hotlist_laguna_xs2.inc`, empty/default until P4.
- Reuse the automatic cache-budget logic (80% working set minus
  non-routed weights, routed-prefill headroom) unchanged.

### C3 — Domain-biased quant pipeline

- Fork `gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py` into a
  Laguna-format renderer (chat template, interleaved thinking, tagged
  tool calls — as defined by the S 2.1 port). Composition per the bias
  mix above; size target ~3M tokens, matching the DS corpus scale.
- Collect the imatrix on the laptop against the official BF16 file.
- **Work item, not footnote**: the imatrix collector targets
  DeepSeek/GLM routed tensors; verify/extend it for Laguna tensor names
  and the Laguna inference graph.
- One custom artifact: `laguna-xs.2` with **routed Q3_K experts using
  the biased imatrix + Q8_0 signal path**, ≈13 GiB. Rationale: the bias
  buys back Q3's quality loss in-domain, and Q3 experts fit ~30% more
  experts per GiB of cache than Q4 — the bias pays out as hit rate too.
  No Q2 fallback artifact (YAGNI).

### C4 — Biased hotlist

- `--expert-profile` runs on the laptop (resident mode — faster), over:
  scripted `ds4-agent` sessions against real Python/web repositories,
  plus raw completion over web/Python corpus slices.
- Small new converter tool: profile output → sorted `{layer, expert}`
  `.inc` (the existing `.inc` files' generator is not in-repo).
- The biased hotlist ships as the XS 2.1 default on this branch.

### C5 — Mini deployment

- `download_model.sh` target for official XS 2.1 Q4_K_M; the biased
  artifact is copied manually (personal branch, no hosting).
- Run config: `ds4-agent -m <gguf> --ssd-streaming --ctx 32768`;
  document optional `sudo sysctl iogpu.wired_limit_mb` bump.
- Memory plan at 32k: signal path ~2.5 GiB + KV ~1.3 GiB (40 KiB/token
  × 32k, 10 global layers; SWA layers cap at 512 tokens) + graph
  scratch ~1 GiB + expert cache 5–6 GiB ≈ 10–11 GiB.

## Phases

```
P0  download official GGUFs → ds4 --inspect → resolve unknowns →
    shape variant → resident bring-up on laptop (Q4_K_M) → fixtures
P1  streaming port → laptop A/B: streamed logits match resident
    (artificially capped cache as stress test)
P2  Mini baseline: official Q4_K_M streamed, no hotlist
    ← first usable milestone
P3  corpus fork → imatrix @ Q8 → biased RoutedQ3_K artifact →
    quality A/B vs P2 artifact
P4  profile runs → hotlist .inc → rebuild
P5  Mini final: biased quant + hotlist → acceptance run → notes
```

## Known unknowns

Resolved at P0 (all recorded in `docs/superpowers/plans/xs2-facts.md`):

1. ~~Per-layer head-count array~~ — RESOLVED: `[48,64,64,64]`×10, i.e. it
   does mirror S 2.1's reduced-heads-on-global-layers pattern (48 global /
   64 SWA), not flat.
2. ~~Leading-dense count~~ — RESOLVED: 1, so 39 sparse layers and 9,984
   routed experts.
3. ~~`rot_swa`~~ — RESOLVED: rot 64 (global), rot_swa 128 (SWA).
4. ~~Tensor names/shapes~~ — RESOLVED: identical to ds4's existing
   `weights_bind_laguna_layer`; no tensor-schema change needed. Q4_K_M
   carries per-layer Q4_K/Q6_K variation on `attn_v`, `ffn_down_exps`,
   and `ffn_down_shexp` that layout validation must tolerate.

Still open:

5. Whether OpenRouter (or Poolside API) serves XS 2.1 for fixture
   collection; fallback reference is the laptop BF16 file.
6. Imatrix collector compatibility with Laguna — resolved in approach by
   using llama.cpp out-of-tree, but unverified against this model.

## Error handling

Die-loudly, matching repo philosophy:

- Unknown XS 2.1 quant layout markers → refuse to load with the marker
  named.
- Streaming on non-Metal backends → refuse.
- S 2.1 + `--ssd-streaming` → keeps its current explicit refusal.
- Cache budget below one full layer's active expert set → exit with an
  explicit `--ssd-streaming-cache-experts` hint.
- Existing automatic-budget capping reused, not reinvented.

## Testing

- **Unit** (`tests/ds4_test.c`, mirroring the S 2.1 port's coverage):
  variant detection, head tables, layout validation.
- **Correctness (the load-bearing test)**: streamed vs resident logit
  comparison on the laptop, same prompt/context, cache artificially
  capped to force evictions. Must match within the tolerance defined
  by the CONTRIBUTING.md correctness-regression procedure.
- **Quality**: new web/Python fixture set plus retained general
  fixtures; report biased-Q3 vs official-Q4 on both domains, so the
  cost/benefit of the bias is measured, not assumed.
- **Performance**: `speed-bench` prefill/decode plus the existing
  cache hit/miss/eviction counters, laptop and Mini.
- **Acceptance**: sustained interactive `ds4-agent` session at 32k on
  the Mini; warm-cache decode ≥ ~10 t/s; session completes a scripted
  multi-tool task.

## Out of scope

CUDA/ROCm/distributed/TP for XS 2.1, S 2.1 streaming validation, FP8 KV
cache, expert pruning/trimming, DFlash/draft models, upstream QA
gates, MoE kernel tile tuning (flagged, deferred).

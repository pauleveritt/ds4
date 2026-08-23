# Mellum footprint: where the weight actually is, and the next target

Re-orientation note. The selective Q4_K artifact landed at 9.33 GiB and the
16 GB question stayed open. This note decomposes that 9.33 GiB by tensor
category, answers whether Laguna XS 2.1's two memory wins transfer to Mellum
(they do not), and names the next target with its arithmetic shown.

Read `MELLUM.md` for current state and
[`2026-08-21-16gb-and-the-prefill-gap.md`](2026-08-21-16gb-and-the-prefill-gap.md)
for the official artifact's tensor inventory and its *measured* llama.cpp
footprint — both are assumed here, not restated.

**Status of the numbers.** The decomposition below is **computed** from GGUF
block geometry, not measured. Three independent checks are why it is trusted:
it reproduces the published 12.03 / 9.33 GiB totals and the 10.98 / 1.05 GiB
expert-vs-rest split to three digits; the selective figure matches the *measured*
`mapped 9554.64 MiB` (= 9.331 GiB) from a real load; and applying the same model
to the official artifact's verified tensor inventory reproduces its known
**7.52 GiB** (gate/up Q4_K 3.876 + down 14×Q8_0/14×Q5_0 3.014 + attn Q4_K 0.311
+ embd Q4_K 0.119 + head Q6_K 0.173 + norms 0.02 = 7.513) — a file this model
was not fitted to. Every *projected* row, though, is arithmetic on a build that
does not exist. Nothing here has been run.

**Reviewed 2026-08-23** by an adversarial pass that corrected four things: the
prefill-scratch figure, the "de-risked" claim about Q5_0, a missing third quant
gate, and a calibration sentence that leaned on evidence its own source
disavows. All four corrections are folded in below rather than appended.

## Where the weight is

Each expert holds three same-shaped matrices — gate (2304→896), up (2304→896),
down (896→2304) — all 2,064,384 elements. Across 64 experts × 28 layers each
projection type totals 3.66 GiB at Q8_0.

| Category | Q8_0 (12.03) | selective (9.33) |
| --- | ---: | ---: |
| Expert gate | 3.66 | 2.30 |
| Expert up | 3.66 | 2.30 |
| **Expert down** (Q8_0 throughout) | **3.66** | **3.66** |
| Attention q/k/v/o | 0.59 | 0.59 |
| Token embedding | 0.22 | 0.22 |
| Output head | 0.22 | 0.22 |
| Router + norms (F32) | 0.02 | 0.02 |

**Down is now 39% of the artifact and is the only expert projection still
untouched.** The selective build compressed gate and up on layers 0–21 and
nothing else, so every further gate/up win scrapes at matrices already halved.

### Why down is stuck, precisely

K-quants (Q4_K, Q3_K, Q6_K) use 256-element superblocks, so the contiguous
dimension must divide by 256. Gate and up present 2304 (÷256 = 9, fine). Down
presents **896**, and 896 ÷ 256 = 3.5. A K-quant block cannot tile the row.
Q8_0 works only because its block is 32 and 896 ÷ 32 = 28.

This is matrix geometry, not a quality tradeoff anyone selected. **Q6_K is
therefore not a candidate for down** — same 256 wall — and on gate/up it is
0.8203 B/elem against Q4_K's 0.5625, i.e. strictly larger than what already
ships. Q6_K is the wrong lever in both directions.

The way out is a 32-element-block format, which divides 896 cleanly:

| Format | B/elem | Divides 896 | ds4 kernel for Mellum |
| --- | ---: | --- | --- |
| Q8_0 | 1.0625 | yes (32) | **shipped** |
| Q4_K | 0.5625 | no (256) | **shipped**, gate/up only |
| Q6_K | 0.8203 | no (256) | none; larger than Q4_K anyway |
| **Q5_0** | **0.6875** | **yes** | **none** |
| Q5_1 | 0.75 | yes | none |
| Q4_1 | 0.625 | yes | none |
| Q4_0 | 0.5625 | yes | none |
| **MXFP4** | **0.5313** | **yes** | **exists for another family** |

**No Q5_0 or Q4_0 code exists in `metal/moe.metal` at all** (verified: zero
matches). So whichever is chosen, this is kernel authoring, not kernel reuse.
Note also that the Q8 down surface is four variants — `down_f32`,
`down_batch_f32`, `down_grouped4_f32`, `down_grouped8_f32`
(`metal/moe.metal:2984, 3038, 3211, 3230`) — so "a kernel" undercounts the
work even for a decode-first slice.

**MXFP4 deserves a look before Q5_0 is assumed.** It is 17 B/32 = 0.5313
B/elem, divides 896, and is the one sub-Q8 32-block format ds4 *already
executes on routed experts* for another family — CPU dot at `ds4.c:3854`,
Metal dequant and kernels at `metal/moe.metal:6, 157, 4529`. It is both
smaller than Q5_0 and partially built. The open question is quality:
requantizing a non-QAT checkpoint into MXFP4 is a real risk that Q5_0 does not
carry, and `ds4.c:842` describes the existing use as "MXFP4 routed experts
**preserved from native checkpoints**" — i.e. not requantized. That caveat is
why it is a candidate to evaluate, not a default.

### What the official artifact's 14/14 split does and does not tell us

**The official artifact's down is 14 layers Q8_0 + 14 layers Q5_0** (inventory
verified in the prefill-gap note). An earlier draft of this note read that as
quality evidence — "down is sensitive and JetBrains did not take it to 4-bit."
**That reading is wrong, and it inverts the conclusion.**

Q4_K_M's heuristic wants `ffn_down` at Q6_K on roughly half the layers and
Q4_K on the rest. Both are 256-block and both are unrepresentable at 896, so
llama.cpp's incompatibility fallback maps Q6_K→Q8_0 and Q4_K→Q5_0 — which
produces exactly the observed 14/14. The split is a **mechanical consequence
of the same 896 geometry**, not a judgment about 5-bit. "They did not use
4-bit" is likewise forced, not chosen. *(Inferred from the fallback behaviour
plus the split's fingerprint; llama.cpp source was not read.)*

The consequence for targeting is direct, and it is the opposite of the earlier
draft's: the 14 layers that got Q5_0 are the ones the heuristic judged **less**
sensitive, and the sensitive half was pushed *up* to Q8_0. So the official
artifact evidences 5-bit down **only on the less-sensitive half**. The
"official pattern" row below (7.94 GiB weights) is the configuration with
external support; **uniform Q5_0 across all 28 (7.29 GiB) has none** and is a
genuine quality bet.

**The placement survives either reading of the split**, which is why the target
does not rest on the fallback inference. Even if the formats are mechanical,
the heuristic's *per-layer ranking* is quality-informed — it chose which layers
deserved the higher format — so copying the official placement copies a real
sensitivity ranking regardless. Stated precisely, the unevidenced claim is not
"the split is arbitrary" but the narrower **"5-bit is safe on the other 14
layers too."** That is what an A/B has to settle, and it is why the split, not
uniformity, is the shipping default. Unlike Laguna, nothing structural forces
uniformity here — the slab class does not apply (below) — so the choice is free
and belongs to measurement.

## The Laguna XS 2.1 wins do not transfer

Both were checked against the code rather than assumed.

**The prefill-chunk scratch win (3.97 → 0.99 GiB) has no Mellum analogue.**
Mellum's per-chunk-token cost has two components, and they should be shown
rather than asserted:

| Component | Site | B/token |
| --- | --- | ---: |
| `ds4_mellum_prefill_scratch` (19 tensors) | `ds4.c:37223` | 132,676 |
| MoE bucketing `s_pairs` + `s_partial` | `ds4_metal_mellum.m:1199, 1269` | 73,984 |

The first alone is ~130 KiB/token ≈ 133 MB at the 1024 cap, matching the
code's own "~120 MB allocation" comment (`ds4.c:62995`). The second matches
its own comment, "about 75 MiB at a 1,024-token chunk"
(`ds4_metal_mellum.m:1271`). **Whether the second belongs in the same budget
is unresolved** — it is a function-static in the MoE path rather than part of
the prefill workspace — so the honest figure is **~0.13 GB, or ~0.20 GB if
the bucketing buffers are counted**. Against Laguna's 3.97 GiB that is 20–30×
smaller either way, so the conclusion does not depend on settling it: there is
nothing to recover. Two further corrections to statements made while
investigating: the ~6.1 GB figure associated with Laguna is a *total-resident
projection*, not scratch; and
Mellum's 1024 chunk cap is **structural, not a stale guard** — the sliding
layers' KV ring is allocated at exactly `DS4_N_SWA`, so a wider chunk would wrap
and clobber keys still in use. Laguna's was an incidental family-level refusal;
Mellum's is ring geometry. Opposite situations.

**The slab-class win (~8.4 GiB) does not exist here.** The entire
slab-class / bypass / cache-eligible machinery is SSD-streaming-only, and Mellum
refuses `--ssd-streaming` outright. No expert cache, no slab, no bypass layers.
This is unrelated to Mellum having no dense prefix, which was the first guess
and was wrong.

**What does transfer is the lesson, through a different mechanism.**
`mellum_batched_prefill_unsupported` is a single whole-model boolean: one Q4_K
layer disables layer-major prefill for all 28, including layers 22–27, which are
pure Q8_0 and whose kernels would run. That is Laguna's uniformity penalty in
miniature — non-uniformity tripping an all-or-nothing gate that costs far more
than the non-uniform part — and it is the mechanical cause of the 0.21× prefill
regression. Scoping it per layer would recover the 21% of the stack already
eligible: **~1.17–1.20×** on prefill by arithmetic on the published path ratios,
i.e. 0.21× → roughly 0.25×. Real and cheap *if* the two paths can be mixed
within one prefill pass — **unverified**, and if they cannot, this is a
prefill-loop restructure rather than a scoping change.

## Two defects found while measuring

**KV is under-reported, and the error grows with context.** Mellum has no
branch in `ds4_context_memory_estimate_with_prefill_mode` and silently inherits
the DeepSeek-4 formula, which drops the `n_head_kv` and K+V factors and uses the
wrong per-layer caps. The reported figure is **ctx-invariant** — generic
`raw_cap = min(align256(1024 + prefill_cap), 8192) = 5120`, so
28 × 5120 × 512 B = 0.068 GiB, printed as `0.07` at every context. Actual is
0.26 at ctx 16384 and 0.48 at 32768, so the error is **3.8× at 16k and ~7× at
32k**, not a flat 7×. Independently corroborated —
`2026-08-01-mellum2-feasibility.md:338` already states "about 0.49 GiB" at
32,768.

The `compressed_kv_rows=3` in the startup line is the same degenerate formula
showing through. Scratch *is* included in the printed total; it prints 0.00
because the same missing branch leaves it at ~99 KB — which is nearly right
today only because the selective artifact never takes the batch path, and stops
being right the moment piece (3) lands.

**The `--prefill-chunk` CLI flag is accepted and then ignored** — stored in
`e->prefill_chunk`, which reaches only the estimator call (`ds4.c:37732`), so it
moves the *printed* `prefill_cap` and nothing else. GLM and non-XS Laguna refuse
it outright. Note the distinction: **the `DS4_MELLUM_PREFILL_CHUNK` env var does
work** — the prefill loop sizes its workspace from
`ds4_mellum_probe_chunk(DS4_N_SWA, DS4_N_SWA)` honoring the env knob
(`ds4.c:63019`), clamped at 1024. It is only the CLI flag that is inert, which
is the worse failure: it silently alters the diagnostics and nothing else.

## Context is nearly free

Only 7 of 28 layers scale with context; the other 21 are sliding-window, pinned
at 1024 rows. So KV grows slowly:

| ctx | KV |
| ---: | ---: |
| 16,384 | 0.26 |
| 32,768 | 0.48 |
| **40,960** | **0.59** |

**Weights are the whole game.** A context target of 40k costs 0.59 GiB, and
doubling it again would cost about 0.5 more.

## The target

At 40k context, adding KV 0.59 to each weight configuration:

| Configuration | weights | +KV@40k | total |
| --- | ---: | ---: | ---: |
| Q8_0 | 12.03 | 0.59 | 12.62 |
| selective, today | 9.33 | 0.59 | **9.92** |
| + gate/up Q4_K on all 28 layers | 8.59 | 0.59 | 9.18 |
| + down Q5_0 on 14 (official pattern) | 7.94 | 0.59 | 8.53 |
| **+ down Q5_0 on all 28** | **7.29** | **0.59** | **7.88** |
| (down Q4_0 on all 28 instead) | 6.86 | 0.59 | 7.45 |

Add ~0.2 GiB scratch to any row only once batch prefill actually runs; on
today's tokenwise path it is ~0.

**Two targets, carrying different evidence:**

- **8.53 GiB at 40k** (down Q5_0 on 14 layers, the official pattern) — the
  configuration a shipped artifact supports.
- **7.88 GiB at 40k** (down Q5_0 on all 28) — 0.65 GiB better and
  **unevidenced**, per the 14/14 analysis above. A quality bet, not a free win.

Either way the move that matters is 9.92 → ~8: the former sits at the edge of
what a 16 GB machine will concede, the latter has headroom.

For calibration, the prefill-gap note measured the official artifact at
**8.24–10.18 GB peak RSS** in llama.cpp across contexts to 131k. Cite that band
only — the same note **disavows its own pressure test**: the zero-page ballast
compressed away, so "no swap at 10.9 GB free" is evidence the ballast deflated,
not that the model survives pressure. It also observes that a real 16 GB Mac's
~10–11 GB Metal wired limit makes 10.18 GB marginal by default. **The 16 GiB
question stays open**; this target improves the odds, it does not settle them.

**The blocking work is a Q5_0 (or MXFP4) expert-down path for Mellum**: the
kernels, plus admission in **three** quant gates, not two — see below. Worth
~0.65–1.3 GiB depending on which target, the largest single win available.

### Three gates, not two

The plan's "two Q8_0 gates, not one" trap recurs here with a third member, and
this one is a **silent-corruption** risk rather than a refusal.
`ds4_engine_bind_mellum_decode_contract` sets
`mellum_batched_prefill_unsupported` by testing **only**
`src->ffn_gate_exps->type == DS4_TENSOR_Q4_K` (`ds4.c:36895`) — it never
inspects down. All four batch/decode down kernels are Q8_0-only
(`metal/moe.metal:2984, 3038, 3211, 3230`).

So the natural A/B artifact for isolating down — **Q8_0 gate/up + Q5_0 down** —
leaves that boolean *false* once validation and the desc builder admit Q5_0.
A long sync would then take the batch path and run
`kernel_mellum_q8_0_down_batch_f32` over Q5_0 bytes: wrong output, no error.
The eligibility test must learn about down *before* any such artifact is
built. **Verified against the code**, and it would have fired during exactly
the quality A/B this note proposes.

## What this does not touch

Attention, embeddings and the output head total ~1.03 GiB and are hard-wired
Q8-only above the MoE seam in ds4's Mellum graph. The official artifact carries
them as Q4_K and Q6_K, so matching it there is worth only ~0.43 GiB for a much
wider kernel surface. Not a priority against down's 1.3.

Nothing here bears on the paused agent-competence work; this is footprint, and
the two tracks are independent.

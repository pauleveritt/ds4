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
block geometry, not measured; it reproduces the published 12.03 / 9.33 GiB
totals and the 10.98 / 1.05 GiB expert-vs-rest split to three digits, which is
why it is trusted. Every projected row is arithmetic on a build that does not
exist. Nothing here has been run.

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
| Q8_0 | 1.0625 | yes | **shipped** |
| Q4_K | 0.5625 | no (256) | **shipped**, gate/up only |
| Q6_K | 0.8203 | no (256) | none; larger than Q4_K anyway |
| **Q5_0** | **0.6875** | **yes** | **none — the gap** |
| Q4_0 | 0.5625 | yes | none |

**The official artifact's down is 14 layers Q8_0 + 14 layers Q5_0**, not
uniform Q5_0 (inventory verified in the prefill-gap note). That split is
llama.cpp's per-layer boost heuristic, and it is evidence about quality: down
is the sensitive projection, and JetBrains did not take it to 4-bit. Uniform
Q5_0 across all 28 is the more aggressive option and is **untested for
quality** — unlike Laguna, no structural pressure forces uniformity here,
because the slab class does not apply (below). It is a free choice, so it
should be made on measured quality rather than on tidiness.

## The Laguna XS 2.1 wins do not transfer

Both were checked against the code rather than assumed.

**The prefill-chunk scratch win (3.97 → 0.99 GiB) has no Mellum analogue.**
Mellum's prefill workspace is ~202 KiB/token — ~0.2 GB at its chunk, against
Laguna's 3.97 GiB, roughly 20× smaller. There is nothing to recover. Two
corrections to statements made while investigating: the ~6.1 GB figure
associated with Laguna is a *total-resident projection*, not scratch; and
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

**KV is under-reported ~7×.** Mellum has no branch in
`ds4_context_memory_estimate_with_prefill_mode` and silently inherits the
DeepSeek-4 formula, which drops the `n_head_kv` and K+V factors and uses the
wrong per-layer caps. Reported 0.07 GiB; actual 0.26 at ctx 16384, 0.48 at
32768. Independently corroborated — `2026-08-01-mellum2-feasibility.md:338`
already states "about 0.49 GiB" at 32,768. The `compressed_kv_rows=3` in the
startup line is the same degenerate formula showing through. Scratch is
*included* in the printed total; it prints 0.00 because the same missing branch
leaves it at ~99 KB, which happens to be nearly right today only because the
selective artifact never takes the batch path.

**`--prefill-chunk` is accepted and then ignored.** GLM and non-XS Laguna
refuse it; Mellum stores it and never consults it, so it moves the *printed*
`prefill_cap` and nothing else. A flag that silently alters only the
diagnostics is worse than a refusal.

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

**Proposed target: ~7.3 GiB of weights, ~7.9 GiB resident at 40k.** The
difference that matters is 9.92 → 7.88: the former sits at the edge of what a
16 GB machine will concede, the latter has real headroom. For calibration, the
prefill-gap note *measured* the official artifact at 8.24–10.18 GB peak RSS in
llama.cpp across contexts to 131k, with no swap at 10.9 GB free — so this target
lands in the class of something already demonstrated on the hardware.

**The blocking work is a Q5_0 expert-down path for Mellum**: the kernel, plus
admission in both places that gate quant type — validation *and* the
decode-contract desc builder, per the "two Q8_0 gates, not one" trap already
recorded in the plan. Worth ~1.3 GiB, the largest single win available, and
de-risked by an existing artifact demonstrating the quality holds at 5-bit.

## What this does not touch

Attention, embeddings and the output head total ~1.03 GiB and are hard-wired
Q8-only above the MoE seam in ds4's Mellum graph. The official artifact carries
them as Q4_K and Q6_K, so matching it there is worth only ~0.43 GiB for a much
wider kernel surface. Not a priority against down's 1.3.

Nothing here bears on the paused agent-competence work; this is footprint, and
the two tracks are independent.

# P2.5 — Laguna XS 2.1 Q3 streamed-cache blocker

**Status:** blocked, 2026-07-28.  The uniform RoutedQ3_K artifact is valid,
but ds4 currently uses a mapped-model fallback for its routed experts during
decode.  It has not demonstrated the footprint reduction that Phase 2 needs.

## Research question

Can a uniform Q3_K routed-expert artifact use the Metal expert cache during
Laguna XS 2.1 streaming decode, and therefore replace the mapped routed-expert
static span with a bounded cache?

This is the gate between the successful P2.0--P2.4 artifact work and any
footprint claim.  Startup reservations alone do not answer it.

## Inputs and completed work

- P2.0 established that explicit llama.cpp overrides can produce a fully
  uniform Q3_K routed artifact.
- P2.1 made `--prefill-chunk 4096` usable for XS 2.1, reducing requested
  graph-tensor payload at context 16384 from 4069.45 MiB to 1017.66 MiB.
- P2.2--P2.3 produced the biased-imatrix artifact:
  `gguf/laguna-xs-2.1-RoutedQ3_K-biased.gguf` (SHA-256
  `b1dc95e586c2fc586a032101cbb3d1b3884e9d70905824b21e07afb821bf1bc8`).
- P2.4 passed resident smoke and undersized-cache streamed-vs-resident A/B,
  but that A/B exercises the correct mapped fallback, not Q3 cache service.

The artifact retains Q8 shared experts and has uniform Q3 routed gate/up/down
tensors.  Its format is not the problem.

## Measurement result

All runs used context 16384, `--prefill-chunk 4096`, and a 256-token greedy
generation.  The figures below distinguish ds4's planned startup total from
the live streaming-expert allocation reported after graph free:

| cache budget (MiB) | planned startup total (GiB) | live streaming experts (GiB) | generation |
|---:|---:|---:|---:|
| 800 | 2.89 | 0.00 | 61.90 t/s |
| 1,600 | 3.89 | 0.00 | 61.88 t/s |
| 3,200 | 5.91 | 0.00 | 62.11 t/s |
| 4,800 | 7.92 | 0.00 | 61.97 t/s |

No run reported cache entries, hits, or a live Q3 expert allocation.  The
stable generation rate across budgets is consistent with a mapped fallback,
not a cache whose capacity is changing.  Therefore the startup totals are
reservations, **not measured streaming footprints**.

## Source-level diagnosis

The zero-live result is intentional current behavior, not a counter bug:

1. The XS 2.1 Q8 signal layout dispatches routed Q3 decode through
   `ds4_gpu_glm_routed_moe_one_tensor` (`ds4.c:47891-47947`).
2. That generic path recognizes Q3 only for resident pair/down pipelines
   (`ds4_metal.m:34040-34070`).
3. Its address-table cache path admits Q2_K and Q4_K only
   (`ds4_metal.m:33767-33788`).
4. Independently, `laguna_decode_experts_cache_servable()` admits Q4_K/Q6_K
   only (`ds4.c:6664-6668`).  Q3 routed tensors consequently remain in the
   decode static spans (`ds4.c:6692-6705`).

The fallback is correct: it preserves output correctness by reading the Q3
weights through mapped model views.  It cannot meet the footprint goal,
because those routed tensors still contribute to the static mapped span.

## Failed implementation experiment — strengthened localization

An experimental Q3 address-table pair/down implementation was tried against
the real biased-Q3 artifact and then fully removed.  The cache path was
deliberately kept outside normal Q3 cache admission, so this was diagnostic
only and never a footprint result.

A streamed run allocated 1.01 GiB of live cache and showed 38,612 hits and
1,012 misses, but its continuation diverged immediately after `To solve the
F...` and subsequently emitted `|UNK|`.  The following readback and byte
checks localize that failure more tightly:

| Check | Result | Consequence |
|---|---|---|
| Q3 gate/up intermediate (`mid`) at decode layer 1 | Exact resident/cache match for all 16 KiB (8 x 512 floats) | Q3 pair arithmetic, selected ids, cache gate/up bytes, and its address-table consumption are not the observed failure. |
| Q3 down output immediately following that `mid` | First differing byte: 16,385 | Divergence begins at the first down-output float. |
| Cached gate/up/down bytes | Exact `memcmp` against each mapped GGUF tensor slice | Not a pread/copy, expert-stride, or slab-content failure. |
| Address-table entries | Exact cached MTL buffer GPU address plus inner offset | Not CPU-side address-table construction. |
| Q3 down address entries replaced with the mapped model-view addresses | Same divergence | Not specific to cached slabs or cache-buffer lifetime. |
| Direct slot-bound cached-down buffers, bypassing raw address-table lookup | Same down divergence | A simple raw-address indirection replacement is not a fix. |

The resident down result has magnitudes around `1e4`; the candidate down
result was around `1e-2`.  The live cache allocation and hit count therefore
prove only that loading and accounting ran, not that its Q3 down output was
valid.

This is now a Q3 down kernel/invocation-contract blocker.  It is not evidence
against the artifact, its cache contents, pair path, or cache accounting.
The removed code must not be restored as a starting point; an isolated down
equivalence test is required before another production-path attempt.

### First isolated control

The committed harness now covers both one expert and eight selected experts at
the production 2048 output / 512 mid dimensions. It uses nonzero deterministic
Q3_K data, the resident 64-thread geometry, distinct direct MTL buffers, and a
permuted selected-id order. Both direct-buffer candidates and a raw-GPU-address
candidate are bit-exact to the resident Q3 down output. This rules out basic
Q3 direct-buffer binding, slot order, selected-id permutation, down geometry,
and raw-address semantics. It does **not** overturn the real-artifact
eight-slot failure; that discrepancy is now an engine-integration or prior
experimental-wiring problem rather than a Q3 down-kernel problem.

## Independent review

Sol independently reviewed commits `538d2ab..2f26d91` and found no blocking
correctness issue in the diagnosis or its current documentation.  The review
confirmed the source-level eligibility chain above and that the current A/B
wording accurately describes mapped-fallback validation.

The review added three safeguards:

1. A working fix requires both numerically correct Q3 address-table kernels
   in the generic MoE path **and** coherent Q3 admission in
   `laguna_decode_experts_cache_servable()`.  Kernel eligibility alone can
   create hits while the Q3 routed tensors remain mapped, so it would not
   prove a footprint reduction.
2. The regression gate must assert nonzero cache entries/hits and live cache
   bytes, plus the expected decode-static-span drop.  Bytewise continuation
   equality alone is insufficient.
3. Preserve raw P2.5 command output and compact memory/A-B transcripts before
   relying on the exact live-memory, throughput, and experimental-hit figures
   above in future comparison work.

The quality result remains a local relative gate only: the current fixture
table supports lower teacher-forced average NLL versus the local Q4 fixtures,
not a general claim that Q3 quality improved.  First-token agreement and LCP
are lower, and the scorer has a documented tokenization/path floor.

## Required next implementation sequence

1. Build an artifact-backed single-layer harness using the real Q3 tensor
   slices, selected ids, and `mid` input captured from a decode step. Compare
   resident, direct-buffer, and raw-address down dispatches before running the
   full routed-MoE engine path.
2. Diagnose the resulting integration discrepancy until the selected candidate is
   exact.  Do not enable Q3 cache service beforehand.
3. Enable coherent Q3 cache eligibility in both the generic Metal path and
   `laguna_decode_experts_cache_servable()`, so cache-service eligibility and
   decode static-span construction agree.
4. Extend the regression test to require nonzero cache entries, hits, and
   live bytes and to assert the expected Q3 routed static-span reduction.
5. Run `tests/xs21_stream_ab.sh`, then rerun the 800/1600/3200/4800 MiB sweep.
   Only those results may replace the provisional footprint projections.

P2.6 hotlist work and any constrained-hardware acceptance claim remain
blocked behind this sequence.

## Evidence and scope

Supporting artifact commands/type map: `gguf-tools/imatrix/laguna-xs21-README.md`.
Quality caveats and fixture results:
`gguf-tools/quality-testing/data/laguna-xs21/README.md`.
Phase-wide roadmap: `docs/superpowers/LAGUNA-XS21.md`.

The current worktree intentionally keeps
`gguf-tools/imatrix/laguna-xs21-biased.imatrix.gguf` untracked.  It is an
intermediate 179 MiB calibration artifact, not a source/documentation change
to stage.

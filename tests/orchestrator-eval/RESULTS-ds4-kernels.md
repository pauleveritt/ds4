# Task-level quality of ds4's Mellum kernels

Run 2026-08-22 with `run_ds4_quality.sh`, which points the committed probes at
`ds4-server` rather than `llama-server`. Every other report in this directory
scores the *model*; this one scores the *engine*, so it moves when ds4's
numerics move.

The question it answers: expert-major MoE reassociates, giving 0.09% relative
rms on logits. Does that cost anything a task can see?

## Results, n=3, no-think, temperature 0.2

| Config | Task A (of 15) | mean | Task B (of 18) | mean |
| --- | --- | ---: | --- | ---: |
| production (`DS4_MELLUM_MOE_GEMM=1`) | 14, 13, 15 | **14.0** | 13, 14, 14 | **13.7** |
| bitwise oracle (`=0`) | 15, 12, 14 | **13.7** | 11, 15, 11 | **12.3** |

## Reading

**No detectable quality drop** — and read that as the strongest claim the data
supports, not as evidence the accelerated path is better.

The within-config spread is 2–3 checks on Task A and 1–4 on Task B, which is
**larger than the difference between the configs** (+0.3 and +1.3). Production
scoring higher on both is noise, not signal. This eval resolves differences of
roughly two checks and nothing finer.

That limit is not new: `REPORT-sft3015.md` records the same thing from the
other direction, where a single run made thinking look like it helped and
repetitions killed the reading. No-think alone spans 11–13 on Task A.

## Why this exists

Kernel gates — fixture max_abs, the bitwise batch-versus-decode check, the
layer-0 FP32 oracle — catch a broken kernel. They cannot tell you whether a
correct-but-different kernel makes the model worse at its job. Before this,
the numerics policy rested on those plus five greedy transcripts. This closes
that gap for the decision that was actually made.

## Caveats

- Two prompts, one model, three repetitions. Scoring is mechanical: it rewards
  contract conformance, not elegance.
- Both arms use exact projections. The looser `DS4_MELLUM_PREFILL_EXACT=0`
  path is not covered here and deviates roughly 47x more on logits.
- ds4-server must be a Metal build. `make cpu` overwrites it, and the failure
  reads as "Mellum 2 inference currently requires --metal".

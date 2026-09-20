# Mellum 2: the `sft_..._joint-iter-3015` snapshot on the orchestrator eval

Run 2026-08-21 against the JetBrains snapshot
`sft_mellum_v23_mixopt_fullx5_p2_joint-iter-3015` at
`683ca310d8e94baec874a8f026c4a738a261e1ce`, the first local artifact whose
config states Mellum's runtime contract canonically
(`MellumForCausalLM`, per-layer-type rope). It re-runs the two task shapes from
`REPORT-thinking-length.md`, which measured `grpo-v23-step-200`.

**Three repetitions per cell.** The earlier report's cells are n=1 and it says
so; this run shows why that matters — see "On variance" below.

## Summary

**Termination is materially better and thinking is still not worth its tokens.**
The grpo checkpoint never terminated with thinking on, on either task, at either
cap. This snapshot terminated **6 of 6** thinking runs given a 4,096-token
budget. At 2,048 it still fails on Task A, 3 of 3. Scores are flat against the
checkpoint it replaces, and thinking buys nothing on either task shape at three
to four times the output.

## Setup

Apple M5 Max. `llama-server` from llama.cpp `0e4a0362`, `--jinja`, Q8_0.

The GGUF is built here rather than downloaded, because no official GGUF of this
snapshot exists:

```sh
python convert_hf_to_gguf.py <snapshot> --outfile mellum-sft3015-BF16.gguf --outtype bf16
llama-quantize mellum-sft3015-BF16.gguf mellum-sft3015-Q8_0.gguf Q8_0 8
```

`conversion/mellum.py` reads `layer_types` and writes the sliding-window
pattern, so the GGUF carries the correct 3:1 contract rather than a flattened
one. Resulting file is 12.3 GiB; decode ran 110–148 tok/s.

Sampling is `temperature 0.2 / top_p 0.9` through the same request payload the
committed `probe.py` sends; `top_k` is the server default on this path, as it
was for the MLX runs in the earlier report. `probe_any.py` is `probe.py` with
the prompt file as an argument so both task shapes share one path; the payload
is unchanged.

## Task A — extraction (`prompt_p2`, 15 checks)

| Config | r1 | r2 | r3 | mean | completion tokens | terminated |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| no-think | 11 | 13 | 11 | **11.7** | 722 / 631 / 655 | 3/3 |
| thinking, 2,048 cap | 2 | 2 | 10 | 4.7 | 2048 / 2048 / 2048 | **0/3** |
| thinking, 4,096 cap | 13 | 10 | 11 | **11.3** | 1768 / 2772 / 2166 | 3/3 |

The 2,048 cells are not scores in any useful sense: two produced **zero** answer
words and the third produced 79 after its thinking was truncated. They record a
budget failure, not packet quality.

## Task B — inference (`prompt_p3`, 18 checks)

| Config | r1 | r2 | r3 | mean | completion tokens | terminated |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| no-think | 14 | 12 | 14 | **13.3** | 668 / 603 / 448 | 3/3 |
| thinking, 4,096 cap | 14 | 13 | 13 | **13.3** | 2847 / 2861 / 2545 | 3/3 |

## Findings

1. **Termination is fixed at a 4,096 budget, not at 2,048.** Six of six
   thinking runs stopped on their own when given 4,096 tokens, against zero of
   four for grpo in the earlier report. But Task A needs more than 2,048: all
   three runs hit that cap, at 792–1,124 words of thinking. The requirement
   moved from *unbounded* to *roughly 2,200–2,900 tokens per orchestration
   call* — a cost question rather than a correctness one, which is a different
   class of problem than the one previously flagged.

2. **Thinking still buys nothing on either shape.** Task A 11.3 with against
   11.7 without; Task B 13.3 against 13.3. The earlier report predicted
   thinking would win on the inference shape and measured that it did not; that
   result reproduces here on a different checkpoint lineage, now at n=3.

3. **The `default_factory` blind spot is unchanged.** Missed **6 of 6**, with
   and without thinking. The earlier report identified this as an association
   gap rather than a compute gap; a new checkpoint and more budget both leave
   it intact.

4. **`INFER models.py` missed 6 of 6 on Task B.** The file the user-story spec
   never names is still never inferred.

5. **Quality is flat, not improved.** Against the `Thinking` checkpoint's 11/15
   and 14/18 in the earlier report, this snapshot lands 11.7 and 13.3 — inside
   the noise band below. It remains behind `Instruct` on Task A (15/15) and
   slightly ahead of it on Task B (13/18).

## On variance, and a correction

After the first single run this looked like a reversal: thinking scored 13 on
Task A against no-think's 11, and the natural reading was that thinking now
helps. **That reading was wrong and repetitions killed it.** No-think alone
spans 11–13 on Task A and 12–14 on Task B at temperature 0.2, so a two-check
gap is inside run-to-run noise.

The lesson is the one Phase 0 of the kernel work already recorded in a
different guise: a single measurement is not a delta. Any future cell added to
this eval should be n≥3, and gaps under about two checks should not be read as
signal.

## Caveats

- Scoring is mechanical: it rewards contract conformance, not elegance. Treat
  absolute scores as ordinal.
- One prompt per task shape, three repetitions, one checkpoint.
- The Q8_0 build is produced here and is not an official artifact.
- This is the `sft` lineage; the earlier report measured `grpo`. These are
  cousins, not successive versions of one thing.

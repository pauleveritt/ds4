# Mellum 2: thinking length vs. orchestration quality

Feedback prepared for the Mellum team, 2026-08-02, in response to their note
that it is time to start reporting thinking length alongside pass rate.

## Summary

On two orchestration task shapes, thinking did not improve packet quality on
either — and the grpo checkpoint did not terminate on either. Disabling
thinking improved grpo's score on both tasks while cutting output 7-16x.

We expected thinking to win on the second (inference) task. It did not. That
result contradicted our own prediction and is reported as-is.

## Setup

Real spec-driven-development orchestration taken from course material, not a
benchmark. The model acts as the **orchestrator**: it reads short specs and
compresses one roadmap phase into a handoff packet for a fresh implementer
subagent that cannot see the specs and will only ever see the packet.

- Apple M5 Max, oMLX, the published MLX Q8 builds.
- `temperature 0.2`, `top_p 0.9`. One prompt per task shape; only checkpoint
  and the thinking toggle vary.
- Scoring is mechanical string and structure checks against our packet
  contract — not human judgement.

Two deliberately different task shapes:

- **Task A — extraction.** Implementation-specific roadmap. Every literal the
  packet needs is already in the prompt. Nothing to derive.
- **Task B — inference.** The same phase rewritten as user stories. The model
  must infer the file set and translate behavioural requirements into
  technical ones.

## Task A — extraction (15 checks)

| Checkpoint | Thinking | Score | Completion tokens | finish_reason |
| --- | --- | ---: | ---: | --- |
| grpo-v23-step-200 | on, 4096 cap | 10/15 | 4096 | **length** |
| grpo-v23-step-200 | on, 2048 cap | 12/15 | 2048 | **length** |
| grpo-v23-step-200 | **off** | **13/15** | **247** | stop |
| Thinking | on, 2048 cap | 11/15 | 2048 | length |
| Thinking | **off** | 11/15 | **314** | stop |
| Instruct | n/a | **15/15** | 439 | stop |

## Task B — inference (18 checks)

| Checkpoint | Thinking | Score | Completion tokens | finish_reason |
| --- | --- | ---: | ---: | --- |
| grpo-v23-step-200 | on | 11/18 | 4096 | **length** |
| grpo-v23-step-200 | **off** | **15/18** | **579** | stop |
| Thinking | on | 14/18 | 3523 | stop |
| Thinking | **off** | 14/18 | **785** | stop |
| Instruct | n/a | 13/18 | 924 | stop |

## Findings

1. **grpo with thinking never terminated.** It hit the cap on both tasks, at
   2048 and again at 4096.

2. **More budget made it worse.** On Task A, raising the cap from 2048 to 4096
   dropped the score from 12 to 10, because the additional budget went to
   reasoning rather than to the answer. At 4096 the split was roughly 2,395
   words of thinking to 60 words of packet — about 97% scratchpad.

3. **Thinking did not help on either task shape.** We predicted it would win on
   Task B, where genuine inference is required. Measured: grpo 15 (off) versus
   11 (on); Thinking 14 versus 14 at 4.5x the tokens.

4. **A shared blind spot that budget does not close.** Task B states a
   requirement behaviourally: *"no two complaints share a literally identical
   filing instant."* Translating that to a Python dataclass requires
   `field(default_factory=...)`, because a plain default is evaluated once at
   import and every instance would share one timestamp. **No configuration
   caught this**, with or without thinking — and Instruct actively recommended
   the buggy form. Extra thinking budget did not help, which suggests an
   association gap rather than a compute gap.

5. **Failures are confident and well-formatted.** Instruct's Task B packet
   proposed a module named `dataclasses.py`, which shadows the standard
   library; marked a file created in Phase 1 as NEW rather than SHARED; and
   inlined template code it had been told not to include. All inside a clean,
   plausible-looking packet that passes a casual read.

## Caveats

- n=1 per cell, two tasks, one phase. This is a smoke test that found a large
  effect, not a benchmark.
- grpo is an in-progress checkpoint with agentic-loop work in flight. This is a
  baseline to re-run, not a verdict; we may be measuring something already
  being fixed.
- Scoring is mechanical. It rewards contract conformance, not elegance, and an
  earlier looser version of the rubric missed the three defects in finding 5 —
  so treat absolute scores as ordinal, not absolute.

## What we would flag hardest

**Termination.** For an orchestrator role, a model that cannot stop is a harder
problem than one that stops slightly wrong. The non-terminating runs were not
so much incorrect as unfinished — the packet was still being deliberated when
the budget ran out.

This matters disproportionately for recursive or agentic use. In a loop that
makes tens of internal calls per query, per-call thinking length multiplies and
non-termination becomes unbounded cost rather than a quality regression.

## Reproducing

Prompts, raw outputs, and scorers are in this directory: `prompt_p2.txt` (Task
A), `prompt_p3.txt` (Task B), `score.py`, `score_p3.py`, and `probe.py` for the
OpenAI-compatible path with the thinking toggle.

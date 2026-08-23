# RLM readiness probe: the `sft_..._joint-iter-3015` snapshot

Run 2026-08-21, same probe and same 475-line document as `RESULTS.md`, against
the JetBrains snapshot `sft_mellum_v23_mixopt_fullx5_p2_joint-iter-3015` at
`683ca310`, served as Q8_0 through `llama-server` (see
`../REPORT-sft3015.md` for the build recipe). Three repetitions per config.

## Results, 2026-08-21

| Config | Hops | Valid code blocks | Terminated | Tokens | Answer |
| --- | ---: | ---: | --- | ---: | --- |
| thinking on, r1 | 6 | 1 | NO | 4,565 | — |
| thinking on, r2 | 6 | 3 | **YES** | 5,236 | grounded but keyword-shaped |
| thinking on, r3 | 6 | 3 | NO | 3,515 | — |
| no-think, r1 | 6 | **0** | NO | 1,306 | — |
| no-think, r2 | 6 | **0** | NO | 12,288 | — |
| no-think, r3 | 6 | **0** | NO | 7,567 | — |

## The zero-code-block result is a harness mismatch, not a model failure

**Read this before comparing the no-think rows to `RESULTS.md`.** The probe
counts a hop as productive only when the reply contains a ` ```python ` fence.
This snapshot's no-think mode does not use fences. It emits a **`<tool_call>`
block carrying JSON**, with the python in a `code` argument:

```text
<tool_call>
{"name": "code", "arguments": {"code": "import re\n# Find all occurrences ...
```

The python inside is valid, sensible exploration code of the same character the
earlier report praised — regex over `doc`, lesson-number extraction, mechanism
matching. So **code generation is intact**; the probe simply no longer speaks
this checkpoint's dialect, scores every such hop as unproductive, and pushes it
into the "reply with a python block" nudge until the hop cap. The no-think
"terminated=NO" rows measure the harness, not the model.

Thinking-on mode does emit fences, which is why those rows have nonzero counts.
The behaviour is mode-dependent, and that asymmetry is itself worth knowing.

**Consequence for the probe:** accept a `<tool_call>` payload as an alternative
to a fenced block before this comparison is run again. Until then, only the
thinking-on rows are commensurable with `RESULTS.md`.

## What can be said

1. **Termination is still unreliable in a loop, even where it improved
   elsewhere.** One of three thinking runs terminated. That is better than the
   grpo baseline's fabricate-and-stop behaviour and far short of dependable —
   and it sits alongside the orchestrator eval, where the same checkpoint
   terminated 6 of 6 single-shot runs at a 4,096 budget. **Single-shot
   termination and in-loop termination are different capabilities**, and this
   checkpoint has clearly acquired the first without the second.

2. **The one termination was not a fabrication.** The r2 answer lists
   mechanisms with lesson numbers derived from code it actually ran, unlike the
   grpo baseline's ten invented mechanisms. Its failure mode is different and
   milder: the "mechanisms" are keyword hits — `stop`, `bound`, `cap`, `guard` —
   rather than the synthesised mechanisms the query asks for. Retrieval
   substituted for synthesis.

3. **The original reading stands.** Bounded, single-hop uses are viable;
   open-ended "explore this corpus and synthesise" is not. Nothing here changes
   that, and the synthesis-and-termination gap named in `RESULTS.md` is still
   the gap.

4. **Cost is unpredictable in a loop.** No-think token totals ranged 1,306 to
   12,288 for the same query — a 9x spread driven entirely by how long the model
   persisted against a harness that kept rejecting its output format. In a
   recursive loop, that variance is the thing that turns non-termination into
   unbounded cost rather than a quality regression.

## Reproducing

`python3 rlm_probe.py <model_id> <tag> <on|off>` against an OpenAI-compatible
endpoint on `127.0.0.1:8125`. Raw final answers are in `sft3015-*.txt`.

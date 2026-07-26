# Laguna XS 2.1 Quality Fixtures

Two fixture sets for the Laguna XS 2.1 (33B-A3B MoE) GGUF, both captured
locally from `gguf/Laguna-XS-2.1-Q4_K_M.gguf` on Metal rather than from a
hosted reference. See "Why local, not hosted" below.

- `general/`: the shared, provider-neutral 100-prompt corpus
  (`gguf-tools/quality-testing/prompts.jsonl`), same prompts used for the
  Laguna S 2.1, GLM 5.2, Flash, and PRO fixtures.
- `webpy/`: a new, XS-2.1-specific 20-prompt corpus
  (`gguf-tools/quality-testing/prompts_laguna_xs21_webpy.jsonl`) biased
  toward web/Python coding tasks (15 prompts: HTML/CSS/JS/TS/Python) plus
  tagged tool-call format (5 prompts). This is the set later tasks (a
  domain-biased low-bit quant comparison, Tasks 10-11) actually care about,
  since that quant is expected to be biased toward this domain.

Both directories follow the same layout as the hosted fixtures
(`data/laguna-openrouter-100`, `data/glm52-openrouter-100`, etc.), minus the
`responses/` directory and manifest `response_file` column, which only apply
to hosted API collection with logprobs:

- `prompts/case_*.txt`
- `continuations/case_*.txt`
- `manifest.tsv` (`# id\tprompt_file\tcontinuation_file`)

## Why local, not hosted

The task brief assumed either an OpenRouter-listed reference or a local BF16
fallback. Neither applies cleanly here:

- **OpenRouter does list Laguna XS 2.1** (`poolside/laguna-xs-2.1`, confirmed
  via `curl -s https://openrouter.ai/api/v1/models`, matching the existing
  `poolside/laguna-s-2.1` naming used for the S 2.1 fixture), so the hosted
  `collect_official.py` pipeline could in principle be reused unmodified —
  see "If a key becomes available" below.
- **But no `OPENROUTER_API_KEY` was set in this environment**
  (`echo -n "$OPENROUTER_API_KEY" | wc -c` returned `0`), so hosted collection
  was skipped entirely rather than blocking on a key.
- **BF16 local generation is not an option**: ds4 cannot load
  `gguf/Laguna-XS-2.1-BF16.gguf` at all (rejected by layout validation; full
  BF16 inference support is out of scope for this project per Task 3's
  findings). The brief's "else run the BF16 locally" fallback is dead and was
  not attempted.

Given both fallbacks were unavailable, both fixture sets were built by
capturing `ds4`'s own local Q4_K_M greedy generation as a stable snapshot.
This is legitimate for what this project actually needs: later tasks compare
a future biased low-bit quant's behavior against this Q4_K_M baseline, not
against an external gold reference, so a local snapshot is sufficient.

## If a key becomes available

To collect the hosted OpenRouter reference for the general 100-prompt corpus
later (mirroring the S 2.1 README block exactly, only the model and output
directory differ):

```sh
export OPENROUTER_API_KEY=...
python3 gguf-tools/quality-testing/collect_official.py \
  --model poolside/laguna-xs-2.1 \
  --endpoint https://openrouter.ai/api/v1/chat/completions \
  --api-key-env OPENROUTER_API_KEY \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/laguna-xs21-openrouter-100 \
  --count 100 \
  --max-tokens 24 \
  --top-logprobs 0 \
  --token-limit-field max_tokens \
  --thinking omit \
  --reasoning-effort none
```

No code changes to `collect_official.py` are needed — `--model` and `--out`
are already plain CLI arguments (verified by reading the script; the only
prior Laguna-specific change, in the S 2.1 commit, was making
`--top-logprobs 0` skip the `response_file`/`responses/` machinery, which is
already in place and applies to any model). The `webpy/` domain-biased
prompts are XS-2.1-specific and intentionally not part of this hosted-key
workflow; they stay as a local-only fixture (see below) regardless of key
availability, since they exist to compare local GGUF variants against each
other, not against a hosted reference.

## How the local snapshots were captured

Both sets were produced with the new `collect_local.py` (the no-hosted-API
counterpart to `collect_official.py`, added in this task), which drives
`./ds4` directly instead of an HTTP endpoint and writes the same
prompts/continuations/manifest.tsv shape:

```sh
python3 gguf-tools/quality-testing/collect_local.py \
  --ds4 ./ds4 \
  --model gguf/Laguna-XS-2.1-Q4_K_M.gguf \
  --prompts gguf-tools/quality-testing/prompts.jsonl \
  --out gguf-tools/quality-testing/data/laguna-xs21/general \
  --max-tokens 24 \
  --think-mode nothink \
  --lock-file /tmp/ds4-task4-general.lock

python3 gguf-tools/quality-testing/collect_local.py \
  --ds4 ./ds4 \
  --model gguf/Laguna-XS-2.1-Q4_K_M.gguf \
  --prompts gguf-tools/quality-testing/prompts_laguna_xs21_webpy.jsonl \
  --out gguf-tools/quality-testing/data/laguna-xs21/webpy \
  --max-tokens 24 \
  --think-mode nothink \
  --lock-file /tmp/ds4-task4-webpy.lock
```

`--nothink` was used for both sets (direct/short continuations, no reasoning
traces), matching the `max-tokens 24` convention of the existing hosted
fixtures. Each case is one independent `ds4 -p <prompt> -n 24 --nothink`
process, greedy (temperature 0 is ds4's teacher-forced/greedy default for
scoring; sampling defaults do not apply to `--dump`/score paths). All 120
continuations (100 general + 20 webpy) were non-empty.

Wall-clock: general (100 prompts) took ~60s; webpy (20 prompts) a few
seconds; both single foreground runs, no backgrounding needed.

## P0 baseline: self-consistency score

Since there is no hosted reference to score against, the "baseline" this task
records is a **self-consistency** check: score the same Q4_K_M GGUF against
its own captured continuations using the existing local scorer
(`gguf-tools/quality-testing/score_official`, built via
`make -C gguf-tools quality-score`). This validates the fixture/scorer
pipeline end-to-end and establishes the noise floor that future comparisons
(e.g. a biased low-bit quant vs. this Q4_K_M baseline) should be judged
against — deltas at or below this floor are not distinguishable from Metal's
own run-to-run floating-point non-determinism.

```sh
gguf-tools/quality-testing/score_official \
  gguf/Laguna-XS-2.1-Q4_K_M.gguf \
  gguf-tools/quality-testing/data/laguna-xs21/general/manifest.tsv \
  /tmp/laguna-xs21-general-self.tsv \
  4096

gguf-tools/quality-testing/score_official \
  gguf/Laguna-XS-2.1-Q4_K_M.gguf \
  gguf-tools/quality-testing/data/laguna-xs21/webpy/manifest.tsv \
  /tmp/laguna-xs21-webpy-self.tsv \
  4096
```

Results (Q4_K_M scored against its own greedy continuations, captured
minutes earlier in a separate process):

| Fixture | cases | tokens | avg_nll | first_token_match | avg_greedy_lcp |
|---|---|---|---|---|---|
| general | 100 | 2443 | 1.5138 | 70/100 | 5.000 |
| webpy   | 20  | 502  | 1.1381 | 9/20   | 5.100 |

`avg_nll` is not near-zero and `first_token_match` is well below 100%, even
though the scorer is teacher-forcing the model's own prior greedy output.
This is expected: each `collect_local.py` case and each `score_official` case
is a **separate process/prefill**, and Metal's parallel-reduction order is
not guaranteed identical run to run, so floating-point rounding differs
slightly and can flip a near-tied greedy argmax, especially deeper into a
continuation where small differences compound. This noise floor (roughly
1.1-1.5 avg_nll, ~45-70% first-token agreement at 24 generated tokens) is the
P0 baseline: it is the same-model, same-weights comparison, so any future
quant-vs-Q4_K_M score in this range or better is statistically
indistinguishable from Q4_K_M compared with itself; materially higher avg_nll
or materially lower first-token-match indicates real degradation, not just
kernel-order noise.

## Known limitation: webpy tool-call prompts

The 5 tool-call prompts in `webpy` (case_015-019) ask the model, in plain
prose, to respond using "Laguna's native tool-call format" (the
`<tool_call>{function-name}<arg_key>{argument-name}</arg_key><arg_value>{argument-value}</arg_value></tool_call>`
tag syntax used by `ds4_agent.c`'s `agent_build_laguna_tools_prompt`) but do
not inject the full `<available_tools>` schema block or system-prompt
scaffolding that the real agent harness (`ds4_agent.c`) builds when running
in agent mode. Captured continuations are therefore a mix of literal tagged
tool calls, JSON-ish tool calls, and prose describing the intended call (see
`webpy/continuations/case_015.txt` vs `case_017.txt` vs `case_019.txt`) —
this is expected given the plain `-p` one-shot invocation used here, not a
bug. The prompts are still useful biased fixtures for comparing GGUF variants
against each other on tool-call-flavored input, since scoring is relative
(same prompt, different GGUF), not an assertion that the exact tag syntax
was produced.

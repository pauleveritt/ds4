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
fixtures. Each case is one independent
`ds4 -p <prompt> -n 24 --temp 0 --nothink` process, explicitly greedy.

**Correction:** the first pass of this fixture set was captured *without*
`--temp 0` in `collect_local.py`'s `cmd = [...]` construction. Laguna's
sampling default is temperature 1.0 (`ds4_engine_sampling_defaults`, and
`./ds4 --help`: "0 is greedy/deterministic"), so every case in the original
capture was randomly sampled, not greedy. `collect_local.py` now passes
`--temp 0` explicitly and both fixture sets were regenerated from scratch.
All 120 continuations (100 general + 20 webpy) were non-empty in both the
original and the regenerated capture.

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
| general | 100 | 2439 | 1.235013 | 71/100 | 6.330 |
| webpy   | 20  | 514  | 0.948913 | 9/20   | 4.700 |

**Correction (superseding the numbers and explanation below this
paragraph in the original version of this section):** the numbers above
were re-measured after fixing `collect_local.py` to pass `--temp 0` (see
"How the local snapshots were captured" above). The *original* capture
(temperature 1.0, unintentionally sampled) scored general=1.5138 avg_nll /
70/100 first-match / 5.000 avg_lcp and webpy=1.1381 avg_nll / 9/20
first-match / 5.100 avg_lcp. Fixing the sampling bug changed avg_nll
noticeably but barely moved `first_token_match` (general 70→71/100, webpy
unchanged at 9/20) and left `avg_greedy_lcp` in the same range. Greedy
capture is still the correct thing to do (a temp=1.0 fixture is not a
reproducible reference), but the hypothesis that missing `--temp 0` was the
main driver of the low first-token-match rate is **not** supported by the
data — something else dominates.

That something else is **not** run-to-run Metal non-determinism either. That
was the original (also wrong) theory in this section; it was tested
directly and falsified:

- Running `./ds4 -m ... --temp 0 -p <same prompt>` twice, in two separate
  process invocations, produced byte-identical stdout both times.
- Running `score_official` twice against the same manifest, in two separate
  process invocations, produced byte-identical output TSVs both times.

So neither the generation path nor the scoring path is internally
non-deterministic across process invocations — each is reproducible on its
own. The actual source of the ~70-71% first-token / non-zero avg_nll gap is
a **systematic (deterministic, reproducible) numerical divergence between
two different code paths inside ds4**: `collect_local.py` captures
continuations via `ds4`'s free-running autoregressive generate loop (`-p`
prefill, then decode-and-sample-argmax one token at a time), while
`score_official` captures the "self-consistency" comparison via
`ds4_session_sync` (batched prompt prefill) followed by
`ds4_session_eval` (teacher-forced one-token-at-a-time decode, feeding the
fixture's *recorded* token back in regardless of what the scorer's own
argmax says). These two paths are not guaranteed to hit identical
floating-point rounding at every step even with identical weights and
identical token history — e.g. prefill batch shape, KV-cache precision, or
kernel selection can differ between "generate" and "sync+eval" — and a
close-margin greedy argmax can flip as a result, compounding over a
24-token continuation (`case_000`'s lcp=5 in the regenerated run is a good
example: the two paths agree for 5 tokens, then drift).

This noise floor (roughly 0.9-1.2 avg_nll, ~45-71% first-token agreement at
24 generated tokens, post-fix) is the corrected P0 baseline: it is a
same-model, same-weights comparison across two different ds4 code paths, so
any future quant-vs-Q4_K_M score in this range or better is not
distinguishable from this floor; materially higher avg_nll or materially
lower first-token-match indicates real degradation. Pinning down exactly
which kernel/batch-path difference between "generate" and "sync+eval"
causes the divergence is out of scope for this task; flagged as a follow-up
if later tasks need tighter precision than this floor allows.

**Follow-up: this is not XS-2.1-specific.** After this section was
originally written, the same self-consistency methodology (`collect_local.py`
+ `score_official`) was run against Laguna S 2.1 — an already-trusted model
whose decode/prefill code this project's changes (including Task 3's
`laguna_graph_forward_token` QKVG dispatch fix) never touched — on a 15-prompt
sample. It scored **~80% first-token-match** under the identical
methodology — the same rough noise-floor magnitude as XS 2.1's 71%/45-71%
range above, on a model with no code path in common with this project's
changes. This was directly measured, not inferred.

This is confirmed evidence that the divergence documented above is a
pre-existing characteristic of the `collect_local.py`/`score_official`
self-consistency comparison itself — a comparison this tooling was not
originally designed or tested for (it was built to score captured
continuations against externally-hosted reference logprobs, not to score a
model's own free-generation output against itself) — and **not** evidence of
an XS-2.1-specific bug or a regression introduced by Task 3's decode-dispatch
fix. Since the same magnitude of mismatch shows up on a completely unrelated,
already-trusted model under the same methodology, the most defensible
conclusion is that the methodology itself has this noise floor, independent
of which model it's pointed at.

The leading (but unverified) hypothesis for the proximate cause is that
`score_official.c` re-tokenizes the *printed continuation text* independently
(via `ds4_tokenize_text`) rather than reusing the token IDs that were
originally sampled during `collect_local.py`'s generate loop. If the
tokenizer's text-to-token mapping isn't perfectly round-trippable at every
BPE boundary, retokenizing printed text can silently produce a different
token sequence than the one that was actually sampled, which would desync
`score_official`'s teacher-forced eval from the generation that produced the
fixture — independent of any model weights, kernel, or dispatch-path
question. This is reasoned from reading `score_official.c`, **not** verified
independently: no one has diffed raw logits between the two paths or traced
the exact token/step where the two paths first disagree. Treat it as the
leading explanation, not a proven root cause.

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

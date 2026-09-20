# RLM readiness probe

Does the model explore a large out-of-context document programmatically, write
valid code, synthesize, and **stop**? A bounded Recursive-Language-Model loop:
the document lives in a REPL variable `doc`, the model emits python blocks that
are executed for real, and output is fed back. Six hops max.

Document: a 475-line lessons file. Query requires collecting mechanisms
scattered across several sections — synthesis, not a single lookup.

## Results, 2026-08-02

| Config | Hops | Valid code blocks | Terminated | Answer |
| --- | ---: | ---: | --- | --- |
| grpo, no-think | 6 | 6/6 | NO | — |
| grpo, thinking | 4 | 1 | YES | fabricated |
| instruct, no-think | 6 | 6/6 | NO | — |
| Thinking, thinking on | 6 | 4 | YES | "Hello, world!" |
| Thinking, no-think | 6 | 6/6 | NO | — |

**No configuration produced a correct answer.**

## Findings

1. **Code generation is solid.** Every no-think configuration wrote valid,
   sensible exploration python on all six hops — no syntax errors, reasonable
   slicing and filtering. This is the code-model strength showing, and it is
   the half of RLM that already works.

2. **No-think never concludes.** All three no-think runs explored diligently
   and hit the hop cap without ever emitting a final answer.

3. **Thinking terminates without exploring.** grpo stopped after one code block
   and invented ten mechanisms; only one of its terms appears in the document
   and every lesson attribution is wrong. The Thinking checkpoint stopped after
   four blocks and answered "Hello, world!".

4. **A false stop signal is worse than no stop signal.** `FINAL:` fired on
   garbage twice. In a recursive loop, a confident termination on a wrong
   answer propagates upward as though it were verified; a missing termination
   at least fails loudly against the hop cap.

## Reading

The gap is not code generation and not context handling. It is
synthesis-and-termination: deciding that enough has been gathered and turning
gathered evidence into an answer. That is the agentic-loop capability the model
authors have said is in progress, so this is a baseline to re-run rather than a
verdict.

Consequence for design: **bounded** RLM uses are viable today, open-ended ones
are not. Verifying a specific claim ("does this file exist", "does this literal
appear in this file") is single-hop and answerable from one print. Open-ended
"explore this corpus and synthesize" is not.

## Reproducing

`python3 rlm_probe.py <model_id> <tag> <on|off>` against an OpenAI-compatible
endpoint. Executed code is restricted to a small import allowlist and blocks
os/sys/subprocess/open/eval/exec.

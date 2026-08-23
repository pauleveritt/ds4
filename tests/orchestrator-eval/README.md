# Orchestrator packet-generation eval

Measures the capability this project actually needs from Mellum: turning a
roadmap phase into a lean handoff packet for a fresh implementer. It is a
text-in/text-out proxy for the orchestrator role, so it needs no tool harness.

It does **not** measure the implementer role. Those failure modes — stale edit
anchors, doom loops, recursive-listing context explosions — are tool-use
behaviours and need a real agent harness.

## Rubric

Fifteen mechanical checks in `score.py`, derived from the "compact packets must
preserve the contract" lesson: name each writable file and mark it NEW or
SHARED, list what must survive in shared files, carry exact literals verbatim,
give one validation command, name framework semantic traps, and inline no code.

Two failure sides are both scored, because both were observed in practice: an
over-short packet that omits required literals, and an over-long one that
inlines files and leaves the child transcribing.

## Prompts

- `prompt_p1.txt` — Phase 1 packet. Also contains a naturally occurring
  contradiction between the specs (`tech-stack.md` says Uvicorn runs via
  `main.py`; the roadmap says create `app.py`). The prompt explicitly asks for
  contradictions to be reported.
- `prompt_p2.txt` — Phase 2 packet. More discriminating: it has shared files
  carried over from Phase 1, so it tests whether preservation requirements are
  stated rather than only new work. This is the *extraction* shape: every
  literal the packet needs is supplied.
- `prompt_p3.txt` — the same Phase 2 written as user stories. The *inference*
  shape: the file set must be inferred and behavioural requirements translated
  into technical ones. Scored by `score_p3.py` (18 checks).

## Running

GGUF via llama.cpp (`run_gguf.sh LABEL MODEL.gguf PROMPT.txt`) or any
OpenAI-compatible server. Then `python3 score.py`.

Sampling was `temp 0.2 / top_p 0.9 / top_k 64`, matching the profile the
lessons recommend for routine work. Output cap 2048, matching the documented
budget — hitting that cap is itself a result, not a harness limitation.

## Results, 2026-08-02, Apple M5 Max

Phase 2 packet, 15 checks:

| Config | Score | Words |
| --- | ---: | ---: |
| Mellum `instruct` Q8 (MLX) | **15/15** | 169 |
| Mellum Thinking Q8_0 (GGUF) | 14/15 | 129 |
| Mellum Thinking Q6_K | 13/15 | 119 |
| Gemma 4 12B it Q8 | 12/15 | 275 |
| Mellum `grpo` Q8 | 12/15 | 1151 (hit cap) |
| Mellum Thinking Q4_K_M | 11/15 | 150 |
| Mellum `thinking` Q8 | 11/15 | 1099 (hit cap) |

Findings:

- **Checkpoint dominates quantization.** Quantization is monotonic but modest
  (Q8 14, Q6 13, Q4 11); checkpoint choice at identical Q8 spans 15 to 11.
- **Q4 fails by omission, not by breaking.** It dropped "Complaints Board", the
  tagline, and the favicon URL — exactly the silent-branding-change failure
  mode the lessons document.
- **Thinking is mismatched to this task.** Both thinking variants spent the
  whole output budget without finishing a packet, while `instruct` finished in
  439 tokens and scored higher. Extraction and compression do not benefit from
  a scratchpad the way inference does.
- **Two failures were universal**: nothing flagged the `main.py`/`app.py`
  contradiction, and only the non-GGUF runs stated what must be preserved in
  the shared `app.py`.

Caveats: n=1 per cell, one task, one phase. This is a smoke test that found a
large effect, not a benchmark. `grpo` is an in-progress checkpoint whose
agentic-loop behaviour is still being trained, so its score should be treated
as a baseline to re-run, not a verdict.

The thinking-length column is worth keeping: it is the cost side of a
pass-rate-versus-thinking-length tradeoff that the model authors have said they
want measured.

## Thinking length

`REPORT-thinking-length.md` holds the paired thinking-on/off results across
both task shapes, written up as feedback for the model authors. Headline: on
neither task shape did thinking improve the score, and the grpo checkpoint did
not terminate on either — while thinking-off cut output 7-16x and scored
higher. We predicted the opposite for the inference shape.

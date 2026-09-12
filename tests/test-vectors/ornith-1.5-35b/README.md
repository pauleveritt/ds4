# Ornith 1.5 35B (Qwen3.5-MoE) llama.cpp reference vectors (O0)

Frozen greedy continuations and top-k raw logits for the Ornith 1.5 35B
`Q4_K_M` GGUF, captured from llama.cpp by the host tool
`scripts/ornith/llama_ref_vectors.c`. This is the independent reference the
engine's `qwen35moe` port is measured against; freezing it changes no engine
code.

## Pins

- GGUF: `Ornith-1.5-35B-Q4_K_M.gguf`
- Size: `21,713,463,040` bytes (20.22 GiB)
- SHA-256: `42739874cc2ccfdb8523b23fbe52e29b2a7555c8176737ca9ca0b5d59859d41f`
- llama.cpp: version `0.4.0`, build `10809`, commit `5266f24da`
- Architecture: `qwen35moe`

A later mismatch in any of these is an invalid test, not a model result.

## Schemas

- Vectors: `ds4-llama-ref-v1` (`reference.vec`)
- Manifest: `ds4-llama-ref-manifest-v1` (`manifest.json`)

The vectors are **raw logits, not logprobs**: each `case` names the greedy
selected token per step and the `top_k` raw logits above it, so the engine's
`--dump-logits` compares by rank and value without a log-softmax. Decode is
greedy (`temperature 0`), 4 steps, top-k 20, 4 threads, context 4096 — the
`decode` block records these.

## Files

- `prompts/*.txt` — 16 exact-byte prompts (below).
- `reference.vec` — 16 `case` blocks, 64 `step` blocks, one `top` line per rank.
- `manifest.json` — the GGUF hash, the llama.cpp build, the decode parameters,
  and every prompt's exact `token_ids`.

## Build the generator

Homebrew's `llama.pc` omits ggml's include and library paths and `llama.h`
includes `ggml.h`, so both must be named (the spec's `llama`-only line does not
compile here):

```sh
cc -O2 $(pkg-config --cflags llama ggml) -o /tmp/llama_ref_vectors \
  scripts/ornith/llama_ref_vectors.c $(pkg-config --libs llama ggml)
```

## Regenerate

Run from the repository root:

```sh
LLAMA_REF_BUILD=10809 LLAMA_REF_COMMIT=5266f24da \
/tmp/llama_ref_vectors \
  --model   ~/models/Ornith-1.5-35B-Q4_K_M.gguf \
  --prompts external/ds4/tests/test-vectors/ornith-1.5-35b/prompts \
  --out     external/ds4/tests/test-vectors/ornith-1.5-35b \
  --threads 4 --force
```

The two environment overrides are required in this environment. The generator
discovers the llama.cpp build and commit by parsing `llama-cli --version`,
which this build writes to **stderr**; the generator's stdout-only `popen`
therefore sees nothing and would record `build 0, commit unknown` (the
reference-vector header would read `commit unknown` too). The values above are
what `llama-cli --version 2>&1` reports.

A run at the same `--threads` is deterministic: a second run to a fresh
directory produced a byte-identical `reference.vec`
(`sha256 52094c1411b0ad9f788e32622b7bcedd63c4f4553c9cf447cdcbb3feb969f20e`).
A different `--threads` is a different run and must be labelled as such.

## Prompt set

16 prompts, aimed at the port's tokenization and logits risk surface rather
than at sampling breadth. `token_ids` are recorded per prompt in the manifest;
the table gives the measured byte and token counts.

| id | category | bytes | tokens |
|---|---|---:|---:|
| `code_python_func` | code completion | 20 | 8 |
| `code_c_string` | code completion | 21 | 8 |
| `code_sql_query` | code completion | 30 | 11 |
| `code_review_offbyone` | code review | 83 | 27 |
| `prose_pangram` | prose | 44 | 10 |
| `prose_prices` | prose | 77 | 35 |
| `reason_sort` | short reasoning | 52 | 23 |
| `cjk_chinese` | CJK | 12 | 2 |
| `cjk_mixed` | CJK | 28 | 8 |
| `cjk_ext_b` | CJK, Unicode extension B (`U+20000`) | 4 | 3 |
| `ws_leading_trailing` | boundary whitespace (` \n `) | 3 | 2 |
| `ws_nbsp` | boundary whitespace (NBSP) | 4 | 3 |
| `ws_ideographic` | boundary whitespace (ideographic space) | 5 | 3 |
| `digits_run` | digit run | 7 | 7 |
| `punct_run` | punctuation run | 5 | 3 |
| `long_near_context` | near context edge | 17099 | 3800 |

Thirteen of the sixteen are byte-exact reuses of strings in
`scripts/ornith/tokcmp.py`, so a tokenizer regression and a logits regression
are exercised by the same bytes. The three intentionally new prompts are
`code_review_offbyone`, `reason_sort`, and `long_near_context`.

`long_near_context` is a repeated pangram of 3,800 tokens — about 93% of the
4,096-token context actually used, leaving room for the 4 greedy steps — so
RoPE/positions near the context edge are exercised. It is the only prompt that
is long by design.

### Trailing newline

**No prompt file ends with a newline.** The generator reads the exact file
bytes, so a trailing newline would be part of the prompt and change its
tokenization. Thirteen prompts reuse `tokcmp.py` corpus strings verbatim, and
those strings carry no trailing newline; appending one would break the exact
reuse. The three new prompts follow the same convention. (The single prompt
whose content contains a newline, `ws_leading_trailing`, ends with a space, not
a newline.)

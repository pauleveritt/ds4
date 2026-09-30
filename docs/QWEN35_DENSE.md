# Dense qwen35 on the Metal graph (experiment)

Branch `qwen35-dense`, from `c97c0da`.  An experiment: nothing here is on
ds4-engine's main or its submodule pin.

**What is measured.** On one M4 Max (36 GB), two dense `qwen35` GGUFs
from `satyrn-ai` on Hugging Face: Ornith-1.5-9B (Q4_K_M, 5,780,090,208
bytes, sha256 prefix `5533e1a703c072fa`, repository
`satyrn-ai/Ornith-1.5-9B-py3.15` at `b66b600`) and Qwen3.8-27B (Q4_K_M,
16,810,715,584 bytes, sha256 prefix `990750f543aa83b2`, repository
`satyrn-ai/Qwen3.8-27B-py3.15` at `f3dbc8b`).  Both load; their logits and
greedy tokens are compared with llama.cpp `19e28a2` (Metal, and for the 9B
its CPU backend as a second reference) on five fixed prompts fed as the
same token ids; and their prompt and generation speed is timed beside
`llama-bench`.

**What is not measured.** Long contexts beyond these prompts, sampling,
tool calls, the MTP block, vision, speed on any other machine, and any
quality claim beyond agreement with llama.cpp.

## What runs

`general.architecture = qwen35` selects a Qwen-family variant
(`DS4_VARIANT_QWEN35`) whose widths are read from the GGUF, so the 9B and the
27B share one path.  Per trunk layer, following llama.cpp's
`src/models/qwen35.cpp`:

    R += mixer(rmsnorm(R) * attn_norm)
    R += ffn_down(silu(ffn_gate(x)) * ffn_up(x)),  x = rmsnorm(R) * post_attention_norm

with the gated DeltaNet mixer on three of every four layers and gated GQA
attention on the fourth; logits are `output(rmsnorm(R) * output_norm)`.

Reused from the Qwen3.8 Flash Next path: the session machinery, ChatML
rendering, the `qwen35` pre-tokenizer, the GDN conv/prep/scan kernels, the
attention prep kernel (its grid stops before the indexer slots) and the
gated decode/prefill attention kernels, and the Q4_K/Q6_K dense matmuls the
Laguna path uses.

New: the validator and weight binding for `qwen35` tensor names, the dense
forward pass (`qwen35_graph_forward_tokens`), and `kernel_qwen35_gdn_out`:
the one numerical difference inside the GDN block is the output gate,
`silu(z)` here against `sigmoid(z)` in Qwen3.8.  The chat template decides
whether a reasoning-effort instruction is rendered (the 9B's does not have
one; the 27B's, a Qwen3.8 template, does).  Prompt batches of more than 8
tokens multiply Q6_K weights through `kernel_mul_mm_q6_K_f32`, the tiled
kernel of `metal/dense.metal` with the Q6_K dequantizer of
`metal/moe.metal`, the same instantiation llama.cpp uses; decode keeps the
Laguna Q6_K matvec (milestone 3).

Not implemented, and refused at load: `--mtp`, `--vision`, KV checkpoint
save/load, and non-Metal backends.  The CPU first-token reference is Qwen3.8
only.

## How to check it

From the repository root, with llama.cpp built and `LLAMA_SERVER` pointing at
its `llama-server`:

    make ds4
    tests/qwen35_parity/ds4_side.sh MODEL OUT/ds4
    uv run --no-project python tests/qwen35_parity/llama_side.py MODEL OUT/ds4 OUT/llama
    uv run --no-project python tests/qwen35_parity/compare.py OUT/ds4 OUT/llama [--table]

`ds4_side.sh` renders each prompt with ds4's chat template (explicit system
prompt "You are a helpful assistant", thinking on) and runs 64 greedy tokens
with the top-5 logprobs of every step.  `llama_side.py` sends those exact
token ids to llama-server (`/completion`, temperature 0, `n_probs` 5), and
records whether llama.cpp's own template and tokenizer give the same ids.

## Milestone 1: the 9B loads and its top-5 matches

All five prompts: llama.cpp's template and tokenizer produce exactly the ids
ds4 renders.  Top-5 at the last prompt position, ds4 (A) against llama.cpp
Metal (B):

| Prompt | Tokens | A top-5 (id: logprob) | B top-5 (id: logprob) | Same ids, same order | Max abs. diff |
|---|---|---|---|---|---|
| en-capital | 27 | 40: -0.466, 1919: -1.207, 760: -2.962, 14773: -6.023, 90700: -6.217 | 40: -0.469, 1919: -1.197, 760: -3.007, 14773: -5.947, 90700: -6.262 | yes | 0.075 |
| en-hashmap | 34 | 40: -0.279, 1919: -1.682, 760: -3.514, 32: -5.289, 14773: -5.683 | 40: -0.303, 1919: -1.603, 760: -3.444, 32: -5.321, 14773: -5.712 | yes | 0.079 |
| ja-seasons | 32 | 40: -0.904, 1919: -1.026, 760: -2.966, 48543: -3.037, 176034: -3.258 | 40: -0.888, 1919: -1.060, 760: -2.943, 48543: -3.046, 176034: -3.220 | yes | 0.038 |
| code-fib | 34 | 40: -0.588, 1919: -1.004, 760: -3.038, 1596: -5.119, 9930: -5.767 | 40: -0.617, 1919: -0.964, 760: -3.015, 1596: -5.106, 9930: -5.667 | yes | 0.100 |
| code-cbug | 72 | 40: -0.349, 1919: -1.770, 760: -2.389, 9930: -5.041, 14773: -6.028 | 40: -0.339, 1919: -1.784, 760: -2.436, 9930: -5.060, 14773: -6.096 | yes | 0.068 |

Scale of those differences: llama.cpp's own CPU backend against its Metal
backend on the same ids gives a different top-5 order on 2 of the 5 prompts
and differences of 0.11 to 0.48 over the shared ids.  A layer-by-layer
comparison on en-capital (llama.cpp's `llama-eval-callback` against a
temporary ds4 dump, not committed) agreed on the layer-0 norm to four
decimals; the GDN intermediates (decay, beta, normalized q, gated-norm
output) agreed to within about 1e-3, and the projections differed at the
1e-3 level expected from llama.cpp's f16 activations in its prefill
matmuls.

## Milestone 2: 64 greedy tokens

ds4 against three llama.cpp configurations on the same token ids: Metal
with its default batched prompt processing, Metal with `-ub 1` (every prompt
token through the one-row kernels decode uses), and the CPU backend
(`-ngl 0`).  "Max diff" is the largest absolute logprob difference of the
chosen token over the steps where both agree.

| Prompt | Metal default | Metal `-ub 1` | CPU |
|---|---|---|---|
| en-capital | 64 of 64, max diff 0.013 | 64 of 64, max diff 0.0012 | 64 of 64, max diff 0.148 |
| en-hashmap | 64 of 64, max diff 0.024 | 64 of 64, max diff 0.0010 | 64 of 64, max diff 0.158 |
| ja-seasons | diverges at step 15, max diff 0.016 before it | 64 of 64, max diff 0.0019 | diverges at step 16 |
| code-fib | 64 of 64, max diff 0.029 | 64 of 64, max diff 0.0010 | 64 of 64, max diff 0.128 |
| code-cbug | 64 of 64, max diff 0.010 | 64 of 64, max diff 0.0010 | 64 of 64, max diff 0.136 |

Against `-ub 1` every prompt matches for all 64 tokens, and the top-5 at the
last prompt position differs by at most 0.004 (against 0.038 to 0.100 with
the default batching, milestone 1).

The one divergence is a near-tie that llama.cpp's own configurations split.
At ja-seasons step 15 ds4 ranks ` with` (440, -0.718) over ` by` (539,
-0.723); default llama.cpp ranks ` by` (-0.720) over ` with` (-0.721), a gap
of 0.001.  llama.cpp with `-ub 1` gives ` with` with ds4's numbers
(-0.718 and -0.723), and so does its CPU backend; so the flip comes from
llama.cpp's batched prompt arithmetic, not from a difference in the model.
The CPU run then parts at step 16 (` spring` over ` a` by 0.026 there, 0.28
the other way in ds4 and in Metal `-ub 1`), within the CPU backend's
spread: its chosen-token differences reach 0.16 even where the tokens agree.

    LLAMA_ARGS="-ngl 99 -ub 1" uv run --no-project python tests/qwen35_parity/llama_side.py MODEL OUT/ds4 OUT/llama-ub1
    LLAMA_ARGS="-ngl 0" uv run --no-project python tests/qwen35_parity/llama_side.py MODEL OUT/ds4 OUT/llama-cpu

**The 27B.** Its template renders the reasoning-effort instruction ahead of
the system prompt, and llama.cpp's template and tokenizer give the same ids
on all five prompts.  All five match in every token against both Metal
configurations; en-capital ends at `<|im_end|>` after 52 tokens in all
three runs, the others run to 64.

| Prompt | Metal default | Metal `-ub 1` |
|---|---|---|
| en-capital | 52 of 52, max diff 0.0050 | 52 of 52, max diff 0.0004 |
| en-hashmap | 64 of 64, max diff 0.0073 | 64 of 64, max diff 0.0006 |
| ja-seasons | 64 of 64, max diff 0.0204 | 64 of 64, max diff 0.0013 |
| code-fib | 64 of 64, max diff 0.0048 | 64 of 64, max diff 0.0005 |
| code-cbug | 64 of 64, max diff 0.0135 | 64 of 64, max diff 0.0007 |

This table is the build with milestone 3's tiled Q6_K prefill; the 9B's
tables above are the build before it.  Between the two builds every run has
the same tokens (max chosen-token difference 0.0012 on the 9B, 0.0005 on
the 27B), and the 9B's comparison with llama.cpp is unchanged: all five
against `-ub 1`, and ja-seasons' step-15 near-tie against the default.

## Milestone 3: speed

`tests/qwen35_parity/bench.sh MODEL` runs 128 greedy tokens after each of
two fixed prompts (502 and 2006 rendered tokens on the 9B, 540 and 2044 on
the 27B), one warm-up then three runs; below are the medians of ds4's own
prefill and generation rates.  llama.cpp's are `llama-bench` at its
defaults (`-b 2048 -ub 512`, flash attention auto), three repetitions:
`pp512`/`pp2048` from an empty context and `tg128` at depth 512/2048.  The
counts differ slightly (502 against 512 tokens and so on), so the columns
compare rates, not identical work.

| Model | ds4 prefill, Q6_K matvec | ds4 prefill, tiled Q6_K | llama.cpp prefill | ds4 generation | llama.cpp generation |
|---|---|---|---|---|---|
| 9B, short | 286.65 t/s | 560.97 t/s | 646.35 t/s | 58.40 t/s | 55.01 t/s |
| 9B, long | 285.69 t/s | 558.98 t/s | 646.38 t/s | 56.36 t/s | 54.00 t/s |
| 27B, short | 85.92 t/s | 167.19 t/s | 188.29 t/s | 19.85 t/s | 18.66 t/s |
| 27B, long | 85.66 t/s | 169.13 t/s | 187.21 t/s | 19.45 t/s | 18.59 t/s |

Generation is within the same few percent before and after (58.43 and
19.74 t/s short before); the tiled kernel changes prefill only.

- **Prefill doubled** with the tiled Q6_K kernel and is now 86 to 87 %
  (9B) and 89 to 90 % (27B) of llama.cpp's.  In these Q4_K_M files
  `ffn_down`, the GDN `attn_qkv` and `attn_v` are largely Q6_K (all of them
  in the 27B's first eight blocks), and the matvec read those weights once
  per prompt token.
- **Generation is 4 to 6 % faster than llama.cpp's** on both models.

## Open problems

- The remaining 10 to 14 % of prefill against llama.cpp is not located.
- Nothing longer than 2044 prompt tokens is timed; GDN's scan and the
  attention kernel both grow with the prompt.

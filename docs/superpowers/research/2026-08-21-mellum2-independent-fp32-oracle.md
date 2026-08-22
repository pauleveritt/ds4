# An independent FP32 oracle for Mellum, and what it says about the envelopes

Run 2026-08-21 against branch `mellum-2.1-overnight` at `94ffd86`. It answers
finding **2.4** of `2026-08-01-mellum2-adversarial-review.md` — that the entire
oracle chain rests on one pinned llama.cpp Metal build, against which ds4's
Mellum graph was written, so an architecture-level misreading shared by both
implementations would be invisible to any amount of llama.cpp comparison.

The review named the fix and called it the most under-used asset on the branch:
an HF Transformers FP32 CPU forward from the pinned safetensors, compared at
`l_out-0` and final logits. This is that run.

Claims below are marked the same way the review marked them. **Verified** means
computed here from a pinned fixture or read out of installed source.
**Inferred** means derived from a recorded gate rather than measured.

## Decision summary

The circularity is broken. Transformers' own `MellumForCausalLM` — separate
code from llama.cpp — reproduces the pinned layer-0 fixture to **0.26%
relative** and the logits to **2.40%**, from the same safetensors, and selects
the same greedy token. The architecture reading behind this port (dual RoPE
with layer-selective YaRN, the 3:1 sliding/full pattern, the bias-free
softmax-then-top-8-renormalize router, Q/K per-head norms) is confirmed by an
implementation that shares no code with the oracle it was written against.

The same run reframes the envelopes. **ds4 reproduces llama.cpp's Q8 arithmetic
one to two orders of magnitude more precisely than Q8 arithmetic reproduces the
FP32 model.** The envelopes are therefore sound regression tripwires and can
never be acceptance criteria for model fidelity — which is what the review
concluded from the fixture pair alone, now with a third independent point on
the line.

Open Question 2 is closed: the old flattened `qwen3_moe` export was never a
faithful runtime contract, and the difference is behaviourally visible on a
26-token prompt.

## Pinned evidence

| Source | Revision or identity | Role |
| --- | --- | --- |
| [Public Thinking release](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking) | `ba4838faad89e968c36f39e76e95319d756714fe` | The weights the pinned fixtures were captured from |
| `tests/test-vectors/mellum-llama-cpp/l-out-0.f32` | SHA-256 `5a99d699…49243` | Layer-0 oracle, 26 x 2,304 |
| `tests/test-vectors/mellum-llama-cpp/l-out-27-tokenwise.f32` | SHA-256 `4ae7a46e…86efe` | All-layer oracle, last row, pre-norm |
| `tests/test-vectors/mellum-llama-cpp/result-output-tokenwise.f32` | SHA-256 `4ec7f9c8…2b1c` | Raw-logit oracle, 98,304 values |
| `tests/test-vectors/mellum-llama-cpp/first-token-logprobs.json` | committed fixture | Independent llama-server cross-check |
| New JetBrains SFT snapshot | private repo `sft_mellum_v23_mixopt_fullx5_p2_joint-iter-3015`, `683ca310d8e94baec874a8f026c4a738a261e1ce` | Canonical `MellumForCausalLM` config; secondary subject |
| Transformers | `5.15.1`, torch `2.13.0`, FP32 on CPU, Apple M5 Max | The independent implementation |

Scripts are committed beside the fixtures at `tests/mellum-fp32-oracle/`, on the
same principle as `capture-layer-output.patch`: the artifact and the thing that
produced it travel together.

## Method

The same 26-token `python_add` ChatML fixture the llama.cpp captures use. The
prompt renders **26 IDs**, matching the callback's count, on a `tokenizer.json`
byte-identical to the one already in the tree — so tokenization is part of the
check rather than an assumption.

One correction worth recording, because it produced a wrong first answer.
HF's `output_hidden_states` tuple has the **final RMSNorm already applied** to
its last entry, so `hidden_states[28]` is not `l_out-27`; read that way it gives
RMS 1.97 against the fixture's 63.36 and looks like a catastrophic mismatch.
The numbers below use a forward hook on `model.model.layers[27]`, which is the
post-residual, pre-norm tensor llama.cpp's `l_out-27` holds. Magnitudes agree
once the hook is right.

Comparing a tokenwise-captured fixture against a single full-sequence HF forward
is legitimate: for causal attention the incremental and full forwards compute
the same function, and only the numerical schedule differs — which is part of
what is being measured, not a confound to remove.

## Result: the matched-weights comparison

**Verified.**

| Checkpoint | HF FP32 | llama.cpp Q8_0 fixture | max abs | RMS | relative | cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `l_out-0`, 26 x 2,304 | RMS 0.1785 | RMS 0.1786 | 0.016188 | 4.6926e-4 | **0.2627%** | +0.999997 |
| `l_out-27`, last row, pre-norm | RMS 66.0094, max 1,689.9 | RMS 63.3624, max 1,492.8 | 197.15 | 5.8942 | **9.3023%** | +0.996684 |
| `result_output` logits | RMS 10.6516 | RMS 10.7742 | 1.2790 | 0.25897 | **2.4036%** | +0.999773 |

Greedy token agrees: **910 = `def`** from both. Top-16 overlap is 14/16, with
order differing among near-ties.

```
HF FP32 : 910, 2116, 889, 45742, 3756, 629, 2591, 2190, 433, 4697, 1017, 20320, 75, 3969, 23, 11174
fixture : 910, 2116, 889,   629, 3756, 45742, 2591, 2190, 1017, 433, 4697, 75, 3969, 20320, 21177, 7846
```

A third point corroborates the top of that list from outside both runs:
`first-token-logprobs.json` was captured from **llama-server** — a different
binary and schedule than the eval-callback fixtures — and gives `def` at
logprob −0.2919, against −0.2867 obtained by softmaxing the pinned logit
fixture.

## Result: the reframing of the envelopes

Converting ds4's own recorded measurements to relative terms and putting them on
one scale with the FP32 gap:

| Gate | ds4 vs llama.cpp @ HEAD | FP32 vs llama.cpp | ratio |
| --- | ---: | ---: | ---: |
| `l_out-0` RMS | **not measured post-dot4** (0.0549% pre-dot4) | 0.2627% | ~4.8x, stale |
| `l_out-27` RMS | **not measured post-dot4**; ~0.072% *inferred* from the 5.9e-2 gate | 9.3023% | ~130x |
| logits RMS | 0.0846% (`0.00911324`) | 2.4036% | 28.4x |

So ds4 sits one to two orders of magnitude closer to llama.cpp's Q8 arithmetic
than llama.cpp's Q8 arithmetic sits to the FP32 model it approximates. The
practical consequence: **below roughly 2.4% on logits, two configurations are
indistinguishable in their faithfulness to the actual weights.** The envelopes
work as "no worse than the day it was measured" tripwires, which is how the
review characterised them and how the feasibility document mostly frames them.
They cannot function as acceptance criteria for fidelity, and tightening them
further buys no fidelity — only regression sensitivity.

Layer 27 is the extreme case. Q8_0 quantization alone moves that hidden state
**9.30%**, against llama.cpp's own tokenwise-versus-batched spread of 3.5% and
ds4's batch-versus-decode drift of well under 0.1% at the same length. That is a
third independent confirmation that absolute RMS at layer 27 is the wrong
tracking metric, on a layer whose channels reach |x| ≈ 1,690.

### Two gaps this exposed in the branch's own record

Both **verified** by grepping this document tree:

1. **Post-dot4 `l_out-0` and `l_out-27` RMS were never measured.** The dot4
   adoption table records maxima and logit RMS, not hidden-state RMS, and the
   layer-0 row was not re-run at all — although dot4 changes the MoE arithmetic
   in *every* layer, layer 0 included. The two "not measured" cells above are
   that gap, not an omission here.
2. **This note's first draft mixed three builds.** The relative conversion was
   originally done with `0.0437417` and `0.00941531`, which are the **"Before"
   column** of the down-accumulation table — a state superseded twice, first by
   the accumulation contract and then by dot4. The qualitative conclusion is
   unchanged, but a note whose central move is converting gates to relative
   terms must not itself compare across builds. That is the same stale-reference
   mistake the feasibility document already catches in itself twice, in Phase 0
   and in the Phase 6 soak.

## Result: what this says about the dot4 trade

dot4 moved logit RMS from `0.00178922` to `0.00911324` — in relative terms
**0.0166% to 0.0846%**, both roughly 28x below the 2.40% gap between the Q8
artifact and the FP32 model it approximates. Measured against the true model
rather than against llama.cpp, no fidelity was given up for the +45% prefill and
+26% decode, and the envelope widening in `c851b89` cost nothing real *in
fidelity terms*.

That is the whole of the claim, and it should not be stretched. The branch's own
record shows dot4 changing greedy continuation on one of three prompts, and
states that the outputs are not interchangeable. The shipped artifact is a Q8
GGUF and the ecosystem's de facto reference is llama.cpp's Q8 behaviour, so
output-level divergence on near-tied continuations is a real cost even where
fidelity to FP32 is not. The defensible statement: **dot4 gave up nothing
measurable in fidelity to the FP32 model, at the cost of output-level divergence
from the ecosystem reference on near-tied continuations.**

## Result: Open Question 2 is closed

The new SFT snapshot self-describes as `MellumForCausalLM` / `model_type:
mellum`, with rope parameters **split by layer type** — YaRN for
`full_attention`, plain unscaled RoPE for `sliding_attention`. The older local
GRPO snapshot's `qwen3_moe` export flattens them into a single YaRN block.

Running the *same weights* under both contracts, the difference is not subtle
(**verified**, `tests/mellum-fp32-oracle/rope_ablation.py`):

| | max | RMS | relative |
| --- | ---: | ---: | ---: |
| `l_out-27`, pre-norm | 274.80 | 25.7904 | 29.57% |
| logits | 6.43979 | 1.49005 | 70.23% |

The greedy token flips, `To` → `We`, with top-16 overlap 13/16 — at 26 tokens,
three orders of magnitude below the 8,192 original context where YaRN
interpolation is nominally inert. The mechanism is the `attention_factor` mscale
of 1.2772588722239782 being applied to sliding layers that should receive
unscaled RoPE: the same failure class as the double-mscale bug that cost a red
gate, but silent, at a magnitude no envelope on this branch would catch.

Two qualifications, both **verified**:

- `MellumForCausalLM` cannot load a flattened config at all — it raises
  `KeyError: 'sliding_attention'` in `MellumRotaryEmbedding.__init__`, because
  it indexes `rope_parameters` by layer type. The upstream implementation
  structurally refuses the contract the old export expressed. The flat case
  above had to be synthesised by duplicating the YaRN block into both keys.
- That synthesis **understates** the old export's damage beyond 1,024 tokens.
  In the installed `modeling_qwen3_moe.py`, `layer_types` is not a config field
  and is not referenced, and the mask selection at `:483` is
  `create_causal_mask if self.config.sliding_window is None else
  create_sliding_window_causal_mask` — applied globally. With the old config's
  `use_sliding_window: true` and `sliding_window: 1024`, stock Qwen3-MoE would
  window *every* layer, including Mellum's seven full-attention layers, on top
  of YaRN-everywhere. At 26 tokens a 1,024 window is inert, so the two are
  behaviourally identical on this fixture and the measurement stands; at longer
  context the real gap is larger than what is tabulated above.

The ablation ran on the SFT snapshot's weights rather than the Thinking weights
the fixtures use. It is a measurement of a config effect, not a member of the
oracle chain.

## What this check cannot exclude

Stated plainly rather than as a footnote, because the limit is load-bearing.

The FP32-versus-fixture gap **conflates quantization with implementation
difference**. It bounds implementation error at no more than the measured value;
it does not isolate it.

A **ds4-only** defect cannot hide in the 9.30%: ds4 is pinned to llama.cpp three
orders of magnitude below that figure, so any ds4-specific error is bounded far
under the FP32 gap. What remains unexcluded is a **shared** ds4-and-llama.cpp
error smaller than the quantization noise floor — a few percent at layer 27, up
to about 2% on logits. The relevant calibration is already in the record: the
feasibility document notes llama.cpp's own tokenwise-versus-batched schedules
differing by `2.18424` RMS at layer 27 — **3.4% relative within a single
build**, and the review's independent recomputation from the two pinned fixtures
gives 2.199, or 3.5%. A subtle shared
misreading contributing one to three percent — one layer's rope type, a norm
epsilon — could sit inside the 9.30% undetected. The end-to-end anchors (greedy
token agreement, 14/16 top-16, 2.40% on logits) make a *consequential* shared
error unlikely. They do not make it impossible.

Separating the two components is a bounded next step and needs no new capture: a
dequantized-Q8 FP32 forward from the pinned GGUF isolates quantization from
implementation, and would let the envelopes be re-derived on a principled basis
rather than as 1.3x-of-measured tripwires.

## A secondary result: the new SFT snapshot is not an oracle

The same forward was run first against the new private snapshot, before the
matched Thinking weights were fetched. It is recorded here so the negative
result is not repeated.

| Checkpoint | relative | cosine |
| --- | ---: | ---: |
| `l_out-0` | 19.20% | +0.9816 |
| `l_out-27`, pre-norm | 76.64% | +0.8097 |
| logits | 93.66% | +0.4045 |

Top-16 overlap 9/16; argmax `To` against the fixture's `def`. The gaps are
dominated by the checkpoint difference and **no conclusion about ds4's
implementation follows from any row**. What survives is narrow: layer-0 hidden
RMS agrees to 0.2% with cosine 0.98 across two different finetunes, and both
checkpoints show the same massive-activation regime at layer 27.

The snapshot's value to this port is its **config**, not its weights — it is the
first artifact in the local record to state Mellum's runtime contract
canonically. Note also that it displaces Open Question 1 further: it is the
`sft` lineage, while the branch's orchestrator evaluations measured
`grpo-v23-step-200`. Those results are a baseline to re-run against it, not a
prediction about it.

## Provenance of this note

The measurements were reviewed by an independent adversarial pass that
re-derived every statistic from the saved arrays and raw fixtures, read the
installed `modeling_mellum.py` and `modeling_qwen3_moe.py`, and checked the
scripts for capture errors. It confirmed the matched-weights numbers, the script
mechanics, and the legitimacy of the tokenwise-versus-full-sequence comparison;
it found the stale-build error recorded above, the dot4 overclaim corrected
above, and supplied the shared-error sharpening and the Qwen3-MoE global-window
reading. Both corrections were verified against this tree before being folded
in.

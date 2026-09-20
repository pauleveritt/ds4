# Semi-adversarial review of the Mellum 2 port

Review date 2026-08-01, against branch `mellum-2.1` at the state described by
`2026-08-01-mellum2-feasibility.md` (1,371 lines). Three independent reviewers
worked from separate lenses — numerical/evidence, strategic/ecosystem, and
process/scope — and did not see each other's findings. This document
synthesises them. It is a review only: no code was modified and no
recommendation here has been acted on.

Claims below are marked by how they were established: **verified** means a
reviewer or the synthesiser read the code or computed it from a pinned fixture;
**reported** means it is taken from the feasibility document; **judgement**
means it is an opinion the evidence supports but does not compel.

## Resolution status (updated same day)

Findings 1.1, 1.2, 1.3, 1.4 and 2.1/2.2 were acted on immediately. Outcome:

- **1.1 confirmed exactly.** Forcing row-exact projections makes the 26-token
  probe and the 1,024-token pre-wrap batch **bit-identical** to sequential
  decode, and cuts post-wrap drift ~2,100x on logits. The drift was projection
  kernel precision, not accumulation. Full detail is now in the feasibility
  document under "Drift resolved".
- **1.2 confirmed.** Chunk-size-1 prefill is bit-identical to decode across all
  28 layers, proving the layer-major graph semantics were always correct.
- **1.4 confirmed as an evidence defect.** The "bit-identical 32-chunk" claim
  was never demonstrated by the code. Both schedules do agree, but the evidence
  was a six-significant-digit printed summary.
- **2.1 confirmed.** Absolute-only reporting was masking scale. Corrected in the
  feasibility document.
- **1.3 (expert-selection audit) is now lower priority**: with exact
  projections the batch and decode paths agree bitwise, so selection cannot
  diverge on that path. It remains worth running if the fast projection path is
  retained.
- **Strategic 3.2 answered by the user, not by measurement:** the deployment
  target is *below* 16 GB, so resident Q4 cannot fit and SSD expert streaming
  is required rather than optional. Open Question 5 is closed by intent.

The performance section's premise also changed. First trustworthy prefill
measurements show the routed MoE is 96.8% of prefill time and gets zero weight
reuse across tokens, making prefill only 1.7x decode. Expert-major grouping is
the fix, and it is the same access pattern SSD streaming needs.

## Bottom line

The engineering discipline on this branch is genuinely above the norm — the
fixture provenance, the staged gates, and the recording of negative results are
better than most production inference work. That is not in dispute, and the
detailed praise in the sections below is meant literally.

But the review found one finding that changes what should happen next, and two
that change what the work is *for*.

The central open problem — batched-prefill drift, currently described as
"unresolved long-sequence numerical accumulation" — has a concrete, verified,
mechanical candidate cause that the document never considers. It is very likely
not accumulation. It is very likely the kernel dispatch policy, and it is
testable in about an hour without any new reference capture.

That matters doubly because the document's planned next step is a multi-day
llama.cpp long-context capture that would not discriminate this cause at all.
The plan is currently pointed away from its own answer.

## Part 1 — The finding that should change the next action

### 1.1 Batched prefill runs at lower arithmetic precision than decode, by construction

**Severity: critical to the diagnosis. Verified.**

`ds4_gpu_matmul_q8_0_legacy_tensor` (`ds4_metal.m:16836` onward) selects a
different Q8_0 kernel according to token count:

| Tokens | Kernel | Precision |
| --- | --- | --- |
| 1 (decode) | `kernel_mul_mv_q8_0_f32` | int8 × F32, **F32 accumulate** |
| 2–16 | `mul_mv_ext` | F32 dequant, reordered reduction |
| 17–31 | generic `kernel_mul_mm_q8_0_f32` | **half weights and half activations** |
| ≥32, %32 | NAX tensor-op path | weight tiles **dequantised to half** |

The NAX branch says so in its own comment at `ds4_metal.m:16945`: it
"dequantizes each 64x32 weight tile to half in threadgroup memory."

A Q8_0 weight is `d(F16) × q(int8)`. That product needs roughly 18 mantissa
bits and is **not exactly representable in half**. So every projection on the
prefill path carries a relative weight error of order `2^-12`–`2^-11` that the
decode path does not have. Activations are additionally rounded on the 17–31
path.

This single mechanism is consistent with every recorded observation:

- Layer-0 drift small but nonzero (`1.35e-5` RMS, about F16 noise relative to
  the layer-0 hidden RMS of 0.179).
- Smooth exponential per-layer amplification rather than a step at one layer.
- Drift growing with sequence length, since all 1,024 tokens' KV passes through
  the degraded projections.
- The 32-chunk and 1024+6 schedules agreeing, since both land in the same
  kernel classes.

The decisive detail: **the repository already contains the fix.**
`ds4_gpu_matmul_q8_0_decode_rows_exact_tensor` (`ds4_metal.m:17093`) exists
precisely to make batched evaluation bit-identical to decode, and the generic
session-batching path already uses it (`ds4.c:65336`, `67122`, `68314`, and
others — verified). The Mellum prefill composition
(`ds4_metal.m:34362`, `34515`) does not call it.

**The experiment.** Re-run `--mellum-true-prefill-probe` and the SWA probe with
Mellum's prefill projections routed through the rows-exact matmul, behind a
flag. If drift collapses by two or three orders of magnitude, the central open
problem is not a mystery: it is a deliberate speed/precision policy, and the
remaining decision is a measured trade — plausibly a quality-mode prefill,
noting that a `g_quality_mode` global already exists in this dispatch function.
If drift persists, the staged-GQA or RoPE batch path is implicated, which is
also a localisation. Either outcome retires the "unresolved accumulation"
framing.

### 1.2 The cheapest discriminator — chunk size 1 — was never run

**Severity: minor as an omission, high value as an experiment. Verified.**

At `n_tok == 1` the prefill composition dispatches the decode matvec. So a
chunk-size-1 run through the already-written `ds4_mellum_prefill_chunks`
(`ds4.c:61830`) should be bit-identical, or nearly so, to sequential decode *if
the layer-major graph is semantically equivalent*. This is the sharpest
available split between "the graph is wired differently" and "the batched
kernels are numerically different," it costs one probe variant, and it does not
exist. Run it alongside 1.1.

### 1.3 Expert-selection divergence is never measured

**Severity: major. Verified.**

The document concludes layer-27 drift is "amplification … not a distinct YaRN or
router failure." But the true-prefill probe compares only hidden states,
attention outputs, and logits (`ds4.c:61677`), and the SWA probe compares only
hidden states and logits (`ds4.c:61997`). The `router_selected` and
`router_weights` buffers exist on both states (`ds4.c:36705`, `36783`) and **no
probe ever reads them.**

The batched router's bitwise-identity regression only holds for *identical input
logits*. Once hidden states drift by percent-scale, top-8 set membership can
flip, converting continuous floating-point noise into a categorical difference —
a different 896×2304 expert matrix applied. At roughly 6% relative hidden drift
at layer 27 over 1,024 positions, flips are close to certain somewhere in the
stack, and a flip at an intermediate layer and position would propagate through
KV while still looking like smooth accumulation in a 2,304-channel RMS.

Note also: the per-layer smooth-accumulation trace exists **only for the
26-token fixture**. The 1,030-token probe passes NULL traces (`ds4.c:61922`).
Smoothness at long context is extrapolated, not measured.

### 1.4 The "bit-identical 32-chunk vs 1024+6" claim is not demonstrated by the code

**Severity: major. Verified.**

The SWA probe compares the 32-chunk run against sequential decode, and the
1024+6 run against sequential decode. There is **no direct comparison between
the two batched schedules anywhere in the probe**, and no `-out` option to diff
files (`ds4_cli.c:110`). At best, identity was inferred from two printed
six-significant-digit summary lines agreeing. That is not a bitwise claim, and
the document should not carry it as one.

Even granting bit-identity, the inference is narrower than stated. It rules out
chunk-size-dependent bookkeeping *between those two schedules*. It does not rule
out any **common-mode** defect shared by all batched invocations — including
1.1. It is also partly a kernel-selection coincidence: both 32 and 1024 are
multiples of 32 and land in the same NAX branch. A 20-token chunk schedule would
likely not be bit-identical, and the "chunk size doesn't matter" conclusion
would not survive it.

Credit where due: the reviewer specifically hunted for a mask or position
off-by-one here and found none. Decode's `key_start`/`key_count`
(`ds4.c:36977`) and prefill's `key_count = min(query_pos+1, cache_cap)`
(`laguna.metal:725`) agree; both kernels iterate oldest-to-newest with the same
online-softmax update; RoPE uses `pos0 + token`. That part is clean.

## Part 2 — The reframing: what the drift numbers actually mean

### 2.1 Everything is reported in absolute error, and the relative picture is different

**Verified by computation over the pinned fixtures.**

The reference layer-27 hidden state (`l-out-27-tokenwise.f32`) has RMS **63.36**
and max |x| **1,492.8** — massive-activation channels. Converting the
document's headline numbers:

| Measurement | Absolute | Relative |
| --- | ---: | ---: |
| Tokenwise all-layer gate | 0.0437 RMS | ~6.9e-4 |
| 26-token batch drift | 0.0735 RMS | ~1.2e-3 |
| 1,024-token batch drift | 3.93 RMS | ~6% |
| Logit drift, pre-wrap | 0.0908 RMS | ~0.8% |
| Logit drift, post-wrap | 0.371 RMS | ~3.4% |

The accepted max-abs tolerance of 1.19 on a hidden state sounds alarming but is
probably a massive-activation channel at under 0.1% relative error. The document
never reports the location or the relative figure, so a reader cannot tell
benign from broken. **Track relative error, not absolute RMS.**

### 2.2 The reference disagrees with itself by more than ds4 does

**Verified by diffing two pinned fixtures.**

Diffing `l-out-27-last.f32` (llama.cpp batched) against `l-out-27-tokenwise.f32`
(llama.cpp tokenwise) gives RMS **2.199**, max abs **48.26** — about **3.5%
relative, at only 26 tokens.** That is roughly 30× larger than ds4's own
batch-versus-decode drift at the same length.

The document *records* the 2.18 figure at line 757, but treats it as a local
observation about that one comparison. It never draws the inference, and the
inference is important in both directions:

- **Supporting the project's instinct:** a magnitude of 3.93 is not by itself
  evidence of a categorical bug. This model's final layers are numerically
  chaotic; RMSNorm over channels reaching 1,493 amplifies directional noise. No
  batched schedule will closely match tokenwise decode, in ds4 *or* in
  llama.cpp.
- **Against the current plan:** the implicit target "batch must match decode" is
  therefore unattainable, and an acceptance envelope built on it would be
  meaningless. The right comparison is ds4-batched against llama.cpp-batched,
  plus behavioural top-k agreement.

On the oddity the brief flagged — hidden RMS falling 3.93 → 2.48 while logit
RMS rises 0.091 → 0.371 — the explanation is mundane: the two measurements are
at different positions (1,023 vs 1,029) with different hidden magnitudes, and
logits derive from the RMS-*normalised* hidden state, so they track directional
error, which grew. Not suspicious. But it is a clean proof that absolute hidden
RMS is the wrong tracking metric.

### 2.3 Every envelope was set after the measurement, at 1.27–1.36× the observed value

**Verified.** Layer-0: measured `9.80e-5`/`4.94e-3` → gate `1.25e-4`/`6e-3`.
All-layer: measured `0.0437`/`1.189` → gate `0.06`/`1.5`. Output head: measured
`0.0094`/`0.0175` → gate `0.012`/`0.025`
(`tests/check_mellum_layer0_oracle.py:20`, `:32`, `:39`).

These are honest **regression tripwires** — "no worse than the day it was
measured" — and the document mostly frames them that way. But it also calls them
acceptance gates, which they are not: they encode zero independent knowledge of
what error is tolerable. The risk is concrete and the document supplies the
proof: the YaRN double-mscale bug was caught only because it produced a 27.8 RMS
explosion. Any defect contributing under ~0.04 RMS at layer 27 — one wrong
expert on one token, a single corrupted KV row, a subtly wrong YaRN correction
dimension — passes every gate in the chain silently.

The most meaningful gate is actually the output-head one (0.0175 max on logits
spanning ±22, top-16 rank order preserved), because it is post-normalisation and
scale-invariant. The document undersells it.

### 2.4 The oracle is not independent, and a genuinely independent one is sitting unused

The entire chain rests on one pinned llama.cpp Metal build on one machine — and
ds4's Mellum graph was written *against* llama.cpp's graph. The YaRN fix made
ds4 match llama.cpp's mscale convention; nothing checks either against the
technical report or Transformers. Correlated-error risk is real: an
architecture-level misreading shared by both implementations is invisible to any
amount of llama.cpp comparison.

The local GRPO **safetensors** checkpoint is pinned (doc line 39). An HF
Transformers FP32/BF16 CPU forward for the existing 26-token fixture, compared
at `l_out-0` and final logits, is one script, breaks the circularity, and is
currently the most under-used asset on the branch.

## Part 3 — Strategy: what the port is for

### 3.1 Every completed gate reproduces llama.cpp; none exceeds it

The gate history is: tokenizer matches llama.cpp; first-token logprobs match
llama.cpp; layer-0 within 9.8e-5 of llama.cpp; all-layer within 0.044 of
llama.cpp; output head within 0.0094 of llama.cpp; top-16 ranking identical to
llama.cpp. Excellent verification practice — and a precise accounting of
capability delivered beyond the oracle, which is zero. Even the YaRN bug that
cost a red gate was a bug llama.cpp does not have.

The sharpest illustration, and it is worth sitting with: **the project had to
build a tokenwise variant of llama.cpp to match ds4's limitation.** The oracle
was degraded to match the port.

### 3.2 The differentiator is deferred at every decision point

SSD expert streaming is the one thing ds4 does that the ecosystem does not.
llama.cpp's answer to "MoE larger than RAM" is mmap demand-paging plus
`--cpu-moe`; a bounded routing-aware expert cache is an *open upstream feature
request* (ggml-org/llama.cpp#20757 — verified). ds4 built one.

It is also deferred at least eight separate times in this document, has no
kernels for Mellum's quant mix, and appears in no completed gate.

Meanwhile the document's own Open Question 5 asks whether resident Q4 plus
layer-specific KV already satisfies the 16 GB target, "making SSD streaming
optional rather than required" — and the document's own KV arithmetic (line 354)
half-answers *yes*. **If that is yes, the entire unique-value thesis for
ds4-Mellum evaporates.** It has never been tested. It could be answered in an
afternoon with llama.cpp on a real 16 GB machine.

That is the single highest-leverage unanswered question on the branch, and it is
strategic, not numerical.

### 3.3 The alternatives, assessed

**llama.cpp** — runs Mellum correctly today, with working batched prefill,
against five official JetBrains GGUFs including the Q4_K_M that ds4 cannot
execute. For a 9–13 GB model on 32 GB machines — the document's own stated first
deployment target — its capability gap versus ds4 is approximately zero.

**MLX / mlx-lm** — verified via Context7 (`/ml-explore/mlx-lm` v0.31.0): native
quantized MoE via `QuantizedSwitchLinear`, existing `qwen3_moe` model classes,
2/3/4/8-bit plus `mxfp4`/`mxfp8`, mixed-bit recipes via `quant_predicate`, and
`batch_generate` giving batched prefill essentially free. Mellum's local
checkpoint self-describes as `qwen3_moe`, and the genuine deltas
(layer-selective YaRN, 3:1 sliding/full, dual RoPE bases) resemble what
`gpt_oss` already demonstrates in mlx-lm. As a *validation and
quantization-experiment vehicle* it is close to ideal. Throughput claims for
MLX on M5 could not be verified from trustworthy sources and should not be
relied on.

**Ollama** — right answer for *users*, irrelevant as a destination for this
project. Useful as a measuring stick: any ds4-Mellum release should be able to
state what it does that `ollama run` does not.

**vLLM / SGLang** — not credible on Apple Silicon for this use case. Correctly
out of scope.

### 3.4 The steelman for continuing

ds4 is explicitly not trying to be llama.cpp. Its README describes a
deliberately narrow engine where loading, prompt rendering, tool calls, KV
state, the server, and the coding agent are built and tested together.
Mellum-in-ds4 inherits that whole stack; Mellum-in-llama.cpp inherits inference
only. The streaming thesis is architecturally real and Mellum's shape suits it —
1,792 routed experts at ~4.3 MiB each in the proposed quant. And
correctness-first ordering is defensible: you cannot run the Laguna XS
streamed-versus-resident A/B without a trusted resident path.

The weakness of the steelman: none of that requires doing resident bring-up,
batched prefill, and tensor-core kernels *before* testing whether streaming is
needed at all.

## Part 4 — Process and scope

### 4.1 Diagnostic scaffolding has become permanent product surface

**Verified: 13 probe commands plus 7 output/trace companions = 20 `--mellum-*`
flags** (`ds4_cli.c:94–113`), all printed unconditionally in public `--help`
(`ds4_help.c:289–306`), all exported in the public header (`ds4.h:284–349`),
backed by roughly 1,500–1,900 lines of probe-only host code in `ds4.c`.

The gates themselves were sound and several caught real bugs. But nothing plans
to retire any of them, and some are already subsumed —
`--mellum-kv-layout-probe` and `--mellum-session-lifecycle-probe` test proper
subsets of the decode and interactive probes. The document even calls the
per-layer dump "diagnostic scaffolding, not a permanent product dependency"
(line 490) while it ships with help text.

The underlying cause is a design smell worth naming: authentic-model gates need
a real 12.9 GB GGUF, which the synthetic-only `ds4_test` cannot carry, so every
real-model gate became a CLI flag instead of one parameterised debug entry
point.

### 4.2 The deployment artifact is validated nowhere

**Verified.** `ds4_gpu_mellum_routed_moe_one_tensor` — the mixed Q4_K/Q8
primitive matching the intended shipping quant — has **zero callers outside its
own test** (declaration `ds4_gpu.h:2266`, definition `ds4_metal.m:34659`, test
`tests/ds4_test.c:2248`). Everything above the MoE seam is hard-wired Q8-only.

So: the ~9.33 GiB deployment target has never been produced as a file; the
imatrix pipeline has never been run for Mellum; all correctness, quality, and
throughput evidence is for a 12 GiB artifact that by the document's own budget
does not comfortably fit the stated 32 GB target, let alone 16 GB; and when the
mixed artifact arrives it needs new layer compositions and an entirely new
oracle chain, since none of the pinned fixtures apply.

This is the branch's largest unpriced work item, and it has no gate.

### 4.3 Real duplication at the kernel layer

**Verified** — these are near-verbatim duplicates with token-major indexing
added, not shared specialisations:

- `kernel_mellum_router_select_one` (`metal/dsv4_misc.metal:4647`) vs
  `..._batch` (`:4711`) — ~60 duplicated lines including the bitonic sort and
  the `6.103515625e-5` denominator floor, written twice.
- `kernel_mellum_q8_0_pair_swiglu_f32` (`metal/moe.metal:557`) vs `..._batch_f32`
  (`:620`).
- `kernel_mellum_q8_0_down_f32` (`moe.metal:2779`) vs `..._batch_f32` (`:2818`).
- Host compositions: attention decode vs prefill (`ds4_metal.m:34228`/`:34312`),
  layer decode vs prefill (`:34389`/`:34471`).

In each case the batch kernel at `n_tokens = 1` subsumes the one-token kernel.
The mitigation deserves real credit — tests assert batch output is **bitwise
identical** to repeated one-token decode (`tests/ds4_test.c:2404`, router at
`:921`), which is the strongest congruence check available. But it holds only
while both are edited in lockstep.

Separately, the *refusals* to generalise are correct judgement, not tax: the GLM
`% 256` checks describe those kernels truthfully, and Q8 rows genuinely cannot
feed the Q4_K packed-scale kernel. And the engine-owned decode state versus
session state is **not** duplication — `ds4_mellum_session_state`
(`ds4.c:36550`) composes a `ds4_mellum_decode_state*`. That is clean.

### 4.4 Claimed versus actually complete

| Gate | Document's claim | Verified state |
| --- | --- | --- |
| Step 1: reference contract | Complete | **Complete.** Fixtures, SHA-256s, capture patch committed |
| Step 2: loader | "Complete… malformed coverage is not" | **Partial, as caveated.** Zero negative fixtures exist despite the gate text requiring them |
| Step 2a: tokenizer/chat | "Complete for canonical fixture" | **Partial.** 89-line test; no punctuation/contraction cases the gate names |
| Step 2b: numerical oracle | Complete for top-10 | **Complete as scoped** |
| Step 3: primitives | Complete | **Complete and well-tested** (synthetic) |
| Step 4: resident decode | Green | **Complete for Q8.** Router parity final-token only |
| Step 5: resident prefill | In progress | **Blocked.** 2 of 4 named attention parity cases missing |
| Mixed-quant artifact | "Not yet an executable artifact" | **Honest — and unstarted.** Test-only dead code |
| SSD streaming | Deferred | Deferred ≥8 times; no kernels |

The honesty verdict matters: **the document does not claim completeness it
lacks.** Every gap the reviewers verified was disclosed in-line. The failure
mode is subtler — disclosed debt with no mechanism forcing repayment, and no
owner or checklist. Deferred three times is how such items become permanent.

### 4.5 The 7.8× claim, and the performance targets

The 14.8 → 111.5 tok/s baseline submitted one command buffer per kernel and
round-tripped the hidden state GPU→CPU→GPU between **every one of 28 layers**.
That is a self-inflicted wound; removing it is table stakes, not a 7.8×
optimisation. The document effectively concedes this ("the first 14.8 t/s agent
trace was not a model limit"), but the framing survives into the summary.

Beyond that the numbers are hygienic and mutually consistent — 111.5 microprobe,
~100 agent path, 117.6/111.6 warmed profile — and the byte-identical-logits
check on the optimisation is exemplary. The output-head profiling (0.459
ms/token, ~5%) correctly redirected effort.

The ≥500 and 1,200–1,500 tok/s prefill targets are extrapolated from a paper
measured on an **M5 Pro** running **Qwen3-30B-A3B Q4** — different model,
quantization, and chip. The document labels them as targets rather than
measurements, which is correct, but the "500 tok/s acceptance floor" should not
be treated as grounded.

### 4.6 The document has outgrown its form

At 1,371 lines it is simultaneously design doc, decision log, test report,
changelog, and plan. A reader asking "what works today?" must reconstruct it
from five stacked "Status update" blocks inside Step 3, gate text stranded 400
lines after the updates that supersede it, and a research log that partly
repeats the narrative.

Most tellingly, the six Open Questions are **untouched by 27 commits**.
Question 1 in particular: the local GRPO checkpoint — described at line 9 as
"the exact downloaded weights we ultimately want to run" — has been silently
displaced. Every artifact, fixture, and measurement on this branch is the public
Thinking release. That should be a recorded decision, not an ambient drift.

## Part 5 — What the work gets genuinely right

Stated plainly, because the critical findings above should not obscure it:

1. **Oracle provenance** is better than most production inference engines:
   SHA-256'd fixtures, the capture patch committed *alongside* them, a
   NaN/Inf-rejecting comparator with its own meta-test, and explicit
   "do not relax this envelope silently" language.
2. **The YaRN double-mscale catch** — red gate at 27.8 RMS, localised to layer
   3, root-caused to an exact 1.2772589 factor, fixed, re-verified. The staged
   process paying for itself.
3. **The tokenwise re-capture was methodologically correct.** Recognising the
   first layer-27 comparison was schedule-confounded and pinning a
   matching-schedule reference, rather than hand-waving a 2.18 RMS gap, is
   exactly right — and it is what made the reframing in §2.2 computable at all.
4. **Boundary enforcement is real, not decorative.** Inspect-only guards,
   layout-only sessions rejecting every selection API (`ds4.c:51969`),
   agent-only capability gating, bit-identical two-session isolation, and the
   prefill ring-aliasing rejection tested at the exact 17/18 boundary.
5. **Primitive-first testing with independent scalar references**, including
   adversarial and tied router logits and the deliberately non-256-divisible 288
   width. By elimination this correctly narrows the drift source to attention
   and dense projections — a conclusion the document never draws but its own
   tests support.
6. **Negative results are recorded as prominently as positive ones.** This is a
   lab notebook, not a press release, and that is why an adversarial review
   could be this productive.

## Part 6 — Recommended sequence

Ordered by information gained per hour, not by plan order.

1. **Precision and equivalence ablation (§1.1, §1.2).** Chunk-size-1 prefill,
   then prefill projections through `rows_exact`. Hours, no new machinery,
   maximum discrimination. Either outcome retires the "unresolved accumulation"
   framing.
2. **Expert-selection audit (§1.3).** Per-position, per-layer top-8 IDs and
   weights for both schedules at 26 and 1,030 tokens; first-flip location and
   flip counts. This determines what kind of gate is even possible: zero flips
   means an RMS envelope is meaningful; nonzero means parity gates must become
   behavioural (top-k rank agreement, greedy-token agreement).
3. **Answer Open Question 5 empirically.** Run the official Q4_K_M under
   llama.cpp on real 16 GB and 32 GB hardware. One afternoon. It determines
   whether ds4-Mellum has a differentiator, and therefore whether steps 4–6 are
   worth doing at all. This is out of order relative to the current plan
   deliberately.
4. **Re-frame the reference work rather than dropping it.** Capture llama.cpp
   at positions 1,023/1,029 in **both** schedules, but gate ds4-batched against
   llama.cpp-**batched** in *relative* terms, calibrated by the reference's own
   3.5% schedule spread. Add one HF Transformers FP32 CPU forward from the
   pinned safetensors to break the circularity.
5. **Consolidate the probe surface.** One `--mellum-probe NAME [--out FILE]`
   dispatcher, hidden from default help. Delete `--mellum-kv-layout-probe` and
   `--mellum-session-lifecycle-probe` outright. Consider deleting the one-token
   router/SwiGLU/down kernels and dispatching the batch kernels at
   `n_tokens = 1` — the existing bitwise tests prove this is safe by
   construction, and it halves the divergence surface before the mixed-quant
   variants double it again.
6. **Write a gate for the deployment artifact,** or explicitly retire the mixed
   quant in favour of the official Q4_K_M plus Q5_0 kernels. Today the shipping
   artifact does not exist and has no oracle.
7. **Split the document** into a short current-state page plus the chronological
   log, and either answer or strike the six Open Questions.

**Single most likely permanent-stall mode**, and both the process and strategic
reviewers converged on it independently: the branch remains a beautifully
validated Q8 diagnostic harness for a 12 GiB artifact that fits only the
development machine, while the actual product artifact never gets built —
because every incremental reward accrues on the Q8 path, and the mixed path
requires redoing the oracle chain from scratch.

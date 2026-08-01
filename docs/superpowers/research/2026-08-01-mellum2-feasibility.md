# Mellum 2 feasibility for DwarfStar

Research begun 2026-08-01 on branch `mellum-2.1`, based on the Laguna XS 2.1
SSD-streaming line at `bcf1b1c`.

The working conversation calls the target "Mellum 2.1". JetBrains' public
artifacts and technical report call the architecture **Mellum 2**. This note
uses *Mellum* for the architecture, *public Thinking release* for the released
JetBrains checkpoint, and *local GRPO checkpoint* for the exact downloaded
weights we ultimately want to run.

## Decision summary

Mellum is a credible DwarfStar target. Resident Metal inference should be a
bounded new-family port: the model uses operations ds4 already has in some
form, and both a canonical llama.cpp implementation and official GGUF files
exist.

Do not implement Mellum as a Laguna variant. The two families share useful
primitives, but differ in layer schedule, router semantics, MLP topology, and
published quantization layout. Mellum should have its own family, shape,
metadata validator, tensor binding, and graph entry points.

Bring up resident inference before SSD streaming. The public Q8_0 GGUF is the
best first correctness artifact because ds4 already supports Q8_0. The public
Q4_K_M file contains Q5_0 expert-down tensors, which ds4 can parse as GGUF but
cannot currently execute on Metal. SSD streaming can reuse the cache lifecycle,
but not the existing Laguna XS streamed expert kernels unchanged.

Keep MTP out of the initial scope. The report describes a one-head MTP draft
model, but neither the local GRPO checkpoint nor the public Thinking release
contains MTP tensors.

## Pinned evidence

| Source | Revision or identity | What it establishes |
| --- | --- | --- |
| [Mellum 2 Technical Report](https://arxiv.org/abs/2605.31268) | arXiv `2605.31268v1`, 2026-05-29 | Intended architecture, training, layer-selective YaRN, MTP design |
| Local GRPO checkpoint | HF cache snapshot `572fe3b9aa78f1476b68eb6017043fb0688bb278` | Exact local weights under investigation |
| [Public Thinking release](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking) | `ba4838faad89e968c36f39e76e95319d756714fe` | Canonical Mellum config and tokenizer |
| [Public Q4_K_M GGUF](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking-GGUF-Q4_K_M) | repo `71a489e7b95efacf89feaaa6fe3b2995f3542409`; file SHA-256 `489cf0d7ca86ef4683e34e2efe8a46a9b52573bfe994a6ad9c86bf57c7173ccb` | Canonical GGUF metadata, tensor names, and published mixed quantization |
| Local llama.cpp | `0e4a0362239713ea95a6864a17a8de4b0ad90d62` | Executable Mellum graph and GGUF reference |
| DwarfStar XS base | `bcf1b1c2773a7cbaddd54416482a8317d2a0a8bc` | Existing Laguna XS resident and SSD-streaming machinery |

The local GRPO snapshot contains five safetensor shards and 24,299,846,144
bytes of tensor payload. Its index has 5,631 tensor entries. The public
Thinking release has the same tensor names, shard assignment, shard sizes, and
total payload size, but different shard hashes. The public release is therefore
an architecture and packaging oracle, not a behavior oracle for the local
weights.

## Canonical architecture

The released config identifies `MellumForCausalLM` with model type `mellum`.
The local GRPO snapshot instead identifies `Qwen3MoeForCausalLM` with model type
`qwen3_moe`. That local config is an older compatibility export and is not a
complete statement of Mellum inference semantics.

The fixed shape is:

| Property | Value |
| --- | ---: |
| Layers | 28 |
| Hidden width | 2,304 |
| Vocabulary | 98,304 |
| Query heads | 32 |
| KV heads | 4 |
| Head dimension | 128 |
| Query projection width | 4,096 |
| Expert count | 64 per layer |
| Active experts | 8 per token |
| Expert hidden width | 896 |
| Routed expert projection tensors | 5,376 (`28 x 64 x 3`) |
| Sliding window | 1,024 tokens |
| Maximum context | 131,072 tokens |

Every layer is sparse. There is no leading dense layer and no shared expert.
Each layer has a standard pre-attention RMS norm, Q/K per-head RMS norms,
GQA, an attention residual, a pre-MoE RMS norm, routed SwiGLU experts, and an
MoE residual.

### Attention and RoPE

The layer pattern is:

```text
sliding, sliding, sliding, full
```

repeated seven times. Full-attention layers are therefore `3, 7, 11, 15, 19,
23, 27`. This is the opposite phase from Laguna XS, whose four-layer groups
begin with the global layer.

The two attention types also use different rotary behavior:

- Full-attention layers use YaRN with theta 500,000, factor 16, original
  context 8,192, beta-fast 32, beta-slow 1, and attention factor
  1.2772588722239782.
- Sliding-attention layers use ordinary, unscaled RoPE with theta 500,000.

The report is explicit that YaRN is applied only to full-attention layers.
The canonical Transformers implementation constructs separate masks and
rotary embeddings for the two layer types. The older `qwen3_moe` config in the
local snapshot flattens the RoPE settings and cannot express this distinction
faithfully under stock Qwen3-MoE inference.

### Router

For every token and layer:

1. Compute 64 bias-free router logits.
2. Apply softmax across all 64 logits in FP32.
3. Select the top eight probabilities.
4. Renormalize those eight selected probabilities to sum to one.
5. Apply the selected weights to the eight SwiGLU expert outputs and sum them.

There is no expert bias, shared expert, or Laguna-style expert weight scale.
The current Laguna graph calls a router helper that requires a bias tensor and
passes an expert scale. A Mellum path needs a bias-free form with scale 1.

### MTP

The technical report says pre-training and post-training retain a single MTP
head. The released and local weight indexes both contain zero tensors named for
MTP, next-token prediction, or a draft module. The released runtime graph also
has no MTP block. Treat this as a report-versus-release packaging distinction;
do not infer missing weights or include speculation in the first port.

## GGUF findings

JetBrains publishes Mellum Thinking GGUFs in BF16, Q8_0, Q6_K, Q4_K_M, and
MXFP4_MOE. This removes the earlier assumption that a new converter is needed
before loader bring-up.

The public Q4_K_M file is 8,071,295,040 bytes and contains 339 tensors. Its
relevant metadata uses the `mellum.*` namespace and records the full shape,
sliding-window pattern, YaRN parameters, and separate SWA RoPE base.

The expert layout is not uniform:

- All expert gate and up tensors are Q4_K.
- Fourteen expert down tensors are Q8_0.
- Fourteen expert down tensors are Q5_0.

The 896-element expert width is not divisible by the 256-element K-quant block,
so the expert-down projections cannot all use Q4_K/Q6_K. This invalidates the
initial rough estimate based on an ideal four-bit payload and prevents direct
reuse of the XS Q4/Q6 streamed address-table kernels.

The GGUF parser knows Q5_0's block geometry, but `DS4_TENSOR_Q5_0` and Metal
Q5_0 matmul dispatch are absent. The official Q4_K_M artifact is therefore not
the smallest-risk first inference target. Start with the official Q8_0 GGUF;
add mixed Q4/Q5/Q8 support only after architecture correctness is established.

### Token termination discrepancy

The tokenizer defines:

| Token | ID |
| --- | ---: |
| `<|endoftext|>` | 0 |
| `<|im_start|>` | 27 |
| `<|im_end|>` | 28 |

The HF model config says `eos_token_id = 0`, while the tokenizer config names
`<|im_end|>` as EOS and the official GGUF records EOS ID 28. DwarfStar should
follow the GGUF/tokenizer contract for the GGUF runtime, but this disagreement
must be captured in the golden generation fixture rather than decided by
assumption.

## Step 1 reference artifact

Completed 2026-08-01. The initial correctness oracle is the pinned public
Thinking Q8_0 GGUF, kept in the Hugging Face cache rather than copied into the
worktree:

```text
/Users/pauleveritt/.cache/huggingface/hub/models--JetBrains--Mellum2-12B-A2.5B-Thinking-GGUF-Q8_0/snapshots/6f5b0031c9ea37740f630362d3c06c54933fc2f4/Mellum2-12B-A2.5B-Thinking-Q8_0.gguf
```

It was fetched with `hf download` at revision
`6f5b0031c9ea37740f630362d3c06c54933fc2f4`. The 12,925,417,536-byte file has
SHA-256 `d4049c2599796d18245523818c0534e8f8605166fc58c60dd0753547facdb3a2`.
The cache snapshot is a symlink to the content-addressed blob carrying that
same digest.

### GGUF contract

The artifact identifies itself as `Mellum2 12B A2.5B Thinking`, architecture
`mellum`, GGUF quantization version 2, and file type 7 (`Q8_0`). Its 339
tensors are exactly 198 `Q8_0` tensors and 141 `F32` tensors: there are no
Q5, K-quant, or other execution types in this bring-up artifact.

The parsed `mellum.*` metadata confirms the fixed contract described above:
28 blocks; context 131,072; embedding 2,304; routed feed-forward width 7,168
(eight active 896-wide experts); 32 query and four KV heads; 128-element K/V
heads; 64 total and eight selected experts; and a 1,024-token sliding window.
It also records YaRN factor 16, original context 8,192, attention factor
1.2772589, beta-fast 32, beta-slow 1, and both ordinary and SWA RoPE bases at
500,000.

The tokenizer contract is `gpt2` with pre-tokenizer `mellum2`, BOS ID 0, and
EOS ID 28. The embedded chat template renders a single user message with
thinking disabled as:

```text
<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

For `Answer with one word: the capital of France is`, this is the 24-token
sequence:

```text
[27, 1397, 233, 5698, 434, 768, 1574, 60, 302, 5782, 332, 8438,
 359, 28, 233, 27, 8091, 233, 23, 233, 233, 24, 233, 233]
```

This tokenization was cross-checked with the local snapshot's matching
`mellum2` tokenizer. It must be rechecked directly against DwarfStar's GGUF
tokenizer implementation when loader support lands; it is a tokenizer fixture,
not evidence that the local GRPO weights have public-release behavior.

### Greedy generation baseline

Reference runtime: local llama.cpp `0e4a0362239713ea95a6864a17a8de4b0ad90d62`
(`b10154-0e4a03622`) on the Apple M5 Max, all layers GPU-offloaded. Each run
used the embedded chat template, `--single-turn --reasoning off`, context 512,
`--temp 0`, seed 123, and `--predict 16`; no sampling penalty or stochastic
sampler was enabled. The observed continuations were:

| User message | First greedy token | 16-token continuation |
| --- | --- | --- |
| `Complete this Python function: def add(a, b):` | `def` (ID 910) | `def add(a, b):\n    """Return the sum of a` |
| `Write only the JavaScript body of function square(x).` | three backticks (ID 889) | `&#96;&#96;&#96;javascript\nreturn x * x;\n&#96;&#96;&#96;` |
| `Answer with one word: the capital of France is` | `Paris` (ID 50195) | `Paris` |

One-token reruns matched those three first-token values. The local CLI build
does not expose a top-logit/probability dump or a standalone server binary, so
this step records exact first-token IDs and continuations but **not** numeric
logits. Add a logits-capable llama.cpp build or a minimal reference evaluator
before using a logit tolerance as a DwarfStar gate.

llama.cpp terminates its generated chat response on the GGUF EOS ID 28. It
does not configure ID 0 as EOS. Keep both IDs in DwarfStar's generation test
matrix: 28 is the public GGUF runtime terminator; 0 remains the conflicting HF
model-config value and must not silently become the GGUF stop condition.

## Reuse assessment

### Useful existing pieces

- GGUF mmap loading and metadata parsing.
- Q8_0, Q4_K, and dense F32/F16 tensor operations.
- Laguna's GQA attention structure, per-head Q/K RMS norm, NeoX RoPE, mixed
  full/SWA scheduling, and layer-specific KV capacity.
- Routed top-k MoE buffer management and general Metal command orchestration.
- The generic streamed expert-cache lifecycle: lookup, load, eviction,
  address buffers, and accounting.
- Laguna XS resident-versus-streamed A/B methodology and quality fixtures as a
  process template.

### Pieces that cannot be reused unchanged

- Laguna model-family detection, fixed shape, metadata names, and tensor
  binding.
- Laguna's `[full, sliding, sliding, sliding]` layer phase.
- Its router helper, which expects a per-expert bias tensor and a scale.
- Its leading-dense and shared-expert branches.
- The fused routed/shared MoE kernel. It requires a shared expert, restricts
  the streamed down type to Q4_K/Q6_K, and rejects an expert width such as 896.
- The current XS streamed expert address-table kernels, which do not support
  the published Mellum Q5_0/Q8_0 down-projection mix.

The likely clean design is a separate Mellum graph that reuses low-level
attention, norm, matmul, and cache primitives. It should not accumulate
family conditionals inside `laguna_graph_forward_*`.

## Memory and SSD relevance

The public Q4_K_M model is about 7.52 GiB. The public Q8_0 model is reported as
12.9 GB. Resident Q4 should fit comfortably on a 16 GB machine at modest
context once its quant types are supported. Q8 is intended only as the first
correctness artifact on the 128 GB development machine.

With an optimized per-layer KV policy, 128K context is less expensive than a
naive all-layer allocation suggests. A BF16 K+V row is 2,048 bytes per layer
and token. Seven full layers at 131,072 tokens consume about 1.75 GiB; the 21
sliding layers need only about 42 MiB for 1,024-token rings. This excludes graph
scratch and other runtime state, but it makes resident Q4 at long context
plausible enough that SSD streaming should be justified by measurement rather
than assumed necessary.

SSD streaming remains interesting for extra memory headroom, smaller machines,
or larger caches and application state. Its feasibility is conditional:

- Positive: only 1,792 routed experts exist, and the cache supports 64 total /
  top-8 routing comfortably.
- Negative: the published Q4 expert-down layout needs Q5_0 and Q8_0 compute
  from streamed addresses, which the XS cache kernels do not implement.
- Unknown: the expert locality and cache size needed to beat mmap fallback have
  not been measured for Mellum.

## First implementation steps

### Step 1: Freeze the target contract and reference outputs

Use the public Q8_0 GGUF for initial bring-up while retaining the local GRPO
checkpoint as the final weight target.

Capture:

- complete GGUF metadata and tensor-type inventory;
- tokenizer IDs and exact chat-template rendering;
- deterministic llama.cpp tokenization and short greedy continuations;
- a small first-token/logit reference set;
- the expected behavior for both EOS IDs seen in the artifacts.

**Gate:** one pinned reference manifest and a compact golden fixture that can
be run before ds4 produces tokens.

### Step 2: Add loader-only Mellum support

Add a Mellum family and fixed shape, then implement:

- `mellum.*` metadata validation;
- the 339-tensor GGUF binding contract;
- tensor layout validation for the Q8_0 bring-up artifact;
- `--inspect` support without constructing a Metal graph.

Keep loading mmap-backed and do not alter the Laguna, GLM, or DeepSeek family
paths.

**Gate:** `ds4 --inspect` accepts the pinned Mellum Q8_0 GGUF and focused tests
reject incorrect layer count, expert shape, head shape, SWA pattern, and RoPE
metadata.

### Step 3: Bring up resident one-token Metal decode

Implement the smallest Mellum decode graph with:

- Q/K per-head RMS normalization before RoPE;
- correct per-layer full/SWA choice and dual RoPE parameters;
- four-KV-head GQA;
- bias-free softmax/top-8/renormalized routing;
- routed SwiGLU with no shared expert;
- final norm and output projection.

Use Q8_0 first so architecture work is not mixed with a new quant kernel.

**Gate:** the first greedy token matches llama.cpp on several prompts, and any
logit or intermediate drift is measured and explained rather than hidden by a
sampling comparison.

### Step 4: Add resident prefill and KV correctness

Add the batch graph and layer-specific KV policy. Validate both sides of the
1,024-token SWA boundary and ensure only full layers retain the entire context.

**Gate:** short continuation parity, a prompt crossing 1,024 tokens, and a
longer chunked-prefill run all agree with the reference within an explicit
numerical tolerance.

### Later: choose the release quant and SSD path

After resident Q8 is green, make a measured choice:

1. Add resident Q5_0 and support the official Q4_K_M artifact as published; or
2. Produce a more uniform Mellum GGUF intended for ds4 and validate its quality.

Only then extend SSD streaming for the chosen down-projection type and run the
same resident-versus-streamed A/B gate used for Laguna XS.

## Open questions

1. Is the local GRPO checkpoint the final intended product target, or should
   the public Thinking release become the shipped default after architecture
   bring-up?
2. What generated the local checkpoint's flattened `qwen3_moe` config, and is
   applying the canonical Mellum config to those weights the intended runtime
   contract?
3. Should the first release artifact preserve the official Q4_K_M mixture, or
   favor a uniform layout designed for DwarfStar streaming?
4. Which EOS behavior matches JetBrains' evaluation and serving setup: ID 0,
   ID 28, or both?
5. Does resident Q4 plus layer-specific KV already satisfy the desired 16 GB
   target, making SSD streaming optional rather than required?
6. What hit rate and cache footprint does Mellum's 8-of-64 routing produce on
   coding-agent workloads?

## Research log

### 2026-08-01

- Created the `mellum-2.1` worktree from `laguna-xs2.1`.
- Located and inspected the local GRPO safetensor checkpoint.
- Compared its config and tensor index with the public Thinking release.
- Verified the canonical per-layer attention and RoPE behavior in the Mellum
  Transformers implementation and the technical report.
- Verified local llama.cpp has a dedicated Mellum model graph.
- Inspected the published Q4_K_M GGUF header and tensor types without
  downloading the full file.
- Revised the feasibility assessment: resident support remains favorable;
  SSD streaming needs new mixed-quant expert compute support.
- Downloaded, checksummed, and inspected the pinned public Q8_0 GGUF.
- Captured its metadata, chat-template tokenization, deterministic greedy
  continuations, first-token IDs, and explicit EOS behavior.
- Added Mellum loader-only support: `--inspect` validates the public Q8_0
  metadata, 28-layer SWA schedule, and all 339 tensor bindings without
  building a Metal graph. Ordinary execution remains explicitly rejected until
  the Mellum decode graph lands.

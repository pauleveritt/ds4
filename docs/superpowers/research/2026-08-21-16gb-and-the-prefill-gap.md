# The 16 GB question, and the prefill gap

Measured 2026-08-21 on Apple M5 Max, 137 GB, against llama.cpp `0e4a0362`
(the revision the Mellum fixtures are pinned to).

Two questions were open. **Does Mellum need SSD expert streaming to run on a
16 GB machine?** — the adversarial review's strategic finding 3.2 called this
the single highest-leverage unanswered question on the branch. And **how far is
ds4's Mellum path from llama.cpp?**

This note records the measurements and, deliberately, the ones that did not
hold up under review. Claims are marked **verified**, **bounded** (the
measurement constrains the answer but does not settle it), or **untested**.

## Pinned artifacts

| Artifact | Identity |
| --- | --- |
| Official Q4_K_M GGUF | repo `71a489e7b95efacf89feaaa6fe3b2995f3542409`, file SHA-256 `489cf0d7…73ccb` |
| Official Q8_0 GGUF | repo `6f5b0031c9ea37740f630362d3c06c54933fc2f4` |
| llama.cpp | `0e4a0362`, Metal, `-ngl 99` |

## The tensor inventory, and what it costs ds4

**Verified** by `GGUFReader` over the pinned Q4_K_M:

| Count | Type | Where |
| ---: | --- | --- |
| 155 | Q4_K | expert gate/up, attention projections, token embeddings |
| 141 | F32 | norms |
| 15 | Q6_K | output head |
| 14 | Q8_0 | expert down, 14 layers |
| 14 | Q5_0 | expert down, the other 14 layers |

The feasibility document's claim that the official artifact contains Q5_0
expert-down tensors ds4 cannot execute on Metal is confirmed — and it is the
*smaller* half of the problem. Everything above the MoE seam in ds4's Mellum
graph is hard-wired Q8-only (adversarial review §4.2), while this file is Q4_K
for attention projections and embeddings and Q6_K for the output head. Running
the official artifact in ds4 is therefore a Q4_K/Q6_K/Q5_0 kernel project, not
a fourteen-tensor patch. Whether that is cheaper or dearer than building the
branch's custom mixed quant — Q4_K only inside the MoE seam, Q8 elsewhere,
which exists precisely to bound this surface — is **untested and genuinely
open**.

## Footprint: the official Q4_K_M under a memory ceiling

Memory was constrained with a ballast process holding physical pages, leaving a
target amount free. Prompt ~800 tokens, `-n 64`, one run per cell.

| free RAM | ctx | peak RSS | prefill | decode | swap |
| ---: | ---: | ---: | ---: | ---: | --- |
| 137 GB | 4K | 8.24 GB | 2,036 t/s *(cold)* | 193.6 t/s | none |
| 15.9 GB | 4K | 8.23 GB | 4,302 t/s | 192.4 t/s | none |
| 15.9 GB | 32K | 8.77 GB | 4,436 t/s | 196.7 t/s | none |
| 15.9 GB | 131K | 10.18 GB | 4,108 t/s | 196.4 t/s | none |
| 10.9 GB | 4K | 8.35 GB | 4,401 t/s | 197.3 t/s | none |
| 10.9 GB | 131K | 10.18 GB | 4,437 t/s | 197.9 t/s | none |

**The footprint numbers are verified and they transfer.** Full 131,072-token
context costs 10.18 GB resident, only 1.9 GB more than 4K, because llama.cpp's
SWA-aware cache caps the 21 sliding layers at their 1,024-token window. The
arithmetic cross-checks: 7 full-attention layers x 131,072 tokens x 2 KB/token
= 1.88 GB, against the 1.83–1.95 GB observed.

**The throughput numbers do not transfer, and the pressure test is invalid.**
Both failures are worth recording:

- The ballast allocated zero-filled pages. Zero pages compress to nearly
  nothing, so under real pressure macOS squashed the *ballast* rather than
  evicting the model — an escape valve a real 16 GB machine does not have.
  "Zero swap at 10.9 GB free" is evidence the ballast deflated, not that the
  model survives pressure. A valid rerun needs incompressible ballast and a
  minutes-long steady-state decode, since a 2.4-second generation cannot show
  the SSD re-read stalls that eviction would cause.
- A ballast constrains capacity, not bandwidth. This is an M5 Max at roughly
  5x the memory bandwidth of the base-M-series chips actual 16 GB Macs carry,
  and decode is bandwidth-bound. Scaled by bandwidth, target-hardware decode
  is plausibly 40–60 t/s. Real 16 GB Macs also default to a Metal wired-memory
  limit near 10–11 GB, which makes a 10.18 GB working set **marginal by
  default**, not comfortable.
- `-c 131072` allocates the KV buffer; the prompt was ~800 tokens, so attention
  depth was never exercised. Decode at position 100K over 7 full-attention
  layers will be slower. **Untested.**
- Every cell is n=1, and the 137 GB row's 2,036 t/s is a cold-cache artifact
  (warm rerun: 4,368 t/s). The branch's own Phase 0 lesson — same-session,
  interleaved, n>=3 — was not applied here.

### What this settles about Open Question 5

**Bounded, not settled.** At a 16 GB target, streaming is unnecessary *by
footprint*: the artifact fits, at maximum context, with room. Below roughly
12 GB it does not fit and streaming or a much smaller quant is required. The
throughput case at 16 GB is open until someone runs a base-M-chip machine.

The question was previously closed by intent rather than measurement — "the
deployment target is *below* 16 GB, so resident Q4 cannot fit and SSD expert
streaming is required." That premise is **weakened at exactly 16 GB and intact
below it**, and the resolution note in the adversarial review should be read
that way rather than as a refutation.

## The gap: ds4 against llama.cpp at matched quantization

The first comparison drawn here was ds4-Q8 against llama.cpp-Q4, which flatters
llama.cpp on decode. Rerun at matched Q8 on the same pinned file:

| | ds4 (post-dot4) | llama.cpp Q8_0 | ratio |
| --- | ---: | ---: | ---: |
| Decode, with output head | 144 t/s | 150–155 t/s | ~1.05x |
| Prefill @ ~1K tokens | 266 t/s | ~4,300 t/s | **~16x** |

**Decode is at parity** — the 5–7% difference is inside the ~5% run-to-run
variance Phase 0 established. There is nothing to chase there, and the
feasibility document already says a tensor-core project should not be judged by
decode speedup.

**Prefill is flat in llama.cpp across length**, so 4,000 t/s is the standing
target rather than a short-prompt artifact:

| Prompt chars | ~tokens | llama.cpp Q8 prefill |
| ---: | ---: | ---: |
| 2,929 | ~800 | 4,341 t/s |
| 9,493 | ~2,600 | 3,955 t/s |
| 18,986 | ~5,200 | 4,130 t/s |
| 27,347 | ~7,000 | 4,057 t/s |

## Why the gap exists, and what the ceiling is

The branch already measured the cause: the routed MoE is **96.8% of prefill
time with zero weight reuse across tokens**. ds4's prefill is structurally
decode-repeated-N-times for the MoE.

Per token *per layer* the MoE touches 8 experts x 3 matrices x 2,064,384
weights = 49.5M weights, about 52 MB at Q8 — and **every one of the 28 layers is
sparse**, so the per-token figure is ~1.47 GB. At 266 t/s that is **~392 GB/s**,
which essentially saturates an M5 Max. The kernel is bandwidth-bound, exactly as
the feasibility document's short-circuit measurement concluded ("the MoE is
96.8% bandwidth-bound, so the win is traffic, not FLOPs").

*A first draft of this note divided by layer and reported ~14 GB/s, concluding
ds4 was overhead-bound rather than bandwidth-bound. That was arithmetic error
and the conclusion drawn from it was wrong; it is recorded here because it
briefly redirected the plan toward the wrong fix.* Reductions, occupancy, and
scalar overhead are real secondary costs — the down-kernel discriminator puts
`mid` re-reads at 38% of that kernel and weight loads, scalar arithmetic, and
the reduction tree at the other 62% — but traffic is the wall.

Group expert-major over a 1,024-token chunk and each of 64 experts sees ~128
tokens, so weights are read once per chunk rather than once per token — a
~128x traffic cut that turns the MoE from repeated GEMV into GEMM, which is the
regime llama.cpp is already in.

| Step | Result | Bitwise oracle survives? |
| --- | --- | --- |
| Expert-major grouping, as built | bit-identical, **neutral** | yes |
| Row-tiled down projection, R=2 | **measured +14.1% mean, +9.9% median** | yes, order-preserving F32 |
| Row-tiled down projection, R=4 | **regression** — 32 KiB threadgroup memory costs occupancy | yes |
| Real expert GEMM: token x row tiling, FP16 tiles, F32 accumulate | 500–1,200 t/s plausible, unmeasured | **no** |

Two corrections to earlier drafts of this plan belong here.

**Row-tiling does not get 4.5x.** It gets about ten percent, and that was
predictable from the branch's own discriminator: `mid` re-reads are 38% of the
down kernel, the down kernel is ~59% of MoE time, and MoE is 96.8% of prefill,
so the ceiling on an order-preserving row tile is ~12% end-to-end. The measured
+9.9% median lands exactly there. See the section below.

**The grouped kernel's neutral result does not disprove expert-major GEMM.**
What is built is not a GEMM: it stages one `(expert, row)` weight row and then
walks that expert's tokens *serially*, running a full 256-lane tree reduction
per token. It trades weight traffic for serialization, which is why it came out
neutral. A genuine GEMM — tiling across tokens *and* output rows, weights held
in registers or threadgroup memory, FP16 tiles with F32 accumulation — is
untested, and it is the only step with a credible path past ~300 t/s.

### The decision hiding in step three

llama.cpp reaches ~4,000 t/s **by dequantizing weight tiles to half** — the
same `mul_mm`/NAX path this session's review identified as the source of ds4's
batched-prefill drift. Matching llama.cpp's prefill means adopting the
precision policy ds4 deliberately rejected, and the bitwise
batch-equals-decode oracle cannot survive it.

The FP32 oracle work argues the trade is defensible. llama.cpp's half-tile
prefill sits 2.4% from the FP32 model on logits and picks the same greedy token,
while ds4's bit-exactness buys agreement with llama.cpp at 0.085% — 28x finer
than the reference's own error against truth. So the thing being given up is a
self-consistency check far finer than the model's own quantization noise.

**That is not a licence to call 2.4% a universal quality floor.** It is one
26-token prompt, and this branch has already recorded a change well inside that
envelope — dot4 — altering greedy continuation on one prompt of three. FP16
tiles with F32 accumulation are transient arithmetic rounding rather than a
second weight quantization, which is a real argument in their favour, but the
release decision should rest on behavioural evidence over many prompts: tool-call
validity, task completion, teacher-forced KL and top-k agreement, router top-8
stability, and long-context behaviour. Keep exact F32 as a diagnostic oracle;
do not make it a release constraint, and do not retire it.

Third-party corroboration that ~4,000 t/s is hardware-allowed rather than a
llama.cpp trick: the BaseRT M5 study reports 2,478–3,907 t/s prefill for
Qwen3-30B-A3B Q4 with tensor-core kernels on an M5 **Pro** — a larger model on
a smaller chip.

## Measured: the row-tiled down projection

`kernel_mellum_q8_0_down_batch_rowtile{2,4}_f32`, behind
`DS4_MELLUM_DOWN_ROWTILE=2|4` (0 or unset keeps the one-row kernel).

The batch kernel spends one 256-lane tree reduction — eight threadgroup
barriers — per `(row, token)`, while each lane contributes a single
four-element partial: `mid_dim` is 896 and the dot4 helper strides
`ntg * 4 = 1024`, so lanes 224..255 contribute nothing and the rest iterate
once. Tiling R rows into one threadgroup reduces all `R * n_expert_used` lanes
in a single tree pass, cutting barriers per row by R, and the R rows read the
same slot activations concurrently.

The arithmetic is untouched by construction: same lane, same elements, same
256-lane tree, per-slot scalars summed in slot order. `./ds4_test
--metal-kernels` **exits 0 under `DS4_MELLUM_DOWN_ROWTILE` of 0, 2 and 4**,
including `Mellum Q8_0 batch routed MoE exact rows=3`, the bitwise
batch-equals-decode regression. Exit codes were read directly, not through a
pipe.

`--mellum-resident-profile`, prefill at the 1,024-token sliding-window cap,
interleaved across configurations, one process per cell:

| Round | R=0 | R=2 | delta |
| --- | ---: | ---: | ---: |
| 6 | 177.0 | 216.4 | +22.3% |
| 7 | 177.6 | 225.7 | +27.1% |
| 8 | 212.1 | 217.8 | +2.7% |
| 9 | 194.9 | 214.1 | +9.9% |
| 10 | 189.8 | 206.3 | +8.7% |

**Mean +14.1%, median +9.9%, 5 of 5 positive** — and 9 of 10 positive counting
an earlier, noisier sweep. R=4 is a **regression** (max 169.7 t/s against R=0's
174.5 in the same sweep): `R * n_expert_used * 256` floats is 32 KiB at eight
experts, which is at the device limit and costs occupancy.

The gain is exactly what the branch's own numbers predicted: `mid` re-reads are
38% of the down kernel, the down kernel is ~59% of MoE time, and MoE is 96.8%
of prefill, so an order-preserving row tile caps out near 12% end-to-end.
**This closes the order-preserving envelope.** Nothing further is available
without reordering the reduction, because with 896 elements and a 256-lane
tree the reduction structure *is* the kernel, and that structure is what the
bitwise oracle pins.

### A note on measurement conditions

The first sweep produced unusable numbers — R=0 alone drifted 217 to 170 to 119
t/s across three rounds — because Spotlight was indexing 35 GB of freshly
written GGUFs alongside the usual desktop load, at load average 7. No thermal
warning was recorded. The rounds above were taken after that subsided; treat
any single-round ds4 throughput figure taken on a busy desktop as worthless,
which is Phase 0's lesson arriving a third time.

## What would actually have to happen

Parity is reachable and the ceiling is hardware-set, not design-set. It costs
one bounded piece of work (row-tiling, to ~1,200 t/s) and one substantial one
(half tiles and tensor ops, forfeiting the bitwise oracle). Neither closes the
separate fact that ds4 cannot load the official artifact at all.

Stated plainly, because it is the strategic content of this note: closing a 16x
gap buys parity with something that already runs on the target hardware today.
The case for doing it rests on the integrated stack — agent, tool calls, KV,
server, tested together — not on the throughput numbers.

## Reproducing

Ballast, prefill sweep, and tensor audit scripts are in the session scratchpad;
the durable commands are:

```sh
llama-cli -m MODEL -f PROMPT --jinja --single-turn --reasoning off \
  --temp 0.2 -c 131072 -n 64 -ngl 99 --no-display-prompt --simple-io
```

with peak footprint from `/usr/bin/time -l`, and the tensor audit from
`gguf.GGUFReader(path).tensors` grouped by `tensor_type.name`.

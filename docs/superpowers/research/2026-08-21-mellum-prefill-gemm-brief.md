# Mellum 2 on ds4: expert-major MoE prefill, and where it stops

Written 2026-08-21 for an agent picking this up cold. Branch
`mellum-2.1-overnight`, worktree `~/projects/ds4/.claude/worktrees/mellum-2.1`.
Everything below was measured on one Apple M5 Max unless stated.

**The one-line result:** prefill at a 1,024-token width went from ~191 t/s to
**941 t/s (4.9x)** by making the MoE expert-major, and the remaining gap to
llama.cpp is now instruction-issue bound in one kernel, not bandwidth bound.
**The one-line surprise, since fixed:** decode, long believed to be at parity
with llama.cpp, was only ever measured from an *empty cache*, and at depth it
collapsed **30x** (154.5 t/s at depth 0 down to 5.1 t/s at 64K). The decode
kernel turned out to be a self-declared placeholder running 1,024 threads per
layer through a serial latency chain. Split-K over eight SIMD groups
(`3c2b379`) cut the decay to **4.5x** and is worth 2.1x at 1K rising to
**6.6x at 64K**. **What remains is an 8x redundant KV read, and that — not a
wider MoE GEMM — is the best-value item left.** See sections 7 and 7a.

---

## 1. The goal, restated by the user

> "I don't need Mellum to be good. We can give it focused tasks. I want it to
> be fast on 16 GB machines."

Read that as: model-quality work is *not* the priority; the artifact is. Two
prior reports (`tests/orchestrator-eval/REPORT-sft3015.md`,
`tests/orchestrator-eval/rlm/RESULTS-sft3015.md`) cover quality and conclude
bounded single-hop tasks work and open-ended synthesis does not. Don't redo
them.

## 2. Model facts you need and should not re-derive

Mellum 2, 12B total / 2.5B active, MoE:

| Property | Value |
| --- | --- |
| Layers | 28, **all sparse** (no dense prefix) |
| Experts | 64, top-8, router bias-free softmax -> top8 -> renormalize |
| Expert dims | in 2304, mid 896, out 2304 |
| Attention | GQA 32 q-heads : 4 kv-heads, head_dim 128 |
| Vocab | 98,304 |
| Sliding window | 1024 |
| Layer pattern | `s,s,s,full` — full attention at layers 3,7,11,15,19,23,27 |
| RoPE | **layer-selective**: YaRN on full-attention layers (attention_factor 1.2772588722239782); plain RoPE theta 500000 on sliding layers |

Two traps in that table:

- The RoPE split is per-layer-type. A flattened rope config silently produces a
  wrong model. `MellumForCausalLM` in transformers 5.15.1 raises
  `KeyError: 'sliding_attention'` on a flattened config — that error is a
  *feature*, it is catching the mistake. llama.cpp's `conversion/mellum.py`
  reads `layer_types` and gets this right.
- HF `hidden_states[28]` is **post-final-norm**, not the layer-27 output.
  Comparing it against a pre-norm fixture gives RMS 1.97 against 63.36 and
  looks like catastrophic breakage. Use a forward hook on
  `model.model.layers[27]`.

## 3. Independent oracle: the architecture is confirmed

Both ds4 and the GGUF path go through llama.cpp's conversion, so agreeing with
llama.cpp proves nothing about the architecture. That circularity was broken
with a direct FP32 HF forward (`tests/mellum-fp32-oracle/`, committed
`d8f5190`, write-up `2026-08-21-mellum2-independent-fp32-oracle.md`).

Result: **0.26% at layer 0** against the independent FP32 reference. The
architecture as implemented is right.

Consequence that matters for judging gates: ds4's internal gates measure
**5–135x finer than the reference's own error against FP32**. They are
*tripwires for regression*, not fidelity criteria. A gate moving from 0.0085%
to 0.0062% is noise-level and does not mean the model got better.

## 4. Why prefill was slow

The original MoE was **token-major**: for each token, gather its 8 experts and
multiply. Zero weight reuse across tokens.

Per-token expert traffic = 8 experts x 6.19 MiB x **28 layers** = ~1.47 GB.
At the observed 266 t/s that is **~392 GB/s — bandwidth-saturated.**

> Note for anyone reading older drafts: an earlier version of this arithmetic
> omitted the 28x layer factor and reported ~14 GB/s, concluding "overhead
> bound". That was wrong. The bandwidth diagnosis is correct.

Going **expert-major** (group tokens by expert, stage the weight row once,
sweep its bucket) cuts weight traffic ~136x: 64 experts x 6.19 MiB x 28 =
11.1 GB per 1,024-token chunk = 10.8 MB/token. Compute then binds at
~5 GFLOP/token, i.e. ~4,000 t/s at 20 TFLOP/s. A scalar (non
`simdgroup_matrix`) kernel at 2–5 TFLOP/s lands in the **400–1,000 t/s** band.
That prediction is what was built, and it landed at 941.

## 5. What was built

All in `metal/moe.metal` and dispatched from `ds4_metal.m`, behind
`DS4_MELLUM_MOE_GEMM=1`. **The default path is untouched and still bitwise
exact.**

Three kernels, dispatched as three encoders:

1. `kernel_mellum_q8_0_pair_swiglu_gemm_f32` — expert-major gate/up, R=1,
   stages the gate and up rows together (18,432 B of threadgroup memory).
2. `kernel_mellum_q8_0_down_grouped{4,8}_f32` — the down projection, template
   `mellum_down_grouped_impl<R>`. Grid is (out_dim/R, n_total_expert). Stages
   R dequantized weight rows per threadgroup via
   `ds4_mellum_stage_q8_rows()`, then each simdgroup walks its bucket's tokens
   with R register accumulators and a single `simd_sum` per row.
3. `kernel_mellum_moe_slot_reduce_f32` — sums the 8 slots **in slot order**.

Two correctness rules that are load-bearing:

- **Never float-atomic into the output.** Results go to per-(token,slot)
  staging and are reduced in slot order. Float addition is not associative;
  atomics would make the result nondeterministic run to run.
- **`slot_reduce` consults the routing table** and contributes 0.0 for an
  out-of-range expert. Without this, a stale partial from a previous chunk
  leaks into the sum. This was a real bug, found and fixed
  (`0cd8ee3`); the alternative fix — memset 75 MiB/layer — was rejected on cost.

One documented **precondition**: a token's 8 selected experts must be
*distinct*. The bitonic router guarantees this. If a future router ever
violates it, or a bucket overflows, the result corrupts **silently**. This is
written into the kernel comments (`75d438e`); keep it there.

Also `kernel_mellum_q8_0_down_batch_rowtile{2,4}_f32` (`123b9bd`), an
order-preserving row-tiled variant of the *old* token-major kernel behind
`DS4_MELLUM_DOWN_ROWTILE`. +9.9% median, default off. Superseded by the
expert-major path but bitwise-safe, so it is kept.

### The optimizations that mattered

| Step | MoE kernel time | Note |
| --- | --- | --- |
| token-major baseline | 128.3 ms | |
| expert-major | 49.8 ms | 2.58x |
| `float4` + `dot()` inner loop | 32.9 ms | **3.9x cumulative** |
| tail-tile clamp w/ compile-time-R fast path | 32.7 ms | correctness fix, ~40% of kernel depends on keeping the fast path |

Accuracy *improved* as a side effect of `float4` (7.6e-6 vs 8.6e-6 max_abs) —
wider accumulation trees reassociate more favourably.

### The optimizations that failed — do not retry these blind

Both are committed as recorded negative results in `1f55c59`.

- **Token-pairing gate/up:** 32.9 -> 34.0 ms. Reverted.
- **f16 threadgroup staging:** 32.7 -> 32.3 ms (**1.2%**) for max_abs
  7.6e-6 -> 1.7e-2, i.e. **2,250x worse accuracy**. Rejected. The negative
  result is itself the finding: **the kernel is not threadgroup-footprint or
  occupancy bound.**

### Where the wall is

Measured DRAM traffic in the new kernel is ~20 GB/s of ~400 available. Not
bandwidth. Not occupancy (see f16 above). Theoretical compute is ~9–10 ms
against 32.7 ms measured. The residual is **instruction issue and latency in
the scalar inner loop.**

The remaining route is a `simdgroup_float8x8` rewrite, keeping F32
accumulation. Estimated 1–2 days. That is the single highest-value prefill
work item left, and it is what closes the band to llama.cpp's ~4,300 t/s.

## 5a. READ THIS BEFORE QUOTING ANY PREFILL NUMBER

**The batched prefill measured in this brief is not wired into any session
path.** `ds4_mellum_prefill_chunks()` (`ds4.c:61802`) has exactly four callers
and all four are the resident profile or probes. The real session path,
`ds4_session_sync_internal` (`ds4.c:63315`), is tokenwise autoregressive and
says so itself at `ds4.c:63327`:

> "This is still tokenwise autoregressive prefill, not the later multi-token
> graph."

Its `sync_batch_tokens = 32` batches *command submission*, not computation.
Every kernel there runs with `n_tokens = 1`. Three consequences:

1. **No user experiences 941 t/s.** A session prefills at decode speed. A 4K
   prompt takes tens of seconds where the layer-major path would take ~6 s.
2. **`DS4_MELLUM_MOE_GEMM` is a no-op for real sessions.** It only affects
   `ds4_gpu_mellum_q8_0_routed_moe_batch_tensor`; the session path calls
   `..._routed_moe_one_tensor`. The flag changes nothing a user runs.
3. **The Laguna prefill comparison in section 8 is not like-for-like.**
   Laguna's `ds4_session_sync_internal` really does call
   `laguna_graph_forward_batch` (`ds4.c:63474`), so Laguna's prefill numbers
   are a shipped path. Mellum's are a lab capability. Section 8 is annotated
   accordingly — do not quote it without that caveat.

The decode numbers are **not** affected: `ds4_mellum_decode_token` is the real
session path, so split-K and head grouping are user-visible. They currently
also speed up session "prefill", because that is just decode in a loop.

Wiring this in is assessed as medium-easy, roughly 1–2 days, and is the
cheapest large win available in this codebase. See section 13.

## 6. Prefill vs context — the decay curve

`DS4_MELLUM_PROFILE_PREFILL_TOKENS=N ./ds4 --mellum-resident-profile`,
cumulative prefill of N tokens from an empty cache, chunk 1024, 3 repeats.
Marginal rates come from differencing adjacent cumulative times — that is the
rate a session actually experiences at that depth.

| Range | Marginal t/s | Cumulative t/s |
| --- | ---: | ---: |
| 0 – 1K | **941** | 941 |
| 1K – 2K | 623 | 749 |
| 2K – 4K | 575 | 651 |
| 4K – 8K | 455 | 535 |
| 8K – 16K | 300 | 384 |
| 16K – 32K | 236 | 292 |
| 32K – 64K | **160** | 207 |

Spread across 0–64K: **5.9x**.

And with the GEMM path off, for the same shapes:

| N | GEMM off | GEMM on | Speedup |
| ---: | ---: | ---: | ---: |
| 1,024 | 191.5 t/s | 940.7 t/s | **4.91x** |
| 4,096 | 182.3 t/s | 650.9 t/s | 3.57x |
| 16,384 | 162.2 t/s | 384.2 t/s | 2.37x |

**Read that last column carefully.** The GEMM win *decays with context*,
because as context grows the cost shifts out of the MoE and into attention.
The MoE work was worth 4.9x on a short prompt and 2.4x at 16K. Any further MoE
optimization has the same shape — it buys the most where the prompt is
shortest. **Attention, not MoE, is what dominates a long session.**

## 7. Decode vs depth — the correction that matters most

Every previous decode measurement in this line of work, including the "decode
is at parity with llama.cpp, do not touch it" note in the working journal,
used `--mellum-resident-profile`, which calls
`ds4_mellum_decode_state_reset()` before each timed pass. **It measured decode
from an empty cache only.** That is not a number any real session sees.

A `DS4_MELLUM_PROFILE_DECODE_DEPTH` knob was added (this brief's only
uncommitted code change, in `ds4.c`) to prime the cache with D tokens before
the clock starts. The result:

| Decode depth | serial t/s | split-K t/s | speedup |
| ---: | ---: | ---: | ---: |
| 0 | 154.5 | 151.4 | — (below threshold) |
| 1,024 | 57.7 | **122.7** | 2.13x |
| 4,096 | 36.3 | **108.6** | 2.99x |
| 16,384 | 15.9 | **67.0** | 4.22x |
| 32,768 | 9.4 | **49.3** | 5.25x |
| 65,536 | 5.1 | **33.4** | 6.55x |

The serial column decayed **30x** from empty cache to 64K. Split-K (section
7a) cuts that to **4.5x**, and the win grows with depth because the serial
chain it removes was proportional to depth.

The empty-cache 154.5 t/s is what was being compared against llama.cpp's
150–155 t/s. **The parity claim does not survive.** llama.cpp's 110–148 t/s
came from real prompts at real depth; ds4 at depth 1,024 is 57.7 t/s.

### Why this is a big, tractable opportunity

Arithmetic, with the assumption stated (f16 KV, 4 kv-heads x 128 dim, k and v):
KV per token is ~2 KiB/layer x 28 = ~57 KiB.

- **Depth 0, 6.47 ms:** ~2.7 GB of Q8 weights (2.5B active x 1.0625 B/w) in
  6.47 ms = **~417 GB/s. Bandwidth-saturated — this kernel is already right.**
- **Depth 1,024, 17.32 ms:** weights 2.7 GB + KV ~58 MB = 2.76 GB in 17.32 ms
  = ~160 GB/s. The *added* 10.85 ms moves only ~58 MB — an effective
  **~5.4 GB/s, roughly 75x off achievable bandwidth.**

So decode at depth is not bandwidth bound and not fundamentally expensive. The
decode attention path is leaving something like an order of magnitude on the
floor. **Whoever picks this up should start here, not in the MoE.** The MoE
rewrite is 1–2 days for maybe 2x on short-prompt prefill; decode attention
looks like a similar effort for a much larger win on the number users feel
during generation.

### Fixed: split-K, committed and measured (section 7a)

Root cause was that `kernel_mellum_attention_decode_gqa_f16` was a
**self-declared placeholder** — "a later split-K specialization can retain
this interface" — and that specialization existed for Laguna and was never
written for Mellum. The dispatch was `MTLSizeMake(n_head,1,1)` x 32 threads:
**1,024 threads for a whole layer's attention**, each threadgroup walking the
key range serially through a loop-carried online-softmax maximum with a
`simd_sum` and two `exp` per iteration.

`kernel_mellum_attention_decode_gqa_split_f16` stripes the key range over the
eight SIMD groups of a 256-thread threadgroup and merges their independently
normalized partials in threadgroup memory, mirroring
`kernel_laguna_attention_decode_gqa_f16`. Histories of 256 keys or fewer keep
the single-SIMD path and stay bitwise identical.

Cost per key-layer, which is the clean way to read this:

| Depth | before | after | issued BW | *useful* BW |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 373.8 ns | 53.9 ns | 304 GB/s | 38 GB/s |
| 4,096 | 417.1 ns | 51.9 ns | 315 GB/s | 39 GB/s |
| 16,384 | 414.1 ns | 61.1 ns | 268 GB/s | 34 GB/s |
| 32,768 | 398.6 ns | 54.5 ns | 301 GB/s | 38 GB/s |

A flat **~7x** at every depth, against a ceiling of 8 stripes. The latency
chain is gone.

### What is left, and it is not the MoE

The kernel now **issues** 268–315 GB/s against a ~400 GB/s ceiling, so it has
become bandwidth-bound. But **only one eighth of that traffic is unique**:
`heads_per_kv = 32/4 = 8`, and all eight query heads sharing a KV head load
the same K/V row independently. Useful bandwidth is 34–39 GB/s.

So the next lever is **head grouping**, not a wider GEMM: evaluate the eight
query heads that share a KV head in one threadgroup so each row is loaded
once. `kernel_laguna_attention_decode_gqa3_split_f16` already does exactly
this three-wide and is the pattern to copy. Mellum's 8:1 ratio makes the prize
larger than Laguna's.

Note this composes with, rather than replaces, a further `nwg` workgroup
split: at 64K each of the 8 stripes still walks ~8,192 keys serially, and
Laguna's gqa3 kernel adds a z-dimension over workgroups for exactly that.

### The sliding-window cap is honoured — so that is not the bug

The window makes a testable prediction: 21 of 28 layers cap their scan at 1,024
keys, so past depth 1,024 only the 7 full-attention layers should keep growing.
Fit a per-key-layer cost from the 0 -> 1,024 step (378.5 ns) and extrapolate
both ways:

| Depth | Measured | If cap honoured | If cap ignored | Fit vs honoured |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 27.5 ms | 25.5 ms | 49.9 ms | 1.08x |
| 16,384 | 63.0 ms | 58.0 ms | 180.1 ms | 1.09x |
| 32,768 | 106.6 ms | 101.4 ms | 353.7 ms | 1.05x |
| 65,536 | 196.1 ms | 188.2 ms | 700.9 ms | 1.04x |

The cap-honoured model tracks within 4–9% across a 16x span; the cap-ignored
model is off by 3.6x at depth 64K. **The sliding window is working correctly.**

That rules out the cheap explanation and localizes the problem precisely: the
cost is **378.5 ns per key-layer**, flat, at every depth. The scan length is
right; the per-key cost is roughly 75x worse than it should be. This is a
kernel efficiency problem in decode attention, not a scheduling or masking bug.
Start by reading the decode attention kernel and asking why moving 2 KiB of KV
costs 378 ns.

## 8. Comparison against Laguna S 2.1

Laguna figures are user-supplied from real sessions; Mellum's are synthetic
tokens in a controlled harness on an M5 Max. **This is indicative, not a
like-for-like benchmark** — different model, different prompts, different
harness. Treat the shapes as comparable and the absolute values as loose.

### Prefill

**Not like-for-like — see section 5a.** Laguna's figures come from its real
session path. Mellum's come from a probe harness with no session-path
consumer, so the Mellum column is what the kernels *can* do, not what a user
gets today. A Mellum session currently prefills at the decode rates in the
next table.

| Context depth | Laguna S 2.1 | Mellum 2 on ds4 (marginal) |
| --- | ---: | ---: |
| ~3.4K | 300–360 t/s | ~575 t/s |
| ~20.7K | 160–170 t/s | ~236 t/s |
| ~36.4K | 85–105 t/s | ~160 t/s |
| ~56.9K | 65–67 t/s | ~160 t/s |
| ~75.2K | 54–55 t/s | not measured |
| ~92.5K | 41–46 t/s | not measured |
| **spread** | **~7x** (3.4K→92.5K) | **5.9x** (0→64K) |

Mellum's prefill *kernels* are ~1.4–2.4x faster than Laguna's shipped prefill
at every depth measured, with slightly gentler decay — but Laguna's number is
one a user gets and Mellum's is not, so this row is a statement about
potential, not about the product. Laguna's golden fixture at 32,768 ctx bursts
145–155 t/s early; Mellum's 16K–32K marginal is 236 t/s.

### Decode

| Depth | Laguna S 2.1 | Mellum (serial) | Mellum (split-K) | vs Laguna now |
| --- | --- | ---: | ---: | --- |
| empty / early | 58–68 t/s | 154.5 | **151.4** | 2.4x faster |
| ~1K | 56–68 t/s | 57.7 | **122.7** | ~2x faster |
| ~4K | 56–68 t/s | 36.3 | **108.6** | ~1.7x faster |
| ~16K | 56–68 t/s | 15.9 | **67.0** | **parity** |
| ~32K | 56–68 t/s | 9.4 | **49.3** | ~1.2x slower |
| ~64K | 56–68 t/s | 5.1 | **33.4** | ~1.8x slower |
| **shape** | **flat** | 30x decay | **4.5x decay** | |

The crossover — the depth past which Laguna's flat decode wins — moved from
**about 1K to about 20K**.

**This is the headline of the comparison.** The user described Laguna's
mechanism precisely: *"prefill collapses with context, decode barely moves —
that asymmetry is the core mechanism behind the ~7x slowdown."*

**Mellum on ds4 does not have that asymmetry, and that is bad news, not good.**
Laguna's flat decode is the *correct* behaviour and the thing that makes a long
session usable. Mellum on ds4 wins the prefill comparison outright and then
gives it all back during generation. The crossover is at roughly 1K of context.
Past that it falls away fast: 3.9x slower at 16K, 12x slower at 64K, while
Laguna is still sitting at 56–68 t/s at 92K.

Put plainly: on a short prompt Mellum on ds4 is the better experience on both
axes. On a long session it is worse on the axis the user stares at.

Section 7 argued the decay was a fixable implementation problem rather than a
property of the model, and split-K has now shown that directly. Mellum on ds4
beats Laguna on both axes out to roughly 20K of context, and head grouping
(section 7a) targets the 8x redundancy that accounts for most of what remains
past that.

## 9. Quality gates — how to not get a false green

This is the trap that cost the most time. **Three of the oracle probes run
decode, not prefill:**

- `--mellum-layer0-probe-out`
- `--mellum-all-layers-probe-out`
- `--mellum-logits-probe-out`

They call the `..._one_tensor` path, which `DS4_MELLUM_MOE_GEMM` **does not
touch**. Run under the flag they return HEAD's numbers to every digit and look
like a perfect pass. They are not testing what you think.

Use these instead — they exercise batched prefill:

- `--mellum-true-prefill-probe`
- `--mellum-true-prefill-swa-probe`

Measured under the GEMM flag:

- 26-token probe logits RMS **0.008627 -> 0.006243** (improved).
- 1,030-token SWA probe within ~2%, chunk invariance preserved.
- Synthetic fixture deviation 3.8e-6 – 7.6e-6 max_abs — F32 reassociation, not
  breakage.
- The default (non-GEMM) path still asserts **exactly 0.0**.

`tests/ds4_test.c` prints the magnitudes before the bitwise asserts and skips
the exactness assert when `DS4_MELLUM_MOE_GEMM` is set. The GEMM path is
non-bitwise **by design**; the default path's bitwise contract is intact and
should stay that way.

## 10. The 16 GB question — still open

The user's actual target. Status: **marginal, not settled.**

- Q8_0 is 12.3 GiB. On a 16 GB machine that leaves ~3.7 GiB for the OS, KV
  cache, and everything else. Tight to infeasible at useful context.
- The official Q4_K_M mix was verified tensor by tensor: **155 Q4_K, 141 F32,
  15 Q6_K, 14 Q8_0, 14 Q5_0** (all `ffn_down_exps`). So ds4 needs real
  **Q4_K, Q6_K, and Q5_0 kernels** — this is not a 14-tensor patch. That work
  does not exist yet.
- A 9.33 GiB selective-precision artifact is the proposed target and has not
  been built.
- ds4 has `--simulate-used-memory NGB`, which wires memory properly. **Use
  that.** A naive zero-filled ballast is compressible and macOS will squash it
  instead of the model, invalidating the test. (A homemade ballast also once
  counted its own pages as free and grew to 94.5 GB before being killed. The
  machine recovered; swap never moved.)

## 11. Reproduction

```bash
cd ~/projects/ds4/.claude/worktrees/mellum-2.1 && make -j8
```

Lock-free MoE microbenchmark — no GGUF, no engine, **does not take the ds4
lock**, safe to run any time:

```bash
./ds4_test --mellum-moe-bench
```

Honour `DS4_BENCH_TOKENS` / `DS4_BENCH_REPS`. This is far steadier than the
resident profile (128.3 / 128.3 / 128.5 ms against 15% swings).

End-to-end prefill and decode:

```bash
DS4_MELLUM_MOE_GEMM=1 DS4_MELLUM_PROFILE_PREFILL_TOKENS=4096 DS4_MELLUM_PROFILE_DECODE_DEPTH=1024 ./ds4 --mellum-resident-profile --ctx 131072 --model <q8.gguf>
```

Env knobs that exist:

| Variable | Effect |
| --- | --- |
| `DS4_MELLUM_MOE_GEMM` | expert-major MoE prefill (default off) |
| `DS4_MELLUM_DOWN_ROWTILE` | row-tiled token-major down (default off) |
| `DS4_MELLUM_PREFILL_CHUNK` | prefill chunk width |
| `DS4_MELLUM_PROFILE_PREFILL_TOKENS` | profile prefill width, 64 .. 2^20 |
| `DS4_MELLUM_PROFILE_DECODE_DEPTH` | prime cache before decode timing |

## 12. Measurement hygiene on this box — read before trusting any number

1. **Read interleaved ratios, never absolutes.** The same baseline measured
   128 ms and 234–323 ms depending on whether PyCharm was at 249% CPU. A
   "66% regression" was once chased that turned out to be PyCharm; the control
   had moved too. Always run the configs interleaved in the same session and
   compare within it.
2. **n >= 5, report the median** for anything end-to-end. An early sweep swung
   217 -> 170 -> 119 t/s under Spotlight.
3. **A single measurement is not a delta.** The orchestrator eval learned the
   same lesson independently: no-think alone spans 11–13 on a 15-check task, so
   a two-point gap is noise.
4. **The ds4 lock.** `flock(LOCK_EX|LOCK_NB)` on `/tmp/ds4.lock`, engine-open
   only; a second process `exit(2)`s. **Do not override `DS4_LOCK_FILE`** —
   AGENT.md says it is intentional and there is kernel VM risk. `ds4_test
   --metal-kernels` needs the GPU but *not* the lock.

## 13. Recommended order of work

0. **Wire `ds4_mellum_prefill_chunks` into `ds4_session_sync`** (§5a). Medium-
   easy, ~1–2 days, no new kernels: `ds4_mellum_prefill_tokens` already drives
   the same `ds4_mellum_decode_state` a session owns, already caps chunks at
   the sliding window, and already advances position and fills logits. Do it
   with `DS4_MELLUM_PREFILL_EXACT` on first so the batch-vs-decode result
   stays bit-identical, measure, then relax. This converts an already-built,
   already-validated 4.9x into something users feel — and makes the headline
   number true.

1. **Head-group the decode attention** (§7a). Split-K is done and banked
   (2.1–6.6x). What remains is an **8x redundant KV read**: all eight query
   heads sharing a KV head load the same row. The kernel is now bandwidth-bound
   on issued traffic (268–315 GB/s of ~400) while only a eighth of it is
   useful. `kernel_laguna_attention_decode_gqa3_split_f16` is a working
   three-wide version of exactly this. Cheaper than item 2 and, on current
   evidence, worth more.
2. **`simdgroup_float8x8` MoE rewrite**, F32 accumulation preserved (§5). 1–2
   days, closes prefill toward llama.cpp — but note §6: this buys less as
   context grows.
3. **Q4_K / Q6_K / Q5_0 kernels**, then build and measure the 9.33 GiB
   artifact under `--simulate-used-memory 16GB` (§10). This is the one that
   actually answers the user's stated goal.
4. Ship the resident Q8 path on 32 GB+ machines meanwhile; it works today.

## 14. Commit trail

| Commit | What |
| --- | --- |
| `d8f5190` | independent FP32 oracle, breaks the llama.cpp circularity |
| `1da0204` | orchestrator evals re-run against sft-iter-3015 |
| `123b9bd` | row-tile the token-major down projection |
| `14cd024` | expert-major simdgroup MoE path |
| `0cd8ee3` | lock-free MoE bench; stale-partial fix |
| `21c6c70` | float4 vectorization, 3.9x |
| `75d438e` | tail-tile clamp; bucket-distinctness precondition |
| `1f55c59` | prefill-width knob; recorded negative results |
| `083bb88` | `DS4_MELLUM_PROFILE_DECODE_DEPTH`; decode measured at depth |
| `3c2b379` | split-K decode attention |
| `418dadf` | split-K hardening after review; curve to 64K |
| `ce757c7` | head-grouped (gqa8) decode attention |

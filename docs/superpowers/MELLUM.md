# Mellum 2 on ds4

Current state and runbook. For how any of this came to be — measurements,
rejected approaches, corrections — see
`research/2026-08-21-mellum-experiment-journal.md`. This document is kept short
on purpose; if it grows a narrative, the narrative belongs in the journal.

## What it is

Mellum 2 is a 12B-A2.5B MoE. The facts that shape every kernel:

| Property | Value |
| --- | --- |
| Layers | 28, **all sparse** — no dense prefix |
| Experts | 64, top-8; bias-free softmax → top-8 → renormalize |
| Expert dims | in 2304, mid 896, out 2304 |
| Attention | GQA **32 q-heads : 4 kv-heads**, head_dim 128 |
| Vocab | 98,304 |
| Sliding window | 1024 |
| Layer pattern | `s,s,s,full` — full attention at layers 3, 7, 11, 15, 19, 23, 27 |
| RoPE | **per-layer-type**: YaRN on full-attention layers (attention_factor 1.2772588722239782); plain RoPE theta 500000 on sliding layers |

Two traps worth knowing before you touch a config:

- The RoPE split is per layer type. A flattened rope config silently produces a
  wrong model. `MellumForCausalLM` raising `KeyError: 'sliding_attention'` is
  that mistake being caught, not a bug.
- HF `hidden_states[28]` is **post-final-norm**, not the layer-27 output.
  Comparing it to a pre-norm fixture looks like catastrophic breakage.

## Where the code is

| Path | What |
| --- | --- |
| `ds4.c` | runtime: KV layout, decode state, layer-major prefill, sessions |
| `ds4_mellum_diag.c` | the twelve `--mellum-diag` entry points (included by `ds4.c`) |
| `ds4_metal_mellum.m` | Metal dispatch (included by `ds4_metal.m`) |
| `metal/laguna.metal` | attention kernels — serial and head-grouped |
| `metal/moe.metal` | MoE kernels — expert-major GEMM, bucketing, slot reduce |

The two `#include`d files are separate for reading, not linkage. Their headers
list exactly what they reach into, which is what a real extraction would have
to sever.

## What it does today

**One path, no configuration.** Exact projections, expert-major MoE,
head-grouped attention above 256 keys, serial below. There is no second
production mode; `DS4_MELLUM_MOE_GEMM=0` selects a bitwise path that exists as
the test oracle.

Decode, 64 steps with the cache primed to depth first:

| Depth | t/s |
| ---: | ---: |
| 0 | 152 |
| 1,024 | 120 |
| 4,096 | 127 |
| 16,384 | 89 |

Re-measured on the current build. The 1,024 and 4,096 rows are within each
other's noise on this machine; do not read the ordering as meaningful.

Session prefill, measured on a real 1,030-token sync:

| Configuration | t/s |
| --- | ---: |
| tokenwise (what shipped before) | 120 |
| current default | ~355 |
| `DS4_MELLUM_PREFILL_EXACT=0` | ~529 |

**Quote ~355 t/s for a session, not the resident profile's 941** — the profile
measures the kernels without session bookkeeping.

## The 9.33 GiB selective artifact

Q4_K expert gate/up on layers 0-21, Q8_0 everywhere else. Built by
`tools/mellum/build-selective-artifact.sh`; kept at
`~/models/mellum-thinking-TARGET.gguf`. ds4 loads and generates from it.

Measured on this box against the Q8_0 model, arms interleaved, n=5 paired
(read the pairs, not the medians -- the run drifted ~40% top to bottom):

| | Q8_0 | selective | ratio |
| --- | ---: | ---: | ---: |
| Decode (isolated, equal load) | ~123 t/s | ~117 t/s | **0.94x** |
| Session prefill, 6045 tokens | ~341 t/s (layer-major) | ~73 t/s (tokenwise) | **0.21x** |
| Planned resident, ctx 16384 | 12.10 GiB | 9.40 GiB | -2.70 GiB |

Decode was 93.5 t/s (0.75x) until Mellum got its own Q4_K kernel.
`kernel_glm_q4_K_pair_swiglu_f32` walks elements one at a time through
`ds4_glm_q4_K_value`, which unpacks the 6-bit packed scale/min pair *per
element* -- though that pair is constant across each 32-element group.
`kernel_mellum_q4_K_pair_swiglu_f32` hoists the unpack out of the inner loop
and walks gate and up together so each `x[]` load serves both rows. Same
threadgroup reduction, same per-element association, bitwise-identical
results, ~25% faster.

The residual ~6% is the inherent cost of nibble extraction, and is close to
llama.cpp's own finding that Q4_K and Q8_0 land within a few percent here.

Prefill is still slow because the expert-major batch kernel stages Q8_0 rows
and cannot run Q4_K, so these models take the tokenwise sync path. That is the
remaining piece.

The memory win is real and is the reason the artifact exists: 9.40 GiB planned
leaves ~6.6 GiB inside 16 GiB where Q8_0 leaves ~3.9 GiB.

**16 GiB is still not demonstrated.** `--simulate-used-memory N` locks N GiB,
so simulating a 16 GiB machine on this 128 GiB box means locking 112 -- and
macOS refuses, failing at `104.00/112.00 GiB`. The tightest reachable
simulation leaves ~24 GiB. Both models load there. Real 16 GiB hardware is the
only way to answer this.

## Hosts

- **`ds4-agent`** — one session per process, and `/tmp/ds4.lock` refuses a
  second process. Parallel agent work is not available this way.
- **`ds4-server`** — resident sessions, `--batched-session N`. Verified with
  concurrent requests. Sessions interleave **serially** on the GPU, so N
  clients share the single-stream rate rather than multiplying it.
- Plain `ds4` refuses to open Mellum for generation; it is for diagnostics.

Thinking mode is a per-request toggle
(`chat_template_kwargs: {"enable_thinking": false}`), honoured on both request
shapes. It costs 3–4× the output tokens for no measured quality gain on this
repo's eval, so prefer it off for focused tasks.

## Running things

```bash
make -j8            # note: this does NOT build ds4_test
make test           # does, along with the server and agent suites
```

Kernel checks, no model or lock needed:

```bash
./ds4_test --metal-kernels
DS4_MELLUM_MOE_GEMM=0 ./ds4_test --metal-kernels
```

Model-backed diagnostics:

```bash
./ds4 --mellum-diag NAME --ctx 4096 --model <q8.gguf>
```

`NAME` is one of: `layer0`, `all-layers`, `kv-layout`, `session-lifecycle`,
`session-decode`, `session-isolation`, `interactive-session`, `swa-boundary`,
`resident-profile`, `true-prefill`, `true-prefill-swa`, `logits`. Add
`--mellum-diag-out FILE` for raw output, `--mellum-diag-trace KIND=FILE`
(`layer`/`attention`/`qk`) for the all-layers traces, `--mellum-diag-top-k N`
for `logits`.

The layer-0 oracle:

```bash
./ds4 --mellum-diag layer0 --mellum-diag-out /tmp/l0.f32 --model <q8.gguf>
python3 tests/check_mellum_layer0_oracle.py /tmp/l0.f32
```

## Settings

All resolved **once** into `ds4_mellum_runtime` (`ds4_mellum_runtime_get()` in
`ds4_gpu.h`). Read that struct to see what Mellum does; never re-derive a
default at a call site.

| Variable | Effect |
| --- | --- |
| `DS4_MELLUM_MOE_GEMM` | default ON; 0 selects the bitwise oracle |
| `DS4_MELLUM_PREFILL_EXACT` | default ON; 0 is faster and looser |
| `DS4_MELLUM_ATTN_GROUP` | default ON; head grouping above 256 keys |
| `DS4_MELLUM_SYNC_BATCH` | 0 forces tokenwise session sync |
| `DS4_MELLUM_PREFILL_CHUNK` | chunk width; also tunes cancellation granularity |
| `DS4_MELLUM_SYNC_TRACE` | which sync path ran, and its throughput |
| `DS4_MELLUM_ATTN_TRACE` | which decode kernel ran, and max key_count |
| `DS4_MELLUM_PROFILE_*` | diagnostic profile geometry |

`DS4_MELLUM_ATTN_SPLIT`, `DS4_MELLUM_ATTN_OVERDISPATCH`,
`DS4_MELLUM_DOWN_ROWTILE` and `DS4_MELLUM_GROUPED_MOE` were removed. If you
find them in an old note, they are gone, not broken.

## Traps that have cost real time

1. **`ds4_test` is not in the `all` target.** `make -j8 && ./ds4_test` runs a
   stale binary. This has already produced one believed-passing test that had
   not been compiled. Use `make test`.
2. **Some probes exercise decode, not prefill.** `--mellum-diag layer0`,
   `all-layers` and `logits` run the decode path. A prefill-only flag changes
   nothing in them, and they will report a clean pass regardless. Use
   `true-prefill` for prefill -- but see trap 6 for `swa-boundary`.
3. **Read interleaved ratios, never absolute times.** The same baseline has
   measured 128 ms and 323 ms depending on what else was running. Interleave
   the arms in one session and compare within it. n≥5, report the median.
4. **Context setting is not context used.** A run at `--ctx 150000` whose
   prompt is 3K tokens measures depth 3K. Decode figures are meaningless
   without a primed cache; use `DS4_MELLUM_PROFILE_DECODE_DEPTH`.
5. **The ds4 lock.** `flock` on `/tmp/ds4.lock`, engine-open only; a second
   process `exit(2)`s. Do not override `DS4_LOCK_FILE`. `ds4_test
   --metal-kernels` needs the GPU but not the lock.

## What to do next

1. **Build the selective 9.33 GiB artifact.** This is the stated goal — fast on
   16 GB machines — and nothing so far addresses it. It needs **Q4_K expert
   gate/up only**; Q6_K and Q5_0 belong to the *official* 7.52 GiB artifact and
   are not needed unless byte-compatibility becomes a goal. Mellum's weight
   validation (`weights_validate_mellum_layout` in `ds4.c`) is written around
   Q8_0 — the token embedding is required to be Q8_0 outright, and only the
   shared-expert tensors have any Q4_K branch today. Scope: mixed-layout
   validation,
   Q4_K/Q8 decode dispatch, expert-major Q4_K prefill, artifact construction,
   Q8-relative quality gates.
2. **Register-cache the eight queries** in the gqa8 inner loop
   (`metal/laguna.metal`) — it reloads eight query `float4`s per key and
   serializes eight `simd_sum`s. Benchmark register pressure; it is already 48
   floats per lane. A bounded win, not the deliverable.
3. **Run the task eval at n≥3** against the current numerics. Expert-major MoE
   shipped on five byte-identical greedy transcripts, which is strong evidence
   for that question but is not a scored benchmark.
4. **Measure N-session throughput** — only if `ds4-server` is the intended
   host, since `ds4-agent` is single-session.

Not recommended: the `simdgroup_float8x8` MoE rewrite. It improves the
component that shrinks as context grows, and does nothing for decode.

6. **`swa-boundary` goes vacuous on a Q4_K artifact.** It compares a tokenwise
   decode fixture against `ds4_session_sync`. When the model takes the
   tokenwise fallback both arms become the same computation, so it reports a
   deviation of exactly `0.000000` and passes while validating nothing. Q8_0
   still measures ~0.101. A perfect score here is the warning sign.
7. **`resident-profile` cannot measure a Q4_K artifact.** It primes the cache
   to depth through the batch prefill path, which refuses Q4_K, so decode
   reports `resident profile decode failed`. Measure through `ds4-server` with
   `DS4_MELLUM_SYNC_TRACE=1` instead; the trace names the path it took, which
   is also how you prove which sync path ran.
8. **`ds4-bench` refuses Mellum entirely** -- "resident sessions require a host
   that owns them" -- so benchmark through `ds4-agent` or `ds4-server`.
9. **A header change can silently corrupt `ds4_test`.** `Makefile:255` omitted
   `ds4_gpu.h` from `ds4_test.o`'s dependencies, so adding a field to a shared
   struct rebuilt `ds4_metal.o` but not the test, and the mismatched layouts
   produced a suite that compiled, linked, ran, returned success, and reported
   seven wrong values. Fixed, but check the dependency list before blaming your
   own numerics.
10. **`timeout(1)` is not on macOS.**

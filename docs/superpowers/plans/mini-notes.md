# Laguna XS 2.1 streaming — deployment sizing notes

Task 8 deployment log. **Retargeted from the plan's original 16 GB Mac Mini
to a 32 GB system** (user decision, 2026-07-27) after the measurements below
showed 16 GB does not fit at a useful context length.

Everything here was measured on the development laptop (M5 Max, 128 GB) with
`gguf/Laguna-XS-2.1-Q4_K_M.gguf` at commit 003733f. **The acceptance run on
real 32 GB hardware has not happened yet** — see "Still to verify".

## 1. Why 16 GB was dropped

Fixed costs at the plan's originally specified `--ctx 32768`, independent of
expert-cache budget:

| component | size |
|---|---|
| non-routed weights | 9.64 GiB |
| KV cache | 1.31 GiB |
| graph scratch | 3.97 GiB |
| **subtotal before any cached expert** | **14.92 GiB** |

That leaves nothing for the expert cache, and nothing for macOS's own ~4 GiB,
on a 16 GB machine. Graph scratch scales with prefill width (ctx 4096 → 0.99
GiB, 8192 → 1.99, 16384 → 3.97, where prefill caps out), and Laguna refuses
`--prefill-chunk` (existing family restriction), so that lever is unavailable.
Dropping to ctx 8192 gets the subtotal to ~13 GiB, which is technically under
16 GB but leaves no working room.

## 2. Expert-cache sizing sweep

`--ctx 16384`, greedy (`--temp 0`), 256 tokens, Python-flavored prompt
("Write a Python class that parses a CSV file, validates rows against a
schema, and yields typed records."):

| experts | hit rate | SSD read | task footprint | gen t/s |
|---:|---:|---:|---:|---:|
| 800 | 0.593 | 30.00 GiB | 6.34 GiB | 46.4 |
| 1,600 | 0.769 | 17.04 GiB | 7.86 GiB | 48.7 |
| 3,200 | 0.902 | 7.20 GiB | 10.92 GiB | 52.1 |
| 4,800 | 0.907 | 6.87 GiB | 12.81 GiB | 51.9 |
| 6,400 | 0.907 | 6.87 GiB | 12.81 GiB | 52.3 |

**Saturates at ~3,600 experts.** The 4,800 and 6,400 rows are byte-identical
(same 3,615 misses, same 6.87 GiB) — the whole Python working set is cached
and further budget is dead weight. Below 1,600 the SSD traffic explodes 4.4x.

The residual 6.87 GiB at saturation is compulsory cold-start: 3,615 experts
fetched exactly once (3,615 x 1.95 MiB = 6.88 GiB). Unavoidable within a
process; it amortizes over a long session.

Context cost is cheap once the cache is saturated — at 4,000 experts,
ctx 16384 = 12.44 GiB footprint, ctx 32768 = 13.07 GiB. **+0.63 GiB for 2x
the context, with identical throughput (51.96 vs 51.88 t/s).** Take the
larger context.

## 3. Recommended 32 GB configuration

```
./ds4-agent -m gguf/Laguna-XS-2.1-Q4_K_M.gguf \
    --ssd-streaming --ssd-streaming-cache-experts 4000 -c 32768
```

| | |
|---|---|
| ds4 task footprint (measured) | 13.1 GiB |
| hot model pages via mmap (estimated) | ~9.6 GiB |
| **total** | **~22.7 GiB** |
| **free for macOS + other apps** | **~9.3 GiB** |

More headroom variant: `--ssd-streaming-cache-experts 3200` gives ~20.5 GiB
total / ~11.5 GiB free, and costs only 0.907 -> 0.902 hit rate.

The mmap figure is an estimate, not a measurement: macOS `phys_footprint`
excludes clean file-backed pages, so the 9.64 GiB of non-routed weights the
model touches every token is not in the 13.1 GiB.

## 4. Why these throughput numbers are optimistic

All of the above ran on a 128 GB laptop where the entire 18.88 GiB model sits
in the OS page cache. The measured `pread` rate proves it: 14.19 GiB in 368 ms
during the Task 7 A/B run is **38.6 GiB/s**, which is RAM bandwidth, not
storage. A 32 GB machine cannot hold the whole file in page cache alongside a
~13 GiB resident process, so misses that resolve from page cache here will be
real SSD reads there (~3-5 GB/s on Apple NVMe, so roughly 8-12x slower on that
path).

Expect the **memory and cache-budget numbers to hold** and the **throughput
numbers not to**. Do not quote 52 t/s as a 32 GB expectation.

## 5. The dominant remaining constraint

Startup reports, on every run:

```
SSD streaming mixed-precision model: 20/39 routed layers off the slab size
class will bypass the expert cache and read experts via mapped model views
WARNING: the majority of routed layers (20/39) are off the slab size class
```

Only 19 of 39 sparse layers use the expert cache at all. The other 20 read
through mmap and do not appear in the hit/miss counters — so the 0.907 hit
rate describes roughly half the model. (Sanity check on the smoke run: 63
decode steps x 19 layers x 8 experts = 9,576 = hits + misses exactly.)

This is what Tasks 10-13 attack. A biased uniform RoutedQ3_K artifact now
exists, but it does **not** yet obtain the cache benefit: its Q3 routed triples
take ds4's correct mapped-model fallback because the generic streaming decode
path has no Q3_K address-table kernels. A 16k-context, 4096-token-prefill,
256-token Q3 run reported the planned 1.01 GiB expert-cache reservation but
**0.00 GiB live streaming experts** after graph free; 800/1,600/3,200/4,800
expert budgets all had no cache statistics and held near 62 t/s. Do not use the
startup reservation as a footprint result.

Q3 cache kernels and a nonzero-hit correctness test now precede the Python
expert hotlist. Re-run this sweep only after that gate passes.

## 6. Download target

`./download_model.sh xs21-q4` added (repo `poolside/Laguna-XS-2.1-GGUF`,
revision pinned to `1a37c0a5fb8c7a18e6106decb6be6327d1b63fa6` per xs2-facts.md
item 5). Verified: re-running against an existing file skips the download.

**Side effect worth knowing:** like every target in that script, `xs21-q4`
relinks `./ds4flash.gguf` to the model it just fetched. `ds4flash.gguf` is
`ds4_test`'s default model (`DS4_TEST_MODEL`), so running this target
repoints the test suite at Laguna XS 2.1 and away from the DeepSeek IQ2XXS
model the suite expects. This was hit and manually reverted during Task 8.
Re-point it before running `./ds4_test`:

```sh
ln -sf "$PWD/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf" ds4flash.gguf
```

## 7. Still to verify on real 32 GB hardware

- [ ] Startup expert-cache budget chosen by the automatic sizing (compare to
      the 4,000 recommended here — the auto-budget derives from Metal's
      `recommendedMaxWorkingSetSize`, ~84% of RAM, then 80% of that, minus
      the 9.64 GiB of non-routed weights)
- [ ] Warm-cache decode t/s (expect below the 52 t/s measured here, per §4)
- [ ] Cache hit rate after a 10-minute agent session
- [ ] Memory pressure / whether the system swaps
- [ ] Whether `iogpu.wired_limit_mb` needed adjusting
- [ ] **Acceptance:** agent completes a simple multi-tool task (read a file
      and summarize, then write a test) without OOM

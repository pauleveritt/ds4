# P2.6--P2.7 — Laguna XS 2.1 hotlist and acceptance preflight

**Status:** local work complete, 2026-07-28.  P2.6 established a compatible
selected-id profile and connected the existing hotlist loader to Laguna's
separate generation graph.  The profile does not improve the constrained
cache benchmark, so it is intentionally not compiled in.  P2.7's real-32-GB
acceptance remains an external-hardware gate; its local memory preflight is
complete.

## P2.6: selected-id hotlist

The profile is `gguf-tools/imatrix/laguna-xs21-biased.hotlist`; its deterministic
web/Python workload is recorded beside it in
`gguf-tools/imatrix/laguna-xs21-hotlist-workload.md`.  Eight 64-token greedy
streaming runs at context 8192 and cache 1600 produced 157,248 selections and
19,656 layer records.  Its top 1,000 rows span all 39 sparse layers, so it is
not a one-layer profiler failure.

The pre-existing file format and loader already agree: comments are ignored
and `layer expert hits weight` supplies each seed priority.  The runtime
override is:

```sh
DS4_METAL_STREAMING_EXPERT_HOTLIST=gguf-tools/imatrix/laguna-xs21-biased.hotlist
```

Two integration omissions prevented it from applying to XS Q3 until now:

1. the shared seed-eligibility predicate did not recognize Laguna Q3/Q4/Q6
   routed triples; and
2. Laguna's dedicated generation graph never called the shared seed routine.

The graph now seeds after prefill.  Seeding before prefill is ineffective,
because the prefill LRU churns it before first-token decode.  A profiled run
confirms `preload=800`, `loaded=800`, all 39 layers, and an 88.6 ms seed.

### Fixed constrained-cache A/B

Both runs used the biased Q3 artifact, context 8192, prefill chunk 4096,
800 cache entries, the retry-HTTP Python prompt, and 256 greedy tokens on the
M5 Max.  They produced the identical continuation.

| condition | cache hit rate | miss pread | decode |
|---|---:|---:|---:|
| ordinary LRU | 45.8% | 54.31 GiB | 32.15 t/s |
| selected-id hotlist | 44.3% | 56.33 GiB | 30.54 t/s |

The generalized eight-prompt popularity list displaces prompt-local working
set at this small cache size.  It is therefore a regression, not a default
optimization.  Retain the file and runtime override for future adaptive or
per-workload experiments, but do **not** generate a compiled-in XS default.

`./tests/xs21_stream_ab.sh` against the biased Q3 artifact passes after the
integration change.

### Stateful-agent closure

Sol noted that the table above only exercises one-shot `./ds4`, while the
intended consumer is the stateful `ds4-agent` session.  The only plausible
remaining static policy was therefore tested with a temporary, environment-
gated post-sync session seed and then removed.

The same Python retry task ran through `ds4-agent --non-interactive`, cache
800, context 8192, and prefill chunk 4096.  The partial run verified a
400-entry seed spanning all 39 layers and returned the identical answer.

| session policy | cache hit rate | miss pread | pread time |
|---|---:|---:|---:|
| ordinary LRU | 44.1% | 41.72 GiB | 898.0 ms |
| 400-of-800 selected-id seed | 42.4% | 43.25 GiB | 912.9 ms |

This rejects both full-cache and partial static seeding on the real Python
agent path.  **Decision: stop Python-hot-expert tuning.**  Keep the profile
artifact and the opt-in one-shot loader as research tools, but do not spend
more roadmap time on static Python hotlists without a materially different,
adaptive cache policy.

## P2.7: 32 GB acceptance preflight

The development machine is a 128 GB M5 Max, so it cannot substantiate the
acceptance criteria that depend on actual memory pressure, page-cache misses,
or a sustained agent session on 32 GB hardware.

The local configuration preflight did pass with context 32768, prefill chunk
4096, and a 3,200-expert target:

| item | measured locally |
|---|---:|
| planned total | 6.53 GiB |
| context runtime (KV + graph buffers) | 2.30 GiB |
| target cache | 4.03 GiB |
| live cache after an 8-token smoke | 1.57 GiB |
| task footprint after smoke | 6.46 GiB |

The live cache is deliberately below target in this short smoke; it only
contains experts selected so far.  This shows the 32k configuration constructs
without an allocation failure, not its steady-state footprint or speed.

Run the checklist in `docs/superpowers/plans/mini-notes.md` §7 on a real 32 GB
machine before declaring product acceptance.  In particular, do not project
the local 25.93 t/s eight-token result onto that hardware: this laptop serves
many mapped-file reads from its large OS page cache.

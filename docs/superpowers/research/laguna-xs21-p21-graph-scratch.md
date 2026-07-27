# P2.1 — Laguna XS 2.1 graph-scratch investigation and fix

**Status:** complete, 2026-07-27.

## Finding

The Phase 2 handoff correctly measured Laguna graph scratch but misidentified
the constraint as a fundamental lack of a `--prefill-chunk` lever.  The actual
state was a stale family-level guard: the CLI accepted the option, then the
Laguna engine rejected it before graph construction.  Beneath that guard,
Laguna already has a chunked batch-prefill loop.

Two implementation defects made the option unusable and its memory opaque:

1. `laguna_graph_alloc` hardcoded `min(ctx, 16384)` for every activation
   workspace and never received `engine.prefill_chunk`.
2. The Laguna branch of `ds4_context_memory_estimate_with_prefill_mode`
   reported `prefill_cap=1` and a small decode-style scratch estimate rather
   than the graph's row-scaled Metal allocations.

## Change

`ds4.c` now:

- accepts `--prefill-chunk` for **Laguna XS 2.1 only**; legacy Laguna S 2.1
  retains its previous rejection;
- threads the configured cap into both one-shot generation and stateful
  session graph allocation;
- uses the cap for every row-scaled Laguna activation tensor; and
- calculates startup memory from the same allocation formula, including
  fixed logits/output tensors.

The generic batch loop already limits each prefill call to `g->prefill_cap`,
so no kernel or routing change was required.

## Measurements

Official `Laguna-XS-2.1-Q4_K_M`, Metal streaming, context 16384, 800-expert
cache, one greedy output token:

| prefill cap | graph scratch | planned graph context (KV + scratch) |
|---:|---:|---:|
| default 16384 | 4069.45 MiB / 3.97 GiB | 4.66 GiB |
| 4096 | 1017.66 MiB / 0.99 GiB | 1.68 GiB |

The practical 4096-token cap saves **3.0 GiB** of graph scratch at context
16384 while leaving the full KV context intact.  This is a measurement, not a
projection.  Throughput needs a separate long-prompt benchmark; these short
runs establish allocation and correctness, not a performance verdict.

## Verification

- `make -j8` passed.
- At context 8192, `--prefill-chunk 4096` reported `prefill_cap=4096` and
  `scratch 1017.66 MiB`; the default is 2034.92 MiB.
- Greedy resident output for `def fizzbuzz(n):` matched bytewise between the
  default and 4096-token cap for 128 tokens.
- Greedy resident and streamed output with the 4096-token cap matched
  bytewise for the same 128-token run.
- A temporary 6,642-token prompt ran under the 4096-token cap as two chunks:
  `4096/6642`, then `6642/6642`.
- `make ds4_test` built, but `./ds4_test` could not run because this dedicated
  worktree intentionally lacks its `ds4flash.gguf` fixture.  No symlink or
  default-model change was made to work around that environmental limitation.

## Consequences for Phase 2

P2.1 is no longer an open investigation.  A 4096-token prefill cap reduces
the uniform-Q4_K projection at context 16384 from about 9.1 GiB to about
6.1 GiB before any cache-budget tuning, and it makes 16 GB substantially more
comfortable.  A production quality artifact must still satisfy the pending
P2.0 stock A/B requirement before those projections become deployment claims.

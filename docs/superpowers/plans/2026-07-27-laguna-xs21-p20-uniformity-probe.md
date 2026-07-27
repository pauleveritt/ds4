# P2.0 — Laguna XS 2.1 uniform routed-Q4_K probe

**Status:** planned.  Preflight only has been completed; no quantization,
imatrix collection, model inference, or GPU work has run.

## Purpose

Test the Phase 2 footprint premise with the smallest meaningful artifact:
whether llama.cpp can quantize the BF16 Laguna XS 2.1 model so that every
routed expert layer has one cache-servable byte-size class.

The official `Q4_K_M` file does not meet that condition.  Its Q4_K_M
heuristic boosts `ffn_down_exps` to Q6_K in 20 of the 39 sparse layers.  ds4
therefore streams only the other 19 layers and keeps the boosted layers in the
decode static span set.  That accounts for roughly 8.4 GiB of apparent
"non-routed" footprint.

This is a layout experiment, not a quality experiment.  It deliberately uses
no imatrix and makes no corpus, quality, hotlist, or hardware-acceptance
claim.

## Why Q4_K first

The shipped Laguna decode cache path can serve routed-down tensors only when
they are Q4_K or Q6_K (`laguna_decode_experts_cache_servable` in `ds4.c`).
Uniform Q3_K or Q2_K would pass the load-time layout validator but remain
mapped rather than cached, so it cannot test the footprint thesis today.

Q4_K is thus the minimum useful probe:

- it can use the current address-table cache path;
- it is already accepted by the XS 2.1 layout validator;
- it isolates the per-layer boost heuristic without introducing imatrix or
  low-bit quality variables; and
- success establishes a useful uniform-Q4_K fallback even if future Q3_K
  cache-kernel support is needed.

## Fixed inputs and expected output

| Item | Value |
|---|---|
| Source | `/Users/pauleveritt/projects/ds4/gguf/Laguna-XS-2.1-BF16.gguf` (63.8 GiB) |
| Quantizer | Local Homebrew `llama.cpp` 9580; record its version and full command line at execution |
| Output (untracked) | `/Users/pauleveritt/projects/ds4/gguf/laguna-xs-2.1-uniform-routed-q4k-probe.gguf` |
| Base recipe | `Q4_K_M` |
| Required overrides | `ffn_gate_exps=q4_k`, `ffn_up_exps=q4_k`, `ffn_down_exps=q4_k` |
| Explicit exclusions | No `--imatrix`; no corpus; no quality decision |

The models live in the primary checkout's shared `gguf/` directory; this
worktree intentionally has no `gguf/` path.  Use the absolute paths above
rather than adding a worktree-local symlink.

The overrides deliberately name all three routed tensors.  Overriding only
the down projection is insufficient: ds4's cache slab eligibility is based on
the complete routed-expert byte layout, not merely the tensor type seen in the
startup warning.

## Preconditions and non-heavy checks

These may be performed before authorizing a quantization run:

1. Record `llama-quantize --version` and `llama-quantize --help` in the run
   log.  The current local help confirms `--tensor-type`,
   `--tensor-type-file`, `--pure`, and `--dry-run` are available.
2. Confirm free disk space for the BF16 source, an approximately 19 GiB probe,
   and temporary quantizer files.  The output must remain untracked.
3. Run `--dry-run` with the exact proposed overrides if its output does not
   read or transform the full model.  Record its size estimate only; do not
   treat it as proof that the overrides take precedence over the Q4_K_M boost
   heuristic.
4. Confirm no `ds4` or `ds4_test` process is active before the later A/B
   check.  `tests/xs21_stream_ab.sh` enforces this safeguard itself.

No model-generation command belongs in this preflight stage.

### Completed preflight — 2026-07-27

- `llama-quantize` resolves to `/opt/homebrew/bin/llama-quantize`, from
  Homebrew `llama.cpp` 9580.  Its executable SHA-256 is
  `ac563c18f7431637f3c191c3fc7fb6a835a7a6aaa6fefdb51335924c525a5ee5`.
- Its local help confirms `--tensor-type`, `--tensor-type-file`, `--pure`,
  and `--dry-run` are available.
- The shared BF16 and official Q4_K_M inputs exist at the absolute paths
  above (63,829.6 MiB and 19,335.1 MiB respectively).
- The source filesystem has 277 GiB free.  No `ds4` or `ds4_test` process was
  active at the time of this check.
- No `--dry-run`, quantization, imatrix collection, inference, or GPU command
  was run.

## Authorized execution only after explicit approval

Quantization is CPU/disk intensive and writes a multi-GiB artifact.  Do not
run this command, collect an imatrix, or run a Metal correctness gate until
the user explicitly authorizes heavy work:

```sh
llama-quantize \
  --tensor-type ffn_gate_exps=q4_k \
  --tensor-type ffn_up_exps=q4_k \
  --tensor-type ffn_down_exps=q4_k \
  /Users/pauleveritt/projects/ds4/gguf/Laguna-XS-2.1-BF16.gguf \
  /Users/pauleveritt/projects/ds4/gguf/laguna-xs-2.1-uniform-routed-q4k-probe.gguf \
  Q4_K_M
```

Before treating the command as canonical, validate the quantizer's matching
semantics on its pinned version.  If suffix overrides do not apply to every
`blk.*` tensor, generate a `--tensor-type-file` that spells out all 117 routed
tensor names (39 layers × gate/up/down) and archive that file with the run
record.

`--pure` is a diagnostic fallback, not the first experiment.  It disables all
K-quant mixtures and changes non-routed tensors too, so it cannot distinguish
whether the required routed overrides themselves defeat the boost heuristic.

## Acceptance checks after the probe exists

### A. Artifact inspection (no inference)

Inspect the GGUF tensor map, grouped by sparse-layer index.  Require:

- every `blk.1` through `blk.39` `ffn_gate_exps`, `ffn_up_exps`, and
  `ffn_down_exps` tensor is Q4_K;
- the calculated per-expert byte layout is identical across all 39 layers;
- no routed tensor remains Q6_K; and
- the complete type map is saved alongside the run command and llama.cpp
  commit/version.

The check must inspect individual tensor names, not only the aggregate GGUF
type histogram.

### B. Engine admission and cache coverage

Run ds4's `--inspect` and a short streamed startup against the probe.  Require
all 39 sparse layers to be on the pinned slab class: no mixed-precision
warning, no bypass-layer count, and no routed expert bytes folded into the
decode static span set.  Compare the reported static footprint with the
official Q4_K_M baseline; the target is the ~8.4 GiB reduction described in
`docs/superpowers/LAGUNA-XS21.md` §5, subject to measurement.

### C. Correctness gate (GPU work)

Only after A and B pass, run:

```sh
./tests/xs21_stream_ab.sh \
  /Users/pauleveritt/projects/ds4/gguf/laguna-xs-2.1-uniform-routed-q4k-probe.gguf
```

Require `xs21 stream A/B: OK`.  This is the authoritative correctness check;
plausible generated text is insufficient evidence for this model family.

## Decision table

| Result | Decision |
|---|---|
| A, B, and C pass | Uniformity premise proven for Q4_K. Continue with P2.1 and later decide whether Q4_K already meets the footprint target. |
| Overrides do not produce 39 uniform layers | Investigate override matching/version behavior before corpus or imatrix work. If llama.cpp cannot force it, revisit the footprint thesis. |
| A passes but B reports bypass layers | Treat as ds4 cache-eligibility/byte-layout work, not a quantizer success. Do not start corpus work on that assumption. |
| A and B pass but C fails | Stop; preserve outputs and debug the stream path before any quality or footprint conclusions. |
| Q3_K is still desired after Q4_K succeeds | Plan Q3_K address-table cache support first. A uniform Q3_K artifact alone will not reduce ds4's mapped decode footprint. |

## Completion record

When execution is authorized, append the exact command, quantizer version or
commit, GGUF tensor-map evidence, ds4 startup lines, A/B transcript, artifact
size, and final decision to this document.  Do not commit the generated GGUF.

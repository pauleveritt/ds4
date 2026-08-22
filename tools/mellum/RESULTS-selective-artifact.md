# The 9.33 GiB selective artifact: built and measured

Built 2026-08-22 with `build-selective-artifact.sh` from the
`Mellum2-12B-A2.5B-Thinking` HF snapshot. This is the "first agentic-planning
release target" from the 2026-08-01 feasibility study, which had never been
produced as a file.

## Composition, as specified

44 tensors at Q4_K — 22 layers × (gate, up) — and 154 at Q8_0, verified from
the quantizer log. Layers 22–27 keep gate/up at Q8_0; every down projection
stays Q8_0.

**The study's size estimate was exact: 9.33 GiB predicted, 9.33 GiB measured.**
Its block-geometry arithmetic can be trusted for the other rows of that table.

| Build | Size | vs Q8_0 |
| --- | ---: | ---: |
| Q8_0 reference | 12.04 GiB | — |
| **Selective target** | **9.33 GiB** | **−2.71 GiB, −22.5%** |
| Q4_K gate/up all layers, no imatrix | 8.59 GiB | −3.45 GiB |
| Official Q4_K_M | 7.52 GiB | −4.52 GiB |

The 8.59 GiB row is the study's *baseline* recipe, not the target; it was built
first by mistake and is kept as a datapoint. It compresses six more layers and
skips calibration.

## Speed, llama.cpp, M5 Max, r=2

| Build | pp1024 | tg64 |
| --- | ---: | ---: |
| Q8_0 | 5132.3 ± 4.9 | 153.4 ± 4.8 |
| Selective | 5184.1 ± 7.6 | **163.5 ± 5.9** |

Prefill is unchanged within noise. **Decode is ~6.6% faster**, which is the
expected direction: decode is bandwidth-bound on weights, and this reads fewer
bytes per token. The compression is not merely free here, it pays.

Note `llama-bench` reports both as "Q8_0" because the file's ftype is Q8_0;
only the size column reveals the mix.

## Quality, n=3, no-think, temperature 0.2

| Task | Q8_0 | Selective | delta |
| --- | --- | --- | ---: |
| A (of 15) | 14, 12, 12 → **12.7** | 12, 14, 12 → **12.7** | 0.0 |
| B (of 18) | 13, 14, 14 → **13.7** | 14, 16, 9 → **13.0** | −0.7 |

**No measurable quality loss** — with the resolution stated honestly. Task A is
identical. Task B's −0.7 sits inside a **7-check spread** on the selective arm,
driven by one run that ran to the 900-token cap and was truncated mid-answer.
That is a length-control artifact, not evidence about the quantization.

This eval resolves about two checks. A −0.7 delta with a spread of 7 is not a
finding.

## What this does not yet show

- **ds4 cannot load it.** `weights_validate_mellum_layout` requires
  `DS4_TENSOR_Q8_0` for all three expert tensors and will `ds4_die`. Every
  number above is llama.cpp measuring the *artifact*; none of it is ds4
  measuring its own kernels.
- Reaching that needs: mixed-layout validation, Q4_K expert gate/up in the
  decode MoE, and a Q4_K expert-major prefill kernel. The GLM/DeepSeek paths
  already carry a `q4_k_pair_swiglu` kernel family and the Mellum MoE args
  already carry a type field, so this is adaptation rather than invention.
- **16 GB has not been demonstrated**, only made plausible: 9.33 GiB resident
  leaves ~6.7 GiB for OS, KV and context buffers where Q8_0 left ~3.7 GiB.
  Confirming it wants `--simulate-used-memory 16GB` at minimum, and real
  hardware to be convincing.

## Calibration provenance

The imatrix was computed over 24 chunks at n_ctx 2048 from ~135 KB: the three
committed orchestrator-eval prompts plus source from this repository
(`ds4_mellum_diag.c`, `metal/laguna.metal`, `tests/ds4_test.c`, `MELLUM.md`).
The eval prompts alone are ~9 KB, which is too thin to calibrate on; source was
chosen because it resembles the work a coding model does. **The eval prompts
are therefore inside the calibration set**, so the quality numbers above are
not fully out-of-sample and should be read as a regression check rather than a
benchmark.

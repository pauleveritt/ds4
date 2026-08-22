#!/bin/zsh
#
# Build the 9.33 GiB agentic-planning artifact described in
# docs/superpowers/research/2026-08-01-mellum2-feasibility.md.
#
#   routed expert gate/up, layers 0-21   imatrix-calibrated Q4_K
#   routed expert gate/up, layers 22-27  Q8_0   (the final six MoE layers)
#   every routed expert down projection  Q8_0
#   token embedding, attention, output   Q8_0
#
# Down stays Q8_0 for two independent reasons: ds4 does not execute Q5_0, and
# Mellum's 896-wide down input is not divisible by the 256-element K-quant
# block, so a K-quant there is not merely worse but unrepresentable.
#
# Usage: build-selective-artifact.sh <hf-snapshot-dir> <out-dir> [calib.txt]
#
# Needs llama.cpp at a revision whose conversion/mellum.py reads layer_types --
# 0e4a0362 is the pinned one -- because a flattened rope config silently
# produces a wrong model.  See MELLUM.md.
set -e
SNAP=$1; OUT=$2; CALIB=${3:-}
[ -n "$SNAP" ] && [ -n "$OUT" ] || { echo "usage: $0 <hf-snapshot> <out-dir> [calib.txt]"; exit 2; }
mkdir -p "$OUT"
LCPP=${LLAMA_CPP:-$HOME/src/llama.cpp}

echo "== 1/4 convert to BF16 =="
python3 "$LCPP/convert_hf_to_gguf.py" "$SNAP" --outfile "$OUT/mellum-BF16.gguf" --outtype bf16

echo "== 2/4 intermediate Q8_0, used only to compute the imatrix =="
llama-quantize "$OUT/mellum-BF16.gguf" "$OUT/mellum-Q8_0.gguf" Q8_0 8

echo "== 3/4 imatrix =="
# Calibration text should look like the work the model will do.  The committed
# eval prompts alone are ~9 KB, which is thin; supplement with source.
[ -n "$CALIB" ] || { echo "no calibration file given"; exit 2; }
llama-imatrix -m "$OUT/mellum-Q8_0.gguf" -f "$CALIB" -o "$OUT/mellum.imatrix" -c 2048 -ngl 99

echo "== 4/4 selective quantize =="
llama-quantize \
  --imatrix "$OUT/mellum.imatrix" \
  --tensor-type 'blk\.([0-9]|1[0-9]|2[01])\.ffn_gate_exps=Q4_K' \
  --tensor-type 'blk\.([0-9]|1[0-9]|2[01])\.ffn_up_exps=Q4_K' \
  "$OUT/mellum-BF16.gguf" "$OUT/mellum-selective.gguf" Q8_0 8

echo
echo "expect 44 q4_K tensors (22 layers x gate+up) and ~154 q8_0; 9.33 GiB"
ls -la "$OUT/mellum-selective.gguf"

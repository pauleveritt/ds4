#!/bin/zsh
#
# Task-level quality of ds4's own kernels.
#
# The other harnesses here point at llama-server, so they score the *model*.
# This one points the same probes at ds4-server, so the scores move when ds4's
# kernels move.  That is the gate to use when changing numerics -- a fixture
# deviation of 0.09% rms says nothing about whether a sampled token changed,
# let alone whether the answer got worse.
#
#   ./run_ds4_quality.sh 1 prod      # expert-major MoE, the production path
#   ./run_ds4_quality.sh 0 oracle    # bitwise path, for comparison
#   # then, per config: cp the saved outputs into /tmp/mellum-eval/out and run
#   #   python3 score.py ; python3 score_p3.py
#
# Read the spread before the delta.  At n=3 and temperature 0.2 the within-
# config spread is 1-4 checks, so this resolves differences of roughly two
# checks and nothing finer.  A one-check gap is noise, not a regression.
#
# $1 = DS4_MELLUM_MOE_GEMM value, $2 = label for the output files.
set -e
W=~/projects/ds4/.claude/worktrees/mellum-2.1
E=$W/tests/orchestrator-eval
M=$(ls ~/.cache/huggingface/hub/models--JetBrains--Mellum2-12B-A2.5B-Thinking-GGUF-Q8_0/snapshots/*/*.gguf | head -1)
OUT=${DS4_QUALITY_OUT:-/tmp/mellum-eval/runs}
mkdir -p /tmp/mellum-eval/out "$OUT"
cp $E/prompt_p2.txt $E/prompt_p3.txt /tmp/mellum-eval/
rm -f /tmp/mellum-eval/out/*.txt
cd $W
HOME=/tmp/mellum2 DS4_MELLUM_MOE_GEMM=$1 ./ds4-server -m "$M" -c 16384 --port 8125 > $OUT/server_$2.log 2>&1 &
SRV=$!
for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:8125/v1/models >/dev/null 2>&1 && break; sleep 2; done
cd $E
for r in 1 2 3; do
  python3 probe_any.py m "$2-r$r"    900 off prompt_p2.txt
  python3 probe_any.py m "p3-$2-r$r" 900 off prompt_p3.txt
done
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true
mkdir -p $OUT/$2 && cp /tmp/mellum-eval/out/*.txt $OUT/$2/
echo "outputs saved to $OUT/$2"

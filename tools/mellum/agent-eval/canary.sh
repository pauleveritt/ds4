#!/bin/zsh
# Forced one-tool canary: measures PROTOCOL compliance, not agent competence.
# Usage: canary.sh <label> <model> <n>
set -u
LABEL=$1; MODEL=$2; N=$3
W=~/projects/ds4/.claude/worktrees/mellum-2.1
source /tmp/agentclinic-runs/metal_env.sh
OUTER=/tmp/agentclinic-runs/canary-$LABEL
mkdir -p "$OUTER"
SEQ=$(( $(ls "$OUTER" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1) + 1 )) 2>/dev/null
[ -z "$SEQ" ] && SEQ=1
BASE="$OUTER/$SEQ"
mkdir -p "$BASE"
ln -sfn "$SEQ" "$OUTER/latest"
cp $W/Makefile $BASE/Makefile
cd $BASE
for i in $(seq 1 $N); do
  $W/ds4-agent --non-interactive ${THINK_ARGS:-} -m "$MODEL" -c 8192 \
    --trace $BASE/trace_$i.txt \
    -p "Read the file Makefile in the current directory and tell me its first line. Use your tools." \
    > $BASE/out_$i.log 2>&1
  T=$BASE/trace_$i.txt
  if grep -aq "recovered_missing_open" $T 2>/dev/null; then V=recovered_missing_open
  elif grep -aq "recovered_wrong_wrapper" $T 2>/dev/null; then V=recovered_wrong_wrapper
  elif grep -aq "tool start detected" $T 2>/dev/null; then V=canonical
  else V=no_call; fi
  if grep -aq "dsml error" $T 2>/dev/null; then E=parse_error; else E=ok; fi
  EXEC=$(grep -ac "tool:read\|CC ?= cc" $BASE/out_$i.log 2>/dev/null || echo 0)
  echo "run=$i variant=$V parse=$E read_executed=$EXEC"
done

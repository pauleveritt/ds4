#!/bin/zsh
# Agentic capability run: one arm (model) against the AgentClinic spec task.
# Usage: run_arm.sh <arm-label> <model-path>
set -u
ARM=$1; MODEL=$2
W=~/projects/ds4/.claude/worktrees/mellum-2.1
SRC=~/PycharmProjects/dlai-local-ai-course
BASE=/tmp/agentclinic-runs/$ARM
mkdir -p "$BASE"
SEQ=$(( $(ls "$BASE" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1) + 1 )) 2>/dev/null
[ -z "$SEQ" ] && SEQ=1
RUN="$BASE/$SEQ"
mkdir -p "$RUN"
ln -sfn "$SEQ" "$BASE/latest"
# Seed from tracked files only: no __pycache__, no prior tests/, no .codex.
git -C $SRC archive HEAD | tar -x -C $RUN
mkdir -p $RUN/.agentlogs

PROMPT='Look in specs/mission.md and specs/tech-stack.md for project details.
Look at the specs/roadmap.md and:
- Implement Phase 1
- Implement Phase 2

Then do a review and provide a list of critical mistakes.'

# ds4 loads its .metal sources relative to cwd, so running from the task
# directory needs every source path pinned back at the worktree.
source /tmp/agentclinic-runs/metal_env.sh

cd $RUN
START=$SECONDS
$W/ds4-agent --non-interactive -n 8192 ${EXTRA_ARGS:-} \
  -m "$MODEL" -c 32768 \
  --trace $RUN/.agentlogs/trace.txt \
  -p "$PROMPT" > $RUN/.agentlogs/stdout.log 2>&1 &
AGENT_PID=$!
( sleep 180; kill -TERM $AGENT_PID 2>/dev/null ) &
WATCH_PID=$!
wait $AGENT_PID
RC=$?
kill $WATCH_PID 2>/dev/null
ELAPSED=$((SECONDS-START))

# ---- deliverables: objective, not judged ----
FILES=0
for f in app.py models.py templates/base.html templates/home.html; do
  [ -f "$RUN/$f" ] && FILES=$((FILES+1))
done
TESTS="no-test-file"
if [ -f "$RUN/tests/test_app.py" ]; then
  TESTS=$(cd $RUN && python3 -m pytest -q tests/ 2>&1 | tail -1)
fi

TOOLCALLS=$(grep -ac "dsml done" $RUN/.agentlogs/trace.txt 2>/dev/null || echo 0)
NUDGES=$(grep -ac "tool nudge" $RUN/.agentlogs/trace.txt 2>/dev/null || echo 0)
ROUNDS=$(grep -ao "tool_round=[0-9]*" $RUN/.agentlogs/trace.txt 2>/dev/null | sort -u | wc -l | tr -d ' ')
MAXPROMPT=$(grep -ao "prompt=[0-9]*" $RUN/.agentlogs/trace.txt 2>/dev/null | sed 's/prompt=//' | sort -n | tail -1)
echo "arm=$ARM rc=$RC elapsed=${ELAPSED}s files=$FILES/4 toolcalls=$TOOLCALLS nudges=$NUDGES rounds=$ROUNDS maxprompt=$MAXPROMPT tests=[$TESTS]"

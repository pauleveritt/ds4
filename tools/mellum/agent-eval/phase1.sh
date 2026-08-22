#!/bin/zsh
# Phase-1-only AgentClinic run with a corrected instrument.
# Usage: phase1.sh <label> <model>   (env: DS4_AGENT_TOOL_NUDGE)
set -u
LABEL=$1; MODEL=$2
W=~/projects/ds4/.claude/worktrees/mellum-2.1
SRC=~/PycharmProjects/dlai-local-ai-course
BASE=/tmp/agentclinic-runs/p1-$LABEL
mkdir -p "$BASE"
SEQ=$(( $(ls "$BASE" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1) + 1 )) 2>/dev/null
[ -z "$SEQ" ] && SEQ=1
RUN="$BASE/$SEQ"
mkdir -p "$RUN/.agentlogs"
ln -sfn "$SEQ" "$BASE/latest"
git -C $SRC archive HEAD | tar -x -C $RUN
# Pinned env, built once and reused, so validation is deterministic.
VENV=/tmp/agentclinic-runs/.venv-shared
[ -d $VENV ] || (python3 -m venv $VENV && $VENV/bin/pip -q install \
  'fastapi==0.115.14' 'jinja2==3.1.4' 'pytest==8.3.4' 'httpx==0.28.1' \
  'python-multipart==0.0.20' 'uvicorn==0.34.0' >/dev/null 2>&1)
cp -R $VENV $RUN/.venv

PROMPT='Implement Phase 1 only from specs/roadmap.md. Read specs/mission.md, specs/tech-stack.md and specs/roadmap.md first with your tools.

Deliverables, exactly these four files:
  app.py
  templates/base.html
  templates/home.html
  tests/test_app.py

Use this API exactly; do not rely on memory for these imports:
  from fastapi import FastAPI, Request
  from fastapi.responses import HTMLResponse, RedirectResponse
  from fastapi.templating import Jinja2Templates
  templates = Jinja2Templates(directory="templates")
  # render with: templates.TemplateResponse(request, "home.html", {...})

Bootstrap CSS:  https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css
Bootstrap JS:   https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js
Do not add integrity or crossorigin attributes. Do not invent hashes.

When all four files exist, validate by running exactly:
  ./.venv/bin/python -m pytest -q tests/
Fix any failure and re-run until it passes. Do not finish while it fails.'

source /tmp/agentclinic-runs/metal_env.sh
cd $RUN
START=$SECONDS
# No timeout(1) on macOS: run in the background, race a watcher against it,
# and kill whichever process is still standing. 180s is the process wall-clock
# cap; -n 8192 caps generation length per model call, which is what actually
# bounds a rambling round -- the process timeout is the backstop if that fails.
$W/ds4-agent --non-interactive -n 8192 -m "$MODEL" -c 32768 \
  --trace $RUN/.agentlogs/trace.txt -p "$PROMPT" > $RUN/.agentlogs/stdout.log 2>&1 &
AGENT_PID=$!
( sleep 180; kill -TERM $AGENT_PID 2>/dev/null ) &
WATCH_PID=$!
wait $AGENT_PID
RC=$?
kill $WATCH_PID 2>/dev/null
ELAPSED=$((SECONDS-START))
[ $ELAPSED -ge 178 ] && echo "TIMED OUT after ${ELAPSED}s" >> $RUN/.agentlogs/stdout.log

FILES=0; MISSING=""
for f in app.py templates/base.html templates/home.html tests/test_app.py; do
  if [ -f "$RUN/$f" ]; then FILES=$((FILES+1)); else MISSING="$MISSING $f"; fi
done
PYTEST="not-run"
[ -f "$RUN/tests/test_app.py" ] && PYTEST=$(cd $RUN && ./.venv/bin/python -m pytest -q tests/ 2>&1 | tail -1)

T=$RUN/.agentlogs/trace.txt; L=$RUN/.agentlogs/stdout.log
ROUNDS=$(grep -ao "tool_round=[0-9]*" $T 2>/dev/null | sort -u | wc -l | tr -d ' ')
CALLS=$(grep -ac "dsml done" $T 2>/dev/null || echo 0)
NUDGE=$(grep -ac "tool nudge" $T 2>/dev/null || echo 0)
MAXP=$(grep -ao "prompt=[0-9]*" $T 2>/dev/null | sed 's/prompt=//' | sort -n | tail -1)
GENB=$(grep -vc "^ds4:" $L >/dev/null 2>&1; grep -v "^ds4:" $L | wc -c | tr -d ' ')
DEGEN=$(grep -ac "degenerated into repeated" $L 2>/dev/null || echo 0)
OVER=$(grep -ac "exceeded the maximum size" $L 2>/dev/null || echo 0)
COMPACT=$(grep -ac "not enough context left" $L 2>/dev/null || echo 0)
echo "label=$LABEL rc=$RC elapsed=${ELAPSED}s files=$FILES/4 rounds=$ROUNDS calls=$CALLS nudge=$NUDGE maxprompt=$MAXP genbytes=$GENB degen=$DEGEN oversize=$OVER compact_fail=$COMPACT pytest=[$PYTEST] missing=[$MISSING]"

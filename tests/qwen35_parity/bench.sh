#!/bin/sh
# bench.sh MODEL [RUNS]
#
# Prompt and generation speed of ./ds4 (run from the repository root) on the
# two fixed chat prompts beside this script (502 and 2006 rendered tokens with
# the 9B's template, 540 and 2044 with the 27B's reasoning-effort line): one
# warm-up, then RUNS runs of 128 greedy tokens each, printing ds4's own
# prefill and generation rates.  llama.cpp's side is
#   llama-bench -m MODEL -p 512,2048 -n 0 -r 3
#   llama-bench -m MODEL -p 0 -n 128 -d 512,2048 -r 3
set -e
MODEL=$1
RUNS=${2:-3}
DIR=$(cd "$(dirname "$0")" && pwd)
for p in p512 p2048; do
    ./ds4 -m "$MODEL" --ctx 8192 --system "You are a helpful assistant" --temp 0 -n 16 \
        --prompt-file "$DIR/$p.txt" > /dev/null 2>&1
    i=0
    while [ "$i" -lt "$RUNS" ]; do
        printf '%s ' "$p"
        ./ds4 -m "$MODEL" --ctx 8192 --system "You are a helpful assistant" --temp 0 -n 128 \
            --prompt-file "$DIR/$p.txt" 2>&1 >/dev/null | grep -E 'prefill: .* t/s'
        i=$((i + 1))
    done
done

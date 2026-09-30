#!/bin/sh
# ds4_side.sh MODEL OUTDIR [ds4 args...]
#
# Runs every prompt of prompts.json through ./ds4 (run from the repository
# root): the rendered chat tokens go to OUTDIR/NAME.tokens and a 64-token
# greedy run with top-5 logprobs per step to OUTDIR/NAME.json.  The system
# prompt is explicit so the dumped tokens are the ones generation uses.
set -e
MODEL=$1
OUT=$2
shift 2
DIR=$(cd "$(dirname "$0")" && pwd)
SYSTEM="You are a helpful assistant"
mkdir -p "$OUT"
python3 -c 'import json,sys; [print(p["name"]) for p in json.load(open(sys.argv[1]))]' "$DIR/prompts.json" |
while read -r name; do
    prompt=$(python3 -c 'import json,sys; print(next(p["prompt"] for p in json.load(open(sys.argv[1])) if p["name"] == sys.argv[2]))' "$DIR/prompts.json" "$name")
    ./ds4 -m "$MODEL" --ctx 4096 --system "$SYSTEM" --dump-chat-tokens -p "$prompt" "$@" 2>/dev/null |
        grep '^\[' > "$OUT/$name.tokens"
    ./ds4 -m "$MODEL" --ctx 4096 --system "$SYSTEM" --dump-logprobs "$OUT/$name.json" \
        --logprobs-top-k 5 -n 64 -p "$prompt" "$@" > "$OUT/$name.log" 2>&1
    echo "$name done"
done

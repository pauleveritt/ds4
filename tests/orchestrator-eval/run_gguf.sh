#!/bin/bash
# usage: run_gguf.sh <label> <gguf> <promptfile>
LB=~/src/llama.cpp/build/bin/llama-cli
BASE=$(basename $3 .txt)
OUT=/tmp/mellum-eval/out/$1.$BASE.txt
$LB -m "$2" -f "$3" --jinja --single-turn --reasoning off \
    --temp 0.2 --top-p 0.9 --top-k 64 \
    -c 8192 -n 2048 -ngl 99 --no-display-prompt --simple-io \
    2>/tmp/mellum-eval/out/$1.$BASE.err > "$OUT"
printf "%-10s %-10s %5s words  " "$1" "$BASE" "$(wc -w < "$OUT")"
grep -oE "eval time =[^,]*" /tmp/mellum-eval/out/$1.$BASE.err | head -2 | tr '\n' ' '; echo

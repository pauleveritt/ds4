### Task 7: Streamed-vs-resident correctness gate

**Files:**
- Create: `tests/xs21_stream_ab.sh`

**Interfaces:**
- Consumes: Tasks 3 (resident) and 6 (streamed).
- Produces: the repeatable A/B gate used after every subsequent engine change.

- [ ] **Step 1: Write the harness**

```bash
#!/bin/zsh
# A/B: resident vs streamed logits/tokens must agree (greedy, fixed seed).
set -e
MODEL=${1:-gguf/Laguna-XS-2.1-Q4_K_M.gguf}
PROMPTS=("def fizzbuzz(n):" "<html><head><title>" "Explain HTTP caching briefly." "import asyncio")
for p in $PROMPTS; do
  A=$(./ds4 -m $MODEL -c 8192 -p "$p" -n 128 --nothink --temp 0)
  B=$(./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts 800 \
        -c 8192 -p "$p" -n 128 --nothink --temp 0)
  if [[ "$A" != "$B" ]]; then
    echo "MISMATCH on prompt: $p"; exit 1
  fi
done
echo "xs21 stream A/B: OK"
```

The 800-expert cache (< one layer's worth × several layers) forces heavy eviction — the stress case. Check `--temp 0` is the repo's greedy flag (`./ds4 --help | grep -i temp`); use the documented greedy mechanism if named differently, and per CONTRIBUTING.md use its output-agreement procedure if bytewise equality is not the repo's standard (adjust the comparison accordingly — the CONTRIBUTING procedure is authoritative).

- [ ] **Step 2: Run it**

```bash
chmod +x tests/xs21_stream_ab.sh && ./tests/xs21_stream_ab.sh
```

Expected: `xs21 stream A/B: OK`. Any mismatch is a Task 6 bug — fix there, re-run, only then proceed.

- [ ] **Step 3: Commit**

```bash
git add tests/xs21_stream_ab.sh
git commit -m "Add XS 2.1 streamed-vs-resident A/B correctness gate"
```

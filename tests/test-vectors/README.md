# DeepSeek V4 Flash Test Vectors

These vectors were captured from the official DeepSeek V4 Flash API using
`deepseek-v4-flash`, greedy decoding, thinking disabled, and
`top_logprobs=20`. The hosted API does not expose full logits, so these files
store the best logprob slice the API provides.

Files:

- `prompts/*.txt`: exact user prompts.
- `official/*.official.json`: official API continuations and top-logprobs.
- `official.vec`: compact C-test fixture generated from the official JSON.
- `local-golden.vec`: local top-k/logit fixture captured from a known-sane DS4
  Flash run. It is used to catch substantial backend drift that can keep the
  same greedy token while damaging the logits distribution.

Regenerate official vectors:

```sh
DEEPSEEK_API_KEY=... ./tests/test-vectors/fetch_official_vectors.py
```

Running the fetcher without `--only` also regenerates `official.vec`.

The C runner consumes `official.vec` directly:

```sh
./ds4_test --logprob-vectors
```

GLM 5.2 OpenRouter vectors are kept in a separate directory:

```sh
OPENROUTER_API_KEY=... ./tests/test-vectors/fetch_openrouter_glm_vectors.py

DS4_TEST_MODEL=models/GLM-5.2-UD-Q4_K_XL.gguf \
DS4_TEST_VECTOR_FILE=tests/test-vectors/glm-openrouter/official.vec \
  ./ds4_test --logprob-vectors
```

The same fetcher also writes `tests/test-vectors/glm-openrouter/manifest.tsv`
for `gguf-tools/quality-testing/score_official`.  By default it routes to
OpenRouter `parasail/fp8` with strict parameter matching so top-logprob slices
are present in the fixture.

It also consumes the local golden fixture:

```sh
./ds4_test --local-golden-vectors
```

## Mellum layer-0 numerical oracle

`mellum-llama-cpp/l-out-0.f32` is the 26 x 2,304 little-endian F32 `l_out-0`
checkpoint captured by the `llama-eval-callback` example at llama.cpp revision
`0e4a0362239713ea95a6864a17a8de4b0ad90d62`. It uses the pinned Mellum Q8_0
GGUF and the `python_add` ChatML fixture. Its SHA-256 is
`5a99d699c83a1a5417f9175f46e30c343026767372546001a112aef34ce49243`.

After writing ds4's diagnostic output, check the durable intermediate gate:

```sh
HOME=/tmp/mellum2 ./ds4 --mellum-layer0-probe-out /tmp/mellum2-ds4-l-out-0.f32 --model MODEL
python3 tests/check_mellum_layer0_oracle.py /tmp/mellum2-ds4-l-out-0.f32
```

The checker verifies the fixture hash and exact byte counts, then requires
maximum absolute error no greater than `0.006` and RMS error no greater than
`0.000125`. `--mellum-layer0-probe-out` stages output beside the requested path
and atomically replaces it only after the complete checkpoint is written.

`mellum-llama-cpp/l-out-27-last.f32` is the corresponding final-token,
post-layer-27 F32 checkpoint (SHA-256
`050436ca257f8ebe36656f32785b9649ddf8b8ad75a08ec47f005ea588b72ebb`). It
is the batched-prefill diagnostic; it does not share ds4's tokenwise schedule.

The matching tokenwise callback checkpoint is
`mellum-llama-cpp/l-out-27-tokenwise.f32` (SHA-256
`4ae7a46e409d6e3bf760ef8f7c671f32ff16d0798b8cd3dd25fe5fcbb3f086fe`). It is
the all-layer acceptance oracle, with maximum absolute error no greater than
`1.5` and RMS error no greater than `0.06`:

```sh
HOME=/tmp/mellum2 ./ds4 --mellum-all-layers-probe-out /tmp/mellum2-ds4-l-out-27.f32 --model MODEL
python3 tests/check_mellum_layer0_oracle.py --layer 27-tokenwise /tmp/mellum2-ds4-l-out-27.f32
```

The checker resolves its pinned fixtures relative to its own source file, so
these commands also work when invoked outside the repository. It rejects a
wrong byte count, fixture hash mismatch, and every non-finite reference,
actual, or delta value. Its dependency-free regression test is part of
`make test`; run it alone with `make test-mellum-oracle-checker`.

`mellum-llama-cpp/result-output-tokenwise.f32` is the matching final raw-logit
checkpoint: one 98,304-value F32 row captured as llama.cpp's `result_output`
after final RMSNorm and the Q8_0 output matrix. Its SHA-256 is
`4ec7f9c838058fc267ec0a2f117aa4db523950df1621f61d25afcf63e3be2b1c`. The
inspect-only ds4 probe does not select a token:

```sh
HOME=/tmp/mellum2 ./ds4 --mellum-logits-probe-out /tmp/mellum2-ds4-result-output.f32 --model MODEL
python3 tests/check_mellum_layer0_oracle.py --layer logits-tokenwise /tmp/mellum2-ds4-result-output.f32
```

The raw-logit gate is maximum absolute error no greater than `0.025` and RMS
no greater than `0.012`.

For an inspect-only ranking report that neither samples nor emits a token:

```sh
HOME=/tmp/mellum2 ./ds4 --mellum-logits-probe-top-k 16 --model MODEL
```

### llama.cpp capture provenance

`capture-layer-output.patch` is the minimal callback patch used to create all
three F32 checkpoints. It applies to the exact llama.cpp revision
`0e4a0362239713ea95a6864a17a8de4b0ad90d62`; use a detached worktree so the
reference checkout remains clean. `MODEL` must be the pinned public Q8_0 GGUF
at revision `6f5b0031c9ea37740f630362d3c06c54933fc2f4`, with SHA-256
`d4049c2599796d18245523818c0534e8f8605166fc58c60dd0753547facdb3a2`.
The following recaptures the matching schedule layer-27 oracle on the original
Apple Metal setup:

```sh
DS4_ROOT=$PWD
LLAMA_SRC=/path/to/llama.cpp
LLAMA_WORKTREE=/tmp/mellum2-llama-callback-src
LLAMA_BUILD=/tmp/mellum2-llama-callback-build
MODEL=/path/to/Mellum2-12B-A2.5B-Thinking-Q8_0.gguf

git -C "$LLAMA_SRC" worktree add --detach "$LLAMA_WORKTREE" 0e4a0362239713ea95a6864a17a8de4b0ad90d62
git -C "$LLAMA_WORKTREE" apply "$DS4_ROOT/tests/test-vectors/mellum-llama-cpp/capture-layer-output.patch"
cmake -S "$LLAMA_WORKTREE" -B "$LLAMA_BUILD" -DGGML_METAL=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build "$LLAMA_BUILD" --target llama-eval-callback -j 8
MELLUM_DUMP_TENSOR=l_out-27 MELLUM_DUMP_PATH=/tmp/l-out-27-tokenwise.f32 MELLUM_TOKENWISE=1 \
  "$LLAMA_BUILD/bin/llama-eval-callback" -m "$MODEL" -ngl all -b 1 --escape \
  -p $'<|im_start|>user\nComplete this Python function: def add(a, b):<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
shasum -a 256 /tmp/l-out-27-tokenwise.f32
```

The command must emit exactly 9,216 bytes and SHA-256
`4ae7a46e409d6e3bf760ef8f7c671f32ff16d0798b8cd3dd25fe5fcbb3f086fe` before it
replaces the pinned fixture. Omit `MELLUM_TOKENWISE=1` to make the batched
`l-out-27-last.f32` capture; set `MELLUM_DUMP_TENSOR=l_out-0` without it to
make the 26-row layer-0 capture. The fixed prompt renders the 26 IDs printed by
the callback, so tokenization is part of the recapture check. Set
`MELLUM_DUMP_TENSOR=result_output` to recapture the one-row raw-logit oracle.

The Metal SSD-streaming cache-pressure repro for issue #384 is a focused
variant of the official-vector check. It forces a 16GiB routed-expert cache and
runs only the `short_code_completion` case that exposes wrong logits when
layer-batched decode reuses expert-cache buffers before the command buffer has
completed:

```sh
DS4_TEST_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  ./ds4_test --metal-ssd-streaming-cache-pressure
```

The runner opens the normal non-quality path with accelerator-specific fast
routes disabled and pins `DS4_METAL_PREFILL_CHUNK=2048` for this strict
official-vector check.

`official.vec` is intentionally trivial to parse from C: each case points to a
prompt file and each expected token is hex-encoded by bytes. The official JSON
files remain in the tree so the compact fixture can be audited against the raw
API response.

To inspect a local top-logprob dump manually:

```sh
./ds4 --metal --nothink -sys "" --temp 0 -n 4 --ctx 16384 \
  --prompt-file tests/test-vectors/prompts/long_code_audit.txt \
  --dump-logprobs /tmp/long_code_audit.ds4.json \
  --logprobs-top-k 20
```

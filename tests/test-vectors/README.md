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
./ds4 --mellum-layer0-probe-out /tmp/mellum2-ds4-l-out-0.f32 --model MODEL
./tests/check_mellum_layer0_oracle.py /tmp/mellum2-ds4-l-out-0.f32
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
./ds4 --mellum-all-layers-probe-out /tmp/mellum2-ds4-l-out-27.f32 --model MODEL
./tests/check_mellum_layer0_oracle.py --layer 27-tokenwise /tmp/mellum2-ds4-l-out-27.f32
```

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

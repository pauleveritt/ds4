# Laguna XS 2.1 biased imatrix and RoutedQ3_K artifact

P2.3 completed on 2026-07-27 using upstream llama.cpp commit
\`0e4a0362239713ea95a6864a17a8de4b0ad90d62\` (build 10154, Metal enabled).
Homebrew llama.cpp build 9580 cannot process \`general.architecture=laguna\` and
must not be substituted for this build.

## Inputs

- BF16 model: \`gguf/Laguna-XS-2.1-BF16.gguf\`
- Corpus: \`dataset/laguna_xs21_rendered_prompts.txt\` (3,000,513 estimated
  tokens; SHA-256 \`d1dfe6a17d75d71897b82cb1fa1fa22e89c41150a7e15da84bdb6c1cb1e1b424\`)
- Imatrix: \`laguna-xs21-biased.imatrix.gguf\` (179.1 MiB, SHA-256
  \`07dfdad0abb4ecd2832565ca20875014f3ce77c6150500d93aff66247edd666f\`)

The pinned tool uses GGUF imatrix format by default. \`--parse-special\` is
intentional: the corpus contains the Laguna chat and tool markers. The 400
chunks completed in about 114 minutes. The warnings about partial entries are
checkpoint snapshots from \`--output-frequency 1\`; the final imatrix reports
476 entries computed on all 400 chunks.

\`\`\`sh
/Users/pauleveritt/src/llama.cpp/build/bin/llama-imatrix \
  -m /Users/pauleveritt/projects/ds4/gguf/Laguna-XS-2.1-BF16.gguf \
  -f gguf-tools/imatrix/dataset/laguna_xs21_rendered_prompts.txt \
  -o gguf-tools/imatrix/laguna-xs21-biased.imatrix.gguf \
  -c 8192 --chunks 400 --parse-special --no-ppl -ngl 99 --output-frequency 1
\`\`\`

## Quantization

The three explicit routed-tensor overrides are the P2.0 uniformity control:
they prevent llama.cpp's layer boost heuristic from creating a mixed routed
precision layout. The result is deliberately untracked at
\`gguf/laguna-xs-2.1-RoutedQ3_K-biased.gguf\`.

\`\`\`sh
/Users/pauleveritt/src/llama.cpp/build/bin/llama-quantize \
  --imatrix gguf-tools/imatrix/laguna-xs21-biased.imatrix.gguf \
  --tensor-type ffn_gate_exps=q3_k \
  --tensor-type ffn_up_exps=q3_k \
  --tensor-type ffn_down_exps=q3_k \
  /Users/pauleveritt/projects/ds4/gguf/Laguna-XS-2.1-BF16.gguf \
  /Users/pauleveritt/projects/ds4/gguf/laguna-xs-2.1-RoutedQ3_K-biased.gguf \
  q8_0
\`\`\`

The command completed in 61.7 seconds. The artifact is 14.64 GiB (3.76 BPW),
SHA-256 \`b1dc95e586c2fc586a032101cbb3d1b3884e9d70905824b21e07afb821bf1bc8\`.
Its inspected tensor map is:

| type | tensors | bytes |
|---|---:|---:|
| F32 | 239 | 0.08 GiB |
| Q8_0 | 322 | 1.99 GiB |
| Q3_K | 117 | 12.57 GiB |

The 117 Q3_K tensors are exactly 39 sparse layers × routed gate/up/down;
there are no Q4_K or Q6_K routed tensors. \`llama-quantize --dry-run\` predicted
the same 14,990.50 MiB size and reported all 117 manual Q3_K overrides.

## P2.4 validation

No new layout branch was necessary: Phase 1's generic Laguna validator already
accepts coherent Q3_K gate/up/down triples with a Q8_0 signal path. A resident
128-token FizzBuzz smoke test produced a normal continuation. The existing
stream gate also passed:

\`\`\`sh
./tests/xs21_stream_ab.sh \
  /Users/pauleveritt/projects/ds4/gguf/laguna-xs-2.1-RoutedQ3_K-biased.gguf
# xs21 stream A/B: OK
\`\`\`

Quality results against the local Q4_K_M snapshot fixtures are recorded in
\`gguf-tools/quality-testing/data/laguna-xs21/README.md\`. They pass the P2.4
gate: average NLL improved on both general and web/Python fixtures. As noted
there, the local self-consistency metric has a documented numerical floor, so
first-token and LCP movement alone is not a regression verdict.


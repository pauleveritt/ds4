# Laguna XS 2.1 P2.6 hotlist workload

The first profile is a small deterministic web/Python workload. Each prompt is
greedy (`--nothink --temp 0`) for 64 generated tokens against the biased Q3
artifact. The selected-id writer merges counts across separate processes.

- `def parse_cache_control(header):`
- `import asyncio\nasync def fetch_all(urls):`
- `@app.get('/health')\ndef health():`
- `Write a Python function that retries an HTTP request with exponential backoff.`
- `<script type="module">\nconst response = await fetch('/api/items')`
- `Explain how HTTP ETags and If-None-Match work.`
- `def pytest_fixture(tmp_path):`
- `Create a TypeScript React component that loads paginated JSON.`

Profiled with the biased RoutedQ3_K artifact, `--ssd-streaming`, cache 1600,
context 8192, `--prefill-chunk 4096`, and 64 generated tokens per prompt.
The first process used `DS4_MOE_RECORD_SELECTED_HOTLIST_FRESH=1`; subsequent
processes used `DS4_MOE_RECORD_SELECTED_HOTLIST_MERGE=1`.  The resulting
`laguna-xs21-biased.hotlist` has 157,248 selections / 19,656 layer records;
its top 1,000 rows cover all 39 sparse layers.

The profile is an input to the runtime override, not a performance claim:

```sh
DS4_METAL_STREAMING_EXPERT_HOTLIST=gguf-tools/imatrix/laguna-xs21-biased.hotlist \
  ./ds4 -m /absolute/path/to/laguna-xs-2.1-RoutedQ3_K-biased.gguf --ssd-streaming ...
```

At an 800-expert constrained cache it regressed the fixed 256-token A/B, so it
is deliberately not baked as the compiled-in XS default.  See
`docs/superpowers/research/laguna-xs21-p26-p27-hotlist-acceptance.md`.

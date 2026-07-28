# Laguna XS 2.1

Laguna XS 2.1 runs in `ds4-agent` through the Metal SSD-streaming path on
Apple silicon.  The validated artifact is a uniform routed-Q3_K build, which
lets all 39 sparse layers use the bounded streaming expert cache.

## Deployment target

Use an Apple-silicon Mac with **32 GB RAM or more**.  The implementation and
resident-vs-streamed correctness gate were validated on a 128 GB M5 Max; the
same configuration still needs final acceptance on real 32 GB hardware for
memory pressure and SSD-miss throughput.  Do not treat 16 GB as supported.

## Install on another Mac

1. Clone or update ds4 and check out `laguna-s2.1`.

2. Build the agent locally:

   ```sh
   make ds4-agent
   ```

3. Copy the generated model artifact.  It is not fetched by
   `download_model.sh` and is not committed to Git:

   ```sh
   rsync -avP \
     /Users/pauleveritt/projects/ds4/gguf/laguna-xs-2.1-RoutedQ3_K-biased.gguf \
     other-mac:/path/to/ds4/gguf/
   ```

4. Start the agent with streaming and a bounded cache:

   ```sh
   ./ds4-agent -m gguf/laguna-xs-2.1-RoutedQ3_K-biased.gguf \
     --ssd-streaming --ssd-streaming-cache-experts 3200 \
     --prefill-chunk 4096 -c 32768
   ```

`--ssd-streaming-cache-experts 3200` is the recommended initial cache target.
The `--prefill-chunk 4096` cap is required to keep graph scratch bounded at
the 32k context setting.

## First-run checks

- Confirm the startup report selects Metal SSD streaming and a 3,200-expert
  target cache.
- Run a real agent task while watching memory pressure in Activity Monitor.
- Record decode speed and miss-read behavior; a 128 GB development laptop can
  serve many misses from its OS page cache, so its throughput does not predict
  a 32 GB machine.

For the engineering evidence, memory measurements, and the remaining 32 GB
acceptance checklist, see
`docs/superpowers/research/laguna-xs21-p26-p27-hotlist-acceptance.md` and
`docs/superpowers/plans/mini-notes.md`.

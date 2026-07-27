#!/usr/bin/env just --justfile

set quiet

@default:
    just --list

# Start ds4-server with the Laguna model at 80K context
@serve:
    ./ds4-server -c 80000

# Start ds4-server at 80K with disk KV cache
@serve-disk:
    ./ds4-server -c 80000 --kv-disk-dir ~/.ds4/server-kv --kv-disk-space-mb 16384

# Start ds4 CLI with the Laguna model at 80K context
@cli ctx="80000":
    ./ds4 -c {{ctx}}

# Show current model and build info
@info:
    @echo "Model: $(readlink ds4flash.gguf)"
    @echo "Build: $(./ds4 --version 2>&1 || echo 'unknown')"

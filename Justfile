# Local ds4 tooling.
#
# `laguna-s` starts ds4-server against the Laguna S 2.1 workspace so it can
# back local agentic coding tools (Pi, etc.) from any directory on this
# machine. Laguna's build lives in a separate worktree/branch (laguna-s2.1)
# because that model line isn't merged into main; this recipe just points at
# it and runs it from here.

laguna_s_workspace := "/Users/pauleveritt/projects/ds4/.claude/worktrees/laguna-s-bench"
laguna_s_model := laguna_s_workspace / "gguf/laguna-s-2.1-Q4_K_M.gguf"
laguna_s_dflash := laguna_s_workspace / "gguf/laguna-s-2.1-DFlash-Q8_0.gguf"

# Dedicated KV disk cache for Laguna S only. ds4's on-disk KV cache is keyed
# purely by a SHA1 of the rendered prompt prefix -- it has no model identity
# in the key. Pointing two different models at the same --kv-disk-dir risks
# a cache hit restoring a checkpoint captured by a different architecture.
# This directory must never be reused for any other ds4-server instance
# (DeepSeek V4 Flash, GLM, etc.) -- keep those on their own directories.
laguna_s_kv_dir := env_var("HOME") / ".local/state/ds4-kv/laguna-s"

# Start ds4-server for Laguna S 2.1, full residency, DFlash-enabled. Runs in
# the foreground -- keep it in its own terminal/tmux pane; other project
# directories talk to it over HTTP.
#
# ds4-server has no --temp flag: sampling is set per-request by the client,
# not the server. DFlash only accelerates greedy (temperature=0) requests and
# silently falls back to ordinary decode otherwise -- correct either way, just
# not sped up unless the client (Pi) sends temperature=0. See the Pi config
# note in the task summary for how to pin that.
laguna-s ctx="32768" port="8000":
    mkdir -p {{laguna_s_kv_dir}}
    cd {{laguna_s_workspace}} && ./ds4-server \
        -m {{laguna_s_model}} \
        --dflash {{laguna_s_dflash}} \
        -c {{ctx}} \
        --port {{port}} \
        --kv-disk-dir {{laguna_s_kv_dir}} \
        --kv-disk-space-mb 8192

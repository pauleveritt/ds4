# Mellum agent evaluation harness

Two instruments, deliberately separate. `canary.sh` measures **protocol
compliance** -- does a tool call get emitted and parsed -- on a single forced
request. `run_arm.sh` measures **agent competence** on the AgentClinic spec
task. A model can pass one and fail the other, and conflating them is how the
first pass through this concluded, wrongly, that Mellum could not call tools.

    ./canary.sh q8 <model.gguf> 20        # n=20 protocol canary
    ./run_arm.sh label <model.gguf>       # one AgentClinic attempt
    DS4_AGENT_TOOL_NUDGE=2 ./run_arm.sh label <model.gguf>

Both need `metal_env.sh` beside them, which pins every `metal/*.metal` source
so ds4 can run from a workspace that is not the worktree:

    for f in metal/*.metal; do b=$(basename $f .metal);
      echo "export DS4_METAL_$(echo $b | tr a-z A-Z)_SOURCE=$PWD/$f"; done > metal_env.sh

`run_arm.sh` seeds each run from `git archive HEAD` of the course repo, so a
run never inherits files from the previous one.

Classify canary failures rather than tuning the prompt against them: immediate
EOS, prose, tool-shaped output missed by detection, or a valid decision not to
call. The third class is a parser bug and the others are not.

# SDD Progress: Laguna XS 2.1 streaming

Plan: docs/superpowers/plans/2026-07-25-laguna-xs2-streaming.md
Spec: docs/superpowers/specs/2026-07-25-laguna-xs2-streaming-design.md

## Decisions made during execution

- 2026-07-25: Model is **Laguna XS 2.1** (`poolside/Laguna-XS-2.1-GGUF`), NOT
  "Laguna XS.2". The XS.2 repo does not exist as a GGUF release. User decision:
  rename throughout to XS 2.1, and do the work on a separate branch/worktree.
- 2026-07-25: No Q8_0 exists upstream. BF16 (63.8 GiB) downloaded instead.
  User decision: Task 10 quantizes **from BF16 directly** (signal path still
  targets Q8_0 baseline type; routed experts Q3_K with biased imatrix).
  Every "Q8_0 file" reference in the plan means the BF16 file.
- 2026-07-25: Downloads totalled ~83 GB (BF16 + Q4_K_M), vs ~54 GB originally
  authorized. Disclosed to user; 311 GB free remained.

## Ground truth (verified directly against the GGUF, not config.json)

- 40 layers, embd 2048, vocab 100352, 256 experts / 8 used, ff_exp 512,
  ff_shared 512, ff_dense 8192, swa 512, kv heads 8, head_dim 128
- head_count array `[48,64,64,64]`×10 → n_head_global 48, n_head_swa 64
  (n_head field, S21-style majority value, must be 64)
- leading_dense 1 → 39 sparse layers → 9,984 routed experts
- rot 64 / rot_swa 128; yarn: freq_base 500000, swa 10000, factor 32.0,
  orig_ctx 8192, beta_fast 64, beta_slow 1, attn_factor 1.0, rms_eps 1e-6
- Tensor names identical to ds4's existing weights_bind_laguna_layer
- Q4_K_M has per-layer Q4_K/Q6_K variation on attn_v, ffn_down_exps,
  ffn_down_shexp (llama.cpp heuristic, uncorrelated with the SWA pattern) —
  layout validation must tolerate it

## Task status

- Task 1: complete (commit 8dd4797, review clean — both verdicts PASS)
  - Minor finding (deferred to final review): xs2-facts.md section 3 Item 1
    table cites S 2.1 n_head=72 without an inline source citation (ds4.c:678).
- Task 2: complete (commits da0e4fa..d7a96fd, review clean — both verdicts PASS)
  - First implementer dispatch stalled mid-task (uncommitted correct code,
    no report, likely a Monitor/background-process mismanagement) — resumed
    the same agent synchronously with no background processes; it finished
    cleanly on the second pass. Final commit: d7a96fd.
  - Design decision: ds4_shape/ds4_model_family/ds4_variant moved from
    ds4.c (static) into ds4.h (extern) so tests can assert on the Laguna
    shape constants directly. Reviewer confirmed no duplicate-definition
    risk and that non-Laguna shapes remain static/private.
  - Minor findings (deferred to final review): (a) --ssd-streaming refusal
    message at ds4.c:57509 says "Laguna S 2.1" but is family-level and
    already blocks XS21 too — stale wording, correct behavior; (b) the new
    test only asserts static struct constants, not the runtime
    block_count-based selection/die path — real coverage arrives with
    Task 3's GGUF load.

- Task 3: complete (commits d7a96fd..4c1d7ab — 618e098 layout validation,
  4c1d7ab decode dispatch fix — review clean, both verdicts PASS)
  - Real bug found and fixed beyond the brief's literal scope, but within
    Task 3's own success criterion (resident bring-up = actually
    generating): `laguna_graph_forward_token`'s QKVG dispatch assumed any
    non-F16 attention weight was Q8_0, silently corrupting XS 2.1's Q4_K
    attention tensors (decode collapsed to UNK after token 1; prefill was
    unaffected since it already dispatched generically). Fixed by adding
    an explicit Q8_0 type check and a new branch using the pre-existing
    generic `laguna_graph_matmul` helper (4 calls: Q/K/V/Gate) instead of
    assuming Q8_0. F16 and true-Q8_0 paths (S 2.1) left byte-identical.
  - Scope decision (mine): full BF16 resident-inference support in ds4 is
    explicitly out of scope for this whole project — BF16 is only a
    quantize-source for Task 10's out-of-tree llama.cpp step, never
    something ds4 itself runs. `--inspect` on the BF16 file dies cleanly
    ("unsupported Laguna quantization layout marker bf16") — confirmed
    acceptable, no BF16 kernel work added anywhere.
  - Layout validation gaps found beyond the brief's generic "mirror S21"
    instruction: XS 2.1's legacy attention tensors (attn_q/k/gate/output)
    are Q4_K not F16 (S 2.1's older legacy recipe used F16); attn_v
    specifically varies per-layer Q4_K/Q6_K, mirroring the existing
    ffn_down_exps/ffn_down_shexp tolerance pattern.
  - Q4_K_M now generates coherent Python continuations (64 and 96 token
    runs verified), full test suite green.

- Task 4: complete (commits 4c1d7ab..026e90a — e6fa93b initial fixtures,
  86c55ca greedy-sampling fix, 026e90a README follow-up — review clean
  after one Important fix)
  - Real bug found and fixed beyond the implementer's own detection: the
    new `collect_local.py` never passed `--temp 0`, so all fixtures were
    randomly sampled (temp 1.0), not greedy/deterministic — would have
    made the "P0 baseline" useless as a stable comparison target for
    Task 11. Fixed, both fixture sets regenerated, rescored.
  - Residual finding after the fix: self-consistency (model scored
    against its own greedy output) only ~71% first-token-match, not
    ~100%. Investigated via a standalone diagnostic (not a plan task):
    ran the identical methodology against Laguna S 2.1 (untouched by
    this project) — got the same magnitude (~80%) on a 15-prompt sample.
    CONCLUSION: this is a pre-existing characteristic of the
    collect_local.py/score_official self-consistency comparison itself
    (built for scoring against hosted references, never validated for
    self-comparison), NOT an XS 2.1-specific bug and NOT related to
    Task 3's decode-dispatch fix. Leading (unverified) cause: likely
    score_official.c retokenizes printed continuation text independently
    rather than reusing originally-sampled token IDs (BPE round-trip
    artifact). Documented in
    gguf-tools/quality-testing/data/laguna-xs21/README.md.
  - Fixture set: 100 "general" cases (existing shared prompts.jsonl,
    local Q4_K_M generation, no OpenRouter key available in this
    environment) + 20 new "webpy" cases (15 web/Python + 5 tool-call,
    authored fresh, tracked at
    gguf-tools/quality-testing/prompts_laguna_xs21_webpy.jsonl).
  - Review's one Important finding (README didn't yet reflect the
    cross-family diagnostic, ran after that commit) — fixed and
    independently verified by the controller (not re-reviewed by a
    fresh reviewer agent, given it was a small, directly-checkable
    doc-only correction to one specific gap).
- Task 5: complete (commits 026e90a..4bed66e — 02e4360 unrelated plan
  doc fix, 4bed66e the actual guard change — review clean, zero findings
  at any severity)
  - `--ssd-streaming` now proceeds past the family-level guard for XS 2.1
    only; S 2.1 still refuses with the same message. Verified XS 2.1
    passes the guard and fails downstream in Metal prefill (expected —
    real streaming wiring is Task 6).

- Task 6: complete (commit bc963c9 — review not yet run)
  - Laguna's fused decode kernel now builds a per-selected-expert GPU
    address table via the same `ds4_gpu_stream_expert_cache_*` primitives
    GLM uses (family-agnostic plumbing); dispatches new Metal kernels
    (`kernel_laguna_q4_K_addr_routed_shared_pair_swiglu_f32` plus Q4_K/Q6_K
    addr-table down variants) when eligible, otherwise falls back
    unchanged to the existing mmap-wrap path. `laguna_graph_forward_token`/
    `_batch` now install correct model-map spans before touching weights.
  - Real bug found and fixed, same signature as Task 3's: down-pipeline
    kernel selection always picked the Q4_K address-table kernel whenever
    streaming was eligible, even though XS 2.1's official file pins its
    cache slab class to Q6_K — silently decoded Q6_K blocks with a Q4_K
    layout, collapsing decode to `<|UNK|>`. Fixed by selecting the kernel
    by actual down tensor type.
  - Smoke test: coherent output, `hits=7127 misses=2449 hit_rate=0.744`.
    `make ds4_test && ./ds4_test` green, independently re-verified solo
    by the controller after the fact.
  - Reviewed by two independent reviewers (A: PASS w/ 1 Important, B:
    FAIL). Both found the same Important finding: the report and this
    ledger described a `laguna_stream_decode_experts_addr_table_eligible`
    helper that never existed in bc963c9 — the shipped code reused the
    generic byte-uniform check unmodified. B additionally found an
    un-enforced cross-module invariant: host-side
    `weights_streaming_layer_experts_uniform` pins the slab class by BYTE
    SIZE only, while Metal's `stream_eligible` serves Q4_K/Q6_K down
    layers only, so a layer could be dropped from the resident span set
    yet be unservable from cache — neither mapped nor cached, Task 3's
    silent-corruption shape.
  - A first fix attempt was caught and rejected BEFORE it was built: it
    added a family-gated `exit(1)` premised on "validation guarantees
    down type is Q4_K or Q6_K". False — ds4.c:5150-5163 accepts
    `down == layer_routed_type` where routed type may be Q4_K, **Q3_K or
    Q2_K** (four types, not two), and the guard had no streaming
    condition despite sitting in a path ds4.c:47584 runs for resident
    Laguna too. It would have killed the existing
    `laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf` at startup in plain
    resident mode, and would reject Task 10-11's planned RoutedQ3_K
    artifact by construction.
  - Shipped fix instead makes the gap structurally impossible rather
    than fatal: `laguna_decode_experts_cache_servable` (ds4.c:6664)
    reports a non-Q4_K/Q6_K routed down layer as not-servable, so it
    stays in the resident span set and takes the same mapped-model
    fallback off-slab-class layers already use. Correct on every model,
    no barrier to later Q2_K/Q3_K work. Stale function-level comment
    (claimed Laguna needed no family special-case) corrected too.
  - Operational incident (not a code finding): the first implementer
    dispatch silently spawned its own background sub-agent instead of
    doing the work itself, then returned a false "done, nothing changed"
    result immediately without waiting on it. A second explicit dispatch
    was launched before this was discovered, so two agents worked the
    same worktree concurrently for ~55 minutes each (vanishing edits,
    competing `ds4_test` processes, one GPU OOM from two model loads at
    once). Both converged on the same single commit (bc963c9); tree was
    clean and tests green on independent solo re-verification afterward.
    No orphaned processes or background tasks remained. Documented in
    task-6-report.md. Lesson for future dispatches: explicitly forbid
    the Agent tool in implementer prompts (now standard practice).

- Task 7: complete (commit 003733f — tests/xs21_stream_ab.sh)
  - **PASSED**: all four prompts byte-identical resident vs streamed,
    under the plan's deliberately undersized 800-expert cache. Real
    eviction pressure confirmed present, so this is a meaningful gate and
    not a vacuous one: hits=11835 misses=7469 hit_rate=0.613
    evictions=6669 miss_pread=14.19 GiB. This validates Task 6.
  - Whole run takes ~24s (8 model loads) — mmap is lazy, so repeated
    "loads" are nearly free. Cheap enough to run after every engine change,
    which is what the plan intends it for.
  - Script adds two things beyond the plan's sketch: dumps both outputs +
    diff on mismatch, and refuses to run if another ds4/ds4_test is live
    (concurrency has already wedged this host once).
  - First clean uncontended resident-vs-streaming perf pair, same prompt,
    back to back (supersedes the contention-contaminated figures in
    task-6-report.md): resident prefill 379.90 t/s / gen 84.59 t/s;
    streamed prefill 197.93 t/s / gen 48.85 t/s. So ~42% decode and ~48%
    prefill penalty — but measured under worst-case cache pressure (800
    experts vs the ~9728 the auto-budget picks), so this is a floor, not
    the expected steady state.
  - Streaming peak footprint at ctx=8192 with the 800-expert cache: 4.04
    GiB task memory (2.36 GiB runtime + 1.52 GiB expert cache). Relevant
    to Task 8's 16 GB Mini target.

- Task 8: partially complete (commit 05965c9). Steps 1 + 3 done on the
  laptop; Step 2 (hardware acceptance run) still open.
  - **User decision 2026-07-27: retargeted from a 16 GB Mac Mini to a
    32 GB system.** 16 GB does not fit: at the plan's specified ctx 32768
    the non-routed weights (9.64 GiB) + KV (1.31) + graph scratch (3.97)
    = 14.92 GiB before a single cached expert, and Laguna refuses
    `--prefill-chunk` so scratch can't be reduced. The plan's Task 8/14
    text still says "Mini"/16 GB and should be read as 32 GB.
  - `xs21-q4` download target added (repo poolside/Laguna-XS-2.1-GGUF,
    revision 1a37c0a pinned per xs2-facts.md item 5). Verified: skips
    when the file already exists.
  - Expert-cache sweep (ctx 16384, greedy, 256 tok, Python prompt)
    saturates at ~3,600 experts — the 4800 and 6400 rows are byte-
    identical (3,615 misses, 6.87 GiB), so the whole working set is
    cached; below 1,600 SSD traffic grows 4.4x (30.00 GiB at 800).
    Context is nearly free once saturated: ctx 32768 costs +0.63 GiB over
    16384 at identical throughput. Recommended 32 GB config: 4000
    experts at ctx 32768 → 13.1 GiB footprint, ~22.7 GiB total including
    hot mmap pages, ~9.3 GiB left for OS + apps.
  - Throughput numbers are explicitly flagged optimistic in mini-notes.md:
    measured where the whole model fits page cache (14.19 GiB pread in
    368 ms = 38.6 GiB/s, i.e. RAM not storage). Memory/budget figures
    should transfer to 32 GB; t/s will not.
  - Gotcha found and documented: every `download_model.sh` target relinks
    `./ds4flash.gguf`, which is `ds4_test`'s default model. Running
    `xs21-q4` repointed the suite from the DeepSeek IQ2XXS model to
    Laguna XS 2.1; manually reverted, restore command recorded in
    mini-notes.md §6. Worth a guard or a warning in the script — deferred,
    out of Task 8's scope.
  - Remaining for Step 2 (needs the 32 GB machine): auto-budget sanity,
    warm decode t/s, hit rate after a 10-min agent session, memory
    pressure / `iogpu.wired_limit_mb`, and the multi-tool acceptance task.
    Checklist is mini-notes.md §7.

## Minor findings roll-up (for final whole-branch review triage)

1. Task 1: missing inline citation for S 2.1 n_head=72 in xs2-facts.md.
2. Task 2: stale "Laguna S 2.1" wording in the family-level streaming
   refusal message (ds4.c:57509) — now also correctly blocks XS21, just
   says the wrong name.
3. Task 2: test_laguna_variant_shapes only covers static constants; the
   block_count variant-selection and die-on-unsupported paths are
   untested until a real GGUF load exists (Task 3).
4. Task 3: attn_v layout-validation error check is slightly redundant
   with the immediately-following tensor_expect_layout call — harmless.

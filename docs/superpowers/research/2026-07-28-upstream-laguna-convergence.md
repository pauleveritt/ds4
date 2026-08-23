# Upstream Laguna convergence and the XS 2.1 branch

Research performed 2026-07-28 against upstream `laguna-s2.1` at `448d569` and
the XS campaign branch `laguna-xs2.1` at `a12297a`, whose merge base is
`3b0ec5f`. **Second pass, same day:** the open premise questions below were
resolved by reading the refactored upstream kernels directly; findings are
folded in and marked *verified*.

## Decision summary

Keep XS on its own branch and **do not integrate upstream with a plain
`git merge`**. The two lines edit the same Laguna graph functions for
different reasons, and the highest-consequence collision is one git resolves
*silently and incorrectly* rather than reporting as a conflict.

**Sequencing decision: run the 32 GB hardware acceptance first, on the
current XS base, before any convergence work.** Phase 2 engineering is
complete and acceptance is the sole remaining handoff; the quality fixtures
and self-consistency baseline were collected on the current base under the
old sampling defaults. Converging first would inject merge risk into a
nearly-finished deliverable and invalidate the baseline the acceptance run
needs — while buying XS no usable speedup, since DFlash and SSD streaming
are mutually exclusive (§Feature-level exclusions). Accept first; then the
rebase has a known-good reference build to diff against, and the fixture
question becomes a measured delta instead of a blind regeneration.

The recommended convergence path (after acceptance) is a deliberate reapply
of the XS work as a small number of logical patches onto the updated upstream
base, sequenced so that each patch is validated by acceptance machinery the
XS branch already carries. The second-pass verification below removed the
worst unknown: XS's central correctness premise **still holds** on updated
upstream, so the plan is no longer hostage to it.

## Topology

Upstream added 12 commits; XS added 50. Every XS commit is the XS campaign —
there is no unrelated engine work entangled in it — and the durable non-XS
documentation has since been extracted onto `paul/laguna`, so XS can be
rebased, rewritten, or abandoned without taking project docs with it.

| Upstream commit | Summary | Bearing on XS |
| --- | --- | --- |
| `448d569` | Tune Laguna sampling defaults | High — family-wide, XS inherits it unvalidated |
| `8f620f3`, `5f3c78d`, `0ee6d45` | DFlash confidence, cross-backend verify pipeline, quantized drafts | Medium |
| `31db3a8`, `30f4e17`, `6aec235` | DFlash and generation on CUDA / DGX Spark | Low (backend we do not target) |
| `62b59a4`, `6bd7fd3` | Laguna ROCm inference and DFlash | Low, but see multi-backend note |
| `bf605b1` | DFlash speculative decoding on Metal | High — restructures the batch graph |
| `5c5f95a` | Harden tagged agent tool-call decoding | Low — `ds4_agent.c` only, XS does not touch it |

## Overlap, classified by how it fails

Grouping by failure mode matters more than grouping by file, because the
classes need different work: Class B is resolved by reading a diff, Class A is
resolved only by knowing what the code was *for*.

### Class A — silent semantic collisions

These do not produce conflict markers. Git merges them cleanly and the result
is wrong or unvalidated.

**Chunked prefill is reverted-by-merge.** XS enabled chunked Laguna prefill
(`0e13e14`) partly by deleting `opt->prefill_chunk != 0 ||` from the Laguna
option-rejection block. Upstream kept that condition and, in `448d569`, added
"custom prefill chunk" to the adjacent error message text. Because only one
side changed the condition line, git takes XS's deletion without comment,
while the *message* a few lines below conflicts and gets human attention. The
likely outcome of a careless resolution is a build that accepts
`--prefill-chunk` while printing that it does not — or, if the resolver takes
upstream's block wholesale, chunked prefill silently disappears and the
XS prefill accounting tests (`538d2ab`, `30746b5`) start failing for a reason
nobody changed.

**New sampling defaults apply to XS without evidence.** `ds4_engine_sampling_defaults`
keys off `DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_LAGUNA`, which is
family-level, not variant-level. XS 2.1 is the same family, so it inherits
temperature 0.7, top-k 20, top-p 0.95, and min-p 0.05 the moment the merge
lands. The XS quality fixtures and self-consistency baseline (`e6fa93b`,
`86c55ca`) were collected under the previous defaults. Any fixture that pinned
`--temp 0` explicitly is unaffected; any that relied on defaults is now
comparing against a different sampler. This must be audited before the
fixtures are trusted as a regression signal.

Worth noting separately: this change also delivers what the roadmap backlog
tracks as `S-MIN-P-DEFAULT`. That item can likely be closed as *shipped
upstream* rather than executed, leaving only the `S-MIN-P-VS-TOP-P`
comparison as open work.

**The routed-MoE workaround's premise — *verified, still holds*.** XS's
prologue in `laguna_graph_forward_batch` widens the resident span set back to
every layer, justified by an explicit comment: the batch kernel
`ds4_gpu_glm_routed_moe_batch_tensor` "always wraps the WHOLE routed-expert
tensor ... it ignores its own `force_resident` argument." Upstream has since
refactored that function into a shared static
`ds4_gpu_glm_routed_moe_batch_tensor_impl`, adding
`ds4_gpu_glm_routed_moe_batch_decode_exact_q2_q3_tensor` and
`..._exact_q4_tensor` alongside the original wrapper. Reading the refactored
kernels settles it:

- The public wrapper still opens with `(void)force_resident;`
  (upstream `ds4_metal.m:36744`). The premise is unchanged; **the
  span-widening prologue stays**, and the largest `ds4.c` conflict must be
  resolved, not dropped.
- The two new exact-decode entry points do not take `force_resident` at all.
  `..._exact_q2_q3_tensor` is a separate ~260-line implementation (it does
  not route through `_impl`) and binds gate/up/down via
  `ds4_gpu_wrap_model_range` over the whole per-tensor range — no expert
  cache, no residency awareness.
- However, the exact-decode paths are **DFlash-verification-only**: they are
  gated on `exact_q8_rows = (row_argmax_out != NULL)`
  (upstream `ds4.c:49672`), and the only call sites passing a non-NULL
  argmax are the two DFlash verify paths (`s->dflash_graph.argmax`,
  upstream `ds4.c:68561`, `ds4.c:68638`). Since upstream rejects support
  models whenever `e->ssd_streaming` is set, these paths are unreachable
  under XS streaming. **The workaround does not need to be extended to
  them** — but their correctness under streaming now *depends on* the
  DFlash/streaming exclusion, which turns the exclusion test (below) from a
  UX nicety into a residency-correctness gate.

### Class B — visible textual conflicts

A dry-run merge produces seven conflict markers across two files. These are
the tractable ones.

| File | Markers | Nature |
| --- | --- | --- |
| `download_model.sh` | 3 | Both sides added a download target at the same insertion points (`xs21-q4` vs `laguna-dflash`). Keep both; purely mechanical. |
| `ds4.c` | 4 | The attention dispatch chain, the batch-graph token ingestion path, and one error-message string. |

The `ds4.c` conflicts concentrate in two functions. In the attention dispatch
chain, XS added a Q4_K legacy-attention branch for the official XS 2.1 layout
(`618e098`, `4c1d7ab`) while upstream added a ROCm pre-quantized Q8 branch to
the same `if`/`else` ladder; the two are independent and can coexist, but the
ladder has to be rebuilt by hand. In `laguna_graph_forward_batch`, XS inserted
the span-install prologue described above while upstream replaced the CPU
token-upload loop with an on-GPU draft-token copy (`gpu_draft_tokens`) and
extended the signature with `row_argmax_out` and a feature-capture parameter.
These edits are compatible in principle — one concerns MoE weight residency,
the other token ingestion — but they occupy the same lines.

### Class C — auto-merged but unverified

`ds4.h`, `ds4_gpu.h`, `ds4_metal.m`, and `tests/ds4_test.c` all changed
substantially on both sides (`ds4_metal.m`: roughly 858 lines from XS against
1,308 from upstream) and produced **no** conflict markers. Textual
independence is not semantic independence. Both sides touch the same Metal
command-buffer and encoder lifecycle primitives — `ds4_gpu_command_buffer`,
`ds4_gpu_compute_encoder`, `ds4_gpu_end_compute_encoder`,
`ds4_gpu_finish_command_buffer`, `ds4_gpu_get_pipeline` — and both add new
machinery around them: XS adds streamed expert-cache load/prune/peek
operations, upstream adds `ds4_gpu_discard_commands` and
`ds4_gpu_close_batch_encoder`.

The specific hazard is that DFlash deliberately keeps drafting and
verification inside **one** command buffer to avoid an intermediate completion
and CPU readback, gating on `ds4_gpu_commands_active()`. XS's expert-cache
loads introduce their own command-boundary requirements during prefill.

*Second-pass narrowing.* Reading both sides bounds this hazard considerably:

- XS never modifies the shared lifecycle helper **bodies**
  (`ds4_gpu_command_buffer`, `ds4_gpu_compute_encoder`,
  `ds4_gpu_end_compute_encoder`, `ds4_gpu_finish_command_buffer`) — it only
  adds new *callers*. After a merge, XS code automatically runs against
  upstream's batch-encoder-aware versions.
- Upstream's helpers now carry hidden state: inside the persistent batch
  command buffer, `ds4_gpu_compute_encoder` returns a shared `g_batch_enc`
  and `ds4_gpu_end_compute_encoder` is a **no-op**; any operation needing a
  real encoder boundary must call the new `ds4_gpu_close_batch_encoder`
  first (upstream added such calls to `ds4_gpu_tensor_copy`,
  `ds4_gpu_flush_encoder`, `ds4_gpu_submit_commands`, the TP paths, etc.).
  XS's Metal additions were written before this discipline existed.
- Upstream's own streamed-expert-cache machinery (which XS's Laguna cache
  rides) was already made batch-encoder-aware — `ds4_gpu_command_buffer`
  even hooks `ds4_gpu_stream_expert_cache_note_owned_created()` — so the
  cache/DFlash composition question is largely answered *by upstream
  itself*. What remains is XS-specific: the diagnostic forced-sync path in
  the routed-shared decode kernel (`q3_artifact_forced_batch_sync` →
  `ds4_gpu_end_commands()`) and any XS-added encoder sequence that assumes
  `end_compute_encoder` really ends encoding.

So the residual Class C work is a **bounded audit**: walk each XS-added
Metal call site of the lifecycle helpers and check it against the
batch-encoder rules, rather than an open-ended "do the disciplines compose"
investigation. At runtime the two features never co-execute (exclusion
below); the audit is about not corrupting encoder state for *non-streamed*
Laguna in the merged binary — the shared-GPU-state hazard `SWIFTSTAR.md`
already documents for `share_session_prefill_workspace`.

### Class D — genuinely disjoint

The Metal shader files do not overlap at all: XS touches `metal/moe.metal`,
upstream touches `metal/dense.metal`, `metal/dflash.metal`, and
`metal/laguna.metal`. The ROCm and CUDA backends are upstream-only and
irrelevant to a Metal-targeted XS. The tool-call parser hardening (`5c5f95a`)
lives entirely in `ds4_agent.c`, which XS never modifies.

## Feature-level exclusions

Three interactions are not merge problems at all — they are product facts that
survive any resolution.

**DFlash and SSD streaming are mutually exclusive.** Upstream rejects support
models when `ssd_streaming` is set: *"--ssd-streaming is not compatible with
support models yet."* SSD-streamed routed experts are the entire reason XS
exists (fitting Laguna on a 32 GB machine). So the headline upstream feature
and the headline XS feature cannot be used together today. XS is not made
obsolete by DFlash, but neither can it borrow the speedup.

*Verified:* the exclusion composes across the merge with no new code. XS
gates its streaming through the same `e->ssd_streaming` flag upstream's
support-model rejection checks (XS variant gate at engine open, `4bed66e`;
upstream rejection in `ds4_engine_open_internal`), and the two checks live in
different blocks that both survive a merge. Note this exclusion is now
**load-bearing for correctness**, not just product shape: it is the only
thing keeping the residency-blind exact-decode verifier paths unreachable
under streaming (§Class A). It must be covered by an explicit test.

**DFlash will not engage under the new defaults.** DFlash requires greedy
decoding and falls back automatically when sampling is stochastic. Upstream
simultaneously changed Laguna's default temperature from 1.0 to 0.7.
Consequently a user who follows the README's `ds4-agent` and `ds4-server`
invocations — neither of which passes `--temp 0` — gets no speculation at all
unless they set it explicitly. Any XS-side benchmark comparing "with and
without DFlash" must pin temperature, or it will measure nothing.

**Laguna is no longer Metal-only.** Upstream now admits Metal, CUDA, and ROCm
for Laguna. XS's streaming prologue calls `metal_graph_install_model_spans`
from inside `laguna_graph_forward_batch`, which is shared across backends. On
the updated base, XS must either gate that call by backend or accept that
XS streaming is Metal-only and say so explicitly at engine open.

## Resolution plan (revised after verification)

The former step 1 — re-read the refactored routed-MoE kernel — is **done**;
the finding is recorded in §Class A. The premise holds, the prologue stays,
and the exact-decode paths need no residency work while the DFlash/streaming
exclusion is enforced. With the plan-invalidating unknown removed, the
sequence is:

0. **Run the 32 GB hardware acceptance on the current XS base first.** This
   is Phase 2's sole remaining handoff and it must happen against the base
   the fixtures and self-consistency baseline were collected on. Everything
   below happens after acceptance, with that build as the reference.
1. **Rebase `laguna-xs2.1` onto the updated `laguna-s2.1`** as roughly six
   logical patches rather than replaying 50 commits: shape variant and
   per-variant head tables; legacy Q4_K attention layout; SSD-streaming gate;
   streamed expert cache; Q3 exact cache; chunked prefill. Replaying the full
   history would re-resolve the same `forward_batch` conflict a dozen times.
   The `forward_batch` prologue is reapplied as-is (premise verified), merged
   around upstream's new signature (`row_argmax_out`, feature capture,
   `gpu_draft_tokens` ingestion).
2. **Decide the prefill-chunk question explicitly** rather than inheriting it
   from a merge: keep XS's variant-gated guard, drop the
   `opt->prefill_chunk != 0 ||` clause from upstream's rejection block, and
   make the error message agree (it currently lists "custom prefill chunk"
   as unsupported).
3. **Batch-encoder audit of XS Metal additions** (the bounded Class C work):
   walk each XS-added call site of `ds4_gpu_compute_encoder` /
   `ds4_gpu_end_compute_encoder` and the diagnostic forced-sync path, and
   apply `ds4_gpu_close_batch_encoder` where a real encoder boundary is
   assumed. Mechanical once the rules in §Class C are in hand.
4. **Add a backend gate** for the Metal-only streaming prologue — Laguna now
   admits CUDA and ROCm upstream, and `metal_graph_install_model_spans` is
   called from shared code. Either gate by backend or reject XS streaming on
   non-Metal backends at engine open with a clear message.
5. **Re-run the XS acceptance suite on the rebased base**: the
   streamed-vs-resident A/B correctness gate (`003733f`), prefill chunk
   accounting (`538d2ab`, `30746b5`), and the Q3 cache equivalence tests.
   These exist already; the rebase's job is to keep them green.
6. **Add two new gates:**
   - `--dflash` with XS streaming fails cleanly with the documented message
     — this is now a residency-correctness gate, not a UX test (§Class A).
   - A same-binary smoke: DFlash on resident Laguna S and streamed XS
     generation both green from one build, so the merged Metal backend is
     shown to serve both feature sets (separately) without encoder-state
     corruption.
7. **Measure the fixture delta** against the new sampling defaults using the
   pre-rebase acceptance build as reference; regenerate only fixtures that
   relied on defaults rather than pinned values, and record the sampler
   change alongside. Any DFlash benchmark must pin `--temp 0` or it measures
   nothing (§Feature-level exclusions).

## What must still be measured

- The fixture/self-consistency delta under the new sampling defaults
  (step 7) — the only remaining quantitative unknown.
- Prefill throughput on the rebased base, to confirm upstream's batch-graph
  restructuring did not regress XS's chunked prefill accounting.

Resolved by this report's second pass, no measurement needed: the
`force_resident` premise (holds), whether the span prologue is redundant (it
is not), and whether the exclusion requires new enforcement code (it does
not — same flag).

## Acceptance signal

XS is considered converged when it sits as a small, readable patch series on
top of an unmodified upstream `laguna-s2.1`; the streamed-vs-resident A/B gate,
prefill accounting tests, and Q3 equivalence tests all pass on that base; the
quality fixtures are either shown unaffected by the sampling change or
regenerated with the change recorded; the DFlash/streaming exclusion and the
same-binary smoke are covered by explicit tests; and the batch-encoder audit
is recorded as done. Hardware acceptance on 32 GB precedes all of it and is
tracked separately as the Phase 2 handoff.

## Backlog notes

- `S-MIN-P-DEFAULT`: upstream `448d569` ships min-p 0.05 (with temperature
  0.7, top-k 20, top-p 0.95) as the Laguna family default — verified in
  `ds4_engine_sampling_defaults`. Recommend closing as **shipped upstream**;
  only `S-MIN-P-VS-TOP-P` remains open, now with a shipped default to argue
  against.
- `X-XS21-REBASE`: scope per the revised plan above; blocked behind the
  32 GB hardware acceptance by design, not by dependency.

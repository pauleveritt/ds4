# P2.5 — Q3_K streamed-cache equivalence plan

**Status:** brainstorming proposal, 2026-07-28.  This is the next blocking
Phase 2 item.  It deliberately plans a correctness proof before cache
admission, footprint measurement, hotlisting, or constrained-hardware work.

## Decision

Implement Q3_K address-table support as a narrow extension of the generic
`ds4_gpu_glm_routed_moe_one_tensor` path, but keep Q3 routed experts mapped
until a committed isolated test proves cache-address output equal to resident
output.

The first failed experiment is useful evidence: cache allocation and the Q3
gate/up pair worked exactly, while divergence began at the first Q3 down
output float. Cached bytes and cache GPU addresses were independently
verified; the prior mapped-model-address toggle was later found inconclusive
because the selected direct-slot kernel did not read its address table. The
investigation should therefore begin with a
minimal Q3 down invocation contract, rather than revisiting quantization, the
cache allocator, or Phase 1's Laguna dispatch.

The direct-buffer controls are exact for both one expert and eight selected
experts at production dimensions (2048 output / 512 mid), with a permuted
selected-id order. The raw-GPU-address candidate is exact over the same
fixture. The artifact-backed generic invocation is now also exact for a
rebound mmap model view, a direct bound owned full-tensor copy, and a raw
address into that owned copy. An initial zero result was a diagnostic bug: it
read a graph-batched command buffer before commit. Basic direct-buffer
binding, slot order, selected ids, Q3 down geometry, and raw-address
consumption are no longer leading hypotheses. The next rung is a
diagnostic-only cache-service integration using the actual selected cache
buffers and address table.

### Integration result (2026-07-28)

The first diagnostic-only Q3 address-table cache run did allocate and use the
real cache (800 entries, 1.01 GiB live, 5,323 hits / 4,349 misses for a
32-token greedy run), but its continuation diverged immediately from the
resident run. This is a useful, bounded failure: generic artifact Q3 pair/down
addressability is proven, while the actual cache-service integration remains
incorrect. Do not proceed to admission or footprint measurement. The next
diagnostic now reads back resident/cache gate-up `mid` and down output by
layer. It localizes the first mismatch to the Q3 pair stage at layer 1
(`mid[0]`: cache `4.06352e-06`, resident `0.0122323`); down is not yet
implicated. Isolate the cache gate/up address binding next.

## Current topology

Laguna XS 2.1 has Q8 shared experts, so its routed Q3 decode takes the generic
GLM-template kernel path rather than the Laguna fused routed/shared-Q4 path:

```text
Laguna decode
  -> ds4_gpu_glm_routed_moe_one_tensor
       resident Q3: pair + down kernels use mapped tensor bases
       cached Q4/Q2: address-table pair + down kernels use cached MTL buffers
       Q3 today: mapped resident fallback
```

Relevant existing pieces:

| Purpose | Reference |
|---|---|
| Resident Q3 pair | `metal/moe.metal:kernel_glm_q3_K_pair_swiglu_f32` |
| Resident Q3 down | `metal/moe.metal:kernel_glm_q3_K_down_f32` |
| Address-table pair pattern | `metal/moe.metal:kernel_glm_q4_K_addr_pair_swiglu_f32` |
| Address-table SIMD-down pattern | `metal/moe.metal:kernel_glm_q4_K_addr_down_simd_f32` |
| Host cache-table construction and dispatch | `ds4_metal.m:ds4_gpu_glm_routed_moe_one_tensor` |
| Decode-static span gate | `ds4.c:laguna_decode_experts_cache_servable` |

The current host eligibility intentionally admits only Q2/Q4 address-table
pipelines.  The Laguna static-span gate admits Q4/Q6 only.  Both are correct
as shipped and must remain so until the new proof succeeds.

## Design hypotheses to test

1. **Raw pointers are now proven for the relevant forms.** The old engine
   mapped-model-address toggle was ineffective because its direct-slot kernel
   did not read the table. The replacement artifact-backed check initially
   read stale graph-batch outputs, then passed exactly after an opt-in
   commit/wait. It covers rebound mapped view, owned full copy, and raw address
   to that copy. The remaining issue is the prior cache integration, not raw
   pointer provenance in the generic Q3 down invocation.
2. **The Q3 down path is the failure locus.** The experiment matched the
   complete gate/up intermediate exactly, then diverged at output byte 16,385
   (the first down float).  Cache bytes and CPU address-table entries were
   exact.  The remaining candidates include a binding-index/signature issue,
   Metal address-space/compiler behavior, dispatch geometry, or a subtle
   departure from the resident kernel.  It is not evidence of a defect in
   `ds4_glm_q3_K_dot2` itself, because the resident kernel uses that helper
   successfully.
3. **Cache service and residency are one contract.** Enabling a Q3 pipeline
   alone is incomplete: `laguna_decode_experts_cache_servable()` must change
   only after the pipeline is proven, otherwise cached Q3 can coexist with
   mapped Q3 and the static footprint will not fall.

## Proposed implementation seams

### 1. Isolated address-table kernels — no admission change

Do not re-add a production Q3 address-table kernel yet.  First create a
test-only down harness that dispatches a resident Q3 down reference and one
candidate binding form over the exact same controlled buffers.  Only after a
candidate is exact should its narrow production kernel be added.  The pair
variant may be retained as a later, independently tested seam because the
real-artifact diagnostic already established its arithmetic/path agreement.

The candidate variant must mirror the Q3 resident down kernel, rather than
porting Q4 arithmetic:

- `kernel_glm_q3_K_addr_down_f32`
  - accepts `down_addrs`;
  - resolves the selected expert's cached base once per slot;
  - retains the resident Q3 `row0`, `N_R0_Q3_K`, SIMD reduction, and
    `ds4_glm_q3_K_dot2` arithmetic exactly; only the expert base changes from
    `down + expert * down_expert_bytes` to
    `reinterpret_cast<device const char *>(down_addrs[expert])`.

Do not start from `kernel_glm_q4_K_addr_down_simd_f32`: its Q4 block decoding
is irrelevant and is a likely route to a plausible-but-wrong Q3 result.

The harness must also test a direct MTL-buffer binding form, but only as a
diagnostic control rather than a presumed fallback.  The real-artifact direct
slot attempt diverged too, so both forms need independent proof.  Preserve the
resident Q3 64-thread geometry and `(dim + 3) / 4` groups unless the isolated
test demonstrates a geometry mismatch.

At this stage, leave `laguna_decode_experts_cache_servable()` unchanged.
Normal Q3 model execution must still take the current safe mapped fallback.

### 2. Committed diagnostic ladder

The first test must isolate pointer indirection from engine routing and cache
policy.  Prefer a focused Metal test mode/harness over a text-generation-only
gate.  It should use deterministic, nonzero Q3_K blocks and the production
dimensions (2048 input/output, 512 expert mid, eight selected experts) so the
same `QK_K`, row stride, group geometry, and expert-table indexing execute.

Run these cases in one committed test, reading back GPU tensors after each:

| Level | Reference | Candidate | Required assertion |
|---|---|---|---|
| A | resident Q3 pair | address-table Q3 pair, same uploaded blocks | exact `mid` equality |
| B | resident Q3 down, supplied `mid` | address-table Q3 down, same uploaded blocks | exact `out` equality |
| C | resident Q3 pair + down | address-table Q3 pair + down | exact `mid` and `out` equality |
| D | normal selected ids | invalid id / zero address slot | same defined zero/skip behavior as existing address-table kernels |

"Exact" is appropriate here: resident and address-table variants should call
the same Q3 arithmetic and differ only in the initial base address.  If an
implementation forces a different reduction order, record and justify a
tight numerical tolerance before accepting it; do not silently weaken the
gate.

Use several block patterns, including nonzero high masks/scales and multiple
selected experts, so a zero-filled fixture cannot hide Q3 packing errors.
The fixture may be generated at test time, but its encoding must be explicit
and deterministic. The committed controls cover one and eight selected experts,
including direct-buffer and raw-address candidates at production shape. The
artifact-backed mapped-model, owned-copy, and raw-address checks are
diagnostic only; they now pass after synchronizing their graph batch. The
next integration check must use cache-style selected buffers/address table.
The local model must not become a portable regression fixture because it is an
untracked 14.64 GiB artifact.

### 3. Cache-integration proof

After A--D pass, enable `stream_addr_q3` only under an explicit diagnostic
switch.  Run a small Q3 model case that forces cache allocation and assert:

- live cache entries, hits, and bytes are nonzero;
- resident and cache-mode continuation (or final logit vectors) agree;
- cache buffer addresses are the inputs used by both pair and down;
- disabling the switch restores the mapped fallback.

This separates kernel correctness from the cache loader, deferred split path,
and address-buffer lifecycle.  Preserve the command and compact output as a
tracked transcript in `docs/superpowers/research/laguna-xs21-p25/`.

### 4. Admit Q3 only after proof

Once isolated and integration gates pass:

1. Make `stream_addr_q3` normal eligibility in
   `ds4_gpu_glm_routed_moe_one_tensor`.
2. Extend `laguna_decode_experts_cache_servable()` to accept Q3_K.
3. Confirm `model_map_span_vec_include_layer_decode()` now excludes the Q3
   routed tensors because cache eligibility and static-span construction agree.
4. Add an engine-level regression assertion for the expected static-span drop,
   not only cache statistics.

This order preserves the safe fallback if any intermediate implementation is
incorrect.

## Verification sequence after implementation

1. Build and run the isolated diagnostic ladder.
2. Run the existing `tests/xs21_stream_ab.sh` against the biased Q3 artifact;
   require bytewise resident/streamed output equality under the deliberately
   undersized 800-expert cache.
3. Capture startup and final cache statistics.  Require nonzero entries,
   hits, and live bytes; confirm all 39 sparse layers are served and Q3 routed
   tensors are absent from decode static spans.
4. Rerun the 800/1600/3200/4800 MiB context-16384, prefill-chunk-4096,
   256-token sweep.  Store command lines and compact raw output beside the
   result table.
5. Only then update `LAGUNA-XS21.md` with measured live footprint/tokens-per-
   second and decide whether P2.6 hotlisting remains worthwhile.

## Non-goals and stop conditions

- Do not change the quant artifact, imatrix, quality verdict, or P2.1
  accounting while diagnosing Q3 cache addressability.
- Do not present startup reservation, text that merely looks coherent, or
  cache hits without a static-span reduction as a footprint result.
- If A or B diverges, stop before changing cache admission.  Preserve a small
  readback diff that identifies the first divergent mid/output element.
- If isolated A--C pass but the integration proof fails, debug cache-buffer
  lifecycle/addresses separately; do not modify Q3 arithmetic speculatively.

## Alternatives considered

| Alternative | Decision | Reason |
|---|---|---|
| Enable Q3 and rely on `xs21_stream_ab.sh` | Reject | Text equality cannot localize pair/down/address defects and the prior experiment already produced plausible infrastructure signals with invalid output. |
| Copy the Q4 address-table down kernel | Reject | Q4 and Q3 block formats and decompression arithmetic differ. |
| Leave Q3 mapped and proceed to P2.6 | Reject | Hotlisting cannot solve the mapped static-span footprint blocker. |
| Ship uniform Q4 instead | Retain as fallback | P2.0 proved it cache-serves all layers and drops static span by 8.44 GiB, but the biased Q3 artifact is the current quality-viable candidate. |

## Relationship to the roadmap

This plan elaborates P2.5 in `docs/superpowers/LAGUNA-XS21.md` and records the
brainstorming decision, not an approved implementation.  The evidence and
Sol review that motivated its gates are in
`docs/superpowers/research/laguna-xs21-p25-q3-cache-blocker.md`.

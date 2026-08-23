# Laguna XS 2.1 Roadmap

Execution index for the Laguna XS 2.1 line: sequence, status, and links. The
designs in `docs/superpowers/specs/` and plans in `docs/superpowers/plans/` are
the source of truth for what each task does; the consolidated state of record is
[LAGUNA-XS21.md](LAGUNA-XS21.md).

**Branch layout.** This branch owns the XS 2.1 engine work and all of its
designs, plans, research, and fixtures. The durable documentation layer — the
SwiftStar design, the Laguna S 2.1 line, and the cross-project backlog — lives
on `paul/laguna` and is not duplicated here. Upstream is `laguna-s2.1`.

**Target:** 16 GB, via SSD-streamed routed experts over the uniform RoutedQ3_K
artifact. With streaming the artifact fits easily under 10 GB including context.
Laguna S 2.1 is a separate line targeting 64 GB class machines with full
residency, and remains explicitly unsupported for SSD streaming.

**Next task:** P2.8 — a real 16 GB constrained-hardware acceptance run. It is
the sole remaining Phase 2 handoff, and it must happen on the current base,
*before* any upstream convergence work, so the acceptance build becomes the
reference the rebase is diffed against.

Promotion and completion are phase-granular: when every task in an active
phase is done, move that phase and its table to **Completed**, adding a
`Completed` date column. Then promote the first planned phase.

---

## Active

### Phase 2 hardware acceptance

Phase 2 engineering is complete. What remains is operational: an acceptance run
on real constrained hardware, which has never been executed. Everything prior
was validated on a developer machine, and the local 32k smoke is an allocation
preflight — explicitly *not* a steady-state or performance claim.

| Phase | Summary | Design | Plan | State |
| --- | --- | --- | --- | --- |
| P2.8 | Real 16 GB constrained-hardware acceptance of the uniform RoutedQ3_K artifact, per the preflight checklist. | [design](specs/2026-07-25-laguna-xs2-streaming-design.md) | [checklist](plans/mini-notes.md) §7 | Next |

**Done condition:** `ds4-agent` runs Laguna XS 2.1 at 32k context on real 16 GB
hardware with the plan's correctness, memory, and quality gates met, using the
validated prefill cap and a 3,200-expert cache target.

**Footprint.** The recorded preflight plans 6.53 GiB (2.30 GiB context runtime
+ 4.03 GiB target cache) and measures a 6.46 GiB task footprint after eight
tokens, with 1.57 GiB cache live. That carries to 16 GB unchanged and needs no
re-sizing.

## Planned

### Upstream convergence

`X-XS21-REBASE` — reapply the XS campaign onto updated upstream `laguna-s2.1`
as roughly six logical patches rather than replaying 50 commits. Scope, verified
premises, conflict classification, and a seven-step sequence are in the
[convergence report](research/2026-07-28-upstream-laguna-convergence.md).

Sequenced *behind* hardware acceptance by deliberate choice, not technical
dependency: converging first would inject merge risk into a nearly-finished
deliverable and invalidate the fixture baseline the acceptance run needs, while
buying XS no speedup — DFlash and SSD streaming are mutually exclusive upstream.

## Completed

| Phase | Summary | Design | Plan | Completed |
| --- | --- | --- | --- | --- |
| Phase 1 | Streaming bring-up: XS 2.1 shape variant and per-variant head tables, official Q4_K_M legacy attention layout, SSD-streaming variant gate, streamed routed experts through the Metal expert cache, streamed-vs-resident A/B correctness gate, `xs21-q4` download target and sizing notes. | [design](specs/2026-07-25-laguna-xs2-streaming-design.md) | [plan](plans/2026-07-25-laguna-xs2-streaming.md) | 2026-07-27 |
| Phase 2 (P2.0–P2.7) | Footprint and quality: uniformity premise validated, graph-scratch investigation and chunked prefill, biased corpus builder, imatrix collection and artifact build, layout acceptance plus quality A/B, Q3 cache equivalence and footprint sweep, Python expert hotlist, acceptance preflight. | [design](specs/2026-07-25-laguna-xs2-streaming-design.md) | [uniformity probe](plans/2026-07-27-laguna-xs21-p20-uniformity-probe.md), [Q3 cache equivalence](plans/2026-07-28-laguna-xs21-p25-q3-cache-equivalence.md) | 2026-07-28 |

The original P0–P4 numbering in the streaming plan is superseded by the Phase 1
/ Phase 2 (P2.0–P2.7) sequence actually executed; the plan's Tasks 9–14 remain
the substrate and are retained as historical sequence in
[LAGUNA-XS21.md](LAGUNA-XS21.md) §8.

**Open minor findings** inherited from Phase 1 — stale streaming-refusal
wording, thin variant-shape test coverage, no unit tests on the streaming
decode/prefill path — are deferred to a final whole-branch review and tracked in
[LAGUNA-XS21.md](LAGUNA-XS21.md) §7 rather than as roadmap rows.

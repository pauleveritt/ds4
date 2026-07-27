# DwarfStar Superpowers Roadmap

The plans in `docs/superpowers/plans/` and designs in
`docs/superpowers/specs/` are the source of truth for what each task does.
This file is the cross-task index: sequence, status, and links. Deferred and
partial work belongs in the [roadmap backlog](superpowers/roadmap-backlog.md).

**Next task:** XS.2 P0, Task 1 — obtain the official Laguna XS.2 GGUFs and
record their ground-truth layout facts before changing the engine.

Promotion and completion are phase-granular: when every task in an active
phase is done, move that phase and its table to **Completed**, adding a
`Completed` date column. Then promote the first planned phase. Keep only
work that is ready to execute in Active; future work stays in the backlog.

**Maintenance (run when a task closes, or on request):**

1. Inspect `git log --oneline` since the last completed entry.
2. Check each active task against its plan and code; move complete phases to
   Completed.
3. If Active is empty, promote the first Planned phase.
4. Ensure every execution row links to an existing design or plan.

---

## Active

### Laguna XS.2 + SSD Streaming + Biased Quantization

Bring Laguna XS.2 to the 16 GB Mac Mini through a staged, Metal-only path:
first prove the official model layout and resident inference, then add SSD
streaming, and only then build the domain-biased quantization and hotlist.

| Phase | Summary | Design | Plan | State |
| --- | --- | --- | --- | --- |
| P0 | Record official GGUF facts and establish resident XS.2 bring-up. | [design](specs/2026-07-25-laguna-xs2-streaming-design.md) | [plan](plans/2026-07-25-laguna-xs2-streaming.md) | Next |
| P1–P4 | Add the XS.2 shape, streaming, biased quantization, and expert hotlist in the plan's order. | [design](specs/2026-07-25-laguna-xs2-streaming-design.md) | [plan](plans/2026-07-25-laguna-xs2-streaming.md) | Planned within initiative |

**Done condition:** `ds4-agent` runs Laguna XS.2 at 32k context on the target
16 GB Mac Mini with the plan's correctness, memory, and quality gates met;
Laguna S 2.1 remains explicitly unsupported for SSD streaming.

## Backlog

Future and partial work deliberately kept out of the active execution queue
lives in the [Superpowers Roadmap Backlog](superpowers/roadmap-backlog.md).

## Planned

No phase is queued for promotion after the active XS.2 initiative.

## Completed

No roadmap phase has been completed since this index was introduced.

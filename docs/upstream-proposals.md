# Upstream proposals

Outbound intentions: what SwiftStar would propose to `antirez/ds4` upstream so
that "upstream-bound" does not quietly become "carried forever." Each entry is
phrased as a proposal — problem, what the patch does, suggested upstream
shape. **Nothing here has been filed.** This is the intention record; filing is
a separate, later decision.

## Proposal #1 — A structured events mode for non-interactive agents (flagship)

**Problem.** A supervisor process (SwiftStar, or any external agent harness)
that spawns `ds4-agent` in `--non-interactive` mode can only observe the agent
by scraping ANSI-formatted text and a `+DWARFSTAR_STATUS` stderr marker. That
is brittle: color codes, cursor motion, and free-form prose all reach stdout,
and the supervisor must reverse-engineer state from presentation. There is no
machine-readable turn boundary, no structured tool-call stream, and no way to
distinguish thinking from answer text programmatically.

**What the patch does.** Adds `--json-events`, which makes every stdout line a
single JSON object with a `t` (type) field: `text`, `think`, `tool`,
`status`, `ready`, `queued`. Tool events carry a zero-based `idx` across all
seven phases (`start`/`tool`/`param_begin`/`param_value`/`param_end`/`output`/
`finish`); `param_begin` carries `name`; `finish` carries `calls` (and
`status` only when the block did not complete cleanly). A fail-closed
`agent_publish` backstop wraps any ungated `agent_publish` site as a generic
text event, so "all stdout is valid JSON when `--json-events` is on" is a
sink-enforced contract rather than an audited one. Wire format documented at
`docs/json-events.md`.

**Suggested upstream shape.** Land the flag + the wire behind it, documented.
The `idx`/`name`/`calls` fields and the fail-closed backstop are the parts
worth keeping verbatim; the per-site gates are an implementation detail a
upstream author may restructure. Acceptable to land piecemeal (e.g. `status`
events first, tool events later) — the wire is forward-compatible via the
`.ignored` unknown-type fallback.

## Proposal #2 — A first-class interrupt for non-interactive mode

**Problem.** A supervisor cannot cleanly interrupt a generation mid-turn when
`ds4-agent` runs `--non-interactive` (no TTY, so SIGINT does not reach
linenoise's handler).

**What the patch does.** `1f18723` supports interrupting a turn in
non-interactive mode; `9bca6d4` clears a stale-interrupt latch on the next
submit so a prior turn's interrupt does not kill the next turn's first
submission.

**Suggested upstream shape.** A first-class interrupt signal/command for
non-interactive mode, with the latch-clearing semantics. The two commits
retire together.

## Proposal #3 — Structured status emission for non-interactive mode

**Problem.** Outside `--json-events`, the only observable agent state is the
`+DWARFSTAR_STATUS` marker (`310d5a4`, `a9eda6d`).

**What the patch does.** Emits `+DWARFSTAR_STATUS` in non-interactive mode and
fixes idle-state emission so a state change bypasses the throttle.

**Suggested upstream shape.** If Proposal #1 lands, this marker is superseded
by the structured `status` events on the `--json-events` path; the marker
remains only for non-`--json-events` non-interactive runs. Retires when
upstream lands structured status emission, or when SwiftStar stops launching
ds4-agent without `--json-events`.

## Proposal #4 — Expose the startup memory plan via a first-class API

**Problem.** A supervisor must know the engine's memory plan at startup to
gate a feasible launch (SwiftStar's P3/P4) *before* committing to a model
load.

**What the patch does.** `83501bb` caches the startup memory plan and exposes
it via `ds4_engine_memory_plan`; `8267745` emits the session memory budget
(`kv_bytes`/`scratch_bytes`/`model_bytes`/`planned_bytes`) on the `ready`
event.

**Suggested upstream shape.** A first-class API for the memory plan, and the
`ready`-event budget fields. Retires when upstream exposes the plan directly.

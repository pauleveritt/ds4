# Fork ledger

> **Standing rule (canonical: `REBASING.md`):** the patch set instruments
> `ds4_agent.c`'s decode loops and emitters, so a rebase can apply cleanly and
> still be semantically wrong. Golden-fixture recapture against the real binary
> is mandatory on every submodule bump.

This ledger gives every divergence a row: *why it exists* and *what would
retire it*. A retired divergence is marked **retired** with the date and the
upstream commit — never deleted, so "carried forever" stays visible rather than
silently accumulating.

**Base:** `laguna-s2.1` (`448d569`, upstream `antirez/laguna-s2.1`). `main`
(`84cc882`) is the pristine antirez mirror, never edited, used as the rebase
source. `swiftstar-integration` = `laguna-s2.1` + the patch set.

## Patch-set divergences

| # | Divergence | Commits (orig → new) | Why it exists | What retires it |
|---|---|---|---|---|
| 1 | status marker (`+DWARFSTAR_STATUS`) | `310d5a4`→`cb4e3d7`, `a9eda6d`→`797c903` | ds4-agent in non-interactive mode must emit observable state; the pre-json-events scraper reads it. (`--json-events` `status` events supersede it on the agent wire, but it remains the mechanism outside `--json-events`.) | upstream lands structured status emission for non-interactive mode, **or** SwiftStar stops launching ds4-agent without `--json-events` and no consumer reads the marker |
| 2 | turn-interrupt | `1f18723`→`3a59c6a` | the app must interrupt a generation mid-turn from a spawned non-interactive process (no TTY, no SIGINT-to-linenoise path) | upstream lands a first-class interrupt signal for non-interactive mode |
| 3 | stale-interrupt latch | `9bca6d4`→`e07327e` | a stale interrupt from a prior turn must not kill the next turn's first submission | retires *with* turn-interrupt — a correctness detail of the interrupt mechanism, not independent |
| 4 | `--json-events` | `66c3de2`→`769bb9a` … `1a14a6c`→`5f7c666` (16) | the app needs a structured NDJSON wire (text/think/tool/status/ready/queued, `idx`, param `name`, finish `calls`) instead of scraping ANSI; documented at `docs/json-events.md` | upstream lands a structured events mode (flagship proposal #1); individual events can retire piecemeal as upstream lands them |
| 5 | startup memory plan | `83501bb`→`a328fb5`, `8267745`→`53dd1a4` | the app must know the memory plan at startup to gate feasibility (P3/P4) before load | upstream exposes the memory plan via a first-class API |
| 6 | integration structure (`swiftstar-integration` = `laguna-s2.1` + patch set) | — | the submodule must pin one branch carrying the model line + patches; fork hygiene, not an engine change | `laguna-s2.1` merges to `antirez/main` (then integration = `main` + patch set); ultimately the whole patch set lands upstream and the fork can be archived |
| 7 | wire handshake + per-event timestamps | (this commit) | binding rule 7: the wire announces itself — a `hello` handshake is the first NDJSON line and every event carries a monotonic `ts`; a consumer refuses a mismatch loudly instead of misparsing a skewed wire | upstream lands a structured events mode with a version handshake and timestamps (flagship proposal #1); retires with #4 |
| 8 | consent flags (`--workspace`, `--shell`) | `8450770`, `0a55a65` | the app's consent model is spawn-time and engine-enforced: `--workspace DIR` confines the file tools (fail closed) and sets the cwd; `--shell off` drops the bash family from the advertised schema and refuses them at dispatch. Without it a `grant`/`toggle` in the app would be cosmetic — the file tools `fopen(path)` with no confinement | upstream lands a wire-level tool-authorization protocol; the flags then become one implementation of it (retires with the P9 wire round-trip) |
| 9 | turn-outcome fields on `ready` | `da341d4`, `81a7dca`, `5c894c9` | binding the roadmap's outcome-telemetry requirement (`main 0ee5f6c`): the turn-end `ready` event carries `stop_reason`/`generated`/`ctx_used` — the wire previously had no stop reason at all (turn end was only inferable from `status.state` → `idle`), so a capture-grade turn outcome could not distinguish EOS from limit from context-full; `81a7dca`/`5c894c9` close the D12 interrupt paths so `stop_reason:interrupt` is honest across every turn-end site | upstream lands a structured events mode with turn-end semantics (flagship proposal #1); retires with #4 |

### Full original→new SHA mapping (cherry-pick order)

| orig | new | subject |
|---|---|---|
| `310d5a4` | `cb4e3d7` | agent: emit +DWARFSTAR_STATUS in non-interactive mode |
| `a9eda6d` | `797c903` | agent: fix idle state emission (state-change bypass) |
| `1f18723` | `3a59c6a` | agent: support interrupting a turn in non-interactive mode |
| `9bca6d4` | `e07327e` | agent: clear a stale interrupt latch on the next submit |
| `66c3de2` | `769bb9a` | agent: add --json-events with text and think events |
| `a3e49d9` | `02e7438` | agent: fix json-events buffer release on error paths and history replay |
| `334db1a` | `1e5b985` | agent: emit structured tool events under --json-events |
| `c8adbc1` | `4e8c94f` | agent: suppress the read-tool ANSI summary under --json-events |
| `7df204e` | `aea5795` | agent: emit status, ready and queued events under --json-events |
| `8bf46bc` | `c377b70` | agent: wrap unguarded publishes as text events under --json-events |
| `f9190ec` | `dd9bff5` | agent: fix status-event dedupe key and add per-call idx to tool events |
| `eda644d` | `7f6d204` | agent: flush pending prose before tool events, add time-based text streaming |
| `ae2c781` | `5fdcf04` | agent: buffer compaction's summary stream with UTF-8 holdback under --json-events |
| `8b6ecc4` | `369e7eb` | agent: fix four json-events robustness issues from deep review |
| `44202f5` | `4c22040` | agent: drop dangling incomplete UTF-8 tail on terminal compaction flush |
| `db8eb73` | `005164c` | agent: disambiguate param_begin by name, add finish call count, document wire format |
| `12ff1fa` | `67b48e7` | agent: fix-round 1 for json-events robustness review |
| `a86cd9f` | `120856e` | agent: fix-round 2 for json-events robustness review - choke-point fix |
| `b3d1600` | `f526b0c` | agent: fix-round 3 - reject overlong/surrogate UTF-8 in the escaper |
| `1a14a6c` | `5f7c666` | docs: state the cross-block output ordering guarantee and the leading-newline range |
| `83501bb` | `a328fb5` | feat: cache the startup memory plan and expose it via ds4_engine_memory_plan |
| `8267745` | `53dd1a4` | feat: emit the session memory budget on the ready event |

Development record: the SDD ledger
`.claude/worktrees/agent-mode/.superpowers/sdd/2026-08-20-agent-json-events/progress.md`
in `~/projects/ds4-control` is the authoritative history of how the 16
`--json-events` commits were built and reviewed.

Note on the original (orig) SHAs: they are historical identifiers from the
pre-fork DS4 Control lineage and are **not fetchable from this fork** — treat
them as provenance references only. The durable authority for that lineage is
the `progress.md` ledger above.

## Development branches

Model lines, not patch divergences. The app never pins them; a line enters the
shipped integration only when SwiftStar ships a `Variant` for it.

- `laguna-s2.1` — upstream mirror (`448d569`); already in the integration (P1).
- `laguna-xs2.1` — local dev branch (`bcf1b1c`), pushed for a durable home;
  enters the integration when SwiftStar ships an XS `Variant` (P12+).
- `mellum-2.1-overnight` — canonical Mellum line (`94ffd86`... actually
  `1da0204` on this fork), pushed for a durable home; enters the integration
  when SwiftStar ships a Mellum `Variant` (P12). Retirement context: the SFT
  snapshot `JetBrains/sft_mellum_v23_mixopt_fullx5_p2_joint-iter-3015` @
  `683ca310…` (2026-08-21) self-describes the canonical `MellumForCausalLM`
  config with split RoPE — removing the reason the branch's fixtures drifted to
  the public Thinking release — while the GGUF/MLX conversion + imatrix
  pipeline remains the branch's largest unpriced work item.

## Contingencies

- **`d0b0caa` (pre-deletion preservation, 2026-08-21).** The pre-existing
  `pauleveritt/ds4` (parented to `notatestuser`) carried one unique commit,
  `d0b0caa` ("Fix merge: restore `wrap_f32_decode_model_range`, `force_model_view`
  param and callers", 38 lines in `ds4_metal.m`). It is a merge-conflict fix
  restoring two laguna-line Metal symbols dropped when `laguna-s2.1` was merged
  into `notatestuser/ds4` v2. Those symbols are native to upstream
  `laguna-s2.1` (5/7 occurrences) and absent from both antirez main and
  notatestuser v2, so P1's `laguna-s2.1`-based integration carries them natively
  and `d0b0caa` is expected to be redundant. **Trigger to restore:** if a future
  rebase drops `wrap_f32_decode_model_range` or `force_model_view` (or the
  recapture/build shows a missing-symbol error), restore from the preserved
  bundle at `~/projects/pauleveritt/swiftstar/.worktrees/ds4-fork-backup/ds4-control-patches-v3.bundle`
  (SwiftStar-side, gitignored) and add a row above. Do **not** port it
  preemptively.

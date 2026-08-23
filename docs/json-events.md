# `ds4-agent --json-events` wire format

Reference for anything consuming `ds4-agent`'s stdout when it is started with
`--json-events`. This document describes the format as implemented in
`ds4_agent.c`; it is derived directly from that source, not from any design
brief, so treat it as authoritative over comments elsewhere. Written so a
parser can be implemented from this file alone, without reading
`ds4_agent.c`.

## Enabling it

`--json-events` is only accepted together with `--non-interactive`; passing
it without `--non-interactive` is a startup error (`ds4-agent` exits before
loading a model). One practical consequence used throughout this document:
**every code path gated on "interactive mode" is unreachable while
`--json-events` is active**, because non-interactive is always on.

## Envelope

Every line `ds4-agent` writes to stdout is a complete, newline-terminated
JSON object (NDJSON — one object per line, no line straddles two objects).
Every object has a `"t"` field naming its kind:

```
hello | text | think | tool | status | ready | queued
```

The **first** non-blank line is always the `hello` handshake (see below).
Every object carries a `"ts"` field — monotonic microseconds since engine
start (`clock_gettime(CLOCK_MONOTONIC)`) — including the handshake. `ts` is
present on every event kind; consumers that do not model a given kind ignore
its `ts` along with the rest of the unmodelled line.

This is enforced structurally, not just by convention: every internal
emitter builds one full JSON object before writing anything, and a backstop
(`agent_publish()`) wraps *any* byte that would otherwise reach stdout
outside those emitters into `{"t":"text","s":"<escaped>"}\n`. So even a
future emission site that forgets to build a proper event still produces a
valid NDJSON line — worst case it is a spurious `text` event, never a raw,
unwrapped byte on the wire.

All string field bodies are escaped by the same routine: `"`, `\`, `\n`,
`\r`, `\t` are escaped, and any other byte `< 0x20` becomes `\u00XX`. Bytes
`>= 0x80` are copied through untouched (see the invalid-UTF-8 quirk below).

## Event kinds

### `hello`

```json
{"t":"hello","v":1,"caps":["status","ready","text","think","tool","queued","ts"],"ts":<µs>}
```

The version/capability handshake, emitted once as the first line when
`--json-events` is active, before any other event. `v` is the wire format
version (currently `1`); `caps` names the event kinds the engine may emit plus
`"ts"` (per-event monotonic timestamps). A consumer that requires a capability
not listed, or a version it does not know, must refuse loudly rather than
continue on a wire it may misparse. With `--host-tools` (P9) `caps` additionally
includes `"tool_request"` — see "Host tools" below.

### `text`

```json
{"t":"text","s":"..."}
```

- `s` (string, always present, never empty): a chunk of the assistant's
  visible reply text (`<think>...</think>` content excluded — that becomes
  `think` events instead).

Assistant text is buffered, not emitted byte-by-byte. The buffer for the
*current* renderer flushes as a `text` (or `think`) event on any of:
a think/text mode switch, a 4096-byte size threshold, 100ms elapsed since
the buffer's last flush, a tool block starting (so preceding prose is not
delayed behind that block's tool events), or the turn finishing. A `text`
event's `s` never mixes think and non-think content — the buffer is flushed
before the mode flips.

The `agent_publish()` backstop described in "Envelope" above is a second,
independent path to a `text` event: any call site that writes through
`agent_publish()`/`agent_publishf()` without going through the buffered
renderer emits its bytes as one immediate, unbuffered `text` event (no
4096-byte/100ms batching). In the current source most narration call sites
are explicitly gated to avoid stdout output under non-interactive mode (and
therefore under `--json-events`, which requires it), so this path is
primarily a defensive safety net for a missed or future gate rather than
something known to fire in normal operation today. Do not assume every
`text` event necessarily went through the 4096-byte/100ms buffering
described above.

### `think`

```json
{"t":"think","s":"..."}
```

- `s` (string, always present, never empty): a chunk of the model's
  `<think>...</think>` content, with the surrounding tags stripped.

Same buffering/flush rules as `text` (they share one buffer, keyed by
whichever mode is currently active).

### `tool`

```json
{"t":"tool","phase":"<phase>","idx":<int>, ...}
```

- `phase` (string, always present): one of `start`, `tool`, `param_begin`,
  `param_value`, `param_end`, `output`, `finish` — see below.
- `idx` (int, always present on every phase): the zero-based index, within
  the *current* DSML tool-call block, of the call this event belongs to.
  See "The `idx` contract" below.

Per-phase fields, in the order they appear on the line:

| phase | extra fields | notes |
|---|---|---|
| `start` | none | Fires once, when a tool-call block opens. `idx` is always `0` (nothing has been announced yet). |
| `tool` | `name` (string) | Fires once per call, when that call's tool name becomes known. `name` is the tool name (`read`, `bash`, `edit`, ...). |
| `param_begin` | `kind` (string), `name` (string) | Fires once per parameter, when that parameter's value starts streaming. `kind` is a coarse styling category (see table below); `name` is the parameter's own name as written in the model's tool call (e.g. `start_line`, `max_lines`, `path`, `command`). Multiple parameter names can share one `kind` — `name` is what disambiguates them. |
| `param_value` | `s` (string) | Zero or more per parameter: the parameter's raw value, chunked (see below). Omitted entirely for an empty-string parameter value (no `param_value` event fires between that parameter's `param_begin` and `param_end`). |
| `param_end` | none | Fires once per parameter, when its value closes. |
| `output` | `s` (string) | See "Which calls get an `output` event" below — **not every tool call produces one.** |
| `finish` | `status` (string, optional), `calls` (int, always present) | Fires once per block, when the block closes. `idx` here is the *last real call's* index (or `0` for a block with zero calls — see the `idx` contract). `status` is present whenever the block did **not** complete cleanly — a DSML parse error (`"[invalid tool call: ...]\n"`), the block still being open when generation was interrupted (`"[tool call interrupted]\n"` or, for the edit-preflight case, `"[tool call stopped: edit old selector failed]\n"`), or a hard turn failure (`"[tool call failed: ...]\n"`) — and absent when the block completed normally. `calls` is the actual number of calls announced in this block (`0` for a block with none) — added specifically so `finish idx:0` for one call and `finish idx:0` for zero calls are distinguishable. |

`param_value` chunking: a parameter's raw value bytes are buffered and
flushed as a `param_value` event at a 4096-byte threshold (holding back any
trailing incomplete UTF-8 sequence so a multi-byte character is never split
across two events) and again, unconditionally, as a final flush when the
parameter closes. There is no time-based flush for `param_value` (unlike
`text`/`think`). A parameter's value may therefore arrive as zero (empty
value), one, or many `param_value` events.

`kind` vocabulary (`agent_tool_param_kind_for`, `agent_tool_param_kind_str`):

| kind | parameter names that map to it | scope |
|---|---|---|
| `bash_command` | `command` | only on tool `bash` |
| `diff_old` | `old` | only on tool `edit` |
| `diff_new` | `new` | only on tool `edit` |
| `path` | `path`, `file`, `filename` | any tool |
| `offset` | `line`, `start_line`, `end_line`, `offset`, `start`, `end`, `count`, `max_lines`, `timeout_sec`, `refresh_sec` | any tool |
| `content` | `content`, `text` | any tool |
| `normal` | anything else (e.g. search's `pattern`, `bash_status`'s `job`) | any tool |

Because `kind` is many-to-one, two parameters on the same call frequently
share a `kind` (e.g. a `read` call's `start_line` and `max_lines` both map to
`offset`) — a consumer must use `name`, not `kind`, to tell them apart.

**Which calls get an `output` event.** Only `bash`, `bash_status`, and
`bash_stop` calls (via `agent_bash_publish_observation`), plus the internal
"unknown tool name" error fallback, ever emit a `tool`/`output` event. The
other built-in tools — `read`, `more`, `write`, `list`, `edit`, `search`,
`google_search`, `visit_page` — never do: their results are appended
directly to the model's own transcript and never surface on the
`--json-events` wire at all. A consumer that wants to show, say, a `read`
call's result has nothing to render from `output` for it; it must be
reconstructed from that call's `param_begin`/`param_value`/`param_end`
events (and the fact that the call reached `finish`) if a preview is wanted
at all. A very large `output` body is capped and, when truncated, gets a
trailing `"\n[output truncated: N of M bytes shown]\n"` note appended inside
`s` itself (not a separate field).

### `tool_request`

```json
{"t":"tool_request","idx":<int>,"name":"<tool>","params":[{"name":"<p>","value":"<v>"},...],"ts":<µs>}
```

Emitted only under `--host-tools` (P9; see "Host tools" below), one per call
in a tool-call block, in place of internally executing the call. `idx` is the
same zero-based, block-scoped index the `tool` event carries; `name` is the
tool name; `params` is the call's parameters as `{"name","value"}` objects in
the order the model wrote them. `ts` is the per-event monotonic timestamp.
Requires `--json-events` (the gate refuses `--host-tools` without it), so a
consumer that negotiated `"tool_request"` in `hello` `caps` knows every line
with `"t":"tool_request"` is a request it is expected to answer over stdin
with a `tool_result` line (see "Host tools" below).

### `status`

```json
{"t":"status","state":"...","prefill_done":0,"prefill_total":0,"prefill_tps":0.0,"generated":0,"gen_tps":0.0,"ctx_used":0,"ctx_size":0,"power":0,"error":"..."}
```

All fields are always present, in this order:

| field | type | meaning |
|---|---|---|
| `state` | string | one of `idle`, `prefill`, `generating`, `compacting`, `draining`, `saving`, `error`, `stopped` |
| `prefill_done` | int | prompt tokens processed so far |
| `prefill_total` | int | prompt tokens to process |
| `prefill_tps` | float, 1 decimal | prefill tokens/sec |
| `generated` | int | tokens generated so far this turn |
| `gen_tps` | float, 1 decimal | generation tokens/sec |
| `ctx_used` | int | context tokens used |
| `ctx_size` | int | effective context size |
| `power` | int | configured power-throttle percentage |
| `error` | string | last error message, or `""` |

Emission is throttled: a `status` event is published immediately when
`state` changes (never delayed), and otherwise at most once per 200ms while
any field differs from the last event actually published. An update whose
full field set is byte-identical to the last published one is suppressed
even if 200ms has elapsed (exact-repeat dedup) — so a consumer cannot assume
`status` arrives on a fixed cadence, only that it arrives promptly on state
changes and is throttled (not silently dropped) otherwise.

### `ready`

```json
{"t":"ready","kv_bytes":1685774336,"scratch_bytes":6146715648,"model_bytes":48254631936,"planned_bytes":56087121920}
```

The `--json-events` replacement for the `+DWARFSTAR_WAITING` stderr marker:
emitted when the worker is idle, has no queued/pending input, and is waiting
for more stdin. Only emitted in the persistent (piped-stdin) non-interactive
loop, never in one-shot (`-p`) mode.

Besides `t`, the four `*_bytes` fields carry the session's startup memory
budget:

| field | type | meaning |
|---|---|---|
| `kv_bytes` | uint64 | raw + compressed KV cache |
| `scratch_bytes` | uint64 | per-context prefill scratch |
| `model_bytes` | uint64 | resident model span |
| `planned_bytes` | uint64 | total, including buffers and reserves |

These values are constant for the life of the session — KV is allocated
upfront for the whole context, so none of them vary with usage — and are
identical on every `ready` of a session; a consumer may take the first and
ignore the rest. They are **absent** (only `{"t":"ready"}` is emitted)
when the engine has no memory plan to report (opened with `ctx_size <= 0`),
so a parser must treat all four fields as optional.

When a turn has ended, `ready` additionally carries that turn's outcome
snapshot (P7, divergence #9):

```json
{"t":"ready",...,"stop_reason":"eos","generated":128,"ctx_used":8192}
```

| field | type | meaning |
|---|---|---|
| `stop_reason` | string, optional | why the turn ended |
| `generated` | int, optional | tokens generated this turn |
| `ctx_used` | int, optional | context tokens used, as of the turn end |

`stop_reason` is one of:

| value | meaning |
|---|---|
| `eos` | the stop token ended the final generation round (clean non-tool exit) |
| `limit` | generation hit the configured token limit |
| `context_full` | the context was already full, so generation could not start |
| `interrupt` | a latched interrupt ended the turn (during prefill, generation, or compaction) |

Like the memory-plan fields, these three are **absent from the startup
`ready`** (no turn has ended yet) and **repeated unchanged on every later
`ready`** until the next turn ends — a consumer that drops a `ready` line
and re-reads on the next one recovers the last turn's outcome. Treat all
three as optional. Before this field landed the wire had no stop reason at
all, and turn end was only inferable from `status.state` transitioning to
`idle`, so a capture-grade turn outcome could not distinguish EOS from
limit from context-full.

### `queued`

```json
{"t":"queued"}
```

No fields besides `t`. The `--json-events` replacement for the
`+DWARFSTAR_QUEUED` stderr marker: emitted when a newly-submitted prompt is
placed on the queue instead of dispatched immediately (worker busy or
dispatch failed). Same one-shot exclusion as `ready`.

## The `idx` contract

- Zero-based.
- Scoped to the *current* DSML tool-call block: it resets to `0` at that
  block's `start` event and is only ever incremented from there.
- It increments exactly once per call, right before that call's own `tool`
  event — never anywhere else. So for calls 0, 1, 2, ... in one block, `idx`
  on their `tool`/`param_begin`/`param_value`/`param_end` events is 0, 1, 2,
  ... in order.
- `start` always carries `idx:0` (nothing has been announced when it fires).
- `finish` carries the *last real call's* `idx` — for one call that is `0`;
  for a block with **zero** calls it is also `0`, which is why `finish` also
  carries `calls` (see the `tool` phase table above) — `idx` alone cannot
  distinguish "one call at idx 0" from "no calls" on `finish`.
- `output` events carry the `idx` of the call whose execution produced them
  (passed straight through from tool dispatch), not necessarily the block's
  *current* streaming idx (dispatch happens after the whole block has
  finished streaming).
- A consumer must scope every `idx` to the block it fell within — i.e. to
  the most recent `start` — not treat it as unique across the whole session
  or turn. `idx` restarts at `0` for every new block.

## Ordering guarantees

Every event is written into one mutex-protected, append-only output buffer
and drained to stdout in append order, so **the order events are appended is
the order they appear on the wire.** Within that constraint, a consumer may
rely on:

- Within one block: `start` (idx 0) is first; `finish` is last, and follows
  every call's `param_end`.
- Within one call: `tool` precedes that call's `param_begin` /
  `param_value`* / `param_end` sequence; each parameter's `param_begin`
  precedes its own `param_value`* and `param_end`.
- A call's parameters are announced (`param_begin`) in the order the model
  wrote them in its tool call.
- Across calls in one block: `idx` is non-decreasing and increments by
  exactly 1 from one call's `tool` event to the next.
- `output` events (where they occur at all) always arrive *after* their
  block's `finish` — tool execution happens only once the whole block has
  finished streaming — and are dispatched and published one call at a time,
  strictly in `idx` order (dispatch is a sequential loop, not concurrent).
- **A block's `output` events all arrive before the next block's `start`.**
  Tool dispatch is synchronous on the single worker thread, and the model
  cannot write a new tool block until it has seen the prior block's results
  in context. This matters because `idx` restarts at `0` every block: a
  consumer that keys tool state by `idx` and clears that state at `start`
  relies on this guarantee to avoid attributing a late `output` to a
  same-`idx` call in the *following* block.
- A `status` event whose `state` differs from the previous one is emitted
  immediately, never delayed by the throttle.

A consumer must **not** rely on:

- Any fixed byte size or fixed count of events for a single logical unit of
  content: `text`/`think`/`param_value` chunk boundaries depend on timing
  and size thresholds, not on semantic boundaries (except that
  `param_value` chunk boundaries always fall on whole UTF-8 codepoints).
  One "line" as a human would read it can be split across several events;
  conversely several distinct pieces of content can be merged into one
  event's `s`.
- Every tool call producing an `output` event — most do not (see the `tool`
  phase table above).
- A precise interleaving of `status` events against concurrently-streaming
  `text`/`tool` events beyond what is stated above — status is checked and
  possibly emitted once per iteration of the non-interactive main loop,
  which runs concurrently with the worker thread that produces text/tool
  events, so their exact interleaving at sub-iteration granularity is not
  specified.
- An explicit turn-end signal — there isn't one (see quirks).

## Known quirks (accepted, not bugs)

These are consumer-facing behaviors identified in review and deliberately
left as-is. A parser should account for them rather than be surprised by
them.

- **A think section's first following `text` event starts with a leading
  newline, usually two.** When `</think>` closes, the renderer unconditionally
  writes one `\n`, plus a second leading `\n` if the last thinking byte
  written was not itself already a newline (the common case, since model
  "thinking" text rarely ends exactly on a newline at the closing tag). That
  byte or byte pair becomes the start of whatever `text` buffer follows —
  an artifact of the ANSI-era paragraph separator between thinking output
  and the reply. Do not treat a leading `"\n\n"` (or occasionally just
  `"\n"`) on the first post-think `text` chunk as meaningful content.
  **Strip all leading whitespace rather than matching a fixed count.** The
  one-or-two above describes only what the renderer itself writes; the
  model's own first content bytes may then add more. A real capture
  contains post-think items beginning with three newlines, so a consumer
  matching the literal `"\n\n"` will under-handle this.
- **Invalid UTF-8 emitted by the model is silently dropped, not passed
  through.** The JSON string escaper validates every multi-byte sequence
  (lead-byte class, correct continuation-byte count, and — per RFC 3629 —
  the narrower first-continuation-byte range required for overlong
  encodings and the UTF-16-surrogate range) before copying it into a JSON
  string body; `"`, `\`, and control bytes `< 0x20` are escaped as usual.
  Any byte that is not part of a complete, well-formed sequence — a torn
  multi-byte character cut by a buffering/size threshold, a lone
  continuation byte, an overlong encoding, a surrogate half, or anything
  else a byte-level tokenizer can emit directly — is dropped from the
  output rather than copied in raw. A consumer therefore never sees invalid
  UTF-8 on the wire, but this means the `s`/`status`/`name` text a consumer
  receives is not always byte-identical to what the model generated: bytes
  can silently vanish rather than arrive mangled. This can affect
  `text`/`think` event bodies and `param_value`/`output`/`tool`/`finish`
  string fields alike, since all of them pass through the same escaper.
- **There is no turn-end event.** A consumer must infer "the turn is over"
  from the `status` event's `state` transitioning from `generating` to
  `idle`. This is reliable specifically because state changes bypass the
  200ms status throttle (see "Ordering guarantees" above), so that
  transition is never delayed or coalesced away.
- **`agent_publish_system_status()`'s narration never reaches the consumer
  under `--json-events`.** That function (source of messages like "Stopped
  by user" and "Compaction interrupted; keeping the previous conversation
  state.") unconditionally returns without publishing anything when
  `cfg->non_interactive` is true — and `--json-events` requires
  `--non-interactive`, so this function is always a no-op while
  `--json-events` is active. These narration messages simply never appear on
  the wire, in any form.
- **A single logical line of output can arrive as several small `text`
  events.** This is a direct consequence of the buffering described under
  the `text`/`think` events above: the 100ms-elapsed and 4096-byte flush
  triggers are not aware of line boundaries, so a run of assistant prose can
  be cut mid-line (or even mid-word) across two or more consecutive `text`
  events. A consumer must reassemble `text` (and `think`) content by
  concatenation, not assume one event equals one line or one paragraph.

## Consent flags (P7; no wire impact)

Two spawn-time flags added in P7 (divergence #8) gate the engine's tool
surface but change **no event kind, field, or ordering guarantee** on this
wire:

- **`--shell on|off`** (default `on`): when `off`, the three bash tools
  (`bash`, `bash_status`, `bash_stop`) are dropped from the advertised tool
  schema and refused at dispatch — a call to any of them returns a tool
  error instead of executing. On the wire this surfaces only as a `tool`
  event whose `output` is that error string (the bash tools are among the
  few that emit `output`; see the `tool` phase table); no new event kind,
  field, or ordering is introduced.
- **`--workspace DIR`**: sets the agent's cwd (unless `--chdir` was given
  explicitly) and confines the six file tools `read`/`more`/`write`/`list`/`edit`/`search` to `DIR`,
  failing closed — a call whose resolved path escapes `DIR` returns a tool
  error rather than touching an arbitrary path. (`more` is confined
  transitively: the confined `read` sets the `w → more_path` it reuses, so a
  `more` launched from a confined `read` cannot escape either.) The
  streaming-time `agent_preflight_edit_old` selector — `edit`'s preflight
  read of the `old` text against `DIR` — is confined too: an `old` whose
  resolved path escapes `DIR` aborts the block with the `[tool call
  stopped: edit old selector failed]` finish status rather than reading an
  arbitrary path. On the wire this is invisible: the file tools never emit
  `output` (see "Which calls get an `output` event"), and a confined or
  refused call differs from a normal one only in its result text (or, for
  the edit preflight, a `finish` `status`), not in any event shape.

Both are parsed at startup and enforced in `ds4_agent.c`; neither adds a
field to any event on this wire. They are noted here only so a consumer
knows the advertised tool surface can be narrower than the built-in set,
and that this narrowing is invisible to the wire format itself.

## Host tools (`--host-tools`; P9, the bidirectional wire)

`--host-tools` (default off; requires `--json-events`) makes the wire
**bidirectional**: the engine **requests** tool execution instead of performing
it. The app (or a fake app) owns execution and answers each request over stdin.
Absent the flag, behavior is byte-for-byte unchanged — the bare CLI keeps
internal execution and emits no `tool_request`.

The protocol is two NDJSON lines keyed by `idx`:

- **Request** (stdout, event kind `tool_request`, one per call in the block):
  ```json
  {"t":"tool_request","idx":<int>,"name":"<tool>","params":[{"name":"<p>","value":"<v>"},...],"ts":<µs>}
  ```
  `idx` is the existing block-scoped index (see "The `idx` contract"). The
  engine emits the request and then **blocks** reading one newline-terminated
  line from stdin for that call's result.
- **Result** (stdin, one NDJSON line the host writes back):
  ```json
  {"t":"tool_result","idx":<int>,"ok":true|false,"s":"<condensed result text>"}
  ```
  The engine matches by `idx` and returns `s` as the call's result text,
  wrapped in the same `Tool result N (name):\n…` envelope as an
  internally-executed call, so the downstream result→KV path is identical. A
  result with `ok:false` is a **refusal**: the engine returns a fixed
  `Tool error: host refused the tool call` text (the host's `s` is discarded).
  A result whose `idx` does not match the call that requested it, or any line
  that is not a valid `tool_result` object, is a **loud refusal** (binding
  rule 7's spirit) — `Tool error: host tool_result idx mismatch …` or
  `… protocol violation …` — rather than trusting a result the engine cannot
  associate with the request that prompted it. EOF on stdin before a result
  line arrives is likewise a refusal (`… EOF before a result line arrived`).

**Stdin ownership during a dispatch.** The blocking read runs on the worker
thread (the same thread the generation/decode loop already blocks on); while
it is in flight, `host_tool_reading` gates the non-interactive loop's stdin
poll so the result line is not drained into the prompt buffer — the worker is
the sole stdin reader for the block's duration. The host writes a `tool_result`
line only after reading the matching `tool_request` from stdout, so the line the
worker blocks for is unambiguous.

The `hello` handshake advertises `"tool_request"` in `caps` iff `--host-tools` is
active, so a consumer can refuse loudly if it does not understand the kind.

## Note on this document's relationship to the brief that requested it

The brief that requested this document described the "several small text
events" fragmentation quirk as specifically about `agent_publish_system_status`.
Investigation of the current source found that function is unconditionally
a no-op under `--json-events` (see the quirk above), so it cannot be the
source of any fragmentation actually observable on the wire. The verified,
reachable source is the generic `text`/`think` buffering thresholds
described above, which is what this document records instead.

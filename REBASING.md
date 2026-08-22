# Rebasing this fork

`pauleveritt/ds4` is the SwiftStar engine fork. `main` is a pristine mirror of
`antirez/main` — never edit it. The shipped integration branch is
`swiftstar-integration`, which the SwiftStar submodule pins.

## Standing rule (load-bearing)

> The patch set instruments `ds4_agent.c`'s decode loops and emitters, so a
> rebase can apply cleanly and still be semantically wrong. Golden-fixture
> recapture against the real binary is mandatory on every submodule bump.

`git rebase` completing without conflict is **not** a success signal. The wire
contract lives at `docs/json-events.md`; the golden fixture lives in the
SwiftStar repo at `fixtures/agent/golden.ndjson` (NDJSON) and
`fixtures/server/golden.sse` (SSE), each with a timestamp sidecar. After any
rebase, rebuild both binaries (`make ds4-server ds4-agent`) and recapture
against the real engine before the bump lands. If the recapture diverges from
the documented wire contract, the rebase broke something git could not see —
re-derive the offending patch against the new upstream; never hand-edit the
capture to match.

## How to rebase the patch set onto new upstream

`swiftstar-integration` = `laguna-s2.1` (the model line) + the 22 app-required
patches on `patch-set`. "Union of model lines," not "main + lines" — at P1 the
one model line is `laguna-s2.1`.

1. Fetch antirez: `git fetch upstream`.
2. Update `laguna-s2.1` to mirror `upstream/laguna-s2.1` (fast-forward only).
3. Recreate `patch-set` from the updated `laguna-s2.1`, then re-apply the
   22-commit series (see `docs/fork-ledger.md` for the original→new SHA mapping).
   `git cherry-pick` is the tool; expect conflicts confined to `ds4_agent.c`
   (the file the patches instrument).
4. Build: `make ds4-server ds4-agent` — must be clean, zero warnings.
5. Recapture (see the standing rule above) — the gate, not the cherry-pick.
6. Move `swiftstar-integration` to the new `patch-set` tip and push.
7. In SwiftStar, bump the submodule (`git -C external/ds4 checkout` + parent
   gitlink commit) and re-run `just engine` + recapture.

## When antirez merges laguna into main

When `laguna-s2.1` eventually lands in `antirez/main`, the model-line merge
becomes upstream's responsibility. At that point `swiftstar-integration` should
be rebased onto `main` (the model line is now in main), and the `laguna-s2.1`
development branch retired. The standing rule still applies — recapture after.

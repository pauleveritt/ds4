#!/usr/bin/env python3
"""Collect local-GGUF continuations for quality-fixture snapshots.

This is the no-hosted-API counterpart to collect_official.py. It drives the
`ds4` binary directly (greedy, deterministic) and writes the same
prompts/continuations/manifest.tsv layout that collect_official.py produces,
minus the responses/ directory and response_file manifest column (there is no
hosted API response to retain provenance for).

Use this when no hosted reference for a model family is available (no API
key, or the model isn't listed by any hosted provider) but a stable
byte-for-byte snapshot of local generation is still useful as a quality
baseline for future GGUF variants (e.g. comparing a later quant against this
one's captured continuations).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_prompts(path: Path) -> list[str]:
    prompts = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") if isinstance(obj, dict) else None
            if not prompt:
                raise SystemExit(f"bad prompt row in {path}: {line[:120]}")
            prompts.append(prompt)
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return prompts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds4", default="./ds4", help="path to the ds4 binary")
    ap.add_argument("--model", required=True, help="GGUF model path passed to ds4 -m")
    ap.add_argument("--prompts", required=True, help="prompts.jsonl path")
    ap.add_argument("--out", required=True, help="output fixture directory")
    ap.add_argument("--count", type=int, default=10_000)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--think-mode", choices=("nothink", "think", "think-max", "default"),
                     default="nothink")
    ap.add_argument("--lock-file", default=None, help="value passed via DS4_LOCK_FILE env")
    ap.add_argument("--extra-arg", action="append", default=[],
                     help="additional ds4 CLI arg, may repeat")
    args = ap.parse_args()

    prompts = load_prompts(Path(args.prompts))
    out = Path(args.out)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    (out / "continuations").mkdir(parents=True, exist_ok=True)

    manifest = out / "manifest.tsv"
    rows = []
    total = min(args.count, len(prompts))

    think_flag = {
        "nothink": "--nothink",
        "think": "--think",
        "think-max": "--think-max",
        "default": None,
    }[args.think_mode]

    env = None
    if args.lock_file:
        import os
        env = dict(os.environ)
        env["DS4_LOCK_FILE"] = args.lock_file

    print(f"model={args.model} ds4={args.ds4} think_mode={args.think_mode} "
          f"max_tokens={args.max_tokens}", file=sys.stderr)

    for i, prompt in enumerate(prompts[: args.count]):
        case_id = f"case_{i:03d}"
        print(f"local {i + 1}/{total}: {case_id}", file=sys.stderr, flush=True)
        cmd = [args.ds4, "-m", args.model, "-n", str(args.max_tokens), "--temp", "0"]
        if think_flag:
            cmd.append(think_flag)
        cmd.extend(args.extra_arg)
        cmd.extend(["-p", prompt])
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"error: {case_id} exited {result.returncode}\n{result.stderr[-2000:]}",
                  file=sys.stderr)
            raise SystemExit(1)
        content = result.stdout
        if not content.strip():
            print(f"warning: empty continuation for {case_id}", file=sys.stderr)

        prompt_path = out / "prompts" / f"{case_id}.txt"
        cont_path = out / "continuations" / f"{case_id}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        cont_path.write_text(content, encoding="utf-8")
        rows.append((case_id, prompt_path, cont_path))

    with manifest.open("w", encoding="utf-8") as fp:
        fp.write("# id\tprompt_file\tcontinuation_file\n")
        for row in rows:
            fp.write("\t".join([row[0], str(row[1]), str(row[2])]) + "\n")
    print(f"wrote {manifest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

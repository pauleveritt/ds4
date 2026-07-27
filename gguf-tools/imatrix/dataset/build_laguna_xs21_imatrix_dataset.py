#!/usr/bin/env python3
"""Build the deterministic Laguna XS 2.1 imatrix calibration corpus.

The corpus is generated entirely in-tree, so it requires no network access.
Its prompt rendering mirrors render_laguna_chat_prompt_text in ds4_server.c,
including the Laguna tool-call and tool-response spellings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

EOS = "〈|EOS|〉"
DEFAULT_SYSTEM = (
    "You are a helpful, conversationally-fluent assistant made by Poolside. "
    "You are here to be helpful to users through natural language conversations."
)
TARGET_BYTES = 3_000_000 * 4
TARGET_SHARES = {"web_python": 0.55, "prose_reasoning": 0.30, "shell_c": 0.15}
FORBIDDEN = re.compile(r"\b(?:java|c#|kotlin|php|go|rust)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Record:
    rid: str
    category: str
    mode: str
    source: str
    rendered: str


def tag_body(text: str, closing: str) -> str:
    """Match the server: escape only the matching closing delimiter."""
    return text.replace(closing, "&lt;" + closing[1:])


def render_call(name: str, arguments: dict[str, object]) -> str:
    body = []
    for key, value in arguments.items():
        value_text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        body.extend((
            "<arg_key>", tag_body(key, "</arg_key>"), "</arg_key>",
            "<arg_value>", tag_body(value_text, "</arg_value>"), "</arg_value>",
        ))
    return f"<tool_call>{name}{''.join(body)}</tool_call>"


def render(messages: list[dict[str, object]], mode: str,
           tools: list[dict[str, object]] | None = None) -> str:
    """Mirror the Laguna branch of render_chat_prompt_text_for_syntax."""
    system, start = DEFAULT_SYSTEM, 0
    if messages and messages[0]["role"] in ("system", "developer"):
        system, start = str(messages[0].get("content", "")), 1
    tool_schemas = "\n".join(
        json.dumps(tool, separators=(",", ":"), ensure_ascii=False) for tool in tools or []
    )
    out = [EOS]
    if system.strip() or tool_schemas or mode == "think":
        system_text = system.rstrip()
        if tool_schemas:
            if system_text:
                system_text += "\n\n"
            system_text += (
                "### Tools\n\n"
                "You may call functions to assist with the user query.\n"
                "All available function signatures are listed below:\n"
                f"<available_tools>\n{tool_schemas}\n</available_tools>"
            )
        out.extend(("<system>", system_text, "</system>\n"))

    pending_assistant = False
    for message in messages[start:]:
        role = message["role"]
        content = str(message.get("content", ""))
        if role in ("system", "developer"):
            out.extend(("<system>", content, "</system>\n"))
        elif role == "user":
            out.extend(("<user>", content, "</user>\n"))
            pending_assistant = True
        elif role in ("tool", "function"):
            out.extend(("<tool_response>", tag_body(content, "</tool_response>"),
                        "</tool_response>\n"))
            pending_assistant = True
        elif role == "assistant":
            if mode == "think":
                prefix = f"<think>{message.get('reasoning', '')}</think>"
            else:
                prefix = "</think>"
            out.extend(("<assistant>", prefix, content))
            for name, arguments in message.get("calls", []):
                out.append(render_call(name, arguments))
            out.append("</assistant>\n")
            pending_assistant = False
    if pending_assistant:
        out.extend(("<assistant>", "<think>" if mode == "think" else "</think>"))
    return "".join(out)


WEB_SNIPPETS = (
    """from pathlib import Path
import json

def {name}(root: Path) -> list[dict]:
    records = []
    for path in root.rglob('*.json'):
        with path.open(encoding='utf-8') as handle:
            records.append(json.load(handle))
    return records
""",
    """type {name}Props = {{ projectId: string; pending: boolean }};

export function {name}Panel(props: {name}Props) {{
  const [query, setQuery] = useState('');
  const visible = useMemo(() => filterRows(query), [query]);
  return <section aria-busy={{props.pending}}>{{visible.map(renderRow)}}</section>;
}}
""",
    """<main class="workspace">
  <nav aria-label="Project">{navigation}</nav>
  <article><h1>{title}</h1><pre>{diagnostic}</pre></article>
</main>
""",
    """.workspace {{ display: grid; grid-template-columns: minmax(14rem, 1fr) 3fr; gap: 1.25rem; }}
.workspace pre {{ overflow: auto; border: 1px solid color-mix(in srgb, currentColor 20%, transparent); }}
@media (max-width: 48rem) {{ .workspace {{ grid-template-columns: 1fr; }} }}
""",
)
PROSE_TOPICS = (
    "a migration plan for a persistent cache with an explicit rollback point",
    "the invariant that makes a bounded queue safe under backpressure",
    "how to distinguish a measured result from an arithmetic projection",
    "a decision record for choosing a smaller prefill chunk",
    "a review of an incident timeline with missing observations called out",
    "why calibration data must resemble production prompts",
)
SHELL_C_SNIPPETS = (
    """int {name}(const uint8_t *src, size_t len, uint32_t *out) {{
    if (!src || !out || len < 4) return -1;
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) value |= (uint32_t)src[i] << (8 * i);
    *out = value;
    return 0;
}}
""",
    """set -eu
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
curl --fail --location --retry 3 "$url" -o "$tmp_dir/artifact"
shasum -a 256 -c "$checksum_file"
mv "$tmp_dir/artifact" "$destination"
""",
)
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "apply_patch", "parameters": {
        "type": "object", "properties": {"patch": {"type": "string"}}, "required": ["patch"]}}},
]


def web_messages(index: int) -> tuple[list[dict[str, object]], list[dict[str, object]] | None]:
    name = f"load_workspace_{index:04d}"
    blocks = "\n".join(snippet.format(
        name=name, navigation="{navigation}", title="{title}", diagnostic="{diagnostic}"
    ) for snippet in WEB_SNIPPETS)
    messages: list[dict[str, object]] = [
        {"role": "system", "content": "Be precise about APIs, state ownership, and failure paths."},
        {"role": "user", "content": (
            f"Review web and Python change set {index}. Identify correctness, accessibility, "
            f"and streaming-memory risks; then give a minimal patch plan.\n\n{blocks}")},
    ]
    if index % 4 == 0:
        messages.extend((
            {"role": "assistant", "reasoning": "Inspect the relevant source before proposing a change.",
             "content": "", "calls": [("read_file", {"path": "src/workspace.py"})]},
            {"role": "tool", "content": (
                "def filter_rows(query):\n"
                "    return [row for row in rows if query.casefold() in row.name.casefold()]")},
            {"role": "user", "content": "Now provide the smallest safe change and its test cases."},
        ))
        return messages, TOOLS
    return messages, None


def prose_messages(index: int) -> tuple[list[dict[str, object]], None]:
    constraints = "\n".join(
        f"- Consider constraint {item}: retain evidence, state uncertainty, and avoid hidden assumptions."
        for item in range(1, 13)
    )
    return ([
        {"role": "system", "content": "Reason from supplied facts. Separate evidence, inference, and recommendation."},
        {"role": "user", "content": (
            f"Write a structured analysis of {PROSE_TOPICS[index % len(PROSE_TOPICS)]}.\n\n"
            f"{constraints}\n\nCase identifier: {index}.")},
    ], None)


def shell_c_messages(index: int) -> tuple[list[dict[str, object]], None]:
    code = "\n".join(snippet.format(
        name=f"parse_frame_{index:04d}", url="https://example.invalid/artifact",
        checksum_file="artifact.sha256", destination="artifact.gguf",
    ) for snippet in SHELL_C_SNIPPETS)
    return ([
        {"role": "system", "content": "Review low-level code conservatively; preserve error handling and quoting."},
        {"role": "user", "content": (
            f"Audit this shell and C excerpt for bounds, ownership, and quoting errors. "
            f"Propose tests.\n\n{code}")},
    ], None)


def stable_id(category: str, mode: str, source: str, rendered: str) -> str:
    digest = hashlib.sha256(f"{category}\0{mode}\0{source}\0{rendered}".encode()).hexdigest()
    return f"laguna-xs21-{digest[:16]}"


def records_for(category: str, index: int) -> list[Record]:
    maker = {
        "web_python": web_messages,
        "prose_reasoning": prose_messages,
        "shell_c": shell_c_messages,
    }[category]
    messages, tools = maker(index)
    source = f"generated:{category}:{index:05d}"
    rows = []
    for mode in ("nothink", "think"):
        rendered = render(messages, mode, tools)
        rows.append(Record(stable_id(category, mode, source, rendered),
                           category, mode, source, rendered))
    return rows


def build_records() -> list[Record]:
    wanted = {category: int(TARGET_BYTES * share) for category, share in TARGET_SHARES.items()}
    totals = {category: 0 for category in TARGET_SHARES}
    records: list[Record] = []
    for category in TARGET_SHARES:
        index = 0
        while totals[category] < wanted[category]:
            for row in records_for(category, index):
                records.append(row)
                totals[category] += len(row.rendered.encode("utf-8"))
            index += 1
    for row in records:
        if FORBIDDEN.search(row.rendered):
            raise ValueError(f"forbidden-language reference in {row.rid}")
    return records


def write_outputs(outdir: Path, records: list[Record]) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda row: (row.category, row.source, row.mode))

    def write_rendered(path: Path, rows: list[Record]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    f"\n\n===== LAGUNA_XS21_IMATRIX_PROMPT {row.rid} {row.category} "
                    f"{row.mode} {row.source} =====\n{row.rendered}"
                )

    write_rendered(outdir / "laguna_xs21_rendered_prompts.txt", records)
    write_rendered(outdir / "laguna_xs21_rendered_prompts_nothink.txt",
                   [row for row in records if row.mode == "nothink"])
    write_rendered(outdir / "laguna_xs21_rendered_prompts_think.txt",
                   [row for row in records if row.mode == "think"])

    bytes_by_category = {category: 0 for category in TARGET_SHARES}
    categories = {category: 0 for category in TARGET_SHARES}
    modes = {"nothink": 0, "think": 0}
    for row in records:
        bytes_by_category[row.category] += len(row.rendered.encode("utf-8"))
        categories[row.category] += 1
        modes[row.mode] += 1
    total = sum(bytes_by_category.values())
    manifest: dict[str, object] = {
        "version": 1,
        "purpose": "Laguna XS 2.1 Python/web-biased imatrix calibration prompts",
        "renderer": "ds4_server.c:render_laguna_chat_prompt_text",
        "record_count": len(records),
        "rendered_utf8_bytes": total,
        "rough_token_estimate_bytes_div_4": total // 4,
        "composition_contract": TARGET_SHARES,
        "composition_actual": {category: bytes_by_category[category] / total
                               for category in TARGET_SHARES},
        "categories": categories,
        "modes": modes,
        "bytes_by_category": bytes_by_category,
        "forbidden_languages_absent": True,
        "files": {
            "all_rendered": "laguna_xs21_rendered_prompts.txt",
            "nothink_rendered": "laguna_xs21_rendered_prompts_nothink.txt",
            "think_rendered": "laguna_xs21_rendered_prompts_think.txt",
        },
    }
    (outdir / "laguna_xs21_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output directory (defaults to this script's directory).")
    args = parser.parse_args()
    manifest = write_outputs(args.out or Path(__file__).resolve().parent, build_records())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

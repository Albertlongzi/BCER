from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def write_tool_manifest(*, tools_list: List[Dict[str, Any]], tool_index: Dict[str, Any], ctx_dir: Path) -> None:
    """
    Persist a tool manifest as both JSON and Markdown for:
    - human debugging
    - prompt/context injection
    """
    ctx_dir.mkdir(parents=True, exist_ok=True)
    tool_manifest_json = ctx_dir / "tool_manifest.json"
    tool_manifest_md = ctx_dir / "tool_manifest.md"
    tool_names = sorted([t.get("name") for t in tools_list if isinstance(t, dict) and "name" in t])
    notes = [
        "register_to_reference processes exactly ONE moving series per call.",
        "Prefer canonical sequence labels: T2w, ADC, DWI_high_b / DWI_low_b / DWI_mid_b (avoid img1/img2).",
        "Naming conventions documented at docs/naming_schema.md.",
    ]
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "tool_names": tool_names,
        "tool_index": tool_index,
        "notes": notes,
        "tools": [
            {
                "name": t.get("name"),
                "description": (
                    (t.get("description", "") or "")
                    + (" NOTE: one moving series per call." if str(t.get("name")) == "register_to_reference" else "")
                ),
                "input_schema": t.get("input_schema", {}),
                "output_schema": t.get("output_schema", {}),
                "tags": t.get("tags", []),
                "version": t.get("version"),
            }
            for t in tools_list
            if isinstance(t, dict) and t.get("name")
        ],
    }
    tool_manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Simple Markdown for easy browsing
    lines = [
        "# Tool Manifest",
        "",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- n_tools: {len(manifest['tools'])}",
        "",
    ]
    lines.append("## Notes")
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")
    for t in manifest["tools"]:
        lines.append(f"## `{t['name']}`")
        desc = (t.get("description") or "").strip()
        if desc:
            lines.append(desc)
        lines.append("")
        lines.append("### input_schema")
        lines.append("```json")
        lines.append(json.dumps(t.get("input_schema", {}), indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
        if t.get("output_schema"):
            lines.append("### output_schema")
            lines.append("```json")
            lines.append(json.dumps(t.get("output_schema", {}), indent=2, ensure_ascii=False))
            lines.append("```")
            lines.append("")
    tool_manifest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

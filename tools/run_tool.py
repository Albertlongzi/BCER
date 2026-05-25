from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict

from commands.schemas import ToolContext, ToolResult, _to_jsonable
from tools.catalog import build_registry


def _load_context(path: Path) -> ToolContext:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ToolContext(
        case_id=str(raw["case_id"]),
        run_id=str(raw["run_id"]),
        run_dir=Path(raw["run_dir"]),
        artifacts_dir=Path(raw["artifacts_dir"]),
        case_state_path=Path(raw["case_state_path"]),
    )


def _normalise_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, ToolResult):
        return payload.to_dict()
    if isinstance(payload, dict):
        return _to_jsonable(payload)
    return {"ok": False, "data": {}, "error": f"Tool returned unsupported payload type: {type(payload).__name__}"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one BCER tool from JSON files.")
    ap.add_argument("--tool-name", required=True)
    ap.add_argument("--args-file", required=True)
    ap.add_argument("--context-file", required=True)
    ap.add_argument("--result-file", required=True)
    ns = ap.parse_args()

    result_path = Path(ns.result_file)
    try:
        args = json.loads(Path(ns.args_file).read_text(encoding="utf-8"))
        ctx = _load_context(Path(ns.context_file))
        registry = build_registry(include_experimental=True)
        tool = registry.get(ns.tool_name)
        payload = _normalise_payload(tool.func(args, ctx))
    except Exception as exc:
        payload = {
            "ok": False,
            "data": {},
            "artifacts": [],
            "warnings": [],
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "stack": traceback.format_exc(),
            },
        }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if payload.get("ok", True) else 0


if __name__ == "__main__":
    raise SystemExit(main())

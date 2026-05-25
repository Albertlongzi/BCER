from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_args(args: Dict[str, Any], *, max_items: int = 6) -> str:
    if not isinstance(args, dict):
        return ""
    parts: list[str] = []
    for i, (k, v) in enumerate(args.items()):
        if i >= max_items:
            parts.append("...")
            break
        if isinstance(v, str):
            vv = v
            if "/" in vv or "\\" in vv:
                vv = Path(vv).name or vv
            if len(vv) > 48:
                vv = vv[:45] + "..."
            parts.append(f"{k}={vv}")
        elif isinstance(v, (int, float, bool)):
            parts.append(f"{k}={v}")
        elif isinstance(v, list):
            parts.append(f"{k}=[len:{len(v)}]")
        elif isinstance(v, dict):
            parts.append(f"{k}={{keys:{len(v)}}}")
        else:
            parts.append(f"{k}={type(v).__name__}")
    return ", ".join(parts)


def summarize_outputs(outputs: Dict[str, Any], *, max_items: int = 6) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(outputs, dict):
        return out

    # Prefer path-like keys first.
    keys = sorted(outputs.keys(), key=lambda x: (0 if x.endswith("_path") else 1, x))
    for k in keys:
        if len(out) >= max_items:
            break
        v = outputs.get(k)
        if isinstance(v, str):
            if k.endswith("_path"):
                out[k] = str(Path(v).name)
            elif len(v) <= 64:
                out[k] = v
        elif isinstance(v, (int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = f"list(len={len(v)})"
        elif isinstance(v, dict):
            out[k] = f"dict(keys={len(v)})"
    return out


def event_record(
    *,
    event_type: str,
    tool_name: str,
    status: str,
    case_id: str,
    run_id: str,
    args_summary: str = "",
    duration_s: Optional[float] = None,
    outputs: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
    attempt: int = 1,
) -> Dict[str, Any]:
    return {
        "event_type": str(event_type),
        "timestamp": now_iso(),
        "tool_name": str(tool_name),
        "args_summary": str(args_summary),
        "status": str(status),
        "duration_s": (round(float(duration_s), 4) if duration_s is not None else None),
        "outputs": outputs or {},
        "error": error,
        "attempt": int(attempt),
        "case_id": str(case_id),
        "run_id": str(run_id),
    }


def append_event(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

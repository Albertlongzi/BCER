from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def to_short_path(path: Optional[str], *, run_dir: Path) -> Optional[str]:
    if not path:
        return None
    try:
        p = Path(path).expanduser().resolve()
    except Exception:
        return path
    try:
        rd = run_dir.expanduser().resolve()
    except Exception:
        rd = run_dir
    try:
        rel = p.relative_to(rd)
        return rel.as_posix()
    except Exception:
        return str(p)


def with_short_path(entry: Dict[str, Any], *, run_dir: Path) -> Dict[str, Any]:
    out = dict(entry)
    p = entry.get("path")
    short = to_short_path(p, run_dir=run_dir)
    if short:
        out["path"] = short
        if p and short != p:
            out["abs_path"] = p
    return out


def list_with_short_paths(items: List[Dict[str, Any]], *, run_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in items:
        if isinstance(it, dict):
            out.append(with_short_path(it, run_dir=run_dir))
    return out

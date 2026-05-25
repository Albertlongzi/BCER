from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .path_utils import to_short_path


def _load_case_state(case_state_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(case_state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _add_item(items: List[Dict[str, Any]], seen: set, *, path: str, kind: str, description: str = "") -> None:
    if not path:
        return
    if path in seen:
        return
    seen.add(path)
    items.append({"path": path, "kind": kind, "description": description})


def build_artifact_index(
    *,
    case_state_path: Path,
    artifacts_dir: Path,
    out_path: Optional[Path] = None,
) -> Tuple[Path, Dict[str, Any]]:
    state = _load_case_state(case_state_path)
    items: List[Dict[str, Any]] = []
    seen: set = set()

    for art in state.get("artifacts_index", []) or []:
        if isinstance(art, dict):
            _add_item(items, seen, path=str(art.get("path") or ""), kind=str(art.get("kind") or ""), description=str(art.get("description") or ""))

    # Add overlay tiles JSONs and any overlays not captured
    for p in sorted((artifacts_dir / "features" / "overlays").glob("**/*")):
        if p.suffix not in (".png", ".json"):
            continue
        _add_item(items, seen, path=str(p), kind=("image" if p.suffix == ".png" else "json"), description="Overlay artifact")

    # Add key feature outputs if present
    for p in [artifacts_dir / "features" / "features.csv", artifacts_dir / "features" / "slice_summary.json"]:
        if p.exists():
            _add_item(items, seen, path=str(p), kind=("csv" if p.suffix == ".csv" else "json"), description="Feature output")

    run_dir = case_state_path.parent
    out_items: List[Dict[str, Any]] = []
    for it in items:
        short = to_short_path(it.get("path"), run_dir=run_dir)
        rec = {
            "path": short or it.get("path"),
            "abs_path": it.get("path"),
            "kind": it.get("kind"),
            "description": it.get("description"),
            "exists": Path(it.get("path")).exists() if it.get("path") else False,
        }
        out_items.append(rec)

    out = {
        "generated_at": state.get("created_at"),
        "run_id": state.get("run_id"),
        "case_id": state.get("case_id"),
        "artifacts": out_items,
    }

    if out_path is None:
        out_path = artifacts_dir / "context" / "artifact_index.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path, out

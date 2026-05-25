from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _discover_report_source(step_results: List[Dict[str, Any]], run_dir: Path) -> Optional[Path]:
    # Prefer explicit generate_report output.
    for rec in reversed(step_results):
        if not isinstance(rec, dict):
            continue
        tool_name = str(rec.get("tool_name") or "")
        if tool_name != "generate_report":
            continue
        data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
        cand = data.get("report_txt_path") or data.get("clinical_report_path")
        if isinstance(cand, str) and cand:
            p = Path(cand).expanduser().resolve()
            if p.exists():
                return p

    # Fallback conventional locations.
    for rel in (
        "artifacts/report/report.md",
        "artifacts/final/final_report.txt",
        "artifacts/dummy/report/report.md",
        "artifacts/dummy/report/report.txt",
    ):
        p = run_dir / rel
        if p.exists():
            return p
    return None


def _synthesize_report(
    *,
    case_id: str,
    goal: str,
    run_dir: Path,
    step_results: List[Dict[str, Any]],
    target_path: Path,
) -> None:
    lines: List[str] = [
        "# MRI Agent Shell Report",
        "",
        f"- case_id: `{case_id}`",
        f"- generated_at: `{_now_iso()}`",
        f"- run_dir: `{run_dir}`",
        "",
        "## Goal",
        goal or "(not provided)",
        "",
        "## Tool Summary",
    ]

    for rec in step_results:
        if not isinstance(rec, dict):
            continue
        tool_name = str(rec.get("tool_name") or "")
        status = str(rec.get("status") or "")
        duration_s = rec.get("duration_s")
        outputs = rec.get("outputs") if isinstance(rec.get("outputs"), dict) else {}
        lines.append(f"- `{tool_name}`: status={status} duration_s={duration_s}")
        for k, v in outputs.items():
            lines.append(f"  - {k}: `{v}`")

    lines += ["", "## Provenance", "- Tool execution trace is stored in workspace logs."]

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_report(
    *,
    workspace_path: Path,
    case_id: str,
    goal: str,
    run_dir: Path,
    step_results: List[Dict[str, Any]],
) -> Path:
    target = workspace_path / "reports" / case_id / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)

    src = _discover_report_source(step_results=step_results, run_dir=run_dir)
    if src is not None:
        shutil.copy2(str(src), str(target))
        return target

    _synthesize_report(case_id=case_id, goal=goal, run_dir=run_dir, step_results=step_results, target_path=target)
    return target

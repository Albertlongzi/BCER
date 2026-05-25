from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent.rules.engine import RuleViolation, validate_tool_call
from core.paths import project_root
from commands.schemas import ToolCall


PLACEHOLDER_PATTERNS = [
    r"/path/to/",
    r"__LATEST__",
    r"<path>",
    r"<PATH>",
    r"<.*?>",
    r"CHANGE_ME",
    r"TODO",
    r"TBD",
    r"YOUR_",
]
PLACEHOLDER_RE = re.compile("|".join(PLACEHOLDER_PATTERNS))


@dataclass
class CallRecord:
    tool_name: str
    stage: str
    arguments: Dict[str, Any]
    step: Optional[int] = None
    source: str = "raw"  # raw|exec


@dataclass
class CallAssessment:
    call: CallRecord
    hard_violations: List[RuleViolation]
    soft_violations: List[RuleViolation]
    placeholder_paths: List[str]
    missing_paths: List[str]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    return out


def _iter_strings(obj: Any) -> Iterable[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)


def _is_path_like(s: str) -> bool:
    if not s:
        return False
    if s.startswith("/"):
        return True
    if "/" in s or "\\" in s:
        return True
    if s.endswith((".nii", ".nii.gz", ".json", ".txt", ".csv", ".png")):
        return True
    return False


def _resolve_path_candidates(s: str, run_dir: Path) -> List[Path]:
    # Preserve absolute path if present.
    p = Path(s)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(run_dir / s)
        candidates.append(run_dir / "artifacts" / s)
    return candidates


def _find_missing_paths(args: Dict[str, Any], run_dir: Path) -> List[str]:
    missing: List[str] = []
    for s in _iter_strings(args):
        if not _is_path_like(s):
            continue
        if PLACEHOLDER_RE.search(s):
            continue
        ok = False
        for cand in _resolve_path_candidates(s, run_dir):
            try:
                if cand.expanduser().resolve().exists():
                    ok = True
                    break
            except Exception:
                continue
        if not ok:
            missing.append(s)
    return missing


def _find_placeholder_paths(args: Dict[str, Any]) -> List[str]:
    placeholders: List[str] = []
    for s in _iter_strings(args):
        if not _is_path_like(s):
            continue
        if PLACEHOLDER_RE.search(s):
            placeholders.append(s)
    return placeholders


def _collect_raw_calls(trace: List[Dict[str, Any]]) -> List[CallRecord]:
    out: List[CallRecord] = []
    for row in trace:
        if row.get("tag") != "reactive":
            continue
        parsed = row.get("parsed")
        if not isinstance(parsed, dict):
            continue
        action = parsed.get("action")
        if action == "tool_call":
            out.append(
                CallRecord(
                    tool_name=str(parsed.get("tool_name") or ""),
                    stage=str(parsed.get("stage") or ""),
                    arguments=parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {},
                    step=row.get("step"),
                    source="raw",
                )
            )
        elif action == "tool_calls":
            calls = parsed.get("calls") or []
            if isinstance(calls, list):
                for c in calls:
                    if not isinstance(c, dict):
                        continue
                    out.append(
                        CallRecord(
                            tool_name=str(c.get("tool_name") or ""),
                            stage=str(c.get("stage") or ""),
                            arguments=c.get("arguments") if isinstance(c.get("arguments"), dict) else {},
                            step=row.get("step"),
                            source="raw",
                        )
                    )
    return out


def _collect_exec_calls(trace: List[Dict[str, Any]]) -> List[CallRecord]:
    out: List[CallRecord] = []
    for row in trace:
        if row.get("tag") != "tool_exec":
            continue
        parsed = row.get("parsed")
        if not isinstance(parsed, dict):
            continue
        out.append(
            CallRecord(
                tool_name=str(parsed.get("tool_name") or ""),
                stage=str(parsed.get("stage") or ""),
                arguments=parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {},
                step=row.get("step"),
                source="exec",
            )
        )
    return out


def _assess_calls(calls: List[CallRecord], case_state_path: Path, run_dir: Path) -> List[CallAssessment]:
    out: List[CallAssessment] = []
    for call in calls:
        tool_call = ToolCall(
            tool_name=call.tool_name,
            arguments=call.arguments,
            call_id="bench",
            case_id="bench",
            stage=call.stage or "misc",
            requested_by="benchmark",
        )
        violations = validate_tool_call(tool_call, case_state_path)
        hard = [v for v in violations if v.level == "hard"]
        soft = [v for v in violations if v.level != "hard"]
        placeholders = _find_placeholder_paths(call.arguments)
        missing = _find_missing_paths(call.arguments, run_dir)
        out.append(
            CallAssessment(
                call=call,
                hard_violations=hard,
                soft_violations=soft,
                placeholder_paths=placeholders,
                missing_paths=missing,
            )
        )
    return out


def _score_assessments(assess: List[CallAssessment]) -> Dict[str, Any]:
    total = len(assess)
    hard_bad = sum(1 for a in assess if a.hard_violations)
    soft_bad = sum(1 for a in assess if a.soft_violations)
    placeholders = sum(1 for a in assess if a.placeholder_paths)
    missing = sum(1 for a in assess if a.missing_paths)
    path_invalid = sum(1 for a in assess if (a.placeholder_paths or a.missing_paths))
    return {
        "total_calls": total,
        "hard_violation_calls": hard_bad,
        "soft_violation_calls": soft_bad,
        "placeholder_path_calls": placeholders,
        "missing_path_calls": missing,
        "hard_valid_rate": (total - hard_bad) / total if total else 0.0,
        "path_valid_rate": (total - path_invalid) / total if total else 0.0,
    }


def _score_execution_log(exec_log: Path) -> Dict[str, Any]:
    rows = _read_jsonl(exec_log)
    if not rows:
        return {"tool_exec_total": 0, "tool_exec_ok": 0, "tool_exec_ok_rate": 0.0}
    ok = 0
    for r in rows:
        if isinstance(r, dict) and r.get("ok") is True:
            ok += 1
    total = len(rows)
    return {"tool_exec_total": total, "tool_exec_ok": ok, "tool_exec_ok_rate": ok / total if total else 0.0}


def _per_tool_breakdown(assess: List[CallAssessment]) -> Dict[str, Any]:
    per: Dict[str, Dict[str, int]] = {}
    for a in assess:
        name = a.call.tool_name or "<unknown>"
        stats = per.setdefault(
            name,
            {
                "total": 0,
                "hard": 0,
                "soft": 0,
                "placeholder": 0,
                "missing": 0,
            },
        )
        stats["total"] += 1
        if a.hard_violations:
            stats["hard"] += 1
        if a.soft_violations:
            stats["soft"] += 1
        if a.placeholder_paths:
            stats["placeholder"] += 1
        if a.missing_paths:
            stats["missing"] += 1
    return per


def _load_case_state(case_state_path: Path) -> Dict[str, Any]:
    if not case_state_path.exists():
        return {}
    try:
        data = json.loads(case_state_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _tool_ok(case_state: Dict[str, Any], tool_name: str) -> bool:
    stage_outputs = case_state.get("stage_outputs", {}) if isinstance(case_state, dict) else {}
    if not isinstance(stage_outputs, dict):
        return False
    for _stage, tools in stage_outputs.items():
        if not isinstance(tools, dict):
            continue
        recs = tools.get(tool_name)
        if not isinstance(recs, list):
            continue
        if any(isinstance(r, dict) and r.get("ok") is True for r in recs):
            return True
    return False


def _infer_domain(case_state: Dict[str, Any]) -> str:
    meta = case_state.get("metadata", {}) if isinstance(case_state, dict) else {}
    if isinstance(meta, dict):
        dom = meta.get("domain")
        if isinstance(dom, str) and dom.strip():
            return dom.strip().lower()
    if _tool_ok(case_state, "brats_mri_segmentation"):
        return "brain"
    return "prostate"


def _pipeline_completeness(case_state: Dict[str, Any]) -> Dict[str, Any]:
    domain = _infer_domain(case_state)
    expected = {
        "brain": [
            "identify_sequences",
            "register_to_reference",
            "brats_mri_segmentation",
            "extract_roi_features",
            "package_vlm_evidence",
            "generate_report",
        ],
        "prostate": [
            "identify_sequences",
            "register_to_reference",
            "segment_prostate",
            "extract_roi_features",
            "package_vlm_evidence",
            "generate_report",
        ],
    }.get(domain, [])
    completed = [t for t in expected if _tool_ok(case_state, t)]
    missing = [t for t in expected if t not in completed]
    total = len(expected)
    return {
        "domain": domain,
        "expected_tools": expected,
        "completed_tools": completed,
        "missing_tools": missing,
        "completeness_rate": (len(completed) / total) if total else 0.0,
    }


def _redundancy_stats(calls: List[CallRecord]) -> Dict[str, Any]:
    total = len(calls)
    if total == 0:
        return {"total_calls": 0, "redundancy_rate": 0.0, "most_repeated": [], "max_streak": 0}
    counts: Dict[str, int] = {}
    max_streak = 0
    cur_streak = 0
    last_tool = None
    for c in calls:
        name = c.tool_name or "<unknown>"
        counts[name] = counts.get(name, 0) + 1
        if name == last_tool:
            cur_streak += 1
        else:
            cur_streak = 1
            last_tool = name
        max_streak = max(max_streak, cur_streak)
    repeated = sum(max(0, n - 1) for n in counts.values())
    most = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "total_calls": total,
        "redundant_calls": repeated,
        "redundancy_rate": repeated / total if total else 0.0,
        "most_repeated": [{"tool_name": k, "count": v} for k, v in most],
        "max_streak": max_streak,
    }


def _resolve_report_paths(run_dir: Path, report_path: Optional[str]) -> Optional[Path]:
    if not report_path:
        return None
    p = Path(report_path)
    if p.is_absolute():
        return p
    cand = run_dir / report_path
    if cand.exists():
        return cand
    cand = run_dir / "artifacts" / report_path
    if cand.exists():
        return cand
    return run_dir / report_path


def _locate_evidence_bundle(run_dir: Path, report: Dict[str, Any]) -> Optional[Path]:
    for key in ("vlm_evidence_bundle_path", "evidence_bundle_path", "vlm_bundle_path"):
        p = report.get(key)
        if isinstance(p, str) and p:
            cand = _resolve_report_paths(run_dir, p)
            if cand and cand.exists():
                return cand
    fallback = [
        run_dir / "artifacts" / "vlm" / "vlm_evidence_bundle.json",
        run_dir / "artifacts" / "vlm_evidence_bundle" / "vlm_evidence_bundle.json",
    ]
    for cand in fallback:
        if cand.exists():
            return cand
    return None


def _report_faithfulness(run_dir: Path, case_state: Dict[str, Any]) -> Dict[str, Any]:
    report_path = run_dir / "artifacts" / "report" / "report.json"
    if not report_path.exists():
        return {"available": False, "score": None, "checks": [], "mismatches": ["report.json missing"]}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {"available": False, "score": None, "checks": [], "mismatches": ["report.json unreadable"]}

    checks: List[Tuple[str, bool]] = []
    mismatches: List[str] = []

    case_id = str(report.get("case_id") or "")
    run_id = str(report.get("run_id") or "")
    if case_id:
        ok = case_id == str(case_state.get("case_id") or case_id)
        checks.append(("case_id_match", ok))
        if not ok:
            mismatches.append("case_id mismatch")
    if run_id:
        ok = run_id == run_dir.name
        checks.append(("run_id_match", ok))
        if not ok:
            mismatches.append("run_id mismatch")

    mapping = report.get("mapping")
    if isinstance(mapping, dict) and mapping:
        for k, v in mapping.items():
            if not isinstance(v, str):
                continue
            exists = any(c.exists() for c in _resolve_path_candidates(v, run_dir))
            checks.append((f"mapping_path_exists:{k}", exists))
            if not exists:
                mismatches.append(f"mapping path missing: {k}")

    feature_table_path = report.get("feature_table_path")
    if isinstance(feature_table_path, str) and feature_table_path:
        fpath = _resolve_report_paths(run_dir, feature_table_path)
        ok = fpath.exists() if fpath else False
        checks.append(("feature_table_exists", ok))
        if not ok:
            mismatches.append("feature_table_path missing")

    evidence_path = _locate_evidence_bundle(run_dir, report)
    ok_ev = bool(evidence_path and evidence_path.exists())
    checks.append(("evidence_bundle_exists", ok_ev))
    if not ok_ev:
        mismatches.append("evidence_bundle missing")

    stage_status = report.get("stage_status")
    if isinstance(stage_status, dict) and stage_status:
        for tool, reported_ok in stage_status.items():
            if reported_ok is None:
                continue
            actual_ok = _tool_ok(case_state, tool)
            ok = bool(reported_ok) == actual_ok
            checks.append((f"stage_status_match:{tool}", ok))
            if not ok:
                mismatches.append(f"stage_status mismatch: {tool}")

    score = sum(1 for _k, v in checks if v) / len(checks) if checks else None
    return {
        "available": True,
        "score": score,
        "checks": [{"check": k, "ok": v} for k, v in checks],
        "mismatches": mismatches,
        "report_json": str(report_path),
        "evidence_bundle": str(evidence_path) if evidence_path else None,
    }


def _pick_latest_run(case_dir: Path) -> Optional[Path]:
    if not case_dir.exists():
        return None
    runs = [p for p in case_dir.iterdir() if p.is_dir()]
    if not runs:
        return None
    # run dirs start with YYYYMMDD_... so lexicographic sort works
    return sorted(runs)[-1]


def score_run(run_dir: Path) -> Dict[str, Any]:
    case_state = run_dir / "case_state.json"
    trace_path = run_dir / "agent_trace.jsonl"
    exec_log = run_dir / "execution_log.jsonl"

    trace = _read_jsonl(trace_path)
    raw_calls = _collect_raw_calls(trace)
    exec_calls = _collect_exec_calls(trace)

    raw_assess = _assess_calls(raw_calls, case_state, run_dir)
    exec_assess = _assess_calls(exec_calls, case_state, run_dir)
    cs = _load_case_state(case_state)

    result = {
        "run_dir": str(run_dir),
        "raw": _score_assessments(raw_assess),
        "exec": _score_assessments(exec_assess),
        "exec_log": _score_execution_log(exec_log),
        "raw_per_tool": _per_tool_breakdown(raw_assess),
        "exec_per_tool": _per_tool_breakdown(exec_assess),
        "pipeline_completeness": _pipeline_completeness(cs),
        "redundancy": {
            "raw": _redundancy_stats(raw_calls),
            "exec": _redundancy_stats(exec_calls),
        },
        "report_faithfulness": _report_faithfulness(run_dir, cs),
    }
    return result


def _print_summary(result: Dict[str, Any]) -> None:
    raw = result.get("raw", {})
    ex = result.get("exec", {})
    elog = result.get("exec_log", {})
    pipe = result.get("pipeline_completeness", {})
    red_raw = (result.get("redundancy") or {}).get("raw", {})
    red_exec = (result.get("redundancy") or {}).get("exec", {})
    rep = result.get("report_faithfulness", {})
    print(f"Run: {result.get('run_dir')}")
    print("Raw model tool-calls:")
    print(f"  total={raw.get('total_calls', 0)} hard_valid_rate={raw.get('hard_valid_rate', 0):.3f} path_valid_rate={raw.get('path_valid_rate', 0):.3f}")
    print(f"  hard_violations={raw.get('hard_violation_calls', 0)} placeholders={raw.get('placeholder_path_calls', 0)} missing_paths={raw.get('missing_path_calls', 0)}")
    print("Executed tool-calls:")
    print(f"  total={ex.get('total_calls', 0)} hard_valid_rate={ex.get('hard_valid_rate', 0):.3f} path_valid_rate={ex.get('path_valid_rate', 0):.3f}")
    print(f"  hard_violations={ex.get('hard_violation_calls', 0)} placeholders={ex.get('placeholder_path_calls', 0)} missing_paths={ex.get('missing_path_calls', 0)}")
    print("Tool execution:")
    print(f"  total={elog.get('tool_exec_total', 0)} ok_rate={elog.get('tool_exec_ok_rate', 0):.3f}")
    if pipe:
        print("Pipeline completeness:")
        print(f"  domain={pipe.get('domain')} completeness_rate={pipe.get('completeness_rate', 0):.3f} missing={pipe.get('missing_tools')}")
    if red_raw:
        print("Redundancy (raw):")
        print(f"  redundancy_rate={red_raw.get('redundancy_rate', 0):.3f} max_streak={red_raw.get('max_streak', 0)}")
    if red_exec:
        print("Redundancy (exec):")
        print(f"  redundancy_rate={red_exec.get('redundancy_rate', 0):.3f} max_streak={red_exec.get('max_streak', 0)}")
    if rep and rep.get("available"):
        print("Report faithfulness:")
        print(f"  score={rep.get('score')} mismatches={rep.get('mismatches')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a single BCER run for tool-call validity.")
    ap.add_argument("--run-dir", default="", help="Path to run directory (runs/<case>/<run_id>).")
    ap.add_argument("--runs-root", default=str(project_root() / "runs"))
    ap.add_argument("--case-id", default="", help="Case ID (used with --runs-root to pick latest run).")
    ap.add_argument("--output-json", default="", help="Write full JSON score report to this path.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    if not run_dir and args.case_id:
        run_dir = _pick_latest_run(Path(args.runs_root).expanduser().resolve() / args.case_id)
    if not run_dir or not run_dir.exists():
        raise SystemExit("run_dir not found (provide --run-dir or --case-id)")

    result = score_run(run_dir)
    _print_summary(result)

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

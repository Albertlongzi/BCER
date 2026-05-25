#!/usr/bin/env python3
"""Aggregate benchmark v2 result JSON files into human-readable summary tables.

Provides two main views:
  1. **Baseline summary** (``--mode baseline``): Category-A capability
     comparison (fault=none).  Pivot table: rows=tasks, columns=arms,
     cells=SR/TCR.  Plus aggregate row at the bottom.
  2. **Ablation summary** (``--mode ablation``): Category-B reflector
     ablation.  Two tables (aggregate by fault, per-task detail) showing
     SR/ERR across the three BCER arms.
  3. ``--mode full`` (default): prints both plus the original per-combo
     detail table.

Usage examples:
    # Quick overview — prints baseline + ablation tables only
    python benchmark/summarize_results.py --results-dir benchmark/ --mode baseline
    python benchmark/summarize_results.py --results-dir benchmark/ --mode ablation

    # Full output with all detail
    python benchmark/summarize_results.py --results-dir benchmark/

    # Export CSV / JSON
    python benchmark/summarize_results.py --results-dir benchmark/ --output-csv benchmark/summary.csv
    python benchmark/summarize_results.py --results-dir benchmark/ --output-json benchmark/summary.json
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _load_result_files(results_dir: Path, pattern: str = "benchmark_results_v2_*.json") -> List[Dict[str, Any]]:
    """Load all matching result JSON files from the given directory."""
    files = sorted(results_dir.glob(pattern))
    records: List[Dict[str, Any]] = []
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data["_source_file"] = str(fp.name)
                records.append(data)
        except Exception as exc:
            print(f"[warn] skipping {fp.name}: {exc}", file=sys.stderr)
    return records


def _extract_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a flat summary row from a single result record."""
    agg = rec.get("aggregate") if isinstance(rec.get("aggregate"), dict) else {}
    row = {
        "task": str(rec.get("task_id") or ""),
        "arm": str(rec.get("arm") or ""),
        "fault": str(rec.get("fault") or ""),
        "runs": int(agg.get("runs") or 0),
        "success_rate": _safe_float(agg.get("success_rate")),
        "avg_tcr": _safe_float(agg.get("avg_tcr")),
        "err_rate": _safe_float(agg.get("err_rate")),
        "fault_applied_rate": _safe_float(agg.get("fault_applied_rate")),
        "safe_halt_rate": _safe_float(agg.get("safe_halt_rate")),
        "semantic_guard_rate": _safe_float(agg.get("semantic_guard_rate")),
        "invariant_pass_rate": _safe_float(agg.get("invariant_pass_rate")),
        "success_pass": int(agg.get("success_pass") or 0),
        "err_eligible": int(agg.get("err_eligible_runs") or 0),
        "err_recovered": int(agg.get("err_recovered") or 0),
        "fault_requested_runs": int(agg.get("fault_requested_runs") or 0),
        "fault_not_applicable_runs": int(agg.get("fault_not_applicable_runs") or 0),
        "fault_evaluable_runs": int(agg.get("fault_evaluable_runs") or 0),
        "fault_applied_runs": int(agg.get("fault_applied_runs") or 0),
        "safe_halt_eligible_runs": int(agg.get("safe_halt_eligible_runs") or 0),
        "safe_halt_pass_runs": int(agg.get("safe_halt_pass_runs") or 0),
        "semantic_guard_eligible_runs": int(agg.get("semantic_guard_eligible_runs") or 0),
        "semantic_guard_pass_runs": int(agg.get("semantic_guard_pass_runs") or 0),
        "inv_evaluable": int(agg.get("invariant_evaluable_runs") or 0),
        "inv_pass": int(agg.get("invariant_pass_runs") or 0),
        "source": str(rec.get("_source_file") or ""),
    }

    # Backward compatibility: old summaries may not have new aggregate fields.
    need_derived = ("fault_applied_rate" not in agg) or ("safe_halt_rate" not in agg) or ("semantic_guard_rate" not in agg)
    if need_derived:
        derived = _derive_fault_stats_from_results(rec)
        for key, val in derived.items():
            if key in row:
                row[key] = val
    return row


def _derive_fault_stats_from_results(rec: Dict[str, Any]) -> Dict[str, Any]:
    results = rec.get("results") if isinstance(rec.get("results"), list) else []
    fault_requested = 0
    fault_not_applicable = 0
    fault_evaluable = 0
    fault_applied = 0
    safe_halt_eligible = 0
    safe_halt_pass = 0
    semantic_guard_eligible = 0
    semantic_guard_pass = 0
    for m in results:
        if not isinstance(m, dict):
            continue
        fault_name = str(m.get("fault") or rec.get("fault") or "").strip().lower()
        fi = m.get("fault_injection") if isinstance(m.get("fault_injection"), dict) else {}
        requested = bool(fi.get("requested")) or str(m.get("fault") or "").strip().lower() != "none"
        not_app = bool(fi.get("not_applicable"))
        applied = bool(fi.get("applied"))
        if requested:
            fault_requested += 1
            if not_app:
                fault_not_applicable += 1
            else:
                fault_evaluable += 1
                if applied:
                    fault_applied += 1
        sh = m.get("safe_halt") if isinstance(m.get("safe_halt"), dict) else {}
        if bool(sh.get("eligible")):
            safe_halt_eligible += 1
            if bool(sh.get("pass")):
                safe_halt_pass += 1
        elif fault_name in {"missing_modality", "scope_violation"} and requested and (not not_app) and applied:
            # Backfill for legacy results that do not include ``safe_halt``.
            safe_halt_eligible += 1
            success = bool((m.get("success") or {}).get("pass"))
            actual_status = str(m.get("actual_status") or "").strip().lower()
            if (not success) and actual_status != "success":
                safe_halt_pass += 1

        sg = m.get("semantic_guard") if isinstance(m.get("semantic_guard"), dict) else {}
        if bool(sg.get("eligible")):
            semantic_guard_eligible += 1
            if bool(sg.get("pass")):
                semantic_guard_pass += 1
        elif fault_name == "semantic_swap" and requested and (not not_app) and applied:
            # Legacy fallback when semantic_guard is absent.
            semantic_guard_eligible += 1
            if bool((m.get("success") or {}).get("pass")):
                semantic_guard_pass += 1

    return {
        "fault_requested_runs": fault_requested,
        "fault_not_applicable_runs": fault_not_applicable,
        "fault_evaluable_runs": fault_evaluable,
        "fault_applied_runs": fault_applied,
        "fault_applied_rate": (float(fault_applied) / float(fault_evaluable)) if fault_evaluable > 0 else None,
        "safe_halt_eligible_runs": safe_halt_eligible,
        "safe_halt_pass_runs": safe_halt_pass,
        "safe_halt_rate": (float(safe_halt_pass) / float(safe_halt_eligible)) if safe_halt_eligible > 0 else None,
        "semantic_guard_eligible_runs": semantic_guard_eligible,
        "semantic_guard_pass_runs": semantic_guard_pass,
        "semantic_guard_rate": (float(semantic_guard_pass) / float(semantic_guard_eligible)) if semantic_guard_eligible > 0 else None,
    }


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Filtering & sorting
# ---------------------------------------------------------------------------


def _filter_rows(
    rows: List[Dict[str, Any]],
    *,
    arm: Optional[str] = None,
    task: Optional[str] = None,
    fault: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out = rows
    if arm:
        out = [r for r in out if r["arm"] == arm]
    if task:
        out = [r for r in out if r["task"] == task]
    if fault:
        out = [r for r in out if r["fault"] == fault]
    return out


def _sort_rows(
    rows: List[Dict[str, Any]],
    sort_by: str = "task",
    ascending: bool = True,
) -> List[Dict[str, Any]]:
    def _key(r: Dict[str, Any]) -> Tuple:
        val = r.get(sort_by)
        if val is None:
            return (1, 0, r.get("task", ""), r.get("arm", ""), r.get("fault", ""))
        if isinstance(val, str):
            return (0, 0, val, r.get("arm", ""), r.get("fault", ""))
        return (0, val, r.get("task", ""), r.get("arm", ""), r.get("fault", ""))

    return sorted(rows, key=_key, reverse=(not ascending))


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

_DISPLAY_COLS = [
    "task",
    "arm",
    "fault",
    "runs",
    "success_rate",
    "avg_tcr",
    "err_rate",
    "fault_applied_rate",
    "safe_halt_rate",
    "semantic_guard_rate",
    "invariant_pass_rate",
]
_DISPLAY_HEADERS = ["Task", "Arm", "Fault", "Runs", "SR", "TCR", "ERR", "FAR", "SHR", "SGR", "INV"]

# Canonical orderings for consistent display
_ARM_ORDER = ["bcer_sketch", "react", "react_token", "react_token_reflector"]
_FAULT_ORDER = [
    "none", "token_mutation", "path_mutation", "argument_omission",
    "semantic_swap", "space_mismatch", "missing_modality", "scope_violation", "timeout",
]
_TASK_ORDER = [
    "short_denoise", "short_superres", "short_segment_brain", "short_recon_grappa",
    "medium_register_prostate", "medium_brain_grade_classify",
    "long_prostate_full", "long_cardiac_full",
]

_BASELINE_ARMS = ["bcer_sketch", "react", "react_token", "react_token_reflector"]
_ABLATION_ARMS = ["bcer_sketch"]
_RECOVERABLE_FAULTS = {"token_mutation", "path_mutation", "argument_omission", "semantic_swap", "space_mismatch"}
_SAFETY_FAULTS = {"missing_modality", "scope_violation"}


def _fmt_val(val: Any, col: str) -> str:
    if val is None:
        return "—"
    if col in {"success_rate", "avg_tcr", "err_rate", "fault_applied_rate", "safe_halt_rate", "semantic_guard_rate", "invariant_pass_rate"}:
        return f"{float(val):.3f}"
    return str(val)


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "  —  "
    return f"{val:.3f}"


def _ordered(items: List[str], canonical: List[str]) -> List[str]:
    """Return *items* sorted by canonical order, with unknowns appended."""
    rank = {v: i for i, v in enumerate(canonical)}
    return sorted(items, key=lambda x: (rank.get(x, 999), x))


# ---- helpers to compute weighted-average aggregates over rows ----

def _weighted_mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    """Weighted average of *key* across rows, using 'runs' as weight."""
    total_w = 0
    total_v = 0.0
    for r in rows:
        w = int(r.get("runs") or 0)
        v = r.get(key)
        if v is None or w == 0:
            continue
        total_w += w
        total_v += float(v) * w
    return (total_v / total_w) if total_w > 0 else None


def _sum_field(rows: List[Dict[str, Any]], key: str) -> int:
    return sum(int(r.get(key) or 0) for r in rows)


def _safe_div(a: int, b: int) -> Optional[float]:
    return (float(a) / float(b)) if b > 0 else None


def _agg_row(rows: List[Dict[str, Any]], *, label: str = "AGGREGATE") -> Dict[str, Any]:
    """Compute aggregate stats from a list of per-combo rows."""
    total_runs = _sum_field(rows, "runs")
    sp = _sum_field(rows, "success_pass")
    ee = _sum_field(rows, "err_eligible")
    er = _sum_field(rows, "err_recovered")
    fe = _sum_field(rows, "fault_evaluable_runs")
    fa = _sum_field(rows, "fault_applied_runs")
    she = _sum_field(rows, "safe_halt_eligible_runs")
    shp = _sum_field(rows, "safe_halt_pass_runs")
    sge = _sum_field(rows, "semantic_guard_eligible_runs")
    sgp = _sum_field(rows, "semantic_guard_pass_runs")
    ie = _sum_field(rows, "inv_evaluable")
    ip = _sum_field(rows, "inv_pass")
    return {
        "task": label,
        "arm": "",
        "fault": "",
        "runs": total_runs,
        "success_rate": _safe_div(sp, total_runs),
        "avg_tcr": _weighted_mean(rows, "avg_tcr"),
        "err_rate": _safe_div(er, ee),
        "fault_applied_rate": _safe_div(fa, fe),
        "safe_halt_rate": _safe_div(shp, she),
        "semantic_guard_rate": _safe_div(sgp, sge),
        "invariant_pass_rate": _safe_div(ip, ie),
        "success_pass": sp,
        "err_eligible": ee,
        "err_recovered": er,
        "fault_evaluable_runs": fe,
        "fault_applied_runs": fa,
        "safe_halt_eligible_runs": she,
        "safe_halt_pass_runs": shp,
        "semantic_guard_eligible_runs": sge,
        "semantic_guard_pass_runs": sgp,
        "inv_evaluable": ie,
        "inv_pass": ip,
    }


# ---- original per-row tables (kept for --mode full) ----

def _render_compact_table(rows: List[Dict[str, Any]], *, cols: Optional[List[str]] = None, headers: Optional[List[str]] = None) -> str:
    if not rows:
        return "(no results)\n"
    use_cols = cols or _DISPLAY_COLS
    use_headers = headers or _DISPLAY_HEADERS

    widths = [len(h) for h in use_headers]
    formatted: List[List[str]] = []
    for r in rows:
        cells = [_fmt_val(r.get(c), c) for c in use_cols]
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))
        formatted.append(cells)

    lines: List[str] = []
    header = "  ".join(h.ljust(widths[i]) for i, h in enumerate(use_headers))
    sep = "  ".join("-" * widths[i] for i in range(len(use_headers)))
    lines.append(header)
    lines.append(sep)
    for cells in formatted:
        row_str = "  ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))
        lines.append(row_str)
    return "\n".join(lines) + "\n"


def _render_markdown_table(rows: List[Dict[str, Any]], *, cols: Optional[List[str]] = None, headers: Optional[List[str]] = None) -> str:
    if not rows:
        return "(no results)\n"
    use_cols = cols or _DISPLAY_COLS
    use_headers = headers or _DISPLAY_HEADERS

    widths = [len(h) for h in use_headers]
    formatted: List[List[str]] = []
    for r in rows:
        cells = [_fmt_val(r.get(c), c) for c in use_cols]
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))
        formatted.append(cells)

    lines: List[str] = []
    header = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(use_headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * widths[i] for i in range(len(use_headers))) + "-|"
    lines.append(header)
    lines.append(sep_line)
    for cells in formatted:
        row_str = "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells))) + " |"
        lines.append(row_str)
    return "\n".join(lines) + "\n"


# ---- NEW: Baseline Summary (Category A) ----

def _render_baseline_summary(rows: List[Dict[str, Any]], *, markdown: bool = False) -> str:
    """Pivot table: rows = tasks, columns = arms, cells = SR / TCR.

    Only includes fault=none rows from baseline arms.
    Adds an AGGREGATE row at the bottom.
    """
    baseline_rows = [r for r in rows if r.get("fault") == "none" and r.get("arm") in set(_BASELINE_ARMS)]
    if not baseline_rows:
        return "(no baseline results found)\n"

    tasks_seen = _ordered(list({r["task"] for r in baseline_rows}), _TASK_ORDER)
    arms_seen = _ordered(list({r["arm"] for r in baseline_rows}), _ARM_ORDER)
    lookup = {(r["task"], r["arm"]): r for r in baseline_rows}

    arm_short = {
        "bcer_sketch": "BCER",
        "react": "ReAct",
        "react_token": "ReAct-Tok",
        "react_token_reflector": "ReAct-Tok+Ref",
    }

    col_w = 12
    task_w = max(28, *(len(t) for t in tasks_seen))
    sep_char = "|" if markdown else " "
    pad = " " if markdown else ""

    lines: List[str] = []
    lines.append("\n" + "=" * 72)
    lines.append("  Category A: Capability Baseline (fault=none)")
    lines.append("=" * 72)

    # Header
    arm_labels = [arm_short.get(a, a).center(col_w) for a in arms_seen]
    h1 = f"{'Task':<{task_w}}  " + "  ".join(arm_labels)
    h2 = f"{'':<{task_w}}  " + "  ".join("SR   TCR ".center(col_w) for _ in arms_seen)
    lines.append(h1)
    lines.append(h2)
    lines.append("-" * len(h1))

    # Per-task rows
    for t in tasks_seen:
        cells: List[str] = []
        for a in arms_seen:
            r = lookup.get((t, a))
            if r is None:
                cells.append("—    —   ".center(col_w))
            else:
                sr = _fmt_pct(r.get("success_rate"))
                tcr = _fmt_pct(r.get("avg_tcr"))
                cells.append(f"{sr} {tcr}".center(col_w))
        lines.append(f"{t:<{task_w}}  " + "  ".join(cells))

    # Aggregate row
    lines.append("-" * len(h1))
    agg_cells: List[str] = []
    for a in arms_seen:
        arm_rows = [r for r in baseline_rows if r.get("arm") == a]
        ag = _agg_row(arm_rows)
        sr = _fmt_pct(ag.get("success_rate"))
        tcr = _fmt_pct(ag.get("avg_tcr"))
        agg_cells.append(f"{sr} {tcr}".center(col_w))
    lines.append(f"{'AGGREGATE':<{task_w}}  " + "  ".join(agg_cells))
    lines.append("")

    return "\n".join(lines) + "\n"


# ---- NEW: Ablation Summary (Category B) ----

def _render_ablation_summary(rows: List[Dict[str, Any]], *, markdown: bool = False) -> str:
    """Two tables for reflector ablation study.

    Table 1: Aggregate by Fault — rows=fault types, cols=arms, cells=SR/ERR
    Table 2: Per-Task Detail  — rows=tasks (grouped by fault), cols=arms, cells=SR/ERR
    """
    ablation_rows = [
        r for r in rows
        if r.get("arm") in set(_ABLATION_ARMS)
        and r.get("fault") in (_RECOVERABLE_FAULTS | _SAFETY_FAULTS)
    ]
    if not ablation_rows:
        return "(no ablation results found)\n"

    arms_seen = _ordered(list({r["arm"] for r in ablation_rows}), _ARM_ORDER)
    faults_seen = _ordered(list({r["fault"] for r in ablation_rows}), _FAULT_ORDER)
    tasks_seen = _ordered(list({r["task"] for r in ablation_rows}), _TASK_ORDER)

    lookup = {(r["task"], r["arm"], r["fault"]): r for r in ablation_rows}

    arm_short = {
        "bcer_sketch": "BCER",
        "react": "ReAct",
        "react_token": "ReAct-Tok",
        "react_token_reflector": "ReAct-Tok+Ref",
    }

    col_w = 14
    fault_w = max(20, *(len(f) for f in faults_seen))
    task_w = max(30, *(len(t) for t in tasks_seen))

    lines: List[str] = []
    lines.append("\n" + "=" * 72)
    lines.append("  Category B: Reflector Ablation (fault injection)")
    lines.append("=" * 72)

    # ---- Table 1: Aggregate by Fault ----
    lines.append("\n--- Aggregate by Fault (weighted across tasks) ---\n")
    arm_labels = [arm_short.get(a, a).center(col_w) for a in arms_seen]
    h1 = f"{'Fault':<{fault_w}}  " + "  ".join(arm_labels)
    h2 = f"{'':<{fault_w}}  " + "  ".join("SR    ERR  ".center(col_w) for _ in arms_seen)
    lines.append(h1)
    lines.append(h2)
    lines.append("-" * len(h1))

    for fault in faults_seen:
        cells: List[str] = []
        for a in arms_seen:
            arm_fault_rows = [r for r in ablation_rows if r.get("arm") == a and r.get("fault") == fault]
            if not arm_fault_rows:
                cells.append("  —     —  ".center(col_w))
            else:
                ag = _agg_row(arm_fault_rows)
                sr = _fmt_pct(ag.get("success_rate"))
                err = _fmt_pct(ag.get("err_rate"))
                cells.append(f"{sr} {err}".center(col_w))
        tag = "†" if fault in _SAFETY_FAULTS else ""
        lines.append(f"{fault + tag:<{fault_w}}  " + "  ".join(cells))

    # Overall aggregate
    lines.append("-" * len(h1))
    ovr_cells: List[str] = []
    for a in arms_seen:
        arm_all = [r for r in ablation_rows if r.get("arm") == a]
        ag = _agg_row(arm_all)
        sr = _fmt_pct(ag.get("success_rate"))
        err = _fmt_pct(ag.get("err_rate"))
        ovr_cells.append(f"{sr} {err}".center(col_w))
    lines.append(f"{'OVERALL':<{fault_w}}  " + "  ".join(ovr_cells))
    lines.append("\n(† = non-recoverable / safety fault; ERR for safety = safe-halt rate)")

    # ---- Table 2: Per-task Detail ----
    lines.append("\n--- Per-Task × Fault Detail ---\n")
    arm_labels2 = [arm_short.get(a, a).center(col_w) for a in arms_seen]
    h1b = f"{'Task / Fault':<{task_w}}  " + "  ".join(arm_labels2)
    h2b = f"{'':<{task_w}}  " + "  ".join("SR    ERR  ".center(col_w) for _ in arms_seen)
    lines.append(h1b)
    lines.append(h2b)
    lines.append("-" * len(h1b))

    for task in tasks_seen:
        # task header
        lines.append(f"  {task}")
        for fault in faults_seen:
            cells: List[str] = []
            has_any = False
            for a in arms_seen:
                r = lookup.get((task, a, fault))
                if r is None:
                    cells.append("  —     —  ".center(col_w))
                else:
                    has_any = True
                    sr = _fmt_pct(r.get("success_rate"))
                    err = _fmt_pct(r.get("err_rate"))
                    cells.append(f"{sr} {err}".center(col_w))
            if has_any:
                lines.append(f"    {fault:<{task_w - 4}}  " + "  ".join(cells))
        # per-task aggregate
        task_rows = [r for r in ablation_rows if r.get("task") == task]
        if task_rows:
            agg_cells2: List[str] = []
            for a in arms_seen:
                arm_task = [r for r in task_rows if r.get("arm") == a]
                if arm_task:
                    ag = _agg_row(arm_task)
                    sr = _fmt_pct(ag.get("success_rate"))
                    err = _fmt_pct(ag.get("err_rate"))
                    agg_cells2.append(f"{sr} {err}".center(col_w))
                else:
                    agg_cells2.append("  —     —  ".center(col_w))
            lines.append(f"    {'(subtotal)':<{task_w - 4}}  " + "  ".join(agg_cells2))
        lines.append("")

    return "\n".join(lines) + "\n"


# ---- Safety Fault Summary ----

def _render_safety_summary(rows: List[Dict[str, Any]]) -> str:
    """Compact table for non-recoverable faults: shows safe-halt rate."""
    safety_rows = [
        r for r in rows
        if r.get("arm") in set(_ABLATION_ARMS)
        and r.get("fault") in _SAFETY_FAULTS
    ]
    if not safety_rows:
        return ""

    arms_seen = _ordered(list({r["arm"] for r in safety_rows}), _ARM_ORDER)
    faults_seen = _ordered(list({r["fault"] for r in safety_rows}), _FAULT_ORDER)
    tasks_seen = _ordered(list({r["task"] for r in safety_rows}), _TASK_ORDER)
    lookup = {(r["task"], r["arm"], r["fault"]): r for r in safety_rows}

    arm_short = {
        "bcer_sketch": "BCER",
        "react": "ReAct",
        "react_token": "ReAct-Tok",
        "react_token_reflector": "ReAct-Tok+Ref",
    }

    col_w = 14
    # Width must accommodate the full "task / fault" label
    max_label = max(
        (len(f"  {t} / {f}") for t in tasks_seen for f in faults_seen),
        default=30,
    )
    task_w = max(35, max_label + 2)

    lines: List[str] = []
    lines.append("\n--- Safety Faults: Safe-Halt Rate ---\n")
    arm_labels = [arm_short.get(a, a).center(col_w) for a in arms_seen]
    h1 = f"{'Task / Fault':<{task_w}}  " + "  ".join(arm_labels)
    h2 = f"{'':<{task_w}}  " + "  ".join("  SHR       ".center(col_w) for _ in arms_seen)
    lines.append(h1)
    lines.append(h2)
    lines.append("-" * len(h1))

    for task in tasks_seen:
        for fault in faults_seen:
            cells: List[str] = []
            has_any = False
            for a in arms_seen:
                r = lookup.get((task, a, fault))
                if r is None:
                    cells.append("  —  ".center(col_w))
                else:
                    has_any = True
                    shr = _fmt_pct(r.get("safe_halt_rate"))
                    cells.append(shr.center(col_w))
            if has_any:
                label = f"  {task} / {fault}"
                lines.append(f"{label:<{task_w}}  " + "  ".join(cells))

    # aggregate
    lines.append("-" * len(h1))
    agg_cells: List[str] = []
    for a in arms_seen:
        arm_all = [r for r in safety_rows if r.get("arm") == a]
        ag = _agg_row(arm_all)
        shr = _fmt_pct(ag.get("safe_halt_rate"))
        agg_cells.append(shr.center(col_w))
    lines.append(f"{'AGGREGATE':<{task_w}}  " + "  ".join(agg_cells))
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# JSON / CSV export
# ---------------------------------------------------------------------------


def _export_json(rows: List[Dict[str, Any]], path: Path) -> None:
    # Remove internal _source_file from export but keep it in the row for reference
    export_rows = []
    for r in rows:
        er = dict(r)
        export_rows.append(er)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": export_rows, "total_combos": len(export_rows)}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[summarize] wrote JSON: {path}")


def _export_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        print("[summarize] no rows to export")
        return
    cols = _DISPLAY_COLS + [
        "success_pass",
        "err_eligible",
        "err_recovered",
        "fault_requested_runs",
        "fault_not_applicable_runs",
        "fault_evaluable_runs",
        "fault_applied_runs",
        "safe_halt_eligible_runs",
        "safe_halt_pass_runs",
        "inv_evaluable",
        "inv_pass",
        "source",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[summarize] wrote CSV: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Aggregate benchmark v2 result JSON files into human-readable summary tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick baseline capability overview
  python benchmark/summarize_results.py --results-dir benchmark/ --mode baseline

  # Reflector ablation study
  python benchmark/summarize_results.py --results-dir benchmark/ --mode ablation

  # Full output (baseline + ablation + per-combo detail)
  python benchmark/summarize_results.py --results-dir benchmark/ --mode full

  # Export to CSV / JSON
  python benchmark/summarize_results.py --results-dir benchmark/ --output-csv summary.csv --output-json summary.json
""",
    )
    ap.add_argument(
        "--results-dir",
        default="benchmark/",
        help="Directory containing benchmark_results_v2_*.json files (default: benchmark/)",
    )
    ap.add_argument(
        "--glob",
        default="benchmark_results_v2_*.json",
        help="Glob pattern for result files (default: benchmark_results_v2_*.json)",
    )
    ap.add_argument(
        "--mode",
        default="full",
        choices=["baseline", "ablation", "full", "detail"],
        help=(
            "Output mode: baseline = Category-A pivot only; "
            "ablation = Category-B reflector tables only; "
            "full = both + safety; "
            "detail = raw per-combo table (default: full)"
        ),
    )
    ap.add_argument("--arm", default="", help="Filter by arm (e.g. bcer_sketch)")
    ap.add_argument("--task", default="", help="Filter by task (e.g. short_superres)")
    ap.add_argument("--fault", default="", help="Filter by fault (e.g. none)")
    ap.add_argument(
        "--sort-by",
        default="task",
        choices=[
            "task", "arm", "fault", "runs",
            "success_rate", "avg_tcr", "err_rate",
            "fault_applied_rate", "safe_halt_rate", "invariant_pass_rate",
        ],
        help="Sort column for detail mode (default: task)",
    )
    ap.add_argument("--sort-asc", action="store_true", help="Sort ascending (default: descending for metrics)")
    ap.add_argument("--output-json", default="", help="Write summary JSON to this path")
    ap.add_argument("--output-csv", default="", help="Write summary CSV to this path")
    ap.add_argument("--format", default="compact", choices=["compact", "markdown"], help="Table format for detail mode (default: compact)")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")

    records = _load_result_files(results_dir, pattern=args.glob)
    if not records:
        raise SystemExit(f"No result files found matching {args.glob} in {results_dir}")

    rows = [_extract_row(r) for r in records]

    # Apply filters
    rows = _filter_rows(
        rows,
        arm=args.arm.strip() or None,
        task=args.task.strip() or None,
        fault=args.fault.strip() or None,
    )
    if not rows:
        raise SystemExit("No results match the given filters.")

    # Sort
    ascending = args.sort_asc or args.sort_by in {"task", "arm", "fault"}
    rows = _sort_rows(rows, sort_by=args.sort_by, ascending=ascending)

    print(f"\n[summarize] {len(rows)} result(s) from {results_dir}\n")

    mode = args.mode

    if mode in {"baseline", "full"}:
        print(_render_baseline_summary(rows, markdown=(args.format == "markdown")))

    if mode in {"ablation", "full"}:
        print(_render_ablation_summary(rows, markdown=(args.format == "markdown")))
        print(_render_safety_summary(rows))

    if mode in {"detail", "full"}:
        print("\n--- Per-Combo Detail ---\n")
        if args.format == "markdown":
            print(_render_markdown_table(rows))
        else:
            print(_render_compact_table(rows))

    # Export
    if args.output_json.strip():
        _export_json(rows, Path(args.output_json).expanduser().resolve())
    if args.output_csv.strip():
        _export_csv(rows, Path(args.output_csv).expanduser().resolve())


if __name__ == "__main__":
    main()

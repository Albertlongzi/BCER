from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def compact_for_memory(obj: Any, *, max_chars: int = 2000) -> Any:
    """
    Compact nested dict/list objects so we can store them in short_term.jsonl without blowing up files/prompts.
    """
    try:
        s = json.dumps(obj, ensure_ascii=False)
        if len(s) <= max_chars:
            return obj
        return {"_truncated": True, "head": s[: max_chars // 2], "tail": s[-max_chars // 2 :]}
    except Exception:
        return {"_unserializable": True, "type": type(obj).__name__}


def extract_key_paths(tool_name: str, tool_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a compact, stable subset of paths/identifiers from a tool_result (truth).
    """
    out: Dict[str, Any] = {}
    data = tool_result.get("data") if isinstance(tool_result, dict) else None
    if not isinstance(data, dict):
        data = {}

    if tool_name == "ingest_dicom_to_nifti":
        for k in ("series_inventory_path", "dicom_meta_path"):
            if data.get(k):
                out[k] = data.get(k)
        if isinstance(data.get("nifti_by_series"), dict):
            out["nifti_by_series"] = data.get("nifti_by_series")
    elif tool_name == "identify_sequences":
        if isinstance(data.get("mapping"), dict):
            out["mapping"] = data.get("mapping")
    elif tool_name == "register_to_reference":
        for k in ("transform_path", "resampled_path"):
            if data.get(k):
                out[k] = data.get(k)
        if isinstance(data.get("resampled_paths"), dict):
            out["resampled_paths"] = data.get("resampled_paths")
        if isinstance(data.get("qc_pngs"), dict):
            out["qc_pngs"] = data.get("qc_pngs")
        if isinstance(data.get("qc_metrics"), dict):
            out["qc_metrics"] = {"method": data.get("qc_metrics", {}).get("method")}
    elif tool_name == "segment_prostate":
        for k in ("prostate_mask_path", "zone_mask_path"):
            if data.get(k):
                out[k] = data.get(k)
    elif tool_name == "segment_cardiac_cine":
        for k in ("seg_path", "rv_mask_path", "myo_mask_path", "lv_mask_path"):
            if data.get(k):
                out[k] = data.get(k)
    elif tool_name == "classify_cardiac_cine_disease":
        for k in ("classification_path", "predicted_group"):
            if data.get(k):
                out[k] = data.get(k)
    elif tool_name == "extract_roi_features":
        for k in ("feature_table_path", "slice_summary_path"):
            if data.get(k):
                out[k] = data.get(k)
        if isinstance(data.get("overlay_pngs"), list):
            out["overlay_pngs"] = (data.get("overlay_pngs") or [])[:6]
    elif tool_name == "detect_lesion_candidates":
        for k in ("candidates_path", "merged_prob_path", "lesion_mask_path"):
            if data.get(k):
                out[k] = data.get(k)
    elif tool_name == "correct_prostate_distortion":
        for k in (
            "out_dir",
            "summary_json_path",
            "backend_out_dir",
            "prediction_npz_path",
            "corrected_b50_nifti",
            "corrected_highb_nifti",
            "corrected_adc_nifti",
            "num_pred_npz",
            "num_panel_png",
            "num_slice_png",
        ):
            if data.get(k) is not None:
                out[k] = data.get(k)
    elif tool_name == "package_vlm_evidence":
        for k in ("vlm_evidence_path",):
            if data.get(k):
                out[k] = data.get(k)
    return out


def append_short_term_event(
    *,
    artifacts_dir: Path,
    step: int,
    tool_name: str,
    tool_result: Dict[str, Any],
    max_chars: int = 2000,
) -> Path:
    mem_dir = artifacts_dir / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    mem_jsonl = mem_dir / "short_term.jsonl"
    event = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "step": int(step),
        "tool_name": str(tool_name),
        "tool_result_compact": compact_for_memory(tool_result, max_chars=max_chars),
        "paths": extract_key_paths(str(tool_name), tool_result),
    }
    with mem_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return mem_jsonl


def build_memory_digest(*, case_state_path: Path, artifacts_dir: Path, max_events: int = 6, autofix_mode: str = "force") -> Dict[str, Any]:
    """
    Structured working memory:
    - recent tool outcomes (events)
    - latest artifact-path summary derived from tool_result truth (case_state.json)
    - coach mode: last reflection first 3 lines (if exists)
    """
    mem_dir = artifacts_dir / "memory"
    mem_jsonl = mem_dir / "short_term.jsonl"

    events: List[Dict[str, Any]] = []
    last_event: Optional[Dict[str, Any]] = None
    if mem_jsonl.exists():
        try:
            lines = mem_jsonl.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = []
        tail = lines[-max_events:] if len(lines) > max_events else lines
        for ln in tail:
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if not isinstance(e, dict):
                continue
            trc = e.get("tool_result_compact") if isinstance(e.get("tool_result_compact"), dict) else {}
            ev = {
                "step": e.get("step"),
                "tool_name": e.get("tool_name"),
                "ok": trc.get("ok") if isinstance(trc, dict) else None,
                "paths": e.get("paths") if isinstance(e.get("paths"), dict) else {},
            }
            events.append(ev)
            last_event = ev

    try:
        s = json.loads(case_state_path.read_text(encoding="utf-8"))
        stage_outputs = s.get("stage_outputs", {}) if isinstance(s, dict) else {}
    except Exception:
        stage_outputs = {}

    def _latest_ok(tool: str) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        best_order = -1
        for _stage, tools in (stage_outputs or {}).items():
            if not isinstance(tools, dict):
                continue
            recs = tools.get(tool)
            if not isinstance(recs, list):
                continue
            for r in recs:
                if not isinstance(r, dict) or not r.get("ok"):
                    continue
                order = int(r.get("stage_order") or 0)
                if order >= best_order:
                    best_order = order
                    d = r.get("data") if isinstance(r.get("data"), dict) else None
                    best = d if isinstance(d, dict) else None
        return best

    latest: Dict[str, Any] = {}
    ingest = _latest_ok("ingest_dicom_to_nifti")
    if not ingest:
        ingest = _latest_ok("identify_sequences")
    if ingest:
        latest["ingest"] = {
            "series_inventory_path": ingest.get("series_inventory_path"),
            "dicom_meta_path": ingest.get("dicom_meta_path"),
            "nifti_by_series": ingest.get("nifti_by_series"),
            "dicom_headers_index_path": ingest.get("dicom_headers_index_path"),
        }
    ident = _latest_ok("identify_sequences")
    if ident and isinstance(ident.get("mapping"), dict):
        latest["mapping"] = ident.get("mapping")

    # collect all successful registration outputs
    try:
        resampled: List[str] = []
        qc_pngs: List[str] = []
        for _stage, tools in (stage_outputs or {}).items():
            if not isinstance(tools, dict):
                continue
            recs = tools.get("register_to_reference")
            if not isinstance(recs, list):
                continue
            for r in recs:
                if not isinstance(r, dict) or not r.get("ok"):
                    continue
                d = r.get("data") if isinstance(r.get("data"), dict) else {}
                rp = d.get("resampled_path")
                if isinstance(rp, str) and rp:
                    resampled.append(rp)
                rps = d.get("resampled_paths")
                if isinstance(rps, dict):
                    for v in rps.values():
                        if isinstance(v, str) and v:
                            resampled.append(v)
                q = d.get("qc_pngs")
                if isinstance(q, dict):
                    for v in q.values():
                        if isinstance(v, str) and v:
                            qc_pngs.append(v)

        def _dedup(xs: List[str]) -> List[str]:
            seen = set()
            outl: List[str] = []
            for x in xs:
                if x in seen:
                    continue
                seen.add(x)
                outl.append(x)
            return outl

        if resampled or qc_pngs:
            latest["registration"] = {"resampled_niftis": _dedup(resampled)[:8], "qc_pngs": _dedup(qc_pngs)[:8]}
    except Exception:
        pass

    seg = _latest_ok("segment_prostate")
    if seg:
        latest["segmentation"] = {"prostate_mask_path": seg.get("prostate_mask_path"), "zone_mask_path": seg.get("zone_mask_path")}
    card_seg = _latest_ok("segment_cardiac_cine")
    if card_seg:
        latest["cardiac_segmentation"] = {
            "seg_path": card_seg.get("seg_path"),
            "rv_mask_path": card_seg.get("rv_mask_path"),
            "myo_mask_path": card_seg.get("myo_mask_path"),
            "lv_mask_path": card_seg.get("lv_mask_path"),
        }
    card_cls = _latest_ok("classify_cardiac_cine_disease")
    if card_cls:
        latest["cardiac_classification"] = {
            "classification_path": card_cls.get("classification_path"),
            "predicted_group": card_cls.get("predicted_group"),
            "needs_vlm_review": card_cls.get("needs_vlm_review"),
        }
    feat = _latest_ok("extract_roi_features")
    if feat:
        latest["features"] = {
            "feature_table_path": feat.get("feature_table_path"),
            "slice_summary_path": feat.get("slice_summary_path"),
            "overlay_pngs": (feat.get("overlay_pngs") or [])[:6] if isinstance(feat.get("overlay_pngs"), list) else None,
        }
    les = _latest_ok("detect_lesion_candidates")
    if les:
        latest["lesion_candidates"] = {
            "candidates_path": les.get("candidates_path"),
            "merged_prob_path": les.get("merged_prob_path"),
            "lesion_mask_path": les.get("lesion_mask_path"),
        }
    dist = _latest_ok("correct_prostate_distortion")
    if dist:
        latest["distortion_correction"] = {
            "out_dir": dist.get("out_dir"),
            "summary_json_path": dist.get("summary_json_path"),
            "backend_out_dir": dist.get("backend_out_dir"),
            "prediction_npz_path": dist.get("prediction_npz_path"),
            "corrected_b50_nifti": dist.get("corrected_b50_nifti"),
            "corrected_highb_nifti": dist.get("corrected_highb_nifti"),
            "corrected_adc_nifti": dist.get("corrected_adc_nifti"),
            "num_pred_npz": dist.get("num_pred_npz"),
            "num_panel_png": dist.get("num_panel_png"),
            "num_slice_png": dist.get("num_slice_png"),
        }
    pkg = _latest_ok("package_vlm_evidence")
    if pkg:
        latest["vlm"] = {"vlm_evidence_path": pkg.get("vlm_evidence_path")}

    # Completed tools list to prevent redundant loops.
    completed_tools: List[Dict[str, Any]] = []
    for _stage, tools in (stage_outputs or {}).items():
        if not isinstance(tools, dict):
            continue
        for tool, recs in tools.items():
            if not isinstance(recs, list):
                continue
            if any(isinstance(r, dict) and r.get("ok") is True for r in recs):
                completed_tools.append({"stage": _stage, "tool_name": tool})

    out: Dict[str, Any] = {
        "events": events,
        "latest": latest,
        "completed_tools": completed_tools,
        "last_event": last_event,
    }
    if autofix_mode == "coach":
        try:
            refls = sorted(mem_dir.glob("reflection_step*.txt"))
            if refls:
                last = refls[-1]
                lines = [ln.strip() for ln in last.read_text(encoding="utf-8").splitlines() if ln.strip()]
                out["coach"] = {"last_reflection_file": str(last), "lines": lines[:3]}
        except Exception:
            pass
    return out

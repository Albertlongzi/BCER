from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from commands.schemas import ToolCall


@dataclass
class CircuitBreakerResult:
    call: Optional[ToolCall]
    calls: Optional[list[ToolCall]]
    note: Optional[Dict[str, Any]]


def _read_case_state(case_state_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(case_state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest_ok_data(case_state: Dict[str, Any], tool_name: str) -> Optional[Dict[str, Any]]:
    stage_outputs = case_state.get("stage_outputs", {}) if isinstance(case_state, dict) else {}
    if not isinstance(stage_outputs, dict):
        return None
    latest_ok = None
    for _stage, tools in stage_outputs.items():
        if not isinstance(tools, dict):
            continue
        recs = tools.get(tool_name)
        if not isinstance(recs, list):
            continue
        for r in recs:
            if isinstance(r, dict) and r.get("ok") is True and isinstance(r.get("data"), dict):
                latest_ok = r.get("data")
    return latest_ok


def _has_success(case_state: Dict[str, Any], tool_name: str) -> bool:
    stage_outputs = case_state.get("stage_outputs", {}) if isinstance(case_state, dict) else {}
    if not isinstance(stage_outputs, dict):
        return False
    for _stage, tools in stage_outputs.items():
        if not isinstance(tools, dict):
            continue
        recs = tools.get(tool_name)
        if not isinstance(recs, list):
            continue
        if any(isinstance(r, dict) and r.get("ok") for r in recs):
            return True
    return False


def _has_attempt(case_state: Dict[str, Any], tool_name: str) -> bool:
    stage_outputs = case_state.get("stage_outputs", {}) if isinstance(case_state, dict) else {}
    if not isinstance(stage_outputs, dict):
        return False
    for _stage, tools in stage_outputs.items():
        if not isinstance(tools, dict):
            continue
        recs = tools.get(tool_name)
        if isinstance(recs, list) and recs:
            return True
    return False


def _best_register_path(case_state: Dict[str, Any], *, subdir_hint: str) -> Optional[Dict[str, Any]]:
    stage_outputs = case_state.get("stage_outputs", {}) if isinstance(case_state, dict) else {}
    if not isinstance(stage_outputs, dict):
        return None
    candidates: list[Dict[str, Any]] = []
    for _stage, tools in stage_outputs.items():
        if not isinstance(tools, dict):
            continue
        recs = tools.get("register_to_reference")
        if not isinstance(recs, list):
            continue
        for r in recs:
            if not isinstance(r, dict) or not r.get("ok"):
                continue
            data = r.get("data") if isinstance(r.get("data"), dict) else {}
            out = str(data.get("output_subdir") or "")
            if subdir_hint.lower() in out.lower():
                candidates.append(data)
    return candidates[-1] if candidates else None


def _pick_highb_from_register(data: Dict[str, Any]) -> Optional[str]:
    resampled_paths = data.get("resampled_paths") if isinstance(data.get("resampled_paths"), dict) else {}
    best_path = ""
    best_b = -1
    for label, path in resampled_paths.items():
        low = str(label).lower()
        m = re.search(r"b(\d+)", low)
        if m:
            bval = int(m.group(1))
            if bval > best_b:
                best_b = bval
                best_path = str(path)
                continue
        if "high" in low and not best_path:
            best_path = str(path)
    if best_path:
        return best_path
    moving = str(data.get("moving") or "").lower()
    resampled = str(data.get("resampled_path") or "")
    if re.search(r"b(1\d{3}|[2-9]\d{3})", moving) or "high" in moving:
        return resampled or None
    return None


def _stage_sort_key(k: str) -> Tuple[int, int, str]:
    try:
        return (0, int(k), "")
    except Exception:
        return (1, 10**9, str(k))


def _latest_error_info(case_state: Dict[str, Any], tool_name: str) -> Tuple[Optional[Dict[str, Any]], int]:
    stage_outputs = case_state.get("stage_outputs", {}) if isinstance(case_state, dict) else {}
    if not isinstance(stage_outputs, dict):
        return None, 0
    errors: list[Dict[str, Any]] = []
    for stage in sorted(stage_outputs.keys(), key=_stage_sort_key):
        tools = stage_outputs.get(stage, {}) if isinstance(stage_outputs.get(stage), dict) else {}
        recs = tools.get(tool_name)
        if not isinstance(recs, list):
            continue
        for r in recs:
            if not isinstance(r, dict):
                continue
            if r.get("ok") is True:
                continue
            err = r.get("error")
            if not isinstance(err, dict):
                data = r.get("data") if isinstance(r.get("data"), dict) else {}
                err = data.get("error")
            if isinstance(err, dict):
                errors.append(err)
    if not errors:
        return None, 0
    last_err = errors[-1]
    last_msg = str(last_err.get("message") or "")
    repeat = 0
    for e in reversed(errors):
        if str(e.get("message") or "") == last_msg:
            repeat += 1
        else:
            break
    return last_err, repeat


def _successful_tool_records(case_state: Dict[str, Any], tool_name: str) -> list[Dict[str, Any]]:
    stage_outputs = case_state.get("stage_outputs", {}) if isinstance(case_state, dict) else {}
    if not isinstance(stage_outputs, dict):
        return []
    out: list[Dict[str, Any]] = []
    for stage in sorted(stage_outputs.keys(), key=_stage_sort_key):
        tools = stage_outputs.get(stage, {}) if isinstance(stage_outputs.get(stage), dict) else {}
        recs = tools.get(tool_name)
        if not isinstance(recs, list):
            continue
        for r in recs:
            if isinstance(r, dict) and r.get("ok") is True:
                out.append(r)
    return out


def _register_signature(rec: Dict[str, Any]) -> str:
    data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
    fixed = str(data.get("fixed") or "")
    moving = str(data.get("moving") or "")
    out_subdir = str(data.get("output_subdir") or "")
    return f"{out_subdir}|{fixed}|{moving}"


def _register_repeat_count(case_state: Dict[str, Any]) -> Tuple[int, str]:
    recs = _successful_tool_records(case_state, "register_to_reference")
    if not recs:
        return 0, ""
    last_sig = _register_signature(recs[-1])
    if not last_sig:
        return 0, ""
    count = 0
    for r in reversed(recs):
        if _register_signature(r) != last_sig:
            break
        count += 1
    return count, last_sig


def _pick_high_b_from_nifti(nifti_by_series: Any) -> Optional[str]:
    if not isinstance(nifti_by_series, dict):
        return None
    best_b = None
    best_path = None
    for key, path in nifti_by_series.items():
        if not isinstance(path, str) or not path:
            continue
        low = str(key).lower()
        m = re.search(r"b(?:=|_|-)?(\d{2,5})", low)
        b_hint = float(m.group(1)) if m else None
        if b_hint is None and ("high" in low and "b" in low):
            b_hint = 2000.0
        if b_hint is None:
            continue
        if best_b is None or b_hint > best_b:
            best_b = b_hint
            best_path = path
    return best_path


def _is_fatal_tool_error(err: Dict[str, Any]) -> bool:
    err_type = str(err.get("type") or "").strip()
    msg = str(err.get("message") or "").lower()
    if err_type in {
        "NameError",
        "UnboundLocalError",
        "AttributeError",
        "TypeError",
        "KeyError",
        "SyntaxError",
        "ImportError",
        "ModuleNotFoundError",
    }:
        return True
    if "is not defined" in msg:
        return True
    if "traceback" in msg and "line" in msg:
        return True
    return False


def _is_resource_exhausted_error(err: Dict[str, Any]) -> bool:
    err_type = str(err.get("type") or "").strip().lower()
    msg = str(err.get("message") or "").lower()
    if "outofmemory" in err_type:
        return True
    if "out of memory" in msg:
        return True
    if "cublas_status_alloc_failed" in msg:
        return True
    if "cuda error" in msg and ("alloc" in msg or "memory" in msg):
        return True
    return False


def apply_circuit_breaker(
    *,
    case_state_path: Path,
    case_id: str,
    last_tool_name: Optional[str],
    last_tool_ok: Optional[bool],
) -> CircuitBreakerResult:
    cs = _read_case_state(case_state_path)
    domain = None
    try:
        if isinstance(cs, dict):
            meta = cs.get("metadata", {}) if isinstance(cs.get("metadata"), dict) else {}
            domain = str(meta.get("domain") or "").strip().lower() or None
    except Exception:
        domain = None

    if last_tool_name == "brats_mri_segmentation" and last_tool_ok is True:
        package_ok = _has_success(cs, "package_vlm_evidence")
        report_ok = _has_success(cs, "generate_report")
        features_ok = _has_success(cs, "extract_roi_features")

        if not features_ok:
            brats = _latest_ok_data(cs, "brats_mri_segmentation") or {}
            tc = brats.get("tc_mask_path")
            wt = brats.get("wt_mask_path")
            et = brats.get("et_mask_path")
            roi_masks = []
            for name, path in (
                ("Tumor_Core", tc),
                ("Whole_Tumor", wt),
                ("Enhancing_Tumor", et),
            ):
                if isinstance(path, str) and path:
                    roi_masks.append({"name": name, "path": path})

            ident = _latest_ok_data(cs, "identify_sequences") or {}
            mapping = ident.get("mapping", {}) if isinstance(ident, dict) else {}
            images = []
            for name in ("T1c", "T1", "T2", "FLAIR"):
                p = mapping.get(name)
                if isinstance(p, str) and p:
                    images.append({"name": name, "path": p})

            if roi_masks and images:
                call = ToolCall(
                    tool_name="extract_roi_features",
                    arguments={
                        "roi_masks": roi_masks,
                        "images": images,
                        "roi_interpretation": "binary_masks",
                        "output_subdir": "features",
                    },
                    call_id=f"cb_{uuid.uuid4().hex[:8]}",
                    case_id=case_id,
                    stage="extract",
                    requested_by="hooks:circuit_breaker",
                )
                note = {
                    "reason": "brats_mri_segmentation succeeded; extract tumor ROI features before reporting.",
                    "next_tool_call": {"tool_name": call.tool_name, "arguments": call.arguments},
                }
                return CircuitBreakerResult(call=call, calls=None, note=note)

        if not package_ok:
            call = ToolCall(
                tool_name="package_vlm_evidence",
                arguments={"case_state_path": str(case_state_path), "output_subdir": "vlm"},
                call_id=f"cb_{uuid.uuid4().hex[:8]}",
                case_id=case_id,
                stage="package",
                requested_by="hooks:circuit_breaker",
            )
            note = {
                "reason": "brats_mri_segmentation succeeded; avoid repeating segmentation and package evidence instead.",
                "next_tool_call": {"tool_name": call.tool_name, "arguments": call.arguments},
            }
            return CircuitBreakerResult(call=call, calls=None, note=note)
        if not report_ok:
            call = ToolCall(
                tool_name="generate_report",
                arguments={
                    "case_state_path": str(case_state_path),
                    "output_subdir": "report",
                    **({"domain": domain} if domain else {}),
                },
                call_id=f"cb_{uuid.uuid4().hex[:8]}",
                case_id=case_id,
                stage="report",
                requested_by="hooks:circuit_breaker",
            )
            note = {
                "reason": "brats_mri_segmentation succeeded; avoid repeating segmentation and generate report instead.",
                "next_tool_call": {"tool_name": call.tool_name, "arguments": call.arguments},
            }
            return CircuitBreakerResult(call=call, calls=None, note=note)

    if last_tool_name == "register_to_reference" and last_tool_ok is True:
        repeat_count, sig = _register_repeat_count(cs)
        if repeat_count >= 3 and sig:
            ident = _latest_ok_data(cs, "identify_sequences") or {}
            mapping = ident.get("mapping", {}) if isinstance(ident, dict) else {}
            nifti_by_series = ident.get("nifti_by_series", {}) if isinstance(ident, dict) else {}
            fixed = str(mapping.get("T2w") or "")
            dwi = str(mapping.get("DWI") or "")
            highb = _pick_high_b_from_nifti(nifti_by_series)
            if highb:
                dwi = str(highb)

            has_dwi_reg = _best_register_path(cs, subdir_hint="registration/DWI") is not None
            has_seg = _has_success(cs, "segment_prostate")
            has_feat = _has_success(cs, "extract_roi_features")
            has_pkg = _has_success(cs, "package_vlm_evidence")
            has_report = _has_success(cs, "generate_report")

            call: Optional[ToolCall] = None
            reason = "Repeated identical register_to_reference calls detected; advancing pipeline."
            if "registration/adc" in sig.lower() and fixed and dwi and not has_dwi_reg:
                call = ToolCall(
                    tool_name="register_to_reference",
                    arguments={
                        "fixed": fixed,
                        "moving": dwi,
                        "output_subdir": "registration/DWI",
                        "split_by_bvalue": True,
                        "save_png_qc": True,
                    },
                    call_id=f"cb_{uuid.uuid4().hex[:8]}",
                    case_id=case_id,
                    stage="register",
                    requested_by="hooks:circuit_breaker",
                )
                reason = "Repeated ADC registration loop detected; forcing DWI registration next."
            elif fixed and not has_seg:
                call = ToolCall(
                    tool_name="segment_prostate",
                    arguments={"t2w_ref": fixed, "output_subdir": "segmentation"},
                    call_id=f"cb_{uuid.uuid4().hex[:8]}",
                    case_id=case_id,
                    stage="segment",
                    requested_by="hooks:circuit_breaker",
                )
            elif not has_feat:
                call = ToolCall(
                    tool_name="extract_roi_features",
                    arguments={"output_subdir": "features"},
                    call_id=f"cb_{uuid.uuid4().hex[:8]}",
                    case_id=case_id,
                    stage="extract",
                    requested_by="hooks:circuit_breaker",
                )
            elif not has_pkg:
                call = ToolCall(
                    tool_name="package_vlm_evidence",
                    arguments={"case_state_path": str(case_state_path), "output_subdir": "vlm"},
                    call_id=f"cb_{uuid.uuid4().hex[:8]}",
                    case_id=case_id,
                    stage="package",
                    requested_by="hooks:circuit_breaker",
                )
            elif not has_report:
                call = ToolCall(
                    tool_name="generate_report",
                    arguments={
                        "case_state_path": str(case_state_path),
                        "output_subdir": "report",
                        **({"domain": domain} if domain else {}),
                    },
                    call_id=f"cb_{uuid.uuid4().hex[:8]}",
                    case_id=case_id,
                    stage="report",
                    requested_by="hooks:circuit_breaker",
                )

            if call is not None:
                note = {
                    "reason": reason,
                    "repeat_count": repeat_count,
                    "register_signature": sig,
                    "next_tool_call": {"tool_name": call.tool_name, "arguments": call.arguments},
                }
                return CircuitBreakerResult(call=call, calls=None, note=note)

    if last_tool_name == "detect_lesion_candidates" and last_tool_ok is False:
        call = ToolCall(
            tool_name="package_vlm_evidence",
            arguments={"case_state_path": str(case_state_path), "output_subdir": "vlm"},
            call_id=f"cb_{uuid.uuid4().hex[:8]}",
            case_id=case_id,
            stage="package",
            requested_by="hooks:circuit_breaker",
        )
        note = {
            "reason": "detect_lesion_candidates failed; stop retrying and proceed by packaging evidence for reporting.",
            "next_tool_call": {"tool_name": call.tool_name, "arguments": call.arguments},
        }
        return CircuitBreakerResult(call=call, calls=None, note=note)

    if last_tool_name == "segment_prostate" and last_tool_ok is False:
        last_err, repeat = _latest_error_info(cs, "segment_prostate")
        if isinstance(last_err, dict):
            resource_exhausted = _is_resource_exhausted_error(last_err)
            if resource_exhausted or repeat >= 2:
                has_report = _has_success(cs, "generate_report")
                if not has_report:
                    call = ToolCall(
                        tool_name="generate_report",
                        arguments={
                            "case_state_path": str(case_state_path),
                            "output_subdir": "report",
                            **({"domain": domain} if domain else {}),
                        },
                        call_id=f"cb_{uuid.uuid4().hex[:8]}",
                        case_id=case_id,
                        stage="report",
                        requested_by="hooks:circuit_breaker",
                    )
                    note = {
                        "reason": (
                            "segment_prostate failed repeatedly or hit resource limits; "
                            "switching to degraded report generation to avoid retry loops."
                        ),
                        "error": last_err,
                        "repeat_count": repeat,
                        "next_tool_call": {"tool_name": call.tool_name, "arguments": call.arguments},
                    }
                    return CircuitBreakerResult(call=call, calls=None, note=note)
                note = {
                    "fatal": True,
                    "tool_name": "segment_prostate",
                    "reason": "segment_prostate failed repeatedly and report already exists; stop retrying.",
                    "error": last_err,
                    "repeat_count": repeat,
                }
                return CircuitBreakerResult(call=None, calls=None, note=note)

    if last_tool_name == "extract_roi_features" and last_tool_ok is True and (domain or "") == "prostate":
        if _has_attempt(cs, "detect_lesion_candidates"):
            return CircuitBreakerResult(call=None, calls=None, note=None)
        adc_reg = _best_register_path(cs, subdir_hint="registration/ADC") or {}
        dwi_reg = _best_register_path(cs, subdir_hint="registration/DWI") or {}
        adc_path = str(adc_reg.get("resampled_path") or "")
        highb_path = _pick_highb_from_register(dwi_reg) or ""
        t2w_path = str(adc_reg.get("fixed") or dwi_reg.get("fixed") or "")

        seg = _latest_ok_data(cs, "segment_prostate") or {}
        prostate_mask = str(seg.get("prostate_mask_path") or "")

        prereq_ok = all([adc_path, highb_path, t2w_path, prostate_mask])
        if prereq_ok:
            call = ToolCall(
                tool_name="detect_lesion_candidates",
                arguments={
                    "t2w_nifti": t2w_path,
                    "adc_nifti": adc_path,
                    "highb_nifti": highb_path,
                    "prostate_mask_nifti": prostate_mask,
                },
                call_id=f"cb_{uuid.uuid4().hex[:8]}",
                case_id=case_id,
                stage="lesion",
                requested_by="hooks:circuit_breaker",
            )
            note = {
                "reason": "Auto-trigger lesion candidate detection after successful ROI features (prostate).",
                "next_tool_call": {"tool_name": call.tool_name, "arguments": call.arguments},
            }
            return CircuitBreakerResult(call=call, calls=None, note=note)

    if last_tool_name == "generate_report" and last_tool_ok is False:
        last_err, repeat = _latest_error_info(cs, "generate_report")
        if isinstance(last_err, dict):
            fatal = _is_fatal_tool_error(last_err) or repeat >= 2
            if fatal:
                note = {
                    "fatal": True,
                    "tool_name": "generate_report",
                    "reason": "generate_report failed with a non-retriable error; stop retrying and surface the failure.",
                    "error": last_err,
                    "repeat_count": repeat,
                }
                return CircuitBreakerResult(call=None, calls=None, note=note)

    if last_tool_name == "detect_lesion_candidates" and last_tool_ok is True:
        cs = _read_case_state(case_state_path)
        stage_outputs = cs.get("stage_outputs", {}) if isinstance(cs, dict) else {}

        lesion_done = False
        if isinstance(stage_outputs, dict):
            for _stage, tools in stage_outputs.items():
                if not isinstance(tools, dict):
                    continue
                recs = tools.get("extract_roi_features")
                if not isinstance(recs, list):
                    continue
                for r in recs:
                    if not isinstance(r, dict) or not r.get("ok"):
                        continue
                    data = r.get("data") if isinstance(r.get("data"), dict) else {}
                    sources = data.get("roi_sources") if isinstance(data, dict) else None
                    if isinstance(sources, list) and any(str(s).lower() == "lesion" for s in sources):
                        lesion_done = True
                        break
                if lesion_done:
                    break

        if not lesion_done:
            lesion_mask = None
            zone_mask = None
            try:
                for _stage, tools in (stage_outputs or {}).items():
                    if not isinstance(tools, dict):
                        continue
                    recs = tools.get("detect_lesion_candidates")
                    if isinstance(recs, list) and recs:
                        rec = recs[-1]
                        if isinstance(rec, dict) and isinstance(rec.get("data"), dict):
                            lesion_mask = rec.get("data", {}).get("lesion_mask_path")
                    seg_recs = tools.get("segment_prostate")
                    if isinstance(seg_recs, list) and seg_recs:
                        seg_rec = seg_recs[-1]
                        if isinstance(seg_rec, dict) and isinstance(seg_rec.get("data"), dict):
                            zone_mask = seg_rec.get("data", {}).get("zone_mask_path")
            except Exception:
                lesion_mask = None
                zone_mask = None

            if isinstance(lesion_mask, str) and lesion_mask:
                roi_masks: list[Dict[str, str]] = []
                try:
                    detect_data = {}
                    for _stage, tools in (stage_outputs or {}).items():
                        if not isinstance(tools, dict):
                            continue
                        recs = tools.get("detect_lesion_candidates")
                        if isinstance(recs, list) and recs:
                            rec = recs[-1]
                            if isinstance(rec, dict) and isinstance(rec.get("data"), dict):
                                detect_data = rec.get("data") or {}
                    comp_masks = detect_data.get("component_mask_paths") if isinstance(detect_data, dict) else None
                    if isinstance(comp_masks, list):
                        for cm in comp_masks:
                            if not isinstance(cm, dict):
                                continue
                            p = str(cm.get("path") or "").strip()
                            if not p:
                                continue
                            cid = cm.get("component_id")
                            roi_name = f"lesion_component_{cid}" if cid is not None else "lesion"
                            roi_masks.append({"name": roi_name, "path": p})
                except Exception:
                    roi_masks = []
                if not roi_masks:
                    roi_masks = [{"name": "lesion", "path": lesion_mask}]

                lesion_call = ToolCall(
                    tool_name="extract_roi_features",
                    arguments={
                        "roi_masks": roi_masks,
                        "images": [],
                        # Keep lesion features separate so we don't overwrite whole-gland stats.
                        "output_subdir": "features_lesion",
                    },
                    call_id=f"cb_{uuid.uuid4().hex[:8]}",
                    case_id=case_id,
                    stage="extract",
                    requested_by="hooks:circuit_breaker",
                )
                # Follow with a main-feature refresh aligned to lesion slices when possible.
                ref_slice_summary = case_state_path.parent / "artifacts" / "features_lesion" / "slice_summary.json"
                main_args: Dict[str, Any] = {
                    "images": [],
                    "output_subdir": "features",
                    "reference_slice_summary_path": str(ref_slice_summary),
                }
                if isinstance(zone_mask, str) and zone_mask:
                    main_args["roi_mask_path"] = zone_mask
                main_call = ToolCall(
                    tool_name="extract_roi_features",
                    arguments=main_args,
                    call_id=f"cb_{uuid.uuid4().hex[:8]}",
                    case_id=case_id,
                    stage="extract",
                    requested_by="hooks:circuit_breaker",
                )
                note = {
                    "reason": "lesion candidates available; extract lesion ROI features, then refresh prostate features aligned to lesion slices.",
                    "next_tool_calls": [
                        {"tool_name": lesion_call.tool_name, "arguments": lesion_call.arguments},
                        {"tool_name": main_call.tool_name, "arguments": main_call.arguments},
                    ],
                }
                return CircuitBreakerResult(call=None, calls=[lesion_call, main_call], note=note)

    return CircuitBreakerResult(call=None, calls=None, note=None)

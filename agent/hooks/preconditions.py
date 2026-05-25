from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from commands.schemas import ToolCall


@dataclass
class PreconditionsResult:
    call: ToolCall
    pre_calls: List[ToolCall]
    notes: List[Dict[str, Any]]


def _read_case_state(case_state_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(case_state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest_tool_data(state_path: Path, stage: str, tool_name: str) -> Dict[str, Any]:
    def _stage_sort_key(k: str):
        try:
            return (0, int(k), "")
        except Exception:
            return (1, 10**9, str(k))

    try:
        s = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    all_stage_outputs: Dict[str, Any] = s.get("stage_outputs", {}) or {}

    stage_outputs = (all_stage_outputs.get(stage, {}) or {}) if isinstance(all_stage_outputs, dict) else {}
    if isinstance(stage_outputs, dict):
        records = stage_outputs.get(tool_name, []) or []
        if records:
            rec = records[-1] or {}
            return rec.get("data", {}) or {}

    if not isinstance(all_stage_outputs, dict):
        return {}
    latest: Dict[str, Any] = {}
    for st in sorted(all_stage_outputs.keys(), key=_stage_sort_key):
        tools = all_stage_outputs.get(st, {}) or {}
        if not isinstance(tools, dict):
            continue
        records = tools.get(tool_name, []) or []
        if not isinstance(records, list) or not records:
            continue
        rec = records[-1] or {}
        latest = rec.get("data", {}) or {}
    return latest or {}


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


def _artifact_glob(case_state_path: Path, patterns: List[str]) -> bool:
    run_art_dir = case_state_path.parent / "artifacts"
    for pat in patterns:
        if list(run_art_dir.glob(pat)):
            return True
    return False


def _has_adc_registration(case_state_path: Path) -> bool:
    run_art_dir = case_state_path.parent / "artifacts"
    try:
        for p in run_art_dir.glob("**/*resampled_to_fixed.nii.gz"):
            s = str(p).lower().replace("\\", "/")
            n = p.name.lower()
            if "/registration/adc/" in s and ("adc" in n or "moving_all_resampled_to_fixed" in n):
                return True
            if "/registered/" in s and ("adc" in n or "moving_all_resampled_to_fixed" in n):
                return True
    except Exception:
        return False
    return False


def _has_dwi_registration(case_state_path: Path) -> bool:
    run_art_dir = case_state_path.parent / "artifacts"
    try:
        for p in run_art_dir.glob("**/*resampled_to_fixed.nii.gz"):
            s = str(p).lower().replace("\\", "/")
            n = p.name.lower()
            if "/registration/adc/" in s:
                continue
            looks_dwi = (
                "moving_high_b_resampled_to_fixed" in n
                or "moving_dwi_resampled_to_fixed" in n
                or "moving_mid_b_resampled_to_fixed" in n
                or "moving_low_b_resampled_to_fixed" in n
                or ("moving_b" in n and "resampled_to_fixed" in n)
            )
            if not looks_dwi:
                continue
            if "/registration/dwi/" in s or "/registered/" in s:
                return True
    except Exception:
        return False
    return False


def _load_alignment_gate(case_state_path: Path) -> Optional[Dict[str, Any]]:
    try:
        p = case_state_path.parent / "artifacts" / "context" / "alignment_gate.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _pick_high_b_from_nifti(nifti_by_series: Any) -> Optional[str]:
    if not isinstance(nifti_by_series, dict):
        return None
    best_b = None
    best_path = None
    for key, path in nifti_by_series.items():
        if not isinstance(path, str) or not path:
            continue
        kl = str(key).lower()
        if "dwi" not in kl and "trace" not in kl:
            continue
        m = re.search(r"b(?:=|_|-)?(\d{2,5})", kl)
        b_hint = float(m.group(1)) if m else None
        if b_hint is None:
            if "high" in kl and "b" in kl:
                b_hint = 2000.0
            elif "mid" in kl and "b" in kl:
                b_hint = 800.0
            elif "low" in kl and "b" in kl:
                b_hint = 50.0
        if b_hint is None:
            continue
        if best_b is None or b_hint > best_b:
            best_b = b_hint
            best_path = path
    return best_path


def _maybe_repair(call: ToolCall, *, autofix_mode: str, auto_repair_fn: Optional[Callable]) -> ToolCall:
    if autofix_mode not in ("force", "coach") or auto_repair_fn is None:
        return call
    repaired = auto_repair_fn(
        call.tool_name,
        dict(call.arguments or {}),
        state_path=call.arguments.get("case_state_path") or call.arguments.get("state_path"),
        ctx_case_state_path=call.arguments.get("case_state_path") or call.arguments.get("state_path"),
        dicom_case_dir=None,
    )
    call.arguments = repaired
    return call


def apply_preconditions(
    call: ToolCall,
    *,
    case_state_path: Path,
    dicom_case_dir: Optional[str],
    autofix_mode: str,
    strategy: str = "run_before",
    auto_repair_fn: Optional[Callable] = None,
) -> PreconditionsResult:
    notes: List[Dict[str, Any]] = []
    pre_calls: List[ToolCall] = []
    tool_name = call.tool_name

    case_state = _read_case_state(case_state_path)
    domain_name = "prostate"
    try:
        meta = case_state.get("metadata", {}) if isinstance(case_state, dict) else {}
        if isinstance(meta, dict):
            domain_name = str(meta.get("domain") or "").strip().lower() or "prostate"
    except Exception:
        domain_name = "prostate"
    mapping = {}
    nifti_by_series = {}
    gate = _load_alignment_gate(case_state_path)
    try:
        ident = _latest_tool_data(case_state_path, "identify", "identify_sequences") or {}
        mapping = ident.get("mapping", {}) if isinstance(ident, dict) else {}
        nifti_by_series = ident.get("nifti_by_series", {}) if isinstance(ident, dict) else {}
    except Exception:
        mapping = {}
        nifti_by_series = {}

    def _make_call(tool: str, args: Dict[str, Any], stage: str, reason: str) -> ToolCall:
        repaired_args = dict(args)
        if autofix_mode in ("force", "coach") and auto_repair_fn is not None:
            repaired_args = auto_repair_fn(
                tool,
                dict(args),
                state_path=case_state_path,
                ctx_case_state_path=case_state_path,
                dicom_case_dir=dicom_case_dir,
            )
        c = ToolCall(
            tool_name=tool,
            arguments=repaired_args,
            call_id=f"pre_{uuid.uuid4().hex[:8]}",
            case_id=call.case_id,
            stage=stage,
            requested_by="hooks:preconditions",
        )
        notes.append({"reason": reason, "tool_name": tool, "arguments": c.arguments})
        return c

    def _gate_skip(seq_key: str) -> bool:
        if not isinstance(gate, dict):
            return False
        decs = gate.get("decisions", {})
        if not isinstance(decs, dict):
            return False
        seq = decs.get(seq_key, {})
        if not isinstance(seq, dict):
            return False
        return seq.get("need_register") is False

    def _pick_cine_path() -> Optional[str]:
        cine = mapping.get("CINE") if isinstance(mapping, dict) else None
        if isinstance(cine, str) and cine:
            return cine
        if isinstance(nifti_by_series, dict):
            for _, p in nifti_by_series.items():
                if isinstance(p, str) and p:
                    return p
        if isinstance(mapping, dict):
            for _, p in mapping.items():
                if isinstance(p, str) and p:
                    return p
        return None

    if tool_name == "identify_sequences":
        # If identify already succeeded, advance the pipeline instead of repeating ingest/identify.
        have_identify = _has_success(case_state, "identify_sequences")
        fixed = mapping.get("T2w") if isinstance(mapping, dict) else None
        cine = _pick_cine_path()
        adc = mapping.get("ADC") if isinstance(mapping, dict) else None
        dwi = mapping.get("DWI") if isinstance(mapping, dict) else None
        dwi_high = _pick_high_b_from_nifti(nifti_by_series)
        if dwi_high:
            dwi = dwi_high
        have_adc_reg = _has_adc_registration(case_state_path)
        have_dwi_reg = _has_dwi_registration(case_state_path)
        replacement: Optional[ToolCall] = None
        if domain_name == "cardiac":
            if have_identify and cine and not _has_success(case_state, "segment_cardiac_cine"):
                replacement = _make_call(
                    "segment_cardiac_cine",
                    {"cine_path": cine, "output_subdir": "segmentation/cardiac_cine"},
                    "segment",
                    "identify_sequences already succeeded; segment cardiac cine next.",
                )
            elif have_identify and _has_success(case_state, "segment_cardiac_cine") and not _has_success(case_state, "classify_cardiac_cine_disease"):
                replacement = _make_call(
                    "classify_cardiac_cine_disease",
                    {"cine_path": cine, "output_subdir": "classification/cardiac_cine"},
                    "report",
                    "cardiac segmentation already succeeded; classify cine disease next.",
                )
            elif have_identify and _has_success(case_state, "classify_cardiac_cine_disease") and not _has_success(case_state, "extract_roi_features"):
                replacement = _make_call(
                    "extract_roi_features",
                    {"output_subdir": "features"},
                    "extract",
                    "cardiac classification already succeeded; extract ROI/radiomics features next.",
                )
        elif have_identify and fixed and adc and not have_adc_reg:
            adc_out_subdir = "registered" if (case_state_path.parent / "artifacts" / "registered").exists() else "registration/ADC"
            if _gate_skip("ADC"):
                replacement = _make_call(
                    "materialize_registration",
                    {"fixed": fixed, "moving": adc, "output_subdir": adc_out_subdir, "label": "adc", "save_png_qc": True},
                    "register",
                    "alignment gate indicates ADC is already aligned; materializing outputs.",
                )
            else:
                replacement = _make_call(
                    "register_to_reference",
                    {
                        "fixed": fixed,
                        "moving": adc,
                        "output_subdir": adc_out_subdir,
                        "split_by_bvalue": False,
                        "save_png_qc": True,
                    },
                    "register",
                    "identify_sequences already succeeded; registering ADC to T2w next.",
                )
        elif have_identify and fixed and dwi and not have_dwi_reg:
            dwi_out_subdir = "registered" if (case_state_path.parent / "artifacts" / "registered").exists() else "registration/DWI"
            if _gate_skip("DWI"):
                replacement = _make_call(
                    "materialize_registration",
                    {"fixed": fixed, "moving": dwi, "output_subdir": dwi_out_subdir, "label": "high_b", "save_png_qc": True},
                    "register",
                    "alignment gate indicates DWI is already aligned; materializing outputs.",
                )
            else:
                replacement = _make_call(
                    "register_to_reference",
                    {
                        "fixed": fixed,
                        "moving": dwi,
                        "output_subdir": dwi_out_subdir,
                        "split_by_bvalue": True,
                        "save_png_qc": True,
                    },
                    "register",
                    "identify_sequences already succeeded; registering DWI to T2w next.",
                )
        elif have_identify and fixed and not _has_success(case_state, "segment_prostate"):
            replacement = _make_call(
                "segment_prostate",
                {"t2w_ref": fixed, "output_subdir": "segmentation"},
                "segment",
                "identify_sequences already succeeded; segment prostate on T2w next.",
            )
        elif have_identify and not _has_success(case_state, "extract_roi_features"):
            replacement = _make_call(
                "extract_roi_features",
                {"output_subdir": "features"},
                "extract",
                "identify_sequences already succeeded; extract ROI features next.",
            )
        if replacement is not None and strategy == "replace":
            return PreconditionsResult(call=replacement, pre_calls=[], notes=notes)

    if tool_name == "extract_roi_features":
        fixed = mapping.get("T2w") if isinstance(mapping, dict) else None
        adc = mapping.get("ADC") if isinstance(mapping, dict) else None
        if fixed and adc:
            have_adc_nifti = _has_adc_registration(case_state_path) or _artifact_glob(case_state_path, ["**/*adc*.nii.gz"])
            if not have_adc_nifti:
                adc_out_subdir = "registered" if (case_state_path.parent / "artifacts" / "registered").exists() else "registration/ADC"
                pre_calls.append(
                    _make_call(
                        "register_to_reference",
                        {
                            "fixed": fixed,
                            "moving": adc,
                            "output_subdir": adc_out_subdir,
                            "split_by_bvalue": False,
                            "save_png_qc": True,
                        },
                        "register",
                        "extract_roi_features requested; registering ADC to T2w first.",
                    )
                )

    if tool_name == "classify_cardiac_cine_disease":
        if domain_name == "cardiac" and not _has_success(case_state, "segment_cardiac_cine"):
            cine = _pick_cine_path()
            if cine:
                pre_calls.append(
                    _make_call(
                        "segment_cardiac_cine",
                        {"cine_path": cine, "output_subdir": "segmentation/cardiac_cine"},
                        "segment",
                        "cardiac classification requested before segmentation; segment cardiac cine first.",
                    )
                )

    if tool_name == "detect_lesion_candidates":
        fixed = mapping.get("T2w") if isinstance(mapping, dict) else None
        adc = mapping.get("ADC") if isinstance(mapping, dict) else None
        dwi = mapping.get("DWI") if isinstance(mapping, dict) else None
        dwi_high = _pick_high_b_from_nifti(nifti_by_series)
        if dwi_high:
            dwi = dwi_high

        have_adc_reg = _has_adc_registration(case_state_path)
        have_dwi_reg = _has_dwi_registration(case_state_path)

        if fixed and adc and not have_adc_reg:
            adc_out_subdir = "registered" if (case_state_path.parent / "artifacts" / "registered").exists() else "registration/ADC"
            pre_calls.append(
                _make_call(
                    "register_to_reference",
                    {
                        "fixed": fixed,
                        "moving": adc,
                        "output_subdir": adc_out_subdir,
                        "split_by_bvalue": False,
                        "save_png_qc": True,
                    },
                    "register",
                    "Lesion detection requested; registering ADC to T2w first.",
                )
            )
        if fixed and dwi and not have_dwi_reg:
            dwi_out_subdir = "registered" if (case_state_path.parent / "artifacts" / "registered").exists() else "registration/DWI"
            pre_calls.append(
                _make_call(
                    "register_to_reference",
                    {
                        "fixed": fixed,
                        "moving": dwi,
                        "output_subdir": dwi_out_subdir,
                        "split_by_bvalue": True,
                        "save_png_qc": True,
                    },
                    "register",
                    "Lesion detection requested; registering DWI (high-b) to T2w first.",
                )
            )

        have_features = _has_success(case_state, "extract_roi_features")
        if not have_features:
            pre_calls.append(
                _make_call(
                    "extract_roi_features",
                    {"output_subdir": "features"},
                    "extract",
                    "Lesion detection requested before extract_roi_features; running extract_roi_features first.",
                )
            )

    if tool_name == "package_vlm_evidence":
        if domain_name == "cardiac":
            if not _has_success(case_state, "extract_roi_features"):
                pre_calls.append(
                    _make_call(
                        "extract_roi_features",
                        {"output_subdir": "features"},
                        "extract",
                        "package_vlm_evidence requested before cardiac ROI/radiomics features; extracting features first.",
                    )
                )
            if not _has_success(case_state, "classify_cardiac_cine_disease"):
                cine = _pick_cine_path()
                pre_calls.append(
                    _make_call(
                        "classify_cardiac_cine_disease",
                        {"cine_path": cine, "output_subdir": "classification/cardiac_cine"},
                        "report",
                        "package_vlm_evidence requested before cardiac disease classification; classifying first.",
                    )
                )
        else:
            fixed = mapping.get("T2w") if isinstance(mapping, dict) else None
            dwi = mapping.get("DWI") if isinstance(mapping, dict) else None
            have_dwi_reg = _has_dwi_registration(case_state_path)
            if fixed and dwi and not have_dwi_reg:
                pre_calls.append(
                    _make_call(
                        "register_to_reference",
                        {
                            "fixed": fixed,
                            "moving": dwi,
                            "output_subdir": "registration/DWI",
                            "split_by_bvalue": True,
                            "save_png_qc": True,
                        },
                        "register",
                        "package_vlm_evidence requested but DWI not registered; registering DWI to T2w first.",
                    )
                )
            if not _has_success(case_state, "extract_roi_features"):
                pre_calls.append(
                    _make_call(
                        "extract_roi_features",
                        {"output_subdir": "features"},
                        "extract",
                        "package_vlm_evidence requested before extract_roi_features; running extract_roi_features first.",
                    )
                )

    if tool_name == "generate_report":
        if domain_name == "cardiac" and not _has_success(case_state, "extract_roi_features"):
            pre_calls.append(
                _make_call(
                    "extract_roi_features",
                    {"output_subdir": "features"},
                    "extract",
                    "generate_report requested before cardiac ROI/radiomics features; extracting features first.",
                )
            )

    if strategy == "replace" and pre_calls:
        # Only return the first precondition call; let the LLM decide next step afterward.
        return PreconditionsResult(call=pre_calls[0], pre_calls=[], notes=notes)

    return PreconditionsResult(call=call, pre_calls=pre_calls, notes=notes)

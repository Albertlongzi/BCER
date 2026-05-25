from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from commands.schemas import ToolCall


@dataclass(frozen=True)
class RuleViolation:
    rule_id: str
    level: str  # hard|soft
    tool_name: str
    message: str
    details: Optional[Dict[str, Any]] = None


def _read_case_state(case_state_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(case_state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def _path_exists(value: str) -> bool:
    try:
        return Path(value).expanduser().resolve().exists()
    except Exception:
        return False


def _looks_like(series: str, tokens: List[str]) -> bool:
    s = str(series or "").lower()
    return any(tok in s for tok in tokens)


def validate_tool_call(call: ToolCall, case_state_path: Path) -> List[RuleViolation]:
    violations: List[RuleViolation] = []
    tool_name = call.tool_name
    args = call.arguments or {}

    if tool_name == "register_to_reference":
        fixed = str(args.get("fixed") or "")
        moving = str(args.get("moving") or "")
        if not fixed or not moving:
            violations.append(
                RuleViolation(
                    rule_id="register_requires_fixed_moving",
                    level="hard",
                    tool_name=tool_name,
                    message="register_to_reference requires both fixed and moving inputs.",
                    details={"fixed": fixed, "moving": moving},
                )
            )
        elif not (_path_exists(fixed) and _path_exists(moving)):
            violations.append(
                RuleViolation(
                    rule_id="register_paths_missing",
                    level="hard",
                    tool_name=tool_name,
                    message="register_to_reference inputs must exist on disk.",
                    details={"fixed": fixed, "moving": moving},
                )
            )
        if fixed and moving and fixed == moving:
            violations.append(
                RuleViolation(
                    rule_id="register_not_t2w_to_t2w",
                    level="hard",
                    tool_name=tool_name,
                    message="Registration fixed and moving are identical; avoid T2w→T2w registration.",
                    details={"fixed": fixed, "moving": moving},
                )
            )
        if "t2" in fixed.lower() and "t2" in moving.lower() and fixed != moving:
            violations.append(
                RuleViolation(
                    rule_id="register_not_t2w_to_t2w_soft",
                    level="soft",
                    tool_name=tool_name,
                    message="Registration appears to be T2w→T2w; verify moving is non-T2w (ADC/DWI).",
                    details={"fixed": fixed, "moving": moving},
                )
            )

    if tool_name == "identify_sequences":
        dicom_case_dir = str(args.get("dicom_case_dir") or "")
        series_inventory_path = str(args.get("series_inventory_path") or "")
        if not dicom_case_dir and not series_inventory_path:
            violations.append(
                RuleViolation(
                    rule_id="identify_requires_input",
                    level="hard",
                    tool_name=tool_name,
                    message="identify_sequences requires dicom_case_dir or series_inventory_path.",
                    details={},
                )
            )
        if dicom_case_dir and not _path_exists(dicom_case_dir):
            violations.append(
                RuleViolation(
                    rule_id="identify_dicom_dir_missing",
                    level="hard",
                    tool_name=tool_name,
                    message="dicom_case_dir does not exist.",
                    details={"dicom_case_dir": dicom_case_dir},
                )
            )
        if series_inventory_path and not _path_exists(series_inventory_path):
            violations.append(
                RuleViolation(
                    rule_id="identify_inventory_missing",
                    level="hard",
                    tool_name=tool_name,
                    message="series_inventory_path does not exist.",
                    details={"series_inventory_path": series_inventory_path},
                )
            )

    if tool_name == "segment_prostate":
        t2w_ref = str(args.get("t2w_ref") or "")
        if not t2w_ref:
            violations.append(
                RuleViolation(
                    rule_id="segment_prostate_requires_t2w",
                    level="hard",
                    tool_name=tool_name,
                    message="segment_prostate requires t2w_ref.",
                    details={},
                )
            )
        elif not _path_exists(t2w_ref):
            violations.append(
                RuleViolation(
                    rule_id="segment_prostate_t2w_missing",
                    level="hard",
                    tool_name=tool_name,
                    message="t2w_ref does not exist on disk.",
                    details={"t2w_ref": t2w_ref},
                )
            )
        elif _looks_like(t2w_ref, ["adc", "dwi", "trace", "t1"]):
            violations.append(
                RuleViolation(
                    rule_id="segment_prostate_t2w_suspect",
                    level="soft",
                    tool_name=tool_name,
                    message="t2w_ref looks like a non-T2w sequence; verify the input.",
                    details={"t2w_ref": t2w_ref},
                )
            )

    if tool_name == "segment_cardiac_cine":
        cine_path = str(args.get("cine_path") or "")
        if not cine_path:
            violations.append(
                RuleViolation(
                    rule_id="segment_cardiac_requires_cine",
                    level="hard",
                    tool_name=tool_name,
                    message="segment_cardiac_cine requires cine_path.",
                    details={},
                )
            )
        elif not _path_exists(cine_path):
            violations.append(
                RuleViolation(
                    rule_id="segment_cardiac_cine_missing",
                    level="hard",
                    tool_name=tool_name,
                    message="cine_path does not exist on disk.",
                    details={"cine_path": cine_path},
                )
            )

    if tool_name == "classify_cardiac_cine_disease":
        seg_path = str(args.get("seg_path") or "")
        info_path = str(args.get("patient_info_path") or "")
        case_state = _read_case_state(case_state_path)
        have_seg = _has_success(case_state, "segment_cardiac_cine")
        if not seg_path and not have_seg:
            violations.append(
                RuleViolation(
                    rule_id="classify_cardiac_requires_segmentation",
                    level="hard",
                    tool_name=tool_name,
                    message="classify_cardiac_cine_disease requires seg_path or a successful prior segment_cardiac_cine run.",
                    details={},
                )
            )
        if seg_path and not _path_exists(seg_path):
            violations.append(
                RuleViolation(
                    rule_id="classify_cardiac_seg_missing",
                    level="hard",
                    tool_name=tool_name,
                    message="seg_path does not exist on disk.",
                    details={"seg_path": seg_path},
                )
            )
        if info_path and not _path_exists(info_path):
            violations.append(
                RuleViolation(
                    rule_id="classify_cardiac_info_missing",
                    level="hard",
                    tool_name=tool_name,
                    message="patient_info_path does not exist on disk.",
                    details={"patient_info_path": info_path},
                )
            )

    if tool_name == "brats_mri_segmentation":
        t1c = str(args.get("t1c_path") or "")
        t1 = str(args.get("t1_path") or "")
        t2 = str(args.get("t2_path") or "")
        flair = str(args.get("flair_path") or "")
        missing = [k for k, v in {"t1c_path": t1c, "t1_path": t1, "t2_path": t2, "flair_path": flair}.items() if not v]
        if missing:
            violations.append(
                RuleViolation(
                    rule_id="brats_requires_four_channels",
                    level="hard",
                    tool_name=tool_name,
                    message="brats_mri_segmentation requires t1c, t1, t2, flair inputs.",
                    details={"missing": missing},
                )
            )
        missing_files = [k for k, v in {"t1c_path": t1c, "t1_path": t1, "t2_path": t2, "flair_path": flair}.items() if v and not _path_exists(v)]
        if missing_files:
            violations.append(
                RuleViolation(
                    rule_id="brats_paths_missing",
                    level="hard",
                    tool_name=tool_name,
                    message="brats_mri_segmentation inputs must exist on disk.",
                    details={"missing": missing_files},
                )
            )
        if len({t1c, t1, t2, flair}) < 4:
            violations.append(
                RuleViolation(
                    rule_id="brats_inputs_duplicate",
                    level="soft",
                    tool_name=tool_name,
                    message="brats_mri_segmentation inputs appear duplicated; verify modality mapping.",
                    details={"t1c_path": t1c, "t1_path": t1, "t2_path": t2, "flair_path": flair},
                )
            )
        if t1c and _looks_like(t1c, ["flair", "t2"]) or t2 and _looks_like(t2, ["t1"]) or flair and _looks_like(flair, ["t1", "t1c"]):
            violations.append(
                RuleViolation(
                    rule_id="brats_modalities_suspect",
                    level="soft",
                    tool_name=tool_name,
                    message="One or more BRATS inputs look mislabeled (e.g., T1/T2/FLAIR).",
                    details={"t1c_path": t1c, "t1_path": t1, "t2_path": t2, "flair_path": flair},
                )
            )

    if tool_name == "extract_roi_features":
        images = args.get("images") or []
        if not isinstance(images, list) or not images:
            violations.append(
                RuleViolation(
                    rule_id="features_requires_images",
                    level="hard",
                    tool_name=tool_name,
                    message="extract_roi_features requires a non-empty images list.",
                    details={},
                )
            )
        else:
            missing_images = []
            for img in images:
                if isinstance(img, dict):
                    path = str(img.get("path") or "")
                else:
                    path = str(img or "")
                if not path or not _path_exists(path):
                    missing_images.append(path or "<missing>")
            if missing_images:
                violations.append(
                    RuleViolation(
                        rule_id="features_image_missing",
                        level="hard",
                        tool_name=tool_name,
                        message="One or more image paths are missing for extract_roi_features.",
                        details={"missing": missing_images},
                    )
                )

        roi_mask_path = str(args.get("roi_mask_path") or "")
        roi_masks = args.get("roi_masks") or []
        if not roi_mask_path and not roi_masks:
            violations.append(
                RuleViolation(
                    rule_id="features_requires_roi",
                    level="hard",
                    tool_name=tool_name,
                    message="extract_roi_features requires roi_mask_path or roi_masks.",
                    details={},
                )
            )
        if roi_mask_path and not _path_exists(roi_mask_path):
            violations.append(
                RuleViolation(
                    rule_id="features_roi_missing",
                    level="hard",
                    tool_name=tool_name,
                    message="roi_mask_path does not exist.",
                    details={"roi_mask_path": roi_mask_path},
                )
            )
        if isinstance(roi_masks, list) and roi_masks:
            missing_rois = []
            for r in roi_masks:
                if isinstance(r, dict):
                    path = str(r.get("path") or "")
                else:
                    path = str(r or "")
                if not path or not _path_exists(path):
                    missing_rois.append(path or "<missing>")
            if missing_rois:
                violations.append(
                    RuleViolation(
                        rule_id="features_roi_masks_missing",
                        level="hard",
                        tool_name=tool_name,
                        message="One or more roi_masks paths are missing.",
                        details={"missing": missing_rois},
                    )
                )

    if tool_name == "detect_lesion_candidates":
        case_state = _read_case_state(case_state_path)
        have_features = _has_success(case_state, "extract_roi_features")
        have_adc = _has_adc_registration(case_state_path)
        have_dwi = _has_dwi_registration(case_state_path)
        have_mask = _artifact_glob(
            case_state_path,
            [
                "**/prostate_whole_gland_mask.nii.gz",
                "**/*_zones.nii.gz",
            ],
        )
        missing = []
        if not have_adc:
            missing.append("registered_adc")
        if not have_dwi:
            missing.append("registered_dwi")
        if not have_mask:
            missing.append("whole_gland_mask")
        if not have_features:
            missing.append("roi_features")
        if missing:
            violations.append(
                RuleViolation(
                    rule_id="lesion_candidates_requires_prereqs",
                    level="hard",
                    tool_name=tool_name,
                    message="Lesion detection missing prerequisites: " + ", ".join(missing),
                    details={"missing": missing},
                )
            )

    if tool_name == "correct_prostate_distortion":
        for key in ("t2w_nifti", "adc_nifti", "highb_nifti"):
            v = str(args.get(key) or "").strip()
            if not v:
                violations.append(
                    RuleViolation(
                        rule_id=f"distortion_{key}_required",
                        level="hard",
                        tool_name=tool_name,
                        message=f"{key} is required.",
                        details={key: v},
                    )
                )
                continue
            if not _path_exists(v):
                violations.append(
                    RuleViolation(
                        rule_id=f"distortion_{key}_missing",
                        level="hard",
                        tool_name=tool_name,
                        message=f"{key} does not exist on disk.",
                        details={key: v},
                    )
                )

    if tool_name == "package_vlm_evidence":
        # Soft reminder: ensure case_state_path provided for deterministic packaging
        if not args.get("case_state_path"):
            violations.append(
                RuleViolation(
                    rule_id="package_requires_case_state",
                    level="soft",
                    tool_name=tool_name,
                    message="package_vlm_evidence should include case_state_path for deterministic packaging.",
                    details={},
                )
            )

    if tool_name == "generate_report":
        case_state_path_arg = str(args.get("case_state_path") or "")
        if not case_state_path_arg:
            violations.append(
                RuleViolation(
                    rule_id="report_requires_case_state",
                    level="hard",
                    tool_name=tool_name,
                    message="generate_report requires case_state_path.",
                    details={},
                )
            )
        elif not _path_exists(case_state_path_arg):
            violations.append(
                RuleViolation(
                    rule_id="report_case_state_missing",
                    level="hard",
                    tool_name=tool_name,
                    message="case_state_path does not exist.",
                    details={"case_state_path": case_state_path_arg},
                )
            )

    return violations

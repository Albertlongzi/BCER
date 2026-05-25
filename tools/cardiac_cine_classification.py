"""
Tool: classify_cardiac_cine_disease

Purpose:
- Rule-based disease classification for cardiac cine MRI using ACDC-style labels.
- Computes core physiological metrics from cardiac segmentation:
  LV/RV EDV/ESV/EF, LV mass, max myocardium thickness, and a local-contraction proxy.
- Applies class rules to predict one of:
  NOR, MINF, DCM, HCM, RV, UNCLASSIFIED.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec

logger = logging.getLogger(__name__)

CARDIAC_CLASSIFY_SPEC = ToolSpec(
    name="classify_cardiac_cine_disease",
    description=(
        "Classify cardiac cine disease group (NOR/MINF/DCM/HCM/RV) from segmentation-derived physiological metrics "
        "using ACDC-style rule logic and ambiguity handling."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "seg_path": {"type": "string"},
            "seg_dir": {
                "type": "string",
                "description": (
                    "Directory containing per-frame segmentation NIfTI files "
                    "(e.g. *_f01.nii.gz … *_f11.nii.gz).  When provided, all "
                    "frames are loaded and stacked for multi-frame ED/ES "
                    "detection by LV-volume extrema.  Takes priority over a "
                    "single seg_path when the directory has more than one file."
                ),
            },
            "ed_seg_path": {"type": "string"},
            "es_seg_path": {"type": "string"},
            "cine_path": {"type": "string"},
            "patient_info_path": {"type": "string"},
            "ed_frame": {"type": "integer"},
            "es_frame": {"type": "integer"},
            "output_subdir": {"type": "string"},
        },
    },
    output_schema={
        "type": "object",
        "properties": {
            "classification_path": {"type": "string"},
            "predicted_group": {"type": "string"},
            "metrics": {"type": "object"},
            "phase_indices": {"type": "object"},
            "rule_trace": {"type": "array"},
        },
        "required": ["classification_path", "predicted_group"],
    },
    version="0.1.0",
    tags=["cardiac", "cine", "classification", "acdc"],
)


def _require_sitk_np():
    try:
        import SimpleITK as sitk  # type: ignore
    except Exception as e:
        raise RuntimeError("SimpleITK is required: pip install SimpleITK") from e
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("numpy is required: pip install numpy") from e
    return sitk, np


def _try_import_distance_transform():
    try:
        from scipy.ndimage import distance_transform_edt  # type: ignore

        return distance_transform_edt
    except Exception:
        return None


def _parse_info_cfg(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not path.exists():
        return out
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            key = k.strip()
            val = v.strip()
            if not key:
                continue
            out[key] = val
    except Exception:
        return {}
    return out


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def _find_info_cfg(
    *,
    seg_path: Path,
    cine_path: Optional[Path],
    user_info_path: Optional[str],
) -> Optional[Path]:
    if user_info_path:
        p = Path(user_info_path).expanduser().resolve()
        if p.exists():
            return p

    search_roots: List[Path] = []
    for p in (cine_path, seg_path):
        if p is None:
            continue
        q = p.parent if p.is_file() else p
        search_roots.append(q)

    for root in search_roots:
        cur = root
        for _ in range(6):
            cfg = cur / "Info.cfg"
            if cfg.exists():
                return cfg
            if cur.parent == cur:
                break
            cur = cur.parent

    patient_id = None
    m = re.search(r"(patient\d{3})", str(seg_path), flags=re.I)
    if not m and cine_path is not None:
        m = re.search(r"(patient\d{3})", str(cine_path), flags=re.I)
    if m:
        patient_id = m.group(1)

    roots = []
    env_root = os.getenv("MRI_AGENT_ACDC_RAW_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser().resolve())

    if patient_id:
        for r in roots:
            for split in ("training", "testing"):
                cfg = r / split / patient_id / "Info.cfg"
                if cfg.exists():
                    return cfg
    return None


def _as_time_zyx(arr: Any, np_mod: Any) -> Any:
    if arr.ndim == 3:
        return arr[np_mod.newaxis, ...]
    if arr.ndim != 4:
        raise RuntimeError(f"Unsupported segmentation rank={arr.ndim}; expected 3D or 4D.")
    return arr


def _frame_volume_ml(frame_zyx: Any, label: int, voxel_ml: float, np_mod: Any) -> float:
    return float(np_mod.count_nonzero(frame_zyx == int(label)) * voxel_ml)


def _safe_ef(edv: float, esv: float) -> Optional[float]:
    if edv <= 1e-6:
        return None
    return float((edv - esv) / edv * 100.0)


def _calc_bsa_m2(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    if height_cm is None or weight_kg is None:
        return None
    if height_cm <= 0.0 or weight_kg <= 0.0:
        return None
    return float(math.sqrt((height_cm * weight_kg) / 3600.0))


def _pick_ed_es_indices(
    *,
    lv_volumes: List[float],
    t_len: int,
    ed_frame: Optional[int],
    es_frame: Optional[int],
    info_cfg: Dict[str, Any],
) -> Tuple[int, int, str]:
    if t_len <= 1:
        return 0, 0, "single_frame"

    if ed_frame is not None and es_frame is not None:
        ed_idx = max(0, min(t_len - 1, int(ed_frame) - 1))
        es_idx = max(0, min(t_len - 1, int(es_frame) - 1))
        return ed_idx, es_idx, "user_provided"

    info_ed = _to_int(info_cfg.get("ED"))
    info_es = _to_int(info_cfg.get("ES"))
    if info_ed is not None and info_es is not None:
        ed_idx = max(0, min(t_len - 1, int(info_ed) - 1))
        es_idx = max(0, min(t_len - 1, int(info_es) - 1))
        return ed_idx, es_idx, "info_cfg"

    if not lv_volumes:
        return 0, 0, "fallback_zero"

    ed_idx = int(max(range(len(lv_volumes)), key=lambda i: lv_volumes[i]))
    es_idx = int(min(range(len(lv_volumes)), key=lambda i: lv_volumes[i]))
    return ed_idx, es_idx, "lv_volume_extrema"


def _resolve_seg_paths(args: Dict[str, Any]) -> Tuple[Path, Optional[Path], Optional[Path]]:
    seg_path_raw = args.get("seg_path")
    if not seg_path_raw:
        raise RuntimeError("seg_path is required (or auto-filled from segment_cardiac_cine).")
    seg_path = Path(str(seg_path_raw)).expanduser().resolve()
    if not seg_path.exists():
        raise FileNotFoundError(f"seg_path not found: {seg_path}")

    ed_raw = str(args.get("ed_seg_path") or "").strip()
    es_raw = str(args.get("es_seg_path") or "").strip()
    ed_path = Path(ed_raw).expanduser().resolve() if ed_raw else None
    es_path = Path(es_raw).expanduser().resolve() if es_raw else None
    if ed_path is not None and (not ed_path.exists()):
        ed_path = None
    if es_path is not None and (not es_path.exists()):
        es_path = None

    # If explicit pair not provided, try to infer from *_ed / *_es naming.
    if ed_path is None and es_path is None:
        stem = seg_path.name
        if "_ed" in stem:
            guess_es = seg_path.parent / stem.replace("_ed", "_es", 1)
            if guess_es.exists():
                ed_path = seg_path
                es_path = guess_es
        elif "_es" in stem:
            guess_ed = seg_path.parent / stem.replace("_es", "_ed", 1)
            if guess_ed.exists():
                ed_path = guess_ed
                es_path = seg_path

    if ed_path is not None and es_path is not None:
        return seg_path, ed_path, es_path
    return seg_path, None, None


def _max_myo_thickness_mm(
    myo_zyx: Any,
    spacing_zyx_mm: Tuple[float, float, float],
    np_mod: Any,
) -> Optional[float]:
    dt = _try_import_distance_transform()
    if dt is None:
        return None

    sy = float(spacing_zyx_mm[1])
    sx = float(spacing_zyx_mm[2])
    best = 0.0
    for z in range(int(myo_zyx.shape[0])):
        sl = myo_zyx[z] > 0
        if int(np_mod.count_nonzero(sl)) < 8:
            continue
        dist = dt(sl, sampling=(sy, sx))
        cur = float(dist.max() * 2.0)
        if cur > best:
            best = cur
    if best <= 0.0:
        return None
    return best


def _local_contraction_proxy(
    myo_ed_zyx: Any,
    myo_es_zyx: Any,
    np_mod: Any,
) -> Dict[str, Any]:
    n_slices = int(myo_ed_zyx.shape[0])
    ratios: List[float] = []
    abnormal = 0
    for z in range(n_slices):
        a_ed = int(np_mod.count_nonzero(myo_ed_zyx[z] > 0))
        a_es = int(np_mod.count_nonzero(myo_es_zyx[z] > 0))
        if a_ed <= 0:
            continue
        ratio = float((a_ed - a_es) / float(a_ed))
        ratios.append(ratio)
        if ratio < 0.10:
            abnormal += 1
    valid = len(ratios)
    several_limit = max(3, int(round(0.40 * valid))) if valid > 0 else 0
    return {
        "valid_slices": int(valid),
        "abnormal_slices": int(abnormal),
        "abnormal_ratio_mean": (float(sum(ratios) / len(ratios)) if ratios else None),
        "abnormal_slices_threshold_for_several": int(several_limit),
        "is_several_segments_abnormal": bool(valid > 0 and 0 < abnormal <= several_limit),
    }


def _flag_high_volume(raw: Optional[float], indexed: Optional[float], *, idx_thr: float, abs_thr: float) -> bool:
    if indexed is not None and indexed > idx_thr:
        return True
    if raw is not None and raw > abs_thr:
        return True
    return False


def _classify_from_metrics(metrics: Dict[str, Any]) -> Tuple[str, List[str], bool]:
    trace: List[str] = []
    lvef = _to_float(metrics.get("lv_ef_percent"))
    rvef = _to_float(metrics.get("rv_ef_percent"))
    lvedv = _to_float(metrics.get("lv_edv_ml"))
    rvedv = _to_float(metrics.get("rv_edv_ml"))
    lvedvi = _to_float(metrics.get("lv_edvi_ml_m2"))
    rvedvi = _to_float(metrics.get("rv_edvi_ml_m2"))
    max_thick = _to_float(metrics.get("max_myo_thickness_mm"))
    lv_mass_index = _to_float(metrics.get("lv_mass_index_g_m2"))
    cproxy = metrics.get("local_contraction_proxy") if isinstance(metrics.get("local_contraction_proxy"), dict) else {}
    several_abn = bool(cproxy.get("is_several_segments_abnormal"))
    borderline = False

    high_lvedv = _flag_high_volume(lvedv, lvedvi, idx_thr=100.0, abs_thr=220.0)
    high_rvedv = _flag_high_volume(rvedv, rvedvi, idx_thr=110.0, abs_thr=250.0)
    low_lvef = (lvef is not None and lvef < 40.0)
    high_lvef = (lvef is not None and lvef > 55.0)
    rvef_abnormal = (rvef is not None and rvef < 40.0)
    if rvef is not None and 40.0 <= rvef < 45.0:
        borderline = True
        trace.append("Borderline RV EF in [40,45): do not force RV class.")

    thick_increase = False
    if max_thick is not None and max_thick >= 15.0:
        thick_increase = True
        trace.append("Detected increased myocardium thickness (max >= 15 mm).")
    if lv_mass_index is not None and lv_mass_index >= 110.0:
        thick_increase = True
        trace.append("Detected increased LV mass index (>= 110 g/m^2).")

    # ACDC ambiguity rule 1:
    # HCM usually has LVEF > 55, but if LVEF < 40 with thickness increase -> MINF.
    if low_lvef and thick_increase:
        trace.append("Rule: LVEF < 40 with local thickness increase -> MINF.")
        return "MINF", trace, borderline

    # ACDC ambiguity rule 2:
    # High LV volume + low LVEF + only several abnormal-contraction segments -> MINF.
    if high_lvedv and low_lvef and several_abn:
        trace.append("Rule: high LVEDV + low LVEF + only several abnormal segments -> MINF.")
        return "MINF", trace, borderline

    # ACDC ambiguity rule 3:
    # Dilated LV and RV (with/without RV dysfunction) -> DCM.
    if high_lvedv and high_rvedv:
        trace.append("Rule: both LV and RV dilated -> DCM.")
        return "DCM", trace, borderline

    # Practical fallback consistent with ACDC class intent:
    # markedly dilated LV with reduced LVEF, without MINF pattern -> DCM.
    if high_lvedv and low_lvef and (not thick_increase) and (not several_abn):
        trace.append("Rule: high LVEDV + low LVEF without MINF pattern -> DCM.")
        return "DCM", trace, borderline

    if high_lvef and thick_increase:
        trace.append("Rule: LVEF > 55 with hypertrophic pattern -> HCM.")
        return "HCM", trace, borderline

    if rvef_abnormal and (not high_lvedv):
        trace.append("Rule: isolated abnormal RV function (RVEF < 40) -> RV.")
        return "RV", trace, borderline

    normal_like = (
        (lvef is not None and lvef >= 55.0)
        and (rvef is not None and rvef >= 45.0)
        and (not high_lvedv)
        and (not high_rvedv)
        and (not thick_increase)
    )
    if normal_like:
        trace.append("Rule: preserved biventricular EF, no dilation, no hypertrophy -> NOR.")
        return "NOR", trace, borderline

    trace.append("No definitive class rule matched -> UNCLASSIFIED.")
    return "UNCLASSIFIED", trace, True


def classify_cardiac_cine_disease(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    sitk, np = _require_sitk_np()

    seg_path, ed_seg_path, es_seg_path = _resolve_seg_paths(args)

    cine_path = None
    cine_raw = args.get("cine_path")
    if cine_raw:
        cine_path = Path(str(cine_raw)).expanduser().resolve()

    info_cfg_path = _find_info_cfg(
        seg_path=seg_path,
        cine_path=cine_path,
        user_info_path=(str(args.get("patient_info_path")) if args.get("patient_info_path") else None),
    )
    info_cfg = _parse_info_cfg(info_cfg_path) if info_cfg_path else {}

    if ed_seg_path is not None and es_seg_path is not None:
        ed_img = sitk.ReadImage(str(ed_seg_path))
        es_img = sitk.ReadImage(str(es_seg_path))
        spacing = tuple(float(x) for x in ed_img.GetSpacing())
        spacing_xyz = spacing[:3] if len(spacing) >= 3 else (1.0, 1.0, 1.0)
        spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
        voxel_ml = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2] / 1000.0)

        ed_arr = np.rint(sitk.GetArrayFromImage(ed_img)).astype("int16")
        es_arr = np.rint(sitk.GetArrayFromImage(es_img)).astype("int16")
        lv_edv = _frame_volume_ml(ed_arr, 3, voxel_ml, np)
        lv_esv = _frame_volume_ml(es_arr, 3, voxel_ml, np)
        rv_edv = _frame_volume_ml(ed_arr, 1, voxel_ml, np)
        rv_esv = _frame_volume_ml(es_arr, 1, voxel_ml, np)
        myo_edv = _frame_volume_ml(ed_arr, 2, voxel_ml, np)

        myo_ed = (ed_arr == 2).astype("uint8")
        myo_es = (es_arr == 2).astype("uint8")
        max_thick = _max_myo_thickness_mm(myo_ed, spacing_zyx, np)
        cproxy = _local_contraction_proxy(myo_ed, myo_es, np)

        ed_frame_cfg = _to_int(info_cfg.get("ED"))
        es_frame_cfg = _to_int(info_cfg.get("ES"))
        ed_idx = int((ed_frame_cfg - 1) if isinstance(ed_frame_cfg, int) and ed_frame_cfg >= 1 else 0)
        es_idx = int((es_frame_cfg - 1) if isinstance(es_frame_cfg, int) and es_frame_cfg >= 1 else 1)
        phase_source = "paired_seg_paths"
    else:
        # ---- Multi-frame loading from seg_dir (preferred) ----
        seg_dir_raw = str(args.get("seg_dir") or "").strip()
        seg_dir_path = Path(seg_dir_raw).expanduser().resolve() if seg_dir_raw else None
        frame_files: List[Path] = []
        if seg_dir_path is not None and seg_dir_path.is_dir():
            frame_files = sorted(seg_dir_path.glob("*.nii.gz"))

        if len(frame_files) > 1:
            # Stack per-frame 3-D seg files into (t, z, y, x)
            first_img = sitk.ReadImage(str(frame_files[0]))
            spacing = tuple(float(x) for x in first_img.GetSpacing())
            spacing_xyz = spacing[:3] if len(spacing) >= 3 else (1.0, 1.0, 1.0)
            spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
            voxel_ml = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2] / 1000.0)
            frame_arrays = []
            for fp in frame_files:
                _img = sitk.ReadImage(str(fp))
                _arr = np.rint(sitk.GetArrayFromImage(_img)).astype("int16")
                frame_arrays.append(_arr)
            seg_tzyx = np.stack(frame_arrays, axis=0)  # (t, z, y, x)
            t_len = int(seg_tzyx.shape[0])
            logger.info(
                "Loaded %d per-frame seg files from %s → seg_tzyx shape %s",
                t_len, seg_dir_path, seg_tzyx.shape,
            )
        else:
            # Fallback: single seg_path file
            seg_img = sitk.ReadImage(str(seg_path))
            spacing = tuple(float(x) for x in seg_img.GetSpacing())
            spacing_xyz = spacing[:3] if len(spacing) >= 3 else (1.0, 1.0, 1.0)
            spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
            voxel_ml = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2] / 1000.0)
            seg_arr = sitk.GetArrayFromImage(seg_img)
            seg_arr = np.rint(seg_arr).astype("int16")
            seg_tzyx = _as_time_zyx(seg_arr, np)
            t_len = int(seg_tzyx.shape[0])

        lv_vols: List[float] = []
        rv_vols: List[float] = []
        myo_vols: List[float] = []
        for t in range(t_len):
            fr = seg_tzyx[t]
            rv_vols.append(_frame_volume_ml(fr, 1, voxel_ml, np))
            myo_vols.append(_frame_volume_ml(fr, 2, voxel_ml, np))
            lv_vols.append(_frame_volume_ml(fr, 3, voxel_ml, np))

        ed_idx, es_idx, phase_source = _pick_ed_es_indices(
            lv_volumes=lv_vols,
            t_len=t_len,
            ed_frame=_to_int(args.get("ed_frame")),
            es_frame=_to_int(args.get("es_frame")),
            info_cfg=info_cfg,
        )

        lv_edv = float(lv_vols[ed_idx]) if lv_vols else 0.0
        lv_esv = float(lv_vols[es_idx]) if lv_vols else 0.0
        rv_edv = float(rv_vols[ed_idx]) if rv_vols else 0.0
        rv_esv = float(rv_vols[es_idx]) if rv_vols else 0.0
        myo_edv = float(myo_vols[ed_idx]) if myo_vols else 0.0

        myo_ed = (seg_tzyx[ed_idx] == 2).astype("uint8")
        myo_es = (seg_tzyx[es_idx] == 2).astype("uint8")
        max_thick = _max_myo_thickness_mm(myo_ed, spacing_zyx, np)
        cproxy = _local_contraction_proxy(myo_ed, myo_es, np)

    lv_ef = _safe_ef(lv_edv, lv_esv)
    rv_ef = _safe_ef(rv_edv, rv_esv)
    lv_mass_g = float(myo_edv * 1.05)  # myocardial density ~1.05 g/ml

    height_cm = _to_float(info_cfg.get("Height"))
    weight_kg = _to_float(info_cfg.get("Weight"))
    bsa_m2 = _calc_bsa_m2(height_cm, weight_kg)

    lv_edvi = (float(lv_edv / bsa_m2) if bsa_m2 else None)
    rv_edvi = (float(rv_edv / bsa_m2) if bsa_m2 else None)
    lv_mass_i = (float(lv_mass_g / bsa_m2) if bsa_m2 else None)

    metrics: Dict[str, Any] = {
        "voxel_volume_ml": voxel_ml,
        "lv_edv_ml": lv_edv,
        "lv_esv_ml": lv_esv,
        "lv_ef_percent": lv_ef,
        "rv_edv_ml": rv_edv,
        "rv_esv_ml": rv_esv,
        "rv_ef_percent": rv_ef,
        "myo_ed_volume_ml": myo_edv,
        "lv_mass_g": lv_mass_g,
        "max_myo_thickness_mm": max_thick,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "bsa_m2": bsa_m2,
        "lv_edvi_ml_m2": lv_edvi,
        "rv_edvi_ml_m2": rv_edvi,
        "lv_mass_index_g_m2": lv_mass_i,
        "local_contraction_proxy": cproxy,
    }

    predicted, rule_trace, borderline = _classify_from_metrics(metrics)
    gt_group = str(info_cfg.get("Group") or "").strip() or None
    gt_match = (bool(gt_group) and gt_group == predicted)

    output_subdir = str(args.get("output_subdir") or "classification/cardiac_cine")
    out_dir = ctx.artifacts_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cardiac_cine_classification.json"

    result_payload = {
        "predicted_group": predicted,
        "ground_truth_group": gt_group,
        "ground_truth_match": bool(gt_match),
        "needs_vlm_review": bool(predicted == "UNCLASSIFIED" or borderline),
        "phase_indices": {
            "ed_index_0based": int(ed_idx),
            "es_index_0based": int(es_idx),
            "ed_frame_1based": int(ed_idx + 1),
            "es_frame_1based": int(es_idx + 1),
            "source": phase_source,
        },
        "metrics": metrics,
        "rule_trace": rule_trace,
        "label_map_note": "ACDC label convention used: 1=RV, 2=MYO, 3=LV.",
    }
    out_path.write_text(json.dumps(result_payload, indent=2) + "\n", encoding="utf-8")

    artifacts = [
        ArtifactRef(path=str(out_path), kind="json", description="Cardiac cine disease classification"),
    ]
    source_artifacts = [
        ArtifactRef(path=str(seg_path), kind="nifti", description="Input cardiac segmentation"),
    ]
    if ed_seg_path is not None:
        source_artifacts.append(ArtifactRef(path=str(ed_seg_path), kind="nifti", description="ED segmentation"))
    if es_seg_path is not None:
        source_artifacts.append(ArtifactRef(path=str(es_seg_path), kind="nifti", description="ES segmentation"))
    if info_cfg_path is not None and info_cfg_path.exists():
        source_artifacts.append(ArtifactRef(path=str(info_cfg_path), kind="text", description="Patient Info.cfg"))

    return {
        "data": {
            "classification_path": str(out_path),
            "predicted_group": predicted,
            "ground_truth_group": gt_group,
            "ground_truth_match": bool(gt_match),
            "needs_vlm_review": bool(predicted == "UNCLASSIFIED" or borderline),
            "metrics": metrics,
            "phase_indices": result_payload["phase_indices"],
            "rule_trace": rule_trace,
        },
        "artifacts": artifacts,
        "source_artifacts": source_artifacts,
        "generated_artifacts": artifacts,
        "warnings": (
            ["Classification is UNCLASSIFIED/BORDERLINE; consider VLM-assisted qualitative review."]
            if (predicted == "UNCLASSIFIED" or borderline)
            else []
        ),
    }


def build_tool() -> Tool:
    return Tool(spec=CARDIAC_CLASSIFY_SPEC, func=classify_cardiac_cine_disease)

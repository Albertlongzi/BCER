"""
Tool: package_vlm_evidence

Purpose:
- Create a standardized `vlm_evidence_bundle.json` for the VLM from existing pipeline artifacts.
- This tool is intentionally "packaging only": it does not write a narrative report.

Bundle contents (high-signal, reproducible evidence):
- mapping (T2w/ADC/DWI)
- features.csv + preview rows
- slice_summary.json + preview
- overlay PNGs + tiles index JSONs (if present)
- QC PNGs (registration central slices, overlays)
- pipeline status snapshot
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec
from runtime.path_utils import list_with_short_paths, to_short_path
from runtime.artifact_index import build_artifact_index


PKG_SPEC = ToolSpec(
    name="package_vlm_evidence",
    description=(
        "Package a reproducible `vlm_evidence_bundle.json` from `case_state.json` + existing artifacts. "
        "Includes mapping, features previews, slice summaries, selected QC/overlay PNGs, and lesion-candidate summaries (if present). "
        "Does NOT generate narrative text."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "case_state_path": {"type": "string"},
            "output_subdir": {"type": "string"},
        },
        "required": ["case_state_path"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "vlm_evidence_path": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["status", "vlm_evidence_path", "summary"],
    },
    version="0.1.0",
    tags=["vlm", "evidence", "packaging"],
)


def _gather_pngs(run_dir: Path) -> List[str]:
    out: List[str] = []
    for p in sorted((run_dir / "artifacts").glob("**/*.png")):
        s = str(p)
        # Keep the set small-ish; prioritize ROIs + central registration PNGs.
        if "/overlays/" in s or s.endswith("_central.png"):
            out.append(s)
    return out[:24]


def _read_features_csv_preview(path: Optional[str], max_rows: int = 64) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= max_rows:
                break
            out: Dict[str, Any] = {}
            for k, v in (r or {}).items():
                if str(k).startswith("rad_"):
                    continue
                if v is None:
                    out[k] = None
                    continue
                vv = str(v)
                try:
                    if k in {"roi", "sequence"}:
                        out[k] = vv
                    else:
                        out[k] = float(vv) if (("." in vv) or vv.isdigit()) else vv
                except Exception:
                    out[k] = vv
            rows.append(out)
    return rows


def _find_feature_md_path(feat_path: Optional[str]) -> Optional[str]:
    if not feat_path:
        return None
    p = Path(str(feat_path)).expanduser().resolve()
    if p.suffix.lower() == ".md" and p.exists():
        return str(p)
    if p.suffix.lower() == ".csv":
        md = p.with_suffix(".md")
        if md.exists():
            return str(md)
    cand = p.parent / "features.md"
    if cand.exists():
        return str(cand)
    return None


def _read_slice_summary_preview(path: Optional[str], per_sequence_cap: int = 12) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    per_seq = obj.get("per_sequence") if isinstance(obj, dict) else None
    if not isinstance(per_seq, dict):
        return {}
    out: Dict[str, Any] = {}
    for name, v in per_seq.items():
        if not isinstance(v, dict):
            continue
        per_slice = v.get("per_slice", [])
        if not isinstance(per_slice, list):
            per_slice = []
        out[str(name)] = {
            "n_slices_with_roi": v.get("n_slices_with_roi"),
            "per_slice_preview": per_slice[:per_sequence_cap],
        }
    return out


def _discover_main_feature_path(run_art_dir: Path) -> Optional[str]:
    """Best-effort discovery of the primary (non-lesion) features.csv."""
    cand = run_art_dir / "features" / "features.csv"
    if cand.exists():
        return str(cand)
    try:
        cands: List[Path] = []
        for p in run_art_dir.rglob("features.csv"):
            parts = [str(x) for x in p.parts]
            if "features_lesion" in parts:
                continue
            if p.parent.name != "features":
                continue
            cands.append(p)
        if not cands:
            return None
        # Prefer the shortest path (closest to artifacts/features/features.csv).
        cands.sort(key=lambda x: len(str(x)))
        return str(cands[0])
    except Exception:
        return None


def _feature_set_from_subdir(run_dir: Path, subdir: str) -> Dict[str, Any]:
    """Collect features artifacts from a conventional subdir (features / features_lesion)."""
    base = run_dir / "artifacts" / subdir
    feat = base / "features.csv"
    slc = base / "slice_summary.json"
    rad = base / "radiomics_summary.json"
    overlays_dir = base / "overlays"

    def _infer_overlay_meta(path: str) -> Dict[str, str]:
        low = path.lower()
        kind = ""
        if "full_frame_montage_4x4" in low:
            kind = "full_frame_montage_4x4"
        elif "full_frame_montage_5x5" in low:
            kind = "full_frame_montage_5x5"
        elif "full_frame_montage" in low:
            kind = "full_frame_montage"
        elif "full_frame_best" in low:
            kind = "full_frame_best_slice"
        elif "roi_crop_montage_4x4" in low:
            kind = "roi_crop_montage_4x4"
        elif "roi_crop_montage_5x5" in low:
            kind = "roi_crop_montage_5x5"
        elif "roi_crop_montage" in low:
            kind = "roi_crop_montage"
        elif "roi_crop_best" in low:
            kind = "roi_crop_best_slice"
        elif low.endswith("_tiles.json"):
            if "full_frame_montage" in low:
                kind = "full_frame_montage_4x4_tiles"
            else:
                kind = "roi_crop_montage_4x4_tiles"
        seq = ""
        if "adc" in low:
            seq = "ADC"
        elif "t2" in low:
            seq = "T2w"
        elif "high_b" in low or "high-b" in low:
            seq = "DWI_high_b"
        elif "low_b" in low or "low-b" in low:
            seq = "DWI_low_b"
        elif "dwi" in low:
            seq = "DWI"
        roi_src = ""
        if "lesion" in low:
            roi_src = "lesion"
        elif "prostate" in low:
            roi_src = "prostate"
        meta: Dict[str, str] = {}
        if kind:
            meta["kind"] = kind
        if seq:
            meta["sequence"] = seq
        if roi_src:
            meta["roi_source"] = roi_src
        return meta

    feat_path = str(feat) if feat.exists() else None
    slc_path = str(slc) if slc.exists() else None
    rad_path = str(rad) if rad.exists() else None

    overlay_items: List[Dict[str, Any]] = []
    if overlays_dir.exists():
        for p in sorted(overlays_dir.glob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".png", ".json"}:
                continue
            meta = _infer_overlay_meta(str(p))
            item: Dict[str, Any] = {"path": str(p)}
            item.update(meta)
            overlay_items.append(item)

    return {
        "ok": bool(feat_path),
        "feature_table_path": to_short_path(feat_path, run_dir=run_dir),
        "feature_table_path_abs": feat_path,
        "slice_summary_path": to_short_path(slc_path, run_dir=run_dir),
        "slice_summary_path_abs": slc_path,
        "radiomics_summary_path": to_short_path(rad_path, run_dir=run_dir),
        "radiomics_summary_path_abs": rad_path,
        "overlay_pngs": list_with_short_paths(overlay_items, run_dir=run_dir),
        "features_preview_rows": _read_features_csv_preview(feat_path, max_rows=64),
        "slice_summary_preview": _read_slice_summary_preview(slc_path, per_sequence_cap=12),
    }


def _uniq_nonempty(values: List[Any], *, max_len: int = 12) -> List[Any]:
    out: List[Any] = []
    seen: set[str] = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(v)
        if len(out) >= max_len:
            break
    return out


def _summarize_dicom_meta(dicom_meta_path: Optional[str]) -> Dict[str, Any]:
    if not dicom_meta_path:
        return {}
    p = Path(str(dicom_meta_path)).expanduser().resolve()
    if not p.exists():
        return {"note": "dicom_meta_path_not_found", "dicom_meta_path": str(p)}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"note": "dicom_meta_unreadable", "dicom_meta_path": str(p)}

    series_meta = obj.get("series_meta") if isinstance(obj, dict) else None
    if not isinstance(series_meta, dict):
        return {"note": "dicom_meta_missing_series_meta", "dicom_meta_path": str(p)}

    manufacturers: List[Any] = []
    models: List[Any] = []
    field_strengths: List[Any] = []
    series_desc: List[Any] = []
    protocol_names: List[Any] = []
    sequence_names: List[Any] = []
    modalities: List[Any] = []
    tr_list: List[Any] = []
    te_list: List[Any] = []
    flip_list: List[Any] = []
    slice_thickness: List[Any] = []
    spacing_between: List[Any] = []
    contrast_agents: List[Any] = []
    contrast_volumes: List[Any] = []
    b_values: List[float] = []
    nifti_spacing_z: List[Any] = []
    spacing_sources: List[Any] = []
    spacing_overrides: List[Any] = []
    spacing_mismatch_series: List[Any] = []

    for _name, meta in series_meta.items():
        if not isinstance(meta, dict):
            continue
        header = meta.get("header") if isinstance(meta.get("header"), dict) else {}
        manufacturers.append(header.get("Manufacturer"))
        models.append(header.get("ManufacturerModelName"))
        field_strengths.append(header.get("MagneticFieldStrength"))
        series_desc.append(header.get("SeriesDescription"))
        protocol_names.append(header.get("ProtocolName"))
        sequence_names.append(header.get("SequenceName"))
        modalities.append(header.get("Modality"))
        tr_list.append(header.get("RepetitionTime"))
        te_list.append(header.get("EchoTime"))
        flip_list.append(header.get("FlipAngle"))
        slice_thickness.append(header.get("SliceThickness"))
        spacing_between.append(header.get("SpacingBetweenSlices"))
        contrast_agents.append(header.get("ContrastBolusAgent"))
        contrast_volumes.append(header.get("ContrastBolusVolume"))
        nifti_info = meta.get("nifti_info") if isinstance(meta.get("nifti_info"), dict) else {}
        spacing_after = nifti_info.get("spacing_after")
        if isinstance(spacing_after, list) and len(spacing_after) >= 3:
            nifti_spacing_z.append(spacing_after[2])
        spacing_sources.append(nifti_info.get("spacing_source"))
        if nifti_info.get("spacing_override_applied"):
            spacing_overrides.append(_name)
        if nifti_info.get("spacing_mismatch"):
            spacing_mismatch_series.append(_name)
        diff = meta.get("diffusion") if isinstance(meta.get("diffusion"), dict) else {}
        b_vals = diff.get("b_values")
        if isinstance(b_vals, list):
            for b in b_vals:
                try:
                    b_values.append(float(b))
                except Exception:
                    continue

    def _sorted_unique_floats(vals: List[Any]) -> List[float]:
        out: List[float] = []
        for v in vals:
            try:
                out.append(float(v))
            except Exception:
                continue
        return sorted(set(out))

    return {
        "dicom_meta_path": str(p),
        "manufacturer": _uniq_nonempty(manufacturers),
        "model": _uniq_nonempty(models),
        "field_strength_t": _sorted_unique_floats(field_strengths),
        "series_descriptions": _uniq_nonempty(series_desc, max_len=24),
        "protocol_names": _uniq_nonempty(protocol_names, max_len=24),
        "sequence_names": _uniq_nonempty(sequence_names, max_len=24),
        "modalities": _uniq_nonempty(modalities, max_len=8),
        "tr_ms": _sorted_unique_floats(tr_list),
        "te_ms": _sorted_unique_floats(te_list),
        "flip_angle_deg": _sorted_unique_floats(flip_list),
        "slice_thickness_mm": _sorted_unique_floats(slice_thickness),
        "spacing_between_slices_mm": _sorted_unique_floats(spacing_between),
        "nifti_spacing_z_mm": _sorted_unique_floats(nifti_spacing_z),
        "nifti_spacing_sources": _uniq_nonempty(spacing_sources, max_len=6),
        "spacing_override_series": _uniq_nonempty(spacing_overrides, max_len=12),
        "spacing_mismatch_series": _uniq_nonempty(spacing_mismatch_series, max_len=12),
        "contrast_agent": _uniq_nonempty(contrast_agents, max_len=6),
        "contrast_volume_ml": _sorted_unique_floats(contrast_volumes),
        "diffusion_b_values": sorted(set(b_values)),
    }


def _summarize_series_list(series: Any, *, max_items: int = 32) -> List[Dict[str, Any]]:
    if not isinstance(series, list):
        return []
    out: List[Dict[str, Any]] = []
    for s in series:
        if not isinstance(s, dict):
            continue
        out.append(
            {
                "series_name": s.get("series_name"),
                "SeriesDescription": s.get("SeriesDescription"),
                "sequence_guess": s.get("sequence_guess"),
                "n_dicoms": s.get("n_dicoms"),
                "b_values": s.get("b_values"),
                "dicom_header_txt": s.get("dicom_header_txt"),
            }
        )
        if len(out) >= max_items:
            break
    return out


def _compute_lesion_geometry(
    *,
    lesion_mask_path: Optional[str],
    prostate_mask_path: Optional[str],
    zone_mask_path: Optional[str],
) -> Dict[str, Any]:
    if not lesion_mask_path or not prostate_mask_path:
        return {"ok": False, "note": "missing_lesion_or_prostate_mask"}
    lm = Path(str(lesion_mask_path)).expanduser().resolve()
    pm = Path(str(prostate_mask_path)).expanduser().resolve()
    if not lm.exists() or not pm.exists():
        return {"ok": False, "note": "lesion_or_prostate_mask_missing_on_disk"}
    try:
        import SimpleITK as sitk  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return {"ok": False, "note": "simpleitk_or_numpy_missing"}

    lesion_img = sitk.ReadImage(str(lm))
    prostate_img = sitk.ReadImage(str(pm))

    # Resample lesion mask into prostate grid if needed.
    if lesion_img.GetSize() != prostate_img.GetSize() or lesion_img.GetSpacing() != prostate_img.GetSpacing():
        lesion_img = sitk.Resample(
            lesion_img,
            prostate_img,
            sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
    lesion_img = sitk.Cast(lesion_img > 0, sitk.sitkUInt8)
    prostate_img = sitk.Cast(prostate_img > 0, sitk.sitkUInt8)

    lesion_arr = sitk.GetArrayViewFromImage(lesion_img)  # z,y,x
    if int(np.count_nonzero(lesion_arr)) == 0:
        return {"ok": False, "note": "lesion_mask_empty"}

    prostate_arr = sitk.GetArrayViewFromImage(prostate_img)
    if int(np.count_nonzero(prostate_arr)) == 0:
        return {"ok": False, "note": "prostate_mask_empty"}

    # Prostate bbox in index space (z,y,x).
    zyx = np.argwhere(prostate_arr > 0)
    zmin, ymin, xmin = zyx.min(axis=0).tolist()
    zmax, ymax, xmax = zyx.max(axis=0).tolist()
    center_z = float(zyx[:, 0].mean())
    center_y = float(zyx[:, 1].mean())
    center_x = float(zyx[:, 2].mean())

    # Capsule boundary for contact check.
    eroded = sitk.BinaryErode(prostate_img, [1, 1, 1])
    boundary = sitk.Subtract(prostate_img, eroded)
    boundary_arr = sitk.GetArrayViewFromImage(boundary) > 0

    zone_arr = None
    if zone_mask_path:
        zp = Path(str(zone_mask_path)).expanduser().resolve()
        if zp.exists():
            zone_img = sitk.ReadImage(str(zp))
            if zone_img.GetSize() != prostate_img.GetSize() or zone_img.GetSpacing() != prostate_img.GetSpacing():
                zone_img = sitk.Resample(
                    zone_img,
                    prostate_img,
                    sitk.Transform(3, sitk.sitkIdentity),
                    sitk.sitkNearestNeighbor,
                    0,
                    sitk.sitkUInt8,
                )
            zone_arr = sitk.GetArrayViewFromImage(zone_img)

    cc = sitk.ConnectedComponent(lesion_img)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    labels = list(stats.GetLabels())
    spacing = prostate_img.GetSpacing()  # x,y,z

    lesions: List[Dict[str, Any]] = []
    cc_arr = sitk.GetArrayFromImage(cc)  # z,y,x

    for lbl in labels:
        vox = int(stats.GetNumberOfPixels(lbl))
        if vox <= 0:
            continue
        bbox = stats.GetBoundingBox(lbl)  # x,y,z,sizeX,sizeY,sizeZ
        centroid_phys = stats.GetCentroid(lbl)
        cx, cy, cz = prostate_img.TransformPhysicalPointToIndex(centroid_phys)

        z_frac = None
        if zmax > zmin:
            z_frac = float(cz - zmin) / float(zmax - zmin)

        # Clock-face in physical space (12 o'clock at anterior = -y).
        try:
            center_phys = prostate_img.TransformContinuousIndexToPhysicalPoint((center_x, center_y, center_z))
            lesion_phys = centroid_phys
            dx = float(lesion_phys[0]) - float(center_phys[0])
            dy = float(lesion_phys[1]) - float(center_phys[1])  # +y = posterior
        except Exception:
            # Fallback to image index space if physical transform fails.
            dx = float(cx) - center_x
            dy = float(cy) - center_y

        angle = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0
        clock = int((angle + 15.0) // 30.0)
        clock = 12 if clock == 0 else clock
        # Empirically, current preprocessing/orientation stack maps positive dx to
        # patient-right in our dataset pipeline. Keep this convention consistent
        # with downstream evaluation labels.
        lr = "right" if dx >= 0 else "left"
        ap = "anterior" if dy < 0 else "posterior"

        comp_mask = cc_arr == int(lbl)
        capsule_contact = bool((comp_mask & boundary_arr).any())

        zone_note = None
        zone_frac: Dict[str, float] = {}
        zone_label = None
        if zone_arr is not None:
            zvals = zone_arr[comp_mask]
            if zvals.size > 0:
                for val in (1, 2):
                    zone_frac[str(val)] = float(int((zvals == val).sum())) / float(zvals.size)
                zone_label = "PZ" if zone_frac.get("2", 0.0) >= zone_frac.get("1", 0.0) else "TZ/CG"
                zone_note = "Zone labels from segmentation (1=CG, 2=PZ)."

        lesions.append(
            {
                "component_id": int(lbl),
                "voxel_count": vox,
                "volume_cc": round(float(vox * spacing[0] * spacing[1] * spacing[2]) / 1000.0, 3),
                "bbox_mm": [
                    round(float(bbox[3]) * float(spacing[0]), 2),
                    round(float(bbox[4]) * float(spacing[1]), 2),
                    round(float(bbox[5]) * float(spacing[2]), 2),
                ],
                "center_index_xyz": [int(cx), int(cy), int(cz)],
                "z_percent_from_inferior": round(float(z_frac) * 100.0, 1) if z_frac is not None else None,
                "cc_segment": ("apex" if z_frac is not None and z_frac < 0.33 else "mid" if z_frac is not None and z_frac < 0.66 else "base" if z_frac is not None else None),
                "clock_face_image": f"{clock} o'clock",
                "axial_quadrant_image": f"{lr}-{ap}",
                "zone": zone_label,
                "zone_overlap_fraction": zone_frac if zone_frac else None,
                "capsule_contact": capsule_contact,
                "notes": zone_note,
            }
        )

    lesions = sorted(lesions, key=lambda x: float(x.get("volume_cc") or 0.0), reverse=True)
    return {
        "ok": True,
        "lesions": lesions,
        "prostate_bbox_index_zyx": [int(zmin), int(ymin), int(xmin), int(zmax), int(ymax), int(xmax)],
        "orientation_note": "Lesion location derived in physical space; left/right uses current pipeline convention (dx>=0 => right).",
    }


def package_vlm_evidence(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    case_state_path = Path(args["case_state_path"]).expanduser().resolve()
    out_subdir = str(args.get("output_subdir") or "vlm").strip() or "vlm"

    state = json.loads(case_state_path.read_text(encoding="utf-8"))
    stage_outputs = state.get("stage_outputs", {}) or {}
    run_dir = case_state_path.parent

    # Where we persist the evidence bundle (stable location)
    vlm_dir = ctx.artifacts_dir / out_subdir
    vlm_dir.mkdir(parents=True, exist_ok=True)

    def _stage_sort_key(k: str):
        try:
            return (0, int(k), "")
        except Exception:
            return (1, 10**9, str(k))

    def best_record(tool_name: str) -> Tuple[Optional[bool], Dict[str, Any]]:
        """
        Prefer a successful record if any exists; otherwise fall back to the latest record.
        """
        if not isinstance(stage_outputs, dict):
            return None, {}
        latest_ok: Optional[bool] = None
        latest_data: Dict[str, Any] = {}
        best_ok: Optional[bool] = None
        best_data: Dict[str, Any] = {}
        for st in sorted(stage_outputs.keys(), key=_stage_sort_key):
            tools = stage_outputs.get(st, {}) or {}
            if not isinstance(tools, dict):
                continue
            records = tools.get(tool_name, []) or []
            if not isinstance(records, list) or not records:
                continue
            rec = records[-1] or {}
            latest_ok = rec.get("ok")
            latest_data = rec.get("data", {}) or {}
            if rec.get("ok") is True:
                best_ok = True
                best_data = latest_data
        if best_ok is True:
            return True, best_data
        return latest_ok, latest_data

    ingest_ok, ingest = best_record("ingest_dicom_to_nifti")
    ident_ok, ident = best_record("identify_sequences")
    reg_ok, reg = best_record("register_to_reference")
    seg_ok, seg = best_record("segment_prostate")
    card_seg_ok, card_seg = best_record("segment_cardiac_cine")
    card_cls_ok, card_cls = best_record("classify_cardiac_cine_disease")
    feat_ok, feat = best_record("extract_roi_features")
    les_ok, les = best_record("detect_lesion_candidates")
    dist_ok, dist = best_record("correct_prostate_distortion")
    seg_data = seg if bool(seg_ok) else (card_seg if isinstance(card_seg, dict) else {})
    seg_any_ok = bool(seg_ok) or bool(card_seg_ok)

    mapping = (ident.get("mapping", {}) if isinstance(ident, dict) else {}) or {}
    series = (ingest.get("series", []) if isinstance(ingest, dict) else []) or []
    present = sorted(
        set((s.get("sequence_guess") for s in series if isinstance(s, dict) and s.get("sequence_guess")))
    )

    feature_path = (feat.get("feature_table_path") if isinstance(feat, dict) else None)
    slice_path = (feat.get("slice_summary_path") if isinstance(feat, dict) else None)
    overlay_list = (feat.get("overlay_pngs") if isinstance(feat, dict) else [])
    run_art_dir = run_dir / "artifacts"

    def _feature_set_from_paths(
        feat_path: Optional[str], slc_path: Optional[str], overlays: Any, *, ok_hint: bool
    ) -> Dict[str, Any]:
        overlay_items = overlays if isinstance(overlays, list) else []
        md_path = _find_feature_md_path(feat_path)
        return {
            "ok": bool(ok_hint or feat_path),
            "feature_table_path": to_short_path(feat_path, run_dir=run_dir),
            "feature_table_path_abs": feat_path,
            "feature_table_md_path": to_short_path(md_path, run_dir=run_dir),
            "feature_table_md_path_abs": md_path,
            "slice_summary_path": to_short_path(slc_path, run_dir=run_dir),
            "slice_summary_path_abs": slc_path,
            "overlay_pngs": list_with_short_paths(overlay_items, run_dir=run_dir),
            "features_preview_rows": _read_features_csv_preview(feat_path, max_rows=64),
            "slice_summary_preview": _read_slice_summary_preview(slc_path, per_sequence_cap=12),
        }

    # Prefer canonical subdirs, but fall back to the latest tool output when needed.
    features_main = _feature_set_from_subdir(run_dir, "features")
    features_lesion = _feature_set_from_subdir(run_dir, "features_lesion")

    last_is_lesion = isinstance(feature_path, str) and "features_lesion" in feature_path.replace("\\", "/")
    if not features_main.get("ok"):
        main_disc = _discover_main_feature_path(run_art_dir)
        if main_disc:
            main_slc = str(Path(main_disc).with_name("slice_summary.json"))
            features_main = _feature_set_from_paths(main_disc, main_slc, [], ok_hint=True)
        elif not last_is_lesion:
            features_main = _feature_set_from_paths(feature_path, slice_path, overlay_list, ok_hint=bool(feat_ok))

    if not features_lesion.get("ok") and last_is_lesion:
        features_lesion = _feature_set_from_paths(feature_path, slice_path, overlay_list, ok_hint=bool(feat_ok))

    features_primary = features_main if features_main.get("ok") else features_lesion
    dicom_meta_path = ident.get("dicom_meta_path") if isinstance(ident, dict) else None
    header_index_path = ident.get("dicom_headers_index_path") if isinstance(ident, dict) else None
    dicom_summary = _summarize_dicom_meta(dicom_meta_path)
    if header_index_path:
        dicom_summary["dicom_headers_index_path"] = str(header_index_path)

    run_dir = case_state_path.parent

    evidence_bundle: Dict[str, Any] = {
        "case_id": state.get("case_id"),
        "run_id": state.get("run_id"),
        "mapping": mapping,
        "dicom_summary": dicom_summary,
        "series_summary": _summarize_series_list(series, max_items=32),
        "pipeline_status": {
            "ingest_dicom_to_nifti_ok": bool(ingest_ok),
            "identify_sequences_ok": bool(ident_ok),
            "register_to_reference_ok": bool(reg_ok),
            "segment_prostate_ok": bool(seg_ok),
            "segment_cardiac_cine_ok": bool(card_seg_ok),
            "classify_cardiac_cine_disease_ok": bool(card_cls_ok),
            "extract_roi_features_ok": bool(feat_ok),
            "extract_roi_features_main_ok": bool(features_main.get("ok")),
            "extract_roi_features_lesion_ok": bool(features_lesion.get("ok")),
            "correct_prostate_distortion_ok": bool(dist_ok),
            "sequences_present": present,
        },
        # Back-compat: `features` points to the primary (prefer main) set.
        "features": features_primary,
        "features_main": features_main,
        "features_lesion": features_lesion,
        "segmentation": {
            "ok": bool(seg_any_ok),
            "prostate_mask_path": to_short_path((seg_data.get("prostate_mask_path") if isinstance(seg_data, dict) else None), run_dir=run_dir),
            "prostate_mask_path_abs": (seg_data.get("prostate_mask_path") if isinstance(seg_data, dict) else None),
            "zone_mask_path": to_short_path((seg_data.get("zone_mask_path") if isinstance(seg_data, dict) else None), run_dir=run_dir),
            "zone_mask_path_abs": (seg_data.get("zone_mask_path") if isinstance(seg_data, dict) else None),
            "cardiac_seg_path": to_short_path((seg_data.get("seg_path") if isinstance(seg_data, dict) else None), run_dir=run_dir),
            "cardiac_seg_path_abs": (seg_data.get("seg_path") if isinstance(seg_data, dict) else None),
            "rv_mask_path": to_short_path((seg_data.get("rv_mask_path") if isinstance(seg_data, dict) else None), run_dir=run_dir),
            "rv_mask_path_abs": (seg_data.get("rv_mask_path") if isinstance(seg_data, dict) else None),
            "myo_mask_path": to_short_path((seg_data.get("myo_mask_path") if isinstance(seg_data, dict) else None), run_dir=run_dir),
            "myo_mask_path_abs": (seg_data.get("myo_mask_path") if isinstance(seg_data, dict) else None),
            "lv_mask_path": to_short_path((seg_data.get("lv_mask_path") if isinstance(seg_data, dict) else None), run_dir=run_dir),
            "lv_mask_path_abs": (seg_data.get("lv_mask_path") if isinstance(seg_data, dict) else None),
        },
        "cardiac_classification": {
            "ok": bool(card_cls_ok),
            "classification_path": to_short_path((card_cls.get("classification_path") if isinstance(card_cls, dict) else None), run_dir=run_dir),
            "classification_path_abs": (card_cls.get("classification_path") if isinstance(card_cls, dict) else None),
            "predicted_group": (card_cls.get("predicted_group") if isinstance(card_cls, dict) else None),
            "ground_truth_group": (card_cls.get("ground_truth_group") if isinstance(card_cls, dict) else None),
            "ground_truth_match": (card_cls.get("ground_truth_match") if isinstance(card_cls, dict) else None),
            "needs_vlm_review": (card_cls.get("needs_vlm_review") if isinstance(card_cls, dict) else None),
            "metrics": (card_cls.get("metrics") if isinstance(card_cls, dict) else None),
            "phase_indices": (card_cls.get("phase_indices") if isinstance(card_cls, dict) else None),
            "rule_trace": (card_cls.get("rule_trace") if isinstance(card_cls, dict) else None),
        },
        "lesion_candidates": {
            "ok": bool(les_ok),
            "candidates_path": to_short_path((les.get("candidates_path") if isinstance(les, dict) else None), run_dir=run_dir),
            "candidates_path_abs": (les.get("candidates_path") if isinstance(les, dict) else None),
            "merged_prob_path": to_short_path((les.get("merged_prob_path") if isinstance(les, dict) else None), run_dir=run_dir),
            "merged_prob_path_abs": (les.get("merged_prob_path") if isinstance(les, dict) else None),
            "lesion_mask_path": to_short_path((les.get("lesion_mask_path") if isinstance(les, dict) else None), run_dir=run_dir),
            "lesion_mask_path_abs": (les.get("lesion_mask_path") if isinstance(les, dict) else None),
            "filter_stats": (les.get("filter_stats") if isinstance(les, dict) else None),
            "candidate_montages": (les.get("candidate_montages") if isinstance(les, dict) else None),
        },
        "distortion_correction": {
            "ok": bool(dist_ok),
            "out_dir": to_short_path((dist.get("out_dir") if isinstance(dist, dict) else None), run_dir=run_dir),
            "out_dir_abs": (dist.get("out_dir") if isinstance(dist, dict) else None),
            "summary_json_path": to_short_path((dist.get("summary_json_path") if isinstance(dist, dict) else None), run_dir=run_dir),
            "summary_json_path_abs": (dist.get("summary_json_path") if isinstance(dist, dict) else None),
            "num_pred_npz": (dist.get("num_pred_npz") if isinstance(dist, dict) else None),
            "num_panel_png": (dist.get("num_panel_png") if isinstance(dist, dict) else None),
        },
        "lesion_geometry": _compute_lesion_geometry(
            lesion_mask_path=(les.get("lesion_mask_path") if isinstance(les, dict) else None),
            prostate_mask_path=(seg_data.get("prostate_mask_path") if isinstance(seg_data, dict) else None),
            zone_mask_path=(seg_data.get("zone_mask_path") if isinstance(seg_data, dict) else None),
        ),
        "qc_pngs": [to_short_path(p, run_dir=run_dir) for p in _gather_pngs(run_dir)],
        "notes": [
            "Ground all statements in evidence. If evidence is insufficient, say 'indeterminate' and list limitations.",
            "Do not invent lesions or numeric measurements not present in features.csv (or features.md) / slice_summary.json.",
        ],
    }

    artifact_index_path, artifact_index = build_artifact_index(case_state_path=case_state_path, artifacts_dir=ctx.artifacts_dir)
    evidence_bundle["artifact_index_path"] = to_short_path(str(artifact_index_path), run_dir=run_dir)
    evidence_bundle["artifact_index"] = {
        "n_artifacts": len(artifact_index.get("artifacts") or []),
    }

    vlm_evidence_bundle_path = vlm_dir / "vlm_evidence_bundle.json"
    vlm_evidence_bundle_path.write_text(json.dumps(evidence_bundle, indent=2) + "\n", encoding="utf-8")

    # Also write a stable CaseContext for downstream workflows (lightweight entry-point).
    context_dir = ctx.artifacts_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    case_context_path = context_dir / "case_context.json"

    meas: Dict[str, Any] = {}
    try:
        rows = evidence_bundle.get("features_main", {}).get("features_preview_rows") or evidence_bundle.get(
            "features", {}
        ).get("features_preview_rows") or []
        if isinstance(rows, list):
            for r in rows:
                if isinstance(r, dict) and r.get("roi") == "whole_gland" and r.get("volume_ml") is not None:
                    meas["prostate_volume_ml"] = float(r["volume_ml"])
                    break
    except Exception:
        pass
    try:
        card_cls = evidence_bundle.get("cardiac_classification", {})
        if isinstance(card_cls, dict):
            if card_cls.get("predicted_group"):
                meas["cardiac_predicted_group"] = card_cls.get("predicted_group")
            metrics = card_cls.get("metrics") if isinstance(card_cls.get("metrics"), dict) else {}
            for key in ("lv_ef_percent", "rv_ef_percent", "lv_edv_ml", "rv_edv_ml", "max_myo_thickness_mm"):
                if key in metrics and metrics.get(key) is not None:
                    meas[key] = metrics.get(key)
    except Exception:
        pass

    case_context = {
        "case_id": state.get("case_id"),
        "run_id": state.get("run_id"),
        "generated_at": str(state.get("created_at") or ""),
        "reference": {"suggested": "T2w"},
        "mapping": mapping,
        "dicom_summary": dicom_summary,
        "lesion_geometry": evidence_bundle.get("lesion_geometry"),
        "stage_meta": state.get("stage_meta", {}) if isinstance(state, dict) else {},
        "artifacts": {
            "segmentation": evidence_bundle.get("segmentation", {}),
            "cardiac_classification": evidence_bundle.get("cardiac_classification", {}),
            "lesion_candidates": evidence_bundle.get("lesion_candidates", {}),
            "distortion_correction": evidence_bundle.get("distortion_correction", {}),
            "features": evidence_bundle.get("features", {}),
            "features_main": evidence_bundle.get("features_main", {}),
            "features_lesion": evidence_bundle.get("features_lesion", {}),
            "qc_pngs": evidence_bundle.get("qc_pngs"),
            "vlm_evidence_bundle_path": to_short_path(str(vlm_evidence_bundle_path), run_dir=run_dir),
            "vlm_evidence_bundle_path_abs": str(vlm_evidence_bundle_path),
            "artifact_index_path": evidence_bundle.get("artifact_index_path"),
        },
        "measurements_summary": meas,
    }
    case_context_path.write_text(json.dumps(case_context, indent=2) + "\n", encoding="utf-8")

    artifacts = [
        ArtifactRef(path=str(vlm_evidence_bundle_path), kind="json", description="VLM evidence bundle"),
        ArtifactRef(path=str(case_context_path), kind="json", description="CaseContext (stable run context)"),
    ]

    qc_count = len(evidence_bundle.get("qc_pngs") or [])
    feature_count = len(evidence_bundle.get("features") or {})
    return {
        "data": {
            "status": "success",
            "vlm_evidence_path": str(vlm_evidence_bundle_path),
            "summary": f"Packaged {feature_count} feature sets and {qc_count} QC images.",
        },
        "artifacts": artifacts,
        "warnings": [],
        "source_artifacts": [ArtifactRef(path=str(case_state_path), kind="json", description="CaseState input")],
        "generated_artifacts": artifacts,
    }


def build_tool() -> Tool:
    return Tool(spec=PKG_SPEC, func=package_vlm_evidence)

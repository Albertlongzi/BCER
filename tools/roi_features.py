"""
Tool: extract_roi_features (Stage 4)

Purpose:
- Produce a structured feature table for ROI × sequence.

MVP status:
- Upgraded to real measurements using SimpleITK + numpy.
- Computes ROI stats for whole gland and (if available) zones.
- Optional: PyRadiomics features (first-order, shape, texture matrices).

Inputs:
- roi_mask_path: path to ROI mask (NIfTI) [legacy single-ROI]
- roi_masks: list of {"name": str, "path": str} for multi-ROI
- images: list of {"name": str, "path": str} for each sequence/volume
- output_subdir (optional)

Outputs:
- feature_table_path (CSV)
- slice_summary_path (JSON)
- overlay_pngs (optional): quick-look ROI overlays/montages (full-frame or ROI-cropped)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec


FEAT_SPEC = ToolSpec(
    name="extract_roi_features",
    description=(
        "Compute quantitative ROI × sequence measurements from NIfTI volumes. "
        "Supports multi-ROI via `roi_masks` (e.g., whole gland + lesion mask). "
        "Optionally accepts `context_mask_path` to control field-of-view (FOV) while keeping ROI selection separate. "
        "Optionally computes PyRadiomics features (first-order, shape, texture matrices). "
        "Given one or more ROI masks and a list of NIfTI images, writes `features.csv` (volume + intensity stats "
        "+ optional radiomics) and `slice_summary.json`, plus QC overlays/montages."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "roi_mask_path": {"type": "string"},
            "roi_masks": {"type": "array"},
            "images": {"type": "array"},
            "output_subdir": {"type": "string"},
            "reference_slices": {"type": "array"},
            "reference_slice_summary_path": {"type": "string"},
            "context_mask_path": {"type": "string"},
            "roi_interpretation": {"type": "string"},
            "overlay_mode": {"type": "string"},
            "radiomics": {"type": "object"},
            "radiomics_mode": {"type": "string"},
            "radiomics_feature_classes": {"type": "array"},
            "radiomics_image_types": {"type": "array"},
            "radiomics_settings": {"type": "object"},
        },
        "required": ["images"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "feature_table_path": {"type": "string"},
            "slice_summary_path": {"type": "string"},
            "overlay_pngs": {"type": "array"},
            "roi_sources": {"type": "array"},
            "roi_summaries": {"type": "array"},
        },
        "required": ["feature_table_path", "slice_summary_path"],
    },
    version="0.6.1",
    tags=["features", "mvp", "simpleitk", "pyradiomics"],
)


def _require_sitk_np():
    try:
        import SimpleITK as sitk  # type: ignore
    except Exception as e:
        raise RuntimeError("SimpleITK required for Stage-4 features: pip install SimpleITK") from e
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("numpy required for Stage-4 features") from e
    return sitk, np


DEFAULT_RADIOMICS_CLASSES = ["firstorder", "shape", "glcm", "glrlm", "glszm", "ngtdm", "gldm"]
LESION_RADIOMICS_FEATURES = {
    "firstorder": ["Entropy", "Skewness"],
    "shape": ["SurfaceVolumeRatio"],
    "glcm": ["Contrast", "Idn"],
    "glszm": ["SizeZoneNonUniformity"],
}


def _normalize_str_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, str):
        parts = [p.strip() for p in val.replace(";", ",").replace("|", ",").split(",")]
        return [p for p in parts if p]
    if isinstance(val, (list, tuple, set)):
        out: List[str] = []
        for it in val:
            s = str(it).strip()
            if s:
                out.append(s)
        return out
    s = str(val).strip()
    return [s] if s else []


def _normalize_radiomics_mode(val: Any) -> str:
    if val is None:
        return "auto"
    if isinstance(val, bool):
        return "on" if val else "off"
    s = str(val).strip().lower()
    if s in ("on", "true", "1", "yes", "y", "enable", "enabled"):
        return "on"
    if s in ("off", "false", "0", "no", "n", "disable", "disabled"):
        return "off"
    if s in ("auto",):
        return "auto"
    return "auto"


def _jsonable(obj: Any, np) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, np) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, np) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _coerce_rad_value(val: Any, np) -> Any:
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, (list, tuple)):
        return [_coerce_rad_value(v, np) for v in val]
    if isinstance(val, dict):
        return _jsonable(val, np)
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return val.reshape(-1)[0].item()
        return [_coerce_rad_value(v, np) for v in val.reshape(-1)]
    try:
        return float(val)
    except Exception:
        return str(val)


def _sanitize_radiomics_output(result: Dict[str, Any], np) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in result.items():
        key = str(k)
        if key.startswith("diagnostics"):
            continue
        out[f"rad_{key}"] = _coerce_rad_value(v, np)
    return out


def _build_radiomics_extractor(
    *,
    mode: str,
    feature_classes: List[str],
    image_types: Any,
    settings: Dict[str, Any],
    warnings: List[str],
    feature_names: Dict[str, List[str]] | None = None,
):
    if mode == "off":
        return None, {"mode": mode, "enabled": False}
    try:
        from radiomics import featureextractor, __version__ as pyrad_version  # type: ignore

        logging.getLogger("radiomics").setLevel(logging.ERROR)
    except Exception as e:
        if mode == "on":
            raise RuntimeError(
                "pyradiomics is required for radiomics features. Install with: pip install pyradiomics"
            ) from e
        warnings.append("pyradiomics not available; radiomics features disabled.")
        return None, {"mode": mode, "enabled": False, "error": "missing_pyradiomics"}

    safe_settings: Dict[str, Any] = {}
    if isinstance(settings, dict):
        safe_settings = dict(settings)
    extractor = featureextractor.RadiomicsFeatureExtractor(**safe_settings)
    extractor.disableAllFeatures()
    if feature_names:
        use_classes = list(feature_names.keys())
        try:
            extractor.enableFeaturesByName(**feature_names)
        except Exception as e:
            warnings.append(f"Radiomics: enableFeaturesByName failed ({type(e).__name__}); falling back to class-level enable.")
            for cls_name in use_classes:
                try:
                    extractor.enableFeatureClassByName(str(cls_name))
                except Exception:
                    warnings.append(f"Radiomics: unknown feature class '{cls_name}' ignored.")
    else:
        use_classes = feature_classes or DEFAULT_RADIOMICS_CLASSES
        for cls_name in use_classes:
            try:
                extractor.enableFeatureClassByName(str(cls_name))
            except Exception:
                warnings.append(f"Radiomics: unknown feature class '{cls_name}' ignored.")
    try:
        extractor.disableAllImageTypes()
        if isinstance(image_types, dict) and image_types:
            extractor.enableImageTypes(image_types)
        else:
            img_list = _normalize_str_list(image_types) if image_types is not None else []
            if not img_list:
                extractor.enableImageTypes(Original={})
            else:
                for it in img_list:
                    try:
                        extractor.enableImageTypeByName(str(it))
                    except Exception:
                        warnings.append(f"Radiomics: unknown image type '{it}' ignored.")
                if not extractor.enabledImagetypes:
                    extractor.enableImageTypes(Original={})
    except Exception:
        extractor.enableImageTypes(Original={})

    meta = {
        "mode": mode,
        "enabled": True,
        "feature_classes": use_classes,
        "feature_names": feature_names,
        "image_types": image_types if image_types is not None else ["Original"],
        "settings": safe_settings,
        "pyradiomics_version": pyrad_version,
    }
    return extractor, meta


def _is_texture_feature_name(name: str) -> bool:
    low = str(name).lower()
    if not low.startswith("rad_"):
        return False
    return any(k in low for k in ("glcm", "glrlm", "glszm", "gldm", "ngtdm"))


def _build_texture_visualization(
    *,
    rows: List[Dict[str, Any]],
    out_dir: Path,
    heatmap_dir: Optional[Path] = None,
) -> Dict[str, Optional[str]]:
    """
    Build texture-feature summary + heatmap for quick QC/interpretation.
    Returns paths (or None) for generated artifacts.
    """
    # Collect numeric texture features per (roi_source, roi, sequence).
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        row_key = "|".join(
            [
                str(r.get("roi_source") or ""),
                str(r.get("roi") or ""),
                str(r.get("sequence") or ""),
            ]
        )
        if row_key not in grouped:
            grouped[row_key] = {}
        for k, v in r.items():
            if not _is_texture_feature_name(str(k)):
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            grouped[row_key].setdefault(str(k), []).append(fv)

    if not grouped:
        return {"summary_csv_path": None, "heatmap_png_path": None}

    # Aggregate mean by key.
    all_feats: set[str] = set()
    agg_rows: List[Dict[str, Any]] = []
    for row_key, feat_map in grouped.items():
        roi_source, roi, sequence = row_key.split("|", 2)
        row: Dict[str, Any] = {"roi_source": roi_source, "roi": roi, "sequence": sequence}
        for k, vals in feat_map.items():
            if not vals:
                continue
            row[k] = float(sum(vals) / len(vals))
            all_feats.add(k)
        agg_rows.append(row)

    if not agg_rows:
        return {"summary_csv_path": None, "heatmap_png_path": None}
    if not all_feats:
        return {"summary_csv_path": None, "heatmap_png_path": None}

    # Write compact summary CSV first.
    summary_csv_path = out_dir / "texture_radiomics_summary.csv"
    feat_names = sorted(all_feats)
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["roi_source", "roi", "sequence"] + feat_names)
        writer.writeheader()
        for r in agg_rows:
            writer.writerow(r)

    # Best-effort heatmap (optional dependency).
    heatmap_png_path: Optional[Path] = None
    heatmap_root = heatmap_dir if isinstance(heatmap_dir, Path) else out_dir
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        # Keep visualization readable: top features by variance.
        if len(feat_names) > 24:
            variances: List[tuple[str, float]] = []
            for k in feat_names:
                vals = []
                for r in agg_rows:
                    v = r.get(k)
                    try:
                        vals.append(float(v))
                    except Exception:
                        continue
                if len(vals) >= 2:
                    variances.append((k, float(np.var(np.asarray(vals, dtype=float)))))
                else:
                    variances.append((k, 0.0))
            feat_names = [k for k, _ in sorted(variances, key=lambda x: x[1], reverse=True)[:24]]

        mat = []
        for r in agg_rows:
            row_vals = []
            for k in feat_names:
                try:
                    row_vals.append(float(r.get(k)))
                except Exception:
                    row_vals.append(float("nan"))
            mat.append(row_vals)
        arr = np.asarray(mat, dtype=float)
        if arr.size > 0:
            # z-score per feature for visual comparability.
            mu = np.nanmean(arr, axis=0, keepdims=True)
            sd = np.nanstd(arr, axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            z = (arr - mu) / sd
            z = np.nan_to_num(z, nan=0.0, posinf=2.5, neginf=-2.5)
            z = np.clip(z, -2.5, 2.5)
            zn = (z + 2.5) / 5.0  # [0,1]

            # Blue-white-red palette without matplotlib.
            # low -> blue, mid -> white, high -> red
            r = (255.0 * zn).astype("uint8")
            g = (255.0 * (1.0 - np.abs(zn - 0.5) * 2.0)).astype("uint8")
            b = (255.0 * (1.0 - zn)).astype("uint8")
            rgb = np.stack([r, g, b], axis=-1)

            img = Image.fromarray(rgb, mode="RGB")
            cell_w = 24
            cell_h = 24
            img = img.resize((max(1, rgb.shape[1] * cell_w), max(1, rgb.shape[0] * cell_h)), resample=Image.NEAREST)
            heatmap_root.mkdir(parents=True, exist_ok=True)
            heatmap_png_path = heatmap_root / "texture_radiomics_heatmap.png"
            img.save(str(heatmap_png_path))
    except Exception:
        heatmap_png_path = None

    return {
        "summary_csv_path": str(summary_csv_path),
        "heatmap_png_path": (str(heatmap_png_path) if heatmap_png_path else None),
    }


def _resample_mask_to_image(mask_img, ref_img):
    sitk, _ = _require_sitk_np()
    tx = sitk.Transform(3, sitk.sitkIdentity)
    return sitk.Resample(mask_img, ref_img, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def _stats(arr, np):
    flat = arr.reshape(-1)
    if flat.size == 0:
        return {}
    flat = flat.astype("float32", copy=False)
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat)),
        "p10": float(np.percentile(flat, 10)),
        "p90": float(np.percentile(flat, 90)),
    }


def _robust_window_uint8(slice_f32, np):
    vals = slice_f32.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (slice_f32 * 0).astype("uint8")
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if hi <= lo:
        hi = lo + 1.0
    x = (slice_f32 - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype("uint8")


def _robust_window_uint8_masked(slice_f32, mask_u8, np):
    """
    Window using ROI-aware but context-preserving statistics.
    Using only ROI voxels can over-saturate tiny lesions; instead compute
    percentiles on a dilated ROI within a padded bbox.
    """
    if mask_u8 is None:
        return _robust_window_uint8(slice_f32, np)
    m = (mask_u8 > 0).astype("uint8", copy=False)
    if int(np.count_nonzero(m)) == 0:
        return _robust_window_uint8(slice_f32, np)

    bb = _bbox_2d(m, np)
    if bb is None:
        return _robust_window_uint8(slice_f32, np)
    x0, x1, y0, y1 = bb
    h, w = m.shape
    pad = 24
    x0 = max(0, x0 - int(pad))
    y0 = max(0, y0 - int(pad))
    x1 = min(w - 1, x1 + int(pad))
    y1 = min(h - 1, y1 + int(pad))

    crop = slice_f32[y0 : y1 + 1, x0 : x1 + 1]
    crop_m = m[y0 : y1 + 1, x0 : x1 + 1]

    # Dilate to include peri-ROI tissue for more stable contrast.
    dil = _binary_dilation_3x3(crop_m, np)
    dil = _binary_dilation_3x3(dil, np)
    vals = crop[dil > 0].reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size < 64:
        # Fallback to any finite values in the padded bbox.
        vals = crop[np.isfinite(crop)].reshape(-1)
    if vals.size == 0:
        return (slice_f32 * 0).astype("uint8")

    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if hi <= lo:
        hi = lo + 1.0
    x = (slice_f32 - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype("uint8")


def _bbox_2d(mask2d_u8, np):
    ys, xs = np.where(mask2d_u8 > 0)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return x0, x1, y0, y1


def _crop_and_mask_slice(*, slice_f32, mask2d_u8, np, pad: int = 10):
    """
    Remove non-mask region by:
    - tight bbox around ROI mask (with padding)
    - zeroing outside-mask pixels (within the crop)
    """
    bb = _bbox_2d(mask2d_u8, np)
    if bb is None:
        return None
    x0, x1, y0, y1 = bb
    h, w = mask2d_u8.shape
    x0 = max(0, x0 - int(pad))
    y0 = max(0, y0 - int(pad))
    x1 = min(w - 1, x1 + int(pad))
    y1 = min(h - 1, y1 + int(pad))

    crop = slice_f32[y0 : y1 + 1, x0 : x1 + 1].astype("float32", copy=False)
    m = mask2d_u8[y0 : y1 + 1, x0 : x1 + 1]
    crop[m <= 0] = 0.0
    return crop


def _crop_slice_bbox(*, slice_f32, mask2d_u8, np, pad: int = 10):
    """
    Crop to a tight bbox around ROI mask (with padding) WITHOUT masking-out background.
    Returns (crop_f32, crop_mask_u8, bbox=(x0,x1,y0,y1)).
    """
    bb = _bbox_2d(mask2d_u8, np)
    if bb is None:
        return None, None, None
    x0, x1, y0, y1 = bb
    h, w = mask2d_u8.shape
    x0 = max(0, x0 - int(pad))
    y0 = max(0, y0 - int(pad))
    x1 = min(w - 1, x1 + int(pad))
    y1 = min(h - 1, y1 + int(pad))
    crop = slice_f32[y0 : y1 + 1, x0 : x1 + 1].astype("float32", copy=False)
    m = mask2d_u8[y0 : y1 + 1, x0 : x1 + 1].astype("uint8", copy=False)
    return crop, m, (x0, x1, y0, y1)


def _binary_erosion_3x3(mask_u8, np):
    """
    Simple 3x3 binary erosion (no scipy dependency).
    """
    m = (mask_u8 > 0).astype("uint8")
    if m.size == 0:
        return m
    p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    # AND over 3x3 neighborhood
    out = p[1:-1, 1:-1].copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            out &= p[1 + dy : 1 + dy + m.shape[0], 1 + dx : 1 + dx + m.shape[1]]
    return out


def _binary_dilation_3x3(mask_u8, np):
    """
    Simple 3x3 binary dilation (no scipy dependency).
    """
    m = (mask_u8 > 0).astype("uint8")
    if m.size == 0:
        return m
    p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    out = p[1:-1, 1:-1].copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            out |= p[1 + dy : 1 + dy + m.shape[0], 1 + dx : 1 + dx + m.shape[1]]
    return out


def _resize_2d(img2d, new_size_xy, sitk):
    """
    Resize a 2D SimpleITK image to (width,height)=new_size_xy using linear interpolation.
    """
    new_w, new_h = int(new_size_xy[0]), int(new_size_xy[1])
    old_w, old_h = img2d.GetSize()
    if old_w <= 0 or old_h <= 0 or new_w <= 0 or new_h <= 0:
        return img2d
    old_sp = img2d.GetSpacing()
    new_sp = (
        float(old_sp[0]) * float(old_w) / float(new_w),
        float(old_sp[1]) * float(old_h) / float(new_h),
    )
    ref = sitk.Image([new_w, new_h], img2d.GetPixelID())
    ref.SetSpacing(new_sp)
    ref.SetOrigin(img2d.GetOrigin())
    ref.SetDirection(img2d.GetDirection())
    tx = sitk.Transform(2, sitk.sitkIdentity)
    return sitk.Resample(img2d, ref, tx, sitk.sitkLinear, 0, img2d.GetPixelID())


def _extract_4d_volume(vol, idx, sitk):
    size = list(vol.GetSize())
    if len(size) < 4:
        return vol
    size[3] = 0
    index = [0, 0, 0, int(idx)]
    return sitk.Extract(vol, size, index)


def _collapse_4d_mask(mask_img, *, sitk, np, warnings: List[str]):
    try:
        size = list(mask_img.GetSize())
        if len(size) < 4 or size[3] <= 1:
            return mask_img
        first = _extract_4d_volume(mask_img, 0, sitk)
        arr = sitk.GetArrayFromImage(mask_img)
        collapsed = (arr > 0).any(axis=0).astype("uint8")
        out = sitk.GetImageFromArray(collapsed)
        out.CopyInformation(first)
        warnings.append("ROI mask is 4D; collapsed across channels to build a 3D mask.")
        return out
    except Exception:
        return mask_img


def _split_4d_image(vol, *, base_name: str, path: str, sitk, warnings: List[str]):
    if vol.GetDimension() != 4:
        return [{"name": base_name, "vol": vol}]
    size = list(vol.GetSize())
    n = int(size[3]) if len(size) > 3 else 0
    if n <= 0:
        warnings.append(f"4D image '{base_name}' has no volumes; skipping.")
        return []
    low = str(path or "").lower()
    labels = None
    if "tumor_subregions" in low or "subregions" in low:
        labels = ["TC", "WT", "ET"]
    out: List[Dict[str, Any]] = []
    for idx in range(n):
        vol3 = _extract_4d_volume(vol, idx, sitk)
        suffix = None
        if labels and idx < len(labels):
            suffix = labels[idx]
        name = f"{base_name}_{suffix}" if suffix else f"{base_name}_c{idx}"
        out.append({"name": name, "vol": vol3})
    warnings.append(f"4D image '{base_name}' split into {len(out)} volume(s) for feature extraction.")
    return out


def _pick_best_roi_slice(mask_zyx, np, reference_slices: List[int] | None = None) -> int:
    best_z = 0
    best = -1
    zs: List[int]
    if reference_slices:
        zs = [int(z) for z in reference_slices if 0 <= int(z) < int(mask_zyx.shape[0])]
        if not zs:
            zs = list(range(int(mask_zyx.shape[0])))
    else:
        zs = list(range(int(mask_zyx.shape[0])))
    for z in zs:
        n = int(np.count_nonzero(mask_zyx[z]))
        if n > best:
            best = n
            best_z = z
    return int(best_z)


def _roi_slice_indices(mask_zyx, np) -> List[int]:
    try:
        idx = np.where(np.any(mask_zyx > 0, axis=(1, 2)))[0]
        return [int(z) for z in idx.tolist()]
    except Exception:
        return []


def _roi_short_label(name: str) -> str:
    low = str(name or "").strip().lower()
    if low in ("tumor_core", "tumor core", "tc"):
        return "TC"
    if low in ("whole_tumor", "whole tumor", "wt"):
        return "WT"
    if low in ("enhancing_tumor", "enhancing tumor", "et"):
        return "ET"
    return str(name or "").strip()


def _save_roi_crop_best_slice_png(
    *,
    vol,
    mask_res,
    out_path: Path,
    reference_slices: List[int] | None = None,
    context_mask_res=None,
    pad: int = 18,
    full_frame: bool = False,
    roi_label: Optional[str] = None,
) -> None:
    """
    Save a zoomed, ROI-outlined best slice to make VLM review easier.
    """
    sitk, np = _require_sitk_np()
    v = sitk.GetArrayViewFromImage(vol).astype("float32", copy=False)  # z,y,x
    m = sitk.GetArrayViewFromImage(mask_res)
    wg = (m > 0).astype("uint8")
    ctx = None
    if context_mask_res is not None:
        try:
            ctx = sitk.GetArrayViewFromImage(context_mask_res)
        except Exception:
            ctx = None
    z = _pick_best_roi_slice(wg, np, reference_slices=reference_slices)

    mask2d = wg[z].astype("uint8")
    if int(mask2d.max()) == 0:
        return

    ctx2d = None
    if ctx is not None:
        ctx2d = (ctx[z] > 0).astype("uint8", copy=False)
        if int(ctx2d.max()) == 0:
            ctx2d = None
    lab2d = m[z].astype("uint8", copy=False)
    if full_frame:
        crop = v[z].astype("float32", copy=False)
        crop_mask = (mask2d > 0).astype("uint8", copy=False)
        x0, y0 = 0, 0
        x1, y1 = crop.shape[1] - 1, crop.shape[0] - 1
        lab_crop = lab2d
        roi_crop = (lab_crop > 0).astype("uint8")
        ctx_crop = ctx2d if ctx2d is not None else None
        win_mask = np.ones_like(crop_mask)
    else:
        bbox_mask = ctx2d if ctx2d is not None else mask2d
        crop, crop_mask, bbox = _crop_slice_bbox(slice_f32=v[z], mask2d_u8=bbox_mask, np=np, pad=pad)
        if crop is None or crop_mask is None or bbox is None:
            return
        x0, x1, y0, y1 = bbox
        lab_crop = lab2d[y0 : y1 + 1, x0 : x1 + 1]
        roi_crop = (lab_crop > 0).astype("uint8")
        ctx_crop = None
        if ctx2d is not None:
            ctx_crop = ctx2d[y0 : y1 + 1, x0 : x1 + 1].astype("uint8", copy=False)
        win_mask = roi_crop if roi_crop is not None else crop_mask

    base_u8 = _robust_window_uint8_masked(crop, win_mask, np)

    # Prefer a colored outline when PIL is available.
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore

        rgb = np.stack([base_u8, base_u8, base_u8], axis=-1)
        cg = (lab_crop == 1).astype("uint8")
        pz = (lab_crop == 2).astype("uint8")
        if ctx_crop is not None:
            ctx_edge = (ctx_crop > 0) & (_binary_erosion_3x3(ctx_crop, np) == 0)
            rgb[ctx_edge] = (0, 255, 255)  # cyan = context (prostate)
            roi_edge = (roi_crop > 0) & (_binary_erosion_3x3(roi_crop, np) == 0)
            rgb[roi_edge] = (255, 0, 0)  # red = ROI (lesion)
        elif int(cg.max()) > 0 or int(pz.max()) > 0:
            if int(cg.max()) > 0:
                cg_edge = (cg > 0) & (_binary_erosion_3x3(cg, np) == 0)
                rgb[cg_edge] = (255, 0, 0)  # red = central gland
            if int(pz.max()) > 0:
                pz_edge = (pz > 0) & (_binary_erosion_3x3(pz, np) == 0)
                rgb[pz_edge] = (0, 255, 255)  # cyan = peripheral zone
        else:
            edge = (crop_mask > 0) & (_binary_erosion_3x3(crop_mask, np) == 0)
            rgb[edge] = (255, 0, 0)

        im = Image.fromarray(rgb, mode="RGB")
        # Keep a consistent, VLM-friendly size.
        im = im.resize((384, 384))
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        draw.text((6, 6), f"z={int(z)}", fill=(255, 255, 0), font=font)
        if ctx_crop is not None:
            draw.rectangle([6, 20, 14, 28], fill=(255, 0, 0))
            draw.text((18, 18), "ROI", fill=(255, 255, 255), font=font)
            draw.rectangle([6, 34, 14, 42], fill=(0, 255, 255))
            draw.text((18, 32), "CTX", fill=(255, 255, 255), font=font)
        elif int(cg.max()) > 0 or int(pz.max()) > 0:
            draw.rectangle([6, 20, 14, 28], fill=(255, 0, 0))
            draw.text((18, 18), "CG", fill=(255, 255, 255), font=font)
            draw.rectangle([6, 34, 14, 42], fill=(0, 255, 255))
            draw.text((18, 32), "PZ", fill=(255, 255, 255), font=font)
        elif roi_label:
            draw.text((6, 18), _roi_short_label(roi_label), fill=(255, 255, 255), font=font)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(str(out_path))
        return
    except Exception:
        pass

    # Fallback: grayscale via SimpleITK.
    base_img = sitk.GetImageFromArray(base_u8)  # 2D uint8
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(base_img, str(out_path))


def _save_roi_crop_montage_png(
    *,
    vol,
    mask_res,
    out_path: Path,
    max_slices: int = 16,
    grid_cols: int = 4,
    grid_rows: int = 4,
    reference_slices: List[int] | None = None,
    context_mask_res=None,
    slice_pad: int = 2,
    pad: int = 18,
    full_frame: bool = False,
    roi_label: Optional[str] = None,
) -> bool:
    """
    Create a tiled montage covering ROI slices, capped to max_slices (evenly sampled),
    using ROI-cropped images with zonal outlines (CG/PZ) when available.
    """
    sitk, np = _require_sitk_np()
    v = sitk.GetArrayViewFromImage(vol).astype("float32", copy=False)  # z,y,x
    m = sitk.GetArrayViewFromImage(mask_res)  # may be binary or zonal labels (0/1/2)
    wg = (m > 0).astype("uint8")
    ctx = None
    if context_mask_res is not None:
        try:
            ctx = sitk.GetArrayViewFromImage(context_mask_res)
        except Exception:
            ctx = None

    # find ROI-present slices
    idx = np.where(np.any(wg > 0, axis=(1, 2)))[0]
    if idx.size == 0:
        return False
    idx = idx.tolist()
    if reference_slices:
        ref = {int(z) for z in reference_slices if 0 <= int(z) < int(v.shape[0])}
        idx_ref = [int(z) for z in idx if int(z) in ref]
        if idx_ref:
            idx = idx_ref
    # If ROI is tiny, pad by a few slices around the lesion (still ROI-centric).
    if slice_pad and idx:
        pad = int(slice_pad)
        if pad > 0 and len(idx) < max_slices:
            extra: set[int] = set()
            for z in idx:
                for dz in range(-pad, pad + 1):
                    zi = int(z) + int(dz)
                    if 0 <= zi < int(v.shape[0]):
                        extra.add(int(zi))
            idx = sorted(set(idx) | extra)
    if len(idx) > max_slices:
        # even sampling across full ROI range to avoid missing apex/base slices
        picks = np.linspace(0, len(idx) - 1, num=max_slices)
        picked: List[int] = []
        seen: set[int] = set()
        for p in picks:
            zi = idx[int(round(p))]
            if zi in seen:
                continue
            seen.add(zi)
            picked.append(int(zi))
        if len(picked) < max_slices:
            for zi in idx:
                if zi in seen:
                    continue
                seen.add(zi)
                picked.append(int(zi))
                if len(picked) >= max_slices:
                    break
        idx = picked

    # Target tile size for VLM-friendly payloads (keeps montage around ~640x640 for 4x4)
    target_tile = 160

    # Prefer Pillow for RGB montage with colored outlines.
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        Image = None  # type: ignore

    tile_imgs = []
    tile_map: List[Dict[str, Any]] = []
    for z in idx:
        mask2d = wg[z].astype("uint8")
        if int(mask2d.max()) == 0:
            continue
        crop_full = v[z].astype("float32", copy=False)
        crop_lab_full = m[z].astype("uint8", copy=False)

        ctx2d = None
        if ctx is not None:
            ctx2d = (ctx[z] > 0).astype("uint8", copy=False)
            if int(ctx2d.max()) == 0:
                ctx2d = None
        if full_frame:
            crop = crop_full
            crop_mask = (mask2d > 0).astype("uint8", copy=False)
            crop_lab = crop_lab_full
            roi_crop = (crop_lab > 0).astype("uint8")
            ctx_crop = ctx2d if ctx2d is not None else None
            win_mask = np.ones_like(crop_mask)
        else:
            bbox_mask = ctx2d if ctx2d is not None else mask2d
            crop, crop_mask, bbox = _crop_slice_bbox(slice_f32=crop_full, mask2d_u8=bbox_mask, np=np, pad=pad)
            if crop is None or crop_mask is None or bbox is None:
                continue
            x0, x1, y0, y1 = bbox
            crop_lab = crop_lab_full[y0 : y1 + 1, x0 : x1 + 1]
            roi_crop = (crop_lab > 0).astype("uint8")
            ctx_crop = None
            if ctx2d is not None:
                ctx_crop = ctx2d[y0 : y1 + 1, x0 : x1 + 1].astype("uint8", copy=False)
            win_mask = roi_crop if roi_crop is not None else crop_mask
        base_u8 = _robust_window_uint8_masked(crop, win_mask, np)

        # Build RGB tile and draw zone outlines if labels exist (1=CG, 2=PZ).
        if Image is not None:
            rgb = np.stack([base_u8, base_u8, base_u8], axis=-1)
            # outlines
            cg = (crop_lab == 1).astype("uint8")
            pz = (crop_lab == 2).astype("uint8")
            if ctx_crop is not None:
                ctx_edge = (ctx_crop > 0) & (_binary_erosion_3x3(ctx_crop, np) == 0)
                rgb[ctx_edge] = (0, 255, 255)  # cyan = context
                roi_edge = (roi_crop > 0) & (_binary_erosion_3x3(roi_crop, np) == 0)
                rgb[roi_edge] = (255, 0, 0)  # red = ROI
            else:
                # If mask is binary, cg/pz will be empty and only grayscale will show (legend still present).
                if int(cg.max()) > 0:
                    cg_edge = (cg > 0) & (_binary_erosion_3x3(cg, np) == 0)
                    rgb[cg_edge] = (255, 0, 0)  # red = Central gland
                if int(pz.max()) > 0:
                    pz_edge = (pz > 0) & (_binary_erosion_3x3(pz, np) == 0)
                    rgb[pz_edge] = (0, 255, 255)  # cyan = Peripheral zone
                if int(cg.max()) == 0 and int(pz.max()) == 0:
                    edge = (crop_mask > 0) & (_binary_erosion_3x3(crop_mask, np) == 0)
                    rgb[edge] = (255, 0, 0)

            im = Image.fromarray(rgb, mode="RGB")
            im = im.resize((int(target_tile), int(target_tile)))
            # Legend (small, repeated per tile to help VLM)
            draw = ImageDraw.Draw(im)
            font = ImageFont.load_default()
            if ctx_crop is not None:
                draw.rectangle([2, 2, 10, 10], fill=(255, 0, 0))
                draw.text((12, 1), "ROI", fill=(255, 255, 255), font=font)
                draw.rectangle([2, 14, 10, 22], fill=(0, 255, 255))
                draw.text((12, 13), "CTX", fill=(255, 255, 255), font=font)
            else:
                if roi_label:
                    draw.text((2, 2), _roi_short_label(roi_label), fill=(255, 255, 255), font=font)
                else:
                    draw.rectangle([2, 2, 10, 10], fill=(255, 0, 0))
                    draw.text((12, 1), "CG", fill=(255, 255, 255), font=font)
                    draw.rectangle([2, 14, 10, 22], fill=(0, 255, 255))
                    draw.text((12, 13), "PZ", fill=(255, 255, 255), font=font)
            tile_imgs.append(im)
        else:
            # Fallback: grayscale tile via SimpleITK
            base_img = sitk.GetImageFromArray(base_u8)  # 2D uint8
            base_img = _resize_2d(base_img, (target_tile, target_tile), sitk)
            # Convert to PNG bytes by writing later via sitk.Tile path (not used when PIL missing)
            tile_imgs.append(base_img)  # type: ignore

        tile_map.append({"z": int(z), "has_roi": True})

    # pad to full grid (fixed size for VLM)
    n_target = int(grid_cols * grid_rows)
    if len(tile_imgs) < n_target:
        if Image is not None:
            blank = Image.new("RGB", (int(target_tile), int(target_tile)), (0, 0, 0))
        else:
            blank = sitk.Image(tile_imgs[0].GetSize(), tile_imgs[0].GetPixelID())  # type: ignore
        for _ in range(n_target - len(tile_imgs)):
            tile_imgs.append(blank)  # type: ignore
            tile_map.append({"z": None, "has_roi": False})
    else:
        tile_imgs = tile_imgs[:n_target]
        tile_map = tile_map[:n_target]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if Image is not None:
        # Compose montage in RGB
        W = int(grid_cols) * int(target_tile)
        H = int(grid_rows) * int(target_tile)
        montage = Image.new("RGB", (W, H), (0, 0, 0))
        for i in range(n_target):
            r = i // int(grid_cols)
            c = i % int(grid_cols)
            montage.paste(tile_imgs[i], (int(c * target_tile), int(r * target_tile)))

        # Add per-tile labels directly onto the montage PNG so the VLM can cite specific tiles.
        draw = ImageDraw.Draw(montage)
        font = ImageFont.load_default()
        for i in range(n_target):
            r = i // int(grid_cols)
            c = i % int(grid_cols)
            z = tile_map[i].get("z")
            label = f"r{r} c{c} z{z}" if z is not None else f"r{r} c{c} blank"
            x0 = int(c * target_tile + 2)
            y0 = int(r * target_tile + 2)
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            draw.rectangle([x0 - 1, y0 - 1, x0 + tw + 3, y0 + th + 3], fill=(0, 0, 0))
            draw.text((x0 + 1, y0 + 1), label, fill=(255, 255, 255), font=font)
        montage.save(str(out_path))
    else:
        # Fallback grayscale montage
        tiled = sitk.Tile(tile_imgs, [int(grid_cols), int(grid_rows)])  # type: ignore
        sitk.WriteImage(tiled, str(out_path))

    # Write tile index JSON next to the montage so text-only models/humans can map tiles -> z.
    try:
        tiles_path = out_path.with_name(out_path.name.replace(".png", "_tiles.json"))
        enriched = []
        for i in range(n_target):
            r = i // int(grid_cols)
            c = i % int(grid_cols)
            enriched.append({"tile_index": int(i), "row": int(r), "col": int(c), **tile_map[i]})
        tiles_path.write_text(json.dumps({"grid": {"rows": int(grid_rows), "cols": int(grid_cols)}, "tiles": enriched}, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass
    return True


def extract_roi_features(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    roi_mask_path = args.get("roi_mask_path")
    roi_masks_arg = args.get("roi_masks")
    images = args.get("images", [])
    context_mask_path = args.get("context_mask_path")
    roi_interpretation = str(args.get("roi_interpretation") or "prostate").strip().lower()
    cardiac_mode = roi_interpretation == "cardiac"
    overlay_mode = str(args.get("overlay_mode") or "").strip().lower()
    if overlay_mode not in ("roi_crop", "full_frame", "both"):
        overlay_mode = ""
    if not overlay_mode:
        overlay_mode = "full_frame" if roi_interpretation in ("prostate", "cardiac") else "roi_crop"
    if cardiac_mode and overlay_mode != "full_frame":
        overlay_mode = "full_frame"
        warnings = ["ROI zoom/crop overlays are disabled for cardiac mode; using full_frame overlays only."]
    else:
        warnings: List[str] = []
    use_roi_crop = overlay_mode in ("roi_crop", "both")
    use_full_frame = overlay_mode in ("full_frame", "both")

    out_subdir = str(args.get("output_subdir", "features") or "features")
    lesion_redirected = False
    # Avoid overwriting whole-gland features with lesion-only runs.
    roi_masks_arg = args.get("roi_masks")
    if out_subdir == "features" and isinstance(roi_masks_arg, list) and len(roi_masks_arg) == 1:
        only = roi_masks_arg[0]
        if isinstance(only, dict) and str(only.get("name", "")).strip().lower() == "lesion":
            out_subdir = "features_lesion"
            lesion_redirected = True
            args = dict(args)
            args["output_subdir"] = out_subdir
            # Note: warning appended later once warnings list exists.
    out_dir = ctx.artifacts_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    feature_table_path = out_dir / "features.csv"
    slice_summary_path = out_dir / "slice_summary.json"

    sitk, np = _require_sitk_np()
    radiomics_cfg = args.get("radiomics")
    radiomics_mode = args.get("radiomics_mode")
    radiomics_feature_classes = args.get("radiomics_feature_classes")
    radiomics_image_types = args.get("radiomics_image_types")
    radiomics_settings = args.get("radiomics_settings")
    if isinstance(radiomics_cfg, dict):
        if "mode" in radiomics_cfg:
            radiomics_mode = radiomics_cfg.get("mode")
        elif "enabled" in radiomics_cfg:
            radiomics_mode = radiomics_cfg.get("enabled")
        if "feature_classes" in radiomics_cfg:
            radiomics_feature_classes = radiomics_cfg.get("feature_classes")
        if "image_types" in radiomics_cfg:
            radiomics_image_types = radiomics_cfg.get("image_types")
        if "settings" in radiomics_cfg:
            radiomics_settings = radiomics_cfg.get("settings")
    elif isinstance(radiomics_cfg, (str, bool)):
        radiomics_mode = radiomics_cfg

    # Default to OFF unless explicitly enabled via args.
    # Auto-enable only for lesion workflows.
    lesion_present_hint = False
    try:
        roi_masks_hint = roi_masks_arg if isinstance(roi_masks_arg, list) else []
        lesion_present_hint = any(
            isinstance(r, dict) and "lesion" in str(r.get("name") or "").lower() for r in roi_masks_hint
        )
    except Exception:
        lesion_present_hint = False
    if (
        radiomics_mode is None
        and radiomics_cfg is None
        and radiomics_feature_classes is None
        and radiomics_image_types is None
        and radiomics_settings is None
    ):
        radiomics_mode = "on" if lesion_present_hint else "off"

    radiomics_mode = _normalize_radiomics_mode(radiomics_mode)
    if isinstance(radiomics_settings, str):
        try:
            radiomics_settings = json.loads(radiomics_settings)
        except Exception:
            radiomics_settings = {}
    if not isinstance(radiomics_settings, dict):
        radiomics_settings = {}
    default_feature_classes = (
        list(DEFAULT_RADIOMICS_CLASSES)
        if cardiac_mode
        else list(LESION_RADIOMICS_FEATURES.keys())
    )
    feature_classes = _normalize_str_list(radiomics_feature_classes) or default_feature_classes
    image_types = radiomics_image_types
    feature_names_subset = None if cardiac_mode else LESION_RADIOMICS_FEATURES
    roi_scope = "all_rois" if cardiac_mode else "lesion_only"
    radiomics_meta = {
        "mode": radiomics_mode,
        "enabled": False,
        "feature_classes": feature_classes,
        "image_types": image_types if image_types is not None else ["Original"],
        "settings": radiomics_settings,
        "roi_scope": roi_scope,
    }
    radiomics_extractor, rad_meta = _build_radiomics_extractor(
        mode=radiomics_mode,
        feature_classes=feature_classes,
        image_types=image_types,
        settings=radiomics_settings,
        warnings=warnings,
        feature_names=feature_names_subset,
    )
    if isinstance(rad_meta, dict):
        radiomics_meta.update(rad_meta)
    radiomics_meta = _jsonable(radiomics_meta, np)
    context_mask_img = None
    if isinstance(context_mask_path, str) and context_mask_path:
        try:
            ctx_path = Path(context_mask_path).expanduser().resolve()
            if ctx_path.exists():
                context_mask_img = sitk.ReadImage(str(ctx_path))
            else:
                context_mask_img = None
                warnings.append(f"context_mask_path not found: {ctx_path}")
        except Exception:
            context_mask_img = None
            warnings.append("Failed to read context_mask_path; ignoring.")
    if context_mask_img is not None and context_mask_img.GetDimension() == 4:
        context_mask_img = _collapse_4d_mask(context_mask_img, sitk=sitk, np=np, warnings=warnings)

    roi_specs: List[Dict[str, str]] = []
    if isinstance(roi_masks_arg, list) and roi_masks_arg:
        for i, r in enumerate(roi_masks_arg):
            if isinstance(r, dict):
                name = str(r.get("name") or f"roi{i}")
                path = str(r.get("path") or "")
                if path:
                    roi_specs.append({"name": name, "path": path})
            elif isinstance(r, str):
                roi_specs.append({"name": f"roi{i}", "path": str(r)})
    if not roi_specs and roi_mask_path:
        fallback_name = "prostate" if roi_interpretation == "prostate" else Path(str(roi_mask_path)).stem
        roi_specs = [{"name": fallback_name, "path": str(roi_mask_path)}]
    if not roi_specs:
        raise ValueError("extract_roi_features requires roi_mask_path or roi_masks")

    if context_mask_img is None:
        try:
            if any("lesion" in str(s.get("name") or "").lower() for s in roi_specs):
                cand = ctx.artifacts_dir / "segmentation" / "prostate_whole_gland_mask.nii.gz"
                if cand.exists():
                    context_mask_img = sitk.ReadImage(str(cand))
        except Exception:
            context_mask_img = None

    rows: List[Dict[str, Any]] = []
    slice_summary: Dict[str, Any] = {"per_sequence": {}, "per_roi": {}}
    if radiomics_meta:
        slice_summary["radiomics"] = radiomics_meta
    overlay_pngs: List[Dict[str, str]] = []
    roi_summaries: List[Dict[str, Any]] = []
    radiomics_rows = 0
    radiomics_error_samples: List[str] = []
    radiomics_feature_names: set[str] = set()
    if lesion_redirected:
        warnings.append("Lesion ROI detected with output_subdir=features; redirected to features_lesion to preserve whole-gland stats.")

    # Optional reference slices to align montages across multiple feature runs.
    reference_slices: List[int] = []
    ref_raw = args.get("reference_slices")
    if isinstance(ref_raw, list):
        for z in ref_raw:
            try:
                reference_slices.append(int(z))
            except Exception:
                continue
    # Prefer lesion-derived reference slices when available to keep montages consistent.
    lesion_ref_slices: List[int] = []
    lesion_mask_path: Path | None = None
    try:
        for spec in roi_specs:
            if "lesion" in str(spec.get("name") or "").lower():
                lesion_mask_path = Path(str(spec.get("path") or "")).expanduser().resolve()
                break
        if lesion_mask_path is None:
            cand = ctx.artifacts_dir / "lesion" / "lesion_mask.nii.gz"
            if cand.exists():
                lesion_mask_path = cand
    except Exception:
        lesion_mask_path = None
    if lesion_mask_path is not None and lesion_mask_path.exists():
        try:
            lesion_mask_img = sitk.ReadImage(str(lesion_mask_path))
            if lesion_mask_img.GetDimension() == 4:
                lesion_mask_img = _collapse_4d_mask(lesion_mask_img, sitk=sitk, np=np, warnings=warnings)
            lesion_arr = sitk.GetArrayViewFromImage(lesion_mask_img)
            lesion_ref_slices = _roi_slice_indices(lesion_arr, np)
        except Exception:
            lesion_ref_slices = []
    ref_summary_path = args.get("reference_slice_summary_path")
    if (not ref_summary_path) and out_subdir == "features":
        # If lesion features exist, align whole-gland overlays to lesion slices for consistency.
        candidate = ctx.artifacts_dir / "features_lesion" / "slice_summary.json"
        if candidate.exists():
            ref_summary_path = str(candidate)
    if (not ref_summary_path) and out_subdir == "features_lesion":
        # If whole-gland features exist, align lesion overlays to those slices for consistency.
        candidate = ctx.artifacts_dir / "features" / "slice_summary.json"
        if candidate.exists():
            ref_summary_path = str(candidate)
    if (not reference_slices) and isinstance(ref_summary_path, str) and ref_summary_path:
        try:
            ref_obj = json.loads(Path(ref_summary_path).expanduser().resolve().read_text(encoding="utf-8"))
            per_seq = ref_obj.get("per_sequence") if isinstance(ref_obj, dict) else None
            if isinstance(per_seq, dict):
                zs: set[int] = set()
                for v in per_seq.values():
                    per_slice = v.get("per_slice") if isinstance(v, dict) else None
                    if not isinstance(per_slice, list):
                        continue
                    for it in per_slice:
                        if isinstance(it, dict) and "z" in it:
                            try:
                                zs.add(int(it.get("z")))
                            except Exception:
                                continue
                reference_slices = sorted(zs)
        except Exception:
            reference_slices = []
    if lesion_ref_slices:
        reference_slices = sorted(set(lesion_ref_slices))
        slice_summary["reference_slices_source"] = "lesion_mask"
    if reference_slices:
        slice_summary["reference_slices_requested"] = reference_slices[:64]
        if isinstance(ref_summary_path, str) and ref_summary_path:
            slice_summary["reference_slice_summary_path"] = str(ref_summary_path)

    # For brain binary masks, avoid ROI zoom-in by using a huge pad (full-frame crop).
    brain_mode = roi_interpretation in ("binary_masks", "brain_binary_masks")
    crop_pad = 10000 if brain_mode else 18

    # Be tolerant: LLM sometimes passes images as list[str] instead of list[{"name","path"}]
    def _canonical_seq_name(name: str, path: str) -> str:
        label = str(name or "").strip()
        low = label.lower()
        if not label or low.startswith("img") or low.startswith("image"):
            try:
                p = Path(path)
                label = p.name
            except Exception:
                label = str(path)
        # strip extensions
        if label.endswith(".nii.gz"):
            label = label[:-7]
        elif label.endswith(".nii"):
            label = label[:-4]
        low = label.lower()
        if "molli" in low:
            return "T1_MOLLI"
        if low in ("t1", "t1_map", "t1mapping", "t1_mapping"):
            return "T1"
        if "t1" in low:
            return "T1"
        if "adc" in low:
            return "ADC"
        if "t2" in low:
            return "T2w"
        if "high" in low and "b" in low:
            return "DWI_high_b"
        if "low" in low and "b" in low:
            return "DWI_low_b"
        if "mid" in low and "b" in low:
            return "DWI_mid_b"
        if "dwi" in low:
            return "DWI"
        return label.replace(" ", "_")

    norm_images: List[Dict[str, str]] = []
    if isinstance(images, list):
        for idx, img in enumerate(images):
            if isinstance(img, str):
                norm_images.append({"name": f"img{idx}", "path": img})
            elif isinstance(img, dict):
                norm_images.append(
                    {
                        "name": str(img.get("name", f"img{idx}")),
                        "path": str(img.get("path", "")),
                    }
                )
            else:
                warnings.append(f"Skipping unsupported images[{idx}] type: {type(img).__name__}")
    else:
        warnings.append("images is not a list; no features extracted.")
        norm_images = []

    # Canonicalize names, drop duplicate paths, and avoid redundant generic DWI.
    canon_images: List[Dict[str, str]] = []
    for it in norm_images:
        p = str(it.get("path") or "").strip()
        if not p:
            continue
        n = _canonical_seq_name(str(it.get("name") or ""), p)
        canon_images.append({"name": n, "path": p})
    has_high_b = any(str(it.get("name")) == "DWI_high_b" for it in canon_images)
    if has_high_b:
        canon_images = [it for it in canon_images if str(it.get("name")) != "DWI"]
    dedup: Dict[str, Dict[str, str]] = {}
    for it in canon_images:
        p = str(it.get("path") or "")
        try:
            rp = str(Path(p).expanduser().resolve())
        except Exception:
            rp = p
        key = f"{it.get('name','')}::{rp}"
        if key not in dedup:
            dedup[key] = {"name": str(it.get("name") or ""), "path": rp}
    norm_images = list(dedup.values())

    for roi_spec in roi_specs:
        roi_name = str(roi_spec.get("name") or "roi")
        roi_path = str(roi_spec.get("path") or "")
        roi_path_p = Path(roi_path).expanduser().resolve()
        if not roi_path_p.exists():
            warnings.append(f"Skipping missing ROI mask '{roi_name}': {roi_path_p}")
            roi_summaries.append({"roi_source": roi_name, "ok": False, "reason": "missing_mask", "n_voxels": 0})
            continue

        mask_img = sitk.ReadImage(str(roi_path_p))
        if mask_img.GetDimension() == 4:
            mask_img = _collapse_4d_mask(mask_img, sitk=sitk, np=np, warnings=warnings)
        mask_arr = sitk.GetArrayViewFromImage(mask_img)
        if int(np.count_nonzero(mask_arr)) == 0:
            warnings.append(f"ROI mask '{roi_name}' is empty; skipping feature extraction for this ROI.")
            roi_summaries.append({"roi_source": roi_name, "ok": False, "reason": "empty_mask", "n_voxels": 0})
            continue

        uniq = set(np.unique(mask_arr).tolist())
        has_zones = (1 in uniq) and (2 in uniq) and roi_interpretation == "prostate"
        ref_in_bounds: List[int] = []
        if reference_slices:
            max_z = int(mask_arr.shape[0])
            ref_in_bounds = sorted({int(z) for z in reference_slices if 0 <= int(z) < max_z})
            if ref_in_bounds and "reference_slices_used" not in slice_summary:
                slice_summary["reference_slices_used"] = ref_in_bounds[:64]

        if roi_interpretation in ("binary_masks", "brain_binary_masks", "cardiac"):
            roi_defs = [(roi_name, lambda m: m > 0)]
        else:
            if "lesion" in roi_name.lower():
                roi_defs = [("lesion", lambda m: m > 0)]
            else:
                roi_defs = [
                    ("whole_gland", lambda m: m > 0),
                ]
                if has_zones:
                    roi_defs.extend(
                        [
                            ("central_gland", lambda m: m == 1),
                            ("peripheral_zone", lambda m: m == 2),
                        ]
                    )

        roi_summaries.append(
            {"roi_source": roi_name, "ok": True, "n_voxels": int(np.count_nonzero(mask_arr))}
        )

        for img in norm_images:
            seq_name_raw = str(img.get("name", "UNKNOWN"))
            img_path = str(img.get("path", ""))
            seq_name = _canonical_seq_name(seq_name_raw, img_path)
            img_path_p = Path(img_path).expanduser().resolve()

            # Only support NIfTI for real measurements in MVP
            if img_path_p.suffix not in (".gz", ".nii") and not img_path_p.name.endswith(".nii.gz"):
                warnings.append(f"Skipping non-NIfTI image '{seq_name}': {img_path_p} (provide a .nii/.nii.gz path).")
                continue
            if not img_path_p.exists():
                warnings.append(f"Skipping missing image '{seq_name}': {img_path_p}")
                continue

            vol = sitk.ReadImage(str(img_path_p))
            vol_items = _split_4d_image(vol, base_name=seq_name, path=str(img_path_p), sitk=sitk, warnings=warnings)
            for vol_item in vol_items:
                seq_name_item = str(vol_item.get("name") or seq_name)
                vol_item_img = vol_item.get("vol")
                if vol_item_img is None:
                    continue
                mask_res = _resample_mask_to_image(mask_img, vol_item_img)
                ctx_res = None
                if context_mask_img is not None:
                    try:
                        ctx_res = _resample_mask_to_image(context_mask_img, vol_item_img)
                    except Exception:
                        ctx_res = None
                m = sitk.GetArrayViewFromImage(mask_res)
                v = sitk.GetArrayViewFromImage(vol_item_img).astype("float32", copy=False)

                sp = vol_item_img.GetSpacing()  # (x,y,z)
                voxel_vol_mm3 = float(sp[0] * sp[1] * sp[2])

                wg = (m > 0)
                per_slice = []
                zs = list(range(int(v.shape[0])))
                if ref_in_bounds:
                    zs = [int(z) for z in ref_in_bounds if 0 <= int(z) < int(v.shape[0])]
                    if not zs:
                        zs = list(range(int(v.shape[0])))
                for zi in zs:
                    vals = v[zi][wg[zi]]
                    if vals.size == 0:
                        continue
                    per_slice.append(
                        {
                            "z": int(zi),
                            "mean": float(vals.mean()),
                            "p10": float(np.percentile(vals, 10)),
                            "p90": float(np.percentile(vals, 90)),
                        }
                    )
                # Keep legacy per_sequence for the first ROI source (best compatibility)
                if roi_specs and roi_name == roi_specs[0].get("name"):
                    seq_entry = {"n_slices_with_roi": len(per_slice), "per_slice": per_slice[:64]}
                    if ref_in_bounds:
                        seq_entry["reference_slices_used"] = ref_in_bounds[:64]
                    slice_summary["per_sequence"][seq_name_item] = seq_entry

                slice_summary.setdefault("per_roi", {}).setdefault(roi_name, {"per_sequence": {}})
                seq_entry_roi = {"n_slices_with_roi": len(per_slice), "per_slice": per_slice[:64]}
                if ref_in_bounds:
                    seq_entry_roi["reference_slices_used"] = ref_in_bounds[:64]
                slice_summary["per_roi"][roi_name]["per_sequence"][seq_name_item] = seq_entry_roi

                for roi_subname, roi_fn in roi_defs:
                    sel = roi_fn(m)
                    vals = v[sel]
                    nvox = int(vals.size)
                    vol_ml = float(nvox * voxel_vol_mm3 / 1000.0)
                    st = _stats(vals, np)
                    rad_feats: Dict[str, Any] = {}
                    radiomics_for_roi = bool(radiomics_meta.get("roi_scope") == "all_rois") or ("lesion" in roi_name.lower())
                    if radiomics_extractor is not None and nvox > 0 and radiomics_for_roi:
                        try:
                            sub_mask_arr = sel.astype("uint8", copy=False)
                            if int(np.count_nonzero(sub_mask_arr)) > 0:
                                sub_mask_img = sitk.GetImageFromArray(sub_mask_arr)
                                sub_mask_img.CopyInformation(mask_res)
                                rad_raw = radiomics_extractor.execute(vol_item_img, sub_mask_img, label=1)
                                rad_feats = _sanitize_radiomics_output(rad_raw, np)
                                if rad_feats:
                                    radiomics_feature_names.update(rad_feats.keys())
                                    radiomics_rows += 1
                        except Exception as e:
                            if len(radiomics_error_samples) < 5:
                                radiomics_error_samples.append(
                                    f"{roi_name}/{roi_subname}/{seq_name_item}: {type(e).__name__}"
                                )
                    rows.append(
                        {
                            "roi_source": roi_name,
                            "roi": roi_subname,
                            "sequence": seq_name_item,
                            "n_voxels": nvox,
                            "volume_ml": vol_ml,
                            **st,
                            **rad_feats,
                        }
                    )

                # Quick-look overlay for QC (include brain Whole_Tumor for VLM guidance)
                if (not brain_mode) or roi_name == "Whole_Tumor":
                    if use_roi_crop:
                        try:
                            out_png = overlay_dir / f"{seq_name_item}_{roi_name}_roi_crop_best.png"
                            _save_roi_crop_best_slice_png(
                                vol=vol_item_img,
                                mask_res=mask_res,
                                out_path=out_png,
                                reference_slices=ref_in_bounds or None,
                                context_mask_res=ctx_res,
                                pad=crop_pad,
                                full_frame=brain_mode,
                                roi_label=roi_name,
                            )
                            overlay_pngs.append(
                                {
                                    "sequence": seq_name_item,
                                    "roi_source": roi_name,
                                    "kind": "roi_crop_best_slice",
                                    "path": str(out_png),
                                }
                            )
                        except Exception:
                            pass
                    if use_full_frame:
                        try:
                            out_png = overlay_dir / f"{seq_name_item}_{roi_name}_full_frame_best.png"
                            _save_roi_crop_best_slice_png(
                                vol=vol_item_img,
                                mask_res=mask_res,
                                out_path=out_png,
                                reference_slices=ref_in_bounds or None,
                                context_mask_res=ctx_res,
                                pad=crop_pad,
                                full_frame=True,
                                roi_label=roi_name,
                            )
                            overlay_pngs.append(
                                {
                                    "sequence": seq_name_item,
                                    "roi_source": roi_name,
                                    "kind": "full_frame_best_slice",
                                    "path": str(out_png),
                                }
                            )
                        except Exception:
                            pass

            # Montage covering ROI slices (best-effort)
            try:
                if (not brain_mode) or roi_name == "Whole_Tumor":
                    if use_roi_crop:
                        out_montage = overlay_dir / f"{seq_name}_{roi_name}_roi_crop_montage_4x4.png"
                        if _save_roi_crop_montage_png(
                            vol=vol,
                            mask_res=mask_res,
                            out_path=out_montage,
                            max_slices=16,
                            grid_cols=4,
                            grid_rows=4,
                            reference_slices=ref_in_bounds or None,
                            context_mask_res=ctx_res,
                            slice_pad=2,
                            pad=crop_pad,
                            full_frame=brain_mode,
                            roi_label=roi_name,
                        ):
                            overlay_pngs.append(
                                {
                                    "sequence": seq_name,
                                    "roi_source": roi_name,
                                    "kind": "roi_crop_montage_4x4",
                                    "path": str(out_montage),
                                }
                            )
                            tiles_json = str(out_montage).replace(".png", "_tiles.json")
                            if Path(tiles_json).exists():
                                overlay_pngs.append(
                                    {
                                        "sequence": seq_name,
                                        "roi_source": roi_name,
                                        "kind": "roi_crop_montage_4x4_tiles",
                                        "path": tiles_json,
                                    }
                                )
                    if use_full_frame:
                        out_montage = overlay_dir / f"{seq_name}_{roi_name}_full_frame_montage_4x4.png"
                        if _save_roi_crop_montage_png(
                            vol=vol,
                            mask_res=mask_res,
                            out_path=out_montage,
                            max_slices=16,
                            grid_cols=4,
                            grid_rows=4,
                            reference_slices=ref_in_bounds or None,
                            context_mask_res=ctx_res,
                            slice_pad=2,
                            pad=crop_pad,
                            full_frame=True,
                            roi_label=roi_name,
                        ):
                            overlay_pngs.append(
                                {
                                    "sequence": seq_name,
                                    "roi_source": roi_name,
                                    "kind": "full_frame_montage_4x4",
                                    "path": str(out_montage),
                                }
                            )
                            tiles_json = str(out_montage).replace(".png", "_tiles.json")
                            if Path(tiles_json).exists():
                                overlay_pngs.append(
                                    {
                                        "sequence": seq_name,
                                        "roi_source": roi_name,
                                        "kind": "full_frame_montage_4x4_tiles",
                                        "path": tiles_json,
                                    }
                                )
            except Exception:
                pass

    if radiomics_meta:
        radiomics_meta["rows_with_features"] = int(radiomics_rows)
        if radiomics_error_samples:
            radiomics_meta["error_samples"] = radiomics_error_samples[:5]
        slice_summary["radiomics"] = radiomics_meta
        if radiomics_meta.get("enabled") and radiomics_rows == 0:
            warnings.append("Radiomics enabled but no features extracted; check ROI masks and image inputs.")
        if radiomics_error_samples:
            warnings.append(
                "Radiomics extraction errors (sampled): " + "; ".join(radiomics_error_samples[:5])
            )
        try:
            if radiomics_meta.get("enabled"):
                summary_path = out_dir / "radiomics_summary.json"
                summary = {
                    "mode": radiomics_meta.get("mode"),
                    "enabled": radiomics_meta.get("enabled"),
                    "feature_classes": radiomics_meta.get("feature_classes"),
                    "image_types": radiomics_meta.get("image_types"),
                    "settings": radiomics_meta.get("settings"),
                    "rows_with_features": int(radiomics_rows),
                    "feature_count": int(len(radiomics_feature_names)),
                    "feature_names_sample": sorted(radiomics_feature_names)[:50],
                }
                summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    skip_overwrite = (
        len(roi_specs) == 1
        and str(roi_specs[0].get("name", "")).strip().lower() == "lesion"
        and not rows
        and any(s.get("reason") == "empty_mask" for s in roi_summaries)
        and feature_table_path.exists()
        and slice_summary_path.exists()
    )
    if brain_mode:
        brain_pngs = [
            o
            for o in overlay_pngs
            if str(o.get("roi_source") or "") == "Whole_Tumor"
            and str(o.get("kind") or "") in {
                "roi_crop_montage_4x4",
                "roi_crop_best_slice",
                "full_frame_montage_4x4",
                "full_frame_best_slice",
            }
        ]
        brain_tiles = [
            o
            for o in overlay_pngs
            if str(o.get("roi_source") or "") == "Whole_Tumor"
            and str(o.get("kind") or "") in {"roi_crop_montage_4x4_tiles", "full_frame_montage_4x4_tiles"}
        ]
        if len(brain_pngs) > 8:
            brain_pngs = brain_pngs[:8]
        overlay_pngs = brain_pngs + brain_tiles
    if skip_overwrite:
        warnings.append("Lesion-only ROI mask empty; preserving existing features outputs.")
        artifacts = [
            ArtifactRef(path=str(feature_table_path), kind="csv", description="ROI x sequence feature table"),
            ArtifactRef(path=str(slice_summary_path), kind="json", description="Slice-wise summary"),
        ]
        md_path = feature_table_path.with_suffix(".md")
        if md_path.exists():
            artifacts.append(ArtifactRef(path=str(md_path), kind="md", description="ROI x sequence feature table (markdown)"))
        return {
            "data": {
                "feature_table_path": str(feature_table_path),
                "feature_table_md_path": str(md_path) if md_path.exists() else None,
                "slice_summary_path": str(slice_summary_path),
                "overlay_pngs": overlay_pngs,
                "roi_sources": [r.get("name") for r in roi_specs],
                "roi_summaries": roi_summaries,
            },
            "artifacts": artifacts,
            "warnings": warnings,
            "source_artifacts": [
                ArtifactRef(path=str(r.get("path")), kind="ref", description=f"ROI mask: {r.get('name')}") for r in roi_specs
            ]
            + [ArtifactRef(path=str(i.get("path", "")), kind="ref", description=f"Image: {i.get('name','')}") for i in norm_images],
            "generated_artifacts": artifacts,
        }

    base_fields = ["roi_source", "roi", "sequence", "n_voxels", "volume_ml", "min", "max", "mean", "median", "std", "p10", "p90"]
    if rows:
        all_fields: set = set()
        for r in rows:
            all_fields.update(r.keys())
        extra_fields = sorted(k for k in all_fields if k not in base_fields)
        fieldnames = base_fields + extra_fields
    else:
        fieldnames = base_fields

    def _format_cell(val: Any) -> Any:
        try:
            if isinstance(val, np.generic):
                val = val.item()
        except Exception:
            pass
        if isinstance(val, float):
            return f"{val:.2f}"
        if isinstance(val, int):
            return val
        if val is None:
            return ""
        return val

    formatted_rows: List[Dict[str, Any]] = []
    for r in rows:
        out_r: Dict[str, Any] = {}
        for k in fieldnames:
            out_r[k] = _format_cell(r.get(k, ""))
        formatted_rows.append(out_r)

    with feature_table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in formatted_rows:
            writer.writerow(r)

    # Also write a markdown table for LLM readability.
    feature_table_md_path = out_dir / "features.md"
    try:
        with feature_table_md_path.open("w", encoding="utf-8") as f:
            header = " | ".join(fieldnames)
            sep = " | ".join(["---"] * len(fieldnames))
            f.write(f"| {header} |\n")
            f.write(f"| {sep} |\n")
            for r in formatted_rows:
                row_vals = []
                for k in fieldnames:
                    v = r.get(k, "")
                    s = str(v).replace("|", "\\|")
                    row_vals.append(s)
                f.write(f"| {' | '.join(row_vals)} |\n")
    except Exception:
        feature_table_md_path = None

    slice_summary_path.write_text(json.dumps(slice_summary, indent=2) + "\n", encoding="utf-8")

    texture_viz = (
        _build_texture_visualization(rows=rows, out_dir=out_dir, heatmap_dir=overlay_dir)
        if rows
        else {"summary_csv_path": None, "heatmap_png_path": None}
    )
    texture_summary_csv = texture_viz.get("summary_csv_path") if isinstance(texture_viz, dict) else None
    texture_heatmap_png = texture_viz.get("heatmap_png_path") if isinstance(texture_viz, dict) else None
    wrote_texture_paths = False
    if isinstance(texture_summary_csv, str) and texture_summary_csv:
        slice_summary["texture_radiomics_summary_path"] = str(texture_summary_csv)
        wrote_texture_paths = True
    if isinstance(texture_heatmap_png, str) and texture_heatmap_png:
        slice_summary["texture_radiomics_heatmap_path"] = str(texture_heatmap_png)
        wrote_texture_paths = True
    if wrote_texture_paths:
        slice_summary_path.write_text(json.dumps(slice_summary, indent=2) + "\n", encoding="utf-8")

    artifacts = [
        ArtifactRef(path=str(feature_table_path), kind="csv", description="ROI x sequence feature table"),
        ArtifactRef(path=str(slice_summary_path), kind="json", description="Slice-wise summary"),
    ]
    if feature_table_md_path:
        artifacts.append(ArtifactRef(path=str(feature_table_md_path), kind="md", description="ROI x sequence feature table (markdown)"))
    if isinstance(texture_summary_csv, str) and texture_summary_csv:
        artifacts.append(ArtifactRef(path=str(texture_summary_csv), kind="csv", description="Texture radiomics summary"))
    if isinstance(texture_heatmap_png, str) and texture_heatmap_png:
        overlay_pngs.append(
            {
                "sequence": "RADIOMICS",
                "roi_source": "multi_roi",
                "kind": "texture_radiomics_heatmap",
                "path": str(texture_heatmap_png),
            }
        )
        artifacts.append(ArtifactRef(path=str(texture_heatmap_png), kind="image", description="Texture radiomics heatmap"))
    for o in overlay_pngs:
        artifacts.append(ArtifactRef(path=o["path"], kind="image", description=f"ROI crop montage: {o.get('sequence','')}"))

    return {
        "data": {
            "feature_table_path": str(feature_table_path),
            "feature_table_md_path": str(feature_table_md_path) if feature_table_md_path else None,
            "slice_summary_path": str(slice_summary_path),
            "overlay_pngs": overlay_pngs,
            "texture_radiomics_summary_path": texture_summary_csv,
            "texture_radiomics_heatmap_path": texture_heatmap_png,
            "roi_sources": [r.get("name") for r in roi_specs],
            "roi_summaries": roi_summaries,
        },
        "artifacts": artifacts,
        "warnings": (warnings if warnings else ([] if rows else ["No measurable NIfTI images found; features.csv may be empty."])),
        "source_artifacts": [
            ArtifactRef(path=str(r.get("path")), kind="ref", description=f"ROI mask: {r.get('name')}") for r in roi_specs
        ]
        + [ArtifactRef(path=str(i.get("path", "")), kind="ref", description=f"Image: {i.get('name','')}") for i in norm_images],
        "generated_artifacts": artifacts,
    }


def build_tool() -> Tool:
    return Tool(spec=FEAT_SPEC, func=extract_roi_features)

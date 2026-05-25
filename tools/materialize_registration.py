"""
Tool: materialize_registration

Purpose:
- Create registration-like outputs without resampling when images are already aligned.
- Writes moving_*_resampled_to_fixed.nii.gz as a pass-through copy of the moving image.
- Emits central-slice PNGs and edge overlays for QC.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec
from .dicom_paths import as_str_list, list_dicom_instance_files, sort_dicom_instance_files


SPEC = ToolSpec(
    name="materialize_registration",
    description=(
        "Materialize registration outputs without resampling. "
        "Use when images are already co-registered; writes moving_*_resampled_to_fixed outputs and QC PNGs."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "fixed": {"type": "string"},
            "moving": {"type": "string"},
            "output_subdir": {"type": "string"},
            "label": {"type": "string"},
            "save_png_qc": {"type": "boolean"},
        },
        "required": ["fixed", "moving"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "resampled_path": {"type": "string"},
            "qc_pngs": {"type": "object"},
            "note": {"type": "string"},
        },
        "required": ["resampled_path"],
    },
    version="0.1.0",
    tags=["registration", "qc", "materialize"],
)


def _require_simpleitk():
    try:
        import SimpleITK as sitk  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "SimpleITK is required for materialize_registration. Install in this env:\n"
            "  pip install SimpleITK\n"
            f"Underlying error: {repr(e)}"
        ) from e
    return sitk


def _is_nifti_path(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _read_volume_any(path: Path, file_list: Optional[List[str]] = None):
    sitk = _require_simpleitk()
    path = Path(path)
    if path.exists() and path.is_file() and _is_nifti_path(path):
        img = sitk.ReadImage(str(path))
        return img, [str(path)]

    reader = sitk.ImageSeriesReader()
    if file_list is None:
        uids = reader.GetGDCMSeriesIDs(str(path))
        if not uids:
            cand = sort_dicom_instance_files(list_dicom_instance_files(path))
            if not cand:
                raise RuntimeError(f"No DICOM series IDs found in: {path}")
            files = as_str_list(cand)
            reader.SetFileNames(files)
            img = reader.Execute()
            return img, files
        files = reader.GetGDCMSeriesFileNames(str(path), uids[0])
    else:
        files = list(file_list)
    reader.SetFileNames(files)
    img = reader.Execute()
    return img, files


def _extract_central_slice(vol):
    sitk = _require_simpleitk()
    size = vol.GetSize()
    z = max(0, int(size[2] // 2)) if len(size) >= 3 else 0
    try:
        arr = sitk.GetArrayViewFromImage(vol)
        if arr.ndim == 3 and arr.shape[0] > 1:
            nz = (arr > 0).sum(axis=(1, 2))
            if int(nz.max()) > 0:
                z = int(nz.argmax())
    except Exception:
        pass
    extractor = sitk.ExtractImageFilter()
    extractor.SetSize([size[0], size[1], 0])
    extractor.SetIndex([0, 0, z])
    return extractor.Execute(vol)


def _to_u8(img2d):
    sitk = _require_simpleitk()
    try:
        sarr = sitk.GetArrayViewFromImage(img2d)
        flat = sarr.reshape(-1)
        nz = flat[flat > 0]
        if nz.size >= 64:
            lo = float(sorted(nz)[int(0.01 * (nz.size - 1))])
            hi = float(sorted(nz)[int(0.99 * (nz.size - 1))])
            if hi > lo:
                img2d = sitk.IntensityWindowing(img2d, windowMinimum=lo, windowMaximum=hi, outputMinimum=0.0, outputMaximum=255.0)
        img2d = sitk.Cast(img2d, sitk.sitkUInt8)
    except Exception:
        img2d = sitk.Cast(sitk.RescaleIntensity(img2d, 0, 255), sitk.sitkUInt8)
    return img2d


def _save_png(img2d, path: Path) -> None:
    sitk = _require_simpleitk()
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img2d, str(path))


def _edge_overlay_png(fixed_sl, moving_sl, out_path: Path) -> None:
    sitk = _require_simpleitk()
    try:
        fixed_edge = sitk.Abs(sitk.SobelEdgeDetection(fixed_sl))
        moving_edge = sitk.Abs(sitk.SobelEdgeDetection(moving_sl))
        fixed_u8 = _to_u8(fixed_edge)
        moving_u8 = _to_u8(moving_edge)
        zero = sitk.Image(fixed_u8.GetSize(), sitk.sitkUInt8)
        zero.CopyInformation(fixed_u8)
        rgb = sitk.Compose(fixed_u8, moving_u8, zero)
        _save_png(rgb, out_path)
    except Exception:
        return


def _infer_label(path: Path) -> str:
    name_l = path.name.lower()
    if "b1400" in name_l or "high_b" in name_l or "high-b" in name_l:
        return "high_b"
    if "b800" in name_l or "mid_b" in name_l or "mid-b" in name_l:
        return "mid_b"
    if "b50" in name_l or "b0" in name_l or "low_b" in name_l or "low-b" in name_l:
        return "low_b"
    if "adc" in name_l:
        return "adc"
    if "dwi" in name_l or "trace" in name_l:
        return "dwi"
    return "moving"


def materialize_registration(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    fixed_path = Path(args["fixed"]).expanduser().resolve()
    moving_path = Path(args["moving"]).expanduser().resolve()
    out_subdir = str(args.get("output_subdir", "registration"))
    label = str(args.get("label") or _infer_label(moving_path))
    save_png_qc = bool(args.get("save_png_qc", True))

    out_dir = ctx.artifacts_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_img, _ = _read_volume_any(fixed_path)
    moving_img, _ = _read_volume_any(moving_path)

    out_path = out_dir / f"moving_{label}_resampled_to_fixed.nii.gz"
    sitk = _require_simpleitk()
    sitk.WriteImage(moving_img, str(out_path))

    artifacts: List[ArtifactRef] = [
        ArtifactRef(path=str(out_path), kind="nifti", description=f"Materialized moving volume ({label}) in fixed space"),
    ]
    qc_pngs: Dict[str, str] = {}

    if save_png_qc:
        fixed_sl = _extract_central_slice(fixed_img)
        moving_sl = _extract_central_slice(moving_img)
        moving_png = out_dir / f"moving_{label}_central.png"
        _save_png(_to_u8(moving_sl), moving_png)
        qc_pngs[f"moving_{label}_central"] = str(moving_png)
        artifacts.append(ArtifactRef(path=str(moving_png), kind="figure", description=f"Moving {label} central slice PNG"))
        edge_png = out_dir / f"moving_{label}_edges_overlay.png"
        _edge_overlay_png(fixed_sl, moving_sl, edge_png)
        if edge_png.exists():
            qc_pngs[f"moving_{label}_edges_overlay"] = str(edge_png)
            artifacts.append(ArtifactRef(path=str(edge_png), kind="figure", description=f"Fixed/moving edge overlay ({label})"))

    return {
        "data": {
            "resampled_path": str(out_path),
            "qc_pngs": qc_pngs,
            "note": "Materialized registration output without resampling (assumed aligned).",
        },
        "artifacts": artifacts,
        "warnings": [],
        "source_artifacts": [
            ArtifactRef(path=str(fixed_path), kind="ref", description="Fixed reference"),
            ArtifactRef(path=str(moving_path), kind="ref", description="Moving image"),
        ],
        "generated_artifacts": artifacts,
    }


def build_tool() -> Tool:
    return Tool(spec=SPEC, func=materialize_registration)

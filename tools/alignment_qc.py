"""
Tool: alignment_qc

Purpose:
- Generate lightweight alignment QC artifacts before registration.
- Export central-slice PNGs for fixed/moving volumes.
- Export edge-overlay PNGs (fixed vs moving) to assess structural alignment.
- Record basic geometry summaries (size/spacing/origin/direction).
"""

from __future__ import annotations

import json
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec
from .dicom_paths import as_str_list, list_dicom_instance_files, sort_dicom_instance_files


ALIGN_QC_SPEC = ToolSpec(
    name="alignment_qc",
    description=(
        "Generate quick alignment QC images and geometry summaries for fixed/moving MRI volumes. "
        "Produces central-slice PNGs and fixed-vs-moving edge overlays without resampling."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "fixed_path": {"type": "string"},
            "moving_paths": {"type": "array"},
            "output_subdir": {"type": "string"},
            "save_edges": {"type": "boolean"},
        },
        "required": ["fixed_path", "moving_paths"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "summary_path": {"type": "string"},
            "images": {"type": "array"},
            "summaries": {"type": "object"},
        },
        "required": ["summary_path"],
    },
    version="0.1.0",
    tags=["qc", "alignment", "prostate"],
)


def _require_simpleitk():
    try:
        import SimpleITK as sitk  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "SimpleITK is required for alignment_qc. Install in this env:\n"
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
    return extractor.Execute(vol), z


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


def _geom_summary(img) -> Dict[str, Any]:
    try:
        return {
            "size": list(img.GetSize()),
            "spacing": list(img.GetSpacing()),
            "origin": list(img.GetOrigin()),
            "direction": list(img.GetDirection()),
        }
    except Exception:
        return {}


def alignment_qc(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    fixed_path = Path(args["fixed_path"]).expanduser().resolve()
    moving_paths = args.get("moving_paths") or []
    out_subdir = str(args.get("output_subdir", "alignment_qc"))
    save_edges = bool(args.get("save_edges", True))

    out_dir = ctx.artifacts_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: List[ArtifactRef] = []
    images: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}

    fixed_img, _ = _read_volume_any(fixed_path)
    fixed_sl, fixed_z = _extract_central_slice(fixed_img)
    fixed_png = out_dir / "fixed_T2w_central.png"
    _save_png(_to_u8(fixed_sl), fixed_png)
    images.append({"name": "T2w", "kind": "central_slice", "path": str(fixed_png), "z": fixed_z})
    artifacts.append(ArtifactRef(path=str(fixed_png), kind="figure", description="Fixed T2w central slice"))
    summaries["T2w"] = _geom_summary(fixed_img)

    for item in moving_paths:
        if isinstance(item, dict):
            name = str(item.get("name") or "moving")
            path = item.get("path")
        else:
            name = "moving"
            path = item
        if not path:
            continue
        mv_path = Path(str(path)).expanduser().resolve()
        mv_img, _ = _read_volume_any(mv_path)
        mv_sl, mv_z = _extract_central_slice(mv_img)
        mv_png = out_dir / f"{name}_central.png"
        _save_png(_to_u8(mv_sl), mv_png)
        images.append({"name": name, "kind": "central_slice", "path": str(mv_png), "z": mv_z})
        artifacts.append(ArtifactRef(path=str(mv_png), kind="figure", description=f"Moving {name} central slice"))
        summaries[name] = _geom_summary(mv_img)
        if save_edges:
            edge_png = out_dir / f"{name}_edges_overlay.png"
            _edge_overlay_png(fixed_sl, mv_sl, edge_png)
            if edge_png.exists():
                images.append({"name": name, "kind": "edges_overlay", "path": str(edge_png), "z": mv_z})
                artifacts.append(ArtifactRef(path=str(edge_png), kind="figure", description=f"Fixed vs {name} edge overlay"))

    summary_path = out_dir / "alignment_qc_summary.json"
    summary_path.write_text(
        json.dumps({"fixed_path": str(fixed_path), "moving_paths": moving_paths, "images": images, "summaries": summaries}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    artifacts.append(ArtifactRef(path=str(summary_path), kind="json", description="Alignment QC summary"))

    return {
        "data": {
            "summary_path": str(summary_path),
            "images": images,
            "summaries": summaries,
        },
        "artifacts": artifacts,
        "warnings": [],
        "source_artifacts": [ArtifactRef(path=str(fixed_path), kind="ref", description="Fixed reference")],
        "generated_artifacts": artifacts,
    }


def build_tool() -> Tool:
    return Tool(spec=ALIGN_QC_SPEC, func=alignment_qc)

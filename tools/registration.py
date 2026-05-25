"""
Tool: register_to_reference

Purpose (Stage 2 MVP):
- Header-based resample: bring moving (DWI/ADC) into fixed (T2w) physical space.
- No optimization/registration yet: uses identity transform with DICOM geometry.
- Produces resampled NIfTI(s) + central-slice QC PNGs.

Inputs:
- fixed: DICOM series directory (T2w) OR a NIfTI path (.nii/.nii.gz)
- moving: DICOM series directory (DWI/ADC) OR a NIfTI path (.nii/.nii.gz)
- split_by_bvalue (optional): attempt to split diffusion series into groups (b-values or halves)
- save_png_qc (optional): save central-slice PNGs for QC

Outputs:
- transform_path: identity transform file
- resampled_path: primary resampled output
- resampled_paths: dict label->nifti path (e.g., part1/part2)
- qc_pngs: dict of png paths
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec
from .dicom_paths import as_str_list, list_dicom_instance_files, sort_dicom_instance_files


REG_SPEC = ToolSpec(
    name="register_to_reference",
    description=(
        "Resample a moving MRI volume into a fixed reference space using SimpleITK. "
        "Inputs can be DICOM series directories or NIfTI paths (.nii/.nii.gz). "
        "Supports methods: identity (header-based), rigid, affine. "
        "For diffusion (DWI/TRACEW), can optionally split the series into low/mid/high-b sub-volumes when possible."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "moving": {"type": "string"},  # DICOM series directory OR NIfTI path
            "fixed": {"type": "string"},  # DICOM series directory (T2w) OR NIfTI path
            "output_subdir": {"type": "string"},
            "split_by_bvalue": {"type": "boolean"},
            "save_png_qc": {"type": "boolean"},
            "interpolation": {
                "type": "string",
                "enum": ["linear", "nearest", "bspline"],
                "description": (
                    "Interpolation mode for resampling. Use 'nearest' when the "
                    "moving image is a binary or label mask to preserve integer "
                    "values. Default: 'linear'."
                ),
                "default": "linear",
            },
        },
        "required": ["moving", "fixed"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "transform_path": {"type": "string"},
            "resampled_path": {"type": "string"},
            "resampled_paths": {"type": "object"},
            "qc_pngs": {"type": "object"},
            "note": {"type": "string"},
        },
        "required": ["transform_path", "resampled_path"],
    },
    version="0.2.0",
    tags=["registration", "mvp", "simpleitk", "resample"],
)


def _require_simpleitk():
    try:
        import SimpleITK as sitk  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "SimpleITK is required for Stage-2 resampling. Install in this env:\n"
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

    # DICOM directory path
    reader = sitk.ImageSeriesReader()
    if file_list is None:
        # If multiple series UIDs exist, this picks the first; for our demo folders it's usually fine.
        uids = reader.GetGDCMSeriesIDs(str(path))
        if not uids:
            # Some datasets store DICOM instances nested or without .dcm extension.
            # Fall back to explicit recursive instance listing.
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


def _split_dwi_by_bvalue(series_dir: Path) -> List[Tuple[str, List[str]]]:
    """
    Returns list of (label, file_list) where label is like 'b50' or 'b1400'.
    If b-values are unavailable, returns one group ('all', all files).
    """
    try:
        import pydicom  # type: ignore
    except Exception:
        # Can't split; SimpleITK will load as-is.
        return [("all", as_str_list(list_dicom_instance_files(series_dir)))]

    def _proj_from_ds(ds) -> Optional[float]:
        """
        Sort slices by physical position using ImagePositionPatient projected onto slice normal.
        Fallback to z if orientation missing.
        """
        ipp = getattr(ds, "ImagePositionPatient", None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        try:
            if ipp is None:
                return None
            ipp = [float(x) for x in ipp]
            if iop is None:
                return float(ipp[2])
            iop = [float(x) for x in iop]
            row = iop[0:3]
            col = iop[3:6]
            # normal = row x col
            n = [
                row[1] * col[2] - row[2] * col[1],
                row[2] * col[0] - row[0] * col[2],
                row[0] * col[1] - row[1] * col[0],
            ]
            return float(ipp[0] * n[0] + ipp[1] * n[1] + ipp[2] * n[2])
        except Exception:
            try:
                return float(ipp[2]) if ipp is not None else None
            except Exception:
                return None

    items: List[Tuple[Path, Optional[float], Optional[float], int]] = []
    for p in list_dicom_instance_files(series_dir):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            b = getattr(ds, "DiffusionBValue", None)
            inst = getattr(ds, "InstanceNumber", None)
            b = float(b) if b is not None else None
            inst = int(inst) if inst is not None else 0
            proj = _proj_from_ds(ds)
            items.append((p, b, proj, inst))
        except Exception:
            continue

    # Group by b
    by_b: Dict[float, List[Tuple[Path, Optional[float], int]]] = {}
    for p, b, proj, inst in items:
        if b is None:
            continue
        by_b.setdefault(float(b), []).append((p, proj, inst))

    if len(by_b) <= 1:
        # No usable b-values; return all instances ordered by physical position when available.
        if items:
            def key(x):
                # x: (p, b, proj, inst)
                proj = x[2]
                inst = x[3]
                return (0, proj) if proj is not None else (1, inst)

            files = [str(p) for p, _, _, _ in sorted(items, key=key)]
        else:
            files = as_str_list(list_dicom_instance_files(series_dir))
        return [("all", files)]

    out: List[Tuple[str, List[str]]] = []
    for b in sorted(by_b.keys()):
        # sort by physical position first, fallback to InstanceNumber
        pairs = sorted(by_b[b], key=lambda x: (0, x[1]) if x[1] is not None else (1, x[2]))
        label = f"b{int(b)}" if float(int(b)) == b else f"b{b}"
        out.append((label, [str(p) for p, _, _ in pairs]))
    return out


def _save_png(img2d, path: Path) -> None:
    sitk = _require_simpleitk()
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img2d, str(path))


def _extract_central_slice(vol):
    sitk = _require_simpleitk()
    size = vol.GetSize()
    # Choose a "best" slice for visualization (avoid all-black central slices).
    # Heuristic: pick slice with max non-zero voxel count; fallback to center.
    z = max(0, int(size[2] // 2)) if len(size) >= 3 else 0
    try:
        arr = sitk.GetArrayViewFromImage(vol)  # z,y,x
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
    # Robust windowing: use percentile on non-zero pixels if possible.
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


def _central_slice_png(vol, out_path: Path) -> None:
    sl = _extract_central_slice(vol)
    sl_u8 = _to_u8(sl)
    _save_png(sl_u8, out_path)


def _edge_overlay_png(fixed_vol, moving_vol, out_path: Path) -> None:
    sitk = _require_simpleitk()
    try:
        fixed_sl = _extract_central_slice(fixed_vol)
        moving_sl = _extract_central_slice(moving_vol)
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


def _geom_match(a, b, tol: float = 1e-3) -> bool:
    try:
        if a.GetSize() != b.GetSize():
            return False
        for aval, bval in zip(a.GetSpacing(), b.GetSpacing()):
            if abs(float(aval) - float(bval)) > tol:
                return False
        for aval, bval in zip(a.GetOrigin(), b.GetOrigin()):
            if abs(float(aval) - float(bval)) > tol:
                return False
        for aval, bval in zip(a.GetDirection(), b.GetDirection()):
            if abs(float(aval) - float(bval)) > tol:
                return False
        return True
    except Exception:
        return False


def register_to_reference(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    sitk = _require_simpleitk()
    moving_path = Path(args["moving"]).expanduser().resolve()
    fixed_path = Path(args["fixed"]).expanduser().resolve()
    out_subdir = args.get("output_subdir", "registration")
    split_by_bvalue = bool(args.get("split_by_bvalue", True))
    save_png_qc = bool(args.get("save_png_qc", True))
    method = str(args.get("method", "identity")).lower().strip()
    interpolation_name = str(args.get("interpolation", "linear")).strip().lower()
    _interp_map = {"linear": sitk.sitkLinear, "nearest": sitk.sitkNearestNeighbor, "bspline": sitk.sitkBSpline}
    if interpolation_name not in _interp_map:
        interpolation_name = "linear"
    interpolation_sitk = _interp_map[interpolation_name]

    out_dir = ctx.artifacts_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []

    fixed_img, fixed_files = _read_volume_any(fixed_path)

    # Note: for NIfTI inputs, we may still be able to split by slice blocks later (if Z == N*fixed_z).
    moving_is_nifti = bool(moving_path.exists() and moving_path.is_file() and _is_nifti_path(moving_path))
    if moving_is_nifti:
        groups: List[Tuple[str, Optional[List[str]]]] = [("all", None)]
    else:
        groups = (
            [(label, files) for (label, files) in _split_dwi_by_bvalue(moving_path)]
            if split_by_bvalue
            else [("all", as_str_list(list_dicom_instance_files(moving_path)))]
        )

    # Heuristic fallback for TRACEW-like series that concatenate multiple b-values without b-tags:
    # if moving has exactly N x the number of fixed slices, split into N equal blocks.
    try:
        fixed_z = int(fixed_img.GetSize()[2])
    except Exception:
        fixed_z = 0
    # For NIfTI, we can infer block boundaries directly from Z. For DICOM, we split by acquisition order.
    nifti_block_ranges: Dict[str, Tuple[int, int]] = {}
    if split_by_bvalue and moving_is_nifti and fixed_z > 0:
        try:
            mv_full, _ = _read_volume_any(moving_path)
            mv_z = int(mv_full.GetSize()[2])
            if mv_z == 3 * fixed_z:
                groups = [("low_b", None), ("mid_b", None), ("high_b", None)]
                nifti_block_ranges = {"low_b": (0, fixed_z), "mid_b": (fixed_z, 2 * fixed_z), "high_b": (2 * fixed_z, 3 * fixed_z)}
            elif mv_z == 2 * fixed_z:
                groups = [("low_b", None), ("high_b", None)]
                nifti_block_ranges = {"low_b": (0, fixed_z), "high_b": (fixed_z, 2 * fixed_z)}
        except Exception:
            nifti_block_ranges = {}

    if split_by_bvalue and (not moving_is_nifti) and len(groups) == 1 and fixed_z > 0 and groups[0][0] in ("all", "dwi"):
        # Build a per-file table with (inst, proj) so we can:
        # - split by acquisition order (InstanceNumber) into low/high b blocks
        # - sort within each block by physical slice position for correct 3D volume assembly
        try:
            import pydicom  # type: ignore
        except Exception:
            pydicom = None  # type: ignore

        file_paths = list_dicom_instance_files(moving_path)
        table: List[Tuple[str, int, Optional[float]]] = []
        if pydicom is not None:
            for p in file_paths:
                try:
                    ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
                    inst = int(getattr(ds, "InstanceNumber", 0) or 0)
                    ipp = getattr(ds, "ImagePositionPatient", None)
                    iop = getattr(ds, "ImageOrientationPatient", None)
                    proj = None
                    try:
                        if ipp is not None:
                            ipp = [float(x) for x in ipp]
                            if iop is None:
                                proj = float(ipp[2])
                            else:
                                iop = [float(x) for x in iop]
                                row = iop[0:3]
                                col = iop[3:6]
                                n = [
                                    row[1] * col[2] - row[2] * col[1],
                                    row[2] * col[0] - row[0] * col[2],
                                    row[0] * col[1] - row[1] * col[0],
                                ]
                                proj = float(ipp[0] * n[0] + ipp[1] * n[1] + ipp[2] * n[2])
                    except Exception:
                        proj = None
                    table.append((str(p), inst, proj))
                except Exception:
                    table.append((str(p), 0, None))
        else:
            table = [(str(p), i, None) for i, p in enumerate(file_paths)]

        # Split by acquisition order first (best proxy when b-tags are missing).
        table_sorted = sorted(table, key=lambda x: x[1])
        if len(table_sorted) == 3 * fixed_z:
            b0 = table_sorted[:fixed_z]
            b1 = table_sorted[fixed_z : 2 * fixed_z]
            b2 = table_sorted[2 * fixed_z :]

            # Sort within each block by physical slice position if available
            def within_key(x):
                _, inst, proj = x
                return (0, proj) if proj is not None else (1, inst)

            f0 = [p for p, _, _ in sorted(b0, key=within_key)]
            f1 = [p for p, _, _ in sorted(b1, key=within_key)]
            f2 = [p for p, _, _ in sorted(b2, key=within_key)]
            groups = [("low_b", f0), ("mid_b", f1), ("high_b", f2)]
        elif len(table_sorted) == 2 * fixed_z:
            b0 = table_sorted[:fixed_z]
            b1 = table_sorted[fixed_z:]

            def within_key(x):
                _, inst, proj = x
                return (0, proj) if proj is not None else (1, inst)

            f0 = [p for p, _, _ in sorted(b0, key=within_key)]
            f1 = [p for p, _, _ in sorted(b1, key=within_key)]
            groups = [("low_b", f0), ("high_b", f1)]

    # If this looks like a known diffusion/ADC series, prefer a stable label instead of "all".
    # (Many series don't carry b-value tags; without this, outputs become moving_all_* which is confusing.)
    try:
        if len(groups) == 1 and groups[0][0] == "all":
            name_l = str(moving_path.name).lower()
            if "b1400" in name_l or "high_b" in name_l or "high-b" in name_l:
                groups = [("high_b", groups[0][1])]
            elif "b800" in name_l or "mid_b" in name_l or "mid-b" in name_l:
                groups = [("mid_b", groups[0][1])]
            elif "b50" in name_l or "b0" in name_l or "low_b" in name_l or "low-b" in name_l:
                groups = [("low_b", groups[0][1])]
            elif "adc" in name_l:
                groups = [("adc", groups[0][1])]
            elif "dwi" in name_l or "trace" in name_l:
                groups = [("dwi", groups[0][1])]
    except Exception:
        pass

    resampled_paths: Dict[str, str] = {}
    qc_pngs: Dict[str, str] = {}
    artifacts: List[ArtifactRef] = []

    # Transform setup
    qc_metrics: Dict[str, Any] = {"method": method}
    if method == "identity":
        tx_global = sitk.Transform(3, sitk.sitkIdentity)
        transform_path = out_dir / "transform_identity.tfm"
        sitk.WriteTransform(tx_global, str(transform_path))
        artifacts.append(ArtifactRef(path=str(transform_path), kind="transform", description="Identity transform (header-based resample)"))
    elif method in ("affine", "rigid"):
        # We'll estimate transform using the first group as representative (typically low_b or adc).
        rep_label, rep_files = groups[0]
        rep_img, _ = _read_volume_any(moving_path, file_list=(rep_files or None))
        rep_img = sitk.Cast(rep_img, sitk.sitkFloat32)
        fixed_f = sitk.Cast(fixed_img, sitk.sitkFloat32)

        if method == "rigid":
            init_tx = sitk.CenteredTransformInitializer(fixed_f, rep_img, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
        else:
            init_tx = sitk.CenteredTransformInitializer(fixed_f, rep_img, sitk.AffineTransform(3), sitk.CenteredTransformInitializerFilter.GEOMETRY)

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.10)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(init_tx, inPlace=False)
        final_tx = reg.Execute(fixed_f, rep_img)

        tx_global = final_tx
        transform_path = out_dir / f"transform_{method}.tfm"
        sitk.WriteTransform(tx_global, str(transform_path))
        artifacts.append(ArtifactRef(path=str(transform_path), kind="transform", description=f"{method} transform (SimpleITK MI)"))
        qc_metrics.update(
            {
                "metric_value": float(reg.GetMetricValue()),
                "optimizer_stop_condition": str(reg.GetOptimizerStopConditionDescription()),
                "iterations": int(reg.GetOptimizerIteration()),
                "representative_label": str(rep_label),
            }
        )
    else:
        raise ValueError(f"Unsupported registration method: {method} (use identity|rigid|affine)")

    # Save fixed central slice for QC
    if save_png_qc:
        fixed_png = out_dir / "fixed_T2w_central.png"
        _central_slice_png(fixed_img, fixed_png)
        qc_pngs["fixed_T2w_central"] = str(fixed_png)
        artifacts.append(ArtifactRef(path=str(fixed_png), kind="figure", description="Fixed (T2w) central slice PNG"))

    primary_resampled: Optional[Path] = None

    # If NIfTI splitting is active, read the full moving volume once.
    moving_full_img = None
    if moving_is_nifti and nifti_block_ranges:
        try:
            moving_full_img, _ = _read_volume_any(moving_path)
        except Exception:
            moving_full_img = None

    for label, file_list in groups:
        if moving_full_img is not None and label in nifti_block_ranges:
            z0, z1 = nifti_block_ranges[label]
            # RegionOfInterest expects size/index in (x,y,z)
            sx, sy, sz = moving_full_img.GetSize()
            size = [int(sx), int(sy), int(z1 - z0)]
            index = [0, 0, int(z0)]
            moving_img = sitk.RegionOfInterest(moving_full_img, size=size, index=index)
            # IMPORTANT: stacked DWI blocks represent the same anatomy at different b-values.
            # Reset origin/direction so each block aligns to the same physical space as the full volume.
            try:
                moving_img.SetOrigin(moving_full_img.GetOrigin())
                moving_img.SetDirection(moving_full_img.GetDirection())
                moving_img.SetSpacing(moving_full_img.GetSpacing())
            except Exception:
                pass
        else:
            moving_img, _ = _read_volume_any(moving_path, file_list=(file_list or None))
        # Heuristic: if ADC series shows non-uniform sampling (missing slices), identity resample is often wrong.
        # We can't fully correct without a better DICOM reader, but we can surface a warning and recommend affine.
        try:
            z = int(moving_img.GetSize()[2])
            fz = int(fixed_img.GetSize()[2])
            if method == "identity" and ("adc" in moving_path.name.lower()) and (abs(z - fz) >= 2):
                warnings.append(
                    f"ADC slice count differs from T2w (moving_z={z} vs fixed_z={fz}); identity resample may misalign. Consider method='affine'."
                )
        except Exception:
            pass
        # If the moving image already matches the fixed geometry, skip resampling.
        resampled = False
        if method == "identity" and _geom_match(moving_img, fixed_img):
            res = moving_img
            resampled = False
        else:
            res = sitk.Resample(moving_img, fixed_img, tx_global, interpolation_sitk, 0.0, moving_img.GetPixelID())
            resampled = True
        out_path = out_dir / f"moving_{label}_resampled_to_fixed.nii.gz"
        sitk.WriteImage(res, str(out_path))
        resampled_paths[label] = str(out_path)
        artifacts.append(ArtifactRef(path=str(out_path), kind="nifti", description=f"Resampled moving volume ({label}) in fixed space"))

        if primary_resampled is None:
            primary_resampled = out_path

        if save_png_qc:
            moving_png = out_dir / f"moving_{label}_central.png"
            _central_slice_png(res, moving_png)
            qc_pngs[f"moving_{label}_central"] = str(moving_png)
            artifacts.append(ArtifactRef(path=str(moving_png), kind="figure", description=f"Resampled moving ({label}) central slice PNG"))
            edge_png = out_dir / f"moving_{label}_edges_overlay.png"
            _edge_overlay_png(fixed_img, res, edge_png)
            if edge_png.exists():
                qc_pngs[f"moving_{label}_edges_overlay"] = str(edge_png)
                artifacts.append(ArtifactRef(path=str(edge_png), kind="figure", description=f"Fixed/moving edge overlay ({label})"))

        if resampled is False:
            warnings.append(f"Moving volume ({label}) already aligned with fixed; skipped resample.")

    # Backward-compatible fields
    resampled_path = str(primary_resampled) if primary_resampled else ""
    if not primary_resampled:
        warnings.append("No resampled output produced (empty moving series?)")

    return {
        "data": {
            "fixed": str(fixed_path),
            "moving": str(moving_path),
            "output_subdir": str(out_subdir),
            "transform_path": str(transform_path),
            "resampled_path": resampled_path,
            "resampled_paths": resampled_paths,
            "qc_pngs": qc_pngs,
            "qc_metrics": qc_metrics,
            "note": f"Stage-2 resample using SimpleITK (method={method}).",
        },
        "artifacts": artifacts,
        "warnings": warnings,
        "source_artifacts": [
            ArtifactRef(path=str(fixed_path), kind="ref", description="Fixed reference (DICOM dir or NIfTI path)"),
            ArtifactRef(path=str(moving_path), kind="ref", description="Moving image (DICOM dir or NIfTI path)"),
        ],
        "generated_artifacts": artifacts,
    }


def build_tool() -> Tool:
    return Tool(spec=REG_SPEC, func=register_to_reference)

"""
Tool: resample_image

Pure-geometry resampling of a NIfTI volume onto a target physical grid
**without** any spatial-alignment optimisation.

Use-cases
---------
* Up/down-sample an ADC map (1.5 mm) to match a T2W grid (0.5 mm).
* Resample a binary segmentation mask to a different grid using
  NearestNeighbor interpolation so label edges stay crisp.
* Resample any NIfTI to an arbitrary spacing / size / direction
  specified either by a reference NIfTI *or* by explicit spacing values.

This is intentionally *not* a registration tool — there is no transform
estimation.  If two images are physically misaligned you should call
``register_to_reference`` first, then use this tool (or the registration
output) to resample.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool specification
# ---------------------------------------------------------------------------

RESAMPLE_SPEC = ToolSpec(
    name="resample_image",
    description=(
        "Resample a NIfTI volume to a target physical grid WITHOUT spatial-alignment "
        "optimisation (no registration).  Specify the target grid via a reference NIfTI "
        "or via explicit target_spacing.  Supports multiple interpolation modes:\n"
        "  - 'linear'  (default) — continuous volumes (T2W, ADC, DWI).\n"
        "  - 'nearest' — binary/label masks (preserves integer labels).\n"
        "  - 'bspline' — high-quality smooth resampling.\n\n"
        "Use this tool when images are already spatially aligned but live on different "
        "voxel grids (e.g. ADC at 1.5 mm vs T2W at 0.5 mm).  Do NOT use if the images "
        "are physically misaligned — call register_to_reference first."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "input_nifti": {
                "type": "string",
                "description": "Path to the NIfTI volume to resample.",
            },
            "reference_nifti": {
                "type": "string",
                "description": (
                    "Path to a reference NIfTI whose grid (size, spacing, origin, "
                    "direction) defines the output geometry.  Mutually exclusive with "
                    "target_spacing."
                ),
            },
            "target_spacing": {
                "type": "array",
                "items": {"type": "number"},
                "description": (
                    "Desired output spacing [sx, sy, sz] in mm.  The output size is "
                    "computed to cover the same physical extent as the input.  Ignored "
                    "if reference_nifti is provided."
                ),
            },
            "interpolation": {
                "type": "string",
                "enum": ["linear", "nearest", "bspline"],
                "description": (
                    "Interpolation mode.  Use 'nearest' for binary/label masks to "
                    "preserve integer values.  Default: 'linear'."
                ),
                "default": "linear",
            },
            "default_pixel_value": {
                "type": "number",
                "description": "Value for voxels outside the input FOV (default 0).",
                "default": 0.0,
            },
            "output_nifti": {
                "type": "string",
                "description": (
                    "Explicit output path.  If omitted, writes "
                    "'resampled_<input_stem>.nii.gz' in the artifacts directory."
                ),
            },
            "output_subdir": {
                "type": "string",
                "description": "Sub-directory under artifacts_dir for outputs.",
                "default": "resample",
            },
        },
        "required": ["input_nifti"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "resampled_nifti": {"type": "string"},
            "input_spacing": {"type": "array"},
            "output_spacing": {"type": "array"},
            "input_size": {"type": "array"},
            "output_size": {"type": "array"},
            "interpolation": {"type": "string"},
            "elapsed_seconds": {"type": "number"},
        },
    },
    version="0.1.0",
    tags=["resample", "preprocessing", "geometry"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INTERP_MAP = None  # populated lazily to avoid import at module level


def _interp_flag(name: str):
    """Map a human-friendly name to a SimpleITK interpolator enum."""
    import SimpleITK as sitk  # type: ignore

    global _INTERP_MAP
    if _INTERP_MAP is None:
        _INTERP_MAP = {
            "linear": sitk.sitkLinear,
            "nearest": sitk.sitkNearestNeighbor,
            "bspline": sitk.sitkBSpline,
        }
    key = name.strip().lower()
    if key not in _INTERP_MAP:
        raise ValueError(
            f"Unknown interpolation '{name}'. Choose from: {list(_INTERP_MAP.keys())}"
        )
    return _INTERP_MAP[key]


def _compute_new_size(old_size, old_spacing, new_spacing):
    """Compute output size so the physical extent is preserved."""
    import math
    return [
        int(math.ceil(old_size[i] * old_spacing[i] / new_spacing[i]))
        for i in range(len(old_size))
    ]


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------

def resample_image(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    """Resample a NIfTI volume to a target grid."""
    import SimpleITK as sitk  # type: ignore

    t0 = time.time()

    # ── Parse arguments ────────────────────────────────────────────────
    input_path = Path(args["input_nifti"]).expanduser().resolve()
    if not input_path.exists():
        return {"ok": False, "error": f"Input NIfTI not found: {input_path}"}

    interp_name = str(args.get("interpolation", "linear")).strip().lower()
    interp_sitk = _interp_flag(interp_name)
    default_pix = float(args.get("default_pixel_value", 0.0))

    output_subdir = args.get("output_subdir", "resample")
    out_dir = ctx.artifacts_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    output_nifti_arg = args.get("output_nifti")
    if output_nifti_arg:
        output_path = Path(output_nifti_arg).expanduser().resolve()
    else:
        stem = input_path.name.replace(".nii.gz", "").replace(".nii", "")
        output_path = out_dir / f"resampled_{stem}.nii.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Read input ─────────────────────────────────────────────────────
    img = sitk.ReadImage(str(input_path))
    in_spacing = list(img.GetSpacing())
    in_size = list(img.GetSize())

    # ── Determine target grid ──────────────────────────────────────────
    ref_path = args.get("reference_nifti")
    target_spacing_arg = args.get("target_spacing")

    if ref_path:
        ref = sitk.ReadImage(str(Path(ref_path).expanduser().resolve()))
        out_size = list(ref.GetSize())
        out_spacing = list(ref.GetSpacing())
        out_origin = list(ref.GetOrigin())
        out_direction = list(ref.GetDirection())
    elif target_spacing_arg:
        out_spacing = [float(s) for s in target_spacing_arg]
        # Pad to 3D if user only gave 2 values
        while len(out_spacing) < 3:
            out_spacing.append(in_spacing[len(out_spacing)])
        out_size = _compute_new_size(in_size, in_spacing, out_spacing)
        out_origin = list(img.GetOrigin())
        out_direction = list(img.GetDirection())
    else:
        return {
            "ok": False,
            "error": (
                "Must provide either 'reference_nifti' or 'target_spacing' "
                "to define the output grid."
            ),
        }

    logger.info(
        "[resample] %s -> spacing %s->%s  size %s->%s  interp=%s",
        input_path.name, in_spacing, out_spacing, in_size, out_size, interp_name,
    )

    # ── Resample ───────────────────────────────────────────────────────
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(out_size)
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetOutputOrigin(out_origin)
    resampler.SetOutputDirection(out_direction)
    resampler.SetInterpolator(interp_sitk)
    resampler.SetDefaultPixelValue(default_pix)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetOutputPixelType(img.GetPixelID())

    resampled = resampler.Execute(img)

    # ── Write output ───────────────────────────────────────────────────
    sitk.WriteImage(resampled, str(output_path))
    logger.info("[resample] Saved: %s", output_path)

    elapsed = time.time() - t0
    artifacts: List[ArtifactRef] = [
        ArtifactRef(
            path=str(output_path),
            kind="nifti",
            description=(
                f"Resampled volume (interp={interp_name}, "
                f"spacing={out_spacing})"
            ),
        ),
    ]

    return {
        "ok": True,
        "data": {
            "resampled_nifti": str(output_path),
            "input_nifti": str(input_path),
            "input_spacing": in_spacing,
            "output_spacing": out_spacing,
            "input_size": in_size,
            "output_size": out_size,
            "interpolation": interp_name,
            "elapsed_seconds": round(elapsed, 3),
        },
        "generated_artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# build_tool() – entry-point for the tool registry
# ---------------------------------------------------------------------------

def build_tool() -> Tool:
    return Tool(spec=RESAMPLE_SPEC, func=resample_image)

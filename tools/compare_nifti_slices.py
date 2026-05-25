"""
Tool: compare_nifti_slices

Deterministic side-by-side comparison of two NIfTI volumes.

Extracts the centre axial slice from each volume and saves a side-by-side
PNG. Handles 2-D, 3-D, and 4-D (takes first volume of 4-D) inputs.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec

logger = logging.getLogger(__name__)

COMPARE_SPEC = ToolSpec(
    name="compare_nifti_slices",
    description=(
        "Load two NIfTI volumes, extract the centre axial slice from each, and "
        "save a side-by-side comparison PNG. Useful for before/after visualisation "
        "of denoising, super-resolution, or other preprocessing steps."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "image_a": {
                "type": "string",
                "description": "Path to the first NIfTI volume (e.g. the *before* image).",
            },
            "image_b": {
                "type": "string",
                "description": "Path to the second NIfTI volume (e.g. the *after* image).",
            },
            "label_a": {
                "type": "string",
                "description": "Title for the left panel.",
                "default": "Before",
            },
            "label_b": {
                "type": "string",
                "description": "Title for the right panel.",
                "default": "After",
            },
            "output_png": {
                "type": "string",
                "description": (
                    "Optional explicit output path for the PNG. If omitted, "
                    "writes 'before_vs_after.png' under artifacts/<output_subdir>."
                ),
            },
            "output_subdir": {
                "type": "string",
                "description": "Sub-directory under artifacts_dir for outputs.",
                "default": "compare",
            },
        },
        "required": ["image_a", "image_b"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "output_png": {"type": "string"},
            "image_a": {"type": "string"},
            "image_b": {"type": "string"},
            "slice_index_a": {"type": "integer"},
            "slice_index_b": {"type": "integer"},
            "elapsed_seconds": {"type": "number"},
        },
    },
    version="0.1.0",
    tags=["compare", "visualisation", "preprocessing"],
)


def _require_deps():
    """Lazy-import heavy dependencies so the module can be imported cheaply."""
    try:
        import numpy as np  # type: ignore
        import SimpleITK as sitk  # type: ignore
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency for compare_nifti_slices. "
            "Install with: pip install SimpleITK matplotlib numpy"
        ) from exc
    return np, sitk, plt


def _centre_axial_slice(arr, spacing, np_mod):
    """Return (2-D slice, slice_index, aspect_ratio) from the centre of a volume.

    SimpleITK ``GetArrayFromImage`` returns arrays in **(z, y, x)** order.
    We extract the centre slice along axis-0 (axial plane), which displays
    the (y, x) grid.  The physical aspect ratio is ``spacing_x / spacing_y``
    so that ``imshow`` renders each voxel at its true physical size.

    For 4-D data the first volume is taken.  For 2-D data, aspect defaults
    to 1.0.
    """
    a = np_mod.asarray(arr, dtype=np_mod.float32)
    if a.ndim == 4:
        a = a[..., 0]  # take first volume
    if a.ndim == 3:
        z = a.shape[0] // 2          # axis-0 == z in (z, y, x)
        slc = a[z, :, :]             # shape (y, x)
        # spacing is (x, y, z) from SimpleITK
        sp_x = float(spacing[0]) if len(spacing) > 0 else 1.0
        sp_y = float(spacing[1]) if len(spacing) > 1 else 1.0
        # imshow rows=y, cols=x  →  aspect = pixel_width / pixel_height = sp_x / sp_y
        aspect = sp_x / sp_y if sp_y > 1e-12 else 1.0
        return slc, int(z), aspect
    if a.ndim == 2:
        return a, 0, 1.0
    raise ValueError(f"Unexpected array dimensionality: {a.ndim}")


def compare_nifti_slices(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    np, sitk, plt = _require_deps()
    t0 = time.time()

    path_a = Path(str(args.get("image_a", ""))).expanduser().resolve()
    path_b = Path(str(args.get("image_b", ""))).expanduser().resolve()

    if not path_a.exists():
        return {"ok": False, "error": f"image_a not found: {path_a}"}
    if not path_b.exists():
        return {"ok": False, "error": f"image_b not found: {path_b}"}

    label_a = str(args.get("label_a", "Before")).strip() or "Before"
    label_b = str(args.get("label_b", "After")).strip() or "After"

    output_subdir = str(args.get("output_subdir", "compare")).strip() or "compare"
    out_dir = ctx.artifacts_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    output_png_arg = str(args.get("output_png") or "").strip()
    if output_png_arg:
        output_path = Path(output_png_arg).expanduser().resolve()
    else:
        output_path = out_dir / "before_vs_after.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load volumes
    img_a = sitk.ReadImage(str(path_a))
    img_b = sitk.ReadImage(str(path_b))
    arr_a = sitk.GetArrayFromImage(img_a)
    arr_b = sitk.GetArrayFromImage(img_b)
    spacing_a = img_a.GetSpacing()  # (x, y, z)
    spacing_b = img_b.GetSpacing()

    slice_a, z_a, aspect_a = _centre_axial_slice(arr_a, spacing_a, np)
    slice_b, z_b, aspect_b = _centre_axial_slice(arr_b, spacing_b, np)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(slice_a, cmap="gray", origin="lower", aspect=aspect_a)
    axes[0].set_title(label_a, fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(slice_b, cmap="gray", origin="lower", aspect=aspect_b)
    axes[1].set_title(label_b, fontsize=12)
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - t0
    artifacts: List[ArtifactRef] = [
        ArtifactRef(
            path=str(output_path),
            kind="png",
            description=f"Side-by-side comparison: {label_a} vs {label_b}",
        ),
    ]

    return {
        "ok": True,
        "data": {
            "output_png": str(output_path),
            "image_a": str(path_a),
            "image_b": str(path_b),
            "slice_index_a": z_a,
            "slice_index_b": z_b,
            "elapsed_seconds": round(elapsed, 3),
        },
        "generated_artifacts": artifacts,
    }


def build_tool() -> Tool:
    return Tool(spec=COMPARE_SPEC, func=compare_nifti_slices)

"""
Tool: denoise_image_bm3d

Classical slice-wise BM3D denoising for MRI NIfTI volumes.

Implementation notes:
- Intensity is min-max normalized to [0, 1] before BM3D.
- BM3D is applied 2D slice-by-slice (flattening leading dims) to keep
  memory usage predictable.
- Output is de-normalized back to the original value range and geometry
  (spacing/origin/direction) is copied exactly from input.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec

logger = logging.getLogger(__name__)


BM3D_SPEC = ToolSpec(
    name="denoise_image_bm3d",
    description=(
        "Denoise a NIfTI volume using classical BM3D denoising. Intensities are "
        "normalized to [0,1], denoised slice-by-slice with BM3D profile='np', then "
        "scaled back to original intensity range. Geometry is preserved exactly."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "input_nifti": {
                "type": "string",
                "description": "Path to input NIfTI volume.",
            },
            "sigma_psd": {
                "type": "number",
                "description": (
                    "Estimated noise standard deviation in normalized [0,1] domain. "
                    "Typical range: 0.03-0.15."
                ),
                "default": 0.08,
            },
            "output_nifti": {
                "type": "string",
                "description": (
                    "Optional explicit output path. If omitted, writes "
                    "'denoised_<input_stem>.nii.gz' under artifacts/output_subdir."
                ),
            },
            "output_subdir": {
                "type": "string",
                "description": "Sub-directory under artifacts_dir for outputs.",
                "default": "denoise",
            },
        },
        "required": ["input_nifti"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "denoised_nifti": {"type": "string"},
            "input_nifti": {"type": "string"},
            "sigma_psd": {"type": "number"},
            "original_min": {"type": "number"},
            "original_max": {"type": "number"},
            "elapsed_seconds": {"type": "number"},
        },
    },
    version="0.1.0",
    tags=["denoise", "bm3d", "preprocessing"],
)


def _require_deps():
    try:
        import numpy as np  # type: ignore
        import SimpleITK as sitk  # type: ignore
        from bm3d import bm3d as bm3d_fn  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency for denoise_image_bm3d. Install with: pip install bm3d SimpleITK"
        ) from exc
    return np, sitk, bm3d_fn


def _minmax_normalize(arr: Any) -> Tuple[Any, float, float]:
    import numpy as np

    work = np.asarray(arr, dtype=np.float32)
    finite_mask = np.isfinite(work)
    if not np.any(finite_mask):
        return np.zeros_like(work, dtype=np.float32), 0.0, 1.0
    vmin = float(np.min(work[finite_mask]))
    vmax = float(np.max(work[finite_mask]))
    scale = vmax - vmin
    if scale <= 1e-12:
        return np.zeros_like(work, dtype=np.float32), vmin, vmax
    norm = (work - vmin) / scale
    norm = np.clip(norm, 0.0, 1.0)
    norm[~finite_mask] = 0.0
    return norm.astype(np.float32, copy=False), vmin, vmax


def _denormalize(norm: Any, vmin: float, vmax: float) -> Any:
    import numpy as np

    scale = float(vmax) - float(vmin)
    if scale <= 1e-12:
        return np.full_like(np.asarray(norm, dtype=np.float32), fill_value=float(vmin), dtype=np.float32)
    return np.asarray(norm, dtype=np.float32) * scale + float(vmin)


def _denoise_slicewise(norm: Any, sigma_psd: float, bm3d_fn: Any) -> Any:
    import numpy as np

    arr = np.asarray(norm, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"Input must be at least 2D, got shape={arr.shape}")
    if arr.ndim == 2:
        den = bm3d_fn(arr, sigma_psd=float(sigma_psd), profile="np")
        return np.clip(np.asarray(den, dtype=np.float32), 0.0, 1.0)

    flat = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
    out = np.empty_like(flat, dtype=np.float32)
    for i in range(flat.shape[0]):
        den = bm3d_fn(flat[i], sigma_psd=float(sigma_psd), profile="np")
        den_arr = np.asarray(den, dtype=np.float32)
        if den_arr.shape != flat[i].shape:
            raise RuntimeError(
                f"BM3D returned unexpected shape {den_arr.shape}; expected {flat[i].shape}"
            )
        out[i] = np.clip(den_arr, 0.0, 1.0)
    return out.reshape(arr.shape)


def denoise_image_bm3d(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    np, sitk, bm3d_fn = _require_deps()
    t0 = time.time()

    input_path = Path(str(args.get("input_nifti", ""))).expanduser().resolve()
    if not input_path.exists():
        return {"ok": False, "error": f"Input NIfTI not found: {input_path}"}

    sigma_psd = float(args.get("sigma_psd", 0.08))
    if not (sigma_psd > 0.0):
        return {"ok": False, "error": f"sigma_psd must be > 0, got {sigma_psd}"}

    output_subdir = str(args.get("output_subdir", "denoise")).strip() or "denoise"
    out_dir = ctx.artifacts_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    output_nifti_arg = str(args.get("output_nifti") or "").strip()
    if output_nifti_arg:
        output_path = Path(output_nifti_arg).expanduser().resolve()
    else:
        stem = input_path.name.replace(".nii.gz", "").replace(".nii", "")
        output_path = out_dir / f"denoised_{stem}.nii.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = sitk.ReadImage(str(input_path))
    arr_in = sitk.GetArrayFromImage(img)
    if np.asarray(arr_in).size == 0:
        return {"ok": False, "error": f"Input image has no voxels: {input_path}"}

    original_dtype = np.asarray(arr_in).dtype
    norm, vmin, vmax = _minmax_normalize(arr_in)

    if abs(vmax - vmin) <= 1e-12:
        logger.info("[BM3D] Constant image range detected; skipping denoise compute.")
        denorm = np.asarray(arr_in, dtype=np.float32)
    else:
        den_norm = _denoise_slicewise(norm, sigma_psd, bm3d_fn)
        denorm = _denormalize(den_norm, vmin, vmax)

    # Keep floating input dtype when possible; otherwise output float32.
    if np.issubdtype(original_dtype, np.floating):
        out_arr = denorm.astype(original_dtype, copy=False)
    else:
        out_arr = denorm.astype(np.float32, copy=False)

    out_img = sitk.GetImageFromArray(out_arr, isVector=False)
    out_img.CopyInformation(img)
    for k in img.GetMetaDataKeys():
        out_img.SetMetaData(k, img.GetMetaData(k))
    sitk.WriteImage(out_img, str(output_path))

    elapsed = time.time() - t0
    artifacts: List[ArtifactRef] = [
        ArtifactRef(
            path=str(output_path),
            kind="nifti",
            description=f"BM3D-denoised volume (sigma_psd={sigma_psd})",
        )
    ]
    return {
        "ok": True,
        "data": {
            "denoised_nifti": str(output_path),
            "input_nifti": str(input_path),
            "sigma_psd": sigma_psd,
            "original_min": float(vmin),
            "original_max": float(vmax),
            "elapsed_seconds": round(elapsed, 3),
        },
        "generated_artifacts": artifacts,
    }


def build_tool() -> Tool:
    return Tool(spec=BM3D_SPEC, func=denoise_image_bm3d)

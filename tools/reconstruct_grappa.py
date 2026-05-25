"""
Tool: reconstruct_grappa

Dual-mode HDF5 cardiac conversion:
- GRAPPA reconstruction for complex multi-coil k-space.
- Direct H5 image->NIfTI conversion for real-valued image volumes.

Pipeline
--------
1. Read HDF5.
2. Prefer complex k-space path (GRAPPA mode):
   - Locate k-space dataset.
   - Reconstruct frame-by-frame via GRAPPA.
   - IFFT + RSS -> magnitude volume.
3. Fallback image mode:
   - Locate real-valued image dataset (e.g. `image`, `reconstruction_rss`).
   - Convert directly to NIfTI.

Memory management
-----------------
Only one ``(kx, ky, coils)`` complex frame is loaded/reconstructed at a
time.  The final output array is a single contiguous ``float32`` buffer.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool specification
# ---------------------------------------------------------------------------

GRAPPA_SPEC = ToolSpec(
    name="reconstruct_grappa",
    description=(
        "Convert cardiac HDF5 inputs to NIfTI for downstream tools. "
        "If the H5 contains complex k-space, run GRAPPA reconstruction "
        "(pygrappa) frame-by-frame and save magnitude NIfTI. If the H5 "
        "already stores image volumes (e.g. 'image' or 'reconstruction_rss'), "
        "convert directly to NIfTI without GRAPPA."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "h5_path": {
                "type": "string",
                "description": "Path to the HDF5 file containing complex k-space data.",
            },
            "kspace_key": {
                "type": "string",
                "description": (
                    "HDF5 dataset key for the k-space data.  If omitted, the "
                    "tool auto-detects by inspecting top-level keys for common "
                    "names (kspace, rawdata, data, etc.)."
                ),
            },
            "image_key": {
                "type": "string",
                "description": (
                    "HDF5 dataset key for real-valued image volumes. Used when "
                    "the input H5 does not contain complex k-space."
                ),
            },
            "calib_key": {
                "type": "string",
                "description": (
                    "HDF5 dataset key for an explicit ACS/calibration dataset.  "
                    "If omitted, the tool auto-crops the fully-sampled centre of "
                    "each frame's k-space."
                ),
            },
            "acs_lines": {
                "type": "integer",
                "description": (
                    "Number of centre ky lines to use as ACS when auto-cropping "
                    "(default: 24).  Ignored when calib_key is provided."
                ),
                "default": 24,
            },
            "kernel_size": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "GRAPPA kernel size [kx, ky] (default: [5, 5])."
                ),
                "default": [5, 5],
            },
            "coil_axis": {
                "type": "integer",
                "description": (
                    "Axis index of the coil dimension in the k-space array *after* "
                    "the non-spatial leading dimensions have been indexed out "
                    "(i.e. in the per-frame sub-array).  Default: auto-detect."
                ),
            },
            "output_nifti": {
                "type": "string",
                "description": (
                    "Explicit output path.  If omitted, writes "
                    "'reconstructed_cine.nii.gz' in the artifacts directory."
                ),
            },
            "output_subdir": {
                "type": "string",
                "description": "Sub-directory under artifacts_dir for outputs.",
                "default": "grappa",
            },
            "pixel_spacing": {
                "type": "array",
                "items": {"type": "number"},
                "description": (
                    "Pixel spacing [sx, sy, sz] in mm for the output NIfTI.  "
                    "If omitted, the tool tries to parse from H5 attributes and "
                    "falls back to [1.0, 1.0, 1.0]."
                ),
            },
        },
        "required": ["h5_path"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "reconstructed_nifti": {"type": "string"},
            "mode": {"type": "string"},
            "source_key": {"type": "string"},
            "kspace_shape": {"type": "array"},
            "output_shape": {"type": "array"},
            "n_coils": {"type": "integer"},
            "acs_lines_used": {"type": "integer"},
            "elapsed_seconds": {"type": "number"},
        },
    },
    version="0.1.0",
    tags=["reconstruction", "grappa", "kspace", "cardiac", "cine"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_deps():
    """Lazy-import heavy dependencies."""
    try:
        import numpy as np
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            f"Missing core dependency for reconstruct_grappa: {exc}"
        ) from exc
    return np, h5py


def _require_grappa():
    try:
        from pygrappa import grappa  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pygrappa is not installed.  Install via: pip install pygrappa\n"
            f"Error: {exc}"
        ) from exc
    return grappa


def _detect_kspace_key(h5file) -> str:
    """
    Auto-detect the primary complex k-space dataset in an HDF5 file.

    Searches top-level keys for common names used across CMR Challenge,
    fastMRI, OCMR, and similar datasets.
    """
    import numpy as np
    candidates = ["kspace", "rawdata", "data", "kData", "k_space", "raw"]
    # Exact match first
    for c in candidates:
        if c in h5file and hasattr(h5file[c], "dtype"):
            return c
    # Case-insensitive fuzzy match
    for key in h5file.keys():
        if hasattr(h5file[key], "dtype") and np.issubdtype(h5file[key].dtype, np.complexfloating):
            return key
    # Last resort: first dataset that is complex
    for key in h5file.keys():
        obj = h5file[key]
        if hasattr(obj, "dtype") and np.issubdtype(obj.dtype, np.complexfloating):
            return key
    raise RuntimeError(
        f"Cannot auto-detect k-space dataset.  Top-level keys: {list(h5file.keys())}.  "
        "Please specify 'kspace_key' explicitly."
    )


def _detect_image_key(h5file) -> str:
    """
    Auto-detect a real-valued image dataset for non-kspace H5 inputs.
    """
    import numpy as np

    candidates = [
        "image",
        "images",
        "reconstruction_rss",
        "reconstruction",
        "recon",
        "volume",
        "img",
    ]
    for c in candidates:
        if c in h5file and hasattr(h5file[c], "dtype") and hasattr(h5file[c], "shape"):
            dt = h5file[c].dtype
            if np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.integer):
                return c

    for key in h5file.keys():
        obj = h5file[key]
        if not hasattr(obj, "dtype") or not hasattr(obj, "shape"):
            continue
        dt = obj.dtype
        if np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.integer):
            if len(tuple(obj.shape)) >= 2:
                return str(key)

    raise RuntimeError(
        f"Cannot auto-detect image dataset. Top-level keys: {list(h5file.keys())}. "
        "Please specify 'image_key' explicitly."
    )


def _normalize_image_dataset_to_vol(arr, np):
    """
    Normalize image-like arrays to (x, y, z[, t]) volume for NIfTI saving.
    """
    x = np.asarray(arr)
    if x.size == 0:
        raise RuntimeError("Image dataset is empty.")
    if np.iscomplexobj(x):
        x = np.abs(x)
    x = x.astype(np.float32, copy=False)

    if x.ndim == 2:
        # (y, x) -> (x, y, z=1)
        return np.transpose(x, (1, 0))[:, :, np.newaxis]

    if x.ndim == 3:
        # Common medical layout in H5: (z, y, x) -> (x, y, z)
        return np.transpose(x, (2, 1, 0))

    if x.ndim == 4:
        # Heuristic:
        # - (t, z, y, x) -> (x, y, z, t)
        # - (z, y, x, t) -> (x, y, z, t)
        if x.shape[0] <= 64 and x.shape[1] <= 128 and x.shape[2] >= 32 and x.shape[3] >= 32:
            return np.transpose(x, (3, 2, 1, 0))
        return np.transpose(x, (2, 1, 0, 3))

    # Flatten higher dimensions into a trailing time-like axis.
    leading = tuple(int(v) for v in x.shape[:-2])
    trailing = tuple(int(v) for v in x.shape[-2:])
    t = int(np.prod(leading))
    reshaped = x.reshape((t, trailing[0], trailing[1]))
    # (t, y, x) -> (x, y, z=1, t)
    vol = np.transpose(reshaped, (2, 1, 0))
    return vol[:, :, np.newaxis, :]


def _infer_axes(shape: Tuple[int, ...], coil_axis_hint: Optional[int] = None):
    """
    Infer which axes are (kx, ky, coils) and which are non-spatial
    from the k-space dataset shape.

    Convention used by CMR Challenge / fastMRI:
        (slices, phases, coils, kx, ky)   – 5-D cine
        (slices, coils, kx, ky)           – 4-D static
        (coils, kx, ky)                   – 3-D single frame

    The heuristic: the last two axes are always (kx, ky).  The coil axis
    is the smallest remaining dimension (typically 8-32 coils).  Any
    dimensions before coils are non-spatial (slices, phases, …).
    """
    ndim = len(shape)
    if ndim < 3:
        raise ValueError(
            f"K-space must be at least 3-D (coils, kx, ky), got shape {shape}"
        )

    kx_ax = ndim - 2
    ky_ax = ndim - 1

    if coil_axis_hint is not None:
        # User-supplied; validate
        if coil_axis_hint < 0:
            coil_axis_hint = ndim + coil_axis_hint
        if coil_axis_hint < 0 or coil_axis_hint >= ndim:
            raise ValueError(
                f"coil_axis_hint={coil_axis_hint} is out of range for shape {shape}"
            )
        if coil_axis_hint in (kx_ax, ky_ax):
            raise ValueError(
                f"coil_axis_hint={coil_axis_hint} collides with k-space axes "
                f"(kx={kx_ax}, ky={ky_ax}) for shape {shape}"
            )
        coil_ax = coil_axis_hint
    else:
        # Auto-detect: among dims 0..ndim-3, pick the one whose size
        # looks like a coil count (smallest, or first dim ≤ 64).
        remaining = list(range(ndim - 2))
        if not remaining:
            raise ValueError(
                f"K-space shape {shape} has no room for a coil axis."
            )
        # Heuristic: coil dim is the last remaining dim (closest to kx/ky)
        # whose size ≤ 64.  If none qualifies, just pick the last remaining.
        coil_candidates = [i for i in remaining if shape[i] <= 64]
        coil_ax = coil_candidates[-1] if coil_candidates else remaining[-1]

    nonspatial_axes = [i for i in range(ndim) if i not in (kx_ax, ky_ax, coil_ax)]
    return nonspatial_axes, coil_ax, kx_ax, ky_ax


def _build_perm_from_surviving_axes(
    surviving_axes: List[int],
    *,
    kx_ax: int,
    ky_ax: int,
    coil_ax: int,
) -> List[int]:
    """Build transpose permutation from surviving axes to (kx, ky, coils)."""
    role: Dict[int, int] = {}
    for src_idx, ax in enumerate(surviving_axes):
        if ax == kx_ax:
            role[src_idx] = 0
        elif ax == ky_ax:
            role[src_idx] = 1
        elif ax == coil_ax:
            role[src_idx] = 2
    if len(role) != 3:
        raise ValueError(
            f"Cannot build permutation for surviving_axes={surviving_axes}; "
            f"expected kx={kx_ax}, ky={ky_ax}, coil={coil_ax}"
        )
    perm = [0, 0, 0]
    for src, dst in role.items():
        perm[dst] = src
    return perm


def _map_calib_nonspatial_indices(
    *,
    calib_shape: Tuple[int, ...],
    calib_nonspatial_axes: List[int],
    full_shape: Tuple[int, ...],
    full_nonspatial_axes: List[int],
    idx_tuple: Tuple[int, ...],
) -> Dict[int, int]:
    """
    Map calibration non-spatial axes to frame indices from the main k-space frame.

    Handles reduced-rank calib tensors (e.g., missing phase axis) by matching axis
    sizes first, then falling back to left-to-right unused full non-spatial axes.
    """
    if not calib_nonspatial_axes:
        return {}

    full_ns_sizes = [int(full_shape[ax]) for ax in full_nonspatial_axes]
    full_ns_indices = [int(v) for v in idx_tuple]
    if len(full_ns_sizes) != len(full_ns_indices):
        raise ValueError(
            f"Nonspatial index mismatch: full_nonspatial_axes={full_nonspatial_axes}, "
            f"idx_tuple={idx_tuple}"
        )

    if not full_ns_indices:
        # Full data has no non-spatial dimensions; default to first slice for calib dims.
        return {ax: 0 for ax in calib_nonspatial_axes}

    used_full: set[int] = set()
    mapped: Dict[int, int] = {}
    for cal_ax in calib_nonspatial_axes:
        cal_size = int(calib_shape[cal_ax])
        pick_full: Optional[int] = None
        for j, fs in enumerate(full_ns_sizes):
            if j in used_full:
                continue
            if fs == cal_size:
                pick_full = j
                break
        if pick_full is None:
            for j in range(len(full_ns_sizes)):
                if j not in used_full:
                    pick_full = j
                    break
        if pick_full is None:
            raise ValueError(
                f"Cannot map calib nonspatial axis {cal_ax} (size={cal_size}) "
                f"to full nonspatial axes {full_nonspatial_axes}"
            )
        used_full.add(pick_full)
        mapped[cal_ax] = int(full_ns_indices[pick_full])
    return mapped


def _infer_3d_calib_axes(shape: Tuple[int, ...], *, n_coils: int, kx_size: int) -> Tuple[int, int, int]:
    """
    Infer (kx, ky, coil) axes for a 3-D global calibration tensor.

    We prioritize an axis matching the known coil count, then pick kx by exact
    size match to the main k-space kx size (or the larger remaining axis).
    """
    if len(shape) != 3:
        raise ValueError(f"Expected 3-D calibration tensor, got shape={shape}")

    coil_candidates = [i for i, s in enumerate(shape) if int(s) == int(n_coils)]
    coil_ax = coil_candidates[-1] if coil_candidates else int(min(range(3), key=lambda i: shape[i]))
    rem = [i for i in range(3) if i != coil_ax]

    kx_matches = [i for i in rem if int(shape[i]) == int(kx_size)]
    if kx_matches:
        kx_ax = kx_matches[0]
    else:
        kx_ax = max(rem, key=lambda i: int(shape[i]))
    ky_ax = rem[0] if rem[1] == kx_ax else rem[1]
    return int(kx_ax), int(ky_ax), int(coil_ax)


def _extract_calib_frame(
    *,
    calib_ds: Any,
    idx_tuple: Tuple[int, ...],
    full_shape: Tuple[int, ...],
    full_nonspatial_axes: List[int],
    n_coils: int,
    kx_size: int,
) -> Any:
    """
    Extract one calibration frame and return it as (kx, ky_calib, coils).

    Supports:
    - full-rank calib tensors (same rank as main k-space),
    - reduced-rank calib tensors (subset of nonspatial axes),
    - global 3-D calib tensors.
    """
    import numpy as np

    calib_shape = tuple(calib_ds.shape)
    if len(calib_shape) < 3:
        raise ValueError(f"Calibration tensor must be >=3D, got shape={calib_shape}")

    if len(calib_shape) == 3:
        ckx_ax, cky_ax, ccoil_ax = _infer_3d_calib_axes(calib_shape, n_coils=n_coils, kx_size=kx_size)
        calib_raw = calib_ds[...]
        perm = _build_perm_from_surviving_axes(
            [0, 1, 2],
            kx_ax=ckx_ax,
            ky_ax=cky_ax,
            coil_ax=ccoil_ax,
        )
        return np.transpose(calib_raw, perm).astype(np.complex128)

    calib_nonspatial, calib_coil_ax, calib_kx_ax, calib_ky_ax = _infer_axes(calib_shape)
    selector: List[Any] = [slice(None)] * len(calib_shape)
    mapped = _map_calib_nonspatial_indices(
        calib_shape=calib_shape,
        calib_nonspatial_axes=calib_nonspatial,
        full_shape=full_shape,
        full_nonspatial_axes=full_nonspatial_axes,
        idx_tuple=idx_tuple,
    )
    for cal_ax, cal_idx in mapped.items():
        selector[cal_ax] = int(cal_idx)

    calib_raw = calib_ds[tuple(selector)]
    calib_raw = np.asarray(calib_raw)
    if calib_raw.ndim != 3:
        raise RuntimeError(
            f"Calibration frame must be 3-D after indexing, got shape={calib_raw.shape} "
            f"(source shape={calib_shape}, selector={selector})"
        )

    surviving_axes = [ax for ax, sel in enumerate(selector) if isinstance(sel, slice)]
    perm = _build_perm_from_surviving_axes(
        surviving_axes,
        kx_ax=calib_kx_ax,
        ky_ax=calib_ky_ax,
        coil_ax=calib_coil_ax,
    )
    return np.transpose(calib_raw, perm).astype(np.complex128)


def _auto_acs_region(frame, acs_lines: int):
    """
    Extract the fully-sampled centre of k-space as calibration data.

    Parameters
    ----------
    frame : ndarray, shape (kx, ky, coils)
    acs_lines : int  –  number of ky lines to extract

    Returns
    -------
    calib : ndarray, shape (kx, acs_lines, coils)
    """
    import numpy as np
    ky = frame.shape[1]
    half = acs_lines // 2
    centre = ky // 2
    start = max(0, centre - half)
    end = min(ky, start + acs_lines)
    calib = frame[:, start:end, :].copy()
    # Sanity: verify the ACS region is actually sampled
    energy = np.sum(np.abs(calib))
    if energy < 1e-12:
        logger.warning(
            "[GRAPPA] ACS region appears empty (energy=%.2e).  "
            "Reconstruction quality may be poor.", energy,
        )
    return calib


def _is_undersampled(frame, threshold: float = 0.90):
    """
    Return True if the fraction of sampled ky lines is below *threshold*.

    Parameters
    ----------
    frame : ndarray, shape (kx, ky, coils)
    """
    import numpy as np
    ky_energy = np.sum(np.abs(frame), axis=(0, 2))  # (ky,)
    frac = np.mean(ky_energy > 0)
    return float(frac) < threshold


def _ifft2_rss(kspace_frame):
    """
    IFFT + RSS coil combination for a single (kx, ky, coils) frame.

    Returns a 2-D float32 magnitude image (kx, ky).
    """
    import numpy as np
    # kspace_frame: (kx, ky, coils)
    img_coils = np.fft.ifftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace_frame, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )
    # RSS
    magnitude = np.sqrt(np.sum(np.abs(img_coils) ** 2, axis=-1)).astype(np.float32)
    return magnitude


def _parse_spacing(h5file, pixel_spacing_arg: Optional[list]) -> List[float]:
    """Try to get voxel spacing from H5 attrs or user arg; fall back to 1mm."""
    if pixel_spacing_arg:
        sp = [float(s) for s in pixel_spacing_arg]
        while len(sp) < 3:
            sp.append(1.0)
        return sp[:3]
    # Try common attribute names
    for attr_name in ("pixel_spacing", "pixelSpacing", "spacing",
                      "voxel_size", "resolution"):
        if attr_name in h5file.attrs:
            val = h5file.attrs[attr_name]
            try:
                sp = [float(v) for v in val]
                while len(sp) < 3:
                    sp.append(1.0)
                return sp[:3]
            except Exception:
                pass
    return [1.0, 1.0, 1.0]


def _save_volume_as_nifti(*, vol, spacing: List[float], output_path: Path, np) -> Optional[str]:
    """
    Save volume shaped as (x, y, z[, t]) to NIfTI.
    Returns None on success, or error message on failure.
    """
    try:
        import SimpleITK as sitk  # type: ignore
    except ImportError:
        try:
            import nibabel as nib  # type: ignore
            nii_affine = np.eye(4)
            for i, sp in enumerate(spacing[: min(3, len(spacing))]):
                nii_affine[i, i] = sp
            nii_img = nib.Nifti1Image(vol, affine=nii_affine)
            nib.save(nii_img, str(output_path))
            return None
        except ImportError:
            return "Neither SimpleITK nor nibabel installed; cannot save NIfTI."
    else:
        if vol.ndim == 3:
            # (x, y, z) -> (z, y, x)
            sitk_arr = np.transpose(vol, (2, 1, 0)).copy()
        elif vol.ndim == 4:
            # (x, y, z, t) -> (t, z, y, x)
            sitk_arr = np.transpose(vol, (3, 2, 1, 0)).copy()
        else:
            trailing = int(np.prod(vol.shape[2:]))
            vol_r = vol.reshape(vol.shape[0], vol.shape[1], trailing)
            sitk_arr = np.transpose(vol_r, (2, 1, 0)).copy()

        sitk_img = sitk.GetImageFromArray(sitk_arr, isVector=False)
        ndim = sitk_img.GetDimension()
        padded_spacing = list(spacing[:ndim])
        while len(padded_spacing) < ndim:
            padded_spacing.append(1.0)
        sitk_img.SetSpacing(padded_spacing[:ndim])
        sitk_img.SetOrigin([0.0] * ndim)
        sitk.WriteImage(sitk_img, str(output_path))
        return None


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------

def reconstruct_grappa(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    """Convert HDF5 cardiac input to NIfTI (GRAPPA mode or image passthrough mode)."""
    t0 = time.time()
    np, h5py = _require_deps()

    h5_path = Path(args["h5_path"]).expanduser().resolve()
    if not h5_path.exists():
        return {"ok": False, "error": f"HDF5 file not found: {h5_path}"}

    acs_lines = int(args.get("acs_lines", 24))
    kernel_size_arg = args.get("kernel_size", [5, 5])
    kernel_size = tuple(int(k) for k in kernel_size_arg)
    coil_axis_hint = args.get("coil_axis")
    if coil_axis_hint is not None:
        coil_axis_hint = int(coil_axis_hint)
    calib_key = args.get("calib_key")
    kspace_key_arg = args.get("kspace_key")
    image_key_arg = args.get("image_key")

    output_subdir_raw = str(args.get("output_subdir", "grappa") or "grappa").strip()
    output_subdir = output_subdir_raw.replace("\\", "/")
    if output_subdir.startswith("./"):
        output_subdir = output_subdir[2:]
    if output_subdir.startswith("artifacts/"):
        output_subdir = output_subdir[len("artifacts/") :]
    if output_subdir.startswith("/"):
        # Absolute paths are not valid artifact subdirs; collapse to basename.
        output_subdir = Path(output_subdir).name
    if (".." in output_subdir) or (not output_subdir):
        output_subdir = "grappa"

    out_dir = ctx.artifacts_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    output_nifti_arg = args.get("output_nifti")
    if output_nifti_arg:
        output_nifti_s = str(output_nifti_arg).strip()
        output_nifti_p = Path(output_nifti_s).expanduser()
        if output_nifti_p.is_absolute():
            output_path = output_nifti_p.resolve()
        else:
            output_path = (out_dir / output_nifti_p).resolve()
    else:
        stem = h5_path.stem.replace(".h5", "")
        output_path = out_dir / f"reconstructed_{stem}.nii.gz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        h5 = h5py.File(str(h5_path), "r")
    except Exception as exc:
        return {"ok": False, "error": f"Failed to open HDF5: {exc}"}

    mode = "grappa"
    source_key = ""
    output_vol = None
    full_shape: Tuple[int, ...] = tuple()
    n_coils = 0
    applied_grappa = 0
    skipped_grappa = 0
    spacing: List[float] = [1.0, 1.0, 1.0]

    try:
        spacing = _parse_spacing(h5, args.get("pixel_spacing"))

        ks_key: Optional[str] = None
        if kspace_key_arg:
            try:
                cand = h5[str(kspace_key_arg)]
                if np.issubdtype(cand.dtype, np.complexfloating):
                    ks_key = str(kspace_key_arg)
                else:
                    mode = "image_passthrough"
                    source_key = str(kspace_key_arg)
            except Exception:
                ks_key = None
        if ks_key is None and mode == "grappa":
            try:
                ks_key = _detect_kspace_key(h5)
            except Exception:
                mode = "image_passthrough"

        grappa_fn = None
        if mode == "grappa" and ks_key is not None:
            try:
                grappa_fn = _require_grappa()
            except Exception as exc:
                fallback_key = str(image_key_arg or "").strip()
                if not fallback_key and "reconstruction_rss" in h5:
                    fallback_key = "reconstruction_rss"
                if not fallback_key:
                    try:
                        fallback_key = _detect_image_key(h5)
                    except Exception:
                        raise exc
                mode = "image_passthrough"
                source_key = str(fallback_key)
                logger.warning(
                    "[GRAPPA] pygrappa unavailable (%s: %s), falling back to image passthrough key=%s",
                    type(exc).__name__,
                    str(exc),
                    source_key,
                )

        if mode == "grappa" and ks_key is not None:
            source_key = str(ks_key)
            ks_ds = h5[ks_key]
            full_shape = tuple(ks_ds.shape)
            full_dtype = ks_ds.dtype
            logger.info(
                "[GRAPPA] mode=grappa key=%s shape=%s dtype=%s",
                ks_key, full_shape, full_dtype,
            )

            calib_ds = None
            if calib_key and calib_key in h5:
                calib_ds = h5[calib_key]
                logger.info("[GRAPPA] Using explicit calib key=%s shape=%s", calib_key, calib_ds.shape)

            nonspatial, coil_ax, kx_ax, ky_ax = _infer_axes(full_shape, coil_axis_hint)
            n_coils = int(full_shape[coil_ax])
            kx_size = int(full_shape[kx_ax])
            ky_size = int(full_shape[ky_ax])
            logger.info(
                "[GRAPPA] Axes: nonspatial=%s coil=%d(n=%d) kx=%d(%d) ky=%d(%d)",
                nonspatial, coil_ax, n_coils, kx_ax, kx_size, ky_ax, ky_size,
            )

            import itertools

            ns_ranges = [range(full_shape[ax]) for ax in nonspatial]
            frame_indices = list(itertools.product(*ns_ranges))
            n_frames = len(frame_indices) if frame_indices else 1
            logger.info("[GRAPPA] Total frames to reconstruct: %d", n_frames)

            ns_sizes = tuple(int(full_shape[ax]) for ax in nonspatial)
            out_shape = (kx_size, ky_size) + ns_sizes
            magnitude_vol = np.zeros(out_shape, dtype=np.float32)

            for fi, idx_tuple in enumerate(frame_indices):
                selector = [None] * len(full_shape)
                ns_iter = iter(idx_tuple)
                for ax in range(len(full_shape)):
                    if ax in (kx_ax, ky_ax, coil_ax):
                        selector[ax] = slice(None)
                    else:
                        selector[ax] = next(ns_iter)
                frame_raw = ks_ds[tuple(selector)]

                surviving = [ax for ax in range(len(full_shape)) if ax in (kx_ax, ky_ax, coil_ax)]
                perm = _build_perm_from_surviving_axes(
                    surviving,
                    kx_ax=kx_ax,
                    ky_ax=ky_ax,
                    coil_ax=coil_ax,
                )
                frame = np.transpose(frame_raw, perm).astype(np.complex128)

                if _is_undersampled(frame):
                    if calib_ds is not None:
                        try:
                            calib = _extract_calib_frame(
                                calib_ds=calib_ds,
                                idx_tuple=idx_tuple,
                                full_shape=full_shape,
                                full_nonspatial_axes=nonspatial,
                                n_coils=n_coils,
                                kx_size=kx_size,
                            )
                        except Exception as exc:
                            logger.warning(
                                "[GRAPPA] calib_key frame %d/%d failed: %s. Falling back to auto-ACS.",
                                fi + 1,
                                n_frames,
                                exc,
                            )
                            calib = _auto_acs_region(frame, acs_lines)
                    else:
                        calib = _auto_acs_region(frame, acs_lines)

                    try:
                        recon = grappa_fn(frame, calib, kernel_size=kernel_size, coil_axis=-1)
                        applied_grappa += 1
                    except Exception as exc:
                        logger.warning(
                            "[GRAPPA] pygrappa failed frame %d/%d: %s. Using zero-filled fallback.",
                            fi + 1,
                            n_frames,
                            exc,
                        )
                        recon = frame
                else:
                    recon = frame
                    skipped_grappa += 1

                mag = _ifft2_rss(recon)
                out_idx = (slice(None), slice(None)) + idx_tuple
                magnitude_vol[out_idx] = mag

                if (fi + 1) % max(1, n_frames // 5) == 0 or fi == n_frames - 1:
                    logger.info("[GRAPPA] Reconstructed %d/%d frames", fi + 1, n_frames)

            output_vol = magnitude_vol
            while output_vol.ndim > 3 and output_vol.shape[-1] == 1:
                output_vol = output_vol[..., 0]
        else:
            mode = "image_passthrough"
            image_key = str(image_key_arg or "").strip()
            if not image_key:
                image_key = _detect_image_key(h5)
            ds = h5[image_key]
            source_key = str(image_key)
            raw = ds[...]
            full_shape = tuple(raw.shape)
            output_vol = _normalize_image_dataset_to_vol(raw, np)
            logger.info(
                "[GRAPPA] mode=image_passthrough key=%s shape=%s -> output_shape=%s",
                image_key,
                full_shape,
                tuple(output_vol.shape),
            )
    except Exception as exc:
        tb = traceback.format_exc(limit=2)
        return {"ok": False, "error": f"reconstruct_grappa failed: {type(exc).__name__}: {exc}", "traceback": tb}
    finally:
        h5.close()

    if output_vol is None:
        return {"ok": False, "error": "No output volume was produced from HDF5 input."}

    save_err = _save_volume_as_nifti(vol=output_vol, spacing=spacing, output_path=output_path, np=np)
    if save_err:
        return {"ok": False, "error": save_err}

    elapsed = time.time() - t0
    logger.info("[GRAPPA] Saved %s in %.1fs (mode=%s)", output_path, elapsed, mode)

    if mode == "grappa":
        desc = f"GRAPPA-reconstructed magnitude image (coils={n_coils}, kernel={kernel_size})"
    else:
        desc = f"H5 image dataset converted to NIfTI (key={source_key})"
    artifacts: List[ArtifactRef] = [
        ArtifactRef(path=str(output_path), kind="nifti", description=desc),
    ]

    out_data: Dict[str, Any] = {
        "reconstructed_nifti": str(output_path),
        "h5_path": str(h5_path),
        "mode": mode,
        "source_key": source_key,
        "output_shape": list(output_vol.shape),
        "pixel_spacing": spacing,
        "elapsed_seconds": round(elapsed, 2),
    }
    if full_shape:
        out_data["kspace_shape"] = list(full_shape)
    if mode == "grappa":
        out_data.update(
            {
                "kspace_key": source_key,
                "n_coils": int(n_coils),
                "acs_lines_used": int(acs_lines),
                "grappa_applied_frames": int(applied_grappa),
                "grappa_skipped_frames": int(skipped_grappa),
                "kernel_size": list(kernel_size),
            }
        )
    else:
        out_data.update({"image_key": source_key})

    return {
        "ok": True,
        "data": out_data,
        "generated_artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# build_tool() – entry-point for the tool registry
# ---------------------------------------------------------------------------

def build_tool() -> Tool:
    return Tool(spec=GRAPPA_SPEC, func=reconstruct_grappa)

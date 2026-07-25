"""
Tool: generate_qa_snapshot

Universal QA visualisation tool for NIfTI volumes.

Loads a single NIfTI file, extracts a representative 2-D slice (handling
3-D and 4-D data), and saves a grayscale PNG snapshot.  Designed as a
robust, deterministic replacement for sandbox_exec-based visualisation.

4-D handling (e.g. cardiac cine):
    1. Select the centre frame along the 4th dimension (time/phase).
    2. From that 3-D sub-volume, select the centre axial slice.

3-D handling:
    Select the centre axial slice directly.

Mask-centred framing:
    When a mask is supplied and the selected slice actually carries labels, the
    preview is cropped to the labelled region plus a margin so the structure
    fills the frame instead of being a few dozen pixels inside a full field of
    view.  Physical proportions are preserved (the crop box is squared in
    millimetres, and the voxel-spacing aspect is still applied), and the crop
    degrades to the full field of view whenever there is no mask, the mask is
    empty on this slice, or the mask grid does not match the anatomy grid.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec

logger = logging.getLogger(__name__)

QA_SNAPSHOT_SPEC = ToolSpec(
    name="generate_qa_snapshot",
    description=(
        "Load a NIfTI volume and save a grayscale PNG of a representative "
        "centre slice.  Handles 3-D and 4-D (cine / time-series) data "
        "automatically.  Optionally overlays a segmentation mask with a "
        "transparent colour map.  Useful as a quick QA check after "
        "reconstruction, segmentation, or any processing step."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "input_nifti": {
                "type": "string",
                "description": "Path to the input NIfTI volume.",
            },
            "mask_nifti": {
                "type": "string",
                "description": (
                    "Optional path to a segmentation mask NIfTI.  If provided, "
                    "the mask is loaded, the same centre slice/frame is extracted, "
                    "and non-zero labels are overlaid on the anatomy with a "
                    "transparent colour map."
                ),
            },
            "seg_dir": {
                "type": "string",
                "description": (
                    "Optional directory of per-frame segmentation NIfTI files "
                    "(e.g. *_f01.nii.gz \u2026 *_f11.nii.gz).  When the anatomy is "
                    "4-D and a centre frame is selected, the tool auto-picks the "
                    "per-frame seg file matching that frame index for the mask "
                    "overlay.  Takes priority over mask_nifti when available."
                ),
            },
            "crop_to_mask": {
                "type": "boolean",
                "description": (
                    "Crop the preview to the labelled region plus a margin so the "
                    "segmented structure fills the frame (default: true).  Only has "
                    "any effect when a mask is supplied AND the selected slice "
                    "carries non-zero labels; with no mask, an empty mask, or a mask "
                    "whose grid does not match the anatomy, the full field of view is "
                    "rendered exactly as before.  Set false to force the full field "
                    "of view even when a mask is present."
                ),
                "default": True,
            },
            "crop_margin_frac": {
                "type": "number",
                "description": (
                    "Margin added around the mask bounding box before cropping, as a "
                    "fraction of the box's longer physical side (default: 0.35)."
                ),
                "default": 0.35,
            },
            "title": {
                "type": "string",
                "description": "Optional title displayed on the snapshot.",
                "default": "",
            },
            "output_png": {
                "type": "string",
                "description": (
                    "Optional explicit output path for the PNG.  If omitted, "
                    "writes 'qa_snapshot.png' under artifacts/<output_subdir>."
                ),
            },
            "output_subdir": {
                "type": "string",
                "description": "Sub-directory under artifacts_dir for outputs.",
                "default": "qa",
            },
        },
        "required": ["input_nifti"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "output_png": {"type": "string"},
            "input_nifti": {"type": "string"},
            "mask_nifti": {"type": "string"},
            "volume_shape": {"type": "array"},
            "selected_frame": {"type": "integer"},
            "selected_slice": {"type": "integer"},
            "crop": {"type": "object"},
            "png_size": {"type": "array"},
            "elapsed_seconds": {"type": "number"},
        },
    },
    version="0.3.0",
    tags=["qa", "visualisation", "snapshot", "overlay"],
)


def _require_deps():
    """Lazy-import heavy dependencies."""
    try:
        import numpy as np  # type: ignore
        import nibabel as nib  # type: ignore
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency for generate_qa_snapshot.  "
            "Install with: pip install nibabel matplotlib numpy"
        ) from exc
    return np, nib, plt


def compute_mask_crop_box(
    mask_2d,
    *,
    np,
    spacing_xy: tuple,
    margin_frac: float = 0.35,
    min_size: int = 8,
):
    """Bounding box of the labelled region, padded and squared in millimetres.

    Parameters
    ----------
    mask_2d : 2-D array in (x, y) index order, same grid as the anatomy slice.
    spacing_xy : (sx, sy) voxel size in mm; used so the box is square in
        *physical* space rather than in voxels, which keeps the anatomy from
        being framed into a sliver on anisotropic grids.
    margin_frac : padding around the label box as a fraction of its longer
        physical side.
    min_size : refuse to return a box smaller than this in either axis; the
        caller then falls back to the full field of view.

    Returns ``(x0, x1, y0, y1)`` as half-open index bounds, or ``None`` when
    there is nothing to crop to (empty mask, degenerate result).
    """
    if mask_2d is None or getattr(mask_2d, "ndim", 0) != 2:
        return None
    nx, ny = int(mask_2d.shape[0]), int(mask_2d.shape[1])
    if nx < min_size or ny < min_size:
        return None

    labelled = np.asarray(mask_2d) > 0
    if not bool(labelled.any()):
        return None

    xs = np.flatnonzero(labelled.any(axis=1))
    ys = np.flatnonzero(labelled.any(axis=0))
    x0, x1 = int(xs[0]), int(xs[-1]) + 1
    y0, y1 = int(ys[0]), int(ys[-1]) + 1

    sx = float(spacing_xy[0]) if spacing_xy and spacing_xy[0] > 1e-9 else 1.0
    sy = float(spacing_xy[1]) if len(spacing_xy) > 1 and spacing_xy[1] > 1e-9 else 1.0

    width_mm = (x1 - x0) * sx
    height_mm = (y1 - y0) * sy
    side_mm = max(width_mm, height_mm)
    if side_mm <= 0:
        return None
    # Margin first, then square the box in millimetres so the rendered frame is
    # not stretched: matplotlib still draws it with the physical aspect ratio.
    side_mm *= 1.0 + 2.0 * max(0.0, float(margin_frac))

    def _expand(lo: int, hi: int, n: int, spacing: float) -> tuple:
        want = int(np.ceil(side_mm / spacing))
        want = max(want, hi - lo)
        want = min(want, n)
        centre = (lo + hi) / 2.0
        new_lo = int(round(centre - want / 2.0))
        new_lo = max(0, min(new_lo, n - want))
        return new_lo, new_lo + want

    cx0, cx1 = _expand(x0, x1, nx, sx)
    cy0, cy1 = _expand(y0, y1, ny, sy)

    if (cx1 - cx0) < min_size or (cy1 - cy0) < min_size:
        return None
    if (cx1 - cx0) >= nx and (cy1 - cy0) >= ny:
        # Box already covers everything: nothing gained, and reporting it as a
        # crop would overstate what happened.
        return None
    return cx0, cx1, cy0, cy1


def _label_overlay_cmap(plt, np, n_labels: int, alpha: float = 0.45):
    """Discrete colour map where label 0 is transparent and 1+ are distinct."""
    from matplotlib.colors import ListedColormap  # noqa: E402

    try:
        base_cmap = plt.get_cmap("tab10")
    except Exception:  # pragma: no cover - very old matplotlib
        base_cmap = plt.cm.get_cmap("tab10", 10)
    overlay_colors = np.zeros((n_labels, 4))
    for lbl in range(1, n_labels):
        # Integer indices address the 10-entry lookup table directly.
        rgba = base_cmap((lbl - 1) % 10)
        overlay_colors[lbl] = (*rgba[:3], alpha)
    return ListedColormap(overlay_colors)


def generate_qa_snapshot(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    np, nib, plt = _require_deps()
    t0 = time.time()

    input_path = Path(str(args.get("input_nifti", ""))).expanduser().resolve()
    if not input_path.exists():
        return {"ok": False, "error": f"Input NIfTI not found: {input_path}"}

    # ---- Resolve mask: prefer seg_dir (frame-matched) over mask_nifti ----
    mask_path_str = str(args.get("mask_nifti") or "").strip()
    mask_path: Path | None = None
    seg_dir_raw = str(args.get("seg_dir") or "").strip()
    seg_dir_path: Path | None = None
    if seg_dir_raw:
        seg_dir_path = Path(seg_dir_raw).expanduser().resolve()
        if not seg_dir_path.is_dir():
            logger.warning("seg_dir not found or not a directory (%s); ignoring.", seg_dir_path)
            seg_dir_path = None
    if mask_path_str:
        mask_path = Path(mask_path_str).expanduser().resolve()
        if not mask_path.exists():
            logger.warning("Mask NIfTI not found (%s); overlay will be skipped.", mask_path)
            mask_path = None

    crop_to_mask = bool(args.get("crop_to_mask", True))
    try:
        crop_margin_frac = float(args.get("crop_margin_frac", 0.35))
    except (TypeError, ValueError):
        return {
            "ok": False,
            "error": f"crop_margin_frac must be a number, got {args.get('crop_margin_frac')!r}",
        }
    if crop_margin_frac < 0:
        return {"ok": False, "error": f"crop_margin_frac must be >= 0, got {crop_margin_frac}"}

    title = str(args.get("title", "")).strip()
    output_subdir = str(args.get("output_subdir", "qa")).strip() or "qa"
    out_dir = ctx.artifacts_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    output_png_arg = str(args.get("output_png") or "").strip()
    if output_png_arg:
        output_path = Path(output_png_arg).expanduser().resolve()
    else:
        output_path = out_dir / "qa_snapshot.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load with nibabel (robust for any dimensionality)
    img = nib.load(str(input_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    vol_shape = list(data.shape)

    selected_frame = -1
    selected_slice = 0

    # 4-D: pick centre frame along dim-3 (time/phase), then centre Z-slice
    if data.ndim == 4:
        selected_frame = data.shape[3] // 2
        vol3d = data[:, :, :, selected_frame]
    elif data.ndim >= 5:
        # Collapse trailing dims, take first
        vol3d = data.reshape(data.shape[0], data.shape[1], data.shape[2], -1)
        selected_frame = vol3d.shape[3] // 2
        vol3d = vol3d[:, :, :, selected_frame]
    elif data.ndim == 3:
        vol3d = data
    elif data.ndim == 2:
        vol3d = data
    else:
        return {"ok": False, "error": f"Unexpected dimensionality: {data.ndim}"}

    # Extract centre axial slice
    # nibabel uses (x, y, z) ordering → axis 2 is Z
    if vol3d.ndim == 3:
        selected_slice = vol3d.shape[2] // 2
        slice_2d = vol3d[:, :, selected_slice]
    else:
        # 2-D already
        slice_2d = vol3d
        selected_slice = 0

    # ---- Mask overlay slice extraction (same frame/slice indices) ----
    # When seg_dir is available and anatomy is 4-D, auto-pick the
    # per-frame seg file that matches the selected anatomy frame.
    mask_slice_2d = None
    effective_mask_path = mask_path  # for metadata reporting
    if seg_dir_path is not None and selected_frame >= 0:
        frame_files = sorted(seg_dir_path.glob("*.nii.gz"))
        if frame_files:
            # Map 0-based selected_frame to sorted file list index
            ff_idx = min(selected_frame, len(frame_files) - 1)
            effective_mask_path = frame_files[ff_idx]
            logger.info(
                "seg_dir: selected frame-matched mask %s (frame=%d, file_idx=%d of %d)",
                effective_mask_path.name, selected_frame, ff_idx, len(frame_files),
            )
            try:
                mask_img = nib.load(str(effective_mask_path))
                mask_data = np.asarray(mask_img.dataobj, dtype=np.float32)
                # Per-frame file is 3-D (x, y, z)
                if mask_data.ndim == 3:
                    _ms = min(selected_slice, mask_data.shape[2] - 1)
                    mask_slice_2d = mask_data[:, :, _ms]
                elif mask_data.ndim == 2:
                    mask_slice_2d = mask_data
                else:
                    # 4-D frame file (unlikely but handle gracefully)
                    _mf = min(selected_frame, mask_data.shape[3] - 1) if mask_data.ndim >= 4 else 0
                    mask_vol3d = mask_data[:, :, :, _mf] if mask_data.ndim >= 4 else mask_data
                    if mask_vol3d.ndim == 3:
                        _ms = min(selected_slice, mask_vol3d.shape[2] - 1)
                        mask_slice_2d = mask_vol3d[:, :, _ms]
            except Exception as exc:
                logger.warning("Failed to load frame-matched mask from seg_dir: %s", exc)

    # Fallback to mask_nifti if seg_dir didn't produce a slice
    if mask_slice_2d is None and mask_path is not None:
        try:
            mask_img = nib.load(str(mask_path))
            mask_data = np.asarray(mask_img.dataobj, dtype=np.float32)
            if mask_data.ndim == 4 and selected_frame >= 0:
                mask_vol3d = mask_data[:, :, :, min(selected_frame, mask_data.shape[3] - 1)]
            elif mask_data.ndim >= 5:
                mask_flat = mask_data.reshape(mask_data.shape[0], mask_data.shape[1], mask_data.shape[2], -1)
                _mf = min(selected_frame, mask_flat.shape[3] - 1) if selected_frame >= 0 else 0
                mask_vol3d = mask_flat[:, :, :, _mf]
            elif mask_data.ndim == 3:
                mask_vol3d = mask_data
            elif mask_data.ndim == 2:
                mask_vol3d = mask_data
            else:
                mask_vol3d = None

            if mask_vol3d is not None:
                if mask_vol3d.ndim == 3:
                    _ms = min(selected_slice, mask_vol3d.shape[2] - 1)
                    mask_slice_2d = mask_vol3d[:, :, _ms]
                else:
                    mask_slice_2d = mask_vol3d
        except Exception as exc:
            logger.warning("Failed to load mask NIfTI for overlay: %s", exc)

    # Compute aspect ratio from voxel spacing
    zooms = img.header.get_zooms() if hasattr(img.header, "get_zooms") else ()
    sp_x = float(zooms[0]) if len(zooms) > 0 else 1.0
    sp_y = float(zooms[1]) if len(zooms) > 1 else 1.0
    aspect = sp_y / sp_x if sp_x > 1e-12 else 1.0

    # ---- Crop the frame to the labelled region so the structure fills it ----
    # Every branch that declines to crop records why, and leaves the full field
    # of view exactly as previous versions rendered it.
    crop_meta: Dict[str, Any] = {
        "applied": False,
        "requested": crop_to_mask,
        "margin_frac": crop_margin_frac,
        "shape_before": [int(slice_2d.shape[0]), int(slice_2d.shape[1])],
        "shape_after": [int(slice_2d.shape[0]), int(slice_2d.shape[1])],
    }
    if mask_slice_2d is not None:
        labelled_before = int(np.count_nonzero(np.asarray(mask_slice_2d) > 0))
        crop_meta["labelled_voxels"] = labelled_before
        crop_meta["labelled_fraction_before"] = round(
            labelled_before / float(max(1, mask_slice_2d.size)), 6
        )

    if not crop_to_mask:
        crop_meta["reason"] = "crop_to_mask=false"
    elif mask_slice_2d is None:
        crop_meta["reason"] = "no mask slice available"
    elif tuple(mask_slice_2d.shape) != tuple(slice_2d.shape):
        crop_meta["reason"] = (
            f"mask grid {tuple(int(v) for v in mask_slice_2d.shape)} does not match anatomy grid "
            f"{tuple(int(v) for v in slice_2d.shape)}"
        )
        logger.warning("Mask-centred crop skipped: %s", crop_meta["reason"])
    else:
        box = compute_mask_crop_box(
            mask_slice_2d,
            np=np,
            spacing_xy=(sp_x, sp_y),
            margin_frac=crop_margin_frac,
        )
        if box is None:
            crop_meta["reason"] = (
                "mask is empty on the selected slice"
                if not bool(np.any(np.asarray(mask_slice_2d) > 0))
                else "mask bounding box plus margin already covers the whole frame"
            )
        else:
            x0, x1, y0, y1 = box
            slice_2d = slice_2d[x0:x1, y0:y1]
            mask_slice_2d = mask_slice_2d[x0:x1, y0:y1]
            crop_meta.update(
                {
                    "applied": True,
                    "bbox_x": [int(x0), int(x1)],
                    "bbox_y": [int(y0), int(y1)],
                    "shape_after": [int(x1 - x0), int(y1 - y0)],
                    "labelled_fraction_after": round(
                        int(np.count_nonzero(np.asarray(mask_slice_2d) > 0))
                        / float(max(1, mask_slice_2d.size)),
                        6,
                    ),
                }
            )
            logger.info(
                "Mask-centred crop: %s -> %s (labelled fraction %.4f -> %.4f)",
                crop_meta["shape_before"],
                crop_meta["shape_after"],
                crop_meta.get("labelled_fraction_before", 0.0),
                crop_meta["labelled_fraction_after"],
            )

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(slice_2d.T, cmap="gray", origin="lower", aspect=aspect)

    # Overlay mask with transparent colour map
    if mask_slice_2d is not None:
        mask_t = mask_slice_2d.T
        # Build a colour map: label 0 → fully transparent; labels 1+ → distinct colours
        unique_labels = np.unique(mask_t)
        n_labels = max(int(unique_labels.max()), 1) + 1
        overlay_cmap = _label_overlay_cmap(plt, np, n_labels)
        ax.imshow(
            mask_t,
            cmap=overlay_cmap,
            vmin=0,
            vmax=n_labels - 1,
            origin="lower",
            aspect=aspect,
            interpolation="nearest",
        )

    if title:
        ax.set_title(title, fontsize=12)
    else:
        stem = input_path.name.replace(".nii.gz", "").replace(".nii", "")
        info_parts = [stem]
        if selected_frame >= 0:
            info_parts.append(f"frame={selected_frame}")
        info_parts.append(f"slice={selected_slice}")
        if mask_slice_2d is not None:
            info_parts.append("mask overlay")
        if crop_meta.get("applied"):
            info_parts.append(
                "cropped to mask {}x{}".format(*crop_meta["shape_after"])
            )
        ax.set_title(" | ".join(info_parts), fontsize=10)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    png_size: List[int] = []
    try:
        from PIL import Image  # type: ignore

        with Image.open(str(output_path)) as im:
            png_size = [int(im.size[0]), int(im.size[1])]
    except Exception as exc:  # pragma: no cover - Pillow ships with matplotlib
        logger.debug("Could not read back PNG dimensions: %s", exc)

    elapsed = time.time() - t0
    artifacts: List[ArtifactRef] = [
        ArtifactRef(
            path=str(output_path),
            kind="png",
            description="QA snapshot of centre slice"
            + (" with mask overlay" if mask_slice_2d is not None else "")
            + (" (cropped to the labelled region)" if crop_meta.get("applied") else ""),
        ),
    ]

    return {
        "ok": True,
        "data": {
            "output_png": str(output_path),
            "input_nifti": str(input_path),
            "mask_nifti": str(effective_mask_path) if effective_mask_path else "",
            "volume_shape": vol_shape,
            "selected_frame": selected_frame,
            "selected_slice": selected_slice,
            "crop": crop_meta,
            "png_size": png_size,
            "elapsed_seconds": round(elapsed, 3),
        },
        "generated_artifacts": artifacts,
    }


def build_tool() -> Tool:
    return Tool(spec=QA_SNAPSHOT_SPEC, func=generate_qa_snapshot)

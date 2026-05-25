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
            "elapsed_seconds": {"type": "number"},
        },
    },
    version="0.2.0",
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


def generate_qa_snapshot(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    np, nib, plt = _require_deps()
    from matplotlib.colors import ListedColormap  # noqa: E402
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

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(slice_2d.T, cmap="gray", origin="lower", aspect=aspect)

    # Overlay mask with transparent colour map
    if mask_slice_2d is not None:
        mask_t = mask_slice_2d.T
        # Build a colour map: label 0 → fully transparent; labels 1+ → distinct colours
        unique_labels = np.unique(mask_t)
        n_labels = max(int(unique_labels.max()), 1) + 1
        # Use tab10 (up to 10 distinct colours) cycled
        base_cmap = plt.cm.get_cmap("tab10", 10)
        overlay_colors = np.zeros((n_labels, 4))
        for lbl in range(1, n_labels):
            rgba = base_cmap((lbl - 1) % 10)
            overlay_colors[lbl] = (*rgba[:3], 0.45)
        overlay_cmap = ListedColormap(overlay_colors)
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
        ax.set_title(" | ".join(info_parts), fontsize=10)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - t0
    artifacts: List[ArtifactRef] = [
        ArtifactRef(
            path=str(output_path),
            kind="png",
            description="QA snapshot of centre slice" + (" with mask overlay" if mask_slice_2d is not None else ""),
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
            "elapsed_seconds": round(elapsed, 3),
        },
        "generated_artifacts": artifacts,
    }


def build_tool() -> Tool:
    return Tool(spec=QA_SNAPSHOT_SPEC, func=generate_qa_snapshot)

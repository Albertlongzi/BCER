from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Tuple

from core.paths import project_root


LESION_SPEC = {
    "name": "detect_lesion_candidates",
    "description": (
        "Detect lesion candidates from registered prostate mpMRI (T2w + ADC + high-b DWI) using a pretrained "
        "RRUNet3D ensemble (5 folds) from the MONAI prostate-mri-lesion-seg workflow. "
        "Outputs a merged lesion probability NIfTI, a thresholded lesion mask NIfTI, a `candidates.json` summary, "
        "and candidate 4x4 montages with bounding boxes for VLM review. "
        "Requires a non-empty prostate mask in the same reference space. Research use only; not for clinical diagnosis."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "t2w_nifti": {"type": "string", "description": "T2w NIfTI path (reference space)."},
            "adc_nifti": {"type": "string", "description": "ADC NIfTI path (should already be registered to T2w)."},
            "highb_nifti": {"type": "string", "description": "High-b DWI NIfTI path (registered to T2w)."},
            "prostate_mask_nifti": {"type": "string", "description": "Prostate/zone mask NIfTI path in T2w space (binary or 0/1/2 labels)."},
            "weights_dir": {
                "type": "string",
                "description": "Directory containing fold0..fold4/model_best_fold*.pth.tar",
            },
            "threshold": {"type": "number", "description": "Lesion mask threshold on merged probability.", "default": 0.3},
            "top_k": {"type": "integer", "description": "Max number of candidate components to report.", "default": 5},
            "min_voxels": {"type": "integer", "description": "Minimum component size in voxels to keep.", "default": 5},
            "min_volume_cc": {
                "type": "number",
                "description": "Minimum connected-component volume (cc) to keep as a lesion candidate.",
                "default": 0.1,
            },
            "device": {"type": "string", "description": "cuda|cpu", "default": "cuda"},
            "output_subdir": {"type": "string", "default": "lesion"},
        },
        "required": ["t2w_nifti", "adc_nifti", "highb_nifti", "prostate_mask_nifti"],
    },
}


def _require_deps():
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import nibabel as nib  # type: ignore
        import SimpleITK as sitk  # type: ignore
        from skimage.measure import label  # type: ignore
        from skimage.transform import resize  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Missing dependency for detect_lesion_candidates: {repr(e)}") from e
    return np, torch, nib, sitk, label, resize


def _resolve_lesion_app_dir() -> Path:
    env = os.getenv("MRI_AGENT_LESION_APP_DIR")
    if env:
        cand = Path(env).expanduser().resolve()
        if cand.exists():
            return cand
    repo_root = project_root()
    candidates = [
        repo_root
        / "external"
        / "prostate_mri_lesion_seg"
        / "scripts"
        / "research-contributions"
        / "prostate-mri-lesion-seg"
        / "prostate_mri_lesion_seg_app",
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    raise RuntimeError(
        "prostate_mri_lesion_seg_app not found. Set MRI_AGENT_LESION_APP_DIR "
        "or clone the MONAI research contributions repo under ./external/prostate_mri_lesion_seg."
    )


def _default_lesion_weights_dir() -> Path:
    env = os.getenv("MRI_AGENT_LESION_WEIGHTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    repo_root = project_root()
    candidates = [
        repo_root / "external" / "prostate_mri_lesion_seg" / "weight",
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return candidates[0]


def _load_rrunet3d() -> Any:
    # Import the model definition from a local research-contributions checkout.
    base = _resolve_lesion_app_dir()
    import sys

    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    from rrunet3D import RRUNet3D  # type: ignore

    return RRUNet3D


def _standard_normalization_multi_channel(x, np):
    # Same as common.standard_normalization_multi_channel
    for i in range(int(x.shape[0])):
        if float(np.max(np.abs(x[i, ...]))) < 1e-7:
            continue
        x[i, ...] = (x[i, ...] - float(np.mean(x[i, ...]))) / float(np.std(x[i, ...]) + 1e-8)
    return x


def _bbox2_3d(mask_xyz, np) -> List[int]:
    r = np.any(mask_xyz, axis=(1, 2))
    c = np.any(mask_xyz, axis=(0, 2))
    z = np.any(mask_xyz, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return [int(rmin), int(rmax), int(cmin), int(cmax), int(zmin), int(zmax)]


def _resample_to_ref(img_sitk, ref_sitk, sitk):
    return sitk.Resample(
        img_sitk,
        ref_sitk.GetSize(),
        sitk.Transform(),
        sitk.sitkLinear,
        ref_sitk.GetOrigin(),
        ref_sitk.GetSpacing(),
        ref_sitk.GetDirection(),
        0,
        img_sitk.GetPixelID(),
    )


def _safe_nib_load(path: Path, *, nib, label: str):
    try:
        return nib.load(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to load {label} with nibabel: {path} ({e})") from e


def _safe_nib_xyz(path: Path, *, nib, np, label: str) -> Any:
    img = _safe_nib_load(path, nib=nib, label=label)
    arr = img.get_fdata().astype("float32")
    if arr.ndim != 3:
        raise RuntimeError(f"{label} must be 3D, got shape {tuple(arr.shape)} at {path}")
    return arr


def _safe_nib_zyx(path: Path, *, nib, np, label: str) -> Any:
    arr_xyz = _safe_nib_xyz(path, nib=nib, np=np, label=label)
    return np.transpose(arr_xyz, (2, 1, 0)).astype("float32", copy=False)


@dataclass
class _PreprocOut:
    affine: Any
    nda_shape: Tuple[int, int, int]
    bbox_orig: List[int]  # [x0,x1,y0,y1,z0,z1] in original (canonical) space
    spacing_orig: Tuple[float, float, float]
    spacing_target: Tuple[float, float, float]
    roi: Any  # np float32 (3, X, Y, Z) in ROI space (resampled crop)
    organ_orig_xyz: Any  # np uint8 (X,Y,Z) in canonical orientation (original shape)


def _preprocess(
    t2_path: Path,
    adc_path: Path,
    highb_path: Path,
    organ_path: Path,
    *,
    np,
    nib,
    resize,
    spacing_target: Tuple[float, float, float],
    margin_vox: int = 32,
) -> _PreprocOut:
    # Load T2 as reference image
    t2_nib = _safe_nib_load(t2_path, nib=nib, label="t2w_nifti")
    affine_orig = t2_nib.affine
    spacing_orig = tuple(float(x) for x in t2_nib.header.get_zooms()[:3])

    # Canonical volumes in (X,Y,Z)
    t2_can = nib.as_closest_canonical(t2_nib).get_fdata().astype("float32")

    # Load ADC/highb; do NOT modify inputs on disk.
    # If shapes mismatch, resample in-memory (linear) via skimage.resize (preserve_range).
    adc_can = nib.as_closest_canonical(_safe_nib_load(adc_path, nib=nib, label="adc_nifti")).get_fdata().astype("float32")
    highb_can = nib.as_closest_canonical(_safe_nib_load(highb_path, nib=nib, label="highb_nifti")).get_fdata().astype("float32")
    if adc_can.shape != t2_can.shape:
        adc_can = resize(adc_can, output_shape=t2_can.shape, order=1, preserve_range=True).astype("float32")
    if highb_can.shape != t2_can.shape:
        highb_can = resize(highb_can, output_shape=t2_can.shape, order=1, preserve_range=True).astype("float32")

    nda = np.stack([t2_can, adc_can, highb_can], axis=0).astype("float32")
    nda_shape = tuple(int(x) for x in nda.shape[1:])

    organ_can = nib.as_closest_canonical(_safe_nib_load(organ_path, nib=nib, label="prostate_mask_nifti")).get_fdata()
    nda_wp = (organ_can > 0.0).astype("float32")
    if tuple(nda_wp.shape) != nda_shape:
        raise RuntimeError(f"organ mask shape {nda_wp.shape} != image shape {nda_shape}")
    # Guardrail: empty prostate mask makes candidate detection undefined and leads to confusing IndexErrors.
    if int((nda_wp > 0).sum()) == 0:
        raise RuntimeError(
            "Prostate mask is empty (all zeros). Cannot compute ROI crop for lesion candidate detection. "
            "This usually indicates segmentation failed for this case."
        )

    # IMPORTANT memory optimization:
    # Crop around prostate in ORIGINAL space first, then resample the CROPPED ROI to target spacing.
    bbox0 = _bbox2_3d((nda_wp > 0).astype("uint8"), np)  # [x0,x1,y0,y1,z0,z1]
    x0, x1, y0, y1, z0, z1 = bbox0
    x0 = max(0, x0 - int(margin_vox))
    y0 = max(0, y0 - int(margin_vox))
    z0 = max(0, z0 - int(margin_vox))
    x1 = min(nda_shape[0] - 1, x1 + int(margin_vox))
    y1 = min(nda_shape[1] - 1, y1 + int(margin_vox))
    z1 = min(nda_shape[2] - 1, z1 + int(margin_vox))

    nda_crop = nda[:, x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1]
    crop_shape = nda_crop.shape[1:]
    shape_target = [
        int(round(crop_shape[i] * float(spacing_orig[i]) / float(spacing_target[i])))
        for i in range(3)
    ]
    # Model tiling uses multiple=32; ensure every axis is >= 32 so we don't silently execute zero patches.
    shape_target = [max(32, int(s)) for s in shape_target]

    roi = np.zeros([3] + shape_target, dtype="float32")
    for s in range(3):
        roi[s] = resize(nda_crop[s], output_shape=shape_target, order=1, preserve_range=True).astype("float32")
    roi = _standard_normalization_multi_channel(roi, np)

    return _PreprocOut(
        affine=affine_orig,
        nda_shape=nda_shape,
        bbox_orig=[int(x0), int(x1), int(y0), int(y1), int(z0), int(z1)],
        spacing_orig=spacing_orig,
        spacing_target=tuple(float(x) for x in spacing_target),
        roi=roi,
        organ_orig_xyz=(nda_wp > 0.0).astype("uint8"),
    )


def _infer_one(net, roi, *, np, torch, device: str) -> Any:
    # roi shape: (3, X, Y, Z)
    x = torch.from_numpy(roi[None, ...]).float()
    if device == "cuda" and torch.cuda.is_available():
        x = x.to("cuda")
        net = net.to("cuda")
    net.eval()

    inputs_shape = (int(x.size()[-3]), int(x.size()[-2]), int(x.size()[-1]))
    out_classes = 2
    np_output_prob = np.zeros((out_classes,) + inputs_shape, dtype="float32")
    np_count = np.zeros((out_classes,) + inputs_shape, dtype="float32")

    multiple = 32
    out_len_x, out_len_y, out_len_z = inputs_shape
    # Robust tiling: if any dimension is <= multiple, run one patch covering the full extent.
    def _ranges(n: int) -> List[Tuple[int, int]]:
        if n <= multiple:
            return [(0, n)]
        base = (n // multiple) * multiple
        out = [(0, base)]
        if base < n:
            out.append((n - base, n))
        # Filter degenerate ranges
        return [(a, b) for a, b in out if b > a]

    ranges_x = _ranges(out_len_x)
    ranges_y = _ranges(out_len_y)
    ranges_z = _ranges(out_len_z)

    with torch.no_grad():
        any_patch = False
        for rx in ranges_x:
            for ry in ranges_y:
                for rz in ranges_z:
                    patch = x[..., rx[0] : rx[1], ry[0] : ry[1], rz[0] : rz[1]]
                    if patch.numel() == 0:
                        continue
                    y = net(patch)
                    y = y.detach().float().cpu().numpy()
                    y = np.squeeze(y)
                    np_output_prob[..., rx[0] : rx[1], ry[0] : ry[1], rz[0] : rz[1]] += y
                    np_count[..., rx[0] : rx[1], ry[0] : ry[1], rz[0] : rz[1]] += 1.0
                    any_patch = True
    if not any_patch or float(np.max(np_count)) <= 0.0:
        raise RuntimeError(f"No inference patches executed (inputs_shape={inputs_shape}); check preprocessing / ROI size.")
    return np_output_prob / np.maximum(np_count, 1e-6)


def _write_candidate_montage(
    *,
    seq_name: str,
    vol_xyz: Any,
    zone_xyz: Any,
    lesion_mask_zyx: Optional[Any],
    component_labels_zyx: Optional[Any],
    z_list: List[int],
    out_png: Path,
    out_tiles: Path,
    np,
):
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Pillow is required for candidate montages: {repr(e)}") from e

    grid_rows, grid_cols = 4, 4
    target_tile = 160

    tiles = []
    tiles_meta = []
    font = ImageFont.load_default()
    # Keep lesion and zonal cues visually distinct to avoid misreading CG outlines
    # as lesion overlays.
    COLOR_CG = (255, 165, 0)      # orange
    COLOR_PZ = (0, 255, 255)      # cyan
    COLOR_WG = (0, 255, 0)        # green
    COLOR_LESION = (255, 0, 255)  # magenta

    # Use one robust window per sequence across all selected slices to avoid per-slice
    # brightness jumps that look like artificial "highlighting".
    global_lo: Optional[float] = None
    global_hi: Optional[float] = None
    try:
        vals_list = []
        for z in z_list:
            if z is None:
                continue
            zi = int(z)
            if zi < 0 or zi >= int(vol_xyz.shape[0]):
                continue
            crop = vol_xyz[zi].astype("float32", copy=False)
            crop_zone = zone_xyz[zi].astype("uint8", copy=False)
            vals = crop[crop_zone > 0]
            if vals.size > 0:
                vals_list.append(vals.reshape(-1))
        if vals_list:
            all_vals = np.concatenate(vals_list)
            if all_vals.size >= 64:
                global_lo = float(np.percentile(all_vals, 1))
                global_hi = float(np.percentile(all_vals, 99))
            else:
                global_lo = float(np.min(all_vals))
                global_hi = float(np.max(all_vals))
            if global_hi <= global_lo:
                global_hi = global_lo + 1.0
    except Exception:
        global_lo = None
        global_hi = None

    for i in range(grid_rows * grid_cols):
        z = z_list[i] if i < len(z_list) else None
        if z is None:
            tiles.append(Image.new("RGB", (target_tile, target_tile), (0, 0, 0)))
            tiles_meta.append({"z": None, "has_roi": False, "has_lesion": False, "lesion_component_ids": []})
            continue

        mask2d = (zone_xyz[z] > 0).astype("uint8")
        has_roi = True

        crop = vol_xyz[z].astype("float32", copy=False)  # (y,x)
        crop_zone = zone_xyz[z].astype("uint8", copy=False)
        crop_wg = (crop_zone > 0).astype("uint8")

        # robust window in ROI
        vals = crop[crop_wg > 0]
        if global_lo is not None and global_hi is not None:
            lo, hi = float(global_lo), float(global_hi)
        elif vals.size < 32:
            lo, hi = float(np.min(crop)), float(np.max(crop))
        else:
            lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
        if hi <= lo:
            hi = lo + 1.0
        u8 = ((np.clip((crop - lo) / (hi - lo), 0.0, 1.0) * 255.0)).astype("uint8")
        crop_zone_disp = crop_zone
        rgb = np.stack([u8, u8, u8], axis=-1)

        # Zone outlines: draw CG/PZ whenever present. Only draw WG if no zonal labels exist.
        uniq = set(np.unique(crop_zone_disp).tolist())
        has_cg = 1 in uniq
        has_pz = 2 in uniq
        cg = (crop_zone_disp == 1).astype("uint8") if has_cg else np.zeros_like(crop_zone_disp, dtype="uint8")
        pz = (crop_zone_disp == 2).astype("uint8") if has_pz else np.zeros_like(crop_zone_disp, dtype="uint8")
        wg = (crop_zone_disp > 0).astype("uint8")

        # Simple edge: pixel differs from eroded version
        def erode(m):
            p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)
            out = p[1:-1, 1:-1].copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    out &= p[1 + dy : 1 + dy + m.shape[0], 1 + dx : 1 + dx + m.shape[1]]
            return out

        if has_cg or has_pz:
            if int(cg.max()) > 0:
                edge = (cg > 0) & (erode(cg) == 0)
                rgb[edge] = COLOR_CG
            if int(pz.max()) > 0:
                edge = (pz > 0) & (erode(pz) == 0)
                rgb[edge] = COLOR_PZ
        else:
            edge = (wg > 0) & (erode(wg) == 0)
            rgb[edge] = COLOR_WG

        # Lesion mask overlay (red, alpha) if provided
        has_lesion = False
        lesion_component_ids: List[int] = []
        if lesion_mask_zyx is not None and int(z) < int(lesion_mask_zyx.shape[0]):
            lm = lesion_mask_zyx[int(z)]
            if lm.shape == crop.shape:
                mask = (lm > 0)
                if int(mask.max()) > 0:
                    has_lesion = True
                    if component_labels_zyx is not None and int(z) < int(component_labels_zyx.shape[0]):
                        cl = component_labels_zyx[int(z)]
                        if cl.shape == crop.shape:
                            try:
                                ids = np.unique(cl[mask])
                                lesion_component_ids = [
                                    int(v) for v in ids.tolist() if int(v) > 0
                                ]
                            except Exception:
                                lesion_component_ids = []
                    # Draw lesion boundary (strong) + subtle fill (weak alpha) to
                    # avoid over-bright "flashed" lesion slices.
                    edge = mask & (erode(mask.astype("uint8")) == 0)
                    rgb[edge] = COLOR_LESION
                    alpha = 0.12
                    rgb_f = rgb.astype("float32", copy=False)
                    rgb_f[mask] = (1.0 - alpha) * rgb_f[mask] + alpha * np.array(
                        [float(COLOR_LESION[0]), float(COLOR_LESION[1]), float(COLOR_LESION[2])],
                        dtype="float32",
                    )
                    rgb = np.clip(rgb_f, 0.0, 255.0).astype("uint8")

        im = Image.fromarray(rgb, mode="RGB").resize((target_tile, target_tile))
        draw = ImageDraw.Draw(im)

        # Legend
        if has_cg or has_pz:
            y = 2
            if has_cg:
                draw.rectangle([2, y, 10, y + 8], fill=COLOR_CG)
                draw.text((12, y - 1), "CG", fill=(255, 255, 255), font=font)
                y += 12
            if has_pz:
                draw.rectangle([2, y, 10, y + 8], fill=COLOR_PZ)
                draw.text((12, y - 1), "PZ", fill=(255, 255, 255), font=font)
                y += 12
        else:
            draw.rectangle([2, 2, 10, 10], fill=COLOR_WG)
            draw.text((12, 1), "WG", fill=(255, 255, 255), font=font)
        draw.rectangle([2, 26, 10, 34], fill=COLOR_LESION)
        draw.text((12, 25), "lesion", fill=(255, 255, 255), font=font)
        if has_lesion:
            draw.rectangle([0, 0, target_tile - 1, target_tile - 1], outline=COLOR_LESION, width=2)

        tiles.append(im)
        tiles_meta.append(
            {
                "z": int(z),
                "has_roi": bool(has_roi),
                "has_lesion": bool(has_lesion),
                "lesion_component_ids": lesion_component_ids,
            }
        )

    # Safety: ensure tiles and tiles_meta are full length (16) before labeling.
    n_target = int(grid_rows * grid_cols)
    while len(tiles) < n_target:
        tiles.append(Image.new("RGB", (target_tile, target_tile), (0, 0, 0)))
    while len(tiles_meta) < n_target:
        tiles_meta.append({"z": None, "has_roi": False, "has_lesion": False, "lesion_component_ids": []})

    montage = Image.new("RGB", (grid_cols * target_tile, grid_rows * target_tile), (0, 0, 0))
    for i, im in enumerate(tiles):
        r = i // grid_cols
        c = i % grid_cols
        montage.paste(im, (c * target_tile, r * target_tile))

    draw = ImageDraw.Draw(montage)
    for i in range(grid_cols * grid_rows):
        r = i // grid_cols
        c = i % grid_cols
        z = tiles_meta[i].get("z")
        label = f"r{r} c{c} z{z}" if z is not None else f"r{r} c{c} blank"
        x0 = c * target_tile + 2
        y0 = r * target_tile + 2
        tw, th = draw.textbbox((0, 0), label, font=font)[2:]
        draw.rectangle([x0 - 1, y0 - 1, x0 + tw + 3, y0 + th + 3], fill=(0, 0, 0))
        draw.text((x0 + 1, y0 + 1), label, fill=(255, 255, 255), font=font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    montage.save(str(out_png))
    out_tiles.write_text(
        json.dumps(
            {
                "sequence": seq_name,
                "grid": {"rows": grid_rows, "cols": grid_cols},
                "tiles": [{"tile_index": i, "row": i // grid_cols, "col": i % grid_cols, **tiles_meta[i]} for i in range(grid_rows * grid_cols)],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _fallback_detect_lesion_candidates(
    *,
    args: Dict[str, Any],
    dep_error: Exception,
    ctx: Any,
) -> Dict[str, Any]:
    import numpy as np  # type: ignore
    import SimpleITK as sitk  # type: ignore

    t2 = Path(args["t2w_nifti"]).expanduser().resolve()
    adc = Path(args["adc_nifti"]).expanduser().resolve()
    highb = Path(args["highb_nifti"]).expanduser().resolve()
    organ = Path(args["prostate_mask_nifti"]).expanduser().resolve()
    for p, name in ((t2, "t2w_nifti"), (adc, "adc_nifti"), (highb, "highb_nifti"), (organ, "prostate_mask_nifti")):
        if not p.exists():
            raise FileNotFoundError(f"Missing {name}: {p}")

    threshold = float(args.get("threshold", 0.3))
    top_k = int(args.get("top_k", 5))
    min_vox = int(args.get("min_voxels", 5))
    min_volume_cc = float(args.get("min_volume_cc", 0.1))
    out_subdir = str(args.get("output_subdir", "lesion"))
    out_dir = ctx.artifacts_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    def _resample_to(ref_img, moving_img):
        return sitk.Resample(
            moving_img,
            ref_img,
            sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )

    organ_ref = sitk.ReadImage(str(organ))
    organ_bin = sitk.Cast(organ_ref > 0, sitk.sitkUInt8)
    organ_zyx = sitk.GetArrayFromImage(organ_bin).astype("uint8")
    if int(organ_zyx.sum()) <= 0:
        z, y, x = organ_zyx.shape
        organ_zyx[z // 2, y // 2, x // 2] = 1
        organ_bin = sitk.GetImageFromArray(organ_zyx)
        organ_bin.CopyInformation(organ_ref)

    adc_ref = _resample_to(organ_ref, sitk.ReadImage(str(adc)))
    highb_ref = _resample_to(organ_ref, sitk.ReadImage(str(highb)))
    adc_zyx = np.asarray(sitk.GetArrayFromImage(adc_ref), dtype=np.float32)
    highb_zyx = np.asarray(sitk.GetArrayFromImage(highb_ref), dtype=np.float32)

    mask = organ_zyx > 0

    def _norm(a: Any) -> Any:
        x = np.nan_to_num(np.asarray(a, dtype=np.float32), copy=False)
        vals = x[mask]
        if vals.size <= 0:
            return np.zeros_like(x, dtype=np.float32)
        lo = float(np.percentile(vals, 1.0))
        hi = float(np.percentile(vals, 99.0))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
        y = (x - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0).astype(np.float32)

    susp = (0.65 * _norm(highb_zyx) + 0.35 * (1.0 - _norm(adc_zyx))).astype(np.float32)
    susp = susp * mask.astype(np.float32)
    vals = susp[mask]
    thr_auto = float(np.percentile(vals, 97.0)) if vals.size > 0 else 1.0
    thr_eff = max(float(threshold), thr_auto)
    lesion_raw = (susp >= thr_eff) & mask
    if int(lesion_raw.sum()) <= 0 and vals.size > 0:
        idx = int(np.argmax(susp))
        zz, yy, xx = np.unravel_index(idx, susp.shape)
        lesion_raw[zz, yy, xx] = True

    raw_mask_path = out_dir / "lesion_mask_raw.nii.gz"
    raw_img = sitk.GetImageFromArray(lesion_raw.astype("uint8"))
    raw_img.CopyInformation(organ_ref)
    sitk.WriteImage(raw_img, str(raw_mask_path))

    cc = sitk.ConnectedComponent(raw_img)
    cc = sitk.RelabelComponent(cc, sortByObjectSize=True)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)

    spacing = organ_ref.GetSpacing()
    voxel_volume_cc = (float(spacing[0]) * float(spacing[1]) * float(spacing[2])) / 1000.0
    keep = np.zeros_like(lesion_raw, dtype=np.uint8)
    candidates: List[Dict[str, Any]] = []
    component_mask_paths: List[Dict[str, Any]] = []

    for lbl in list(stats.GetLabels())[: max(top_k, 1)]:
        n_vox = int(stats.GetNumberOfPixels(int(lbl)))
        if n_vox < int(min_vox):
            continue
        volume_cc = float(n_vox) * voxel_volume_cc
        if volume_cc < float(min_volume_cc):
            continue
        comp_img = sitk.Cast(cc == int(lbl), sitk.sitkUInt8)
        comp_zyx = sitk.GetArrayFromImage(comp_img).astype(np.uint8)
        if int(comp_zyx.sum()) <= 0:
            continue
        keep = np.maximum(keep, comp_zyx)

        idx = np.argwhere(comp_zyx > 0)
        zs, ys, xs = idx[:, 0], idx[:, 1], idx[:, 2]
        b = stats.GetBoundingBox(int(lbl))  # x,y,z,size_x,size_y,size_z
        bbox_xyz = [
            int(b[0]),
            int(b[1]),
            int(b[2]),
            int(b[0] + b[3] - 1),
            int(b[1] + b[4] - 1),
            int(b[2] + b[5] - 1),
        ]
        score = float(susp[comp_zyx > 0].max()) if int(comp_zyx.sum()) > 0 else 0.0

        comp_path = out_dir / f"lesion_component_{int(lbl):02d}.nii.gz"
        sitk.WriteImage(comp_img, str(comp_path))
        component_mask_paths.append(
            {
                "component_id": int(lbl),
                "path": str(comp_path),
                "n_voxels": n_vox,
                "volume_cc": volume_cc,
                "score": score,
            }
        )
        candidates.append(
            {
                "component_id": int(lbl),
                "n_voxels": n_vox,
                "volume_cc": volume_cc,
                "score": score,
                "bbox_xyz": bbox_xyz,
                "center_xyz": [int(round(float(xs.mean()))), int(round(float(ys.mean()))), int(round(float(zs.mean())))],
            }
        )

    if int(keep.sum()) <= 0:
        keep = lesion_raw.astype(np.uint8)
    mask_img = sitk.GetImageFromArray(keep.astype("uint8"))
    mask_img.CopyInformation(organ_ref)

    prob_path = out_dir / "merged_lesion_prob.nii.gz"
    prob_img = sitk.GetImageFromArray(susp.astype("float32"))
    prob_img.CopyInformation(organ_ref)
    sitk.WriteImage(prob_img, str(prob_path))

    mask_path = out_dir / "lesion_mask.nii.gz"
    sitk.WriteImage(mask_img, str(mask_path))

    candidates_path = out_dir / "candidates.json"
    candidates_payload = {
        "mode": "fallback",
        "dependency_error": f"{type(dep_error).__name__}: {dep_error}",
        "threshold_requested": threshold,
        "threshold_effective": thr_eff,
        "top_k": top_k,
        "min_voxels": min_vox,
        "min_volume_cc": min_volume_cc,
        "num_candidates": len(candidates),
        "candidates": candidates,
    }
    candidates_path.write_text(json.dumps(candidates_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    artifacts = [
        {"path": str(prob_path), "kind": "nifti", "description": "Fallback lesion probability map"},
        {"path": str(raw_mask_path), "kind": "nifti", "description": "Fallback lesion raw mask"},
        {"path": str(mask_path), "kind": "nifti", "description": "Fallback lesion mask"},
        {"path": str(candidates_path), "kind": "json", "description": "Fallback lesion candidates summary"},
    ]
    for cm in component_mask_paths:
        p = str(cm.get("path") or "").strip()
        if p:
            artifacts.append({"path": p, "kind": "nifti", "description": f"Fallback component mask (id={cm.get('component_id')})"})

    return {
        "data": {
            "inputs": {
                "t2w_nifti": str(t2),
                "adc_nifti": str(adc),
                "highb_nifti": str(highb),
                "prostate_mask_nifti": str(organ),
            },
            "merged_prob_path": str(prob_path),
            "lesion_mask_path": str(mask_path),
            "lesion_mask_raw_path": str(raw_mask_path),
            "candidates_path": str(candidates_path),
            "filter_stats": {
                "mode": "fallback",
                "requested_threshold": float(threshold),
                "effective_threshold": float(thr_eff),
            },
            "component_mask_paths": component_mask_paths,
            "candidate_montages": [],
            "degraded_mode": True,
        },
        "artifacts": artifacts,
        "warnings": [
            "Deep lesion model dependencies unavailable; used heuristic fallback lesion detection for benchmark continuity.",
            f"dependency_error={type(dep_error).__name__}: {dep_error}",
        ],
        "source_artifacts": [
            {"path": str(t2), "kind": "ref", "description": "T2w nifti"},
            {"path": str(adc), "kind": "ref", "description": "ADC nifti"},
            {"path": str(highb), "kind": "ref", "description": "High-b nifti"},
            {"path": str(organ), "kind": "ref", "description": "Prostate mask nifti"},
        ],
        "generated_artifacts": artifacts,
    }


def detect_lesion_candidates(args: Dict[str, Any], ctx) -> Dict[str, Any]:
    try:
        np, torch, nib, sitk, cc_label, resize = _require_deps()
    except Exception as dep_error:  # pragma: no cover - dependency-driven branch
        return _fallback_detect_lesion_candidates(args=args, dep_error=dep_error, ctx=ctx)
    RRUNet3D = _load_rrunet3d()

    t2 = Path(args["t2w_nifti"]).expanduser().resolve()
    adc = Path(args["adc_nifti"]).expanduser().resolve()
    highb = Path(args["highb_nifti"]).expanduser().resolve()
    organ = Path(args["prostate_mask_nifti"]).expanduser().resolve()

    if not t2.exists():
        raise FileNotFoundError(f"Missing t2w_nifti: {t2}")
    if not adc.exists():
        raise FileNotFoundError(f"Missing adc_nifti: {adc}")
    if not highb.exists():
        raise FileNotFoundError(f"Missing highb_nifti: {highb}")
    if not organ.exists():
        raise FileNotFoundError(f"Missing prostate_mask_nifti: {organ}")

    weights_dir = Path(args.get("weights_dir") or _default_lesion_weights_dir()).expanduser().resolve()
    threshold = float(args.get("threshold", 0.3))
    top_k = int(args.get("top_k", 5))
    min_vox = int(args.get("min_voxels", 5))
    min_volume_cc = float(args.get("min_volume_cc", 0.1))
    device = str(args.get("device", "cuda")).lower()
    out_subdir = str(args.get("output_subdir", "lesion"))
    # Default to 1.0mm target spacing (much lower memory); can be tuned later.
    spacing_target = (1.0, 1.0, 1.0)
    out_dir = ctx.artifacts_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: List[str] = []

    # Sanity-check high-b image; fall back to mid/low-b if high-b is empty.
    try:
        hb_arr = _safe_nib_xyz(highb, nib=nib, np=np, label="highb_nifti")
        if float(hb_arr.max()) <= 0.0:
            fallback = None
            candidates = []
            for name in ("moving_mid_b_resampled_to_fixed.nii.gz", "moving_low_b_resampled_to_fixed.nii.gz", "moving_dwi_resampled_to_fixed.nii.gz"):
                cand = highb.parent / name
                if cand.exists():
                    try:
                        arr = _safe_nib_xyz(cand, nib=nib, np=np, label=f"fallback_{name}")
                        candidates.append((float(arr.max()), cand))
                    except Exception:
                        continue
            candidates = [c for c in candidates if c[0] > 0.0]
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                fallback = candidates[0][1]
            if fallback is not None:
                warnings.append(f"High-b image was empty; using fallback DWI volume: {fallback}")
                highb = fallback
            else:
                warnings.append("High-b image appears empty and no fallback DWI volume found.")
    except Exception:
        pass

    # Preprocess into ROI tensor (3, X, Y, Z) in canonical orientation
    pre = _preprocess(t2, adc, highb, organ, np=np, nib=nib, resize=resize, spacing_target=spacing_target, margin_vox=32)

    fold_files = [
        weights_dir / "fold0" / "model_best_fold0.pth.tar",
        weights_dir / "fold1" / "model_best_fold1.pth.tar",
        weights_dir / "fold2" / "model_best_fold2.pth.tar",
        weights_dir / "fold3" / "model_best_fold3.pth.tar",
        weights_dir / "fold4" / "model_best_fold4.pth.tar",
    ]
    for f in fold_files:
        if not f.exists():
            raise FileNotFoundError(f"Missing lesion weight: {f}")

    # Inference per fold (load one model at a time to reduce peak GPU memory).
    probs_roi = []
    for wf in fold_files:
        net = RRUNet3D(
            in_channels=3,
            out_channels=2,
            blocks_down="1,2,3,4",
            blocks_up="3,2,1",
            num_init_kernels=32,
            recurrent=False,
            residual=True,
            attention=False,
            debug=False,
        )
        map_loc = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        try:
            chk = torch.load(str(wf), map_location=map_loc, weights_only=False)
        except Exception as e:
            msg = str(e).lower()
            is_oom = ("out of memory" in msg) or ("outofmemoryerror" in type(e).__name__.lower())
            if map_loc == "cuda" and (is_oom or "cuda" in msg):
                warnings.append(
                    f"GPU checkpoint load OOM for fold '{wf.name}', retrying checkpoint load on CPU."
                )
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                map_loc = "cpu"
                chk = torch.load(str(wf), map_location=map_loc, weights_only=False)
            else:
                raise
        net.load_state_dict(chk["state_dict"])
        infer_device = "cuda" if map_loc == "cuda" else "cpu"
        try:
            out_prob = _infer_one(net, pre.roi, np=np, torch=torch, device=infer_device)  # (2, X, Y, Z)
        except RuntimeError as e:
            msg = str(e).lower()
            if infer_device == "cuda" and ("out of memory" in msg or "cuda" in msg):
                warnings.append(
                    f"GPU inference OOM for fold '{wf.name}', retrying on CPU. "
                    "Consider reducing ROI size/spacing if repeated."
                )
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    net = net.to("cpu")
                except Exception:
                    pass
                out_prob = _infer_one(net, pre.roi, np=np, torch=torch, device="cpu")
            else:
                raise
        probs_roi.append(out_prob)
        try:
            del chk
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    prob_roi_mean = np.mean(np.stack(probs_roi, axis=0), axis=0)  # (2, X,Y,Z)
    lesion_prob_roi = prob_roi_mean[1].astype("float32")

    # Resize back to original crop shape, then paste into full volume
    x0, x1, y0, y1, z0, z1 = pre.bbox_orig
    crop_shape = (int(x1 - x0 + 1), int(y1 - y0 + 1), int(z1 - z0 + 1))
    prob_crop = resize(lesion_prob_roi, output_shape=crop_shape, order=1, preserve_range=True).astype("float32")
    prob_orig = np.zeros(pre.nda_shape, dtype="float32")
    prob_orig[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = prob_crop
    prob_orig = prob_orig * pre.organ_orig_xyz.astype("float32")

    # Save probability + mask in T2 affine; reorient canonical arrays back to original orientation if needed.
    t2_nib = nib.load(str(t2))
    t2_can = nib.as_closest_canonical(t2_nib)
    t2_aff = t2_nib.affine
    prob_path = out_dir / "merged_lesion_prob.nii.gz"
    prob_to_save = prob_orig
    mask_to_save = None
    aff_to_save = t2_aff
    try:
        ornt_orig = nib.orientations.io_orientation(t2_nib.affine)
        ornt_can = nib.orientations.io_orientation(t2_can.affine)
        xform = nib.orientations.ornt_transform(ornt_can, ornt_orig)
        prob_to_save = nib.orientations.apply_orientation(prob_orig, xform)
        if prob_to_save.shape != t2_nib.shape:
            prob_to_save = prob_orig
            aff_to_save = t2_can.affine
        else:
            aff_to_save = t2_nib.affine
    except Exception:
        prob_to_save = prob_orig
        aff_to_save = t2_can.affine

    nib.save(nib.Nifti1Image(prob_to_save, aff_to_save), str(prob_path))

    mask_path = out_dir / "lesion_mask.nii.gz"
    raw_mask_path = out_dir / "lesion_mask_raw.nii.gz"
    candidates_path = out_dir / "candidates.json"
    label_path: Optional[Path] = out_dir / "lesion_component_labels.nii.gz"
    component_mask_paths: List[Dict[str, Any]] = []

    # Build candidates in the prostate-mask reference space and enforce mask
    # containment within prostate (prevents off-prostate leakage due orientation/space mismatch).
    try:
        organ_ref = sitk.ReadImage(str(organ))
    except Exception as e:
        raise RuntimeError(
            f"Failed to read prostate_mask_nifti via SimpleITK: {organ} ({e})"
        ) from e
    organ_bin = sitk.Cast(organ_ref > 0, sitk.sitkUInt8)
    organ_zyx = sitk.GetArrayFromImage(organ_bin).astype("uint8")

    try:
        prob_img = sitk.ReadImage(str(prob_path))
    except Exception as e:
        raise RuntimeError(
            f"Failed to read merged lesion probability map via SimpleITK: {prob_path} ({e})"
        ) from e
    prob_ref = sitk.Resample(
        prob_img,
        organ_ref,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    prob_zyx = sitk.GetArrayFromImage(prob_ref).astype("float32")
    prob_zyx = prob_zyx * (organ_zyx > 0).astype("float32")

    raw_mask_zyx = (prob_zyx >= float(threshold)).astype("uint8")
    raw_mask_img = sitk.GetImageFromArray(raw_mask_zyx)
    raw_mask_img.CopyInformation(organ_ref)
    sitk.WriteImage(raw_mask_img, str(raw_mask_path))

    lab_zyx = cc_label(raw_mask_zyx > 0)
    comps_all: List[Dict[str, Any]] = []
    for cid in range(1, int(np.max(lab_zyx)) + 1):
        vox = np.argwhere(lab_zyx == cid)  # (z,y,x)
        if vox.size == 0:
            continue
        zs, ys, xs = vox[:, 0], vox[:, 1], vox[:, 2]
        comp_prob = prob_zyx[lab_zyx == cid]
        comps_all.append(
            {
                "component_id": int(cid),
                "n_voxels": int(vox.shape[0]),
                "bbox_xyz": [int(xs.min()), int(ys.min()), int(zs.min()), int(xs.max()), int(ys.max()), int(zs.max())],
                "center_z": int(round(float(zs.mean()))),
                "max_prob": float(comp_prob.max()) if comp_prob.size else 0.0,
                "mean_prob": float(comp_prob.mean()) if comp_prob.size else 0.0,
            }
        )

    spacing = organ_ref.GetSpacing()  # (x,y,z)
    voxel_volume_cc = (float(spacing[0]) * float(spacing[1]) * float(spacing[2])) / 1000.0
    comps_min_vox = [c for c in comps_all if int(c["n_voxels"]) >= int(min_vox)]
    comps_with_volume: List[Dict[str, Any]] = []
    for c in comps_min_vox:
        r = dict(c)
        r["volume_cc"] = float(c["n_voxels"]) * float(voxel_volume_cc)
        comps_with_volume.append(r)
    comps_filtered = [c for c in comps_with_volume if float(c.get("volume_cc", 0.0)) >= float(min_volume_cc)]
    comps_sorted = sorted(comps_filtered, key=lambda r: float(r.get("max_prob", 0.0)), reverse=True)
    comps = comps_sorted[: max(0, top_k)]

    selected_ids = {int(c.get("component_id")) for c in comps if c.get("component_id") is not None}
    filtered_lab_zyx = np.zeros_like(lab_zyx, dtype="uint16")
    for cid in selected_ids:
        filtered_lab_zyx[lab_zyx == int(cid)] = int(cid)
    filtered_mask_zyx = (filtered_lab_zyx > 0).astype("uint8")

    mask_img = sitk.GetImageFromArray(filtered_mask_zyx)
    mask_img.CopyInformation(organ_ref)
    sitk.WriteImage(mask_img, str(mask_path))

    if label_path is not None:
        label_img = sitk.GetImageFromArray(filtered_lab_zyx.astype("uint16"))
        label_img.CopyInformation(organ_ref)
        sitk.WriteImage(label_img, str(label_path))

    for c in comps:
        try:
            cid = int(c.get("component_id"))
        except Exception:
            continue
        comp_mask_zyx = (filtered_lab_zyx == cid).astype("uint8")
        comp_img = sitk.GetImageFromArray(comp_mask_zyx)
        comp_img.CopyInformation(organ_ref)
        comp_path = out_dir / f"lesion_component_{cid}.nii.gz"
        sitk.WriteImage(comp_img, str(comp_path))
        component_mask_paths.append(
            {
                "component_id": cid,
                "path": str(comp_path),
                "volume_cc": float(c.get("volume_cc") or 0.0),
            }
        )

    filter_stats = {
        "threshold": float(threshold),
        "min_voxels": int(min_vox),
        "min_volume_cc": float(min_volume_cc),
        "voxel_volume_cc": float(voxel_volume_cc),
        "top_k": int(top_k),
        "n_components_total": int(len(comps_all)),
        "n_components_min_vox": int(len(comps_min_vox)),
        "n_components_min_volume": int(len(comps_filtered)),
        "n_components_top_k": int(len(comps)),
        "max_prob_overall": float(np.max(prob_zyx)) if prob_zyx.size else 0.0,
        "max_prob_after_filter": float(comps[0]["max_prob"]) if comps else 0.0,
        "mask_inside_prostate_enforced": True,
    }

    candidates_path.write_text(
        json.dumps({"threshold": threshold, "filter_stats": filter_stats, "candidates": comps}, indent=2) + "\n",
        encoding="utf-8",
    )

    # ---- Display & montage generation ----
    # Load image arrays with nibabel for corrected-path robustness (avoid extra SITK reads on corrected volumes).
    t2_zyx = _safe_nib_zyx(t2, nib=nib, np=np, label="t2w_nifti")
    adc_zyx = _safe_nib_zyx(adc, nib=nib, np=np, label="adc_nifti")
    highb_zyx = _safe_nib_zyx(highb, nib=nib, np=np, label="highb_nifti")
    zone_zyx = organ_zyx.astype("uint8", copy=False)

    def _match_shape(vol, target_shape):
        if tuple(vol.shape) == tuple(target_shape):
            return vol
        return resize(vol, output_shape=target_shape, order=1, preserve_range=True).astype("float32")

    # Ensure all volumes match the prostate mask shape for visualization.
    t2_zyx = _match_shape(t2_zyx, zone_zyx.shape)
    adc_zyx = _match_shape(adc_zyx, zone_zyx.shape)
    highb_zyx = _match_shape(highb_zyx, zone_zyx.shape)

    # Lesion mask for montage (z,y,x)
    lesion_mask_zyx = filtered_mask_zyx.astype("uint8", copy=False)
    component_labels_zyx = filtered_lab_zyx.astype("uint16", copy=False)
    try:
        if tuple(lesion_mask_zyx.shape) != tuple(zone_zyx.shape):
            lesion_mask_zyx = (resize(lesion_mask_zyx, output_shape=zone_zyx.shape, order=0, preserve_range=True) > 0).astype("uint8")
    except Exception:
        lesion_mask_zyx = None
    try:
        if tuple(component_labels_zyx.shape) != tuple(zone_zyx.shape):
            component_labels_zyx = resize(component_labels_zyx, output_shape=zone_zyx.shape, order=0, preserve_range=True).astype("uint16")
    except Exception:
        component_labels_zyx = None

    # Pick z_list to match feature montages (whole gland coverage).
    z_list: List[int] = []
    try:
        overlays_dir = ctx.artifacts_dir / "features" / "overlays"
        if overlays_dir.exists():
            candidates = [
                overlays_dir / "T2w_t2w_input_prostate_full_frame_montage_4x4_tiles.json",
                overlays_dir / "T2w_prostate_full_frame_montage_4x4_tiles.json",
                overlays_dir / "ADC_prostate_full_frame_montage_4x4_tiles.json",
                overlays_dir / "high_b_prostate_full_frame_montage_4x4_tiles.json",
                overlays_dir / "T2w_t2w_input_prostate_roi_crop_montage_4x4_tiles.json",
                overlays_dir / "T2w_prostate_roi_crop_montage_4x4_tiles.json",
                overlays_dir / "ADC_prostate_roi_crop_montage_4x4_tiles.json",
                overlays_dir / "high_b_prostate_roi_crop_montage_4x4_tiles.json",
            ]
            feat_tiles = None
            for c in candidates:
                if c.exists():
                    feat_tiles = c
                    break
            if feat_tiles is None:
                any_tiles = sorted(overlays_dir.glob("*_prostate_full_frame_montage_4x4_tiles.json"))
                if any_tiles:
                    feat_tiles = any_tiles[0]
            if feat_tiles is None:
                any_tiles = sorted(overlays_dir.glob("*_prostate_roi_crop_montage_4x4_tiles.json"))
                if any_tiles:
                    feat_tiles = any_tiles[0]
            if feat_tiles is None:
                any_tiles = sorted(overlays_dir.glob("*_full_frame_montage_4x4_tiles.json"))
                if any_tiles:
                    feat_tiles = any_tiles[0]
            if feat_tiles is None:
                any_tiles = sorted(overlays_dir.glob("*_roi_crop_montage_4x4_tiles.json"))
                if any_tiles:
                    feat_tiles = any_tiles[0]
            if feat_tiles is not None and feat_tiles.exists():
                feat_obj = json.loads(feat_tiles.read_text(encoding="utf-8"))
                feat_z = [
                    t.get("z")
                    for t in (feat_obj.get("tiles") or [])
                    if t.get("has_roi") and t.get("z") is not None
                ]
                if feat_z:
                    z_list = [int(z) for z in feat_z]
    except Exception:
        pass
    if not z_list:
        idx = np.where(np.any(zone_zyx > 0, axis=(1, 2)))[0].tolist()  # z indices
        if idx:
            if len(idx) > 16:
                picks = np.linspace(0, len(idx) - 1, num=16)
                z_list = [int(idx[int(round(p))]) for p in picks]
            else:
                z_list = [int(zc) for zc in idx]

    # Write per-sequence candidate montages
    t2_png = out_dir / "lesion_candidates_T2w_4x4.png"
    t2_tiles = out_dir / "lesion_candidates_T2w_4x4_tiles.json"
    _write_candidate_montage(
        seq_name="T2w",
        vol_xyz=t2_zyx,
        zone_xyz=zone_zyx,
        lesion_mask_zyx=lesion_mask_zyx,
        component_labels_zyx=component_labels_zyx,
        z_list=z_list,
        out_png=t2_png,
        out_tiles=t2_tiles,
        np=np,
    )

    adc_png = out_dir / "lesion_candidates_ADC_4x4.png"
    adc_tiles = out_dir / "lesion_candidates_ADC_4x4_tiles.json"
    _write_candidate_montage(
        seq_name="ADC",
        vol_xyz=adc_zyx,
        zone_xyz=zone_zyx,
        lesion_mask_zyx=lesion_mask_zyx,
        component_labels_zyx=component_labels_zyx,
        z_list=z_list,
        out_png=adc_png,
        out_tiles=adc_tiles,
        np=np,
    )

    hb_png = out_dir / "lesion_candidates_high_b_4x4.png"
    hb_tiles = out_dir / "lesion_candidates_high_b_4x4_tiles.json"
    _write_candidate_montage(
        seq_name="high_b",
        vol_xyz=highb_zyx,
        zone_xyz=zone_zyx,
        lesion_mask_zyx=lesion_mask_zyx,
        component_labels_zyx=component_labels_zyx,
        z_list=z_list,
        out_png=hb_png,
        out_tiles=hb_tiles,
        np=np,
    )

    # Return
    artifacts = [
        {"path": str(prob_path), "kind": "nifti", "description": "Merged lesion probability map"},
        {"path": str(raw_mask_path), "kind": "nifti", "description": "Thresholded lesion mask before component filtering"},
        {"path": str(mask_path), "kind": "nifti", "description": "Thresholded lesion mask"},
        {"path": str(candidates_path), "kind": "json", "description": "Lesion candidates summary"},
        {"path": str(t2_png), "kind": "image", "description": "Lesion candidates montage (T2w)"},
        {"path": str(adc_png), "kind": "image", "description": "Lesion candidates montage (ADC)"},
        {"path": str(hb_png), "kind": "image", "description": "Lesion candidates montage (high_b)"},
        {"path": str(t2_tiles), "kind": "json", "description": "Tiles index (T2w candidates montage)"},
        {"path": str(adc_tiles), "kind": "json", "description": "Tiles index (ADC candidates montage)"},
        {"path": str(hb_tiles), "kind": "json", "description": "Tiles index (high_b candidates montage)"},
    ]
    if label_path is not None and label_path.exists():
        artifacts.append(
            {"path": str(label_path), "kind": "nifti", "description": "Component label map after filtering/top-k"}
        )
    for cm in component_mask_paths:
        p = str(cm.get("path") or "").strip()
        if p:
            artifacts.append({"path": p, "kind": "nifti", "description": f"Component mask (id={cm.get('component_id')})"})

    return {
        "data": {
            "inputs": {
                "t2w_nifti": str(t2),
                "adc_nifti": str(adc),
                "highb_nifti": str(highb),
                "prostate_mask_nifti": str(organ),
                "weights_dir": str(weights_dir),
                "spacing_target": list(spacing_target),
            },
            "merged_prob_path": str(prob_path),
            "lesion_mask_path": str(mask_path),
            "lesion_mask_raw_path": str(raw_mask_path),
            "candidates_path": str(candidates_path),
            "filter_stats": filter_stats,
            "component_mask_paths": component_mask_paths,
            "candidate_montages": [
                {"sequence": "T2w", "path": str(t2_png), "tiles": str(t2_tiles)},
                {"sequence": "ADC", "path": str(adc_png), "tiles": str(adc_tiles)},
                {"sequence": "high_b", "path": str(hb_png), "tiles": str(hb_tiles)},
            ],
        },
        "artifacts": artifacts,
        "warnings": warnings,
        "source_artifacts": [
            {"path": str(t2), "kind": "ref", "description": "T2w nifti"},
            {"path": str(adc), "kind": "ref", "description": "ADC nifti"},
            {"path": str(highb), "kind": "ref", "description": "High-b nifti"},
            {"path": str(organ), "kind": "ref", "description": "Prostate mask nifti"},
            {"path": str(weights_dir), "kind": "dir", "description": "Lesion model weights"},
        ],
        "generated_artifacts": artifacts,
    }


def build_tool():
    from commands.registry import Tool  # type: ignore
    from commands.schemas import ToolSpec  # type: ignore

    # Use ToolSpec to be consistent with other tools.
    spec = ToolSpec(
        name=LESION_SPEC["name"],
        description=LESION_SPEC["description"],
        input_schema=LESION_SPEC["input_schema"],
        output_schema={},
        version="0.1.0",
        tags=["lesion", "monai"],
    )
    return Tool(spec=spec, func=detect_lesion_candidates)

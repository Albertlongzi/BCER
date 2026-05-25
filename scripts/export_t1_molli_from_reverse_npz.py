#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np


DEFAULT_TI_MS = [120.0, 200.0, 280.0, 360.0, 440.0, 1000.0, 1080.0, 1160.0]


def _parse_ti_list(raw: str) -> List[float]:
    vals: List[float] = []
    for token in str(raw or "").split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(float(t))
    return vals or list(DEFAULT_TI_MS)


def _as_slice_first_posterior(arr: np.ndarray) -> np.ndarray:
    """
    Expected layout from cmr_reverse reverse_single_cine.py:
      posterior shape = (S, 3, H, W), channels=[PD, T1(ms), T2(ms)].
    """
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D posterior, got shape={arr.shape}")
    if arr.shape[1] == 3:
        return arr
    if arr.shape[0] == 3:
        return np.transpose(arr, (1, 0, 2, 3))
    raise ValueError(f"Cannot infer channel axis for posterior shape={arr.shape}")


def _to_xyz(slice_first: np.ndarray) -> np.ndarray:
    # (S, H, W) -> (H, W, S) for NIfTI
    return np.transpose(slice_first, (1, 2, 0))


def _look_locker_signal(pd: np.ndarray, t1_ms: np.ndarray, t2_ms: np.ndarray, *, ti_ms: float, fa_deg: float) -> np.ndarray:
    eps = 1e-6
    t1 = np.clip(t1_ms.astype(np.float32), eps, None)
    t2 = np.clip(t2_ms.astype(np.float32), eps, None)
    rho = pd.astype(np.float32)

    fa = np.deg2rad(float(fa_deg))
    sin_fa = np.sin(fa)
    cos_fa = np.cos(fa)
    ratio = t1 / t2

    steady_state = rho / (1.0 + cos_fa + (1.0 - cos_fa) * ratio)
    inv_factor = 1.0 + (np.sin(fa * 0.5) / max(float(sin_fa), eps)) * (ratio * (1.0 - cos_fa) + 1.0 + cos_fa)
    t1app_inv = (np.cos(fa * 0.5) ** 2) / t1 + (np.sin(fa * 0.5) ** 2) / t2
    t1app = 1.0 / np.clip(t1app_inv, eps, None)

    signal = steady_state * (1.0 - inv_factor * np.exp(-float(ti_ms) / t1app))
    return np.abs(signal)


def _save_like_ref(data_xyz: np.ndarray, out_path: Path, ref_img: nib.Nifti1Image) -> None:
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.float32)
    out = nib.Nifti1Image(data_xyz.astype(np.float32), ref_img.affine, header=hdr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser(description="Export T1 MOLLI-style NIfTI maps from cmr_reverse reverse_single_cine NPZ output.")
    ap.add_argument("--npz", required=True, help="Path to reverse_single_cine output .npz")
    ap.add_argument("--reference-nifti", required=True, help="Reference NIfTI for affine/header/shape")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--prefix", default="", help="Output file prefix (default: npz stem)")
    ap.add_argument("--pd-div", type=float, default=1.0, help="Optional PD normalization divisor")
    ap.add_argument("--fa-deg", type=float, default=35.0, help="Flip angle used for MOLLI signal synthesis")
    ap.add_argument("--ti-ms", default=",".join([str(int(x)) for x in DEFAULT_TI_MS]), help="Comma-separated inversion times in ms")
    ap.add_argument("--export-molli-series", action="store_true", help="Also export synthetic MOLLI images at --ti-ms")
    ap.add_argument("--save-all-maps", action="store_true", help="Also save PD/T2 maps")
    args = ap.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    ref_path = Path(args.reference_nifti).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    prefix = str(args.prefix or npz_path.stem).strip()
    ti_list = _parse_ti_list(str(args.ti_ms))

    bundle = np.load(npz_path)
    if "posterior" not in bundle.files:
        raise KeyError(f"{npz_path} missing 'posterior'. Found keys: {bundle.files}")

    posterior = _as_slice_first_posterior(np.asarray(bundle["posterior"]))
    pd = posterior[:, 0].astype(np.float32)
    t1 = posterior[:, 1].astype(np.float32)
    t2 = posterior[:, 2].astype(np.float32)

    pd_div = float(args.pd_div)
    if pd_div and pd_div != 1.0:
        pd = pd / pd_div

    ref_img = nib.load(str(ref_path))
    t1_xyz = _to_xyz(t1)
    _save_like_ref(t1_xyz, out_dir / f"{prefix}_t1_molli_map.nii.gz", ref_img)

    if args.save_all_maps:
        _save_like_ref(_to_xyz(pd), out_dir / f"{prefix}_pd_map.nii.gz", ref_img)
        _save_like_ref(_to_xyz(t2), out_dir / f"{prefix}_t2_map.nii.gz", ref_img)

    if args.export_molli_series:
        for ti in ti_list:
            sig = _look_locker_signal(pd, t1, t2, ti_ms=float(ti), fa_deg=float(args.fa_deg))
            _save_like_ref(_to_xyz(sig), out_dir / f"{prefix}_molli_ti{int(round(ti)):04d}.nii.gz", ref_img)

    print(f"[OK] Exported T1 MOLLI outputs to: {out_dir}")


if __name__ == "__main__":
    main()


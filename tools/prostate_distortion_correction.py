"""
Tool: correct_prostate_distortion

Purpose:
- Run prostate DWI/ADC distortion correction using a case-local NPZ payload generated
  from current run artifacts (T2w + ADC + high-b DWI).
- Keep all generated inputs/outputs under the current run artifact directory.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec


SPEC = ToolSpec(
    name="correct_prostate_distortion",
    description=(
        "Correct prostate DWI/ADC distortion using an external diffusion backend. "
        "Reads case-local NIfTI inputs (T2w/ADC/high-b), writes corrected NIfTI volumes "
        "and backend artifacts under the run artifact directory."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "t2w_nifti": {"type": "string"},
            "adc_nifti": {"type": "string"},
            "highb_nifti": {"type": "string"},
            "python_exec": {"type": "string"},
            "output_subdir": {"type": "string"},
            "device": {"type": "string"},
            "radius": {"type": "integer"},
            "max_cases": {"type": "integer"},
            "steps": {"type": "integer"},
            "strength": {"type": "number"},
            "eta": {"type": "number"},
            "sampler": {"type": "string"},
            "b_low": {"type": "number"},
            "b_high": {"type": "number"},
            "slice_mode": {"type": "string"},
            "save_npz": {"type": "boolean"},
            "save_slices": {"type": "boolean"},
            "gamma_b14": {"type": "number"},
            "gamma_adc": {"type": "number"},
            "adc_plo": {"type": "number"},
            "adc_phi": {"type": "number"},
            "t2_cond_channels": {"type": "integer"},
            "t2_contrast_mod": {"type": "string"},
            "t2_canny_low": {"type": "integer"},
            "t2_canny_high": {"type": "integer"},
            "cnn_base_channels": {"type": "integer"},
            "cnn_latent_dim": {"type": "integer"},
            "cnn_prompt_k": {"type": "integer"},
            "cnn_prompt_temp": {"type": "number"},
            "num_gpus": {"type": "integer"},
            "target_inplane_h": {"type": "integer"},
            "target_inplane_w": {"type": "integer"},
            "target_spacing_x_mm": {"type": "number"},
            "target_spacing_y_mm": {"type": "number"},
            "timeout_sec": {"type": "integer"},
        },
        "required": ["t2w_nifti", "adc_nifti", "highb_nifti"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "out_dir": {"type": "string"},
            "summary_json_path": {"type": "string"},
            "backend_out_dir": {"type": "string"},
            "input_npz_path": {"type": "string"},
            "prediction_npz_path": {"type": "string"},
            "panel_png_path": {"type": "string"},
            "t2w_nifti": {"type": "string"},
            "corrected_b50_nifti": {"type": "string"},
            "corrected_highb_nifti": {"type": "string"},
            "corrected_adc_nifti": {"type": "string"},
            "command": {"type": "string"},
            "num_pred_npz": {"type": "integer"},
            "num_panel_png": {"type": "integer"},
            "num_slice_png": {"type": "integer"},
            "runtime_sec": {"type": "number"},
            "stdout_tail": {"type": "string"},
            "stderr_tail": {"type": "string"},
            "note": {"type": "string"},
        },
        "required": ["out_dir", "summary_json_path", "corrected_highb_nifti", "corrected_adc_nifti"],
    },
    version="0.2.0",
    tags=["prostate", "distortion", "diffusion", "sandboxed"],
)


def _require_deps():
    try:
        import nibabel as nib  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Missing dependency for correct_prostate_distortion: {repr(e)}") from e
    return nib, np


def _as_int(args: Dict[str, Any], key: str, default: int) -> int:
    try:
        if key in args and args[key] is not None:
            return int(args[key])
    except Exception:
        pass
    return int(default)


def _as_float(args: Dict[str, Any], key: str, default: float) -> float:
    try:
        if key in args and args[key] is not None:
            return float(args[key])
    except Exception:
        pass
    return float(default)


def _as_str(args: Dict[str, Any], key: str, default: str) -> str:
    v = args.get(key)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _as_bool(args: Dict[str, Any], key: str, default: bool) -> bool:
    if key not in args:
        return bool(default)
    v = args.get(key)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _tail_text(text: str, max_lines: int = 120, max_chars: int = 6000) -> str:
    lines = str(text or "").splitlines()
    tail = lines if len(lines) <= max_lines else lines[-max_lines:]
    out = "\n".join(tail)
    if len(out) <= max_chars:
        return out
    return "[TRUNCATED]\n" + out[-max_chars:]


def _resolve_existing_file(raw: str, field: str) -> Path:
    p = Path(str(raw or "")).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f"{field} not found: {p}")
    if not p.is_file():
        raise RuntimeError(f"{field} must be a file path: {p}")
    return p


def _pick_from_env(names: List[str]) -> str:
    for key in names:
        val = str(os.getenv(key, "")).strip()
        if val:
            return val
    return ""


def _infer_xy_spacing_from_img(img: Any, *, np) -> Tuple[float, float]:
    try:
        z = img.header.get_zooms()
        if len(z) >= 2:
            sx = float(z[0])
            sy = float(z[1])
            if np.isfinite(sx) and np.isfinite(sy) and sx > 0.0 and sy > 0.0:
                return (sx, sy)
    except Exception:
        pass
    return (0.5625, 0.5625)


def _resample_xy_and_center_crop(
    vol: Any,
    *,
    src_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
    target_hw: Tuple[int, int],
    interp_order: int = 1,
    np,
) -> Any:
    arr = vol.astype(np.float32)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D array for XY resample/crop, got shape {tuple(arr.shape)}")

    zx = float(src_xy[0]) / max(1e-6, float(target_xy[0]))
    zy = float(src_xy[1]) / max(1e-6, float(target_xy[1]))
    zx = max(1e-6, zx)
    zy = max(1e-6, zy)
    out_h = max(1, int(round(arr.shape[0] * zx)))
    out_w = max(1, int(round(arr.shape[1] * zy)))

    ord_i = int(interp_order)
    if ord_i < 0:
        ord_i = 0
    if ord_i > 3:
        ord_i = 3

    if out_h != arr.shape[0] or out_w != arr.shape[1]:
        try:
            from scipy.ndimage import zoom  # type: ignore

            arr = zoom(arr, (zx, zy, 1.0), order=ord_i, mode="nearest").astype(np.float32)
        except Exception:
            try:
                from skimage.transform import resize  # type: ignore

                arr = resize(
                    arr,
                    output_shape=(out_h, out_w, int(arr.shape[2])),
                    order=ord_i,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.float32)
            except Exception as e:
                raise RuntimeError(
                    "Unable to resample XY plane: scipy.ndimage.zoom and skimage.transform.resize are both unavailable."
                ) from e

    target_h = max(1, int(target_hw[0]))
    target_w = max(1, int(target_hw[1]))
    h, w, z = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])

    h_start = max(0, (h - target_h) // 2)
    w_start = max(0, (w - target_w) // 2)
    h_end = min(h, h_start + target_h)
    w_end = min(w, w_start + target_w)
    cropped = arr[h_start:h_end, w_start:w_end, :]

    if int(cropped.shape[0]) == target_h and int(cropped.shape[1]) == target_w:
        return cropped.astype(np.float32)

    out = np.zeros((target_h, target_w, z), dtype=np.float32)
    ph = max(0, (target_h - int(cropped.shape[0])) // 2)
    pw = max(0, (target_w - int(cropped.shape[1])) // 2)
    out[ph : ph + int(cropped.shape[0]), pw : pw + int(cropped.shape[1]), :] = cropped
    return out.astype(np.float32)


def _normalize_adc_to_mm2_per_s(adc: Any, *, np) -> Tuple[Any, Dict[str, Any]]:
    arr = adc.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    p99_abs = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 0.0

    # Reference pipeline expects ADC in mm^2/s (~0..0.003). Clinical exports are often x10^-6 or x10^-3.
    adc_scale = 1.0
    unit_guess = "mm2_per_s"
    if p99_abs > 10.0:
        adc_scale = 1e-6
        unit_guess = "um2_per_s_x1e6"
    elif p99_abs > 0.02:
        adc_scale = 1e-3
        unit_guess = "um2_per_s_x1e3"

    adc_mm = (arr * float(adc_scale)).astype(np.float32)
    adc_mm = np.clip(adc_mm, 0.0, 0.003).astype(np.float32)
    meta = {
        "adc_unit_guess": unit_guess,
        "adc_scale_applied": float(adc_scale),
        "adc_p99_abs_before_scale": float(p99_abs),
        "adc_p99_after_scale": float(np.percentile(adc_mm, 99.0)) if adc_mm.size else 0.0,
    }
    return adc_mm, meta


def _align_shape(
    vol: Any,
    target_shape: Tuple[int, int, int],
    *,
    interp_order: int = 1,
    np,
    label: str,
) -> Any:
    if tuple(vol.shape) == tuple(target_shape):
        return vol
    try:
        from skimage.transform import resize  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"{label} shape {tuple(vol.shape)} != target {tuple(target_shape)} and skimage is unavailable: {repr(e)}"
        ) from e
    ord_i = int(interp_order)
    if ord_i < 0:
        ord_i = 0
    if ord_i > 3:
        ord_i = 3
    return resize(
        vol.astype(np.float32),
        output_shape=target_shape,
        order=ord_i,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32)


def _load_volume(path: Path, *, nib, np) -> Tuple[Any, Any]:
    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)
    if vol.ndim != 3:
        raise RuntimeError(f"Expected 3D volume at {path}, got shape {tuple(vol.shape)}")
    return img, vol


def _save_like(path: Path, ref_img: Any, vol: Any, *, nib, np) -> Path:
    hdr = ref_img.header.copy()
    out = nib.Nifti1Image(vol.astype(np.float32), ref_img.affine, header=hdr)
    out.set_data_dtype(np.float32)
    try:
        qform, qcode = ref_img.get_qform(coded=True)
        sform, scode = ref_img.get_sform(coded=True)
        if qform is not None:
            out.set_qform(qform, int(qcode) if qcode is not None else 1)
        if sform is not None:
            out.set_sform(sform, int(scode) if scode is not None else 1)
    except Exception:
        pass
    try:
        # Keep scaling explicit/identity to avoid downstream reader surprises.
        out.header["scl_slope"] = 1.0
        out.header["scl_inter"] = 0.0
    except Exception:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(path))
    return path


def _build_backend_input_npz(
    *,
    t2_path: Path,
    adc_path: Path,
    highb_path: Path,
    npz_path: Path,
    b_low: float,
    b_high: float,
    target_hw: Tuple[int, int],
    target_spacing_xy: Tuple[float, float],
    nib,
    np,
) -> Dict[str, Any]:
    t2_img, t2 = _load_volume(t2_path, nib=nib, np=np)
    _, adc = _load_volume(adc_path, nib=nib, np=np)
    _, highb = _load_volume(highb_path, nib=nib, np=np)

    target_shape = tuple(int(x) for x in t2.shape)
    adc = _align_shape(adc, target_shape, interp_order=1, np=np, label="adc_nifti")
    # Preserve sparse bright foci in high-b diffusion by avoiding linear smoothing here.
    highb = _align_shape(highb, target_shape, interp_order=0, np=np, label="highb_nifti")

    delta_b = max(1.0, float(b_high - b_low))
    adc_mm, adc_meta = _normalize_adc_to_mm2_per_s(adc, np=np)
    highb_nonneg = np.clip(highb.astype(np.float32), 0.0, None)
    b50_raw = (highb_nonneg * np.exp(adc_mm * delta_b)).astype(np.float32)
    src_xy = _infer_xy_spacing_from_img(t2_img, np=np)

    b50 = _resample_xy_and_center_crop(
        b50_raw,
        src_xy=src_xy,
        target_xy=target_spacing_xy,
        target_hw=target_hw,
        interp_order=0,
        np=np,
    )
    b1400 = _resample_xy_and_center_crop(
        highb_nonneg,
        src_xy=src_xy,
        target_xy=target_spacing_xy,
        target_hw=target_hw,
        interp_order=0,
        np=np,
    )
    adc_in = _resample_xy_and_center_crop(
        adc_mm,
        src_xy=src_xy,
        target_xy=target_spacing_xy,
        target_hw=target_hw,
        interp_order=1,
        np=np,
    )
    t2_backend = _resample_xy_and_center_crop(
        t2.astype(np.float32),
        src_xy=src_xy,
        target_xy=target_spacing_xy,
        target_hw=target_hw,
        interp_order=1,
        np=np,
    )

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(npz_path),
        dwi_b50_in=b50,
        dwi_b1400_in=b1400.astype(np.float32),
        adc_in=adc_in.astype(np.float32),
        t2=t2_backend.astype(np.float32),
    )
    return {
        "t2_ref_img": t2_img,
        "target_shape": target_shape,
        "input_npz_shape": tuple(int(x) for x in b50.shape),
        "preprocess": {
            "source_xy_mm": [float(src_xy[0]), float(src_xy[1])],
            "target_xy_mm": [float(target_spacing_xy[0]), float(target_spacing_xy[1])],
            "target_inplane_hw": [int(target_hw[0]), int(target_hw[1])],
            "interp_order": {
                "t2": 1,
                "adc": 1,
                "highb": 0,
                "b50": 0,
            },
            **adc_meta,
        },
    }


def _resolve_backend_config(args: Dict[str, Any]) -> Dict[str, str]:
    python_exec = _as_str(
        args,
        "python_exec",
        _pick_from_env(["MRI_AGENT_PROSTATE_DISTORTION_PYTHON", "MRI_AGENT_PYTHON", "PYTHON"]),
    ) or "python"

    script_raw = _pick_from_env(
        ["MRI_AGENT_PROSTATE_DISTORTION_SCRIPT_PATH", "MRI_AGENT_PROSTATE_DISTORTION_SCRIPT"]
    )
    ckpt_raw = _pick_from_env(["MRI_AGENT_PROSTATE_DISTORTION_DIFF_CKPT"])
    cnn_ckpt_raw = _pick_from_env(["MRI_AGENT_PROSTATE_DISTORTION_CNN_CKPT"])

    if not script_raw:
        raise RuntimeError(
            "Missing backend script path. Set MRI_AGENT_PROSTATE_DISTORTION_SCRIPT_PATH "
            "(or MRI_AGENT_PROSTATE_DISTORTION_SCRIPT)."
        )
    if not ckpt_raw:
        raise RuntimeError("Missing diffusion checkpoint path. Set MRI_AGENT_PROSTATE_DISTORTION_DIFF_CKPT.")
    if not cnn_ckpt_raw:
        raise RuntimeError("Missing CNN checkpoint path. Set MRI_AGENT_PROSTATE_DISTORTION_CNN_CKPT.")

    script_path = _resolve_existing_file(script_raw, "backend_script_path")
    ckpt = _resolve_existing_file(ckpt_raw, "diffusion_checkpoint")
    cnn_ckpt = _resolve_existing_file(cnn_ckpt_raw, "cnn_checkpoint")

    return {
        "python_exec": python_exec,
        "script_path": str(script_path),
        "ckpt": str(ckpt),
        "cnn_ckpt": str(cnn_ckpt),
    }


def _pick_prediction_volumes(pred_npz_path: Path, *, np) -> Dict[str, Any]:
    with np.load(str(pred_npz_path), allow_pickle=False) as d:
        adc_diff = d.get("adc_diff")
        adc_cnn = d.get("adc_cnn")
        b50_diff = d.get("dwi_b50_diff")
        b50_cnn = d.get("dwi_b50_cnn")
        b1400_diff = d.get("dwi_b1400_diff")
        b1400_cnn = d.get("dwi_b1400_cnn")

        adc = adc_diff if adc_diff is not None else adc_cnn
        b50 = b50_diff if b50_diff is not None else b50_cnn
        b1400 = b1400_diff if b1400_diff is not None else b1400_cnn

    if adc is None or b1400 is None:
        raise RuntimeError(f"Prediction NPZ missing corrected ADC/high-b outputs: {pred_npz_path}")

    adc = adc.astype(np.float32)
    b1400 = b1400.astype(np.float32)
    if b50 is None:
        b50 = b1400.astype(np.float32)
    else:
        b50 = b50.astype(np.float32)
    return {
        "adc": adc,
        "b1400": b1400,
        "b50": b50,
    }


def correct_prostate_distortion(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    nib, np = _require_deps()

    t2_path = _resolve_existing_file(str(args.get("t2w_nifti") or ""), "t2w_nifti")
    adc_path = _resolve_existing_file(str(args.get("adc_nifti") or ""), "adc_nifti")
    highb_path = _resolve_existing_file(str(args.get("highb_nifti") or ""), "highb_nifti")

    out_subdir = _as_str(args, "output_subdir", "distortion_correction")
    out_dir = (ctx.artifacts_dir / out_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    input_root = out_dir / "input_npz"
    backend_out_dir = out_dir / "raw_predictions"
    input_npz_path = input_root / "case_input.npz"
    target_inplane_h = max(32, _as_int(args, "target_inplane_h", 256))
    target_inplane_w = max(32, _as_int(args, "target_inplane_w", 256))
    target_spacing_x_mm = max(1e-6, _as_float(args, "target_spacing_x_mm", 0.5625))
    target_spacing_y_mm = max(1e-6, _as_float(args, "target_spacing_y_mm", 0.5625))

    b_low = _as_float(args, "b_low", 50.0)
    b_high = _as_float(args, "b_high", 1400.0)
    prep = _build_backend_input_npz(
        t2_path=t2_path,
        adc_path=adc_path,
        highb_path=highb_path,
        npz_path=input_npz_path,
        b_low=b_low,
        b_high=b_high,
        target_hw=(target_inplane_h, target_inplane_w),
        target_spacing_xy=(target_spacing_x_mm, target_spacing_y_mm),
        nib=nib,
        np=np,
    )

    backend = _resolve_backend_config(args)
    backend_out_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        backend["python_exec"],
        "-u",
        backend["script_path"],
        "--ckpt",
        backend["ckpt"],
        "--cnn_ckpt",
        backend["cnn_ckpt"],
        "--test_root",
        str(input_root),
        "--out_dir",
        str(backend_out_dir),
        "--radius",
        str(_as_int(args, "radius", 2)),
        "--steps",
        str(_as_int(args, "steps", 100)),
        "--strength",
        str(_as_float(args, "strength", 0.3)),
        "--eta",
        str(_as_float(args, "eta", 0.0)),
        "--sampler",
        _as_str(args, "sampler", "dpmsolver"),
        "--slice_mode",
        _as_str(args, "slice_mode", "all"),
        "--b_low",
        str(b_low),
        "--b_high",
        str(b_high),
        "--gamma_b14",
        str(_as_float(args, "gamma_b14", 0.4)),
        "--gamma_adc",
        str(_as_float(args, "gamma_adc", 0.7)),
        "--adc_plo",
        str(_as_float(args, "adc_plo", 2.0)),
        "--adc_phi",
        str(_as_float(args, "adc_phi", 99.0)),
        "--t2_cond_channels",
        str(_as_int(args, "t2_cond_channels", 64)),
        "--t2_contrast_mod",
        _as_str(args, "t2_contrast_mod", "none"),
        "--t2_canny_low",
        str(_as_int(args, "t2_canny_low", 50)),
        "--t2_canny_high",
        str(_as_int(args, "t2_canny_high", 150)),
        "--cnn_base_channels",
        str(_as_int(args, "cnn_base_channels", 64)),
        "--cnn_latent_dim",
        str(_as_int(args, "cnn_latent_dim", 8)),
        "--cnn_prompt_k",
        str(_as_int(args, "cnn_prompt_k", 8)),
        "--cnn_prompt_temp",
        str(_as_float(args, "cnn_prompt_temp", 1.0)),
    ]
    device = str(args.get("device") or "").strip()
    if device:
        cmd.extend(["--device", device])
    max_cases = args.get("max_cases")
    if max_cases is not None:
        cmd.extend(["--max_cases", str(int(max_cases))])
    num_gpus = args.get("num_gpus")
    if num_gpus is not None:
        cmd.extend(["--num_gpus", str(int(num_gpus))])
    if _as_bool(args, "save_npz", True):
        cmd.append("--save_npz")
    if _as_bool(args, "save_slices", True):
        cmd.append("--save_slices")

    timeout_sec = _as_int(args, "timeout_sec", 6 * 60 * 60)
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(Path(backend["script_path"]).resolve().parent.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    runtime_sec = float(time.time() - t0)

    stdout_tail = _tail_text(proc.stdout, max_lines=120)
    stderr_tail = _tail_text(proc.stderr, max_lines=120)
    if proc.returncode != 0:
        raise RuntimeError(
            "correct_prostate_distortion failed "
            f"(exit={proc.returncode}).\n"
            f"STDOUT tail:\n{stdout_tail}\n\nSTDERR tail:\n{stderr_tail}"
        )

    pred_npz = sorted(backend_out_dir.glob("*_pred.npz"))
    panel_png = sorted(backend_out_dir.glob("*_panel.png"))
    slice_png = sorted(backend_out_dir.glob("*_slices/*.png"))
    if not pred_npz:
        raise RuntimeError(f"No backend prediction NPZ generated under {backend_out_dir}")

    pred_path = pred_npz[0]
    corrected = _pick_prediction_volumes(pred_path, np=np)
    target_shape = tuple(int(x) for x in prep["target_shape"])
    corr_adc = _align_shape(corrected["adc"], target_shape, interp_order=1, np=np, label="adc_corrected")
    # Keep nearest-neighbor back-projection for diffusion outputs to reduce blur accumulation.
    corr_highb = _align_shape(corrected["b1400"], target_shape, interp_order=0, np=np, label="highb_corrected")
    corr_b50 = _align_shape(corrected["b50"], target_shape, interp_order=0, np=np, label="b50_corrected")

    corrected_dir = out_dir / "corrected_nifti"
    corrected_adc_path = _save_like(
        corrected_dir / "adc_corrected.nii.gz", prep["t2_ref_img"], corr_adc, nib=nib, np=np
    )
    corrected_highb_path = _save_like(
        corrected_dir / "highb_corrected.nii.gz", prep["t2_ref_img"], corr_highb, nib=nib, np=np
    )
    corrected_b50_path = _save_like(
        corrected_dir / "b50_corrected.nii.gz", prep["t2_ref_img"], corr_b50, nib=nib, np=np
    )

    summary_path = out_dir / "distortion_correction_summary.json"
    summary = {
        "tool": SPEC.name,
        "version": SPEC.version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "t2w_nifti": str(t2_path),
            "adc_nifti": str(adc_path),
            "highb_nifti": str(highb_path),
            "b_low": b_low,
            "b_high": b_high,
            "target_inplane_h": int(target_inplane_h),
            "target_inplane_w": int(target_inplane_w),
            "target_spacing_x_mm": float(target_spacing_x_mm),
            "target_spacing_y_mm": float(target_spacing_y_mm),
        },
        "backend": {
            "script_path": backend["script_path"],
            "ckpt": backend["ckpt"],
            "cnn_ckpt": backend["cnn_ckpt"],
            "python_exec": backend["python_exec"],
        },
        "out_dir": str(out_dir),
        "backend_out_dir": str(backend_out_dir),
        "input_npz_path": str(input_npz_path),
        "prediction_npz_path": str(pred_path),
        "corrected_adc_nifti": str(corrected_adc_path),
        "corrected_highb_nifti": str(corrected_highb_path),
        "corrected_b50_nifti": str(corrected_b50_path),
        "command": shlex.join(cmd),
        "runtime_sec": runtime_sec,
        "return_code": int(proc.returncode),
        "input_npz_shape": prep.get("input_npz_shape"),
        "input_preprocess": prep.get("preprocess"),
        "num_pred_npz": len(pred_npz),
        "num_panel_png": len(panel_png),
        "num_slice_png": len(slice_png),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    artifacts: List[ArtifactRef] = [
        ArtifactRef(path=str(summary_path), kind="json", description="Distortion correction run summary"),
        ArtifactRef(path=str(out_dir), kind="directory", description="Distortion correction output directory"),
        ArtifactRef(path=str(corrected_adc_path), kind="nifti", description="Corrected ADC in T2w space"),
        ArtifactRef(path=str(corrected_highb_path), kind="nifti", description="Corrected high-b DWI in T2w space"),
        ArtifactRef(path=str(corrected_b50_path), kind="nifti", description="Corrected b50 DWI in T2w space"),
        ArtifactRef(path=str(pred_path), kind="npz", description="Backend prediction NPZ"),
        ArtifactRef(path=str(input_npz_path), kind="npz", description="Case-local backend input NPZ"),
    ]
    if panel_png:
        artifacts.append(
            ArtifactRef(path=str(panel_png[0]), kind="figure", description="Distortion correction central panel")
        )
    for p in slice_png[:8]:
        artifacts.append(ArtifactRef(path=str(p), kind="figure", description="Distortion correction slice panel"))

    source_artifacts: List[ArtifactRef] = [
        ArtifactRef(path=str(t2_path), kind="nifti", description="Input T2w volume"),
        ArtifactRef(path=str(adc_path), kind="nifti", description="Input ADC volume"),
        ArtifactRef(path=str(highb_path), kind="nifti", description="Input high-b volume"),
        ArtifactRef(path=str(backend["script_path"]), kind="python", description="Backend inference script"),
        ArtifactRef(path=str(backend["ckpt"]), kind="checkpoint", description="Diffusion checkpoint"),
        ArtifactRef(path=str(backend["cnn_ckpt"]), kind="checkpoint", description="CNN checkpoint"),
    ]

    return {
        "data": {
            "out_dir": str(out_dir),
            "summary_json_path": str(summary_path),
            "backend_out_dir": str(backend_out_dir),
            "input_npz_path": str(input_npz_path),
            "prediction_npz_path": str(pred_path),
            "panel_png_path": str(panel_png[0]) if panel_png else None,
            "t2w_nifti": str(t2_path),
            "corrected_b50_nifti": str(corrected_b50_path),
            "corrected_highb_nifti": str(corrected_highb_path),
            "corrected_adc_nifti": str(corrected_adc_path),
            "command": shlex.join(cmd),
            "num_pred_npz": len(pred_npz),
            "num_panel_png": len(panel_png),
            "num_slice_png": len(slice_png),
            "runtime_sec": runtime_sec,
            "input_npz_shape": prep.get("input_npz_shape"),
            "input_preprocess": prep.get("preprocess"),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "note": "Case-local distortion correction completed.",
        },
        "artifacts": artifacts,
        "warnings": [],
        "source_artifacts": source_artifacts,
        "generated_artifacts": artifacts,
    }


def build_tool() -> Tool:
    return Tool(spec=SPEC, func=correct_prostate_distortion)

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _require_libs():
    try:
        import numpy as np  # type: ignore
    except Exception as e:
        raise RuntimeError("numpy is required (pip install numpy).") from e
    try:
        import SimpleITK as sitk  # type: ignore
    except Exception as e:
        raise RuntimeError("SimpleITK is required (pip install SimpleITK).") from e
    return np, sitk


def _stats(vals: Any, np_mod: Any) -> Dict[str, float]:
    if vals is None or int(vals.size) == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np_mod.mean(vals)),
        "std": float(np_mod.std(vals)),
        "median": float(np_mod.median(vals)),
        "p10": float(np_mod.percentile(vals, 10)),
        "p90": float(np_mod.percentile(vals, 90)),
    }


def _same_grid(a: Any, b: Any, *, atol: float = 1e-4) -> bool:
    if tuple(a.GetSize()) != tuple(b.GetSize()):
        return False
    for x, y in zip(a.GetSpacing(), b.GetSpacing()):
        if abs(float(x) - float(y)) > atol:
            return False
    for x, y in zip(a.GetOrigin(), b.GetOrigin()):
        if abs(float(x) - float(y)) > atol:
            return False
    da = list(a.GetDirection())
    db = list(b.GetDirection())
    if len(da) != len(db):
        return False
    for x, y in zip(da, db):
        if abs(float(x) - float(y)) > atol:
            return False
    return True


def _resample_mask_to_ref(mask_img: Any, ref_img: Any, sitk: Any) -> Any:
    return sitk.Resample(
        mask_img,
        ref_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0.0,
        sitk.sitkUInt8,
    )


def _load_masks(
    *,
    label_seg_path: Optional[Path],
    myo_mask_path: Optional[Path],
    lv_mask_path: Optional[Path],
    sitk: Any,
) -> Tuple[Any, Any]:
    if label_seg_path is not None:
        seg = sitk.ReadImage(str(label_seg_path))
        seg_arr = sitk.GetArrayFromImage(seg)
        myo = sitk.GetImageFromArray((seg_arr == 2).astype("uint8"))
        lv = sitk.GetImageFromArray((seg_arr == 3).astype("uint8"))
        myo.CopyInformation(seg)
        lv.CopyInformation(seg)
        return myo, lv
    if myo_mask_path is None:
        raise ValueError("Provide --label-seg or --myo-mask.")
    myo = sitk.ReadImage(str(myo_mask_path))
    lv = sitk.ReadImage(str(lv_mask_path)) if lv_mask_path is not None else sitk.ReadImage(str(myo_mask_path))
    return myo, lv


def _zone_name(z: int) -> str:
    if z == 0:
        return "basal"
    if z == 1:
        return "mid"
    return "apical"


def _build_zone_map(valid_slices: List[int], np_mod: Any) -> Dict[int, int]:
    out: Dict[int, int] = {}
    if not valid_slices:
        return out
    parts = np_mod.array_split(np_mod.array(valid_slices, dtype=int), 3)
    for zi in parts[0].tolist():
        out[int(zi)] = 0
    for zi in parts[1].tolist():
        out[int(zi)] = 1
    for zi in parts[2].tolist():
        out[int(zi)] = 2
    return out


def _score_differential(
    *,
    cine_group: Optional[str],
    global_mean: float,
    global_std: float,
    normal_low: float,
    normal_high: float,
    high_thr: float,
    low_thr: float,
    sd_high_thr: float,
    local_high: bool,
) -> Dict[str, Any]:
    scores: Dict[str, float] = {}
    evidence: List[str] = []

    def add(name: str, val: float) -> None:
        scores[name] = float(scores.get(name, 0.0) + float(val))

    global_high = bool(global_mean >= high_thr)
    global_low = bool(global_mean <= low_thr)
    global_normal = bool(normal_low <= global_mean <= normal_high)
    sd_high = bool(global_std >= sd_high_thr)

    if global_high:
        evidence.append(f"Global mean T1 high ({global_mean:.1f} ms >= {high_thr:.1f} ms).")
        for n in ("DCM", "HCM", "Myocarditis", "Cardiac_Amyloidosis", "Diffuse_Fibrosis"):
            add(n, 2.0)

    if global_normal and local_high:
        evidence.append(
            f"Global T1 normal ({global_mean:.1f} ms) but local segments elevated."
        )
        add("MINF", 2.0)
        add("Focal_Fibrosis", 2.0)

    if sd_high:
        evidence.append(f"T1 heterogeneity high (SD={global_std:.1f} ms >= {sd_high_thr:.1f} ms).")
        add("MINF", 1.0)
        add("HCM", 1.0)
        add("Focal_Fibrosis", 1.0)

    if global_low:
        evidence.append(f"Global mean T1 low ({global_mean:.1f} ms <= {low_thr:.1f} ms).")
        add("Iron_or_Lipid_Deposition", 3.0)

    cine_norm = str(cine_group or "").strip().upper()
    if cine_norm in {"NOR", "MINF", "DCM", "HCM", "RV"}:
        add(cine_norm, 3.0)
        evidence.append(f"Cine phenotype suggests {cine_norm}.")

    ranked = sorted(
        [{"label": k, "score": float(v)} for k, v in scores.items() if float(v) > 0.0],
        key=lambda x: x["score"],
        reverse=True,
    )
    return {
        "global_high": global_high,
        "global_low": global_low,
        "global_normal": global_normal,
        "sd_high": sd_high,
        "local_segment_t1_high": bool(local_high),
        "evidence": evidence,
        "ranked_suspicions": ranked,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="T1 MOLLI feature pyramid + rule-based differential diagnosis using cardiac masks.")
    ap.add_argument("--t1-map", required=True, help="T1 map NIfTI (e.g., *_t1_molli_map.nii.gz)")
    ap.add_argument("--label-seg", default="", help="Label segmentation NIfTI (1=RV,2=MYO,3=LV)")
    ap.add_argument("--myo-mask", default="", help="Optional MYO mask NIfTI")
    ap.add_argument("--lv-mask", default="", help="Optional LV mask NIfTI")
    ap.add_argument("--cine-classification-json", default="", help="Optional cardiac_cine_classification.json")
    ap.add_argument("--subject-id", default="", help="Optional subject id")
    ap.add_argument("--phase", default="", help="Optional phase label (ED/ES)")
    ap.add_argument("--n-sectors", type=int, default=6)
    ap.add_argument("--segment-delta-high-ms", type=float, default=80.0)
    ap.add_argument("--global-normal-low-ms", type=float, default=950.0)
    ap.add_argument("--global-normal-high-ms", type=float, default=1300.0)
    ap.add_argument("--global-high-ms", type=float, default=1300.0)
    ap.add_argument("--global-low-ms", type=float, default=900.0)
    ap.add_argument("--sd-high-ms", type=float, default=120.0)
    ap.add_argument("--auto-resample-masks", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    np, sitk = _require_libs()

    t1_path = Path(args.t1_map).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    label_seg = Path(args.label_seg).expanduser().resolve() if str(args.label_seg).strip() else None
    myo_mask = Path(args.myo_mask).expanduser().resolve() if str(args.myo_mask).strip() else None
    lv_mask = Path(args.lv_mask).expanduser().resolve() if str(args.lv_mask).strip() else None

    t1_img = sitk.ReadImage(str(t1_path))
    myo_img, lv_img = _load_masks(label_seg_path=label_seg, myo_mask_path=myo_mask, lv_mask_path=lv_mask, sitk=sitk)

    resampled = False
    if not _same_grid(t1_img, myo_img):
        if not bool(args.auto_resample_masks):
            raise RuntimeError("Mask grid and T1 grid mismatch; rerun with --auto-resample-masks.")
        myo_img = _resample_mask_to_ref(myo_img, t1_img, sitk=sitk)
        lv_img = _resample_mask_to_ref(lv_img, t1_img, sitk=sitk)
        resampled = True

    t1 = sitk.GetArrayFromImage(t1_img).astype("float32")
    myo = (sitk.GetArrayFromImage(myo_img) > 0).astype("uint8")
    lv = (sitk.GetArrayFromImage(lv_img) > 0).astype("uint8")

    myo_vals = t1[myo > 0]
    if int(myo_vals.size) == 0:
        raise RuntimeError("Myocardium mask is empty after alignment.")

    global_stat = _stats(myo_vals, np)
    valid_slices = [int(z) for z in range(int(t1.shape[0])) if int(np.count_nonzero(myo[z] > 0)) > 0]
    zone_map = _build_zone_map(valid_slices, np)

    slice_rows: List[Dict[str, Any]] = []
    for z in valid_slices:
        vals = t1[z][myo[z] > 0]
        st = _stats(vals, np)
        slice_rows.append(
            {
                "slice_index": int(z),
                "zone": _zone_name(zone_map.get(int(z), 1)),
                "n_voxels": int(vals.size),
                **st,
            }
        )

    n_sectors = int(max(2, args.n_sectors))
    sector_vals_global: Dict[str, List[float]] = {f"s{i}": [] for i in range(n_sectors)}
    sector_vals_zone: Dict[str, Dict[str, List[float]]] = {
        _zone_name(zi): {f"s{i}": [] for i in range(n_sectors)} for zi in (0, 1, 2)
    }

    for z in valid_slices:
        m = myo[z] > 0
        if int(np.count_nonzero(m)) == 0:
            continue
        ys, xs = np.where(m)
        if int(np.count_nonzero(lv[z] > 0)) > 0:
            ly, lx = np.where(lv[z] > 0)
            cy = float(np.mean(ly))
            cx = float(np.mean(lx))
        else:
            cy = float(np.mean(ys))
            cx = float(np.mean(xs))
        vals = t1[z][m]
        ang = np.arctan2(ys.astype("float32") - cy, xs.astype("float32") - cx)
        ang = (ang + 2.0 * math.pi) % (2.0 * math.pi)
        sec = np.floor(ang / (2.0 * math.pi / float(n_sectors))).astype("int32")
        sec = np.clip(sec, 0, n_sectors - 1)
        zone_name = _zone_name(zone_map.get(int(z), 1))
        for i in range(n_sectors):
            sv = vals[sec == i]
            if int(sv.size) == 0:
                continue
            key = f"s{i}"
            sector_vals_global[key].extend([float(x) for x in sv.tolist()])
            sector_vals_zone[zone_name][key].extend([float(x) for x in sv.tolist()])

    sector_rows: List[Dict[str, Any]] = []
    for key in sorted(sector_vals_global.keys()):
        st = _stats(np.asarray(sector_vals_global[key], dtype="float32"), np)
        sector_rows.append({"level": "global_sector", "zone": "all", "sector": key, "n_voxels": int(len(sector_vals_global[key])), **st})
    for zn in ("basal", "mid", "apical"):
        for key in sorted(sector_vals_zone[zn].keys()):
            st = _stats(np.asarray(sector_vals_zone[zn][key], dtype="float32"), np)
            sector_rows.append(
                {"level": "zone_sector", "zone": zn, "sector": key, "n_voxels": int(len(sector_vals_zone[zn][key])), **st}
            )

    global_mean = float(global_stat["mean"])
    global_std = float(global_stat["std"])
    segment_delta = float(args.segment_delta_high_ms)
    local_high = False
    for r in sector_rows:
        mean_v = float(r.get("mean") or float("nan"))
        if not math.isnan(mean_v) and mean_v >= global_mean + segment_delta:
            local_high = True
            break

    cine_group = None
    if str(args.cine_classification_json).strip():
        cj = Path(args.cine_classification_json).expanduser().resolve()
        if cj.exists():
            try:
                obj = json.loads(cj.read_text(encoding="utf-8"))
                cine_group = obj.get("predicted_group") if isinstance(obj, dict) else None
            except Exception:
                cine_group = None

    differential = _score_differential(
        cine_group=cine_group,
        global_mean=global_mean,
        global_std=global_std,
        normal_low=float(args.global_normal_low_ms),
        normal_high=float(args.global_normal_high_ms),
        high_thr=float(args.global_high_ms),
        low_thr=float(args.global_low_ms),
        sd_high_thr=float(args.sd_high_ms),
        local_high=bool(local_high),
    )

    subject_id = str(args.subject_id or t1_path.stem).strip()
    phase = str(args.phase or "").strip().upper() or None
    summary = {
        "subject_id": subject_id,
        "phase": phase,
        "t1_map_path": str(t1_path),
        "cine_predicted_group": cine_group,
        "mask_resampled_to_t1": bool(resampled),
        "n_myo_voxels": int(myo_vals.size),
        "global_myo_t1": global_stat,
        "thresholds_ms": {
            "segment_delta_high": float(args.segment_delta_high_ms),
            "global_normal_low": float(args.global_normal_low_ms),
            "global_normal_high": float(args.global_normal_high_ms),
            "global_high": float(args.global_high_ms),
            "global_low": float(args.global_low_ms),
            "sd_high": float(args.sd_high_ms),
        },
        "differential": differential,
    }

    out_json = out_dir / "t1_pyramid_diagnosis.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    slice_csv = out_dir / "t1_slice_features.csv"
    with slice_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["slice_index", "zone", "n_voxels", "mean", "std", "median", "p10", "p90"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in slice_rows:
            w.writerow(r)

    sector_csv = out_dir / "t1_sector_pyramid_features.csv"
    with sector_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["level", "zone", "sector", "n_voxels", "mean", "std", "median", "p10", "p90"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sector_rows:
            w.writerow(r)

    print(f"[OK] JSON: {out_json}")
    print(f"[OK] Slice CSV: {slice_csv}")
    print(f"[OK] Sector CSV: {sector_csv}")


if __name__ == "__main__":
    main()


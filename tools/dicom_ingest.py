"""
Tool: identify_sequences (Stage 1; DICOM headers + optional DICOM->NIfTI conversion)

Purpose:
- Read DICOM headers (pydicom, stop_before_pixels) to build:
    - series_inventory.json (series list + key tags)
    - dicom_meta.json (curated per-series header + diffusion summary)
    - dicom_headers_index.txt + per-series header text (for LLM-friendly inspection)
- Optionally convert each series directory to a NIfTI volume (.nii.gz) for downstream tools.
- Identify likely T2w / ADC / DWI series and return a mapping.

Inputs (either):
- dicom_case_dir: folder containing series subfolders (preferred; enables conversion + header text)
- series_inventory_path: precomputed inventory (legacy mode)

Optional inputs:
- output_subdir (optional): artifacts subdir name (default: ingest)
- require_pydicom (optional): if true, fail when pydicom is unavailable
- convert_to_nifti (optional): if true, attempt SimpleITK DICOM->NIfTI conversion per series
- deep_dump (optional): if true (default), include recursive unrolled DICOM dump in header text output

Outputs:
- mapping + confidence
- series_inventory_path, dicom_meta_path
- series list + nifti_by_series
- dicom_headers_index_path + per-series header text files
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec
from .dicom_paths import first_dicom_instance_file, list_dicom_instance_files, sort_dicom_instance_files


def _require_simpleitk():
    try:
        import SimpleITK as sitk  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "SimpleITK is required for DICOM->NIfTI conversion. Install in this env:\n"
            "  pip install SimpleITK\n"
            f"Underlying error: {repr(e)}"
        ) from e
    return sitk


_SPACING_MISMATCH_MM = 0.5
_SPACING_MISMATCH_RATIO = 0.2


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _header_z_spacing(header: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(header, dict):
        return None
    for key in ("SpacingBetweenSlices", "SliceThickness"):
        v = _as_float(header.get(key))
        if v and v > 0:
            return float(v)
    return None


def _apply_spacing_policy(
    img,
    *,
    header: Optional[Dict[str, Any]],
    spacing_policy: str,
) -> Tuple[Any, Dict[str, Any]]:
    spacing_before = img.GetSpacing()
    header_z = _header_z_spacing(header)
    spacing_after = spacing_before
    spacing_source = "simpleitk"
    override_applied = False
    mismatch = False
    delta_mm: Optional[float] = None
    delta_pct: Optional[float] = None

    if header_z is not None and len(spacing_before) >= 3:
        dz = float(spacing_before[2])
        delta_mm = abs(header_z - dz)
        delta_pct = (delta_mm / dz) if dz != 0 else None
        mismatch = bool(
            delta_mm > _SPACING_MISMATCH_MM
            and (delta_pct is None or delta_pct > _SPACING_MISMATCH_RATIO)
        )
        if spacing_policy in ("header", "auto"):
            if spacing_policy == "header" or mismatch:
                spacing_after = (spacing_before[0], spacing_before[1], float(header_z))
                img.SetSpacing(spacing_after)
                spacing_source = "dicom_header"
                override_applied = True

    info = {
        "size": list(img.GetSize()),
        "spacing_before": list(spacing_before),
        "spacing_after": list(spacing_after),
        "origin": list(img.GetOrigin()),
        "direction": list(img.GetDirection()),
        "spacing_policy": spacing_policy,
        "spacing_source": spacing_source,
        "header_spacing_z_mm": header_z,
        "spacing_mismatch": mismatch,
        "spacing_override_applied": override_applied,
        "spacing_delta_z_mm": delta_mm,
        "spacing_delta_pct": delta_pct,
    }
    return img, info


def _dicom_series_to_nifti(
    series_dir: Path,
    out_path: Path,
    *,
    header: Optional[Dict[str, Any]] = None,
    spacing_policy: str = "auto",
) -> Tuple[Path, Dict[str, Any]]:
    """
    Convert a single DICOM series folder to NIfTI using SimpleITK.
    Supports nested/no-extension instance layouts by using the instance file list fallback.
    """
    sitk = _require_simpleitk()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reader = sitk.ImageSeriesReader()
    uids = reader.GetGDCMSeriesIDs(str(series_dir))
    if uids:
        files = reader.GetGDCMSeriesFileNames(str(series_dir), uids[0])
    else:
        files = [str(p) for p in sort_dicom_instance_files(list_dicom_instance_files(series_dir))]

    if not files:
        raise RuntimeError(f"No DICOM instances found for conversion in: {series_dir}")

    reader.SetFileNames(list(files))
    img = reader.Execute()
    img, nifti_info = _apply_spacing_policy(img, header=header, spacing_policy=spacing_policy)
    sitk.WriteImage(img, str(out_path))
    return out_path, nifti_info


def _dicom_files_to_nifti(
    files: List[str],
    out_path: Path,
    *,
    header: Optional[Dict[str, Any]] = None,
    spacing_policy: str = "auto",
) -> Tuple[Path, Dict[str, Any]]:
    """
    Convert a list of DICOM instance files to NIfTI using SimpleITK.
    """
    sitk = _require_simpleitk()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reader = sitk.ImageSeriesReader()
    if not files:
        raise RuntimeError("No DICOM files provided for conversion")
    files = [str(p) for p in sort_dicom_instance_files([Path(f) for f in files])]
    reader.SetFileNames(list(files))
    img = reader.Execute()
    img, nifti_info = _apply_spacing_policy(img, header=header, spacing_policy=spacing_policy)
    sitk.WriteImage(img, str(out_path))
    return out_path, nifti_info


def _list_series_dirs(dicom_case_dir: Path) -> List[Path]:
    if (not dicom_case_dir.exists()) or (not dicom_case_dir.is_dir()):
        return []
    direct = sorted([p for p in dicom_case_dir.iterdir() if p.is_dir()])
    if not direct:
        return []
    # Handle wrapper layouts like: <case>/DICOMS/<SERIES_NAME>/...
    # If there is a single top-level wrapper directory and its children look
    # like real series folders, descend one level for sequence discovery.
    if len(direct) == 1:
        wrapper = direct[0]
        nested = sorted([p for p in wrapper.iterdir() if p.is_dir()])
        if nested and any(_count_dicoms(p) > 0 for p in nested):
            return nested
    return direct


def _count_dicoms(series_dir: Path) -> int:
    return len(list_dicom_instance_files(series_dir))


def _guess_sequence_from_name(name: str) -> Tuple[str, float]:
    n = name.lower()
    if re.search(r"patient\d{3}(_4d|_frame\d+)", n):
        return "CINE", 0.9
    if "cine" in n or "bssfp" in n or "ssfp" in n:
        return "CINE", 0.95
    if re.search(r"(^|[^a-z0-9])t2([^a-z0-9]|$)", n):
        return "T2w", 0.9
    if re.search(r"(^|[^a-z0-9])adc([^a-z0-9]|$)", n):
        return "ADC", 0.9
    if "trace" in n or "dwi" in n:
        return "DWI", 0.7
    if "bval" in n:
        return "DWI_BVAL", 0.6
    return "UNKNOWN", 0.1


_BVAL_RE = re.compile(r"b(?:=|_|-)?(\d{2,5})", re.IGNORECASE)


def _infer_b_value_from_text(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    t = str(text).lower()
    m = _BVAL_RE.search(t)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    if "high" in t and "b" in t:
        return 2000.0
    if "mid" in t and "b" in t:
        return 800.0
    if "low" in t and "b" in t:
        return 50.0
    return None


def _list_nifti_files(case_dir: Path) -> List[Path]:
    out: List[Path] = []
    if not case_dir.exists():
        return out
    if case_dir.is_file():
        p = case_dir
        name = p.name.lower()
        if re.search(r"_gt(\.nii(\.gz)?)?$", name):
            return out
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            return [p]
        return out
    if not case_dir.is_dir():
        return out
    for p in sorted(case_dir.iterdir()):
        if not p.is_file():
            continue
        name = p.name.lower()
        # Exclude annotation/label volumes (e.g., ACDC *_gt.nii.gz) from sequence identification.
        if re.search(r"_gt(\.nii(\.gz)?)?$", name):
            continue
        if name.endswith(".nii") or name.endswith(".nii.gz"):
            out.append(p)
    return out


def _guess_sequence_from_nifti_name(name: str) -> Tuple[str, float]:
    n = name.lower()
    if re.search(r"patient\d{3}(_4d|_frame\d+)", n):
        return "CINE", 0.9
    if "cine" in n or "bssfp" in n or "ssfp" in n:
        return "CINE", 0.95
    if re.search(r"(^|[^a-z0-9])(t1ce|t1c|t1gd)([^a-z0-9]|$)", n):
        return "T1c", 0.95
    if "flair" in n:
        return "FLAIR", 0.95
    if re.search(r"(^|[^a-z0-9])t2([^a-z0-9]|$)", n):
        return "T2", 0.9
    if re.search(r"(^|[^a-z0-9])t1([^a-z0-9]|$)", n):
        return "T1", 0.8
    return "UNKNOWN", 0.1


def _build_nifti_inventory(
    *,
    nifti_case_dir: Path,
    out_dir: Path,
) -> Tuple[Path, Optional[Path], Optional[Path], List[Dict[str, Any]], Dict[str, Any], Dict[str, str], List[ArtifactRef], List[str]]:
    nifti_files = _list_nifti_files(nifti_case_dir)
    series: List[Dict[str, Any]] = []
    nifti_by_series: Dict[str, str] = {}
    artifacts: List[ArtifactRef] = []
    warnings: List[str] = []

    for p in nifti_files:
        label, conf = _guess_sequence_from_nifti_name(p.name)
        series_name = p.stem
        series.append(
            {
                "series_name": series_name,
                "series_dir": str(p),
                "n_dicoms": 1,
                "sequence_guess": label,
                "sequence_confidence": conf,
                "SeriesDescription": p.name,
                "nifti_path": str(p),
            }
        )
        nifti_by_series[series_name] = str(p)

    inv = {"series": series}
    series_inventory_path = out_dir / "series_inventory.json"
    series_inventory_path.write_text(json.dumps(inv, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    artifacts.append(ArtifactRef(path=str(series_inventory_path), kind="json", description="Series inventory (from NIfTI filenames)"))
    warnings.append("identify_sequences inferred mapping from NIfTI filenames; no DICOM headers available.")

    dicom_meta_path = out_dir / "dicom_meta.json"
    dicom_meta = {
        "note": "Synthetic DICOM meta generated from NIfTI-only directory.",
        "PatientID": nifti_case_dir.name,
        "StudyDate": "N/A",
        "series_meta": {},
    }
    dicom_meta_path.write_text(json.dumps(dicom_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    artifacts.append(ArtifactRef(path=str(dicom_meta_path), kind="json", description="Synthetic DICOM meta (NIfTI-only)"))

    header_index_path = out_dir / "dicom_headers_index.json"
    header_index = {
        "note": "Synthetic header index generated from NIfTI-only directory.",
        "series_headers": {},
    }
    header_index_path.write_text(json.dumps(header_index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    artifacts.append(ArtifactRef(path=str(header_index_path), kind="json", description="Synthetic DICOM header index (NIfTI-only)"))

    return (
        series_inventory_path,
        dicom_meta_path,
        header_index_path,
        series,
        {},
        nifti_by_series,
        artifacts,
        warnings,
    )


def _first_dicom(series_dir: Path) -> Optional[Path]:
    return first_dicom_instance_file(series_dir)


def _safe_get(ds: Any, key: str) -> Any:
    try:
        v = getattr(ds, key, None)
    except Exception:
        v = None
    return _to_jsonable_value(v)


def _to_jsonable_value(v: Any) -> Any:
    """
    Convert pydicom values (MultiValue, DSfloat, UID, PersonName, etc.) into JSON-serializable
    Python primitives (str/float/int/bool/list/dict/None).
    """
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (bytes, bytearray)):
        return None
    # pydicom value wrappers often have .value
    try:
        if hasattr(v, "value"):
            return _to_jsonable_value(v.value)
    except Exception:
        pass
    # MultiValue / sequences behave like list/tuple
    if isinstance(v, (list, tuple)):
        return [_to_jsonable_value(x) for x in v]
    # UID, PersonName, DSfloat, IS, etc. are safely stringifiable
    try:
        return str(v)
    except Exception:
        return None


def _read_dicom_header(path: Path) -> Dict[str, Any]:
    try:
        import pydicom  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pydicom_import_failed: {repr(e)}") from e

    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    # A curated minimal set of tags; expand later as needed.
    return {
        "StudyInstanceUID": _safe_get(ds, "StudyInstanceUID"),
        "SeriesInstanceUID": _safe_get(ds, "SeriesInstanceUID"),
        "SOPInstanceUID": _safe_get(ds, "SOPInstanceUID"),
        "Modality": _safe_get(ds, "Modality"),
        "Manufacturer": _safe_get(ds, "Manufacturer"),
        "ManufacturerModelName": _safe_get(ds, "ManufacturerModelName"),
        "MagneticFieldStrength": _safe_get(ds, "MagneticFieldStrength"),
        "SeriesDescription": _safe_get(ds, "SeriesDescription"),
        "ProtocolName": _safe_get(ds, "ProtocolName"),
        "SequenceName": _safe_get(ds, "SequenceName"),
        "ImageType": _safe_get(ds, "ImageType"),
        "ScanningSequence": _safe_get(ds, "ScanningSequence"),
        "SequenceVariant": _safe_get(ds, "SequenceVariant"),
        "ScanOptions": _safe_get(ds, "ScanOptions"),
        "BodyPartExamined": _safe_get(ds, "BodyPartExamined"),
        "PatientPosition": _safe_get(ds, "PatientPosition"),
        "Rows": _safe_get(ds, "Rows"),
        "Columns": _safe_get(ds, "Columns"),
        "PixelSpacing": _safe_get(ds, "PixelSpacing"),
        "SliceThickness": _safe_get(ds, "SliceThickness"),
        "SpacingBetweenSlices": _safe_get(ds, "SpacingBetweenSlices"),
        "RepetitionTime": _safe_get(ds, "RepetitionTime"),
        "EchoTime": _safe_get(ds, "EchoTime"),
        "FlipAngle": _safe_get(ds, "FlipAngle"),
        "DiffusionBValue": _safe_get(ds, "DiffusionBValue"),
        "ContrastBolusAgent": _safe_get(ds, "ContrastBolusAgent"),
        "ContrastBolusVolume": _safe_get(ds, "ContrastBolusVolume"),
        "ContrastBolusTotalDose": _safe_get(ds, "ContrastBolusTotalDose"),
        "ContrastBolusRoute": _safe_get(ds, "ContrastBolusRoute"),
    }


def _get_b_value(ds: Any) -> Optional[float]:
    """
    Best-effort extraction of diffusion b-value from DICOM dataset.
    Checks standard and nested sequence tags as well as common private tags.
    """
    try:
        bv = _to_jsonable_value(getattr(ds, "DiffusionBValue", None))
        if bv is not None:
            return float(bv)
    except Exception:
        pass

    # MR Diffusion Sequence (0018,9117) -> item -> Diffusion b-value (0018,9087)
    try:
        seq = ds.get((0x0018, 0x9117), None)
        if seq and getattr(seq, "value", None):
            item = seq.value[0]
            bv2 = _to_jsonable_value(getattr(item, "DiffusionBValue", None) or item.get((0x0018, 0x9087), None))
            if bv2 is not None:
                return float(bv2)
    except Exception:
        pass

    # Siemens private tag [B_value] (0019,100c)
    try:
        pv = ds.get((0x0019, 0x100C), None)
        if pv is not None:
            bv3 = _to_jsonable_value(getattr(pv, "value", None))
            if bv3 is not None:
                return float(bv3)
    except Exception:
        pass

    return None


def _scan_b_values(series_dir: Path) -> Dict[str, Any]:
    """
    Scan all instances in a series dir to summarize diffusion b-values.
    Uses pydicom header-only reads. Returns JSON-serializable dict.
    """
    try:
        import pydicom  # type: ignore
    except Exception as e:
        return {"ok": False, "error": f"pydicom_import_failed: {e}"}

    bvals: List[float] = []
    counts: Dict[str, int] = {}
    for p in list_dicom_instance_files(series_dir):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            bv = _get_b_value(ds)
            if bv is None:
                continue
            bvals.append(float(bv))
            key = str(int(bv)) if float(int(bv)) == float(bv) else str(bv)
            counts[key] = counts.get(key, 0) + 1
        except Exception:
            continue

    uniq = sorted(set(bvals))
    return {
        "ok": True,
        "b_values": uniq,
        "n_b_values": len(uniq),
        "counts_by_b": counts,
        "n_instances": len(list_dicom_instance_files(series_dir)),
    }


def _group_dwi_by_bvalue(series_dir: Path) -> Dict[float, List[str]]:
    """
    Group DWI instances by b-value using DICOM headers.
    Returns mapping: b_value -> list of file paths.
    """
    try:
        import pydicom  # type: ignore
    except Exception:
        return {}

    groups: Dict[float, List[str]] = {}
    for p in list_dicom_instance_files(series_dir):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            bv = _get_b_value(ds)
            if bv is None:
                continue
            b = float(bv)
            groups.setdefault(b, []).append(str(p))
        except Exception:
            continue
    return groups


def _iter_dicom_text_tags(ds: Any, *, max_value_len: int = 200) -> List[str]:
    lines: List[str] = []
    for elem in ds:
        try:
            tag = f"({elem.tag.group:04x},{elem.tag.elem:04x})"
            vr = str(getattr(elem, "VR", ""))
            name = str(getattr(elem, "keyword", "") or getattr(elem, "name", ""))
            raw_val = _to_jsonable_value(getattr(elem, "value", None))
            sval = str(raw_val)
            if len(sval) > max_value_len:
                sval = sval[: max_value_len - 3] + "..."
            lines.append(f"{tag}\t{vr}\t{name}\t{sval}")
        except Exception:
            continue
    return lines


_BULK_DATA_TAGS = {
    (0x7FE0, 0x0010),  # Pixel Data
    (0x7FE0, 0x0008),  # Float Pixel Data
    (0x7FE0, 0x0009),  # Double Float Pixel Data
    (0x5400, 0x1010),  # Waveform Data
    (0x5600, 0x0020),  # Spectroscopy Data
}

_BULK_KEYWORD_HINTS = (
    "PixelData",
    "FloatPixelData",
    "DoubleFloatPixelData",
    "WaveformData",
    "SpectroscopyData",
    "OverlayData",
    "EncapsulatedDocument",
)


def _is_bulk_data_element(elem: Any, *, raw_value: Any) -> bool:
    try:
        group = int(getattr(elem.tag, "group", -1))
        item = int(getattr(elem.tag, "elem", -1))
    except Exception:
        group, item = -1, -1

    if (group, item) in _BULK_DATA_TAGS:
        return True
    # Overlay Data family: 60xx,3000
    if item == 0x3000 and (group & 0xFF00) == 0x6000:
        return True

    keyword = str(getattr(elem, "keyword", "") or "")
    name = str(getattr(elem, "name", "") or "")
    probe = f"{keyword} {name}"
    if any(h.lower() in probe.lower() for h in _BULK_KEYWORD_HINTS):
        return True

    if isinstance(raw_value, (bytes, bytearray)):
        return True

    return False


def _value_preview_for_dump(value: Any, *, max_value_len: int = 500) -> str:
    try:
        v = _to_jsonable_value(value)
    except Exception:
        v = None

    if isinstance(v, (list, dict)):
        try:
            txt = json.dumps(v, ensure_ascii=False)
        except Exception:
            txt = str(v)
    else:
        txt = str(v)
    txt = txt.replace("\n", "\\n").replace("\r", "\\r")
    if len(txt) > max_value_len:
        txt = txt[: max_value_len - 3] + "..."
    return txt


def _iter_dicom_recursive_lines(
    ds: Any,
    *,
    depth: int = 0,
    max_depth: int = 8,
    max_value_len: int = 500,
) -> List[str]:
    lines: List[str] = []
    indent = "  " * max(0, int(depth))
    for elem in ds:
        try:
            tag = f"({elem.tag.group:04x},{elem.tag.elem:04x})"
            keyword = str(getattr(elem, "keyword", "") or getattr(elem, "name", "") or "Unknown")
            vr = str(getattr(elem, "VR", "") or "")
            raw_val = getattr(elem, "value", None)
            if _is_bulk_data_element(elem, raw_value=raw_val):
                lines.append(f"{indent}{tag} {keyword}: <omitted bulk data>")
                continue

            if vr == "SQ":
                seq_val = raw_val if isinstance(raw_val, (list, tuple)) else []
                n_items = len(seq_val)
                lines.append(f"{indent}{tag} {keyword}: <Sequence {n_items} item(s)>")
                if depth >= max_depth:
                    lines.append(f"{indent}  <max recursion depth reached>")
                    continue
                for i, item_ds in enumerate(seq_val):
                    lines.append(f"{indent}  Item[{i}]")
                    try:
                        lines.extend(
                            _iter_dicom_recursive_lines(
                                item_ds,
                                depth=depth + 2,
                                max_depth=max_depth,
                                max_value_len=max_value_len,
                            )
                        )
                    except Exception:
                        lines.append(f"{indent}    <unable to parse sequence item>")
                continue

            sval = _value_preview_for_dump(raw_val, max_value_len=max_value_len)
            lines.append(f"{indent}{tag} {keyword}: {sval}")
        except Exception:
            continue
    return lines


def _scan_te_tr_from_series(series_dir: Path, *, max_instances: int = 32) -> Tuple[Any, Any]:
    te = None
    tr = None
    try:
        import pydicom  # type: ignore
    except Exception:
        return te, tr

    files = list_dicom_instance_files(series_dir)
    for p in files[: max(1, int(max_instances))]:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if te is None:
            te = _to_jsonable_value(getattr(ds, "EchoTime", None))
        if tr is None:
            tr = _to_jsonable_value(getattr(ds, "RepetitionTime", None))
        if te is not None and tr is not None:
            break
    return te, tr


def _format_physical_param(value: Any) -> str:
    if value is None:
        return "NOT_FOUND"
    s = str(value).strip()
    return s if s else "NOT_FOUND"


def _extract_ascii_block(path: Path, *, start: str = "### ASCCONV BEGIN", end: str = "### ASCCONV END") -> Optional[str]:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    try:
        text = data.decode("latin1", errors="ignore")
    except Exception:
        return None
    i = text.find(start)
    if i < 0:
        return None
    j = text.find(end, i)
    if j < 0:
        return text[i:]
    return text[i : j + len(end)]


def _write_dicom_header_text(
    *,
    series_name: str,
    series_dir: Path,
    first_dicom: Optional[Path],
    out_dir: Path,
    header: Optional[Dict[str, Any]] = None,
    deep_dump: bool = True,
) -> Optional[Path]:
    if not first_dicom:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{series_name}_header.txt"
    read_error: Optional[str] = None
    tag_lines: List[str] = []
    recursive_lines: List[str] = []
    te = header.get("EchoTime") if isinstance(header, dict) else None
    tr = header.get("RepetitionTime") if isinstance(header, dict) else None
    try:
        import pydicom  # type: ignore

        ds = pydicom.dcmread(str(first_dicom), stop_before_pixels=True, force=True)
        tag_lines = _iter_dicom_text_tags(ds)
        if deep_dump:
            recursive_lines = _iter_dicom_recursive_lines(ds)
        if te is None:
            te = _to_jsonable_value(getattr(ds, "EchoTime", None))
        if tr is None:
            tr = _to_jsonable_value(getattr(ds, "RepetitionTime", None))
    except Exception as e:
        read_error = str(e)

    # Fallback scan across instances so TE/TR are present even if first header is missing fields.
    if te is None or tr is None:
        te_scan, tr_scan = _scan_te_tr_from_series(series_dir)
        if te is None:
            te = te_scan
        if tr is None:
            tr = tr_scan

    asconv = _extract_ascii_block(first_dicom)
    lines: List[str] = []
    lines.append(f"series_name: {series_name}")
    lines.append(f"series_dir: {series_dir}")
    lines.append(f"first_dicom: {first_dicom}")
    lines.append("")
    lines.append("=== KEY_PHYSICAL_PARAMETERS ===")
    lines.append(f"EchoTime: {_format_physical_param(te)}")
    lines.append(f"RepetitionTime: {_format_physical_param(tr)}")
    if read_error:
        lines.append(f"header_dump_error: {read_error}")
    lines.append("")
    lines.append("=== ASCCONV_BLOCK ===")
    lines.append(asconv if asconv else "(not found)")
    lines.append("")
    lines.append("=== DICOM_TAGS_TSV (tag, vr, keyword, value) ===")
    if tag_lines:
        lines.extend(tag_lines)
    else:
        lines.append("(no tags extracted)")
    lines.append("")
    lines.append("=== DICOM_RECURSIVE_DUMP (unrolled sequences; bulk data omitted) ===")
    if deep_dump:
        lines.extend(recursive_lines if recursive_lines else ["(no recursive tags extracted)"])
    else:
        lines.append("(deep_dump disabled)")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


IDENTIFY_SPEC = ToolSpec(
    name="identify_sequences",
    description=(
        "Identify likely T2w / ADC / DWI series. If `dicom_case_dir` is provided, this tool also "
        "builds `series_inventory.json`, `dicom_meta.json`, and per-series DICOM header text files, "
        "optionally converting series to NIfTI for downstream tools. "
        "Returns a `mapping` from sequence tokens to series directories plus confidence scores."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "dicom_case_dir": {"type": "string"},
            "series_inventory_path": {"type": "string"},
            "output_subdir": {"type": "string"},
            "require_pydicom": {"type": "boolean"},
            "convert_to_nifti": {"type": "boolean"},
            "spacing_policy": {"type": "string"},
            "deep_dump": {"type": "boolean"},
        },
        "required": [],
    },
    output_schema={
        "type": "object",
        "properties": {
            "mapping": {"type": "object"},
            "confidence": {"type": "object"},
            "series_inventory_path": {"type": "string"},
            "dicom_meta_path": {"type": "string"},
            "dicom_headers_index_path": {"type": "string"},
            "series": {"type": "array"},
            "nifti_by_series": {"type": "object"},
            "note": {"type": "string"},
        },
        "required": ["mapping"],
    },
    version="0.1.0",
    tags=["dicom", "mvp", "heuristic"],
)


def _build_series_inventory(
    *,
    dicom_case_dir: Path,
    out_dir: Path,
    require_pydicom: bool,
    convert_to_nifti: bool,
    spacing_policy: str,
    deep_dump: bool,
) -> Tuple[Path, Path, Optional[Path], List[Dict[str, Any]], Dict[str, Any], Dict[str, str], List[ArtifactRef], List[str]]:
    series_dirs = _list_series_dirs(dicom_case_dir)
    counts_by_series = {sd.name: _count_dicoms(sd) for sd in series_dirs}
    series: List[Dict[str, Any]] = []
    meta_by_series: Dict[str, Any] = {}
    pydicom_ok = True
    nifti_by_series: Dict[str, str] = {}
    header_txt_dir = out_dir / "dicom_txt"
    header_index_path = out_dir / "dicom_headers_index.txt"
    header_index_lines: List[str] = []
    spacing_warnings: List[str] = []

    for sd in series_dirs:
        label, conf = _guess_sequence_from_name(sd.name)
        first = _first_dicom(sd)
        header: Dict[str, Any]
        if first:
            try:
                header = _read_dicom_header(first)
            except Exception as e:
                pydicom_ok = False
                if require_pydicom:
                    raise RuntimeError(
                        "pydicom is required for real header parsing. Install in this env:\n"
                        "  pip install pydicom\n"
                        f"Underlying error: {e}"
                    ) from e
                header = {"note": "pydicom_not_installed_or_failed", "error": str(e)}
        else:
            header = {"note": "no_dicom_files"}

        b_summary = None
        if label in ("DWI", "ADC", "DWI_BVAL") or "diffusion" in sd.name.lower():
            b_summary = _scan_b_values(sd)

        nifti_path: Optional[str] = None
        nifti_error: Optional[str] = None
        nifti_by_bvalue: Dict[str, str] = {}
        nifti_info: Optional[Dict[str, Any]] = None
        if convert_to_nifti:
            try:
                out_nii = out_dir / "nifti" / f"{sd.name}.nii.gz"
                nifti_path_obj, nifti_info = _dicom_series_to_nifti(
                    sd,
                    out_nii,
                    header=header,
                    spacing_policy=spacing_policy,
                )
                nifti_path = str(nifti_path_obj)
                nifti_by_series[sd.name] = nifti_path
            except Exception as e:
                nifti_error = str(e)
                nifti_by_series[sd.name] = ""

            # If DWI, try to split by b-value and save per-b NIfTI.
            if label == "DWI":
                groups = _group_dwi_by_bvalue(sd)
                if groups:
                    for bval, files in groups.items():
                        try:
                            tag = str(int(bval)) if float(int(bval)) == float(bval) else str(bval)
                            key = f"{sd.name}_b{tag}"
                            out_b = out_dir / "nifti" / f"{key}.nii.gz"
                            _dicom_files_to_nifti(
                                files,
                                out_b,
                                header=header,
                                spacing_policy=spacing_policy,
                            )
                            nifti_by_series[key] = str(out_b)
                            nifti_by_bvalue[tag] = str(out_b)
                        except Exception:
                            continue
                else:
                    # Fallback heuristic: if DWI slice count is a multiple of a reference series, split evenly.
                    ref_counts = [c for n, c in counts_by_series.items() if n != sd.name]
                    ref_n = min(ref_counts) if ref_counts else 0
                    n_dicoms = counts_by_series.get(sd.name, 0)
                    if ref_n and n_dicoms % ref_n == 0:
                        ratio = int(n_dicoms // ref_n)
                        if ratio in (2, 3):
                            files = [str(p) for p in list_dicom_instance_files(sd)]
                            for i in range(ratio):
                                chunk = files[i * ref_n : (i + 1) * ref_n]
                                if not chunk:
                                    continue
                                key = f"{sd.name}_b{i}"
                                out_b = out_dir / "nifti" / f"{key}.nii.gz"
                                try:
                                    _dicom_files_to_nifti(
                                        chunk,
                                        out_b,
                                        header=header,
                                        spacing_policy=spacing_policy,
                                    )
                                    nifti_by_series[key] = str(out_b)
                                    nifti_by_bvalue[str(i)] = str(out_b)
                                except Exception:
                                    continue

        if nifti_info and nifti_info.get("spacing_mismatch"):
            header_z = nifti_info.get("header_spacing_z_mm")
            before = nifti_info.get("spacing_before") or []
            z_before = before[2] if isinstance(before, list) and len(before) > 2 else None
            spacing_warnings.append(
                "Spacing mismatch for series "
                f"'{sd.name}': header z={header_z} vs SimpleITK z={z_before}; "
                f"spacing_policy={spacing_policy}."
            )
        if spacing_policy in ("header", "auto") and nifti_info and nifti_info.get("header_spacing_z_mm") is None:
            spacing_warnings.append(
                f"Spacing policy '{spacing_policy}' requested but header z spacing is missing for series '{sd.name}'."
            )

        header_txt_path = _write_dicom_header_text(
            series_name=sd.name,
            series_dir=sd,
            first_dicom=first,
            out_dir=header_txt_dir,
            header=header,
            deep_dump=deep_dump,
        )
        if header_txt_path:
            header_index_lines.append(f"{sd.name}\t{header_txt_path}")

        meta_by_series[sd.name] = {
            "series_dir": str(sd),
            "first_dicom": str(first) if first else None,
            "header": header,
            "diffusion": b_summary,
            "nifti_path": nifti_path,
            "nifti_error": nifti_error,
            "nifti_by_bvalue": nifti_by_bvalue,
            "nifti_info": nifti_info,
            "dicom_header_txt": str(header_txt_path) if header_txt_path else None,
        }
        series.append(
            {
                "series_name": sd.name,
                "series_dir": str(sd),
                "n_dicoms": _count_dicoms(sd),
                "nifti_path": nifti_path,
                "sequence_guess": label,
                "sequence_confidence": conf,
                "SeriesInstanceUID": header.get("SeriesInstanceUID"),
                "SeriesDescription": header.get("SeriesDescription"),
                "Modality": header.get("Modality"),
                "b_values": (b_summary.get("b_values") if isinstance(b_summary, dict) else None),
                "counts_by_b": (b_summary.get("counts_by_b") if isinstance(b_summary, dict) else None),
                "dicom_header_txt": str(header_txt_path) if header_txt_path else None,
                "nifti_spacing": (nifti_info.get("spacing_after") if isinstance(nifti_info, dict) else None),
                "nifti_spacing_source": (nifti_info.get("spacing_source") if isinstance(nifti_info, dict) else None),
                "nifti_spacing_mismatch": (nifti_info.get("spacing_mismatch") if isinstance(nifti_info, dict) else None),
            }
        )

    series_inventory_path = out_dir / "series_inventory.json"
    series_inventory_path.write_text(json.dumps({"series": series}, indent=2) + "\n", encoding="utf-8")

    dicom_meta_path = out_dir / "dicom_meta.json"
    dicom_meta_path.write_text(
        json.dumps(
            {
                "dicom_case_dir": str(dicom_case_dir),
                "note": (
                    "Stage-1: parsed headers via pydicom (stop_before_pixels). "
                    + ("NIfTI conversion enabled." if convert_to_nifti else "NIfTI conversion disabled.")
                ),
                "series_meta": meta_by_series,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    header_index_path.write_text("\n".join(header_index_lines) + ("\n" if header_index_lines else ""), encoding="utf-8")

    artifacts = [
        ArtifactRef(path=str(series_inventory_path), kind="json", description="Series inventory"),
        ArtifactRef(path=str(dicom_meta_path), kind="json", description="Curated DICOM metadata (headers only)"),
        ArtifactRef(path=str(header_index_path), kind="text", description="Index of per-series DICOM header text files"),
    ]
    if convert_to_nifti:
        for name, p in nifti_by_series.items():
            if p:
                artifacts.append(
                    ArtifactRef(path=str(p), kind="nifti", description=f"NIfTI converted from DICOM series '{name}'")
                )

    warnings = (
        ([] if convert_to_nifti else ["NIfTI conversion disabled; set convert_to_nifti=true to write .nii.gz volumes."])
        + ([] if pydicom_ok else ["pydicom not available: headers were not parsed; install `pip install pydicom`"])
        + spacing_warnings
    )

    return (
        series_inventory_path,
        dicom_meta_path,
        header_index_path,
        series,
        meta_by_series,
        nifti_by_series,
        artifacts,
        warnings,
    )


def identify_sequences(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    dicom_case_dir_raw = args.get("dicom_case_dir")
    inv_path_raw = args.get("series_inventory_path")
    out_subdir = args.get("output_subdir", "ingest")
    require_pydicom = bool(args.get("require_pydicom", False))
    convert_to_nifti = bool(args.get("convert_to_nifti", True))
    deep_dump = bool(args.get("deep_dump", True))
    spacing_policy = str(args.get("spacing_policy", "auto")).lower()
    if spacing_policy not in ("auto", "header", "simpleitk"):
        spacing_policy = "auto"
    out_dir = ctx.artifacts_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    series: List[Dict[str, Any]] = []
    series_inventory_path: Optional[Path] = None
    dicom_meta_path: Optional[Path] = None
    header_index_path: Optional[Path] = None
    nifti_by_series: Dict[str, str] = {}
    artifacts: List[ArtifactRef] = []
    warnings: List[str] = []

    if dicom_case_dir_raw:
        dicom_case_dir = Path(str(dicom_case_dir_raw)).expanduser().resolve()
        if not dicom_case_dir.exists():
            raise FileNotFoundError(f"dicom_case_dir not found: {dicom_case_dir}")
        nifti_files = _list_nifti_files(dicom_case_dir)
        series_dirs = _list_series_dirs(dicom_case_dir)
        has_dicom = any(_count_dicoms(sd) > 0 for sd in series_dirs)
        if nifti_files and not has_dicom:
            (
                series_inventory_path,
                dicom_meta_path,
                header_index_path,
                series,
                _meta_by_series,
                nifti_by_series,
                artifacts,
                warnings,
            ) = _build_nifti_inventory(nifti_case_dir=dicom_case_dir, out_dir=out_dir)
        else:
            (
                series_inventory_path,
                dicom_meta_path,
                header_index_path,
                series,
                _meta_by_series,
                nifti_by_series,
                artifacts,
                warnings,
            ) = _build_series_inventory(
                dicom_case_dir=dicom_case_dir,
                out_dir=out_dir,
                require_pydicom=require_pydicom,
                convert_to_nifti=convert_to_nifti,
                spacing_policy=str(spacing_policy),
                deep_dump=deep_dump,
            )
    elif inv_path_raw:
        inv_path = Path(str(inv_path_raw)).expanduser().resolve()
        inv = json.loads(inv_path.read_text(encoding="utf-8"))
        series = inv.get("series", [])
        series_inventory_path = inv_path
        warnings = ["identify_sequences used series_inventory_path only; DICOM header text + NIfTI conversion not generated."]
    else:
        raise ValueError("identify_sequences requires dicom_case_dir or series_inventory_path")

    # Remove GT/label entries if present in precomputed inventories.
    def _is_gt_entry(item: Dict[str, Any]) -> bool:
        probe_vals = [
            item.get("series_name"),
            item.get("SeriesDescription"),
            item.get("nifti_path"),
            item.get("series_dir"),
        ]
        for v in probe_vals:
            txt = str(v or "").lower()
            if re.search(r"_gt(\.nii(\.gz)?)?$", txt):
                return True
        return False

    series = [s for s in series if isinstance(s, dict) and (not _is_gt_entry(s))]
    if isinstance(nifti_by_series, dict) and nifti_by_series:
        nifti_by_series = {
            k: v
            for k, v in nifti_by_series.items()
            if not re.search(r"_gt(\.nii(\.gz)?)?$", str(k).lower())
            and not re.search(r"_gt(\.nii(\.gz)?)?$", str(v or "").lower())
        }

    # Choose the highest-confidence matching series name for each sequence.
    best: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    for s in series:
        seq = s.get("sequence_guess", "UNKNOWN")
        conf = float(s.get("sequence_confidence", 0.0))
        if seq == "UNKNOWN":
            continue
        if seq not in best or conf > best[seq][0]:
            best[seq] = (conf, s)

    mapping: Dict[str, str] = {}
    confidence: Dict[str, float] = {}
    for seq, (conf, s) in best.items():
        # Prefer NIfTI outputs when available to avoid passing DICOM directories downstream.
        nifti_path = s.get("nifti_path")
        if not nifti_path:
            series_name = s.get("series_name")
            if series_name and series_name in nifti_by_series and nifti_by_series.get(series_name):
                nifti_path = nifti_by_series.get(series_name)
        mapping[seq] = nifti_path or s.get("series_dir") or s.get("series_name")
        confidence[seq] = conf

    # For DWI, prefer the highest b-value NIfTI if we generated per-b splits.
    try:
        dwi_series_names = [
            s.get("series_name")
            for s in series
            if isinstance(s, dict) and s.get("sequence_guess") == "DWI" and s.get("series_name")
        ]
        prefix_hints = [f"{str(name).lower()}_b" for name in dwi_series_names]
        best_b = None
        best_path = None
        for key, path in (nifti_by_series or {}).items():
            if not isinstance(path, str) or not path:
                continue
            kl = str(key).lower()
            if prefix_hints:
                if not any(kl.startswith(ph) for ph in prefix_hints):
                    if "dwi" not in kl and "trace" not in kl:
                        continue
            else:
                if "dwi" not in kl and "trace" not in kl:
                    continue
            b_hint = _infer_b_value_from_text(kl) or _infer_b_value_from_text(path)
            if b_hint is None:
                continue
            if best_b is None or b_hint > best_b:
                best_b = b_hint
                best_path = path
        if best_path:
            mapping["DWI"] = best_path
            confidence["DWI"] = max(confidence.get("DWI", 0.0), 0.7)
    except Exception:
        pass

    data = {
        "mapping": mapping,
        "confidence": confidence,
        "series_inventory_path": str(series_inventory_path) if series_inventory_path else None,
        "dicom_meta_path": str(dicom_meta_path) if dicom_meta_path else None,
        "dicom_headers_index_path": str(header_index_path) if header_index_path else None,
        "series": series,
        "nifti_by_series": nifti_by_series,
        "note": (
            "Stage-1 identify+ingest: headers parsed; NIfTI conversion enabled."
            if dicom_case_dir_raw and convert_to_nifti
            else "Stage-1 identify+ingest: headers parsed; NIfTI conversion disabled." if dicom_case_dir_raw else "Identify from inventory."
        ),
    }
    source_artifacts = []
    if dicom_case_dir_raw:
        _src = Path(str(dicom_case_dir_raw)).expanduser().resolve()
        source_artifacts.append(
            ArtifactRef(
                path=str(_src),
                kind=("file" if _src.is_file() else "dir"),
                description=("Input case file" if _src.is_file() else "Input DICOM case directory"),
            )
        )
    if series_inventory_path:
        source_artifacts.append(ArtifactRef(path=str(series_inventory_path), kind="json", description="Series inventory input"))

    return {
        "data": data,
        "artifacts": artifacts,
        "warnings": warnings + ["Heuristic mapping based on folder names; replace with DICOM-tag-based mapping later."],
        "source_artifacts": source_artifacts,
        "generated_artifacts": artifacts,
    }


def build_tools() -> List[Tool]:
    return [Tool(spec=IDENTIFY_SPEC, func=identify_sequences)]

from __future__ import annotations

"""
Utility helpers for dealing with different on-disk DICOM layouts.

Some datasets store instances as `*.dcm` in a flat series folder.
Others (e.g., some Siemens exports) store instance files as `I0000000` nested under
intermediate directories, alongside control files like DICOMDIR/LOCKFILE/VERSION.

These helpers provide a robust way to find instance files without assuming extensions.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple


_IGNORE_NAMES: Set[str] = {"DICOMDIR", "LOCKFILE", "VERSION"}
_IGNORE_SUFFIXES: Set[str] = {".nii", ".nii.gz", ".json", ".png", ".csv", ".txt"}


def list_dicom_instance_files(series_dir: Path) -> List[Path]:
    """
    Return a sorted list of candidate DICOM instance files under series_dir.
    This is intentionally permissive and does NOT validate that every file is a DICOM;
    downstream readers (pydicom/SimpleITK) will skip/raise on invalid ones.
    """
    series_dir = Path(series_dir)
    if not series_dir.exists():
        return []

    # 1) Common case: *.dcm anywhere under series_dir
    files = [p for p in series_dir.rglob("*.dcm") if p.is_file()]
    if files:
        return sorted(files)

    # 2) Common Siemens-ish case: "I*" instance files (nested)
    files = [p for p in series_dir.rglob("I*") if p.is_file() and p.name not in _IGNORE_NAMES]
    if files:
        return sorted(files)

    # 3) Fallback: any non-trivial file except known control files / obvious non-DICOM artifacts
    out: List[Path] = []
    for p in series_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name in _IGNORE_NAMES:
            continue
        s = p.name.lower()
        if any(s.endswith(suf) for suf in _IGNORE_SUFFIXES):
            continue
        try:
            if p.stat().st_size < 256:
                continue
        except Exception:
            continue
        out.append(p)
    return sorted(out)


def first_dicom_instance_file(series_dir: Path) -> Optional[Path]:
    files = list_dicom_instance_files(series_dir)
    return files[0] if files else None


def as_str_list(files: Iterable[Path]) -> List[str]:
    return [str(p) for p in files]


def sort_dicom_instance_files(files: Iterable[Path]) -> List[Path]:
    """
    Best-effort sort of DICOM instances by physical position (ImagePositionPatient),
    falling back to InstanceNumber, then filename.
    """
    paths = [Path(p) for p in files]
    try:
        import pydicom  # type: ignore
    except Exception:
        return sorted(paths)

    items: List[Tuple[Optional[float], Optional[int], str, Path]] = []
    has_proj = False
    has_inst = False
    for p in paths:
        proj = None
        inst = None
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            ipp = getattr(ds, "ImagePositionPatient", None)
            iop = getattr(ds, "ImageOrientationPatient", None)
            if ipp is not None:
                ipp = [float(x) for x in ipp]
                if iop is not None and len(iop) >= 6:
                    iop = [float(x) for x in iop]
                    row = iop[0:3]
                    col = iop[3:6]
                    n = [
                        row[1] * col[2] - row[2] * col[1],
                        row[2] * col[0] - row[0] * col[2],
                        row[0] * col[1] - row[1] * col[0],
                    ]
                    proj = float(ipp[0] * n[0] + ipp[1] * n[1] + ipp[2] * n[2])
                else:
                    proj = float(ipp[2])
            inst_v = getattr(ds, "InstanceNumber", None)
            if inst_v is not None:
                inst = int(inst_v)
        except Exception:
            pass
        if proj is not None:
            has_proj = True
        if inst is not None:
            has_inst = True
        items.append((proj, inst, p.name, p))

    if has_proj:
        items.sort(key=lambda x: (0, x[0]) if x[0] is not None else (1, x[1] if x[1] is not None else 1_000_000, x[2]))
    elif has_inst:
        items.sort(key=lambda x: (x[1] if x[1] is not None else 1_000_000, x[2]))
    else:
        items.sort(key=lambda x: x[2])

    return [p for _proj, _inst, _name, p in items]

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_DATA_EXTENSIONS = (".nii", ".nii.gz", ".dcm", ".h5", ".hdf5")


@dataclass(frozen=True)
class RootSpec:
    domain: str
    root: Path
    bucket: str
    split_h5_files: bool = False
    source_tag: str = ""


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object at: {path}")
    return obj


def _tasks_obj(path: Path) -> Dict[str, Any]:
    raw = _read_json(path)
    if isinstance(raw.get("tasks"), dict):
        return raw["tasks"]
    return raw


def _infer_domain_from_path(path: Path) -> str:
    low = str(path).lower()
    if any(k in low for k in ("prostate", "fastmri_prostate", "pirads")):
        return "prostate"
    if any(k in low for k in ("brats", "brain", "miccai")):
        return "brain"
    if any(k in low for k in ("acdc", "cardiac", "cine")):
        return "cardiac"
    return "prostate"


def _looks_like_dicom_file(path: Path) -> bool:
    if path.suffix.lower() == ".dcm":
        return True
    # Fallback for extension-less files with DICM signature.
    try:
        if path.suffix:
            return False
        if not path.is_file():
            return False
        with path.open("rb") as f:
            header = f.read(132)
        return len(header) >= 132 and header[128:132] == b"DICM"
    except Exception:
        return False


def _scan_case_files(case_root: Path, *, max_depth: int, max_files: int) -> List[Path]:
    root = case_root.resolve()
    if root.is_file():
        return [root]
    if not root.exists() or not root.is_dir():
        return []

    out: List[Path] = []
    root_depth = len(root.parts)

    for dirpath, dirnames, filenames in os.walk(root):
        cur = Path(dirpath)
        depth = len(cur.parts) - root_depth
        if depth >= max_depth:
            dirnames[:] = []

        for fn in filenames:
            fp = cur / fn
            out.append(fp)
            if len(out) >= max_files:
                return out
    return out


def _has_data_files(path: Path, *, max_depth: int, max_files: int) -> bool:
    for fp in _scan_case_files(path, max_depth=max_depth, max_files=max_files):
        name = fp.name.lower()
        if name.endswith(_DATA_EXTENSIONS):
            return True
        if _looks_like_dicom_file(fp):
            return True
    return False


def _discover_h5_case_files(root: Path, *, max_depth: int, max_files: int) -> List[Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        return []
    if root.is_file():
        low = root.name.lower()
        if low.endswith(".h5") or low.endswith(".hdf5"):
            return [root]
        return []

    out: List[Path] = []
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        cur = Path(dirpath)
        depth = len(cur.parts) - root_depth
        if depth >= max_depth:
            dirnames[:] = []
        for fn in filenames:
            low = fn.lower()
            if low.endswith(".h5") or low.endswith(".hdf5"):
                out.append((cur / fn).resolve())
                if len(out) >= max_files:
                    return out
    return sorted(out)


def _discover_case_dirs(root: Path, domain: str, *, max_depth: int, max_files: int) -> List[Path]:
    root = root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return []

    # BraTS layout: root/HGG|LGG/<case>
    if domain == "brain":
        grade_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name.upper() in {"HGG", "LGG"}]
        if grade_dirs:
            out: List[Path] = []
            for gd in grade_dirs:
                for c in sorted(gd.iterdir()):
                    if c.is_dir() and _has_data_files(c, max_depth=max_depth, max_files=max_files):
                        out.append(c.resolve())
            if out:
                return out

    out: List[Path] = []
    children = [d for d in sorted(root.iterdir()) if d.is_dir()]

    # ACDC raw layout: root/training|testing/patientXXX
    if domain == "cardiac":
        split_dirs = [d for d in children if d.name.lower() in {"training", "testing"}]
        if split_dirs:
            patient_cases: List[Path] = []
            for sd in split_dirs:
                for p in sorted(sd.iterdir()):
                    if not p.is_dir():
                        continue
                    if _has_data_files(p, max_depth=max_depth, max_files=max_files):
                        patient_cases.append(p.resolve())
            if patient_cases:
                return patient_cases

    for d in children:
        if _has_data_files(d, max_depth=max_depth, max_files=max_files):
            out.append(d.resolve())
    if out:
        return out

    # One more nesting level for uncommon layouts.
    nested: List[Path] = []
    for d in children:
        for c in sorted(d.iterdir()):
            if c.is_dir() and _has_data_files(c, max_depth=max_depth, max_files=max_files):
                nested.append(c.resolve())
    if nested:
        return nested

    # Single-case root.
    if _has_data_files(root, max_depth=max_depth, max_files=max_files):
        return [root]
    return []


def _discover_case_refs(root: Path, domain: str, *, split_h5_files: bool, max_depth: int, max_files: int) -> List[Path]:
    root = root.expanduser().resolve()
    if not root.exists():
        return []

    if split_h5_files:
        h5_cases = _discover_h5_case_files(root, max_depth=max_depth, max_files=max_files)
        if h5_cases:
            return h5_cases

    if root.is_file():
        return [root] if _has_data_files(root, max_depth=1, max_files=1) else []

    return _discover_case_dirs(root, domain, max_depth=max_depth, max_files=max_files)


def _infer_input_format(files: Iterable[Path]) -> str:
    has_nifti = False
    has_h5 = False
    has_dicom = False

    for p in files:
        low = p.name.lower()
        if low.endswith(".nii") or low.endswith(".nii.gz"):
            has_nifti = True
        elif low.endswith(".h5") or low.endswith(".hdf5"):
            has_h5 = True
        elif low.endswith(".dcm"):
            has_dicom = True
        elif _looks_like_dicom_file(p):
            has_dicom = True

    if has_h5 and not (has_nifti or has_dicom):
        return "raw_kspace"
    if has_nifti and not (has_h5 or has_dicom):
        return "nifti"
    if has_dicom and not (has_h5 or has_nifti):
        return "dicom"
    return "mixed"


def _is_high_b_text(text: str) -> bool:
    low = text.lower()
    if any(k in low for k in ("high_b", "high-b", "highb")):
        return True
    m = re.search(r"b(?:=|_|-)?(\d{3,5})", low)
    if not m:
        return False
    try:
        return int(m.group(1)) >= 800
    except Exception:
        return False


def _infer_prostate_modalities(files: Iterable[Path]) -> Dict[str, bool]:
    t2w = False
    adc = False
    dwi_highb = False

    for p in files:
        low = str(p).lower()
        if re.search(r"(^|[^a-z0-9])t2w?([^a-z0-9]|$)", low):
            t2w = True
        if "adc" in low or "diffusion_adc" in low:
            adc = True
        if any(k in low for k in ("dwi", "trace", "diffusion")) or _is_high_b_text(low):
            dwi_highb = True

    return {
        "T2w": t2w,
        "ADC": adc,
        "DWI_highb": dwi_highb,
    }


def _infer_brain_modalities(files: Iterable[Path]) -> Dict[str, bool]:
    t1 = False
    t1c = False
    t2 = False
    flair = False

    for p in files:
        low = str(p.name).lower()
        if any(k in low for k in ("t1ce", "t1c", "t1gd")):
            t1c = True
        if "flair" in low:
            flair = True
        if re.search(r"(^|[^a-z0-9])t2([^a-z0-9]|$)", low):
            t2 = True
        if re.search(r"(^|[^a-z0-9])t1([^a-z0-9]|$)", low) and not any(k in low for k in ("t1ce", "t1c", "t1gd")):
            t1 = True

    return {
        "T1": t1,
        "T1c": t1c,
        "T2": t2,
        "FLAIR": flair,
    }


def _infer_cardiac_modalities(files: Iterable[Path]) -> Dict[str, bool]:
    h5 = False
    cine_h5 = False
    cine_nifti = False

    for p in files:
        low = p.name.lower()
        if low.endswith(".h5") or low.endswith(".hdf5"):
            h5 = True
            if any(k in low for k in ("cine", "bssfp", "ssfp", "sax", "lax", "shortaxis", "short_axis")):
                cine_h5 = True
        if low.endswith(".nii") or low.endswith(".nii.gz"):
            if any(k in low for k in ("cine", "_4d", "frame", "bssfp", "ssfp", "sax")):
                cine_nifti = True

    # Cardiac datasets frequently use generic names without explicit "cine" tokens.
    if not cine_nifti:
        for p in files:
            low = p.name.lower()
            if low.endswith(".nii") or low.endswith(".nii.gz"):
                cine_nifti = True
                break

    return {
        "h5": h5,
        "cine_h5": cine_h5,
        "cine_nifti": cine_nifti,
        "raw_kspace": h5,
    }


def _infer_modalities(domain: str, files: Iterable[Path]) -> Dict[str, bool]:
    if domain == "prostate":
        return _infer_prostate_modalities(files)
    if domain == "brain":
        return _infer_brain_modalities(files)
    if domain == "cardiac":
        return _infer_cardiac_modalities(files)
    return {}


def _normalize_domain_field(raw: Any) -> List[str]:
    if isinstance(raw, str):
        s = raw.strip().lower()
        return [s] if s else []
    if isinstance(raw, list):
        out: List[str] = []
        for x in raw:
            s = str(x or "").strip().lower()
            if s:
                out.append(s)
        return out
    return []


def _eval_modal_rule(rule: Any, modalities: Dict[str, bool]) -> bool:
    if rule is None:
        return True
    if isinstance(rule, str):
        return bool(modalities.get(rule, False))
    if isinstance(rule, list):
        return all(bool(modalities.get(str(k), False)) for k in rule)
    if not isinstance(rule, dict):
        return True

    all_of = [str(x) for x in (rule.get("all_of") or [])]
    any_of = [str(x) for x in (rule.get("any_of") or [])]
    none_of = [str(x) for x in (rule.get("none_of") or [])]

    if all_of and not all(bool(modalities.get(k, False)) for k in all_of):
        return False
    if any_of and not any(bool(modalities.get(k, False)) for k in any_of):
        return False
    if none_of and any(bool(modalities.get(k, False)) for k in none_of):
        return False
    return True


def _task_supported(*, domain: str, modalities: Dict[str, bool], contract: Dict[str, Any]) -> bool:
    domains = _normalize_domain_field(contract.get("domain"))
    if domains and domain not in domains:
        return False

    req = contract.get("required_modalities")
    if isinstance(req, dict):
        # domain-specific rule takes precedence.
        if domain in req and isinstance(req.get(domain), (dict, list, str)):
            return _eval_modal_rule(req.get(domain), modalities)
        # otherwise interpret as direct rule object.
        if any(k in req for k in ("all_of", "any_of", "none_of")):
            return _eval_modal_rule(req, modalities)
        return True

    return _eval_modal_rule(req, modalities)


def _build_case_id(domain: str, case_root: Path, seen: set[str]) -> str:
    base = f"{domain}_{case_root.name}"
    base = re.sub(r"[^a-zA-Z0-9_.-]+", "_", base)
    if base not in seen:
        seen.add(base)
        return base
    suffix = hashlib.sha1(str(case_root).encode("utf-8")).hexdigest()[:8]
    cid = f"{base}_{suffix}"
    seen.add(cid)
    return cid


def _build_manifest_entry(
    *,
    case_root: Path,
    domain: str,
    files: List[Path],
    tasks: Dict[str, Any],
    seen_case_ids: set[str],
    note_hints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    modalities = _infer_modalities(domain, files)
    supports: List[str] = []
    for task_id in sorted(tasks.keys()):
        contract = tasks.get(task_id)
        if not isinstance(contract, dict):
            continue
        if _task_supported(domain=domain, modalities=modalities, contract=contract):
            supports.append(task_id)

    input_format = _infer_input_format(files)
    case_id = _build_case_id(domain, case_root, seen_case_ids)

    notes: List[str] = [str(x) for x in (note_hints or []) if str(x).strip()]
    if not files:
        notes.append("No data file found under scan depth limit.")
    if case_root.is_file():
        notes.append("Case root is a single file path.")
    if not supports:
        notes.append("No task contract matched required_modalities.")

    out: Dict[str, Any] = {
        "case_id": case_id,
        "domain": domain,
        "case_root": str(case_root.resolve()),
        "input_format": input_format,
        "modalities": modalities,
        "supports_tasks": supports,
    }
    if notes:
        out["notes"] = " ".join(notes)
    return out


def _bucket_limited(bucket: str, limits: Dict[str, int], counts: Dict[str, int]) -> bool:
    lim = int(limits.get(bucket, 0) or 0)
    if lim <= 0:
        return False
    return int(counts.get(bucket, 0) or 0) >= lim


def build_manifest(
    *,
    root_specs: List[RootSpec],
    tasks_registry_path: Path,
    max_depth: int,
    max_files_per_case: int,
    max_cases: int,
    bucket_limits: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    tasks = _tasks_obj(tasks_registry_path)
    entries: List[Dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    limits = dict(bucket_limits or {})
    bucket_counts: Dict[str, int] = {}

    for spec in root_specs:
        case_refs = _discover_case_refs(
            spec.root,
            spec.domain,
            split_h5_files=bool(spec.split_h5_files),
            max_depth=max_depth,
            max_files=max_files_per_case,
        )
        for case_ref in case_refs:
            if _bucket_limited(spec.bucket, limits, bucket_counts):
                continue
            files = _scan_case_files(case_ref, max_depth=max_depth, max_files=max_files_per_case)
            note_hints: List[str] = []
            if spec.source_tag:
                note_hints.append(f"source_group={spec.source_tag}")
            if spec.split_h5_files and case_ref.is_file():
                note_hints.append("split_h5_case=true")
            entry = _build_manifest_entry(
                case_root=case_ref,
                domain=spec.domain,
                files=files,
                tasks=tasks,
                seen_case_ids=seen_case_ids,
                note_hints=note_hints,
            )
            entries.append(entry)
            bucket_counts[spec.bucket] = int(bucket_counts.get(spec.bucket, 0) or 0) + 1
            if max_cases > 0 and len(entries) >= max_cases:
                return entries
    return entries


def _add_roots(
    root_specs: List[RootSpec],
    roots: List[str],
    *,
    domain: str,
    bucket: str,
    split_h5_files: bool,
    source_tag: str,
) -> None:
    for r in roots:
        s = str(r or "").strip()
        if not s:
            continue
        root_specs.append(
            RootSpec(
                domain=domain,
                root=Path(s).expanduser().resolve(),
                bucket=bucket,
                split_h5_files=split_h5_files,
                source_tag=source_tag,
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build cases_manifest.jsonl for benchmark v2.")
    ap.add_argument("--prostate-root", action="append", default=[], help="Prostate cases root (repeatable).")
    ap.add_argument("--brain-root", action="append", default=[], help="Brain cases root (repeatable).")
    ap.add_argument("--cardiac-root", action="append", default=[], help="Cardiac cases root (repeatable).")
    ap.add_argument(
        "--cardiac-raw-root",
        action="append",
        default=[],
        help="Cardiac raw-kspace root (repeatable). Each .h5/.hdf5 file is treated as one case.",
    )
    ap.add_argument("--case-root", action="append", default=[], help="Generic case root. Domain is inferred from path.")
    ap.add_argument("--tasks-registry", default="configs/tasks_registry.json", help="Path to tasks_registry.json")
    ap.add_argument("--output", default="cases_manifest.jsonl", help="Output manifest jsonl path.")
    ap.add_argument("--max-depth", type=int, default=4, help="Recursive scan depth per case root.")
    ap.add_argument("--max-files-per-case", type=int, default=4000, help="File scan cap per case.")
    ap.add_argument("--max-cases", type=int, default=0, help="Optional cap of total manifest cases (0 means no limit).")
    ap.add_argument("--max-prostate-cases", type=int, default=0, help="Optional cap for prostate bucket.")
    ap.add_argument("--max-brain-cases", type=int, default=0, help="Optional cap for brain bucket.")
    ap.add_argument("--max-cardiac-cases", type=int, default=0, help="Optional cap for cardiac bucket.")
    ap.add_argument("--max-cardiac-raw-cases", type=int, default=0, help="Optional cap for cardiac_raw bucket.")
    args = ap.parse_args()

    root_specs: List[RootSpec] = []
    _add_roots(
        root_specs,
        list(args.prostate_root or []),
        domain="prostate",
        bucket="prostate",
        split_h5_files=False,
        source_tag="prostate",
    )
    _add_roots(
        root_specs,
        list(args.brain_root or []),
        domain="brain",
        bucket="brain",
        split_h5_files=False,
        source_tag="brain",
    )
    _add_roots(
        root_specs,
        list(args.cardiac_root or []),
        domain="cardiac",
        bucket="cardiac",
        split_h5_files=True,
        source_tag="cardiac",
    )
    _add_roots(
        root_specs,
        list(args.cardiac_raw_root or []),
        domain="cardiac",
        bucket="cardiac_raw",
        split_h5_files=True,
        source_tag="cardiac_raw",
    )

    for raw in list(args.case_root or []):
        p = Path(str(raw)).expanduser().resolve()
        dom = _infer_domain_from_path(p)
        root_specs.append(
            RootSpec(
                domain=dom,
                root=p,
                bucket=dom,
                split_h5_files=(dom == "cardiac"),
                source_tag="generic",
            )
        )

    if not root_specs:
        raise SystemExit(
            "No case roots provided. Use --prostate-root/--brain-root/--cardiac-root/--cardiac-raw-root/--case-root."
        )

    tasks_registry_path = Path(str(args.tasks_registry)).expanduser().resolve()
    bucket_limits = {
        "prostate": max(0, int(args.max_prostate_cases)),
        "brain": max(0, int(args.max_brain_cases)),
        "cardiac": max(0, int(args.max_cardiac_cases)),
        "cardiac_raw": max(0, int(args.max_cardiac_raw_cases)),
    }
    entries = build_manifest(
        root_specs=root_specs,
        tasks_registry_path=tasks_registry_path,
        max_depth=max(1, int(args.max_depth)),
        max_files_per_case=max(64, int(args.max_files_per_case)),
        max_cases=max(0, int(args.max_cases)),
        bucket_limits=bucket_limits,
    )

    out_path = Path(str(args.output)).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in entries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    domain_counts: Dict[str, int] = {}
    for row in entries:
        dom = str(row.get("domain") or "").strip().lower()
        if dom:
            domain_counts[dom] = int(domain_counts.get(dom, 0) or 0) + 1
    print(f"[OK] wrote {len(entries)} case records -> {out_path}")
    if domain_counts:
        print("[INFO] counts_by_domain:", json.dumps(domain_counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

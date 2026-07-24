"""
Tool: segment_cardiac_cine

Purpose:
- Run cardiac cine short-axis segmentation using the vendored, Apache-2.0
  physics-augmented nnUNet (V1) backend under ``external/nnunet_phys_seg`` and
  nnUNet model weights fetched separately under ``assets/``.

Inputs:
- cine_path: NIfTI file (.nii.gz/.nii) or directory containing cine NIfTI(s)
- backend_root (optional): vendored nnUNet backend root (default
  ``{PROJECT_ROOT}/external/nnunet_phys_seg``). The legacy ``cmr_reverse_root``
  key is still honored for back-compat.
- nnunet_python (optional): Python executable with the nnUNet backend installed
  (default: MRI_AGENT_CARDIAC_SEG_PYTHON or the current interpreter)
- results_folder (optional): nnUNet RESULTS_FOLDER path (default
  ``{ASSETS_ROOT}/models/cardiac_nnunet/results``)
- output_subdir (optional)
- task_name / trainer_class_name / model / folds / checkpoint (optional overrides)

Outputs:
- seg_path: primary segmentation path
- rv_mask_path / myo_mask_path / lv_mask_path: primary class masks
- case_results: per-case outputs
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec
from core.paths import project_root

CARDIAC_SEG_SPEC = ToolSpec(
    name="segment_cardiac_cine",
    description=(
        "Segment cardiac cine MRI using the vendored physics-augmented nnUNet "
        "backend and separately-fetched weights "
        "(default Task900_ACDC_Phys, trainer nnUNetTrainerV2_InvGreAug; both overridable). "
        "Produces multi-class segmentation and class-specific masks for RV/MYO/LV."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "cine_path": {"type": "string"},
            "acdc_info_cfg": {"type": "string"},
            "use_ed_es_only": {"type": "boolean"},
            "backend_root": {"type": "string"},
            "cmr_reverse_root": {"type": "string"},
            "nnunet_python": {"type": "string"},
            "results_folder": {"type": "string"},
            "output_subdir": {"type": "string"},
            "task_name": {"type": "string"},
            "trainer_class_name": {"type": "string"},
            "model": {"type": "string"},
            "folds": {"type": "array", "items": {"type": "integer"}},
            "checkpoint": {"type": "string"},
            "disable_tta": {"type": "boolean"},
            "num_threads_preprocessing": {"type": "integer"},
            "num_threads_nifti_save": {"type": "integer"},
        },
        "required": ["cine_path"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "seg_path": {"type": "string"},
            "rv_mask_path": {"type": "string"},
            "myo_mask_path": {"type": "string"},
            "lv_mask_path": {"type": "string"},
            "case_results": {"type": "array"},
            "note": {"type": "string"},
        },
        "required": ["seg_path"],
    },
    version="0.1.0",
    tags=["segmentation", "cardiac", "cine", "nnunet"],
)


def _require_sitk():
    try:
        import SimpleITK as sitk  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "SimpleITK is required for cardiac cine segmentation preprocessing/postprocessing. "
            "Install via: pip install SimpleITK"
        ) from e
    return sitk


def _repo_root() -> Path:
    env = os.getenv("BCER_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return project_root()


def _assets_root() -> Path:
    env = os.getenv("BCER_ASSETS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return (_repo_root() / "assets").resolve()


def _vendored_backend_src() -> Path:
    """Vendored, Apache-2.0 nnUNet backend package parent (importable as ``nnunet``)."""
    return (_repo_root() / "external" / "nnunet_phys_seg").resolve()


def _default_backend_root() -> Path:
    # Neutral var wins; legacy MRI_AGENT_CARDIAC_CMR_REVERSE_ROOT still honored.
    env = os.getenv("MRI_AGENT_CARDIAC_BACKEND_ROOT") or os.getenv(
        "MRI_AGENT_CARDIAC_CMR_REVERSE_ROOT"
    )
    if env:
        return Path(env).expanduser().resolve()
    return _vendored_backend_src()


def _default_seg_python() -> Path:
    env = os.getenv("MRI_AGENT_CARDIAC_SEG_PYTHON") or os.getenv(
        "MRI_AGENT_CARDIAC_NNUNET_PYTHON"
    )
    if env:
        return Path(os.path.abspath(os.path.expanduser(env)))
    return Path(sys.executable)


def _default_results_folder() -> Path:
    env = os.getenv("MRI_AGENT_CARDIAC_RESULTS_FOLDER")
    if env:
        return Path(env).expanduser().resolve()
    return (_assets_root() / "models" / "cardiac_nnunet" / "results").resolve()


def _stem_nii(path: Path) -> str:
    n = path.name
    if n.endswith(".nii.gz"):
        return n[:-7]
    if n.endswith(".nii"):
        return n[:-4]
    return path.stem


def _safe_case_id(raw: str) -> str:
    s = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(raw))
    s = s.strip("_")
    return s or "case"


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        return
    except Exception:
        pass
    shutil.copy2(str(src), str(dst))


def _ensure_nii_gz(src: Path, dst_nii_gz: Path) -> None:
    if str(src).lower().endswith(".nii.gz"):
        _link_or_copy(src, dst_nii_gz)
        return
    sitk = _require_sitk()
    img = sitk.ReadImage(str(src))
    sitk.WriteImage(img, str(dst_nii_gz))


def _parse_acdc_ed_es(info_cfg: Path) -> Tuple[Optional[int], Optional[int]]:
    if not info_cfg.exists() or not info_cfg.is_file():
        return None, None
    ed: Optional[int] = None
    es: Optional[int] = None
    try:
        for line in info_cfg.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            key = k.strip().lower()
            val = v.strip()
            if key == "ed":
                ed = int(val)
            elif key == "es":
                es = int(val)
    except Exception:
        return None, None
    return ed, es


def _prepare_single_nifti_case(
    src: Path,
    case_id_base: str,
    input_dir: Path,
    *,
    use_ed_es_only: bool = True,
    info_cfg_path: Optional[Path] = None,
) -> List[Tuple[str, Path]]:
    """
    Prepare one NIfTI path for nnUNet input.
    - 3D -> one case: <case_id>_0000.nii.gz
    - 4D cine -> split into per-frame 3D cases: <case_id>_fXX_0000.nii.gz
    """
    sitk = _require_sitk()
    img = sitk.ReadImage(str(src))
    dim = int(img.GetDimension())

    if dim == 4:
        size = img.GetSize()
        n_frames = int(size[3]) if len(size) >= 4 else 0
        if n_frames <= 0:
            raise RuntimeError(f"4D cine has no temporal frames: {src}")

        selected_frames: List[int] = list(range(n_frames))
        frame_alias: Dict[int, str] = {}
        if use_ed_es_only:
            cfg = info_cfg_path if info_cfg_path is not None else (src.parent / "Info.cfg")
            ed, es = _parse_acdc_ed_es(cfg)
            picked: List[int] = []
            if isinstance(ed, int) and 1 <= ed <= n_frames:
                picked.append(ed - 1)
                frame_alias[ed - 1] = "ed"
            if isinstance(es, int) and 1 <= es <= n_frames and (es - 1) not in picked:
                picked.append(es - 1)
                frame_alias[es - 1] = "es"
            if picked:
                selected_frames = picked

        out: List[Tuple[str, Path]] = []
        for t in selected_frames:
            frame = img[:, :, :, t]
            suffix = frame_alias.get(t, f"f{t + 1:02d}")
            cid = f"{case_id_base}_{suffix}"
            dst = input_dir / f"{cid}_0000.nii.gz"
            sitk.WriteImage(frame, str(dst))
            out.append((cid, dst))
        return out

    if dim in (2, 3):
        dst = input_dir / f"{case_id_base}_0000.nii.gz"
        _ensure_nii_gz(src, dst)
        return [(case_id_base, dst)]

    raise RuntimeError(f"Unsupported cine dimensionality ({dim}D): {src}")


def _prepare_cases(
    cine_path: Path,
    input_dir: Path,
    *,
    use_ed_es_only: bool = True,
    acdc_info_cfg: Optional[Path] = None,
) -> List[Tuple[str, Path]]:
    """
    Prepare nnUNet input files in the form: CASE_0000.nii.gz.
    Returns list of (case_id, prepared_path).
    """
    sitk = _require_sitk()
    input_dir.mkdir(parents=True, exist_ok=True)
    cases: List[Tuple[str, Path]] = []
    used: set[str] = set()

    def _unique_case_id(base: str) -> str:
        cid = _safe_case_id(base)
        if cid not in used:
            used.add(cid)
            return cid
        idx = 1
        while f"{cid}_{idx}" in used:
            idx += 1
        out = f"{cid}_{idx}"
        used.add(out)
        return out

    if cine_path.is_file():
        cid = _unique_case_id(_stem_nii(cine_path))
        prepared = _prepare_single_nifti_case(
            cine_path,
            cid,
            input_dir,
            use_ed_es_only=bool(use_ed_es_only),
            info_cfg_path=acdc_info_cfg,
        )
        cases.extend(prepared)
        return cases

    if not cine_path.is_dir():
        raise FileNotFoundError(f"cine_path not found: {cine_path}")

    nifti_files = sorted(
        [p for p in cine_path.iterdir() if p.is_file() and (p.name.lower().endswith(".nii.gz") or p.name.lower().endswith(".nii"))]
    )
    if nifti_files:
        for p in nifti_files:
            cid = _unique_case_id(_stem_nii(p))
            prepared = _prepare_single_nifti_case(
                p,
                cid,
                input_dir,
                use_ed_es_only=bool(use_ed_es_only),
                info_cfg_path=acdc_info_cfg,
            )
            cases.extend(prepared)
        return cases

    # Fallback: treat input as a DICOM series directory and convert one volume.
    reader = sitk.ImageSeriesReader()
    uids = reader.GetGDCMSeriesIDs(str(cine_path))
    if not uids:
        raise RuntimeError(
            f"No NIfTI files or DICOM series found under {cine_path}. "
            "Pass a cine NIfTI path or run identify_sequences first."
        )
    files = reader.GetGDCMSeriesFileNames(str(cine_path), uids[0])
    if not files:
        raise RuntimeError(f"DICOM series was found but no readable instances in {cine_path}")
    reader.SetFileNames(list(files))
    img = reader.Execute()
    cid = _unique_case_id(_safe_case_id(cine_path.name))
    dst = input_dir / f"{cid}_0000.nii.gz"
    sitk.WriteImage(img, str(dst))
    cases.append((cid, dst))
    return cases


def _run_nnunet_predict(
    *,
    nnunet_python: Path,
    backend_root: Path,
    results_folder: Path,
    input_dir: Path,
    pred_dir: Path,
    task_name: str,
    trainer_class_name: str,
    model: str,
    folds: List[int],
    checkpoint: str,
    disable_tta: bool,
    num_threads_preprocessing: int,
    num_threads_nifti_save: int,
) -> Tuple[str, str, List[str]]:
    # Inference only needs RESULTS_FOLDER (the trained weights). nnUNet V1
    # paths.py merely warns when raw/preprocessed are unset, so those dirs are
    # optional here; we pass them through only when they actually exist (e.g.
    # when backend_root points at a full training tree).
    raw_base = backend_root / "nnunetdata" / "raw"
    preprocessed = backend_root / "nnunetdata" / "preprocessed"
    # An existing-but-empty results dir is the common case after a fresh clone
    # (assets/models/cardiac_nnunet/results is created by download_assets.sh
    # before the weights land), and it would otherwise surface as an opaque
    # nnUNet "model output folder not found" assertion deep in the child.
    if not results_folder.exists() or not any(results_folder.iterdir()):
        raise FileNotFoundError(
            "Missing or empty nnUNet RESULTS_FOLDER (download the cardiac weights "
            "first -- see docs/ASSETS.md): "
            f"{results_folder}"
        )
    if not nnunet_python.exists():
        raise FileNotFoundError(f"nnunet_python not found: {nnunet_python}")

    cmd = [
        str(nnunet_python),
        "-m",
        "nnunet.inference.predict_simple",
        "-i",
        str(input_dir),
        "-o",
        str(pred_dir),
        "-t",
        str(task_name),
        "-tr",
        str(trainer_class_name),
        "-m",
        str(model),
        "-f",
    ]
    cmd.extend([str(int(x)) for x in folds])
    cmd.extend(
        [
            "-chk",
            str(checkpoint),
            "--overwrite_existing",
            "--num_threads_preprocessing",
            str(int(num_threads_preprocessing)),
            "--num_threads_nifti_save",
            str(int(num_threads_nifti_save)),
        ]
    )
    if disable_tta:
        cmd.append("--disable_tta")

    env = os.environ.copy()
    # Inference requires only RESULTS_FOLDER; raw/preprocessed are attached only
    # when present so paths.py does not create empty dirs inside the source tree.
    if raw_base.exists():
        env["nnUNet_raw_data_base"] = str(raw_base)
    if preprocessed.exists():
        env["nnUNet_preprocessed"] = str(preprocessed)
    env["RESULTS_FOLDER"] = str(results_folder)
    # Import the vendored, Apache-2.0 nnUNet backend from external/nnunet_phys_seg.
    nnunet_src = _vendored_backend_src()
    run_cwd = backend_root if backend_root.exists() else _repo_root()
    if nnunet_src.exists():
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{nnunet_src}:{prev}" if prev else str(nnunet_src)
        # Provenance guard. Under `python -m`, sys.path[0] is the child's cwd,
        # which therefore beats the PYTHONPATH entry above. backend_root is
        # meant to be overridable to a full nnUNet *training* tree (we read
        # nnunetdata/{raw,preprocessed} from it), and such a tree contains its
        # own nnunet/ package -- which would then silently shadow the vendored
        # fork. Run from the vendored dir so the code that executes is always
        # the one we ship; raw/preprocessed still come from backend_root.
        run_cwd = nnunet_src
        # A populated ~/.local/lib/pythonX.Y/site-packages sits ahead of the
        # interpreter's own site-packages and overrides env pins (e.g. the
        # numpy<2 that envs/cardiac.yml exists to enforce).
        env["PYTHONNOUSERSITE"] = "1"

    proc = subprocess.run(
        cmd,
        cwd=str(run_cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stdout_tail = "\n".join((proc.stdout or "").splitlines()[-120:])
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-120:])
        raise RuntimeError(
            "nnUNet cardiac cine segmentation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Return code: {proc.returncode}\n"
            f"STDOUT (tail):\n{stdout_tail}\n"
            f"STDERR (tail):\n{stderr_tail}"
        )
    return proc.stdout or "", proc.stderr or "", cmd


def _normalize_nnunet_task_and_model(task_name: str, model: str) -> Tuple[str, str]:
    task_raw = str(task_name or "").strip()
    model_raw = str(model or "").strip()
    model_allowed = {"2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"}

    # Handle common task/model swap from free-form LLM output.
    if model_raw.lower().startswith("task") and (not task_raw.lower().startswith("task")):
        task_raw = model_raw
        model_raw = ""

    if (not task_raw) or task_raw.lower() in {"acdc", "acdc_phys", "task900", "task900_acdc", "900"}:
        task_norm = "Task900_ACDC_Phys"
    elif task_raw.isdigit():
        tid = int(task_raw)
        task_norm = "Task900_ACDC_Phys" if tid == 900 else f"Task{tid:03d}"
    elif task_raw.lower().startswith("task"):
        task_norm = task_raw
    else:
        task_norm = "Task900_ACDC_Phys"

    model_norm = model_raw.lower()
    if model_norm not in model_allowed:
        model_norm = "2d"
    return task_norm, model_norm


def _split_cardiac_labels(pred_path: Path, out_dir: Path) -> Dict[str, str]:
    """
    ACDC label convention:
    - 1: RV blood pool
    - 2: Myocardium
    - 3: LV blood pool
    """
    sitk = _require_sitk()
    seg_img = sitk.ReadImage(str(pred_path))
    case_id = _stem_nii(pred_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    rv = sitk.Cast(seg_img == 1, sitk.sitkUInt8)
    myo = sitk.Cast(seg_img == 2, sitk.sitkUInt8)
    lv = sitk.Cast(seg_img == 3, sitk.sitkUInt8)

    rv_path = out_dir / f"{case_id}_rv_mask.nii.gz"
    myo_path = out_dir / f"{case_id}_myo_mask.nii.gz"
    lv_path = out_dir / f"{case_id}_lv_mask.nii.gz"
    sitk.WriteImage(rv, str(rv_path))
    sitk.WriteImage(myo, str(myo_path))
    sitk.WriteImage(lv, str(lv_path))
    return {
        "rv_mask_path": str(rv_path),
        "myo_mask_path": str(myo_path),
        "lv_mask_path": str(lv_path),
    }


def run_cardiac_cine_segmentation(
    *,
    cine_path: Path,
    output_dir: Path,
    backend_root: Path,
    nnunet_python: Path,
    results_folder: Path,
    task_name: str = "Task900_ACDC_Phys",
    trainer_class_name: str = "nnUNetTrainerV2_InvGreAug",
    model: str = "2d",
    folds: Optional[List[int]] = None,
    checkpoint: str = "model_final_checkpoint",
    disable_tta: bool = True,
    num_threads_preprocessing: int = 2,
    num_threads_nifti_save: int = 2,
    use_ed_es_only: bool = True,
    acdc_info_cfg: Optional[Path] = None,
) -> Dict[str, Any]:
    folds_use = [0, 1, 2, 3, 4] if not folds else [int(x) for x in folds]
    output_dir.mkdir(parents=True, exist_ok=True)

    io_dir = output_dir / "nnunet_io"
    input_dir = io_dir / "input"
    pred_dir = io_dir / "pred"
    masks_dir = output_dir / "masks"
    if input_dir.exists():
        shutil.rmtree(input_dir, ignore_errors=True)
    if pred_dir.exists():
        shutil.rmtree(pred_dir, ignore_errors=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    prepared_cases = _prepare_cases(
        cine_path,
        input_dir,
        use_ed_es_only=bool(use_ed_es_only),
        acdc_info_cfg=acdc_info_cfg,
    )
    if not prepared_cases:
        raise RuntimeError("No valid cine case prepared for nnUNet input.")

    task_name_norm, model_norm = _normalize_nnunet_task_and_model(task_name, model)

    stdout, stderr, cmd = _run_nnunet_predict(
        nnunet_python=nnunet_python,
        backend_root=backend_root,
        results_folder=results_folder,
        input_dir=input_dir,
        pred_dir=pred_dir,
        task_name=task_name_norm,
        trainer_class_name=trainer_class_name,
        model=model_norm,
        folds=folds_use,
        checkpoint=checkpoint,
        disable_tta=bool(disable_tta),
        num_threads_preprocessing=int(num_threads_preprocessing),
        num_threads_nifti_save=int(num_threads_nifti_save),
    )

    log_path = output_dir / "nnunet_predict.log"
    log_text = "\n".join(
        [
            f"CMD: {' '.join(cmd)}",
            "",
            "STDOUT:",
            stdout,
            "",
            "STDERR:",
            stderr,
        ]
    )
    log_path.write_text(log_text, encoding="utf-8")

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        raise RuntimeError(f"nnUNet finished but no prediction files found in {pred_dir}")

    case_results: List[Dict[str, Any]] = []
    for p in pred_files:
        split = _split_cardiac_labels(p, masks_dir)
        case_results.append(
            {
                "case_id": _stem_nii(p),
                "seg_path": str(p),
                **split,
            }
        )

    primary = case_results[0]
    return {
        "seg_path": primary["seg_path"],
        "rv_mask_path": primary["rv_mask_path"],
        "myo_mask_path": primary["myo_mask_path"],
        "lv_mask_path": primary["lv_mask_path"],
        "case_results": case_results,
        "log_path": str(log_path),
        "input_dir": str(input_dir),
        "pred_dir": str(pred_dir),
        "backend_root": str(backend_root),
        "results_folder": str(results_folder),
        "task_name": str(task_name_norm),
        "trainer_class_name": str(trainer_class_name),
        "model": str(model_norm),
        "folds": folds_use,
        "checkpoint": str(checkpoint),
        "note": "Cardiac cine segmentation labels use ACDC convention: 1=RV, 2=MYO, 3=LV.",
        "use_ed_es_only": bool(use_ed_es_only),
    }


def segment_cardiac_cine(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    cine_path = Path(str(args["cine_path"])).expanduser().resolve()
    # backend_root is the neutral key; cmr_reverse_root stays for back-compat.
    backend_root_arg = args.get("backend_root") or args.get("cmr_reverse_root")
    backend_root = Path(str(backend_root_arg or _default_backend_root())).expanduser().resolve()
    nnunet_python = Path(
        os.path.abspath(os.path.expanduser(str(args.get("nnunet_python"))))
    ) if args.get("nnunet_python") else _default_seg_python()
    acdc_info_cfg_raw = str(args.get("acdc_info_cfg") or "").strip()
    acdc_info_cfg = Path(acdc_info_cfg_raw).expanduser().resolve() if acdc_info_cfg_raw else None
    use_ed_es_only = bool(args.get("use_ed_es_only", True))
    results_folder = Path(str(args.get("results_folder") or _default_results_folder())).expanduser().resolve()
    out_subdir = str(args.get("output_subdir") or "segmentation/cardiac_cine")
    out_dir = ctx.artifacts_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_cardiac_cine_segmentation(
        cine_path=cine_path,
        output_dir=out_dir,
        backend_root=backend_root,
        nnunet_python=nnunet_python,
        results_folder=results_folder,
        task_name=str(
            args.get("task_name")
            or os.getenv("MRI_AGENT_CARDIAC_TASK")
            or "Task900_ACDC_Phys"
        ),
        trainer_class_name=str(
            args.get("trainer_class_name")
            or os.getenv("MRI_AGENT_CARDIAC_TRAINER")
            or "nnUNetTrainerV2_InvGreAug"
        ),
        model=str(args.get("model") or "2d"),
        folds=list(args.get("folds") or [0, 1, 2, 3, 4]),
        checkpoint=str(args.get("checkpoint") or "model_final_checkpoint"),
        disable_tta=bool(args.get("disable_tta", True)),
        num_threads_preprocessing=int(args.get("num_threads_preprocessing", 2)),
        num_threads_nifti_save=int(args.get("num_threads_nifti_save", 2)),
        use_ed_es_only=use_ed_es_only,
        acdc_info_cfg=acdc_info_cfg,
    )

    artifacts: List[ArtifactRef] = []
    for item in result.get("case_results", []) or []:
        if not isinstance(item, dict):
            continue
        if item.get("seg_path"):
            artifacts.append(ArtifactRef(path=str(item["seg_path"]), kind="nifti", description=f"Cardiac segmentation ({item.get('case_id')})"))
        for k, desc in (
            ("rv_mask_path", "RV mask"),
            ("myo_mask_path", "Myocardium mask"),
            ("lv_mask_path", "LV mask"),
        ):
            if item.get(k):
                artifacts.append(ArtifactRef(path=str(item[k]), kind="nifti", description=f"{desc} ({item.get('case_id')})"))
    if result.get("log_path"):
        artifacts.append(ArtifactRef(path=str(result["log_path"]), kind="text", description="nnUNet prediction log"))

    return {
        "data": result,
        "artifacts": artifacts,
        "warnings": [],
    }


def build_tool() -> Tool:
    return Tool(spec=CARDIAC_SEG_SPEC, func=segment_cardiac_cine)

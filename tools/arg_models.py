from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator

from core.domain_config import DomainConfig, get_domain_config
from core.paths import project_root


# repair_tool_args() logic classes (for ablation clarity):
# A) type/schema repair: coercion, defaults, schema normalization (allowed in noToken)
# B) semantic binding repair: sequence-name->path resolve / prior-node artifact autowire
#    (suppressed in noToken via suppress_sequence_resolve / suppress_node_output_autowire)


def _latest_tool_data(state_path: Path, stage: str, tool_name: str) -> Dict[str, Any]:
    """
    Return the latest stored `data` for a tool.

    NOTE:
    Historically, stages were named like "identify"/"segment"/... but some plans
    use numeric stage keys. If the requested stage is not present we fall back to
    searching ALL stages and returning the latest record.
    """

    def _stage_sort_key(k: str):
        try:
            return (0, int(k), "")
        except Exception:
            return (1, 10**9, str(k))

    try:
        s = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    all_stage_outputs: Dict[str, Any] = s.get("stage_outputs", {}) or {}

    stage_outputs = (all_stage_outputs.get(stage, {}) or {}) if isinstance(all_stage_outputs, dict) else {}
    if isinstance(stage_outputs, dict):
        records = stage_outputs.get(tool_name, []) or []
        if records:
            rec = records[-1] or {}
            return rec.get("data", {}) or {}

    if not isinstance(all_stage_outputs, dict):
        return {}
    latest: Dict[str, Any] = {}
    for st in sorted(all_stage_outputs.keys(), key=_stage_sort_key):
        tools = all_stage_outputs.get(st, {}) or {}
        if not isinstance(tools, dict):
            continue
        records = tools.get(tool_name, []) or []
        if not isinstance(records, list) or not records:
            continue
        rec = records[-1] or {}
        latest = rec.get("data", {}) or {}
    return latest or {}


@dataclass(frozen=True)
class SequenceMappingConfig:
    mapping: Dict[str, str]
    domain: DomainConfig
    suppress: bool = False

    def resolve_token(self, token: Any) -> Optional[str]:
        if self.suppress:
            return None
        return self.domain.resolve_token(token, self.mapping)


@dataclass(frozen=True)
class RepairContext:
    tool_name: str
    state_path: Path
    ctx_case_state_path: Path
    dicom_case_dir: Optional[str]
    mapping: Dict[str, str]
    sequence_mapping: SequenceMappingConfig
    run_artifacts_dir: Path
    segment_data: Dict[str, Any]
    config: "RepairConfig"
    domain: DomainConfig
    suppress_sequence_resolve: bool = False
    suppress_node_output_autowire: bool = False

    # B-class semantic binding repair helper:
    # sequence_mapping.resolve_token("ADC"/"T2w") -> concrete NIfTI path.
    def resolve_sequence_token(self, token: Any) -> Optional[str]:
        return self.sequence_mapping.resolve_token(token)

    # B-class semantic binding repair helper:
    # reads prior node outputs from case_state.json (disable for noToken).
    def latest_tool_data(self, stage: str, tool_name: str) -> Dict[str, Any]:
        if self.suppress_node_output_autowire:
            return {}
        return _latest_tool_data(self.state_path, stage, tool_name)


@dataclass(frozen=True)
class RepairConfig:
    model_registry_path: Path
    prostate_bundle_dir: Path
    lesion_weights_dir: Path
    brain_bundle_dir: Path
    cardiac_cmr_reverse_root: Path
    cardiac_nnunet_python: Path
    cardiac_results_folder: Path
    distortion_repo_root: Path
    distortion_diff_ckpt: Path
    distortion_cnn_ckpt: Path
    distortion_test_roots: List[str]

    @staticmethod
    def from_env(repo_root: Path) -> "RepairConfig":
        model_registry_default = repo_root / "models"
        model_registry = Path(os.getenv("MRI_AGENT_MODEL_REGISTRY", str(model_registry_default))).expanduser().resolve()

        prostate_bundle = Path(
            os.getenv("MRI_AGENT_PROSTATE_BUNDLE_DIR", str(model_registry / "prostate_mri_anatomy"))
        ).expanduser().resolve()

        lesion_default = repo_root / "external" / "prostate_mri_lesion_seg" / "weight"
        lesion_weights = Path(os.getenv("MRI_AGENT_LESION_WEIGHTS_DIR", str(lesion_default))).expanduser().resolve()

        brain_bundle = Path(
            os.getenv(
                "MRI_AGENT_BRATS_BUNDLE_DIR",
                str(model_registry / "brats_mri_segmentation"),
            )
        ).expanduser().resolve()
        cardiac_root_default = repo_root / "external" / "cmr_reverse"
        cardiac_root = Path(
            os.getenv("MRI_AGENT_CARDIAC_CMR_REVERSE_ROOT", str(cardiac_root_default))
        ).expanduser().resolve()
        # Keep symlink path as-is (do not resolve), otherwise a venv shim like
        # revimg/bin/python -> python3.11 can collapse to /usr/bin/python3.11.
        cardiac_nnunet_python = Path(
            os.path.abspath(
                os.path.expanduser(
                    os.getenv(
                        "MRI_AGENT_CARDIAC_NNUNET_PYTHON",
                        str(cardiac_root / "revimg" / "bin" / "python"),
                    )
                )
            )
        )
        cardiac_results = Path(
            os.getenv(
                "MRI_AGENT_CARDIAC_RESULTS_FOLDER",
                str(cardiac_root / "nnunetdata" / "results"),
            )
        ).expanduser().resolve()
        distortion_root_default = repo_root / "external" / "Prostate_distortion_recover"
        distortion_root = Path(
            os.getenv("MRI_AGENT_PROSTATE_DISTORTION_ROOT", str(distortion_root_default))
        ).expanduser().resolve()
        distortion_diff_ckpt = Path(
            os.getenv(
                "MRI_AGENT_PROSTATE_DISTORTION_DIFF_CKPT",
                str(
                    distortion_root
                    / "network_runs"
                    / "diffusion_clean_v2"
                    / "diff_t2cnn_clean_epoch_092.pt"
                ),
            )
        ).expanduser().resolve()
        distortion_cnn_ckpt = Path(
            os.getenv(
                "MRI_AGENT_PROSTATE_DISTORTION_CNN_CKPT",
                str(
                    distortion_root
                    / "network_runs"
                    / "mageultra_dualb_axis01_quick_local_v25"
                    / "mageultra_epoch_025.pt"
                ),
            )
        ).expanduser().resolve()
        test_roots_env = str(os.getenv("MRI_AGENT_PROSTATE_DISTORTION_TEST_ROOTS", "")).strip()
        distortion_test_roots: List[str] = []
        if test_roots_env:
            parts = [x.strip() for x in re.split(r"[,:;]", test_roots_env) if x.strip()]
            for p in parts:
                distortion_test_roots.append(str(Path(p).expanduser().resolve()))
        if not distortion_test_roots:
            distortion_test_roots = [
                str((distortion_root / "validation_data2_TSE").expanduser().resolve()),
                str((distortion_root / "preprocessed_infer_npz").expanduser().resolve()),
            ]
        return RepairConfig(
            model_registry_path=model_registry,
            prostate_bundle_dir=prostate_bundle,
            lesion_weights_dir=lesion_weights,
            brain_bundle_dir=brain_bundle,
            cardiac_cmr_reverse_root=cardiac_root,
            cardiac_nnunet_python=cardiac_nnunet_python,
            cardiac_results_folder=cardiac_results,
            distortion_repo_root=distortion_root,
            distortion_diff_ckpt=distortion_diff_ckpt,
            distortion_cnn_ckpt=distortion_cnn_ckpt,
            distortion_test_roots=distortion_test_roots,
        )


@dataclass(frozen=True)
class ArtifactLocator:
    run_artifacts_dir: Path
    segment_data: Dict[str, Any]

    def t2w_nifti_path(self) -> Optional[Path]:
        try:
            zone_p = self.segment_data.get("zone_mask_path") if isinstance(self.segment_data, dict) else None
            if isinstance(zone_p, str) and Path(zone_p).exists():
                cand = Path(zone_p).expanduser().resolve().with_name("t2w_input.nii.gz")
                if cand.exists():
                    return cand
        except Exception:
            pass
        cand = self.run_artifacts_dir / "segmentation" / "t2w_input.nii.gz"
        return cand if cand.exists() else None

    def list_intensity_niftis(self) -> List[Path]:
        nifti_paths: List[Path] = []
        for p in sorted(self.run_artifacts_dir.glob("**/*.nii.gz")):
            s = p.name.lower()
            if (
                "mask" in s
                or "segmentation" in s
                or "tumor" in s
                or "subregion" in s
                or "label" in s
                or s.endswith("_zones.nii.gz")
                or s.endswith("t2w_input_zones.nii.gz")
            ):
                continue
            nifti_paths.append(p)
        return nifti_paths

    def _pick_by_hint(self, nifti_paths: List[Path], hint: str) -> Optional[str]:
        h = hint.lower()
        for p in nifti_paths:
            s = str(p).lower()
            if h in s:
                return str(p)
        return None

    def pick_adc_nifti(self, nifti_paths: List[Path]) -> Optional[str]:
        for p in nifti_paths:
            s = str(p).lower().replace("\\", "/")
            n = p.name.lower()
            if "moving_adc_resampled_to_fixed" in n:
                return str(p)
            if "/registration/adc/" in s and "resampled_to_fixed" in n and ("high_b" not in n and "low_b" not in n and "mid_b" not in n):
                return str(p)
        for p in nifti_paths:
            n = p.name.lower()
            if "adc" in n and ("high_b" not in n and "low_b" not in n and "mid_b" not in n):
                return str(p)
        for p in nifti_paths:
            if p.name.lower() == "moving_all_resampled_to_fixed.nii.gz":
                return str(p)
        return None

    def pick_highb_nifti(self, nifti_paths: List[Path]) -> Optional[str]:
        # Prefer explicit DWI registration outputs first.
        for p in nifti_paths:
            s = str(p).lower().replace("\\", "/")
            n = p.name.lower()
            if "/registration/dwi/" in s and "moving_high_b_resampled_to_fixed" in n:
                return str(p)
        for p in nifti_paths:
            s = str(p).lower().replace("\\", "/")
            n = p.name.lower()
            if "/registration/dwi/" in s and ("moving_dwi_resampled_to_fixed" in n or "moving_mid_b_resampled_to_fixed" in n):
                return str(p)
        # Fallback to global matching, but avoid ADC registration directory.
        for p in nifti_paths:
            s = str(p).lower().replace("\\", "/")
            n = p.name.lower()
            if "/registration/adc/" in s:
                continue
            if "moving_high_b_resampled_to_fixed" in n:
                return str(p)
        for p in nifti_paths:
            s = str(p).lower().replace("\\", "/")
            n = p.name.lower()
            if "/registration/adc/" in s:
                continue
            if "moving_dwi_resampled_to_fixed" in n:
                return str(p)
        for p in nifti_paths:
            s = str(p).lower().replace("\\", "/")
            n = p.name.lower()
            if "/registration/adc/" in s:
                continue
            if "high_b" in n or "b1400" in n:
                return str(p)
        return None


def _build_context(
    tool_name: str,
    state_path: Path,
    ctx_case_state_path: Path,
    dicom_case_dir: Optional[str],
    domain: Optional[DomainConfig],
    suppress_sequence_resolve: bool = False,
    suppress_node_output_autowire: bool = False,
) -> RepairContext:
    repo_root = project_root()
    domain_cfg = domain or get_domain_config("prostate")
    config = RepairConfig.from_env(repo_root)
    ident = _latest_tool_data(state_path, "identify", "identify_sequences") or {}
    raw_mapping = ident.get("mapping", {}) if isinstance(ident, dict) else {}
    if not isinstance(raw_mapping, dict):
        raw_mapping = {}
    # B-class semantic binding repair: sequence-name->path mapping lookups.
    # noToken keeps type/schema repair but suppresses this mapping at repair-time.
    mapping = {} if suppress_sequence_resolve else raw_mapping

    seg: Dict[str, Any] = {}
    if not suppress_node_output_autowire:
        seg = _latest_tool_data(state_path, "segment", "segment_prostate") or {}
        if not isinstance(seg, dict):
            seg = {}
        if domain_cfg.name == "brain":
            brain_seg = _latest_tool_data(state_path, "segment", "brats_mri_segmentation") or {}
            if isinstance(brain_seg, dict) and brain_seg:
                seg = brain_seg
        elif not seg and domain_cfg.name != "cardiac":
            brain_seg = _latest_tool_data(state_path, "segment", "brats_mri_segmentation") or {}
            if isinstance(brain_seg, dict) and brain_seg:
                seg = brain_seg
        if domain_cfg.name == "cardiac":
            cardiac_seg = _latest_tool_data(state_path, "segment", "segment_cardiac_cine") or {}
            if isinstance(cardiac_seg, dict) and cardiac_seg:
                seg = cardiac_seg

    run_art_dir = Path(ctx_case_state_path).parent / "artifacts"
    if suppress_node_output_autowire:
        # B-class semantic binding repair: artifact scans / prior-node output autowire.
        # Point to a guaranteed-missing subdir so ArtifactLocator/glob fallbacks no-op.
        run_art_dir = run_art_dir / "__repair_node_output_autowire_suppressed__"
    return RepairContext(
        tool_name=tool_name,
        state_path=state_path,
        ctx_case_state_path=ctx_case_state_path,
        dicom_case_dir=dicom_case_dir,
        mapping=mapping,
        sequence_mapping=SequenceMappingConfig(mapping=mapping, domain=domain_cfg, suppress=suppress_sequence_resolve),
        run_artifacts_dir=run_art_dir,
        segment_data=seg,
        config=config,
        domain=domain_cfg,
        suppress_sequence_resolve=suppress_sequence_resolve,
        suppress_node_output_autowire=suppress_node_output_autowire,
    )


def _ctx(info: ValidationInfo) -> Optional[RepairContext]:
    if not info or not info.context:
        return None
    ctx = info.context.get("ctx")
    if isinstance(ctx, RepairContext):
        return ctx
    return None


def _normalize_output_subdir(data: Dict[str, Any]) -> Dict[str, Any]:
    osub = data.get("output_subdir")
    if isinstance(osub, str) and osub:
        raw = osub.replace("\\", "/")
        if "/artifacts/" in raw:
            raw = raw.rsplit("/artifacts/", 1)[-1]
        s = raw.lstrip("./")
        if s.startswith("artifacts/"):
            s = s[len("artifacts/") :]
        s = s.lstrip("/").replace("..", "")
        if s:
            data["output_subdir"] = s
        else:
            data.pop("output_subdir", None)
    return data


def _normalize_images_list(val: Any) -> List[Dict[str, str]]:
    out_imgs: List[Dict[str, str]] = []
    if isinstance(val, dict):
        val = [val]
    elif isinstance(val, str):
        val = [{"name": Path(val).stem or "img0", "path": val}]
    if not isinstance(val, list):
        return out_imgs
    for i, it in enumerate(val):
        if isinstance(it, str):
            out_imgs.append({"name": Path(it).stem or f"img{i}", "path": it})
        elif isinstance(it, dict):
            out_imgs.append({"name": str(it.get("name", f"img{i}")), "path": str(it.get("path", ""))})
    return out_imgs


def _normalize_roi_masks(val: Any) -> List[Dict[str, str]]:
    out_masks: List[Dict[str, str]] = []
    if isinstance(val, dict):
        val = [val]
    elif isinstance(val, str):
        val = [val]
    if not isinstance(val, list):
        return out_masks
    for i, it in enumerate(val):
        if isinstance(it, dict):
            path = it.get("path") or it.get("mask_path") or it.get("roi_mask_path")
            if not path:
                continue
            name = it.get("name") or Path(str(path)).stem or f"roi{i}"
            out_masks.append({"name": str(name), "path": str(path)})
        elif isinstance(it, str):
            name = Path(it).stem or f"roi{i}"
            out_masks.append({"name": str(name), "path": str(it)})
    return out_masks


class ToolArgsBase(BaseModel):
    model_config = ConfigDict(extra="allow")
    output_subdir: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_subdir(cls, data: Any, info: ValidationInfo) -> Any:
        if isinstance(data, dict):
            return _normalize_output_subdir(dict(data))
        return data


class GenericToolArgs(ToolArgsBase):
    pass


class IngestDicomArgs(ToolArgsBase):
    dicom_case_dir: Optional[str] = None
    output_subdir: Optional[str] = None
    require_pydicom: Optional[bool] = None
    convert_to_nifti: Optional[bool] = None
    deep_dump: Optional[bool] = None

    @model_validator(mode="after")
    def _defaults(self, info: ValidationInfo) -> "IngestDicomArgs":
        ctx = _ctx(info)
        fields_set = self.model_fields_set
        if ctx and ctx.dicom_case_dir and "dicom_case_dir" not in fields_set:
            self.dicom_case_dir = ctx.dicom_case_dir
        if "output_subdir" not in fields_set:
            self.output_subdir = "ingest"
        if "require_pydicom" not in fields_set:
            # Keep identify robust in lightweight benchmark envs where pydicom may
            # be absent; explicit plans can still force require_pydicom=True.
            self.require_pydicom = False
        if "convert_to_nifti" not in fields_set:
            self.convert_to_nifti = True
        if "deep_dump" not in fields_set:
            self.deep_dump = True
        return self


class RegisterToReferenceArgs(ToolArgsBase):
    moving: Optional[str] = None
    fixed: Optional[str] = None
    output_subdir: Optional[str] = None
    split_by_bvalue: Optional[bool] = None
    save_png_qc: Optional[bool] = None

    @model_validator(mode="after")
    def _repair(self, info: ValidationInfo) -> "RegisterToReferenceArgs":
        ctx = _ctx(info)
        if not ctx:
            return self

        # B-class semantic binding repair: sequence token -> path.
        fixed_res = ctx.resolve_sequence_token(self.fixed) if self.fixed is not None else None
        moving_res = ctx.resolve_sequence_token(self.moving) if self.moving is not None else None
        if fixed_res and moving_res:
            try:
                if ctx.domain.should_swap_registration(str(Path(fixed_res).name), str(Path(moving_res).name)):
                    fixed_res, moving_res = moving_res, fixed_res
            except Exception:
                pass
            try:
                if Path(fixed_res).expanduser().resolve() == Path(moving_res).expanduser().resolve():
                    for cand_key in ("ADC", "DWI"):
                        cand = ctx.mapping.get(cand_key) if isinstance(ctx.mapping, dict) else None
                        if isinstance(cand, str) and Path(cand).exists():
                            moving_res = cand
                            break
            except Exception:
                pass

        if fixed_res:
            self.fixed = fixed_res
        if moving_res:
            self.moving = moving_res

        try:
            mv = str(self.moving or "")
            mv_l = mv.lower()
            osub = self.output_subdir
            if isinstance(osub, str):
                osub_l = osub.lower().replace("\\", "/").lstrip("./")
            else:
                osub_l = ""
            canonical_subdir = ctx.domain.registration_subdir(mv) if mv else None
            if (not osub_l) or (osub_l in ("register", "registration")):
                if canonical_subdir:
                    self.output_subdir = canonical_subdir
                else:
                    self.output_subdir = f"registration/{Path(str(self.moving or 'moving')).name}"
            elif canonical_subdir and osub_l.startswith("registration/") and osub_l != canonical_subdir.lower():
                # Prevent inconsistent subdirs like DWI data written into registration/ADC.
                self.output_subdir = canonical_subdir
        except Exception:
            if not self.output_subdir:
                self.output_subdir = f"registration/{Path(str(self.moving or 'moving')).name}"

        fields_set = self.model_fields_set
        mv_l = str(self.moving or "").lower().replace("\\", "/")
        mv_name = Path(str(self.moving or "")).name.lower()
        is_adc_like = ("adc" in mv_name) or ("/adc." in mv_l) or ("/registration/adc/" in mv_l)
        if is_adc_like:
            # Force ADC registration to remain single-volume. Splitting ADC creates fake low/mid/high-b outputs
            # and can trigger repeated register loops in lesion preconditions.
            self.split_by_bvalue = False
        elif "split_by_bvalue" not in fields_set:
            self.split_by_bvalue = True
        if "save_png_qc" not in fields_set:
            self.save_png_qc = True
        return self


class SegmentProstateArgs(ToolArgsBase):
    t2w_ref: Optional[str] = None
    bundle_dir: Optional[str] = None
    device: Optional[str] = None
    output_subdir: Optional[str] = None
    spacing_policy: Optional[str] = None

    @model_validator(mode="after")
    def _repair(self, info: ValidationInfo) -> "SegmentProstateArgs":
        ctx = _ctx(info)
        if not ctx:
            return self

        def _is_functional_path(txt: Any) -> bool:
            low = str(txt or "").lower().replace("\\", "/")
            return any(
                k in low
                for k in (
                    "adc",
                    "dwi",
                    "diff",
                    "high_b",
                    "high-b",
                    "low_b",
                    "low-b",
                    "mid_b",
                    "mid-b",
                    "b1400",
                    "b800",
                    "b50",
                    "/registration/dwi/",
                    "/registration/adc/",
                )
            )

        def _pick_ingested_t2w_path() -> Optional[str]:
            ing_dir = ctx.run_artifacts_dir / "ingested" / "nifti"
            if ing_dir.exists():
                for p in sorted(ing_dir.glob("*.nii*")):
                    nm = p.name.lower()
                    if ("t2" in nm) and (not _is_functional_path(nm)):
                        return str(p)
            ident = ctx.latest_tool_data("ingest", "identify_sequences") or ctx.latest_tool_data(
                "identify", "identify_sequences"
            )
            series = ident.get("series", []) if isinstance(ident, dict) else []
            if isinstance(series, list):
                for s in series:
                    if not isinstance(s, dict):
                        continue
                    nifti_path = str(s.get("nifti_path") or "").strip()
                    if not nifti_path:
                        continue
                    guess = str(s.get("sequence_guess") or "").lower()
                    name = str(s.get("series_name") or "").lower()
                    if (
                        guess == "t2w"
                        or ("t2" in name)
                        or ("t2" in Path(nifti_path).name.lower())
                    ) and (not _is_functional_path(nifti_path)):
                        return nifti_path
            return None

        t2w_ref = self.t2w_ref
        # B-class semantic binding repair: sequence token -> path.
        t2w_res = ctx.resolve_sequence_token(t2w_ref) if t2w_ref is not None else None
        canonical_t2w = ctx.domain.structural_path(ctx.mapping) if isinstance(ctx.mapping, dict) else None
        fallback_t2w = canonical_t2w or _pick_ingested_t2w_path()
        try:
            sref = str(t2w_ref or "").lower()
            pref = Path(str(t2w_ref)).expanduser().resolve() if t2w_ref is not None else None
            looks_wrong_modality = any(k in sref for k in ctx.domain.functional_keywords) or _is_functional_path(sref)
            looks_file = bool(pref and pref.exists() and pref.is_file())
            if fallback_t2w and (looks_wrong_modality or looks_file or not t2w_res):
                self.t2w_ref = fallback_t2w
            elif t2w_res:
                self.t2w_ref = t2w_res
        except Exception:
            if fallback_t2w:
                self.t2w_ref = fallback_t2w

        default_bundle = ctx.config.prostate_bundle_dir
        bd_raw = self.bundle_dir
        bd_str = str(bd_raw) if bd_raw is not None else ""
        bd_path = Path(bd_str).expanduser().resolve() if bd_str else default_bundle.resolve()
        looks_placeholder = ("/path/to/" in bd_str) or (bd_str.strip().lower() in ("", "none"))
        has_model = (bd_path / "models" / "model.ts").exists() or (bd_path / "models" / "model.pt").exists()
        if looks_placeholder or (not bd_path.exists()) or (not has_model):
            self.bundle_dir = str(default_bundle.resolve())
        else:
            self.bundle_dir = str(bd_path)

        if "device" not in self.model_fields_set:
            self.device = "cuda"
        self.output_subdir = "segmentation"
        return self


class SegmentBrainTumorArgs(ToolArgsBase):
    t1c_path: Optional[str] = None
    t1_path: Optional[str] = None
    t2_path: Optional[str] = None
    flair_path: Optional[str] = None
    bundle_root: Optional[str] = None
    device: Optional[str] = None
    output_subdir: Optional[str] = None

    @model_validator(mode="after")
    def _repair(self, info: ValidationInfo) -> "SegmentBrainTumorArgs":
        ctx = _ctx(info)
        if not ctx:
            return self

        fields_set = self.model_fields_set
        mapping = ctx.mapping if isinstance(ctx.mapping, dict) else {}

        if self.t1c_path:
            resolved = ctx.resolve_sequence_token(self.t1c_path)
            if resolved:
                self.t1c_path = resolved
        if self.flair_path:
            resolved = ctx.resolve_sequence_token(self.flair_path)
            if resolved:
                self.flair_path = resolved
        if self.t1_path:
            resolved = ctx.resolve_sequence_token(self.t1_path)
            if resolved:
                self.t1_path = resolved
        if self.t2_path:
            resolved = ctx.resolve_sequence_token(self.t2_path)
            if resolved:
                self.t2_path = resolved

        if "t1c_path" not in fields_set:
            cand = ctx.domain.structural_path(mapping)
            if cand:
                self.t1c_path = cand
            elif mapping.get("T1"):
                self.t1c_path = mapping.get("T1")
        if "flair_path" not in fields_set and mapping.get("FLAIR"):
            self.flair_path = mapping.get("FLAIR")
        if "t1_path" not in fields_set and mapping.get("T1"):
            self.t1_path = mapping.get("T1")
        if "t2_path" not in fields_set and mapping.get("T2"):
            self.t2_path = mapping.get("T2")

        # Bundle root: prefer configured default if missing/invalid.
        default_bundle = ctx.config.brain_bundle_dir
        bd_raw = self.bundle_root
        bd_str = str(bd_raw) if bd_raw is not None else ""
        bd_path = Path(bd_str).expanduser().resolve() if bd_str else default_bundle.resolve()
        looks_placeholder = ("/path/to/" in bd_str) or (bd_str.strip().lower() in ("", "none"))
        has_bundle = (bd_path / "configs" / "inference.json").exists()
        if looks_placeholder or (not bd_path.exists()) or (not has_bundle):
            self.bundle_root = str(default_bundle.resolve())
        else:
            self.bundle_root = str(bd_path)

        if "device" not in fields_set:
            self.device = "auto"
        if isinstance(self.device, str):
            d = self.device.strip().lower()
            if d.startswith("cuda"):
                self.device = "cuda"
            elif "cpu" in d:
                self.device = "cpu"
            elif d not in ("auto", "cpu", "cuda"):
                self.device = "auto"
        if "output_subdir" not in fields_set:
            self.output_subdir = "brats_mri_segmentation"
        return self


class SegmentCardiacCineArgs(ToolArgsBase):
    cine_path: Optional[str] = None
    cmr_reverse_root: Optional[str] = None
    nnunet_python: Optional[str] = None
    results_folder: Optional[str] = None
    task_name: Optional[str] = None
    trainer_class_name: Optional[str] = None
    model: Optional[str] = None
    folds: Optional[Any] = None
    checkpoint: Optional[str] = None
    disable_tta: Optional[bool] = None
    output_subdir: Optional[str] = None

    @model_validator(mode="after")
    def _repair(self, info: ValidationInfo) -> "SegmentCardiacCineArgs":
        ctx = _ctx(info)
        if not ctx:
            return self

        fields_set = self.model_fields_set
        mapping = ctx.mapping if isinstance(ctx.mapping, dict) else {}

        if self.cine_path:
            resolved = ctx.resolve_sequence_token(self.cine_path)
            if resolved:
                self.cine_path = resolved
        if "cine_path" not in fields_set:
            cand = ctx.domain.structural_path(mapping)
            if cand:
                self.cine_path = cand
            elif mapping.get("CINE"):
                self.cine_path = mapping.get("CINE")
            elif mapping:
                # Last-resort fallback for imperfect identify mapping.
                for _, path in mapping.items():
                    if isinstance(path, str) and path:
                        self.cine_path = path
                        break

        default_root = ctx.config.cardiac_cmr_reverse_root
        default_py = ctx.config.cardiac_nnunet_python
        default_results = ctx.config.cardiac_results_folder

        root_raw = str(self.cmr_reverse_root or "").strip()
        root_path = Path(root_raw).expanduser().resolve() if root_raw else default_root
        # Guardrail: reject paths that exist but are clearly not a cmr_reverse repo root.
        try:
            required = [
                root_path / "nnunetdata" / "raw",
                root_path / "nnunetdata" / "preprocessed",
                root_path / "nnunet",
            ]
            if not all(p.exists() for p in required):
                root_path = default_root
        except Exception:
            root_path = default_root
        self.cmr_reverse_root = str(root_path)

        py_raw = str(self.nnunet_python or "").strip()
        py_path = Path(os.path.abspath(os.path.expanduser(py_raw))) if py_raw else default_py
        if not py_path.exists():
            py_path = default_py
        self.nnunet_python = str(py_path)

        res_raw = str(self.results_folder or "").strip()
        res_path = Path(res_raw).expanduser().resolve() if res_raw else default_results
        if not res_path.exists():
            res_path = default_results
        self.results_folder = str(res_path)

        def _looks_task_name(v: str) -> bool:
            s = str(v or "").strip()
            if not s:
                return False
            low = s.lower()
            return low.startswith("task") or s.isdigit() or low in {"acdc", "acdc_phys"}

        task_raw = str(self.task_name or "").strip()
        model_raw = str(self.model or "").strip()
        model_allowed = {"2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"}

        # Common LLM error for this tool: task/model swapped
        # (e.g. task_name="ACDC", model="Task900_ACDC_Phys").
        if _looks_task_name(model_raw) and (not model_raw.lower() in model_allowed):
            if (not _looks_task_name(task_raw)) or task_raw.lower() in {"acdc", "acdc_phys"}:
                task_raw = model_raw
                model_raw = ""

        if (not task_raw) or task_raw.lower() in {"acdc", "acdc_phys", "task900", "task900_acdc", "900"}:
            self.task_name = "Task900_ACDC_Phys"
        elif task_raw.isdigit():
            tid = int(task_raw)
            self.task_name = "Task900_ACDC_Phys" if tid == 900 else f"Task{tid:03d}"
        elif task_raw.lower().startswith("task"):
            self.task_name = task_raw
        else:
            # nnUNet v1 predict_simple expects numeric id or "TaskXXX_*".
            self.task_name = "Task900_ACDC_Phys"

        if "trainer_class_name" not in fields_set:
            self.trainer_class_name = "nnUNetTrainerV2_InvGreAug"

        model_norm = model_raw.lower()
        if model_norm in model_allowed:
            self.model = model_norm
        else:
            # Keep cardiac inference stable on nnUNet-v1 style CLI.
            self.model = "2d"
        folds_val = self.folds
        folds_norm: Optional[List[int]] = None
        if isinstance(folds_val, list):
            tmp: List[int] = []
            for x in folds_val:
                try:
                    tmp.append(int(x))
                except Exception:
                    continue
            folds_norm = tmp or None
        elif isinstance(folds_val, str):
            low = folds_val.strip().lower()
            if low in ("all", "*"):
                folds_norm = [0, 1, 2, 3, 4]
            else:
                parts = [p for p in re.split(r"[,\s]+", low) if p]
                tmp = []
                for p in parts:
                    try:
                        tmp.append(int(p))
                    except Exception:
                        continue
                folds_norm = tmp or None
        elif folds_val is not None:
            try:
                folds_norm = [int(folds_val)]
            except Exception:
                folds_norm = None
        if ("folds" not in fields_set) or (not folds_norm):
            folds_norm = [0, 1, 2, 3, 4]
        self.folds = folds_norm
        if "checkpoint" not in fields_set:
            self.checkpoint = "model_final_checkpoint"
        if "disable_tta" not in fields_set:
            self.disable_tta = True
        if "output_subdir" not in fields_set:
            self.output_subdir = "segmentation/cardiac_cine"
        return self


class ClassifyCardiacCineDiseaseArgs(ToolArgsBase):
    seg_path: Optional[str] = None
    ed_seg_path: Optional[str] = None
    es_seg_path: Optional[str] = None
    cine_path: Optional[str] = None
    patient_info_path: Optional[str] = None
    output_subdir: Optional[str] = None
    ed_frame: Optional[int] = None
    es_frame: Optional[int] = None

    @model_validator(mode="after")
    def _repair(self, info: ValidationInfo) -> "ClassifyCardiacCineDiseaseArgs":
        ctx = _ctx(info)
        if not ctx:
            return self

        fields_set = self.model_fields_set
        mapping = ctx.mapping if isinstance(ctx.mapping, dict) else {}
        seg = ctx.segment_data or {}

        if isinstance(seg, dict):
            if "seg_path" not in fields_set and isinstance(seg.get("seg_path"), str):
                self.seg_path = str(seg.get("seg_path"))
            if ("ed_seg_path" not in fields_set or "es_seg_path" not in fields_set) and isinstance(seg.get("case_results"), list):
                ed_cand = None
                es_cand = None
                for item in seg.get("case_results") or []:
                    if not isinstance(item, dict):
                        continue
                    case_id = str(item.get("case_id") or "").lower()
                    p = str(item.get("seg_path") or "")
                    if not p:
                        continue
                    if ("_ed" in case_id or "_ed." in p.lower() or p.lower().endswith("_ed.nii.gz")) and ed_cand is None:
                        ed_cand = p
                    if ("_es" in case_id or "_es." in p.lower() or p.lower().endswith("_es.nii.gz")) and es_cand is None:
                        es_cand = p
                if "ed_seg_path" not in fields_set and ed_cand:
                    self.ed_seg_path = ed_cand
                if "es_seg_path" not in fields_set and es_cand:
                    self.es_seg_path = es_cand
                if "seg_path" not in fields_set and self.ed_seg_path:
                    self.seg_path = self.ed_seg_path
            if "cine_path" not in fields_set and not self.cine_path:
                cand = ctx.domain.structural_path(mapping)
                if cand:
                    self.cine_path = cand

        if self.cine_path:
            resolved = ctx.resolve_sequence_token(self.cine_path)
            if resolved:
                self.cine_path = resolved
        if (not self.cine_path) and isinstance(mapping.get("CINE"), str):
            self.cine_path = str(mapping.get("CINE"))

        if "patient_info_path" not in fields_set:
            # 1) Near cine path.
            cine_p = Path(str(self.cine_path)).expanduser().resolve() if self.cine_path else None
            if cine_p is not None:
                info_cfg = (cine_p.parent if cine_p.is_file() else cine_p) / "Info.cfg"
                if info_cfg.exists():
                    self.patient_info_path = str(info_cfg)

            # 2) Near seg path.
            if not self.patient_info_path and (self.ed_seg_path or self.seg_path):
                seg_probe = self.ed_seg_path or self.seg_path
                seg_p = Path(str(seg_probe)).expanduser().resolve()
                info_cfg = seg_p.parent / "Info.cfg"
                if info_cfg.exists():
                    self.patient_info_path = str(info_cfg)

            # 3) Search known ACDC roots by patient ID.
            if not self.patient_info_path:
                patient_id = None
                for probe in (self.cine_path, self.seg_path):
                    if not probe:
                        continue
                    m = re.search(r"(patient\d{3})", str(probe), flags=re.I)
                    if m:
                        patient_id = m.group(1)
                        break
                if patient_id:
                    roots: List[Path] = []
                    env_root = os.getenv("MRI_AGENT_ACDC_RAW_ROOT")
                    if env_root:
                        roots.append(Path(env_root).expanduser().resolve())
                    for root in roots:
                        for split in ("training", "testing"):
                            cfg = root / split / patient_id / "Info.cfg"
                            if cfg.exists():
                                self.patient_info_path = str(cfg)
                                break
                        if self.patient_info_path:
                            break

        if "output_subdir" not in fields_set:
            self.output_subdir = "classification/cardiac_cine"
        return self


class ExtractRoiFeaturesArgs(ToolArgsBase):
    roi_mask_path: Optional[str] = None
    roi_masks: Optional[Any] = None
    images: Optional[Any] = None
    output_subdir: Optional[str] = None
    roi_interpretation: Optional[str] = None
    radiomics: Optional[Any] = None
    radiomics_mode: Optional[str] = None
    radiomics_feature_classes: Optional[Any] = None
    radiomics_image_types: Optional[Any] = None
    radiomics_settings: Optional[Any] = None

    @model_validator(mode="after")
    def _repair(self, info: ValidationInfo) -> "ExtractRoiFeaturesArgs":
        ctx = _ctx(info)
        if not ctx:
            return self

        seg = ctx.segment_data or {}
        roi_mask = self.roi_mask_path
        roi_masks = _normalize_roi_masks(self.roi_masks)
        if roi_masks:
            self.roi_masks = roi_masks
        else:
            self.roi_masks = None
            roi_masks = None
        lesion_mode = False
        try:
            if isinstance(roi_masks, list) and roi_masks:
                for r in roi_masks:
                    if isinstance(r, dict) and str(r.get("name", "")).strip().lower() == "lesion":
                        lesion_mode = True
                        break
            if isinstance(roi_mask, str) and "lesion" in roi_mask.lower():
                lesion_mode = True
        except Exception:
            lesion_mode = False

        if ctx.domain.name == "prostate" and not lesion_mode:
            zone_p = seg.get("zone_mask_path") if isinstance(seg, dict) else None
            if isinstance(zone_p, str) and Path(zone_p).exists():
                self.roi_mask_path = zone_p
            elif (not isinstance(roi_mask, str)) or (roi_mask and not Path(roi_mask).exists()):
                if isinstance(seg, dict) and seg.get("prostate_mask_path"):
                    self.roi_mask_path = seg["prostate_mask_path"]
        if ctx.domain.name == "brain":
            def _find_mask(key: str, pattern: str) -> Optional[str]:
                p = seg.get(key) if isinstance(seg, dict) else None
                if isinstance(p, str) and Path(p).exists():
                    return str(Path(p).expanduser().resolve())
                try:
                    for cand in ctx.run_artifacts_dir.glob(f"**/{pattern}"):
                        return str(cand.resolve())
                except Exception:
                    return None
                return None

            def _looks_prostate_roi(items: Optional[List[Dict[str, str]]]) -> bool:
                if not items:
                    return False
                for it in items:
                    name = str(it.get("name") or "").lower()
                    path = str(it.get("path") or "").lower()
                    if any(tok in name for tok in ("prostate", "whole_gland", "peripheral", "central", "zones")):
                        return True
                    if any(tok in path for tok in ("prostate", "t2w_input_zones", "zones")):
                        return True
                return False

            tc = _find_mask("tc_mask_path", "brats_tc_mask.nii.gz")
            wt = _find_mask("wt_mask_path", "brats_wt_mask.nii.gz")
            et = _find_mask("et_mask_path", "brats_et_mask.nii.gz")
            brain_masks: List[Dict[str, str]] = []
            if tc:
                brain_masks.append({"name": "Tumor_Core", "path": tc})
            if wt:
                brain_masks.append({"name": "Whole_Tumor", "path": wt})
            if et:
                brain_masks.append({"name": "Enhancing_Tumor", "path": et})
            if brain_masks and (not roi_masks or _looks_prostate_roi(roi_masks)):
                self.roi_masks = brain_masks
                roi_masks = brain_masks
            if isinstance(self.roi_mask_path, str):
                low = self.roi_mask_path.lower()
                if any(tok in low for tok in ("prostate", "t2w_input_zones", "zones")):
                    self.roi_mask_path = None
        if ctx.domain.name == "brain" and "roi_interpretation" not in self.model_fields_set:
            if (self.roi_masks is not None) or (self.roi_mask_path is not None):
                self.roi_interpretation = "binary_masks"
        if ctx.domain.name == "cardiac":
            # Auto-wire cardiac masks from segment_cardiac_cine outputs:
            # RV/MYO/LV masks for ED/ES frames.
            if not roi_masks and isinstance(seg, dict):
                case_results = seg.get("case_results") if isinstance(seg.get("case_results"), list) else []
                cardiac_masks: List[Dict[str, str]] = []
                for item in case_results:
                    if not isinstance(item, dict):
                        continue
                    case_id = str(item.get("case_id") or "").strip()
                    phase = "ed" if "_ed" in case_id.lower() else ("es" if "_es" in case_id.lower() else case_id.lower())
                    for key, label in (
                        ("rv_mask_path", "RV"),
                        ("myo_mask_path", "MYO"),
                        ("lv_mask_path", "LV"),
                    ):
                        p = item.get(key)
                        if isinstance(p, str) and p and Path(p).exists():
                            name = f"{label}_{phase.upper()}" if phase else label
                            cardiac_masks.append({"name": name, "path": str(Path(p).expanduser().resolve())})
                if cardiac_masks:
                    self.roi_masks = cardiac_masks
                    self.roi_mask_path = None
                    roi_masks = cardiac_masks

            # Use dedicated cardiac interpretation so ROI names are preserved (not zonal remapping).
            if "roi_interpretation" not in self.model_fields_set and (self.roi_masks or self.roi_mask_path):
                self.roi_interpretation = "cardiac"

        locator = ArtifactLocator(ctx.run_artifacts_dir, seg)
        images = _normalize_images_list(self.images)
        roi_mask_paths: set[str] = set()
        for r in roi_masks or []:
            p = r.get("path") if isinstance(r, dict) else None
            if isinstance(p, str) and p:
                try:
                    roi_mask_paths.add(str(Path(p).expanduser().resolve()))
                except Exception:
                    roi_mask_paths.add(p)
        if isinstance(self.roi_mask_path, str) and self.roi_mask_path:
            try:
                roi_mask_paths.add(str(Path(self.roi_mask_path).expanduser().resolve()))
            except Exception:
                roi_mask_paths.add(self.roi_mask_path)

        def _looks_like_mask(path: str) -> bool:
            s = Path(path).name.lower()
            if ctx.domain.name == "brain" and "t2w_input" in s:
                return True
            if "prostate" in s and ctx.domain.name == "brain":
                return True
            return any(tok in s for tok in ("mask", "segmentation", "tumor", "lesion", "zones", "label", "subregion"))

        filtered_images: List[Dict[str, str]] = []
        for it in images:
            p = str(it.get("path", "")).strip()
            if not p:
                continue
            try:
                rp = str(Path(p).expanduser().resolve())
            except Exception:
                rp = p
            if rp in roi_mask_paths or _looks_like_mask(rp):
                continue
            filtered_images.append({"name": str(it.get("name") or ""), "path": p})
        images = filtered_images

        nifti_paths = locator.list_intensity_niftis()
        t2_nifti = locator.t2w_nifti_path()
        if t2_nifti and t2_nifti not in nifti_paths:
            nifti_paths.append(t2_nifti)
        if ctx.domain.name == "cardiac" and isinstance(seg, dict):
            # Prefer ED/ES input frames prepared for nnUNet inference when available.
            added_cardiac_frames = False
            input_dir_raw = seg.get("input_dir")
            if isinstance(input_dir_raw, str) and input_dir_raw:
                input_dir = Path(input_dir_raw).expanduser().resolve()
                case_results = seg.get("case_results") if isinstance(seg.get("case_results"), list) else []
                for item in case_results:
                    if not isinstance(item, dict):
                        continue
                    case_id = str(item.get("case_id") or "").strip()
                    if not case_id:
                        continue
                    p = input_dir / f"{case_id}_0000.nii.gz"
                    if p.exists():
                        low = case_id.lower()
                        if "_ed" in low:
                            nm = "CINE_ED"
                        elif "_es" in low:
                            nm = "CINE_ES"
                        else:
                            nm = f"CINE_{case_id}"
                        images.append({"name": nm, "path": str(p)})
                        added_cardiac_frames = True
            if (not added_cardiac_frames) and isinstance(ctx.mapping, dict) and isinstance(ctx.mapping.get("CINE"), str):
                cine_map = str(ctx.mapping.get("CINE"))
                if cine_map and Path(cine_map).exists():
                    images.append({"name": "CINE", "path": cine_map})
            # Include T1/T1-mapping when available so cardiac ROI features can be extracted on MOLLI maps.
            if isinstance(ctx.mapping, dict):
                for t1_key, t1_name in (("T1_MOLLI", "T1_MOLLI"), ("T1", "T1"), ("T1c", "T1c")):
                    t1_path = ctx.mapping.get(t1_key)
                    if isinstance(t1_path, str) and t1_path:
                        tp = Path(t1_path).expanduser().resolve()
                        if tp.exists() and tp.is_file():
                            images.append({"name": t1_name, "path": str(tp)})
        if ctx.domain.name == "brain" and isinstance(ctx.mapping, dict):
            for key in ("T1c", "T1", "T2", "FLAIR"):
                p = ctx.mapping.get(key)
                if isinstance(p, str) and p:
                    try:
                        rp = Path(p).expanduser().resolve()
                    except Exception:
                        rp = Path(p)
                    if rp.exists() and rp.is_file():
                        images.append({"name": key, "path": str(rp)})

        def _is_registered(p: str) -> bool:
            return "resampled_to_fixed" in p.lower()

        repaired: List[Dict[str, str]] = []
        for it in images:
            p = str(it.get("path", ""))
            name = str(it.get("name", ""))
            if p and Path(p).exists() and Path(p).is_dir():
                if "adc" in name.lower() or "adc" in p.lower():
                    cand = locator.pick_adc_nifti(nifti_paths)
                    if cand:
                        repaired.append({"name": "ADC", "path": cand})
                        continue
            if (("adc" in name.lower()) or ("adc" in p.lower())) and (p and not Path(p).exists()):
                cand = locator.pick_adc_nifti(nifti_paths)
                if cand:
                    repaired.append({"name": "ADC", "path": cand})
                    continue
            if "high_b" in name.lower() or "high_b" in p.lower() or ("dwi" in name.lower() and "low_b" not in name.lower()):
                cand = locator.pick_highb_nifti(nifti_paths)
                if cand and (not _is_registered(p)):
                    repaired.append({"name": "high_b", "path": cand})
                    continue
            repaired.append({"name": name, "path": p})
        images = repaired

        renamed: List[Dict[str, str]] = []
        for it in images:
            p = str(it.get("path", ""))
            n = str(it.get("name", ""))
            pl = p.lower()
            if "high_b" in pl or "high-b" in pl:
                n = "high_b"
            elif "low_b" in pl or "low-b" in pl:
                n = "low_b"
            elif "adc" in pl:
                n = "ADC"
            elif p.endswith("t2w_input.nii.gz") or "t2w_input" in pl:
                n = "T2w_t2w_input"
            renamed.append({"name": n, "path": p})
        images = renamed

        try:
            want_adc = bool(
                "ADC" in ctx.domain.functional_tags
                and isinstance(ctx.mapping, dict)
                and ctx.mapping.get("ADC")
            )
            have_adc = any(
                str(it.get("name", "")).lower() == "adc" or "adc" in str(it.get("path", "")).lower()
                for it in images
            )
            if want_adc and not have_adc:
                adc_path = locator.pick_adc_nifti(nifti_paths)
                if adc_path and Path(adc_path).exists():
                    images.append({"name": "ADC", "path": adc_path})
        except Exception:
            pass

        try:
            if "DWI" in ctx.domain.functional_tags:
                have_highb = any(
                    "high_b" in str(it.get("name", "")).lower() or "high_b" in str(it.get("path", "")).lower()
                    for it in images
                )
                if not have_highb:
                    highb_path = locator.pick_highb_nifti(nifti_paths)
                    if highb_path and Path(highb_path).exists():
                        images.append({"name": "high_b", "path": highb_path})
        except Exception:
            pass

        if t2_nifti and t2_nifti.exists():
            rp = str(t2_nifti.resolve())
            if all(str(Path(it.get("path", "")).resolve()) != rp for it in images if it.get("path")):
                images.append({"name": "T2w_t2w_input", "path": rp})

        if not images:
            images = []
        if len(images) < 2 and nifti_paths:
            seen = set(str(Path(it["path"]).resolve()) for it in images if it.get("path"))
            for p in nifti_paths:
                rp = str(p.resolve())
                if rp in seen:
                    continue
                seen.add(rp)
                images.append({"name": p.parent.name if p.name != "t2w_input.nii.gz" else "T2w_t2w_input", "path": rp})

        cleaned: List[Dict[str, str]] = []
        for it in images:
            p = str(it.get("path") or "").strip()
            if not p:
                continue
            try:
                rp = str(Path(p).expanduser().resolve())
            except Exception:
                rp = p
            cleaned.append({"name": ctx.domain.canonical_image_name(str(it.get("name") or ""), rp), "path": rp})
        has_high_b = any(str(it.get("name")) == "high_b" for it in cleaned)
        if has_high_b:
            cleaned = [it for it in cleaned if str(it.get("name")) != "DWI"]
        dedup: Dict[str, Dict[str, str]] = {}
        for it in cleaned:
            key = f"{it.get('name','')}::{it.get('path','')}"
            if key not in dedup:
                dedup[key] = {"name": str(it.get("name") or ""), "path": str(it.get("path") or "")}
        order = {
            "T2w_t2w_input": 0,
            "T2w": 1,
            "ADC": 2,
            "high_b": 3,
            "low_b": 4,
            "mid_b": 5,
            "DWI": 6,
            "T1_MOLLI": 7,
            "T1": 8,
            "T1c": 9,
            "CINE": 10,
            "CINE_ED": 11,
            "CINE_ES": 12,
        }
        images = sorted(dedup.values(), key=lambda it: order.get(str(it.get("name") or ""), 99))

        if ctx.domain.name == "cardiac":
            # For cardiac feature extraction, default to T1/MOLLI-only features.
            # Cine remains available to segmentation/classification tools.
            t1_only: List[Dict[str, str]] = []
            for it in images:
                nm = str(it.get("name") or "")
                low = nm.lower()
                path_low = str(it.get("path") or "").lower()
                if nm in ("T1_MOLLI", "T1", "T1c") or ("molli" in low) or ("t1" in low) or ("molli" in path_low):
                    t1_only.append(it)
            if t1_only:
                images = t1_only

        self.images = images
        if not self.output_subdir:
            self.output_subdir = "features"
        if ctx.domain.name == "cardiac":
            # Keep cardiac defaults compact and interpretable: turn radiomics off unless explicitly requested.
            if "radiomics_mode" not in self.model_fields_set and "radiomics" not in self.model_fields_set:
                self.radiomics_mode = "off"
            if "radiomics_feature_classes" not in self.model_fields_set and self.radiomics_feature_classes is None:
                self.radiomics_feature_classes = ["firstorder", "shape"]
            if "radiomics_image_types" not in self.model_fields_set and self.radiomics_image_types is None:
                self.radiomics_image_types = ["Original"]
            if "radiomics_settings" not in self.model_fields_set and self.radiomics_settings is None:
                self.radiomics_settings = {"binWidth": 25}
        return self


class DetectLesionCandidatesArgs(ToolArgsBase):
    t2w_nifti: Optional[str] = None
    adc_nifti: Optional[str] = None
    highb_nifti: Optional[str] = None
    prostate_mask_nifti: Optional[str] = None
    weights_dir: Optional[str] = None
    min_volume_cc: Optional[float] = None
    device: Optional[str] = None
    output_subdir: Optional[str] = None

    @model_validator(mode="after")
    def _repair(self, info: ValidationInfo) -> "DetectLesionCandidatesArgs":
        ctx = _ctx(info)
        if not ctx:
            return self

        seg = ctx.segment_data or {}
        locator = ArtifactLocator(ctx.run_artifacts_dir, seg)
        nifti_paths = locator.list_intensity_niftis()
        fields_set = self.model_fields_set

        t2_nifti = locator.t2w_nifti_path()
        if (not self.t2w_nifti) and t2_nifti and t2_nifti.exists():
            self.t2w_nifti = str(t2_nifti)

        # Prefer corrected diffusion outputs when distortion correction succeeded.
        try:
            dist = ctx.latest_tool_data("register", "correct_prostate_distortion")
        except Exception:
            dist = {}
        if isinstance(dist, dict):
            corr_t2 = dist.get("t2w_nifti")
            corr_adc = dist.get("corrected_adc_nifti")
            corr_highb = dist.get("corrected_highb_nifti")
            have_corr = False
            if isinstance(corr_adc, str) and corr_adc and Path(corr_adc).exists():
                have_corr = True
            if (
                isinstance(corr_t2, str)
                and corr_t2
                and Path(corr_t2).exists()
            ):
                # Keep lesion detection in the same reference space used by distortion correction.
                self.t2w_nifti = corr_t2
                have_corr = True
            if isinstance(corr_adc, str) and corr_adc and Path(corr_adc).exists():
                self.adc_nifti = corr_adc
                have_corr = True
            if isinstance(corr_highb, str) and corr_highb and Path(corr_highb).exists():
                self.highb_nifti = corr_highb
                have_corr = True
            if have_corr and t2_nifti and t2_nifti.exists():
                # Force T2 reference to segmentation-space T2 input when corrected DWI/ADC are selected.
                self.t2w_nifti = str(t2_nifti)

        if not self.adc_nifti:
            adc_nifti = locator.pick_adc_nifti(nifti_paths)
            if adc_nifti:
                self.adc_nifti = adc_nifti

        if not self.highb_nifti:
            highb_nifti = locator.pick_highb_nifti(nifti_paths)
            if highb_nifti:
                self.highb_nifti = highb_nifti

        wg = seg.get("prostate_mask_path") if isinstance(seg, dict) else None
        zone = seg.get("zone_mask_path") if isinstance(seg, dict) else None
        if isinstance(wg, str) and Path(wg).exists():
            self.prostate_mask_nifti = wg
        elif isinstance(zone, str) and Path(zone).exists():
            self.prostate_mask_nifti = zone

        default_w = ctx.config.lesion_weights_dir
        wd_raw = self.weights_dir
        wd = Path(str(wd_raw)).expanduser().resolve() if wd_raw else default_w
        need_override = True
        try:
            need_override = not (
                wd.exists()
                and (wd / "fold0" / "model_best_fold0.pth.tar").exists()
                and (wd / "fold4" / "model_best_fold4.pth.tar").exists()
            )
        except Exception:
            need_override = True
        if need_override and default_w.exists():
            self.weights_dir = str(default_w)
        else:
            self.weights_dir = str(wd)

        if "min_volume_cc" not in fields_set:
            self.min_volume_cc = 0.1
        if "device" not in fields_set:
            self.device = "cuda"
        if "output_subdir" not in fields_set:
            self.output_subdir = "lesion"
        return self


class CorrectProstateDistortionArgs(ToolArgsBase):
    t2w_nifti: Optional[str] = None
    adc_nifti: Optional[str] = None
    highb_nifti: Optional[str] = None
    python_exec: Optional[str] = None
    output_subdir: Optional[str] = None
    device: Optional[str] = None
    b_low: Optional[float] = None
    b_high: Optional[float] = None
    steps: Optional[int] = None
    strength: Optional[float] = None
    sampler: Optional[str] = None
    radius: Optional[int] = None
    num_gpus: Optional[int] = None
    save_npz: Optional[bool] = None
    save_slices: Optional[bool] = None

    @model_validator(mode="after")
    def _repair(self, info: ValidationInfo) -> "CorrectProstateDistortionArgs":
        ctx = _ctx(info)
        if not ctx:
            return self
        fields_set = self.model_fields_set
        seg = ctx.segment_data or {}
        locator = ArtifactLocator(ctx.run_artifacts_dir, seg)
        nifti_paths = locator.list_intensity_niftis()

        if "t2w_nifti" not in fields_set:
            t2 = locator.t2w_nifti_path()
            if t2 and t2.exists():
                self.t2w_nifti = str(t2)
        if "adc_nifti" not in fields_set:
            adc = locator.pick_adc_nifti(nifti_paths)
            if adc:
                self.adc_nifti = str(adc)
        if "highb_nifti" not in fields_set:
            highb = locator.pick_highb_nifti(nifti_paths)
            if highb:
                self.highb_nifti = str(highb)

        if "python_exec" not in fields_set:
            self.python_exec = str(os.getenv("MRI_AGENT_PROSTATE_DISTORTION_PYTHON", "python"))

        out_subdir = str(self.output_subdir or "distortion_correction").strip() or "distortion_correction"
        if "output_subdir" not in fields_set:
            self.output_subdir = out_subdir

        if "device" not in fields_set:
            self.device = "cuda"
        if "b_low" not in fields_set:
            self.b_low = 50.0
        if "b_high" not in fields_set:
            self.b_high = 1400.0
        if "steps" not in fields_set:
            self.steps = 100
        if "strength" not in fields_set:
            self.strength = 0.3
        if "sampler" not in fields_set:
            self.sampler = "dpmsolver"
        if "radius" not in fields_set:
            self.radius = 2
        if "num_gpus" not in fields_set:
            self.num_gpus = 1
        if "save_npz" not in fields_set:
            self.save_npz = True
        if "save_slices" not in fields_set:
            self.save_slices = True
        return self


class PackageVlmEvidenceArgs(ToolArgsBase):
    case_state_path: Optional[str] = None
    output_subdir: Optional[str] = None

    @model_validator(mode="after")
    def _repair(self, info: ValidationInfo) -> "PackageVlmEvidenceArgs":
        ctx = _ctx(info)
        if not ctx:
            return self
        self.case_state_path = str(ctx.ctx_case_state_path)
        if "output_subdir" not in self.model_fields_set:
            self.output_subdir = "vlm"
        return self


class GenerateReportArgs(ToolArgsBase):
    case_state_path: Optional[str] = None
    feature_table_path: Optional[str] = None
    output_subdir: Optional[str] = None
    domain: Optional[str] = None
    degradation_context: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _repair(self, info: ValidationInfo) -> "GenerateReportArgs":
        ctx = _ctx(info)
        if not ctx:
            return self
        self.case_state_path = str(ctx.ctx_case_state_path)
        if "output_subdir" not in self.model_fields_set:
            self.output_subdir = "report"
        if "domain" not in self.model_fields_set and ctx.domain is not None:
            self.domain = str(ctx.domain.name)
        try:
            main_feat = ctx.run_artifacts_dir / "features" / "features.csv"
            if main_feat.exists():
                self.feature_table_path = str(main_feat)
        except Exception:
            pass
        return self


_ARG_MODEL_REGISTRY: Dict[str, type[ToolArgsBase]] = {
    "ingest_dicom_to_nifti": IngestDicomArgs,
    "identify_sequences": IngestDicomArgs,
    "register_to_reference": RegisterToReferenceArgs,
    "segment_prostate": SegmentProstateArgs,
    "brats_mri_segmentation": SegmentBrainTumorArgs,
    "segment_cardiac_cine": SegmentCardiacCineArgs,
    "classify_cardiac_cine_disease": ClassifyCardiacCineDiseaseArgs,
    "extract_roi_features": ExtractRoiFeaturesArgs,
    "detect_lesion_candidates": DetectLesionCandidatesArgs,
    "correct_prostate_distortion": CorrectProstateDistortionArgs,
    "package_vlm_evidence": PackageVlmEvidenceArgs,
    "generate_report": GenerateReportArgs,
}


def repair_tool_args(
    tool_name: str,
    args: Dict[str, Any],
    *,
    state_path: Path,
    ctx_case_state_path: Path,
    dicom_case_dir: Optional[str],
    domain: Optional[DomainConfig] = None,
    suppress_sequence_resolve: bool = False,
    suppress_node_output_autowire: bool = False,
) -> Dict[str, Any]:
    """Repair tool args via Pydantic models.

    Logic is intentionally split for ablation clarity:
    A) type/schema repair (allowed in noToken): type coercion, defaults, shape normalization
    B) semantic binding repair (disabled in noToken via suppress_* flags): sequence-name->path
       resolution and prior-node output / artifact autowire.
    """
    ctx = _build_context(
        tool_name,
        state_path,
        ctx_case_state_path,
        dicom_case_dir,
        domain,
        suppress_sequence_resolve=suppress_sequence_resolve,
        suppress_node_output_autowire=suppress_node_output_autowire,
    )
    model_cls = _ARG_MODEL_REGISTRY.get(tool_name, GenericToolArgs)
    try:
        safe_args = dict(args or {})
    except Exception:
        safe_args = {}
    try:
        model = model_cls.model_validate(safe_args, context={"ctx": ctx})
        return model.model_dump(exclude_none=True)
    except Exception as e:
        print(f"WARNING: Arg repair failed for {tool_name}: {e}")
        return safe_args

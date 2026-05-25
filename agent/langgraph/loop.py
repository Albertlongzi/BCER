from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from functools import partial
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

try:
    from langgraph.graph import END, StateGraph  # type: ignore
except Exception:  # pragma: no cover - optional dependency in test/runtime envs
    END = "__END__"
    StateGraph = None  # type: ignore

from commands.dispatcher import ToolDispatcher
from commands.registry import ToolRegistry
from commands.schemas import CaseState, ToolCall
from core.parser import parse_model_output
from llm.adapter_vllm_server import VLLMOpenAIChatAdapter, VLLMServerConfig
from llm.adapter_openai_api import OpenAIChatAdapter, OpenAIConfig
from llm.adapter_anthropic_api import AnthropicChatAdapter, AnthropicConfig
from llm.adapter_gemini_api import GeminiChatAdapter, GeminiConfig
from core.plan_dag import AgentPlanDAG, CaseScope, DagPolicy, PlanNode
from ..loop import (  # reuse proven helpers; we can refactor later
    _auto_repair_args_for_tool,
    _compact_tool_index,
    summarize_case_state,
    _latest_tool_data,
)
from ..subagents.planner import PlannerSubagent
from ..subagents.sketch_planner import SketchPlannerSubagent
from ..subagents.policy import PolicySubagent, build_guided_schema
from ..subagents.reflector import ReflectorSubagent
from ..subagents.alignment_gate import AlignmentGateSubagent
from ..subagents.distortion_gate import DistortionGateSubagent
from ..subagents.prompts import (
    LANGGRAPH_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    REFLECT_SYSTEM_PROMPT,
    build_sketch_planner_system_prompt,
)
from ..skills.registry import SkillRegistry
from ..hooks.preconditions import apply_preconditions
from ..hooks.circuit_breaker import apply_circuit_breaker
from ..rules.engine import validate_tool_call
from core.domain_config import DomainConfig, get_domain_config
from core.paths import project_root
from tools.dicom_ingest import build_tools as build_dicom_tools
from tools.roi_features import build_tool as build_feature_tool
from tools.prostate_lesion_candidates import build_tool as build_lesion_candidates_tool
from tools.prostate_distortion_correction import build_tool as build_prostate_distortion_tool
from tools.vlm_evidence import build_tool as build_package_vlm_evidence_tool
from tools.report_generation import build_tool as build_report_tool
from tools.registration import build_tool as build_registration_tool
from tools.alignment_qc import build_tool as build_alignment_qc_tool
from tools.materialize_registration import build_tool as build_materialize_registration_tool
from tools.prostate_segmentation import build_tool as build_segmentation_tool
from tools.brain_tumor_segmentation import build_tool as build_brats_mri_segmentation_tool
from tools.brain_glioma_grade_classification import build_tool as build_brain_glioma_grade_classification_tool
from tools.cardiac_cine_segmentation import build_tool as build_cardiac_cine_segmentation_tool
from tools.cardiac_cine_classification import build_tool as build_cardiac_cine_classification_tool
from tools.bm3d_denoising import build_tool as build_bm3d_denoise_tool
from tools.resample_image import build_tool as build_resample_image_tool
from tools.reconstruct_grappa import build_tool as build_grappa_tool
from tools.compare_nifti_slices import build_tool as build_compare_nifti_tool
from tools.generate_qa_snapshot import build_tool as build_qa_snapshot_tool
from tools.arg_models import RepairConfig
from runtime.memory import append_short_term_event, build_memory_digest, compact_for_memory, extract_key_paths
from runtime.tool_manifest import write_tool_manifest
from ..plans.template_loader import load_plan_template
from ..plans.sketch_schema import ConstrainedPlanSketch, constrained_plan_sketch_guided_json_schema
from ..plans.sketch_compiler import compile_constrained_plan_sketch


def _flag_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def _register_experimental_tools(reg: ToolRegistry) -> None:
    enable_exp = _flag_enabled("MRI_AGENT_ENABLE_EXPERIMENTAL")
    enable_rag = True
    enable_sandbox = True
    if enable_rag:
        from tools.rag_search import build_tool as build_rag_search_tool

        reg.register(build_rag_search_tool())
    if enable_sandbox:
        from tools.sandbox_exec import build_tool as build_sandbox_tool

        reg.register(build_sandbox_tool())


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for t in build_dicom_tools():
        reg.register(t)
    reg.register(build_registration_tool())
    reg.register(build_alignment_qc_tool())
    reg.register(build_materialize_registration_tool())
    reg.register(build_segmentation_tool())
    reg.register(build_brats_mri_segmentation_tool())
    reg.register(build_brain_glioma_grade_classification_tool())
    reg.register(build_cardiac_cine_segmentation_tool())
    reg.register(build_cardiac_cine_classification_tool())
    reg.register(build_feature_tool())
    reg.register(build_lesion_candidates_tool())
    reg.register(build_prostate_distortion_tool())
    reg.register(build_package_vlm_evidence_tool())
    reg.register(build_report_tool())
    reg.register(build_bm3d_denoise_tool())
    reg.register(build_resample_image_tool())
    reg.register(build_grappa_tool())
    reg.register(build_compare_nifti_tool())
    reg.register(build_qa_snapshot_tool())
    _register_experimental_tools(reg)
    return reg


def _build_planner_llm(
    *,
    llm_mode: str,
    max_new_tokens: int,
    server_cfg: Optional[VLLMServerConfig],
    api_model: Optional[str],
    api_base_url: Optional[str],
) -> Any:
    if llm_mode == "server":
        cfg = server_cfg or VLLMServerConfig(max_tokens=int(max_new_tokens))
        return VLLMOpenAIChatAdapter(cfg)
    if llm_mode == "openai":
        return OpenAIChatAdapter(OpenAIConfig(model=api_model or OpenAIConfig().model, base_url=api_base_url))
    if llm_mode == "anthropic":
        return AnthropicChatAdapter(AnthropicConfig(model=api_model or AnthropicConfig().model))
    if llm_mode == "gemini":
        return GeminiChatAdapter(GeminiConfig(model=api_model or GeminiConfig().model))

    from ..loop import FakeLLM  # type: ignore

    return FakeLLM()


def _planner_default_external_model_roots() -> List[str]:
    out: List[str] = []
    seen: set[str] = set()

    def _push(path_like: Any) -> None:
        raw = str(path_like or "").strip()
        if not raw:
            return
        try:
            val = str(Path(raw).expanduser().resolve())
        except Exception:
            return
        if val in seen:
            return
        seen.add(val)
        out.append(val)

    _push(Path("~/.cache/torch/hub/bundle").expanduser())
    _push(Path("~/.cache/torch/hub").expanduser())
    try:
        cfg = RepairConfig.from_env(project_root())
        _push(cfg.model_registry_path)
        _push(cfg.prostate_bundle_dir)
        _push(cfg.lesion_weights_dir)
        _push(cfg.brain_bundle_dir)
        _push(cfg.cardiac_cmr_reverse_root)
        _push(cfg.cardiac_results_folder)
        _push(cfg.cardiac_nnunet_python)
        _push(Path(cfg.cardiac_nnunet_python).expanduser().resolve().parent)
        _push(cfg.distortion_repo_root)
        _push(cfg.distortion_diff_ckpt)
        _push(cfg.distortion_cnn_ckpt)
        for p in (cfg.distortion_test_roots or []):
            _push(p)
    except Exception:
        pass
    return out


def _abs_path_no_resolve(path_like: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path_like or ""))))


def _scan_case_ref(case_ref: Path, *, max_files: int = 200) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "exists": case_ref.exists(),
        "is_dir": case_ref.is_dir(),
        "nifti_files": [],
        "h5_files": [],
        "dicom_files": 0,
    }
    if not case_ref.exists():
        return info
    if case_ref.is_file():
        s = str(case_ref).lower()
        if s.endswith(".nii.gz") or s.endswith(".nii"):
            info["nifti_files"] = [str(case_ref)]
        elif s.endswith(".h5") or s.endswith(".hdf5"):
            info["h5_files"] = [str(case_ref)]
        elif s.endswith(".dcm"):
            info["dicom_files"] = 1
        elif "." not in case_ref.name:
            try:
                with open(case_ref, "rb") as _fh:
                    _fh.seek(128)
                    if _fh.read(4) == b"DICM":
                        info["dicom_files"] = 1
            except Exception:
                pass
        return info

    nifti_files: List[str] = []
    h5_files: List[str] = []
    dicom_count = 0
    seen = 0
    for p in case_ref.rglob("*"):
        if not p.is_file():
            continue
        seen += 1
        if seen > max_files:
            break
        sl = str(p).lower()
        if sl.endswith(".nii.gz") or sl.endswith(".nii"):
            nifti_files.append(str(p))
            continue
        if sl.endswith(".h5") or sl.endswith(".hdf5"):
            h5_files.append(str(p))
            continue
        if sl.endswith(".dcm"):
            dicom_count += 1
            continue
        # Extension-less file – probe for DICOM magic bytes (b"DICM" at offset 128)
        if "." not in p.name and p.name.lower() not in ("dicomdir", "lockfile", "version"):
            try:
                with open(p, "rb") as _fh:
                    _fh.seek(128)
                    if _fh.read(4) == b"DICM":
                        dicom_count += 1
            except Exception:
                pass
    info["nifti_files"] = sorted(nifti_files)[:64]
    info["h5_files"] = sorted(h5_files)[:64]
    info["dicom_files"] = int(dicom_count)
    return info


def _infer_modalities_from_case(domain_name: str, scan: Dict[str, Any]) -> Dict[str, Optional[bool]]:
    paths: List[str] = []
    raw = scan.get("nifti_files")
    if isinstance(raw, list):
        paths = [str(x).lower() for x in raw if isinstance(x, str)]
    h5_paths: List[str] = []
    raw_h5 = scan.get("h5_files")
    if isinstance(raw_h5, list):
        h5_paths = [str(x).lower() for x in raw_h5 if isinstance(x, str)]
    has_observations = bool(paths) or bool(h5_paths) or int(scan.get("dicom_files") or 0) > 0

    def _has_any(keys: List[str]) -> Optional[bool]:
        if not has_observations:
            return None
        # DICOM-only case at planning time: keep modality as unknown.
        if not paths:
            return None
        for p in paths:
            if any(k in p for k in keys):
                return True
        return False

    if domain_name == "prostate":
        return {
            "T2w": _has_any(["t2w", "/t2/", "_t2", "-t2"]),
            "ADC": _has_any(["adc"]),
            "DWI": _has_any(["dwi", "trace", "high_b", "b800", "b1000", "b1400", "b1500", "b2000"]),
        }
    if domain_name == "brain":
        return {
            "T1c": _has_any(["t1c", "t1ce", "t1gd", "t1_gd", "t1_ce"]),
            "T1": _has_any(["/t1/", "_t1", "-t1", "/t1.", "t1.nii"]),
            "T2": _has_any(["/t2/", "_t2", "-t2", "/t2.", "t2.nii"]),
            "FLAIR": _has_any(["flair"]),
        }
    if domain_name == "cardiac":
        cine_from_nifti = _has_any(["cine", "bssfp", "ssfp", "sax", "shortaxis", "short_axis"])
        if cine_from_nifti is True:
            return {"CINE": True}
        if h5_paths:
            # HDF5 cardiac exams can be converted by reconstruct_grappa before segmentation.
            return {"CINE": True}
        return {"CINE": cine_from_nifti}
    return {}


def _collect_directory_evidence(case_ref: Path, *, max_files: int = 240, max_entries: int = 80) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "case_ref": str(case_ref),
        "entries": [],
        "nifti_files": [],
    }
    if not case_ref.exists():
        return info
    paths: List[Path] = []
    if case_ref.is_file():
        paths = [case_ref]
    else:
        seen = 0
        for p in case_ref.rglob("*"):
            if not p.is_file():
                continue
            seen += 1
            if seen > max_files:
                break
            paths.append(p)
    paths = sorted(paths, key=lambda x: str(x).lower())

    entries: List[Dict[str, str]] = []
    nifti_files: List[str] = []
    for p in paths:
        p_abs = str(_abs_path_no_resolve(str(p)))
        try:
            rel = str(p.relative_to(case_ref))
        except Exception:
            rel = p.name
        if len(entries) < max_entries:
            entries.append({"name": str(p.name), "relative_path": rel, "path": p_abs})
        pl = p_abs.lower()
        if pl.endswith(".nii.gz") or pl.endswith(".nii"):
            nifti_files.append(p_abs)
    info["entries"] = entries
    info["nifti_files"] = nifti_files[:128]
    return info


def _extract_abs_paths_by_suffix(text: str, suffixes: Tuple[str, ...]) -> List[str]:
    """Extract absolute-like paths from free text, filtering by file suffix."""
    if not suffixes:
        return []
    wanted = tuple(str(s).lower() for s in suffixes)
    s = str(text or "")
    candidates: List[str] = []

    for q in re.findall(r"""["']([^"']+)["']""", s):
        qq = str(q or "").strip()
        if qq:
            candidates.append(qq)
    for tok in s.split():
        t = str(tok or "").strip().strip(",;:.!?)]}(")
        if t.startswith("/") or t.startswith("~/"):
            candidates.append(t)

    out: List[str] = []
    seen: set[str] = set()
    for raw in candidates:
        try:
            p = _abs_path_no_resolve(str(raw))
            low = str(p).lower()
            if not low.endswith(wanted):
                continue
            val = str(p)
        except Exception:
            continue
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _find_case_file_by_suffix(case_ref: Path, suffixes: Tuple[str, ...], *, max_files: int = 1200) -> Optional[str]:
    """Best-effort scan for the first matching file under case_ref."""
    wanted = tuple(str(s).lower() for s in suffixes)
    if not wanted:
        return None
    if case_ref.is_file():
        low = str(case_ref).lower()
        if low.endswith(wanted):
            return str(_abs_path_no_resolve(str(case_ref)))
        return None
    if not case_ref.exists() or not case_ref.is_dir():
        return None

    seen = 0
    try:
        for p in case_ref.rglob("*"):
            if not p.is_file():
                continue
            seen += 1
            if seen > max_files:
                break
            if str(p).lower().endswith(wanted):
                return str(_abs_path_no_resolve(str(p)))
    except Exception:
        return None
    return None


def _score_cardiac_cine_candidate(path_like: str) -> int:
    p = str(path_like or "").strip().lower()
    if not p:
        return 0
    if re.search(r"_gt(\.nii(\.gz)?)?$", p):
        return 0
    score = 0
    if "cine" in p:
        score = max(score, 10)
    if "bssfp" in p or "ssfp" in p:
        score = max(score, 9)
    if "shortaxis" in p or "short_axis" in p or "sax" in p:
        score = max(score, 8)
    if re.search(r"(?:^|[^a-z0-9])4d(?:[^a-z0-9]|$)", p):
        score = max(score, 7)
    if p.endswith("_4d.nii.gz") or p.endswith("_4d.nii"):
        score = max(score, 7)
    if re.search(r"_frame\d{2,3}(?:\.nii(?:\.gz)?)?$", p):
        score = max(score, 6)
    return score


def _directory_modality_candidates(domain_name: str, directory_evidence: Dict[str, Any]) -> Dict[str, List[str]]:
    raw_paths = directory_evidence.get("nifti_files")
    nifti_paths = [str(x) for x in raw_paths if isinstance(x, str)] if isinstance(raw_paths, list) else []
    dom = str(domain_name or "").strip().lower()

    if dom == "cardiac":
        ranked: List[Tuple[int, str]] = []
        for p in nifti_paths:
            score = _score_cardiac_cine_candidate(p)
            if score <= 0:
                continue
            ranked.append((score, p))
        ranked = sorted(ranked, key=lambda x: (-int(x[0]), str(x[1]).lower()))
        return {"CINE": [p for _, p in ranked]}

    return {}


def _merge_modality_hints_with_directory_evidence(
    *,
    modality_hints: Dict[str, Optional[bool]],
    modality_candidates: Dict[str, List[str]],
) -> Tuple[Dict[str, Optional[bool]], List[str]]:
    merged: Dict[str, Optional[bool]] = dict(modality_hints or {})
    notes: List[str] = []
    for modality, cands in (modality_candidates or {}).items():
        candidates = [str(x) for x in (cands or []) if str(x).strip()]
        if not candidates:
            continue
        prev = merged.get(modality)
        if prev is True:
            continue
        merged[modality] = True
        sample = str(Path(candidates[0]).name)
        if prev is False:
            notes.append(f"Directory evidence overrode missing modality hint: {modality} (example: {sample}).")
        else:
            notes.append(f"Directory evidence inferred modality: {modality} (example: {sample}).")
    return merged, notes


def _compact_directory_evidence_for_prompt(
    *,
    directory_evidence: Dict[str, Any],
    modality_candidates: Dict[str, List[str]],
    max_entries: int = 40,
) -> Dict[str, Any]:
    raw_entries = directory_evidence.get("entries")
    entries = [x for x in (raw_entries or []) if isinstance(x, dict)] if isinstance(raw_entries, list) else []
    entries = entries[:max_entries]
    raw_nifti = directory_evidence.get("nifti_files")
    nifti_files = [str(x) for x in (raw_nifti or []) if isinstance(x, str)] if isinstance(raw_nifti, list) else []
    candidate_preview: Dict[str, List[str]] = {}
    for k, vals in (modality_candidates or {}).items():
        vv = [str(Path(x).name) for x in (vals or []) if str(x).strip()]
        if vv:
            candidate_preview[str(k)] = vv[:8]
    return {
        "nifti_count": len(nifti_files),
        "entries_preview": entries,
        "candidate_modalities": candidate_preview,
    }


def _merge_modality_candidate_maps(*maps: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for mp in maps:
        if not isinstance(mp, dict):
            continue
        for key, vals in mp.items():
            k = str(key or "").strip()
            if not k:
                continue
            bucket = out.setdefault(k, [])
            seen = set(bucket)
            for v in (vals or []):
                sv = str(v or "").strip()
                if not sv or sv in seen:
                    continue
                seen.add(sv)
                bucket.append(sv)
    return out


def _explicit_input_alias_candidates(
    *,
    domain_name: str,
    explicit_input_aliases: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, List[str]], List[str]]:
    aliases = explicit_input_aliases if isinstance(explicit_input_aliases, dict) else {}
    dom = str(domain_name or "").strip().lower()
    if not aliases:
        return {}, []

    def _vals(*keys: str) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for k in keys:
            raw = aliases.get(k)
            if isinstance(raw, str):
                s = str(raw).strip()
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
            elif isinstance(raw, list):
                for item in raw:
                    s = str(item or "").strip()
                    if s and s not in seen:
                        seen.add(s)
                        out.append(s)
        return out

    cands: Dict[str, List[str]] = {}
    if dom == "brain":
        for modal, keys in (
            ("T1", ("t1_nifti",)),
            ("T1c", ("t1c_nifti", "t1ce_nifti")),
            ("T2", ("t2_nifti", "t2w_nifti")),
            ("FLAIR", ("flair_nifti",)),
        ):
            vv = _vals(*keys)
            if vv:
                cands[modal] = vv
    elif dom == "prostate":
        for modal, keys in (
            ("T2w", ("t2w_nifti", "t2_nifti")),
            ("ADC", ("adc_nifti",)),
            ("DWI", ("dwi_highb_nifti", "highb_nifti", "dwi_nifti")),
        ):
            vv = _vals(*keys)
            if vv:
                cands[modal] = vv
    elif dom == "cardiac":
        vv = _vals("cine_nifti")
        if vv:
            cands["CINE"] = vv

    notes: List[str] = []
    if cands:
        preview = ", ".join(
            [f"{k}={Path(v[0]).name}" for k, v in sorted(cands.items()) if isinstance(v, list) and v]
        )
        if preview:
            notes.append("Loaded explicit input aliases from manifest: " + preview)
    return cands, notes


_ALIAS_PLACEHOLDER_RE = re.compile(r"^\$\{alias:([a-zA-Z0-9_.-]+)\}$")


def _inject_explicit_input_aliases(
    *,
    nodes: List[PlanNode],
    explicit_input_aliases: Optional[Dict[str, Any]],
) -> Tuple[List[PlanNode], List[str], List[str]]:
    aliases = explicit_input_aliases if isinstance(explicit_input_aliases, dict) else {}
    if not aliases:
        return [n.model_copy(deep=True) for n in nodes], [], []

    used: set[str] = set()
    missing: List[str] = []
    missing_seen: set[str] = set()

    def _resolve(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _resolve(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_resolve(x) for x in v]
        if not isinstance(v, str):
            return v
        m = _ALIAS_PLACEHOLDER_RE.match(v.strip())
        if not m:
            return v
        key = str(m.group(1) or "").strip()
        raw = aliases.get(key)
        if isinstance(raw, str):
            s = str(raw).strip()
            if s:
                used.add(key)
                return s
        elif isinstance(raw, list):
            vals = [str(x).strip() for x in raw if str(x).strip()]
            if vals:
                used.add(key)
                return vals
        if key not in missing_seen:
            missing_seen.add(key)
            missing.append(key)
        return v

    out: List[PlanNode] = []
    changed = 0
    for node in nodes:
        before = dict(node.arguments or {})
        after = _resolve(before)
        if after != before:
            out.append(node.model_copy(update={"arguments": after}))
            changed += 1
        else:
            out.append(node.model_copy(deep=True))

    notes: List[str] = []
    if changed > 0 and used:
        notes.append("Injected explicit input aliases into placeholders: " + ", ".join(sorted(used)))
    if missing:
        notes.append("Missing explicit input aliases for placeholders: " + ", ".join(sorted(missing)))
    return out, notes, missing


def _bind_direct_modality_paths(
    *,
    nodes: List[PlanNode],
    domain_name: str,
    modality_candidates: Dict[str, List[str]],
) -> Tuple[List[PlanNode], List[str]]:
    dom = str(domain_name or "").strip().lower()
    out = [n.model_copy(deep=True) for n in nodes]
    changed: List[str] = []

    if dom == "cardiac":
        cine_candidates = [str(x) for x in (modality_candidates.get("CINE") or []) if str(x).strip()]
        if not cine_candidates:
            return nodes, []
        cine_path = cine_candidates[0]
        for i, node in enumerate(out):
            tool = str(node.tool_name or "").strip()
            if tool not in {"segment_cardiac_cine", "classify_cardiac_cine_disease"}:
                continue
            args = dict(node.arguments or {})
            cur = str(args.get("cine_path") or "").strip()
            if cur not in {"", "CINE", "@seq.CINE"}:
                continue
            args["cine_path"] = cine_path
            out[i] = node.model_copy(update={"arguments": args})
            changed.append(f"{node.node_id}.cine_path")

    elif dom == "brain":
        for i, node in enumerate(out):
            tool = str(node.tool_name or "").strip()
            args = dict(node.arguments or {})
            patched = False
            if tool == "brats_mri_segmentation":
                for arg_key, modal in (
                    ("t1_path", "T1"),
                    ("t1c_path", "T1c"),
                    ("t2_path", "T2"),
                    ("flair_path", "FLAIR"),
                ):
                    cands = [str(x) for x in (modality_candidates.get(modal) or []) if str(x).strip()]
                    cur = str(args.get(arg_key) or "").strip()
                    if cands and cur in {"", modal, f"@seq.{modal}"}:
                        args[arg_key] = cands[0]
                        changed.append(f"{node.node_id}.{arg_key}")
                        patched = True
            elif tool == "register_to_reference":
                for arg_key in ("fixed", "moving"):
                    cur = str(args.get(arg_key) or "").strip()
                    cands = [str(x) for x in (modality_candidates.get(cur) or []) if str(x).strip()]
                    if cands and cur in {"T1", "T1c", "T2", "FLAIR"}:
                        args[arg_key] = cands[0]
                        changed.append(f"{node.node_id}.{arg_key}")
                        patched = True
            if patched:
                out[i] = node.model_copy(update={"arguments": args})

    elif dom == "prostate":
        for i, node in enumerate(out):
            tool = str(node.tool_name or "").strip()
            args = dict(node.arguments or {})
            patched = False
            if tool == "register_to_reference":
                for arg_key in ("fixed", "moving"):
                    cur = str(args.get(arg_key) or "").strip()
                    cands = [str(x) for x in (modality_candidates.get(cur) or []) if str(x).strip()]
                    if cands and cur in {"T2w", "ADC", "DWI"}:
                        args[arg_key] = cands[0]
                        changed.append(f"{node.node_id}.{arg_key}")
                        patched = True
            elif tool == "segment_prostate":
                cur = str(args.get("t2w_ref") or "").strip()
                cands = [str(x) for x in (modality_candidates.get("T2w") or []) if str(x).strip()]
                if cands and cur in {"", "T2w", "@seq.T2w"}:
                    args["t2w_ref"] = cands[0]
                    changed.append(f"{node.node_id}.t2w_ref")
                    patched = True
            if patched:
                out[i] = node.model_copy(update={"arguments": args})

    notes: List[str] = []
    if changed:
        notes.append(
            "Bound direct modality paths to concrete files before execution: "
            + ", ".join(sorted(set(changed)))
        )
    return out, notes


def _extract_tool_mentions(plan_text: str, tool_names: List[str]) -> List[str]:
    if not str(plan_text or "").strip():
        return []
    out: List[str] = []
    seen = set()
    low_text = str(plan_text).lower()
    for t in tool_names:
        tl = str(t).strip().lower()
        if not tl:
            continue
        if tl in low_text and tl not in seen:
            out.append(str(t))
            seen.add(tl)
    return out


def _canonical_modality_token(domain_name: str, raw: str) -> str:
    tok = str(raw or "").strip().lower().replace("-", "").replace("_", "")
    if not tok:
        return ""
    aliases: Dict[str, Dict[str, str]] = {
        "brain": {
            "t1c": "T1c",
            "t1ce": "T1c",
            "t1gd": "T1c",
            "t1": "T1",
            "t2": "T2",
            "t2w": "T2",
            "flair": "FLAIR",
        },
        "prostate": {
            "t2": "T2w",
            "t2w": "T2w",
            "adc": "ADC",
            "dwi": "DWI",
            "highb": "DWI",
            "trace": "DWI",
        },
        "cardiac": {
            "cine": "CINE",
            "bssfp": "CINE",
            "ssfp": "CINE",
            "sax": "CINE",
        },
    }
    table = aliases.get(str(domain_name or "").strip().lower(), {})
    return str(table.get(tok) or "")


def _seq_ref_for_modality(domain_name: str, token: str) -> str:
    t = str(token or "").strip()
    d = str(domain_name or "").strip().lower()
    if d == "brain":
        mapping = {"T1c": "@seq.T1c", "T1": "@seq.T1", "T2": "@seq.T2", "FLAIR": "@seq.FLAIR"}
        return str(mapping.get(t) or t)
    if d == "prostate":
        mapping = {"T2w": "@seq.T2w", "ADC": "@seq.ADC", "DWI": "@seq.DWI"}
        return str(mapping.get(t) or t)
    if d == "cardiac":
        mapping = {"CINE": "@seq.CINE"}
        return str(mapping.get(t) or t)
    return t


def _extract_registration_override_pairs(text: str, domain_name: str) -> List[Tuple[str, str]]:
    src = str(text or "")
    if not src:
        return []
    raw_tokens = sorted(
        set(
            [
                x
                for x in [
                    "t1c",
                    "t1ce",
                    "t1gd",
                    "t1",
                    "t2w",
                    "t2",
                    "flair",
                    "adc",
                    "dwi",
                    "high_b",
                    "high-b",
                    "trace",
                    "cine",
                    "bssfp",
                    "ssfp",
                    "sax",
                ]
                if _canonical_modality_token(domain_name, x)
            ]
        ),
        key=len,
        reverse=True,
    )
    if not raw_tokens:
        return []
    alt = "|".join([re.escape(x) for x in raw_tokens])
    pairs: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for m in re.finditer(rf"\b({alt})\b\s*(?:to|->)\s*\b({alt})\b", src, flags=re.IGNORECASE):
        moving = _canonical_modality_token(domain_name, str(m.group(1) or ""))
        fixed = _canonical_modality_token(domain_name, str(m.group(2) or ""))
        if not moving or not fixed or moving == fixed:
            continue
        pair = (moving, fixed)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _apply_registration_overrides(
    *,
    nodes: List[PlanNode],
    domain_name: str,
    pairs: List[Tuple[str, str]],
) -> Tuple[List[PlanNode], List[str]]:
    if not pairs:
        return nodes, []
    out = [n.model_copy(deep=True) for n in nodes]
    reg_idx = [i for i, n in enumerate(out) if str(n.tool_name or "").strip() == "register_to_reference"]
    if not reg_idx:
        return out, ["Registration override requested but no register_to_reference node exists in this plan."]

    notes: List[str] = []
    used_indices: set[int] = set()
    for moving_tok, fixed_tok in pairs:
        picked: Optional[int] = None
        for idx in reg_idx:
            if idx in used_indices:
                continue
            current_moving = _canonical_modality_token(domain_name, str((out[idx].arguments or {}).get("moving") or ""))
            if current_moving and current_moving == moving_tok:
                picked = idx
                break
        if picked is None:
            for idx in reg_idx:
                if idx not in used_indices:
                    picked = idx
                    break
        if picked is None:
            notes.append(f"Ignored extra registration override {moving_tok}->{fixed_tok}; no free register node remained.")
            continue

        used_indices.add(picked)
        args = dict(out[picked].arguments or {})
        args["moving"] = _seq_ref_for_modality(domain_name, moving_tok)
        args["fixed"] = _seq_ref_for_modality(domain_name, fixed_tok)
        out_subdir = str(args.get("output_subdir") or "").strip()
        if out_subdir.lower().startswith("registration/"):
            args["output_subdir"] = f"registration/{moving_tok}"
        new_label = f"Register {moving_tok} to {fixed_tok}"
        out[picked] = out[picked].model_copy(update={"arguments": args, "label": new_label, "required": True})
        notes.append(f"Applied registration override: moving={moving_tok} -> fixed={fixed_tok}.")
    return out, notes


def _missing_modalities_for_planner(
    *,
    domain_name: str,
    request_type: str,
    modality_hints: Dict[str, Optional[bool]],
    nodes: Optional[List[PlanNode]] = None,
) -> List[str]:
    req = _normalize_request_type(request_type) or "full_pipeline"
    if req in {"report", "qa", "custom_analysis", "denoise", "super_resolution", "raw_recon"}:
        return []

    def _miss(key: str) -> bool:
        return modality_hints.get(key) is False

    def _token_from_arg_value(v: Any) -> str:
        if not isinstance(v, str):
            return ""
        s = str(v).strip()
        if not s:
            return ""
        if s.startswith("@seq."):
            return _canonical_modality_token(domain_name, s[len("@seq.") :])
        return _canonical_modality_token(domain_name, s)

    if isinstance(nodes, list):
        required_tokens: set[str] = set()
        for node in nodes:
            if not bool(node.required):
                continue
            args = node.arguments or {}
            if not isinstance(args, dict):
                continue
            for av in args.values():
                tok = _token_from_arg_value(av)
                if tok:
                    required_tokens.add(tok)
        return [tok for tok in sorted(required_tokens) if _miss(tok)]

    missing: List[str] = []
    dom = str(domain_name or "").strip().lower()
    if dom == "cardiac":
        if req in {"full_pipeline", "segment", "classify"} and _miss("CINE"):
            missing.append("CINE")
        return missing
    if dom == "prostate":
        if req == "segment":
            if _miss("T2w"):
                missing.append("T2w")
            return missing
        if req in {"full_pipeline", "register", "lesion"}:
            for k in ("T2w", "ADC", "DWI"):
                if _miss(k):
                    missing.append(k)
            return missing
        return missing
    if dom == "brain":
        if req in {"full_pipeline", "segment"}:
            for k in ("T1c", "T2", "FLAIR"):
                if _miss(k):
                    missing.append(k)
            return missing
        if req == "register":
            for k in ("T1c", "T2", "FLAIR"):
                if _miss(k):
                    missing.append(k)
            return missing
        return missing
    return missing


def _planner_natural_response(
    *,
    llm: Any,
    status: str,
    domain_name: str,
    request_type: str,
    case_ref: Path,
    goal: str,
    blocking_reasons: List[str],
    modality_hints: Dict[str, Optional[bool]],
) -> str:
    blocked_bits = ", ".join([str(x) for x in (blocking_reasons or []) if str(x).strip()])
    req = _normalize_request_type(request_type) or "full_pipeline"
    if str(status) != "blocked":
        return (
            f"I prepared a {domain_name} plan for request_type={req} "
            f"using case '{case_ref.name}'."
        )

    fallback = (
        f"I see you requested {domain_name} {req}, but I can't proceed yet. "
        f"I found missing required modalities: {blocked_bits}. "
        "Please load a compatible case or adjust the request."
    )
    prompt_payload = {
        "status": "blocked",
        "domain": domain_name,
        "request_type": req,
        "goal": str(goal or ""),
        "case_ref": str(case_ref),
        "modality_hints": modality_hints,
        "blocking_reasons": blocking_reasons,
        "style": "Return one concise, polite sentence for a clinician user.",
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are an MRI planning assistant. "
                "Given structured planning facts, return only a short natural-language response."
            ),
        },
        {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
    ]
    try:
        text = str(llm.generate(messages)).strip()
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            if blocked_bits and blocked_bits.lower() not in text.lower():
                text = f"{text} Missing requirements: {blocked_bits}."
            return text
    except Exception:
        pass
    return fallback


def _tool_line_is_optional(plan_text: str, tool_name: str) -> bool:
    tl = str(tool_name or "").strip().lower()
    if not tl:
        return False
    lines = [ln.strip().lower() for ln in str(plan_text or "").splitlines() if ln.strip()]
    optional_tokens = ("optional", "if needed", "if available", "when available", "if present", "can skip")
    required_tokens = ("required", "must", "always")
    for ln in lines:
        if tl not in ln:
            continue
        if any(tok in ln for tok in required_tokens):
            return False
        if any(tok in ln for tok in optional_tokens):
            return True
    return False


def _is_negated_goal_term(goal_text: str, term: str) -> bool:
    txt = str(goal_text or "").lower()
    tm = re.escape(str(term or "").lower())
    if not txt or not tm:
        return False
    patterns = [
        rf"\b(?:no|without|skip|omit|exclude)\b[^.\n]{{0,48}}\b{tm}\b",
        rf"\b(?:do\s+not|don'?t|dont|no\s+need|not\s+need(?:ed)?|not\s+necessary)\b[^.\n]{{0,48}}\b{tm}\b",
        rf"\b{tm}\b[^.\n]{{0,48}}\b(?:not\s+needed|unnecessary|to\s+skip)\b",
    ]
    return any(bool(re.search(p, txt)) for p in patterns)


def _prune_nodes_by_constraints(*, nodes: List[PlanNode], intent: Dict[str, bool]) -> Tuple[List[PlanNode], List[str]]:
    block_tools: set[str] = set()
    if bool(intent.get("forbid_report")):
        block_tools.update({"package_vlm_evidence", "generate_report"})
    if bool(intent.get("forbid_features")):
        block_tools.add("extract_roi_features")
    if bool(intent.get("forbid_lesion")):
        block_tools.add("detect_lesion_candidates")
    if bool(intent.get("forbid_classification")):
        block_tools.add("classify_cardiac_cine_disease")
    if bool(intent.get("segment_only")):
        block_tools.update(
            {
                "extract_roi_features",
                "detect_lesion_candidates",
                "classify_cardiac_cine_disease",
                "package_vlm_evidence",
                "generate_report",
            }
        )
    if not block_tools:
        return nodes, []

    removed = [n.tool_name for n in nodes if n.tool_name in block_tools]
    kept = [n for n in nodes if n.tool_name not in block_tools]
    valid_ids = {n.node_id for n in kept}
    normalized: List[PlanNode] = []
    for node in kept:
        deps = [d for d in (node.depends_on or []) if d in valid_ids]
        if deps != list(node.depends_on or []):
            normalized.append(node.model_copy(update={"depends_on": deps}))
        else:
            normalized.append(node)
    notes = []
    if removed:
        notes.append("Planner constraints pruned nodes: " + ", ".join(sorted(set([str(x) for x in removed]))))
    return normalized, notes


def _normalize_request_type(raw: str) -> str:
    s = str(raw or "").strip().lower()
    aliases = {
        "registration": "register",
        "segmentation": "segment",
        "classification": "classify",
        "denoising": "denoise",
        "bm3d": "denoise",
        "superres": "super_resolution",
        "superresolution": "super_resolution",
        "super-resolution": "super_resolution",
        "upsample": "super_resolution",
        "upsample2x": "super_resolution",
        "resample": "super_resolution",
        "resampling": "super_resolution",
        "regrid": "super_resolution",
        "rawrecon": "raw_recon",
        "raw-recon": "raw_recon",
        "raw_reconstruction": "raw_recon",
        "reconstruct_raw": "raw_recon",
        "read_metadata": "qa",
        "metadata": "qa",
        "question_answer": "qa",
        "question-answer": "qa",
        "question": "qa",
        "qa": "qa",
        "custom": "custom_analysis",
        "custom_analysis": "custom_analysis",
        "code_interpreter": "custom_analysis",
        "code-interpreter": "custom_analysis",
        "sandbox": "custom_analysis",
    }
    s = aliases.get(s, s)
    if s in {
        "full_pipeline",
        "segment",
        "register",
        "lesion",
        "report",
        "classify",
        "denoise",
        "super_resolution",
        "raw_recon",
        "qa",
        "custom_analysis",
    }:
        return s
    return ""


def _goal_implies_metadata_qa(goal: str) -> bool:
    txt = str(goal or "").strip().lower()
    if not txt:
        return False
    if any(k in txt for k in ("dicom header", "header information", "metadata only", "read metadata", "read header")):
        return True
    if bool(re.search(r"\b(what|show|read|tell)\b[^.\n]{0,64}\b(tr|te|echo time|repetition time)\b", txt)):
        return True
    if bool(re.search(r"\b(only|just)\b[^.\n]{0,40}\b(read|show|inspect)\b[^.\n]{0,80}\b(header|metadata)\b", txt)):
        return True
    return False


def _goal_implies_custom_analysis(goal: str) -> bool:
    txt = str(goal or "").strip().lower()
    if not txt:
        return False
    keys = (
        "custom statistical",
        "custom calculation",
        "custom analysis",
        "code interpreter",
        "python script",
        "nibabel",
        "numpy",
        "run python",
        "sandbox",
    )
    return any(k in txt for k in keys)


def _goal_implies_denoise(goal: str) -> bool:
    txt = str(goal or "").strip().lower()
    if not txt:
        return False
    keys = (
        "denoise",
        "de-noise",
        "bm3d",
        "self-supervised denois",
    )
    return any(k in txt for k in keys)


def _goal_implies_super_resolution(goal: str) -> bool:
    txt = str(goal or "").strip().lower()
    if not txt:
        return False
    keys = (
        "super resolution",
        "super-resolution",
        "superres",
        "upsample",
        "up-sample",
        "resample",
        "regrid",
        "2x",
    )
    return any(k in txt for k in keys)


def _goal_implies_raw_recon(goal: str) -> bool:
    txt = str(goal or "").strip().lower()
    if not txt:
        return False
    has_raw = any(k in txt for k in ("raw", "k-space", "kspace", ".h5", "hdf5", "grappa", "reconstruct_grappa"))
    has_cardiac = any(k in txt for k in ("cardiac", "cine", "sax"))
    return bool(has_raw and has_cardiac)


def _extract_requested_workflow_type(goal: str, explicit_request_type: Optional[str]) -> str:
    req = _normalize_request_type(str(explicit_request_type or ""))

    # Honour an explicit, specific request_type from the Shell classifier.
    # This prevents goal-content heuristics (e.g. _goal_implies_raw_recon)
    # from overriding a precise intent like "full_pipeline", "classify", etc.
    _specific_types = {
        "denoise", "super_resolution", "raw_recon", "qa",
        "full_pipeline", "classify", "segment", "report", "lesion",
        "register",
    }
    if req in _specific_types:
        return req

    # Goal-content heuristics — ordered from most specific to least.
    if _goal_implies_metadata_qa(goal):
        return "qa"
    if _goal_implies_denoise(goal):
        return "denoise"
    if _goal_implies_super_resolution(goal):
        return "super_resolution"
    if _goal_implies_raw_recon(goal):
        return "raw_recon"
    if _goal_implies_custom_analysis(goal):
        return "custom_analysis"

    if req:
        return req
    goal_s = str(goal or "").strip().lower()
    m = re.search(r"requested workflow type:\s*([a-z_]+)", goal_s)
    if m:
        req = _normalize_request_type(str(m.group(1) or ""))
        if req:
            return req
    m2 = re.search(r"\brequest_type\b\s*[:=]\s*([a-z_]+)", goal_s)
    if m2:
        req = _normalize_request_type(str(m2.group(1) or ""))
        if req:
            return req
    return ""


def _prune_nodes_by_request_type(*, nodes: List[PlanNode], request_type: str) -> Tuple[List[PlanNode], List[str]]:
    req = _normalize_request_type(request_type)
    if req in {"", "full_pipeline", "report"}:
        return nodes, []

    allowed_stages: Optional[set[str]] = None
    allowed_tools: Optional[set[str]] = None
    if req == "register":
        allowed_stages = {"identify", "register"}
    elif req == "segment":
        allowed_stages = {"identify", "segment"}
    elif req == "lesion":
        allowed_stages = {"identify", "register", "segment", "lesion"}
    elif req == "classify":
        allowed_tools = {
            "identify_sequences",
            "reconstruct_grappa",
            "brats_mri_segmentation",
            "extract_roi_features",
            "classify_brain_glioma_grade",
            "segment_cardiac_cine",
            "classify_cardiac_cine_disease",
            "generate_qa_snapshot",
        }
    elif req == "denoise":
        allowed_tools = {"identify_sequences", "denoise_image_bm3d", "compare_nifti_slices"}
    elif req == "super_resolution":
        allowed_tools = {"identify_sequences", "resample_image", "compare_nifti_slices"}
    elif req == "raw_recon":
        allowed_tools = {
            "reconstruct_grappa",
            "generate_qa_snapshot",
        }
    elif req == "qa":
        allowed_tools = {"identify_sequences", "rag_search", "search_repo"}
    elif req == "custom_analysis":
        allowed_tools = {"identify_sequences", "sandbox_exec"}
    else:
        return nodes, []

    selected: set[str] = set()
    for node in nodes:
        stage = str(node.stage or "").strip().lower()
        tool = str(node.tool_name or "").strip()
        if allowed_tools is not None and tool in allowed_tools:
            selected.add(str(node.node_id))
            continue
        if allowed_stages is not None and stage in allowed_stages:
            selected.add(str(node.node_id))

    if not selected:
        return nodes, []

    by_id: Dict[str, PlanNode] = {str(n.node_id): n for n in nodes}
    keep_ids = set(selected)
    stack = list(selected)
    while stack:
        nid = stack.pop()
        node = by_id.get(nid)
        if node is None:
            continue
        for dep in (node.depends_on or []):
            dep_id = str(dep).strip()
            if not dep_id or dep_id in keep_ids:
                continue
            if dep_id in by_id:
                keep_ids.add(dep_id)
                stack.append(dep_id)

    kept = [n for n in nodes if str(n.node_id) in keep_ids]
    removed = [str(n.tool_name) for n in nodes if str(n.node_id) not in keep_ids]
    notes: List[str] = []
    if removed:
        notes.append(
            "Request-type pruning applied "
            f"(request_type={req}): removed " + ", ".join(sorted(set([x for x in removed if x])))
        )
    return kept, notes


def _force_required_for_terminal_request_type(*, nodes: List[PlanNode], request_type: str) -> Tuple[List[PlanNode], List[str]]:
    req = _normalize_request_type(request_type)
    if req in {"", "full_pipeline"}:
        return nodes, []

    out: List[PlanNode] = []
    forced: List[str] = []
    for node in nodes:
        match = False
        stage = str(node.stage or "").strip().lower()
        tool = str(node.tool_name or "").strip()
        if req in {"register", "segment", "lesion"} and stage == req:
            match = True
        elif req == "classify" and tool == "classify_cardiac_cine_disease":
            match = True
        elif req == "denoise" and tool == "denoise_image_bm3d":
            match = True
        elif req == "super_resolution" and tool == "resample_image":
            match = True
        elif req == "raw_recon" and tool in {
            "reconstruct_grappa",
            "segment_cardiac_cine",
            "classify_cardiac_cine_disease",
        }:
            match = True
        elif req == "report" and tool == "generate_report":
            match = True
        elif req == "qa" and tool in {"rag_search", "search_repo"}:
            match = True
        elif req == "custom_analysis" and tool == "sandbox_exec":
            match = True

        if match and (not bool(node.required)):
            out.append(node.model_copy(update={"required": True}))
            forced.append(str(node.node_id))
            continue
        out.append(node)

    notes: List[str] = []
    if forced:
        notes.append(
            "Forced required=True for terminal request_type nodes "
            f"(request_type={req}): " + ", ".join(sorted(set(forced)))
        )
    return out, notes


def _infer_goal_intent(goal: str, domain_name: str, plan_text: str, tool_mentions: List[str]) -> Dict[str, bool]:
    goal_txt = str(goal or "").lower()
    txt = f"{goal_txt}\n{str(plan_text or '')}".lower()
    req_type = ""
    m_req = re.search(r"requested workflow type:\s*([a-z_]+)", goal_txt)
    if m_req:
        req_type = str(m_req.group(1) or "").strip().lower()

    wants_seg_only = any(k in txt for k in ("segmentation only", "segment only", "only segment", "mask only"))
    wants_seg_only = wants_seg_only or bool(re.search(r"\bjust\b[^.\n]{0,24}\bsegment(?:ation)?\b", goal_txt))
    if req_type == "segment":
        wants_seg_only = True

    forbid_report = _is_negated_goal_term(goal_txt, "report")
    forbid_features = _is_negated_goal_term(goal_txt, "feature") or _is_negated_goal_term(goal_txt, "radiomic")
    forbid_lesion = _is_negated_goal_term(goal_txt, "lesion")
    forbid_classification = _is_negated_goal_term(goal_txt, "classif")
    if wants_seg_only:
        forbid_report = True
        forbid_features = True
        forbid_lesion = True
        forbid_classification = True

    wants_report = any(k in txt for k in ("report", "impression", "findings", "diagnos", "assessment"))
    if "generate_report" in tool_mentions and not forbid_report:
        wants_report = True
    if wants_seg_only or forbid_report:
        wants_report = False

    wants_features = any(k in txt for k in ("feature", "radiomic", "quantitative", "roi"))
    if "extract_roi_features" in tool_mentions and not forbid_features:
        wants_features = True
    if forbid_features:
        wants_features = False

    wants_lesion = any(k in txt for k in ("lesion", "pirads", "pi-rads", "suspicious focus", "tumor candidate"))
    if "detect_lesion_candidates" in tool_mentions and not forbid_lesion:
        wants_lesion = True
    if forbid_lesion:
        wants_lesion = False

    wants_classification = any(
        k in txt
        for k in (
            "classif",
            "disease group",
            "acdc",
            "cardiomyopathy",
            "hgg",
            "lgg",
            "glioma grade",
            "tumor grade",
        )
    )
    if "classify_cardiac_cine_disease" in tool_mentions and not forbid_classification:
        wants_classification = True
    if "classify_brain_glioma_grade" in tool_mentions and not forbid_classification:
        wants_classification = True
    if forbid_classification:
        wants_classification = False

    if domain_name == "prostate" and wants_report and (not forbid_features):
        wants_features = True
    if domain_name == "cardiac" and wants_report and (not forbid_classification):
        wants_classification = True

    return {
        "wants_report": bool(wants_report),
        "wants_features": bool(wants_features),
        "wants_lesion": bool(wants_lesion),
        "wants_classification": bool(wants_classification),
        "forbid_report": bool(forbid_report),
        "forbid_features": bool(forbid_features),
        "forbid_lesion": bool(forbid_lesion),
        "forbid_classification": bool(forbid_classification),
        "segment_only": bool(wants_seg_only),
    }


def _drop_nodes_and_repair_dependencies(*, nodes: List[PlanNode], drop_ids: set[str]) -> List[PlanNode]:
    if not drop_ids:
        return [n.model_copy(deep=True) for n in nodes]
    kept = [n.model_copy(deep=True) for n in nodes if str(n.node_id) not in drop_ids]
    valid_ids = {str(n.node_id) for n in kept}
    out: List[PlanNode] = []
    for node in kept:
        deps = [str(d) for d in (node.depends_on or []) if str(d) in valid_ids]
        if deps != list(node.depends_on or []):
            out.append(node.model_copy(update={"depends_on": deps}))
        else:
            out.append(node)
    return out


def _apply_node_updates(*, nodes: List[PlanNode], updates: Dict[str, Dict[str, Any]]) -> List[PlanNode]:
    if not updates:
        return [n.model_copy(deep=True) for n in nodes]
    out: List[PlanNode] = []
    for node in nodes:
        patch = updates.get(str(node.node_id))
        if patch:
            out.append(node.model_copy(update=patch))
        else:
            out.append(node.model_copy(deep=True))
    return out


def _patch_qa_template_nodes(
    *,
    nodes: List[PlanNode],
    goal: str,
    domain_name: str,
    llm_mode: str,
    server_cfg: Optional[VLLMServerConfig],
    api_model: Optional[str],
    api_base_url: Optional[str],
) -> Tuple[List[PlanNode], List[str], List[str]]:
    rag_args: Dict[str, Any] = {
        "question": str(goal or "").strip(),
        "case_state_path": "@runtime.case_state_path",
        "output_subdir": "qa",
        "domain": str(domain_name or ""),
    }
    mode = str(llm_mode or "").strip().lower()
    if mode in {"server", "openai", "anthropic", "gemini"}:
        rag_args["llm_mode"] = mode
        if mode == "server":
            rag_args["server_base_url"] = str((server_cfg.base_url if server_cfg else "") or "http://127.0.0.1:8000/v1")
            rag_args["server_model"] = str((server_cfg.model if server_cfg else "") or "")
        elif mode == "openai":
            rag_args["api_model"] = str(api_model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini")
            if str(api_base_url or "").strip():
                rag_args["api_base_url"] = str(api_base_url).strip()
        elif mode == "anthropic":
            rag_args["api_model"] = str(api_model or os.environ.get("ANTHROPIC_MODEL") or "claude-3-7-sonnet-20250219")
        elif mode == "gemini":
            rag_args["api_model"] = str(api_model or os.environ.get("GEMINI_MODEL") or "gemini-1.5-pro")

    patched = _apply_node_updates(
        nodes=nodes,
        updates={
            "rag_search_020": {
                "arguments": rag_args,
                "required": True,
                "depends_on": ["identify_sequences_001"],
            }
        },
    )
    notes = ["QA/read-metadata intent detected: planned minimal DAG identify_sequences -> rag_search."]
    return patched, notes, []


def _patch_custom_analysis_template_nodes(
    *,
    nodes: List[PlanNode],
    goal: str,
    domain_name: str,
) -> Tuple[List[PlanNode], List[str], List[str]]:
    safe_goal = repr(str(goal or ""))
    cmd = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        f"Path('analysis_request.txt').write_text({safe_goal}, encoding='utf-8')\n"
        "print('custom_analysis scaffold prepared: analysis_request.txt')\n"
        "PY"
    )
    patched = _apply_node_updates(
        nodes=nodes,
        updates={
            "sandbox_exec_020": {
                "arguments": {
                    "cmd": cmd,
                    "output_subdir": "sandbox/custom_analysis",
                    "capture_globs": ["*.txt", "*.json"],
                    "max_output_chars": 4000,
                },
                "required": True,
                "depends_on": ["identify_sequences_001"],
                "label": f"Run sandbox custom analysis scaffold ({domain_name})",
            }
        },
    )
    notes = ["Custom-analysis intent detected: planned identify_sequences -> sandbox_exec scaffold DAG."]
    return patched, notes, []


def _patch_denoise_template_nodes(
    *,
    nodes: List[PlanNode],
    goal: str,
    case_scan: Dict[str, Any],
) -> Tuple[List[PlanNode], List[str], List[str]]:
    notes: List[str] = []
    blocking: List[str] = []

    input_nifti: Optional[str] = None
    goal_nifti = _extract_abs_paths_by_suffix(goal, (".nii.gz", ".nii"))
    if goal_nifti:
        input_nifti = goal_nifti[0]
    else:
        scan_nifti = case_scan.get("nifti_files")
        if isinstance(scan_nifti, list):
            for p in scan_nifti:
                sp = str(p or "").strip()
                if not sp:
                    continue
                input_nifti = sp
                break
        if input_nifti:
            notes.append(f"Denoise input inferred from case scan: {input_nifti}")

    prepend_ingest = False
    if not input_nifti:
        dicom_count = int(case_scan.get("dicom_files") or 0)
        if dicom_count > 0:
            prepend_ingest = True
            notes.append(
                f"No NIfTI files found but {dicom_count} DICOM file(s) detected; "
                "prepending identify_sequences to convert DICOM → NIfTI before denoising."
            )
        else:
            blocking.append("denoise request requires an input NIfTI path (.nii/.nii.gz) or a DICOM case directory.")
            return [], notes, blocking

    updates: Dict[str, Dict[str, Any]] = {}
    if prepend_ingest:
        updates["denoise_bm3d_010"] = {
            "arguments": {"input_nifti": "@auto", "output_subdir": "denoise"},
            "depends_on": ["identify_sequences_001"],
            "required": True,
        }
        patched = _apply_node_updates(nodes=nodes, updates=updates)
    else:
        updates["denoise_bm3d_010"] = {
            "arguments": {"input_nifti": str(input_nifti), "output_subdir": "denoise"},
            "depends_on": [],
            "required": True,
        }
        patched = _apply_node_updates(nodes=nodes, updates=updates)
        patched = _drop_nodes_and_repair_dependencies(nodes=patched, drop_ids={"identify_sequences_001"})

    compare_required = _goal_requests_comparison_png(goal)
    patched = _apply_node_updates(
        nodes=patched,
        updates={"compare_denoise_020": {"required": bool(compare_required), "depends_on": ["denoise_bm3d_010"]}},
    )

    notes.append("Denoise request detected: planned denoise_image_bm3d with optional comparison PNG generation.")
    return patched, notes, blocking


def _patch_super_resolution_template_nodes(
    *,
    nodes: List[PlanNode],
    goal: str,
    case_scan: Dict[str, Any],
) -> Tuple[List[PlanNode], List[str], List[str]]:
    notes: List[str] = []
    blocking: List[str] = []

    input_nifti: Optional[str] = None
    goal_nifti = _extract_abs_paths_by_suffix(goal, (".nii.gz", ".nii"))
    if goal_nifti:
        input_nifti = goal_nifti[0]
    else:
        scan_nifti = case_scan.get("nifti_files")
        if isinstance(scan_nifti, list):
            for p in scan_nifti:
                sp = str(p or "").strip()
                if not sp:
                    continue
                input_nifti = sp
                break
        if input_nifti:
            notes.append(f"Super-resolution input inferred from case scan: {input_nifti}")

    prepend_ingest = False
    if not input_nifti:
        dicom_count = int(case_scan.get("dicom_files") or 0)
        if dicom_count > 0:
            prepend_ingest = True
            notes.append(
                f"No NIfTI files found but {dicom_count} DICOM file(s) detected; "
                "prepending identify_sequences to convert DICOM → NIfTI before super-resolution."
            )
        else:
            blocking.append("super_resolution request requires an input NIfTI path (.nii/.nii.gz) or a DICOM case directory.")
            return [], notes, blocking

    target_spacing = _infer_half_spacing_from_nifti(str(input_nifti)) if input_nifti else None
    if target_spacing is None:
        target_spacing = [0.5, 0.5, 0.5]
        notes.append("Could not read input spacing at planning time; defaulting target_spacing to [0.5, 0.5, 0.5].")

    updates: Dict[str, Dict[str, Any]] = {}
    if prepend_ingest:
        updates["resample_sr_010"] = {
            "arguments": {
                "input_nifti": "@auto",
                "target_spacing": [float(x) for x in target_spacing],
                "interpolation": "bspline",
                "output_subdir": "resample",
            },
            "depends_on": ["identify_sequences_001"],
            "required": True,
        }
        patched = _apply_node_updates(nodes=nodes, updates=updates)
    else:
        updates["resample_sr_010"] = {
            "arguments": {
                "input_nifti": str(input_nifti),
                "target_spacing": [float(x) for x in target_spacing],
                "interpolation": "bspline",
                "output_subdir": "resample",
            },
            "depends_on": [],
            "required": True,
        }
        patched = _apply_node_updates(nodes=nodes, updates=updates)
        patched = _drop_nodes_and_repair_dependencies(nodes=patched, drop_ids={"identify_sequences_001"})

    compare_required = _goal_requests_comparison_png(goal)
    patched = _apply_node_updates(
        nodes=patched,
        updates={"compare_superres_020": {"required": bool(compare_required), "depends_on": ["resample_sr_010"]}},
    )

    notes.append("Super-resolution request detected: planned resample_image with optional comparison PNG generation.")
    return patched, notes, blocking


def _patch_raw_recon_template_nodes(
    *,
    nodes: List[PlanNode],
    goal: str,
    domain_name: str,
    case_ref: Path,
) -> Tuple[List[PlanNode], List[str], List[str]]:
    notes: List[str] = []
    blocking: List[str] = []
    dom = str(domain_name or "").strip().lower()
    if dom != "cardiac":
        blocking.append(f"raw_recon is currently supported for cardiac domain only (got domain={dom or 'unknown'}).")
        return [], notes, blocking

    h5_path: Optional[str] = None
    goal_h5 = _extract_abs_paths_by_suffix(goal, (".h5",))
    if goal_h5:
        h5_path = goal_h5[0]
    else:
        h5_path = _find_case_file_by_suffix(case_ref, (".h5",))
        if h5_path:
            notes.append(f"Raw reconstruction input inferred from case directory: {h5_path}")

    if not h5_path:
        blocking.append("raw_recon request requires an HDF5 k-space path (.h5).")
        return [], notes, blocking

    patched = _apply_node_updates(
        nodes=nodes,
        updates={
            "reconstruct_grappa_010": {
                "arguments": {"h5_path": str(h5_path), "output_subdir": "grappa"},
                "required": True,
                "depends_on": [],
            }
        },
    )
    notes.append("Raw reconstruction request detected: planned reconstruct_grappa -> generate_qa_snapshot.")
    return patched, notes, blocking


def _patch_prostate_template_nodes(
    *,
    nodes: List[PlanNode],
    intent: Dict[str, bool],
    modalities: Dict[str, Optional[bool]],
    optional_overrides: Dict[str, bool],
) -> Tuple[List[PlanNode], List[str], List[str]]:
    notes: List[str] = []
    missing_modalities: List[str] = []
    omitted_tools: List[str] = []

    adc_required = True
    if modalities.get("ADC") is False:
        adc_required = False
        missing_modalities.append("ADC")
        omitted_tools.append("register_adc")
        notes.append("ADC was not detected in case scan; ADC registration marked optional.")

    dwi_required = True
    if modalities.get("DWI") is False:
        dwi_required = False
        missing_modalities.append("DWI")
        omitted_tools.append("register_dwi")
        notes.append("DWI/high-b was not detected in case scan; DWI registration marked optional.")

    lesion_required = True

    features_required = bool(intent.get("wants_features"))
    if optional_overrides.get("extract_roi_features"):
        features_required = False

    lesion_deps = ["segment_prostate_030"]
    if adc_required:
        lesion_deps.append("register_adc_010")
    if dwi_required:
        lesion_deps.append("register_dwi_020")

    updates: Dict[str, Dict[str, Any]] = {
        "register_adc_010": {"required": adc_required},
        "register_dwi_020": {"required": dwi_required},
        "extract_features_040": {"required": features_required},
        "detect_lesions_050": {"required": lesion_required, "depends_on": lesion_deps},
    }

    if bool(intent.get("wants_report")):
        package_deps = ["segment_prostate_030"]
        if features_required:
            package_deps.append("extract_features_040")
        if lesion_required:
            package_deps.append("detect_lesions_050")
        report_args: Dict[str, Any] = {
            "case_state_path": "@runtime.case_state_path",
            "output_subdir": "report",
            "domain": "prostate",
        }
        if missing_modalities or omitted_tools:
            report_args["degradation_context"] = {
                "missing_modalities": missing_modalities,
                "omitted_tools": omitted_tools,
                "degraded_mode": True,
                "advisory": (
                    f"Limited exam: {', '.join(missing_modalities)} modality not acquired. "
                    f"Omitted: {', '.join(omitted_tools)}. "
                    "Report must state these limitations explicitly."
                ),
            }
        updates["package_evidence_060"] = {"depends_on": package_deps}
        updates["generate_report_070"] = {"arguments": report_args}
        patched = _apply_node_updates(nodes=nodes, updates=updates)
    else:
        patched = _apply_node_updates(nodes=nodes, updates=updates)
        patched = _drop_nodes_and_repair_dependencies(
            nodes=patched,
            drop_ids={"package_evidence_060", "generate_report_070"},
        )
    return patched, notes, []


def _patch_brain_template_nodes(
    *,
    nodes: List[PlanNode],
    intent: Dict[str, bool],
    optional_overrides: Dict[str, bool],
) -> Tuple[List[PlanNode], List[str], List[str]]:
    notes: List[str] = []
    reg_required = bool(intent.get("wants_lesion"))
    classify_required = bool(intent.get("wants_classification"))
    features_required = bool(intent.get("wants_features"))
    if classify_required:
        features_required = True
    if optional_overrides.get("extract_roi_features") and (not classify_required):
        features_required = False

    updates: Dict[str, Dict[str, Any]] = {
        "register_t2_010": {"required": reg_required},
        "register_flair_020": {"required": reg_required},
        "extract_features_040": {"required": features_required},
        "classify_brain_grade_050": {
            "required": classify_required,
            "depends_on": ["extract_features_040"],
            "arguments": {
                "feature_table_path": "@node.extract_features_040.feature_table_path",
                "output_subdir": "classification/brain_grade",
            },
        },
    }
    if bool(intent.get("wants_report")):
        package_deps = ["segment_brats_030"]
        if features_required:
            package_deps.append("extract_features_040")
        if classify_required:
            package_deps.append("classify_brain_grade_050")
        updates["package_evidence_060"] = {"depends_on": package_deps}
        patched = _apply_node_updates(nodes=nodes, updates=updates)
    else:
        patched = _apply_node_updates(nodes=nodes, updates=updates)
        patched = _drop_nodes_and_repair_dependencies(
            nodes=patched,
            drop_ids={"package_evidence_060", "generate_report_070"},
        )
    return patched, notes, []


def _patch_cardiac_template_nodes(
    *,
    nodes: List[PlanNode],
    intent: Dict[str, bool],
    optional_overrides: Dict[str, bool],
    case_ref: Optional[Path],
) -> Tuple[List[PlanNode], List[str], List[str]]:
    notes: List[str] = []
    h5_path: Optional[str] = None
    if case_ref is not None:
        try:
            h5_path = _find_case_file_by_suffix(case_ref, (".h5",))
        except Exception:
            h5_path = None

    classify_required = bool(intent.get("wants_classification")) or bool(intent.get("wants_report"))
    features_required = bool(intent.get("wants_features"))
    if optional_overrides.get("extract_roi_features"):
        features_required = False

    updates: Dict[str, Dict[str, Any]] = {
        "classify_cine_030": {"required": classify_required},
        "extract_features_040": {"required": features_required},
    }

    if h5_path:
        patched = _drop_nodes_and_repair_dependencies(
            nodes=nodes,
            drop_ids={"reconstruct_grappa_005"},
        )
        identify_case_arg = "@case.input"
        if case_ref is not None and case_ref.is_file():
            identify_case_arg = str(case_ref.parent)
            notes.append("Cardiac full pipeline keeps identify_sequences first; file case input was remapped to parent directory for identify step.")
        updates.update(
            {
                "identify_sequences_001": {
                    "arguments": {
                        "dicom_case_dir": identify_case_arg,
                        "convert_to_nifti": True,
                        "output_subdir": "ingest",
                    },
                    "depends_on": [],
                    "required": True,
                },
                "segment_cine_020": {
                    "arguments": {
                        "cine_path": "@node.reconstruct_grappa_005.reconstructed_nifti",
                        "output_subdir": "segmentation/cardiac_cine",
                    },
                    "depends_on": ["reconstruct_grappa_005"],
                },
                "classify_cine_030": {
                    "required": classify_required,
                    "arguments": {
                        "cine_path": "@node.reconstruct_grappa_005.reconstructed_nifti",
                        "seg_path": "@node.segment_cine_020.seg_path",
                        "seg_dir": "@node.segment_cine_020.pred_dir",
                        "output_subdir": "classification/cardiac_cine",
                    },
                    "depends_on": ["segment_cine_020"],
                },
                "qa_snapshot_050": {
                    "arguments": {
                        "input_nifti": "@node.reconstruct_grappa_005.reconstructed_nifti",
                        "seg_dir": "@node.segment_cine_020.pred_dir",
                        "title": "Cardiac cine – segmentation overlay",
                        "output_subdir": "qa/cardiac",
                    },
                    "depends_on": ["segment_cine_020"],
                },
            }
        )
        patched = _apply_node_updates(nodes=patched, updates=updates)
        recon_node = PlanNode(
            node_id="reconstruct_grappa_005",
            tool_name="reconstruct_grappa",
            stage="reconstruct",
            arguments={"h5_path": str(h5_path), "output_subdir": "grappa"},
            required=True,
            depends_on=["identify_sequences_001"],
            label="Reconstruct raw k-space/H5 cine to NIfTI",
        )
        out_nodes: List[PlanNode] = []
        inserted = False
        for n in patched:
            out_nodes.append(n)
            if str(n.node_id) == "identify_sequences_001":
                out_nodes.append(recon_node)
                inserted = True
        if not inserted:
            out_nodes = [
                PlanNode(
                    node_id="identify_sequences_001",
                    tool_name="identify_sequences",
                    stage="identify",
                    arguments={
                        "dicom_case_dir": identify_case_arg,
                        "convert_to_nifti": True,
                        "output_subdir": "ingest",
                    },
                    required=True,
                    depends_on=[],
                    label="Identify cardiac cine sequence",
                ),
                recon_node,
                *out_nodes,
            ]
        patched = out_nodes
        notes.append(f"H5 input detected ({h5_path}): identify_sequences kept as first stage, then reconstruct_grappa → segment_cardiac_cine.")
    else:
        patched = _drop_nodes_and_repair_dependencies(nodes=nodes, drop_ids={"reconstruct_grappa_005"})
        updates.update(
            {
                "segment_cine_020": {
                    "arguments": {"cine_path": "CINE", "output_subdir": "segmentation/cardiac_cine"},
                    "depends_on": ["identify_sequences_001"],
                },
                "classify_cine_030": {
                    "required": classify_required,
                    "arguments": {
                        "cine_path": "CINE",
                        "seg_path": "@node.segment_cine_020.seg_path",
                        "seg_dir": "@node.segment_cine_020.pred_dir",
                        "output_subdir": "classification/cardiac_cine",
                    },
                    "depends_on": ["segment_cine_020"],
                },
                "qa_snapshot_050": {
                    "arguments": {
                        "input_nifti": "@auto",
                        "seg_dir": "@node.segment_cine_020.pred_dir",
                        "title": "Cardiac cine – segmentation overlay",
                        "output_subdir": "qa/cardiac",
                    },
                    "depends_on": ["segment_cine_020"],
                },
            }
        )
        patched = _apply_node_updates(nodes=patched, updates=updates)

    if bool(intent.get("wants_report")):
        package_deps = ["segment_cine_020"]
        if classify_required:
            package_deps.append("classify_cine_030")
        if features_required:
            package_deps.append("extract_features_040")
        updates["package_evidence_060"] = {"depends_on": package_deps}
        patched = _apply_node_updates(nodes=patched, updates=updates)
    else:
        patched = _apply_node_updates(nodes=patched, updates=updates)
        patched = _drop_nodes_and_repair_dependencies(
            nodes=patched,
            drop_ids={"package_evidence_060", "generate_report_070"},
        )
    return patched, notes, []


def _patch_template_nodes(
    *,
    nodes: List[PlanNode],
    domain_name: str,
    request_type: str,
    goal: str,
    intent: Dict[str, bool],
    modalities: Dict[str, Optional[bool]],
    optional_overrides: Dict[str, bool],
    case_scan: Dict[str, Any],
    case_ref: Path,
    llm_mode: str,
    server_cfg: Optional[VLLMServerConfig],
    api_model: Optional[str],
    api_base_url: Optional[str],
) -> Tuple[List[PlanNode], List[str], List[str]]:
    req = _normalize_request_type(request_type) or "full_pipeline"
    dom = str(domain_name or "").strip().lower()

    if req == "qa":
        return _patch_qa_template_nodes(
            nodes=nodes,
            goal=goal,
            domain_name=dom,
            llm_mode=llm_mode,
            server_cfg=server_cfg,
            api_model=api_model,
            api_base_url=api_base_url,
        )
    if req == "custom_analysis":
        return _patch_custom_analysis_template_nodes(nodes=nodes, goal=goal, domain_name=dom)
    if req == "denoise":
        return _patch_denoise_template_nodes(nodes=nodes, goal=goal, case_scan=case_scan)
    if req == "super_resolution":
        return _patch_super_resolution_template_nodes(nodes=nodes, goal=goal, case_scan=case_scan)
    if req == "raw_recon":
        return _patch_raw_recon_template_nodes(nodes=nodes, goal=goal, domain_name=dom, case_ref=case_ref)

    if dom == "prostate":
        return _patch_prostate_template_nodes(
            nodes=nodes,
            intent=intent,
            modalities=modalities,
            optional_overrides=optional_overrides,
        )
    if dom == "brain":
        return _patch_brain_template_nodes(
            nodes=nodes,
            intent=intent,
            optional_overrides=optional_overrides,
        )
    if dom == "cardiac":
        return _patch_cardiac_template_nodes(
            nodes=nodes,
            intent=intent,
            optional_overrides=optional_overrides,
            case_ref=case_ref,
        )

    return nodes, [f"Unsupported domain for planner DAG: {dom}"], []


def _build_qa_plan_nodes(
    *,
    goal: str,
    domain_name: str,
    llm_mode: str,
    server_cfg: Optional[VLLMServerConfig],
    api_model: Optional[str],
    api_base_url: Optional[str],
) -> Tuple[List[PlanNode], List[str]]:
    rag_args: Dict[str, Any] = {
        "question": str(goal or "").strip(),
        "case_state_path": "@runtime.case_state_path",
        "output_subdir": "qa",
        "domain": str(domain_name or ""),
    }
    mode = str(llm_mode or "").strip().lower()
    if mode in {"server", "openai", "anthropic", "gemini"}:
        rag_args["llm_mode"] = mode
        if mode == "server":
            rag_args["server_base_url"] = str((server_cfg.base_url if server_cfg else "") or "http://127.0.0.1:8000/v1")
            rag_args["server_model"] = str((server_cfg.model if server_cfg else "") or "")
        elif mode == "openai":
            rag_args["api_model"] = str(api_model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini")
            if str(api_base_url or "").strip():
                rag_args["api_base_url"] = str(api_base_url).strip()
        elif mode == "anthropic":
            rag_args["api_model"] = str(api_model or os.environ.get("ANTHROPIC_MODEL") or "claude-3-7-sonnet-20250219")
        elif mode == "gemini":
            rag_args["api_model"] = str(api_model or os.environ.get("GEMINI_MODEL") or "gemini-1.5-pro")

    nodes = [
        PlanNode(
            node_id="identify_sequences_001",
            tool_name="identify_sequences",
            stage="identify",
            arguments={
                "dicom_case_dir": "@case.input",
                "convert_to_nifti": False,
                "require_pydicom": True,
                "deep_dump": True,
                "output_subdir": "ingest",
            },
            required=True,
            depends_on=[],
            label="Identify sequences and parse headers",
        ),
        PlanNode(
            node_id="rag_search_020",
            tool_name="rag_search",
            stage="qa",
            arguments=rag_args,
            required=True,
            depends_on=["identify_sequences_001"],
            label="Answer metadata question from local case outputs",
        ),
    ]
    notes = ["QA/read-metadata intent detected: planned minimal DAG identify_sequences -> rag_search."]
    return nodes, notes


def _build_custom_analysis_plan_nodes(*, goal: str, domain_name: str) -> Tuple[List[PlanNode], List[str]]:
    # Use repr() to safely escape any quotes / special chars in the goal text
    # so the generated heredoc Python script never has broken string literals.
    safe_goal = repr(str(goal or ""))
    cmd = (
        'python - <<\'PY\'\n'
        'from pathlib import Path\n'
        f'Path(\'analysis_request.txt\').write_text({safe_goal}, encoding=\'utf-8\')\n'
        'print(\'custom_analysis scaffold prepared: analysis_request.txt\')\n'
        'PY'
    )
    nodes = [
        PlanNode(
            node_id="identify_sequences_001",
            tool_name="identify_sequences",
            stage="identify",
            arguments={"dicom_case_dir": "@case.input", "convert_to_nifti": True, "output_subdir": "ingest"},
            required=True,
            depends_on=[],
            label="Identify sequences for custom analysis context",
        ),
        PlanNode(
            node_id="sandbox_exec_020",
            tool_name="sandbox_exec",
            stage="custom_analysis",
            arguments={
                "cmd": cmd,
                "output_subdir": "sandbox/custom_analysis",
                "capture_globs": ["*.txt", "*.json"],
                "max_output_chars": 4000,
            },
            required=True,
            depends_on=["identify_sequences_001"],
            label=f"Run sandbox custom analysis scaffold ({domain_name})",
        ),
    ]
    notes = ["Custom-analysis intent detected: planned identify_sequences -> sandbox_exec scaffold DAG."]
    return nodes, notes


def _goal_requests_comparison_png(goal: str) -> bool:
    txt = str(goal or "").lower()
    if not txt:
        return False
    keys = (
        "before/after",
        "before after",
        "comparison",
        "compare",
        "png",
        "sandbox_exec",
    )
    return any(k in txt for k in keys)


def _infer_half_spacing_from_nifti(path_like: str) -> Optional[List[float]]:
    p = Path(str(path_like or "")).expanduser()
    if not p.exists() or not p.is_file():
        return None
    try:
        import nibabel as nib  # type: ignore
    except Exception:
        return None
    try:
        img = nib.load(str(p))
        zooms = list(img.header.get_zooms())
    except Exception:
        return None

    spacing: List[float] = []
    for i in range(min(3, len(zooms))):
        try:
            spacing.append(max(1e-6, float(zooms[i]) / 2.0))
        except Exception:
            spacing.append(0.5)
    while len(spacing) < 3:
        spacing.append(0.5)
    return spacing[:3]


def _build_denoise_plan_nodes(
    *,
    goal: str,
    case_scan: Dict[str, Any],
) -> Tuple[List[PlanNode], List[str], List[str]]:
    notes: List[str] = []
    blocking: List[str] = []

    input_nifti: Optional[str] = None
    goal_nifti = _extract_abs_paths_by_suffix(goal, (".nii.gz", ".nii"))
    if goal_nifti:
        input_nifti = goal_nifti[0]
    else:
        scan_nifti = case_scan.get("nifti_files")
        if isinstance(scan_nifti, list):
            for p in scan_nifti:
                sp = str(p or "").strip()
                if not sp:
                    continue
                input_nifti = sp
                break
        if input_nifti:
            notes.append(f"Denoise input inferred from case scan: {input_nifti}")

    # If no NIfTI found but DICOM files exist, prepend an identify_sequences
    # (DICOM-to-NIfTI) step so the denoise tool receives a NIfTI volume.
    prepend_ingest = False
    if not input_nifti:
        dicom_count = int(case_scan.get("dicom_files") or 0)
        if dicom_count > 0:
            prepend_ingest = True
            notes.append(
                f"No NIfTI files found but {dicom_count} DICOM file(s) detected; "
                "prepending identify_sequences to convert DICOM → NIfTI before denoising."
            )
        else:
            blocking.append("denoise request requires an input NIfTI path (.nii/.nii.gz) or a DICOM case directory.")
            return [], notes, blocking

    nodes: List[PlanNode] = []

    if prepend_ingest:
        nodes.append(
            PlanNode(
                node_id="identify_sequences_001",
                tool_name="identify_sequences",
                stage="identify",
                arguments={"dicom_case_dir": "@case.input", "convert_to_nifti": True, "output_subdir": "ingest"},
                required=True,
                depends_on=[],
                label="Convert DICOM to NIfTI before denoising",
            )
        )
        # Use @auto so that cerebellum._normalize_step_args auto-discovers the
        # first NIfTI produced by the ingest step (via seq_paths or file glob).
        denoise_input = "@auto"
        denoise_deps = ["identify_sequences_001"]
    else:
        denoise_input = str(input_nifti)
        denoise_deps = []

    nodes.append(
        PlanNode(
            node_id="denoise_bm3d_010",
            tool_name="denoise_image_bm3d",
            stage="denoise",
            arguments={
                "input_nifti": denoise_input,
                "output_subdir": "denoise",
            },
            required=True,
            depends_on=denoise_deps,
            label="Denoise input volume with BM3D",
        )
    )

    compare_required = _goal_requests_comparison_png(goal)
    nodes.append(
        PlanNode(
            node_id="compare_denoise_020",
            tool_name="compare_nifti_slices",
            stage="compare",
            arguments={
                "image_a": "@node.denoise_bm3d_010.input_nifti",
                "image_b": "@node.denoise_bm3d_010.denoised_nifti",
                "label_a": "Before (original)",
                "label_b": "After (BM3D denoised)",
                "output_subdir": "compare/denoise",
            },
            required=bool(compare_required),
            depends_on=["denoise_bm3d_010"],
            label="Generate before/after denoise comparison PNG",
        )
    )
    notes.append("Denoise request detected: planned denoise_image_bm3d with optional comparison PNG generation.")
    return nodes, notes, blocking


def _build_super_resolution_plan_nodes(
    *,
    goal: str,
    case_scan: Dict[str, Any],
) -> Tuple[List[PlanNode], List[str], List[str]]:
    notes: List[str] = []
    blocking: List[str] = []

    input_nifti: Optional[str] = None
    goal_nifti = _extract_abs_paths_by_suffix(goal, (".nii.gz", ".nii"))
    if goal_nifti:
        input_nifti = goal_nifti[0]
    else:
        scan_nifti = case_scan.get("nifti_files")
        if isinstance(scan_nifti, list):
            for p in scan_nifti:
                sp = str(p or "").strip()
                if not sp:
                    continue
                input_nifti = sp
                break
        if input_nifti:
            notes.append(f"Super-resolution input inferred from case scan: {input_nifti}")

    # If no NIfTI found but DICOM files exist, prepend an identify_sequences
    # (DICOM-to-NIfTI) step so the resample tool receives a NIfTI volume.
    prepend_ingest = False
    if not input_nifti:
        dicom_count = int(case_scan.get("dicom_files") or 0)
        if dicom_count > 0:
            prepend_ingest = True
            notes.append(
                f"No NIfTI files found but {dicom_count} DICOM file(s) detected; "
                "prepending identify_sequences to convert DICOM → NIfTI before super-resolution."
            )
        else:
            blocking.append("super_resolution request requires an input NIfTI path (.nii/.nii.gz) or a DICOM case directory.")
            return [], notes, blocking

    nodes: List[PlanNode] = []

    if prepend_ingest:
        nodes.append(
            PlanNode(
                node_id="identify_sequences_001",
                tool_name="identify_sequences",
                stage="identify",
                arguments={"dicom_case_dir": "@case.input", "convert_to_nifti": True, "output_subdir": "ingest"},
                required=True,
                depends_on=[],
                label="Convert DICOM to NIfTI before super-resolution",
            )
        )
        resample_input = "@auto"
        resample_deps = ["identify_sequences_001"]
    else:
        resample_input = str(input_nifti)
        resample_deps = []

    target_spacing = _infer_half_spacing_from_nifti(str(input_nifti)) if input_nifti else None
    if target_spacing is None:
        target_spacing = [0.5, 0.5, 0.5]
        notes.append("Could not read input spacing at planning time; defaulting target_spacing to [0.5, 0.5, 0.5].")

    nodes.append(
        PlanNode(
            node_id="resample_sr_010",
            tool_name="resample_image",
            stage="super_resolution",
            arguments={
                "input_nifti": resample_input,
                "target_spacing": [float(x) for x in target_spacing],
                "interpolation": "bspline",
                "output_subdir": "resample",
            },
            required=True,
            depends_on=resample_deps,
            label="Upsample input volume with classical resampling (2x spacing reduction)",
        )
    )

    compare_required = _goal_requests_comparison_png(goal)
    nodes.append(
        PlanNode(
            node_id="compare_superres_020",
            tool_name="compare_nifti_slices",
            stage="compare",
            arguments={
                "image_a": "@node.resample_sr_010.input_nifti",
                "image_b": "@node.resample_sr_010.resampled_nifti",
                "label_a": "Before (original)",
                "label_b": "After (2x super-resolution)",
                "output_subdir": "compare/super_resolution",
            },
            required=bool(compare_required),
            depends_on=["resample_sr_010"],
            label="Generate before/after super-resolution comparison PNG",
        )
    )
    notes.append("Super-resolution request detected: planned resample_image with optional comparison PNG generation.")
    return nodes, notes, blocking


def _build_raw_recon_plan_nodes(
    *,
    goal: str,
    domain_name: str,
    case_ref: Path,
) -> Tuple[List[PlanNode], List[str], List[str]]:
    notes: List[str] = []
    blocking: List[str] = []
    dom = str(domain_name or "").strip().lower()
    if dom != "cardiac":
        blocking.append(f"raw_recon is currently supported for cardiac domain only (got domain={dom or 'unknown'}).")
        return [], notes, blocking

    h5_path: Optional[str] = None
    goal_h5 = _extract_abs_paths_by_suffix(goal, (".h5",))
    if goal_h5:
        h5_path = goal_h5[0]
    else:
        h5_path = _find_case_file_by_suffix(case_ref, (".h5",))
        if h5_path:
            notes.append(f"Raw reconstruction input inferred from case directory: {h5_path}")

    if not h5_path:
        blocking.append("raw_recon request requires an HDF5 k-space path (.h5).")
        return [], notes, blocking

    nodes: List[PlanNode] = [
        PlanNode(
            node_id="reconstruct_grappa_010",
            tool_name="reconstruct_grappa",
            stage="reconstruct",
            arguments={
                "h5_path": str(h5_path),
                "output_subdir": "grappa",
            },
            required=True,
            depends_on=[],
            label="Reconstruct raw k-space with GRAPPA",
        ),
        PlanNode(
            node_id="qa_snapshot_020",
            tool_name="generate_qa_snapshot",
            stage="qa",
            arguments={
                "input_nifti": "@node.reconstruct_grappa_010.reconstructed_nifti",
                "title": "GRAPPA reconstruction",
                "output_subdir": "qa/grappa",
            },
            required=True,
            depends_on=["reconstruct_grappa_010"],
            label="Generate QA snapshot of reconstructed volume",
        ),
    ]
    notes.append(
        "Raw reconstruction request detected: planned reconstruct_grappa -> generate_qa_snapshot."
    )
    return nodes, notes, blocking


def _build_prostate_plan_nodes(
    *,
    intent: Dict[str, bool],
    modalities: Dict[str, Optional[bool]],
    optional_overrides: Dict[str, bool],
) -> Tuple[List[PlanNode], List[str]]:
    notes: List[str] = []
    nodes: List[PlanNode] = []
    missing_modalities: List[str] = []
    omitted_tools: List[str] = []

    adc_required = True
    if modalities.get("ADC") is False:
        adc_required = False
        missing_modalities.append("ADC")
        omitted_tools.append("register_adc")
        notes.append("ADC was not detected in case scan; ADC registration marked optional.")

    dwi_required = True
    if modalities.get("DWI") is False:
        dwi_required = False
        missing_modalities.append("DWI")
        omitted_tools.append("register_dwi")
        notes.append("DWI/high-b was not detected in case scan; DWI registration marked optional.")

    lesion_required = True
    lesion_deps = ["segment_prostate_030"]
    if adc_required:
        lesion_deps.append("register_adc_010")
    if dwi_required:
        lesion_deps.append("register_dwi_020")

    nodes.append(
        PlanNode(
            node_id="identify_sequences_001",
            tool_name="identify_sequences",
            stage="identify",
            arguments={"dicom_case_dir": "@case.input", "convert_to_nifti": True, "output_subdir": "ingest"},
            required=True,
            depends_on=[],
            label="Identify and map prostate sequences",
        )
    )
    nodes.append(
        PlanNode(
            node_id="register_adc_010",
            tool_name="register_to_reference",
            stage="register",
            arguments={
                "fixed": "T2w",
                "moving": "ADC",
                "output_subdir": "registration/ADC",
                "split_by_bvalue": False,
                "save_png_qc": True,
            },
            required=adc_required,
            depends_on=["identify_sequences_001"],
            label="Register ADC to T2w",
        )
    )
    nodes.append(
        PlanNode(
            node_id="register_dwi_020",
            tool_name="register_to_reference",
            stage="register",
            arguments={
                "fixed": "T2w",
                "moving": "DWI",
                "output_subdir": "registration/DWI",
                "split_by_bvalue": True,
                "save_png_qc": True,
            },
            required=dwi_required,
            depends_on=["identify_sequences_001"],
            label="Register DWI/high-b to T2w",
        )
    )
    nodes.append(
        PlanNode(
            node_id="segment_prostate_030",
            tool_name="segment_prostate",
            stage="segment",
            arguments={"t2w_ref": "T2w", "output_subdir": "segmentation"},
            required=True,
            depends_on=["identify_sequences_001"],
            label="Segment whole-gland prostate mask",
        )
    )

    features_required = bool(intent.get("wants_features"))
    if optional_overrides.get("extract_roi_features"):
        features_required = False
    nodes.append(
        PlanNode(
            node_id="extract_features_040",
            tool_name="extract_roi_features",
            stage="extract",
            arguments={"output_subdir": "features"},
            required=features_required,
            depends_on=["segment_prostate_030"],
            label="Extract ROI/radiomics features",
        )
    )

    nodes.append(
        PlanNode(
            node_id="detect_lesions_050",
            tool_name="detect_lesion_candidates",
            stage="lesion",
            arguments={"output_subdir": "lesion"},
            required=lesion_required,
            depends_on=lesion_deps,
            label="Detect lesion candidates",
        )
    )

    if bool(intent.get("wants_report")):
        package_deps = ["segment_prostate_030"]
        if features_required:
            package_deps.append("extract_features_040")
        if lesion_required:
            package_deps.append("detect_lesions_050")
        nodes.append(
            PlanNode(
                node_id="package_evidence_060",
                tool_name="package_vlm_evidence",
                stage="package",
                arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "vlm"},
                required=True,
                depends_on=package_deps,
                label="Package VLM evidence",
            )
        )
        report_args: Dict[str, Any] = {
            "case_state_path": "@runtime.case_state_path",
            "output_subdir": "report",
            "domain": "prostate",
        }
        # Track 2: inject degradation_context when modalities are missing.
        if missing_modalities or omitted_tools:
            report_args["degradation_context"] = {
                "missing_modalities": missing_modalities,
                "omitted_tools": omitted_tools,
                "degraded_mode": True,
                "advisory": (
                    f"Limited exam: {', '.join(missing_modalities)} modality not acquired. "
                    f"Omitted: {', '.join(omitted_tools)}. "
                    "Report must state these limitations explicitly."
                ),
            }
        nodes.append(
            PlanNode(
                node_id="generate_report_070",
                tool_name="generate_report",
                stage="report",
                arguments=report_args,
                required=True,
                depends_on=["package_evidence_060"],
                label="Generate prostate report",
            )
        )

    return nodes, notes


def _build_brain_plan_nodes(
    *,
    intent: Dict[str, bool],
    optional_overrides: Dict[str, bool],
) -> Tuple[List[PlanNode], List[str]]:
    nodes: List[PlanNode] = []
    notes: List[str] = []

    reg_optional = not bool(intent.get("wants_lesion"))
    classify_required = bool(intent.get("wants_classification"))
    nodes.append(
        PlanNode(
            node_id="identify_sequences_001",
            tool_name="identify_sequences",
            stage="identify",
            arguments={"dicom_case_dir": "@case.input", "convert_to_nifti": True, "output_subdir": "ingest"},
            required=True,
            depends_on=[],
            label="Identify brain sequences",
        )
    )
    nodes.append(
        PlanNode(
            node_id="register_t2_010",
            tool_name="register_to_reference",
            stage="register",
            arguments={"fixed": "T1c", "moving": "T2", "output_subdir": "registration/T2", "save_png_qc": True},
            required=(not reg_optional),
            depends_on=["identify_sequences_001"],
            label="Register T2 to T1c",
        )
    )
    nodes.append(
        PlanNode(
            node_id="register_flair_020",
            tool_name="register_to_reference",
            stage="register",
            arguments={"fixed": "T1c", "moving": "FLAIR", "output_subdir": "registration/FLAIR", "save_png_qc": True},
            required=(not reg_optional),
            depends_on=["identify_sequences_001"],
            label="Register FLAIR to T1c",
        )
    )
    nodes.append(
        PlanNode(
            node_id="segment_brats_030",
            tool_name="brats_mri_segmentation",
            stage="segment",
            arguments={
                "t1c_path": "T1c",
                "t1_path": "T1",
                "t2_path": "T2",
                "flair_path": "FLAIR",
                "output_subdir": "segmentation/brats_mri_segmentation",
            },
            required=True,
            depends_on=["identify_sequences_001"],
            label="Segment brain tumor subregions (WT/TC/ET)",
        )
    )

    features_required = bool(intent.get("wants_features"))
    if classify_required:
        features_required = True
    if optional_overrides.get("extract_roi_features") and (not classify_required):
        features_required = False
    nodes.append(
        PlanNode(
            node_id="extract_features_040",
            tool_name="extract_roi_features",
            stage="extract",
            arguments={"output_subdir": "features", "roi_interpretation": "binary_masks"},
            required=features_required,
            depends_on=["segment_brats_030"],
            label="Extract tumor ROI features",
        )
    )
    nodes.append(
        PlanNode(
            node_id="classify_brain_grade_050",
            tool_name="classify_brain_glioma_grade",
            stage="classify",
            arguments={
                "feature_table_path": "@node.extract_features_040.feature_table_path",
                "output_subdir": "classification/brain_grade",
            },
            required=classify_required,
            depends_on=["extract_features_040"],
            label="Classify glioma grade (HGG/LGG) from tumor features",
        )
    )

    if bool(intent.get("wants_report")):
        package_deps = ["segment_brats_030"]
        if features_required:
            package_deps.append("extract_features_040")
        if classify_required:
            package_deps.append("classify_brain_grade_050")
        nodes.append(
            PlanNode(
                node_id="package_evidence_060",
                tool_name="package_vlm_evidence",
                stage="package",
                arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "vlm"},
                required=True,
                depends_on=package_deps,
                label="Package VLM evidence",
            )
        )
        nodes.append(
            PlanNode(
                node_id="generate_report_070",
                tool_name="generate_report",
                stage="report",
                arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "report", "domain": "brain"},
                required=True,
                depends_on=["package_evidence_060"],
                label="Generate brain tumor report",
            )
        )
    return nodes, notes


def _build_cardiac_plan_nodes(
    *,
    intent: Dict[str, bool],
    optional_overrides: Dict[str, bool],
    case_ref: Optional[Path] = None,
    request_type: str = "",
) -> Tuple[List[PlanNode], List[str]]:
    nodes: List[PlanNode] = []
    notes: List[str] = []

    classify_required = bool(intent.get("wants_classification"))
    if bool(intent.get("wants_report")):
        classify_required = True

    # ---- Detect raw k-space H5 input ----
    h5_path: Optional[str] = None
    if case_ref is not None:
        try:
            h5_path = _find_case_file_by_suffix(case_ref, (".h5",))
        except Exception:
            pass

    resolved_req = _normalize_request_type(request_type) or "full_pipeline"

    # ---- SHORT DAG: raw_recon only (reconstruct → qa_snapshot) ----
    if resolved_req == "raw_recon":
        if not h5_path:
            notes.append("raw_recon requested but no .h5 file found in case directory.")
            return [], notes
        nodes.append(
            PlanNode(
                node_id="reconstruct_grappa_010",
                tool_name="reconstruct_grappa",
                stage="reconstruct",
                arguments={
                    "h5_path": str(h5_path),
                    "output_subdir": "grappa",
                },
                required=True,
                depends_on=[],
                label="Reconstruct raw k-space with GRAPPA",
            )
        )
        nodes.append(
            PlanNode(
                node_id="qa_snapshot_020",
                tool_name="generate_qa_snapshot",
                stage="qa",
                arguments={
                    "input_nifti": "@node.reconstruct_grappa_010.reconstructed_nifti",
                    "title": "GRAPPA reconstruction",
                    "output_subdir": "qa/grappa",
                },
                required=True,
                depends_on=["reconstruct_grappa_010"],
                label="Generate QA snapshot of reconstructed volume",
            )
        )
        notes.append("raw_recon request: planned reconstruct_grappa → generate_qa_snapshot.")
        return nodes, notes

    # ---- FULL / CLASSIFY / SEGMENT pipeline ----
    identify_case_arg = "@case.input"
    if h5_path and case_ref is not None and case_ref.is_file():
        identify_case_arg = str(case_ref.parent)
        notes.append("Cardiac file-based H5 case: identify_sequences input remapped to parent directory.")
    nodes.append(
        PlanNode(
            node_id="identify_sequences_001",
            tool_name="identify_sequences",
            stage="identify",
            arguments={"dicom_case_dir": identify_case_arg, "convert_to_nifti": True, "output_subdir": "ingest"},
            required=True,
            depends_on=[],
            label="Identify cardiac cine sequence",
        )
    )
    if h5_path:
        nodes.append(
            PlanNode(
                node_id="reconstruct_grappa_005",
                tool_name="reconstruct_grappa",
                stage="reconstruct",
                arguments={
                    "h5_path": str(h5_path),
                    "output_subdir": "grappa",
                },
                required=True,
                depends_on=["identify_sequences_001"],
                label="Convert H5/raw cardiac input to NIfTI",
            )
        )
        nodes.append(
            PlanNode(
                node_id="segment_cine_020",
                tool_name="segment_cardiac_cine",
                stage="segment",
                arguments={
                    "cine_path": "@node.reconstruct_grappa_005.reconstructed_nifti",
                    "output_subdir": "segmentation/cardiac_cine",
                },
                required=True,
                depends_on=["reconstruct_grappa_005"],
                label="Segment cardiac cine (from reconstructed NIfTI)",
            )
        )
        notes.append(f"H5 input detected ({h5_path}): identify_sequences first, then reconstruct_grappa → segment_cardiac_cine.")
    else:
        nodes.append(
            PlanNode(
                node_id="segment_cine_020",
                tool_name="segment_cardiac_cine",
                stage="segment",
                arguments={"cine_path": "CINE", "output_subdir": "segmentation/cardiac_cine"},
                required=True,
                depends_on=["identify_sequences_001"],
                label="Segment cardiac cine",
            )
        )
    # When H5: cine_path for classify should point to the GRAPPA output
    classify_cine_path = (
        "@node.reconstruct_grappa_005.reconstructed_nifti" if h5_path
        else "CINE"
    )
    nodes.append(
        PlanNode(
            node_id="classify_cine_030",
            tool_name="classify_cardiac_cine_disease",
            stage="report",
            arguments={
                "seg_path": "@node.segment_cine_020.seg_path",
                "seg_dir": "@node.segment_cine_020.pred_dir",
                "cine_path": classify_cine_path,
                "output_subdir": "classification/cardiac_cine",
            },
            required=classify_required,
            depends_on=["segment_cine_020"],
            label="Classify cardiac disease group",
        )
    )

    features_required = bool(intent.get("wants_features"))
    if optional_overrides.get("extract_roi_features"):
        features_required = False
    nodes.append(
        PlanNode(
            node_id="extract_features_040",
            tool_name="extract_roi_features",
            stage="extract",
            arguments={"output_subdir": "features", "roi_interpretation": "cardiac"},
            required=features_required,
            depends_on=["segment_cine_020"],
            label="Extract cardiac ROI features",
        )
    )

    # ---- QA snapshot with segmentation mask overlay ----
    qa_input_nifti = (
        "@node.reconstruct_grappa_005.reconstructed_nifti" if h5_path
        else "@auto"
    )
    nodes.append(
        PlanNode(
            node_id="qa_snapshot_050",
            tool_name="generate_qa_snapshot",
            stage="qa",
            arguments={
                "input_nifti": qa_input_nifti,
                "seg_dir": "@node.segment_cine_020.pred_dir",
                "title": "Cardiac cine – segmentation overlay",
                "output_subdir": "qa/cardiac",
            },
            required=True,
            depends_on=["segment_cine_020"],
            label="QA snapshot with segmentation overlay",
        )
    )

    if bool(intent.get("wants_report")):
        package_deps = ["segment_cine_020"]
        if classify_required:
            package_deps.append("classify_cine_030")
        if features_required:
            package_deps.append("extract_features_040")
        nodes.append(
            PlanNode(
                node_id="package_evidence_060",
                tool_name="package_vlm_evidence",
                stage="package",
                arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "vlm"},
                required=True,
                depends_on=package_deps,
                label="Package VLM evidence",
            )
        )
        nodes.append(
            PlanNode(
                node_id="generate_report_070",
                tool_name="generate_report",
                stage="report",
                arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "report", "domain": "cardiac"},
                required=True,
                depends_on=["package_evidence_060"],
                label="Generate cardiac report",
            )
        )
    return nodes, notes


def plan_agent_dag(
    *,
    goal: str,
    domain: str,
    case_ref: str,
    case_id: Optional[str] = None,
    request_type: Optional[str] = None,
    llm_mode: str = "server",
    max_new_tokens: int = 2048,
    server_cfg: Optional[VLLMServerConfig] = None,
    api_model: Optional[str] = None,
    api_base_url: Optional[str] = None,
    workspace_root: Optional[str] = None,
    runs_root: Optional[str] = None,
    allow_external_model_roots: Optional[List[str]] = None,
    template_variant: str = "default",
    explicit_input_aliases: Optional[Dict[str, Any]] = None,
    planning_mode: str = "template",
    sketch_fallback_to_template: bool = False,
    planner_task_id: Optional[str] = None,
) -> AgentPlanDAG:
    """
    Planning-only API for LangGraph Brain.
    This function uses LLM + skills + case scan to build AgentPlanDAG and does NOT execute any tool.
    """
    domain_cfg = get_domain_config(str(domain))
    case_ref_path = _abs_path_no_resolve(str(case_ref))
    case_id_val = str(case_id or case_ref_path.name or "case").strip()
    requested_workflow_type = _extract_requested_workflow_type(goal, request_type)
    workspace_root_path = Path(str(workspace_root or project_root())).expanduser().resolve()
    runs_root_path = Path(str(runs_root or (workspace_root_path / "runs"))).expanduser().resolve()

    reg = build_registry()
    tools_list = reg.list_specs()
    tool_index = _compact_tool_index(tools_list)
    tool_names = sorted([str(t.get("name")) for t in tools_list if isinstance(t, dict) and t.get("name")])

    skills_path = project_root() / "config" / "skills.json"
    skills_registry = SkillRegistry.load(skills_path)
    skills_summary = skills_registry.to_prompt(set_name=None, tags=[domain_cfg.name], max_items=8)
    skills_guidance = skills_registry.to_guidance(set_name=None, tags=[domain_cfg.name], max_items=8)

    case_scan = _scan_case_ref(case_ref_path)
    directory_evidence = _collect_directory_evidence(case_ref_path)
    modality_hints_raw = _infer_modalities_from_case(domain_cfg.name, case_scan)
    directory_modality_candidates = _directory_modality_candidates(domain_cfg.name, directory_evidence)
    explicit_alias_candidates, explicit_alias_notes = _explicit_input_alias_candidates(
        domain_name=domain_cfg.name,
        explicit_input_aliases=explicit_input_aliases,
    )
    modality_candidates = _merge_modality_candidate_maps(explicit_alias_candidates, directory_modality_candidates)
    modality_hints, evidence_override_notes = _merge_modality_hints_with_directory_evidence(
        modality_hints=modality_hints_raw,
        modality_candidates=modality_candidates,
    )
    planner_directory_evidence = _compact_directory_evidence_for_prompt(
        directory_evidence=directory_evidence,
        modality_candidates=modality_candidates,
    )

    llm = _build_planner_llm(
        llm_mode=str(llm_mode),
        max_new_tokens=int(max_new_tokens),
        server_cfg=server_cfg,
        api_model=api_model,
        api_base_url=api_base_url,
    )
    resolved_request_type = requested_workflow_type or "full_pipeline"
    planning_mode_norm = str(planning_mode or "template").strip().lower() or "template"
    if planning_mode_norm in {"default", "dag"}:
        planning_mode_norm = "template"
    if planning_mode_norm in {"constrained_sketch"}:
        planning_mode_norm = "sketch"
    if planning_mode_norm not in {"template", "sketch"}:
        raise ValueError(f"Unsupported planning_mode: {planning_mode!r}")

    plan_input_text = (
        f"Goal: {str(goal)}\n"
        f"Domain: {domain_cfg.name}\n"
        f"case_id: {case_id_val}\n"
        f"case_ref: {str(case_ref_path)}\n"
        "You are planning ONLY. Do not execute tools.\n\n"
        "Conflict resolution rule:\n"
        "- If modality_hints_raw says a modality is missing, but directory_evidence lists concrete candidate files for that modality,\n"
        "  treat that modality as available and continue planning.\n"
        "- When such evidence exists, prefer binding the concrete absolute file path in tool arguments (example: cine_path=/abs/path/patient061_4d.nii.gz).\n\n"
        "tool_index:\n"
        f"{json.dumps(tool_index, ensure_ascii=False, indent=2)}\n\n"
        "skills_summary:\n"
        f"{skills_summary}\n\n"
        "skills_guidance:\n"
        f"{json.dumps(skills_guidance, ensure_ascii=False, indent=2)}\n\n"
        "case_scan_and_evidence:\n"
        f"{json.dumps({'scan': case_scan, 'modality_hints_raw': modality_hints_raw, 'modality_hints_effective': modality_hints, 'directory_evidence': planner_directory_evidence}, ensure_ascii=False, indent=2)}\n"
    )

    plan_text = ""
    planner_error: Optional[str] = None
    planner_raw_messages: List[Dict[str, Any]] = []
    sketch_parsed_obj: Optional[Dict[str, Any]] = None
    sketch_model: Optional[ConstrainedPlanSketch] = None
    sketch_compile_diagnostics: Optional[Dict[str, Any]] = None
    planner_artifacts: Dict[str, str] = {}
    planner_metadata: Dict[str, Any] = {"planning_mode": planning_mode_norm}

    if planning_mode_norm == "sketch":
        sketch_allowed_tools_prompt: List[str] = []
        try:
            pre_template = load_plan_template(
                domain=domain_cfg.name,
                request_type=resolved_request_type,
                variant=str(template_variant or "default"),
            )
            draft_nodes = [n.model_copy(deep=True) for n in (pre_template.nodes or [])]
            draft_intent = _infer_goal_intent(str(goal), domain_cfg.name, "", [])
            draft_optional_overrides = {
                "extract_roi_features": False,
                "detect_lesion_candidates": False,
            }
            draft_nodes, _draft_patch_notes, _draft_patch_blocking = _patch_template_nodes(
                nodes=draft_nodes,
                domain_name=domain_cfg.name,
                request_type=resolved_request_type,
                goal=str(goal or ""),
                intent=draft_intent,
                modalities=modality_hints,
                optional_overrides=draft_optional_overrides,
                case_scan=case_scan,
                case_ref=case_ref_path,
                llm_mode=str(llm_mode),
                server_cfg=server_cfg,
                api_model=api_model,
                api_base_url=api_base_url,
            )
            if not _draft_patch_blocking:
                draft_nodes, _ = _prune_nodes_by_constraints(nodes=draft_nodes, intent=draft_intent)
                draft_nodes, _ = _prune_nodes_by_request_type(nodes=draft_nodes, request_type=resolved_request_type)
                draft_nodes, _ = _force_required_for_terminal_request_type(nodes=draft_nodes, request_type=resolved_request_type)
            sketch_allowed_tools_prompt = _allowed_tools_from_nodes(draft_nodes)
        except Exception:
            sketch_allowed_tools_prompt = []

        sketch_task_name = str(planner_task_id or resolved_request_type or "task").strip() or "task"
        sketch_system_prompt = build_sketch_planner_system_prompt(
            domain=domain_cfg.name,
            request_type=resolved_request_type,
            allowed_tools=(sketch_allowed_tools_prompt or tool_names),
        )
        sketch_planner = SketchPlannerSubagent(llm=llm, system_prompt=sketch_system_prompt)
        sketch_input_text = (
            f"Task: {sketch_task_name}\n"
            + plan_input_text
            + "\nOutput target: Constrained Plan Sketch JSON.\n"
            + "Reminder: Use symbolic refs only in step.inputs; do not use absolute file paths.\n"
            + f"Allowed tools (task/domain scope): {json.dumps(sketch_allowed_tools_prompt or tool_names, ensure_ascii=False)}\n"
        )
        try:
            plan_res = sketch_planner.run(
                {
                    "plan_input_text": sketch_input_text,
                    "guided_json_schema": constrained_plan_sketch_guided_json_schema(),
                }
            )
            planner_raw_messages = [m for m in (plan_res.get("messages") or []) if isinstance(m, dict)]
            plan_text = str(plan_res.get("sketch_text") or "").strip()
            if plan_res.get("error"):
                planner_error = str(plan_res.get("error"))
        except Exception as e:
            planner_error = repr(e)
            plan_text = ""

        parsed = _try_parse_json(plan_text)
        if isinstance(parsed, dict):
            sketch_parsed_obj = parsed
            try:
                sketch_model = ConstrainedPlanSketch.model_validate(parsed)
            except Exception as e:
                sketch_compile_diagnostics = {
                    "compile_ok": False,
                    "schema_valid": False,
                    "errors": [f"Sketch schema validation failed: {repr(e)}"],
                    "warnings": [],
                }
        else:
            if plan_text:
                sketch_compile_diagnostics = {
                    "compile_ok": False,
                    "schema_valid": False,
                    "errors": ["Sketch JSON parse failed (expected JSON object)."],
                    "warnings": [],
                }
    else:
        planner = PlannerSubagent(llm=llm)
        try:
            plan_res = planner.run({"plan_input_text": plan_input_text})
            planner_raw_messages = [m for m in (plan_res.get("messages") or []) if isinstance(m, dict)]
            plan_text = str(plan_res.get("plan_text") or "").strip()
        except Exception as e:
            planner_error = repr(e)
            plan_text = ""

    if planning_mode_norm == "sketch" and sketch_model is not None:
        tool_mentions = [str(s.tool) for s in (sketch_model.steps or []) if str(s.tool).strip()]
    else:
        tool_mentions = _extract_tool_mentions(plan_text, tool_names)
    intent = _infer_goal_intent(str(goal), domain_cfg.name, plan_text, tool_mentions)
    optional_overrides = {
        "extract_roi_features": (
            False if planning_mode_norm == "sketch" else _tool_line_is_optional(plan_text, "extract_roi_features")
        ),
        "detect_lesion_candidates": (
            False if planning_mode_norm == "sketch" else _tool_line_is_optional(plan_text, "detect_lesion_candidates")
        ),
    }
    extra_blocking_reasons: List[str] = []
    template_id = ""
    template_version = ""
    template_hash = ""
    template_source = ""
    nodes: List[PlanNode] = []
    notes: List[str] = []

    try:
        template = load_plan_template(
            domain=domain_cfg.name,
            request_type=resolved_request_type,
            variant=str(template_variant or "default"),
        )
        template_id = str(template.template_id or "").strip()
        template_version = str(template.template_version or "").strip()
        template_hash = str(template.template_hash or "").strip()
        template_source = str(template.source_path or "").strip()
        nodes = [n.model_copy(deep=True) for n in (template.nodes or [])]
        notes.extend([str(x) for x in (template.notes or []) if str(x).strip()])
        notes.append(
            f"Loaded template skeleton: id={template_id}, version={template_version}, hash={template_hash[:12]}"
        )
    except Exception as e:
        err = f"template load failed ({domain_cfg.name}/{resolved_request_type}): {repr(e)}"
        extra_blocking_reasons.append(err)
        notes.append(err)

    if not extra_blocking_reasons:
        nodes, patch_notes, patch_blocking = _patch_template_nodes(
            nodes=nodes,
            domain_name=domain_cfg.name,
            request_type=resolved_request_type,
            goal=str(goal or ""),
            intent=intent,
            modalities=modality_hints,
            optional_overrides=optional_overrides,
            case_scan=case_scan,
            case_ref=case_ref_path,
            llm_mode=str(llm_mode),
            server_cfg=server_cfg,
            api_model=api_model,
            api_base_url=api_base_url,
        )
        notes.extend([str(x) for x in (patch_notes or []) if str(x).strip()])
        extra_blocking_reasons.extend([str(x) for x in (patch_blocking or []) if str(x).strip()])

    if planning_mode_norm == "sketch":
        reg_override_pairs = []
        override_notes = ["Skipped planner registration overrides for sketch mode (compiler follows sketch + blueprints)."]
    else:
        reg_override_pairs = _extract_registration_override_pairs(
            text=f"{str(goal or '')}\n{plan_text}",
            domain_name=domain_cfg.name,
        )
    if planning_mode_norm != "sketch" and str(template_variant or "").strip().lower() in {"notoken", "no_token"}:
        override_notes = ["Skipped planner registration overrides for noToken template variant (preserve explicit contract inputs)."]
    elif planning_mode_norm != "sketch":
        nodes, override_notes = _apply_registration_overrides(
            nodes=nodes,
            domain_name=domain_cfg.name,
            pairs=reg_override_pairs,
        )
    nodes, prune_notes = _prune_nodes_by_constraints(nodes=nodes, intent=intent)
    nodes, req_prune_notes = _prune_nodes_by_request_type(nodes=nodes, request_type=resolved_request_type)
    nodes, force_req_notes = _force_required_for_terminal_request_type(nodes=nodes, request_type=resolved_request_type)
    nodes, explicit_alias_inject_notes, explicit_alias_missing = _inject_explicit_input_aliases(
        nodes=nodes,
        explicit_input_aliases=explicit_input_aliases,
    )
    nodes, direct_bind_notes = _bind_direct_modality_paths(
        nodes=nodes,
        domain_name=domain_cfg.name,
        modality_candidates=modality_candidates,
    )

    sketch_compile_notes: List[str] = []
    if planning_mode_norm == "sketch":
        if sketch_model is None:
            if sketch_compile_diagnostics is None:
                sketch_compile_diagnostics = {
                    "compile_ok": False,
                    "schema_valid": False,
                    "errors": ["Sketch planner did not return a valid parsed sketch object."],
                    "warnings": [],
                }
            extra_blocking_reasons.append("sketch compile failed")
        else:
            compile_res = compile_constrained_plan_sketch(
                sketch=sketch_model,
                blueprints=nodes,
                allowed_tools=_allowed_tools_from_nodes(nodes),
                require_blueprint_match=True,
                ensure_required_blueprint_nodes=True,
                case_ref_is_file=bool(case_ref_path.is_file()),
            )
            sketch_compile_diagnostics = dict(compile_res.diagnostics or {})
            planner_metadata["sketch_compile_ok"] = bool(compile_res.ok)
            planner_metadata["sketch_step_count"] = len(sketch_model.steps or [])
            planner_metadata["sketch_task"] = str(sketch_model.task or "")
            if compile_res.ok:
                nodes = [n.model_copy(deep=True) for n in (compile_res.nodes or [])]
                sketch_compile_notes.append(
                    "BCER-Sketch compile succeeded: "
                    f"explicit={int((sketch_compile_diagnostics.get('compiled_summary') or {}).get('explicit_sketch_nodes', 0))}, "
                    f"autofilled={int((sketch_compile_diagnostics.get('compiled_summary') or {}).get('autofilled_nodes', 0))}, "
                    f"total={int((sketch_compile_diagnostics.get('compiled_summary') or {}).get('node_count', len(nodes)))}"
                )
            else:
                if not bool(sketch_fallback_to_template):
                    extra_blocking_reasons.append("sketch compile failed")
                    for _msg in [str(x) for x in (sketch_compile_diagnostics.get("errors") or []) if str(x).strip()][:5]:
                        extra_blocking_reasons.append(f"sketch compile error: {_msg}")
                    sketch_compile_notes.append("BCER-Sketch compile failed; no template fallback (strict mode).")
                else:
                    sketch_compile_notes.append("BCER-Sketch compile failed; falling back to template DAG (fallback enabled).")
                    planner_metadata["sketch_fallback_used"] = True

    blocking_reasons: List[str] = []
    for key in _missing_modalities_for_planner(
        domain_name=domain_cfg.name,
        request_type=resolved_request_type,
        modality_hints=modality_hints,
        nodes=nodes,
    ):
        blocking_reasons.append(f"missing modality: {key}")
    for moving_tok, fixed_tok in reg_override_pairs:
        if modality_hints.get(moving_tok) is False:
            blocking_reasons.append(f"registration override requests moving={moving_tok}, but it was not found")
        if modality_hints.get(fixed_tok) is False:
            blocking_reasons.append(f"registration override requests fixed={fixed_tok}, but it was not found")
    for miss_key in (explicit_alias_missing or []):
        blocking_reasons.append(f"missing explicit input alias: {miss_key}")
    blocking_reasons.extend([str(x) for x in (extra_blocking_reasons or []) if str(x).strip()])
    dedup_blocking: List[str] = []
    seen_blocking: set[str] = set()
    for reason in blocking_reasons:
        rr = str(reason or "").strip()
        if not rr or rr in seen_blocking:
            continue
        seen_blocking.add(rr)
        dedup_blocking.append(rr)
    blocking_reasons = dedup_blocking
    planner_status = "blocked" if blocking_reasons else "ready"

    plan_notes = list(notes)
    plan_notes.extend(explicit_alias_notes)
    plan_notes.extend(evidence_override_notes)
    plan_notes.extend(override_notes)
    plan_notes.extend(prune_notes)
    plan_notes.extend(req_prune_notes)
    plan_notes.extend(force_req_notes)
    plan_notes.extend(explicit_alias_inject_notes)
    plan_notes.extend(direct_bind_notes)
    plan_notes.extend(sketch_compile_notes)
    if resolved_request_type:
        plan_notes.append(f"Requested workflow type (detected): {resolved_request_type}")
    plan_notes.append(f"Planner planning_mode: {planning_mode_norm}")
    if str(template_variant or "").strip():
        plan_notes.append(f"Planner template_variant: {str(template_variant).strip()}")
    if template_id:
        plan_notes.append(f"Planner template_id: {template_id}")
    if template_version:
        plan_notes.append(f"Planner template_version: {template_version}")
    if template_hash:
        plan_notes.append(f"Planner template_hash: {template_hash}")
    if template_source:
        plan_notes.append(f"Planner template_source: {template_source}")
    if blocking_reasons:
        plan_notes.append("Planner blocking reasons: " + "; ".join(blocking_reasons))
    plan_notes.append(f"Planner model mode: {llm_mode}")
    if plan_text:
        plan_notes.append("Planner output (raw):\n" + plan_text)
    if planner_error:
        plan_notes.append("Planner fallback due to error: " + planner_error)
    if tool_mentions:
        plan_notes.append("Planner-mentioned tools: " + ", ".join(tool_mentions))
    if planning_mode_norm == "sketch" and sketch_compile_diagnostics is not None:
        ok_flag = bool(sketch_compile_diagnostics.get("compile_ok"))
        err_n = len([x for x in (sketch_compile_diagnostics.get("errors") or []) if str(x).strip()])
        warn_n = len([x for x in (sketch_compile_diagnostics.get("warnings") or []) if str(x).strip()])
        plan_notes.append(f"Sketch compiler diagnostics: compile_ok={ok_flag}, errors={err_n}, warnings={warn_n}")
        if err_n:
            for msg in [str(x) for x in (sketch_compile_diagnostics.get("errors") or []) if str(x).strip()][:8]:
                plan_notes.append("Sketch compiler error: " + msg)
        if warn_n:
            for msg in [str(x) for x in (sketch_compile_diagnostics.get("warnings") or []) if str(x).strip()][:8]:
                plan_notes.append("Sketch compiler warning: " + msg)
    plan_notes.append("This DAG was generated in planning-only mode (no tool execution).")
    natural_response = _planner_natural_response(
        llm=llm,
        status=planner_status,
        domain_name=domain_cfg.name,
        request_type=resolved_request_type,
        case_ref=case_ref_path,
        goal=str(goal or ""),
        blocking_reasons=blocking_reasons,
        modality_hints=modality_hints,
    )

    scoped_external_roots: List[str] = []
    seen_external: set[str] = set()
    for root in [
        *[str(x) for x in (allow_external_model_roots or []) if str(x).strip()],
        *_planner_default_external_model_roots(),
    ]:
        rr = str(root or "").strip()
        if not rr or rr in seen_external:
            continue
        seen_external.add(rr)
        scoped_external_roots.append(rr)

    plan_id = f"dag_{case_id_val}_{uuid.uuid4().hex[:8]}"
    dag_obj = AgentPlanDAG(
        plan_id=plan_id,
        template_id=template_id,
        template_version=template_version,
        template_hash=template_hash,
        goal=str(goal),
        case_scope=CaseScope(
            domain=domain_cfg.name,  # type: ignore[arg-type]
            case_id=case_id_val,
            case_ref=str(case_ref_path),
            workspace_root=str(workspace_root_path),
            runs_root=str(runs_root_path),
            allow_external_model_roots=scoped_external_roots,
        ),
        planner_status=str(planner_status),
        natural_language_response=natural_response,
        blocking_reasons=blocking_reasons,
        requested_request_type=resolved_request_type,
        policy=DagPolicy(skip_optional_by_default=True, stop_on_required_failure=True, max_attempts_default=2),
        nodes=([] if planner_status == "blocked" else nodes),
        notes=plan_notes,
        planner_metadata=planner_metadata,
        planner_artifacts={},
    )
    if planning_mode_norm == "sketch":
        planner_artifacts = _persist_planner_artifacts(
            runs_root=runs_root_path,
            case_id=case_id_val,
            planner_kind="bcer_sketch",
            raw_output=plan_text,
            parsed_obj=sketch_parsed_obj,
            compile_diagnostics=sketch_compile_diagnostics,
            compiled_dag=(dag_obj if str(dag_obj.planner_status) != "blocked" else None),
            prompt_messages=planner_raw_messages,
            planner_error=planner_error,
        )
        dag_obj.planner_artifacts = dict(planner_artifacts or {})
        if dag_obj.planner_artifacts:
            for k, v in dag_obj.planner_artifacts.items():
                dag_obj.notes.append(f"Planner artifact [{k}]: {v}")
    return dag_obj


def _sanitize_server_model(model: str, base_url: str) -> str:
    model_s = str(model or "").strip()
    base = str(base_url or "").strip().lower()
    is_local = ("127.0.0.1" in base) or ("localhost" in base)
    default_model = os.environ.get("MEDGEMMA_SERVER_MODEL", "Qwen/Qwen3-VL-30B-A3B-Thinking")
    if not model_s:
        return default_model
    model_l = model_s.lower()
    if is_local and (model_l.startswith("gpt-") or model_l.startswith("o1") or model_l.startswith("o3")):
        return default_model
    if model_l in ("qwen-vl", "qwen_vl", "qwenvl", "vl", "qwen"):
        return default_model
    return model_s


def _apply_report_llm_args(run: LangGraphRunCtx, args: Dict[str, Any]) -> Dict[str, Any]:
    if run.llm_mode == "server":
        args["llm_mode"] = "server"
        if run.server_cfg is not None:
            args.setdefault("server_base_url", str(run.server_cfg.base_url))
            args.setdefault("server_model", str(run.server_cfg.model))
            args.setdefault("max_tokens", int(run.server_cfg.max_tokens or 900))
            args.setdefault("temperature", float(run.server_cfg.temperature or 0.0))
    elif run.llm_mode in ("openai", "anthropic", "gemini"):
        args["llm_mode"] = run.llm_mode
        if run.api_model:
            args.setdefault("api_model", str(run.api_model))
        if run.llm_mode == "openai" and run.api_base_url:
            args.setdefault("api_base_url", str(run.api_base_url))
    return args


@dataclass(frozen=True)
class LangGraphRunCtx:
    runs_root: Path
    case_id: str
    run_id: str
    llm_mode: str
    autofix_mode: str  # force|coach|off
    finalize_with_llm: bool
    dicom_case_dir: Optional[str]
    max_steps: int

    dispatcher: ToolDispatcher
    state: CaseState
    tool_ctx: Any

    tools_list: List[Dict[str, Any]]
    tool_index: Dict[str, Any]
    skills_registry: SkillRegistry
    skills_guidance: List[Dict[str, Any]]

    llm: Any
    server_cfg: Optional[VLLMServerConfig]
    api_model: Optional[str]
    api_base_url: Optional[str]
    planner: PlannerSubagent
    reflector: ReflectorSubagent
    policy: PolicySubagent
    alignment_gate: AlignmentGateSubagent
    distortion_gate: DistortionGateSubagent
    domain: DomainConfig
    lesion_threshold_default: float
    lesion_min_volume_cc: float
    lesion_threshold_adaptive: bool
    lesion_threshold_candidates: List[float]
    lesion_threshold_max_retries: int
    distortion_gate_override: Optional[str]  # None|"off"|"forced_on"  (Track 2 arms)
    progress_emit: Optional[Callable[[Dict[str, Any]], None]]


class LGState(TypedDict, total=False):
    step: int
    messages: List[Dict[str, Any]]
    last_model_raw: str
    last_model_parsed: Optional[Dict[str, Any]]
    last_error: Optional[str]
    last_tool_name: Optional[str]
    last_tool_ok: Optional[bool]
    plan: Optional[Dict[str, Any]]
    reflection: Optional[str]
    pending_calls: List[Dict[str, Any]]
    force_tool_exec: Optional[bool]
    done: bool
    final_report_path: Optional[str]

def _write_agent_trace(run_dir: Path, rec: Dict[str, Any]) -> None:
    p = run_dir / "agent_trace.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_plan_trace(run_dir: Path, rec: Dict[str, Any]) -> None:
    p = run_dir / "plan_trace.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_gate_trace(run_dir: Path, rec: Dict[str, Any]) -> None:
    """Append a JSONL record to gate_trace.jsonl for Track 2 auditing."""
    p = run_dir / "gate_trace.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_message_trace(run_dir: Path, rec: Dict[str, Any]) -> None:
    p = run_dir / "conversation_trace.md"
    ts = rec.get("ts") or ""
    tag = rec.get("tag") or ""
    step = rec.get("step")
    header = f"## {ts} | {tag} | step {step}\n"
    lines: List[str] = [header]

    def _strip_think(text: str) -> str:
        if not isinstance(text, str):
            return str(text)
        t = text.replace("<think>", "").replace("</think>", "")
        return t.strip()

    def _compact_model_output(tag_name: str, text: str) -> str:
        cleaned = _strip_think(text)
        if tag_name == "policy_response":
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                cleaned = cleaned[start : end + 1].strip()
        elif tag_name == "plan_response":
            lines = [ln.rstrip() for ln in cleaned.splitlines()]
            numbered = [ln for ln in lines if ln.lstrip().startswith(tuple(f"{i}." for i in range(1, 10)))]
            if numbered:
                cleaned = "\n".join(numbered)
            else:
                cleaned = "\n".join(lines[:10]).strip()
        elif tag_name == "reflect":
            lines = [ln for ln in cleaned.splitlines() if ln.strip()]
            cleaned = "\n".join(lines[:6]).strip()
        if len(cleaned) > 1500:
            cleaned = cleaned[:1500].rstrip() + "\n[TRUNCATED]"
        return cleaned

    if "messages" in rec:
        lines.append("**messages (system prompt omitted after first)**\n")
        for i, m in enumerate(rec.get("messages") or []):
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            # Skip repeated system prompts to reduce noise.
            if role == "system":
                continue
            lines.append(f"- [{i}] {role}: {content}\n")

    if "reflection" in rec:
        lines.append("**reflection**\n")
        lines.append(f"> {_compact_model_output(tag, str(rec.get('reflection')))}\n")

    if "model_output" in rec:
        lines.append("**model_output**\n")
        lines.append(f"> {_compact_model_output(tag, str(rec.get('model_output')))}\n")

    with p.open("a", encoding="utf-8") as f:
        f.write("".join(lines) + "\n")


def _emit_progress(run: LangGraphRunCtx, payload: Dict[str, Any]) -> None:
    emit = run.progress_emit
    if emit is None:
        return
    try:
        emit(payload)
    except Exception:
        # Best-effort telemetry only; never affect agent execution.
        pass


def _alignment_gate_paths(run_dir: Path) -> Dict[str, Path]:
    ctx_dir = run_dir / "artifacts" / "context"
    qc_dir = run_dir / "artifacts" / "alignment_qc"
    return {
        "qc_summary": qc_dir / "alignment_qc_summary.json",
        "gate_decision": ctx_dir / "alignment_gate.json",
    }


def _distortion_gate_paths(run_dir: Path) -> Dict[str, Path]:
    ctx_dir = run_dir / "artifacts" / "context"
    return {
        "gate_decision": ctx_dir / "distortion_gate.json",
    }


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_case_state(case_state_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(case_state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tool_attempt_status(case_state_path: Path, tool_name: str) -> Tuple[bool, bool]:
    cs = _read_case_state(case_state_path)
    stage_outputs = cs.get("stage_outputs", {}) if isinstance(cs, dict) else {}
    attempted = False
    succeeded = False
    if not isinstance(stage_outputs, dict):
        return attempted, succeeded
    for _stage, tools in stage_outputs.items():
        if not isinstance(tools, dict):
            continue
        recs = tools.get(tool_name)
        if not isinstance(recs, list):
            continue
        for r in recs:
            if not isinstance(r, dict):
                continue
            attempted = True
            if r.get("ok") is True:
                succeeded = True
                return attempted, succeeded
    return attempted, succeeded


def _has_adc_registration_artifact(run_art_dir: Path) -> bool:
    try:
        for p in run_art_dir.glob("**/*resampled_to_fixed.nii.gz"):
            s = str(p).lower().replace("\\", "/")
            n = p.name.lower()
            if "/registration/adc/" in s and ("adc" in n or "moving_all_resampled_to_fixed" in n):
                return True
            if "/registered/" in s and ("adc" in n or "moving_all_resampled_to_fixed" in n):
                return True
    except Exception:
        return False
    return False


def _has_dwi_registration_artifact(run_art_dir: Path) -> bool:
    try:
        for p in run_art_dir.glob("**/*resampled_to_fixed.nii.gz"):
            s = str(p).lower().replace("\\", "/")
            n = p.name.lower()
            if "/registration/adc/" in s:
                continue
            looks_dwi = (
                "moving_high_b_resampled_to_fixed" in n
                or "moving_dwi_resampled_to_fixed" in n
                or "moving_mid_b_resampled_to_fixed" in n
                or "moving_low_b_resampled_to_fixed" in n
                or ("moving_b" in n and "resampled_to_fixed" in n)
            )
            if not looks_dwi:
                continue
            if "/registration/dwi/" in s or "/registered/" in s:
                return True
    except Exception:
        return False
    return False


def _infer_series_name_from_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(str(path))
    name = p.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return name


def _build_alignment_text_summary(ident: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    series_inventory_path = ident.get("series_inventory_path")
    dicom_meta_path = ident.get("dicom_meta_path")
    mapping = ident.get("mapping", {}) if isinstance(ident, dict) else {}

    meta = _load_json(Path(str(dicom_meta_path))) if dicom_meta_path else None
    series_meta = meta.get("series_meta", {}) if isinstance(meta, dict) else {}

    def _header_for_series(series_name: Optional[str]) -> Dict[str, Any]:
        if not series_name:
            return {}
        info = series_meta.get(series_name, {})
        header = info.get("header", {}) if isinstance(info, dict) else {}
        if not isinstance(header, dict):
            return {}
        keys = [
            "SeriesDescription",
            "Modality",
            "PixelSpacing",
            "SliceThickness",
            "SpacingBetweenSlices",
            "Rows",
            "Columns",
            "ImageOrientationPatient",
            "ImagePositionPatient",
        ]
        return {k: header.get(k) for k in keys if k in header}

    t2_name = _infer_series_name_from_path(mapping.get("T2w"))
    adc_name = _infer_series_name_from_path(mapping.get("ADC"))
    dwi_name = _infer_series_name_from_path(mapping.get("DWI"))

    if t2_name:
        summary["T2w"] = {"series_name": t2_name, "dicom_header": _header_for_series(t2_name)}
    if adc_name:
        summary["ADC"] = {"series_name": adc_name, "dicom_header": _header_for_series(adc_name)}
    if dwi_name:
        base = dwi_name.split("_b")[0] if "_b" in dwi_name else dwi_name
        summary["DWI"] = {"series_name": dwi_name, "dicom_header": _header_for_series(base)}

    if series_inventory_path and Path(str(series_inventory_path)).exists():
        inv = _load_json(Path(str(series_inventory_path))) or {}
        series_list = inv.get("series", []) if isinstance(inv, dict) else []
        if isinstance(series_list, list):
            summary["series_inventory"] = [
                {
                    "series_name": s.get("series_name"),
                    "sequence_guess": s.get("sequence_guess"),
                    "nifti_spacing": s.get("nifti_spacing"),
                    "n_dicoms": s.get("n_dicoms"),
                }
                for s in series_list
                if isinstance(s, dict)
            ]

    return summary


def _select_alignment_images(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(images, list):
        return []
    wanted = []
    # Prefer central slices first.
    def _pick(name: str, kind: str):
        for it in images:
            if not isinstance(it, dict):
                continue
            if str(it.get("name")) == name and str(it.get("kind")) == kind:
                wanted.append(it)
                return
    _pick("T2w", "central_slice")
    _pick("ADC", "central_slice")
    _pick("DWI", "central_slice")
    _pick("ADC", "edges_overlay")
    _pick("DWI", "edges_overlay")
    # If we have fewer than 3, fall back to any central slices.
    if len(wanted) < 3:
        for it in images:
            if not isinstance(it, dict):
                continue
            if str(it.get("kind")) == "central_slice" and it not in wanted:
                wanted.append(it)
            if len(wanted) >= 5:
                break
    return wanted


def _load_alignment_gate_decision(run_dir: Path) -> Optional[Dict[str, Any]]:
    paths = _alignment_gate_paths(run_dir)
    if paths["gate_decision"].exists():
        return _load_json(paths["gate_decision"])
    return None


def _write_alignment_gate_decision(run_dir: Path, decision: Dict[str, Any]) -> None:
    paths = _alignment_gate_paths(run_dir)
    paths["gate_decision"].parent.mkdir(parents=True, exist_ok=True)
    paths["gate_decision"].write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sequence_key_from_path(path: str) -> str:
    name_l = str(path).lower()
    if "adc" in name_l:
        return "ADC"
    return "DWI"


def _maybe_build_alignment_qc_call(run: LangGraphRunCtx, ident: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    mapping = ident.get("mapping", {}) if isinstance(ident, dict) else {}
    fixed = mapping.get("T2w")
    adc = mapping.get("ADC")
    dwi = mapping.get("DWI")
    moving_paths = []
    if adc:
        moving_paths.append({"name": "ADC", "path": adc})
    if dwi:
        moving_paths.append({"name": "DWI", "path": dwi})
    if not fixed or not moving_paths:
        return None
    return {
        "action": "tool_call",
        "tool_name": "alignment_qc",
        "arguments": {"fixed_path": fixed, "moving_paths": moving_paths, "output_subdir": "alignment_qc", "save_edges": True},
        "stage": "misc",
    }


def _alignment_gate_for_action(
    run: LangGraphRunCtx,
    action_obj: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[str]]:
    # Only gate register_to_reference for prostate.
    if getattr(run.domain, "name", None) != "prostate":
        return action_obj, None
    if action_obj.get("action") != "tool_call" or action_obj.get("tool_name") != "register_to_reference":
        return action_obj, None

    ident = _latest_tool_data(run.tool_ctx.case_state_path, "identify", "identify_sequences") or {}
    paths = _alignment_gate_paths(run.tool_ctx.run_dir)
    gate = _load_alignment_gate_decision(run.tool_ctx.run_dir)

    # Ensure we have QC images first.
    if gate is None and not paths["qc_summary"].exists():
        qc_call = _maybe_build_alignment_qc_call(run, ident)
        if qc_call:
            return qc_call, "alignment_qc_first"

    if gate is None and paths["qc_summary"].exists():
        qc = _load_json(paths["qc_summary"]) or {}
        text_summary = _build_alignment_text_summary(ident)
        images = _select_alignment_images(qc.get("images", []))
        payload = {
            "case_id": run.case_id,
            "run_id": run.run_id,
            "text_summary": text_summary,
            "qc_summary": qc.get("summaries", {}),
            "image_manifest": images,
        }
        res = run.alignment_gate.decide(payload=payload, image_paths=images, use_schema=False)
        decision = res.parsed or {
            "decisions": {},
            "overall": {"need_register_any": True, "confidence": 0.1, "notes": "alignment gate parse failed"},
        }
        _write_alignment_gate_decision(run.tool_ctx.run_dir, decision)
        gate = decision

    if not isinstance(gate, dict):
        return action_obj, None

    moving = action_obj.get("arguments", {}).get("moving")
    seq_key = _sequence_key_from_path(str(moving or ""))
    dec = gate.get("decisions", {}).get(seq_key, {}) if isinstance(gate.get("decisions"), dict) else {}
    need_register = bool(dec.get("need_register", True))

    if not need_register:
        # Replace with materialize_registration (pass-through) to avoid resample.
        mat_args = dict(action_obj.get("arguments", {}) or {})
        mat_args.pop("split_by_bvalue", None)
        mat_args["label"] = "adc" if seq_key == "ADC" else "high_b"
        _write_gate_trace(run.tool_ctx.run_dir, {
            "ts": datetime.utcnow().isoformat() + "Z",
            "gate": "alignment",
            "sequence": seq_key,
            "decision": "skip_register",
            "confidence": dec.get("confidence"),
            "reasons": dec.get("reasons", dec.get("notes")),
            "action_taken": "materialize_registration",
        })
        return {
            "action": "tool_call",
            "tool_name": "materialize_registration",
            "arguments": mat_args,
            "stage": "register",
        }, "alignment_gate_skip_register"

    _write_gate_trace(run.tool_ctx.run_dir, {
        "ts": datetime.utcnow().isoformat() + "Z",
        "gate": "alignment",
        "sequence": seq_key,
        "decision": "allow_register",
        "confidence": dec.get("confidence"),
        "reasons": dec.get("reasons", dec.get("notes")),
        "action_taken": "register_to_reference",
    })
    return action_obj, None


def _alignment_gate_for_calls(
    run: LangGraphRunCtx,
    calls_obj: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[str]]:
    # Only gate for prostate and only when tool_calls is used.
    if getattr(run.domain, "name", None) != "prostate":
        return calls_obj, None
    if calls_obj.get("action") != "tool_calls":
        return calls_obj, None
    calls = calls_obj.get("calls")
    if not isinstance(calls, list) or not calls:
        return calls_obj, None

    # If any register_to_reference is present and we don't have QC/gate yet, run alignment_qc first.
    needs_register = any(isinstance(c, dict) and c.get("tool_name") == "register_to_reference" for c in calls)
    if not needs_register:
        return calls_obj, None

    paths = _alignment_gate_paths(run.tool_ctx.run_dir)
    gate = _load_alignment_gate_decision(run.tool_ctx.run_dir)
    if gate is None and not paths["qc_summary"].exists():
        ident = _latest_tool_data(run.tool_ctx.case_state_path, "identify", "identify_sequences") or {}
        qc_call = _maybe_build_alignment_qc_call(run, ident)
        if qc_call:
            return {"action": "tool_calls", "calls": [qc_call] + calls}, "alignment_qc_inserted"

    # If gate exists, replace register_to_reference with materialize_registration when allowed.
    if isinstance(gate, dict):
        new_calls: List[Dict[str, Any]] = []
        changed = False
        for c in calls:
            if not isinstance(c, dict) or c.get("tool_name") != "register_to_reference":
                new_calls.append(c)
                continue
            moving = c.get("arguments", {}).get("moving")
            seq_key = _sequence_key_from_path(str(moving or ""))
            dec = gate.get("decisions", {}).get(seq_key, {}) if isinstance(gate.get("decisions"), dict) else {}
            if dec.get("need_register") is False:
                mat_args = dict(c.get("arguments", {}) or {})
                mat_args.pop("split_by_bvalue", None)
                mat_args["label"] = "adc" if seq_key == "ADC" else "high_b"
                new_calls.append(
                    {
                        "tool_name": "materialize_registration",
                        "arguments": mat_args,
                        "stage": c.get("stage") or "register",
                    }
                )
                changed = True
            else:
                new_calls.append(c)
        if changed:
            return {"action": "tool_calls", "calls": new_calls}, "alignment_gate_skip_register"

    return calls_obj, None


def _load_distortion_gate_decision(run_dir: Path) -> Optional[Dict[str, Any]]:
    paths = _distortion_gate_paths(run_dir)
    if paths["gate_decision"].exists():
        return _load_json(paths["gate_decision"])
    return None


def _write_distortion_gate_decision(run_dir: Path, decision: Dict[str, Any]) -> None:
    paths = _distortion_gate_paths(run_dir)
    paths["gate_decision"].parent.mkdir(parents=True, exist_ok=True)
    paths["gate_decision"].write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _collect_distortion_gate_images(run_dir: Path) -> List[Dict[str, Any]]:
    art_dir = run_dir / "artifacts"
    patterns = [
        "registration/DWI/*edges_overlay*.png",
        "registration/ADC/*edges_overlay*.png",
        "registration/DWI/*central*.png",
        "registration/ADC/*central*.png",
        "alignment_qc/*edges_overlay*.png",
        "alignment_qc/*central*.png",
    ]
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for pat in patterns:
        for p in sorted(art_dir.glob(pat)):
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            label = p.name.replace("_", " ")
            out.append({"name": p.name, "label": label, "path": rp})
            if len(out) >= 10:
                return out
    return out


def _registration_summaries_for_distortion(case_state_path: Path) -> List[Dict[str, Any]]:
    cs = _read_case_state(case_state_path)
    stage_outputs = cs.get("stage_outputs", {}) if isinstance(cs, dict) else {}
    out: List[Dict[str, Any]] = []
    if not isinstance(stage_outputs, dict):
        return out
    for _stage, tools in stage_outputs.items():
        if not isinstance(tools, dict):
            continue
        recs = tools.get("register_to_reference")
        if not isinstance(recs, list):
            continue
        for r in recs:
            if not isinstance(r, dict):
                continue
            data = r.get("data") if isinstance(r.get("data"), dict) else {}
            if not isinstance(data, dict):
                continue
            out.append(
                {
                    "ok": bool(r.get("ok")),
                    "moving": data.get("moving"),
                    "fixed": data.get("fixed"),
                    "output_subdir": data.get("output_subdir"),
                    "qc_metrics": data.get("qc_metrics"),
                    "qc_pngs": data.get("qc_pngs"),
                }
            )
    return out[-6:]


def _should_consider_distortion_gate(run: LangGraphRunCtx, parsed_obj: Dict[str, Any]) -> bool:
    if getattr(run.domain, "name", None) != "prostate":
        return False
    action = str(parsed_obj.get("action") or "")
    downstream = {
        "extract_roi_features",
        "detect_lesion_candidates",
        "package_vlm_evidence",
        "generate_report",
    }
    if action == "tool_call":
        t = str(parsed_obj.get("tool_name") or "")
        return t in downstream
    if action == "tool_calls":
        calls = parsed_obj.get("calls")
        if not isinstance(calls, list):
            return False
        return any(isinstance(c, dict) and str(c.get("tool_name") or "") in downstream for c in calls)
    return False


def _distortion_gate_wants_correction(decision: Dict[str, Any]) -> bool:
    d = decision.get("decision") if isinstance(decision, dict) else None
    if not isinstance(d, dict):
        return False
    return bool(d.get("should_run_correction"))


def _distortion_gate_for_policy_action(
    run: LangGraphRunCtx,
    parsed_obj: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[str]]:
    if not _should_consider_distortion_gate(run, parsed_obj):
        return parsed_obj, None

    # Track 2 arm override: short-circuit the VLM gate entirely.
    override = getattr(run, "distortion_gate_override", None)
    if override == "off":
        _write_gate_trace(run.tool_ctx.run_dir, {
            "ts": datetime.utcnow().isoformat() + "Z",
            "gate": "distortion",
            "decision": "pass",
            "override": "off",
            "action_taken": "passthrough",
            "original_action": parsed_obj.get("tool_name") or parsed_obj.get("action"),
        })
        return parsed_obj, None
    if override == "forced_on":
        attempted, succeeded = _tool_attempt_status(Path(run.tool_ctx.case_state_path), "correct_prostate_distortion")
        if attempted or succeeded:
            return parsed_obj, None
        _write_gate_trace(run.tool_ctx.run_dir, {
            "ts": datetime.utcnow().isoformat() + "Z",
            "gate": "distortion",
            "decision": "correct",
            "override": "forced_on",
            "action_taken": "correct_prostate_distortion",
            "original_action": parsed_obj.get("tool_name") or parsed_obj.get("action"),
        })
        return {
            "action": "tool_call",
            "tool_name": "correct_prostate_distortion",
            "arguments": {
                "output_subdir": "distortion_correction",
                "strength": 0.3,
                "save_npz": True,
                "save_slices": True,
            },
            "stage": "register",
        }, "distortion_gate_forced_on"

    attempted, succeeded = _tool_attempt_status(Path(run.tool_ctx.case_state_path), "correct_prostate_distortion")
    if attempted or succeeded:
        return parsed_obj, None

    decision = _load_distortion_gate_decision(run.tool_ctx.run_dir)
    images: List[Dict[str, Any]] = []
    if decision is None:
        images = _collect_distortion_gate_images(run.tool_ctx.run_dir)
        ident = _latest_tool_data(run.tool_ctx.case_state_path, "identify", "identify_sequences") or {}
        payload = {
            "case_id": run.case_id,
            "run_id": run.run_id,
            "mapping": ident.get("mapping") if isinstance(ident, dict) else {},
            "registration_history": _registration_summaries_for_distortion(Path(run.tool_ctx.case_state_path)),
            "image_manifest": images,
            "instruction": "Decide if prostate distortion correction should run before lesion/reporting tools.",
        }
        if not images:
            decision = {
                "decision": {
                    "should_run_correction": False,
                    "confidence": 0.0,
                    "reasons": ["no_registration_qc_images_available"],
                },
                "overall": {"note": "distortion gate skipped due to missing QC images."},
            }
        else:
            res = run.distortion_gate.decide(payload=payload, image_paths=images, use_schema=False)
            decision = res.parsed or {
                "decision": {
                    "should_run_correction": False,
                    "confidence": 0.1,
                    "reasons": ["distortion_gate_parse_failed"],
                },
                "overall": {"note": str(res.error or "distortion gate parse failed")},
            }
        _write_distortion_gate_decision(run.tool_ctx.run_dir, decision)

    d_inner = decision.get("decision", {}) if isinstance(decision, dict) else {}
    wants_correction = _distortion_gate_wants_correction(decision)

    _write_gate_trace(run.tool_ctx.run_dir, {
        "ts": datetime.utcnow().isoformat() + "Z",
        "gate": "distortion",
        "decision": "correct" if wants_correction else "pass",
        "should_run_correction": wants_correction,
        "confidence": d_inner.get("confidence") if isinstance(d_inner, dict) else None,
        "reasons": d_inner.get("reasons") if isinstance(d_inner, dict) else None,
        "images_analyzed": len(images),
        "action_taken": "correct_prostate_distortion" if wants_correction else "passthrough",
        "original_action": parsed_obj.get("tool_name") or parsed_obj.get("action"),
    })

    if not isinstance(decision, dict) or not wants_correction:
        return parsed_obj, None

    return {
        "action": "tool_call",
        "tool_name": "correct_prostate_distortion",
        "arguments": {
            "output_subdir": "distortion_correction",
            "strength": 0.3,
            "save_npz": True,
            "save_slices": True,
        },
        "stage": "register",
    }, "distortion_gate_inserted_correction"


def _summarize_message_content(content: Any) -> Any:
    def _strip_named_block(text: str, label: str) -> str:
        marker = f"{label}:"
        if marker not in text:
            return text
        pre, rest = text.split(marker, 1)
        end_markers = ["\nstate_summary:", "\nmemory_digest:", "\nprior_plan", "\n## "]
        end_idx = min([i for i in (rest.find(m) for m in end_markers) if i >= 0], default=-1)
        tail = rest[end_idx:] if end_idx >= 0 else ""
        return pre + f"{marker} (omitted)\n" + tail

    def _strip_schema_blocks(text: str) -> str:
        lines = text.splitlines()
        out: List[str] = []
        skip = False
        for line in lines:
            stripped = line.strip()
            if not skip and (stripped.startswith("### input_schema") or stripped.startswith("### output_schema")):
                skip = True
                continue
            if skip:
                if stripped.startswith("## "):
                    skip = False
                    out.append(line)
                    continue
                if stripped.startswith("### ") and not (
                    stripped.startswith("### input_schema") or stripped.startswith("### output_schema")
                ):
                    skip = False
                    out.append(line)
                    continue
                if stripped.startswith("state_summary:") or stripped.startswith("memory_digest:") or stripped.startswith("prior_plan"):
                    skip = False
                    out.append(line)
                    continue
                continue
            out.append(line)
        return "\n".join(out)

    if isinstance(content, str):
        s = content.strip()
        if "tool_manifest_contents:" in s:
            try:
                pre, rest = s.split("tool_manifest_contents:", 1)
                end_markers = ["\nstate_summary:", "\nmemory_digest:", "\nprior_plan"]
                end_idx = min([i for i in (rest.find(m) for m in end_markers) if i >= 0], default=-1)
                if end_idx >= 0:
                    tail = rest[end_idx:]
                else:
                    tail = ""
                s = pre + "tool_manifest_contents: (omitted)\n" + tail
            except Exception:
                pass
        if "memory_digest:" in s:
            s = _strip_named_block(s, "memory_digest")
        if "prior_plan" in s:
            s = _strip_named_block(s, "prior_plan")
        if "### input_schema" in s or "### output_schema" in s:
            s = _strip_schema_blocks(s)
        if s.startswith("{") or s.startswith("["):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return {"_json": "omitted", "keys": sorted(list(obj.keys()))}
                if isinstance(obj, list):
                    return {"_json": "omitted", "type": "list", "len": len(obj)}
            except Exception:
                return "<json omitted>"
        return s
    if isinstance(content, list):
        out = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "image_url":
                out.append({"type": "image_url", "image_url": "<omitted>"})
            else:
                out.append(blk)
        return out
    return content


def _compact_plan_for_prompt(plan: Any) -> str:
    if not plan:
        return "(none)"
    plan_text = ""
    if isinstance(plan, dict):
        plan_text = str(plan.get("plan_text") or "")
    else:
        plan_text = str(plan)
    lines = [ln.strip() for ln in plan_text.splitlines() if ln.strip()]
    numbered = [ln for ln in lines if re.match(r"^\d+\.", ln)]
    if numbered:
        lines = numbered
    if not lines:
        return "(none)"
    compact = "\n".join(lines[:6])
    if len(lines) > 6:
        compact = compact + "\n[TRUNCATED]"
    return compact


def _truncate_text(text: str, *, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[TRUNCATED]"


def _compact_prompt_obj(obj: Any, *, max_chars: int) -> Any:
    compact = compact_for_memory(obj, max_chars=max_chars)
    if isinstance(compact, (dict, list, str, int, float, bool)) or compact is None:
        return compact
    return {"value": str(compact)}


def _should_run_plan_reflect(run: LangGraphRunCtx, st: LGState) -> bool:
    step = int(st.get("step", 0))
    if step == 0:
        return True
    if st.get("last_tool_ok") is False:
        return True
    if st.get("last_error"):
        return True
    return False


def _summarize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        out.append({"role": m.get("role"), "content": _summarize_message_content(m.get("content"))})
    return out


def _trim_messages_for_context(messages: List[Dict[str, Any]], *, max_chars: int = 48_000) -> List[Dict[str, Any]]:
    """
    Simple, model-agnostic context guardrail.
    vLLM errors can occur when input tokens alone exceed the context window; we keep messages bounded by characters.
    """
    if not messages:
        return messages
    # Keep the first system message and the tail.
    sys = messages[0:1]
    tail = messages[1:]
    # Hard cap tail length by dropping older messages.
    out = list(sys)
    cur = len(json.dumps(sys, ensure_ascii=False))
    # Take from the end
    for m in reversed(tail):
        add = len(json.dumps(m, ensure_ascii=False))
        if cur + add > max_chars:
            break
        out.append(m)
        cur += add
    # Restore original order: sys + kept tail
    kept_tail = list(reversed(out[1:]))
    return sys + kept_tail


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    # Best-effort extraction of outermost JSON object
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                frag = text[start : i + 1]
                try:
                    obj = json.loads(frag)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _planner_debug_dir(*, runs_root: Path, case_id: str, planner_kind: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return (runs_root / "_planner_debug" / str(case_id or "case") / f"{ts}_{planner_kind}_{uuid.uuid4().hex[:8]}").resolve()


def _persist_planner_artifacts(
    *,
    runs_root: Path,
    case_id: str,
    planner_kind: str,
    raw_output: str,
    parsed_obj: Optional[Dict[str, Any]] = None,
    compile_diagnostics: Optional[Dict[str, Any]] = None,
    compiled_dag: Optional[AgentPlanDAG] = None,
    prompt_messages: Optional[List[Dict[str, Any]]] = None,
    planner_error: Optional[str] = None,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        dbg_dir = _planner_debug_dir(runs_root=runs_root, case_id=case_id, planner_kind=planner_kind)
        dbg_dir.mkdir(parents=True, exist_ok=True)

        raw_path = dbg_dir / "planner_raw_output.txt"
        raw_path.write_text(str(raw_output or ""), encoding="utf-8")
        out["raw_output"] = str(raw_path)

        if parsed_obj is not None:
            parsed_path = dbg_dir / "planner_parsed.json"
            _write_json_file(parsed_path, parsed_obj)
            out["parsed_json"] = str(parsed_path)

        if compile_diagnostics is not None:
            diag_path = dbg_dir / "compile_diagnostics.json"
            _write_json_file(diag_path, compile_diagnostics)
            out["compile_diagnostics"] = str(diag_path)

        if compiled_dag is not None:
            dag_path = dbg_dir / "compiled_dag.json"
            _write_json_file(dag_path, compiled_dag.model_dump())
            out["compiled_dag"] = str(dag_path)

        if prompt_messages:
            msg_path = dbg_dir / "planner_messages.json"
            _write_json_file(msg_path, _summarize_messages(prompt_messages))
            out["planner_messages"] = str(msg_path)

        if str(planner_error or "").strip():
            err_path = dbg_dir / "planner_error.txt"
            err_path.write_text(str(planner_error or ""), encoding="utf-8")
            out["planner_error"] = str(err_path)
    except Exception:
        return out
    return out


def _allowed_tools_from_nodes(nodes: List[PlanNode]) -> List[str]:
    return sorted(
        {
            str(n.tool_name).strip()
            for n in (nodes or [])
            if isinstance(n, PlanNode) and str(getattr(n, "tool_name", "")).strip()
        }
    )


def _parse_threshold_candidates(text: str) -> List[float]:
    vals: List[float] = []
    for tok in str(text or "").split(","):
        s = tok.strip()
        if not s:
            continue
        try:
            v = float(s)
        except Exception:
            continue
        if 0.0 < v < 1.0:
            vals.append(v)
    uniq = sorted(set(vals), reverse=True)
    return uniq or [0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05]


def _build_adaptive_lesion_redetect_call(run: LangGraphRunCtx) -> Tuple[Optional[ToolCall], Optional[Dict[str, Any]]]:
    if getattr(run.domain, "name", None) != "prostate":
        return None, None
    if not bool(run.lesion_threshold_adaptive):
        return None, None
    case_state = _load_json(Path(run.tool_ctx.case_state_path)) or {}
    stage_outputs = case_state.get("stage_outputs", {}) if isinstance(case_state, dict) else {}
    lesion_stage = stage_outputs.get("lesion", {}) if isinstance(stage_outputs, dict) else {}
    recs = lesion_stage.get("detect_lesion_candidates", []) if isinstance(lesion_stage, dict) else []
    if not isinstance(recs, list) or not recs:
        return None, None
    if len(recs) > int(max(0, run.lesion_threshold_max_retries)):
        return None, None
    last = recs[-1] if isinstance(recs[-1], dict) else {}
    if not bool(last.get("ok")):
        return None, None
    data = last.get("data", {}) if isinstance(last.get("data"), dict) else {}
    filter_stats = data.get("filter_stats", {}) if isinstance(data.get("filter_stats"), dict) else {}
    cands_path = Path(str(data.get("candidates_path") or "")).expanduser()
    cands_obj = _load_json(cands_path) if cands_path.exists() else {}
    cands = cands_obj.get("candidates", []) if isinstance(cands_obj, dict) else []
    if isinstance(cands, list) and cands:
        return None, None

    current_th = float(filter_stats.get("threshold", run.lesion_threshold_default))
    if current_th <= min(run.lesion_threshold_candidates or [current_th]):
        return None, None

    montages = data.get("candidate_montages", []) if isinstance(data.get("candidate_montages"), list) else []
    image_refs: List[str] = []
    for m in montages[:6]:
        if not isinstance(m, dict):
            continue
        for key in ("png_path", "tiles_json"):
            p = str(m.get(key) or "").strip()
            if p:
                image_refs.append(p)

    allowed = [float(x) for x in (run.lesion_threshold_candidates or []) if 0.0 < float(x) < float(current_th)]
    if not allowed:
        return None, None

    # Deterministic fallback: one rerun at 0.1 when available; otherwise use
    # the lowest configured threshold below current threshold.
    target_th = 0.1 if 0.1 in allowed else min(allowed)
    if target_th >= current_th:
        return None, None

    inputs = data.get("inputs", {}) if isinstance(data.get("inputs"), dict) else {}
    required = ("t2w_nifti", "adc_nifti", "highb_nifti", "prostate_mask_nifti")
    if not all(str(inputs.get(k) or "").strip() for k in required):
        return None, None

    args: Dict[str, Any] = {
        "t2w_nifti": str(inputs.get("t2w_nifti")),
        "adc_nifti": str(inputs.get("adc_nifti")),
        "highb_nifti": str(inputs.get("highb_nifti")),
        "prostate_mask_nifti": str(inputs.get("prostate_mask_nifti")),
        "output_subdir": "lesion",
        "threshold": float(target_th),
        "min_volume_cc": float(run.lesion_min_volume_cc),
    }
    if str(inputs.get("weights_dir") or "").strip():
        args["weights_dir"] = str(inputs.get("weights_dir"))
    call = ToolCall(
        tool_name="detect_lesion_candidates",
        arguments=args,
        call_id=f"cb_{uuid.uuid4().hex[:8]}",
        case_id=run.case_id,
        stage="lesion",
        requested_by="langgraph_agent:adaptive_threshold",
    )
    note = {
        "reason": "adaptive_lesion_threshold_rerun",
        "from_threshold": current_th,
        "to_threshold": float(target_th),
        "max_prob_overall": float(filter_stats.get("max_prob_overall", 0.0)),
        "decision_basis": "deterministic_empty_candidates_fallback",
        "image_refs_considered": image_refs,
    }
    return call, note


def _normalize_tool_calls(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    if parsed.get("action") == "tool_call":
        return [
            {
                "tool_name": parsed.get("tool_name"),
                "arguments": parsed.get("arguments"),
                "stage": parsed.get("stage"),
            }
        ]
    if parsed.get("action") == "tool_calls":
        calls = parsed.get("calls")
        if isinstance(calls, list):
            return [c for c in calls if isinstance(c, dict)]
    return []


def _build_observation(*, tool_name: str, tool_result: Dict[str, Any]) -> Dict[str, Any]:
    ok = bool(tool_result.get("ok")) if isinstance(tool_result, dict) else False
    warnings = tool_result.get("warnings") if isinstance(tool_result, dict) else None
    warnings_list = warnings if isinstance(warnings, list) else []
    error = tool_result.get("error") if isinstance(tool_result, dict) else None
    uncertainty: List[str] = []
    if not ok:
        uncertainty.append("tool_failed")
    if warnings_list:
        uncertainty.append("tool_warnings_present")
    if isinstance(error, dict) and error.get("message"):
        uncertainty.append("error_message_present")

    facts = extract_key_paths(tool_name, tool_result) if isinstance(tool_result, dict) else {}
    facts = _compact_prompt_obj(facts, max_chars=1200) if isinstance(facts, dict) else facts
    err_compact = None
    if isinstance(error, dict):
        err_compact = {
            "type": error.get("type"),
            "message": str(error.get("message") or "")[:400],
        }
    return {
        "tool_name": tool_name,
        "ok": ok,
        "facts": facts,
        "warnings": warnings_list,
        "error": err_compact,
        "uncertainty": uncertainty,
    }


def _fallback_next_tool_call(run: LangGraphRunCtx, st: LGState) -> Optional[Dict[str, Any]]:
    """
    Deterministic safety net when the policy model fails to return a valid action.
    Returns a minimal tool_call dict or None.
    """
    state_path = Path(run.tool_ctx.case_state_path)
    run_art_dir = state_path.parent / "artifacts"
    domain_name = getattr(run.domain, "name", None) or "prostate"

    try:
        case_state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        case_state = {}

    def _has_success(tool_name: str) -> bool:
        stage_outputs = case_state.get("stage_outputs", {}) if isinstance(case_state, dict) else {}
        if not isinstance(stage_outputs, dict):
            return False
        for _stage, tools in stage_outputs.items():
            if not isinstance(tools, dict):
                continue
            recs = tools.get(tool_name)
            if not isinstance(recs, list):
                continue
            if any(isinstance(r, dict) and r.get("ok") for r in recs):
                return True
        return False

    def _has_file(rel: str) -> bool:
        try:
            return (run_art_dir / rel).exists()
        except Exception:
            return False

    def _glob_any(patterns: List[str]) -> bool:
        try:
            for pat in patterns:
                if any(run_art_dir.glob(pat)):
                    return True
        except Exception:
            return False
        return False

    def _report_args() -> Dict[str, Any]:
        args = {"case_state_path": str(state_path), "output_subdir": "report", "domain": domain_name}
        return _apply_report_llm_args(run, args)

    def _final_action(note: str = "report_exists") -> Dict[str, Any]:
        return {"action": "final", "final_report": {"status": "ok", "note": note}}

    ident = _latest_tool_data(state_path, "identify", "identify_sequences") or {}
    mapping = ident.get("mapping", {}) if isinstance(ident, dict) else {}
    nifti_by_series = ident.get("nifti_by_series", {}) if isinstance(ident, dict) else {}
    have_identify = _has_success("identify_sequences")
    fixed = mapping.get("T2w") if isinstance(mapping, dict) else None
    adc = mapping.get("ADC") if isinstance(mapping, dict) else None
    dwi = mapping.get("DWI") if isinstance(mapping, dict) else None

    def _pick_high_b_from_nifti() -> Optional[str]:
        if not isinstance(nifti_by_series, dict):
            return None
        best_b = None
        best_path = None
        for key, path in nifti_by_series.items():
            if not isinstance(path, str) or not path:
                continue
            kl = str(key).lower()
            if "dwi" not in kl and "trace" not in kl:
                continue
            m = re.search(r"b(?:=|_|-)?(\d{2,5})", kl)
            b_hint = float(m.group(1)) if m else None
            if b_hint is None:
                if "high" in kl and "b" in kl:
                    b_hint = 2000.0
                elif "mid" in kl and "b" in kl:
                    b_hint = 800.0
                elif "low" in kl and "b" in kl:
                    b_hint = 50.0
            if b_hint is None:
                continue
            if best_b is None or b_hint > best_b:
                best_b = b_hint
                best_path = path
        return best_path

    if domain_name == "cardiac":
        if not have_identify:
            args: Dict[str, Any] = {"convert_to_nifti": True}
            if run.dicom_case_dir:
                args["dicom_case_dir"] = str(run.dicom_case_dir)
            return {"action": "tool_call", "tool_name": "identify_sequences", "arguments": args, "stage": "identify"}

        cine = mapping.get("CINE") if isinstance(mapping, dict) else None
        if not cine and isinstance(nifti_by_series, dict):
            for _, p in nifti_by_series.items():
                if isinstance(p, str) and p:
                    cine = p
                    break
        if not cine and isinstance(mapping, dict):
            for _, p in mapping.items():
                if isinstance(p, str) and p:
                    cine = p
                    break

        have_cardiac_seg = _glob_any(["**/segmentation/cardiac_cine/nnunet_io/pred/*.nii.gz", "**/*_rv_mask.nii.gz", "**/*_lv_mask.nii.gz"])
        if (not have_cardiac_seg) and cine:
            return {
                "action": "tool_call",
                "tool_name": "segment_cardiac_cine",
                "arguments": {"cine_path": cine, "output_subdir": "segmentation/cardiac_cine"},
                "stage": "segment",
            }

        have_cardiac_cls = _glob_any(["**/classification/cardiac_cine/cardiac_cine_classification.json"])
        if (not have_cardiac_cls) and have_cardiac_seg:
            return {
                "action": "tool_call",
                "tool_name": "classify_cardiac_cine_disease",
                "arguments": {"cine_path": cine, "output_subdir": "classification/cardiac_cine"},
                "stage": "report",
            }

        if not _has_file("vlm/vlm_evidence_bundle.json"):
            return {
                "action": "tool_call",
                "tool_name": "package_vlm_evidence",
                "arguments": {"case_state_path": str(state_path), "output_subdir": "vlm"},
                "stage": "package",
            }
        if _has_file("report/report.json"):
            return _final_action("report_already_generated")
        return {
            "action": "tool_call",
            "tool_name": "generate_report",
            "arguments": _report_args(),
            "stage": "report",
        }

    if domain_name == "brain":
        # 1) Identify sequences if we haven't already.
        if not have_identify:
            args: Dict[str, Any] = {"convert_to_nifti": True}
            if run.dicom_case_dir:
                args["dicom_case_dir"] = str(run.dicom_case_dir)
            return {"action": "tool_call", "tool_name": "identify_sequences", "arguments": args, "stage": "identify"}

        def _pick_from_nifti(hints: List[str]) -> Optional[str]:
            if not isinstance(nifti_by_series, dict):
                return None
            for p in nifti_by_series.values():
                if not isinstance(p, str):
                    continue
                pl = p.lower()
                if any(h in pl for h in hints):
                    return p
            return None

        t1c = mapping.get("T1c") or mapping.get("T1") or _pick_from_nifti(["t1c", "t1ce", "t1gd"])
        t1 = mapping.get("T1") or _pick_from_nifti(["t1"])
        t2 = mapping.get("T2") or _pick_from_nifti(["t2"])
        flair = mapping.get("FLAIR") or _pick_from_nifti(["flair"])

        have_brats_masks = _glob_any(["**/brats_tc_mask.nii.gz", "**/brats_wt_mask.nii.gz", "**/brats_et_mask.nii.gz"])
        if not have_brats_masks and t1c and t1 and t2 and flair:
            return {
                "action": "tool_call",
                "tool_name": "brats_mri_segmentation",
                "arguments": {
                    "t1c_path": t1c,
                    "t1_path": t1,
                    "t2_path": t2,
                    "flair_path": flair,
                    "output_subdir": "segmentation/brats_mri_segmentation",
                },
                "stage": "segment",
            }

        if not _has_file("features/features.csv"):
            def _find_one(pattern: str) -> Optional[str]:
                try:
                    hits = list(run_art_dir.glob(pattern))
                    if hits:
                        return str(hits[0])
                except Exception:
                    return None
                return None

            roi_masks: List[Dict[str, str]] = []
            tc = _find_one("**/brats_tc_mask.nii.gz")
            wt = _find_one("**/brats_wt_mask.nii.gz")
            et = _find_one("**/brats_et_mask.nii.gz")
            for name, path in (("Tumor_Core", tc), ("Whole_Tumor", wt), ("Enhancing_Tumor", et)):
                if isinstance(path, str) and path:
                    roi_masks.append({"name": name, "path": path})

            images: List[Dict[str, str]] = []
            for name, path in (("T1c", t1c), ("T1", t1), ("T2", t2), ("FLAIR", flair)):
                if isinstance(path, str) and path:
                    images.append({"name": name, "path": path})

            if roi_masks and images:
                return {
                    "action": "tool_call",
                    "tool_name": "extract_roi_features",
                    "arguments": {
                        "roi_masks": roi_masks,
                        "images": images,
                        "roi_interpretation": "binary_masks",
                        "output_subdir": "features",
                    },
                    "stage": "extract",
                }

        if not _has_file("vlm/vlm_evidence_bundle.json"):
            return {
                "action": "tool_call",
                "tool_name": "package_vlm_evidence",
                "arguments": {"case_state_path": str(state_path), "output_subdir": "vlm"},
                "stage": "package",
            }
        if _has_file("report/report.json"):
            return _final_action("report_already_generated")

        return {
            "action": "tool_call",
            "tool_name": "generate_report",
            "arguments": _report_args(),
            "stage": "report",
        }

    # Prostate defaults: prefer high-b DWI if we have per-b NIfTI outputs.
    dwi_high = _pick_high_b_from_nifti()
    if dwi_high:
        dwi = dwi_high

    # 1) Identify sequences first.
    if not fixed and not have_identify:
        args: Dict[str, Any] = {"convert_to_nifti": True}
        if run.dicom_case_dir:
            args["dicom_case_dir"] = str(run.dicom_case_dir)
        return {"action": "tool_call", "tool_name": "identify_sequences", "arguments": args, "stage": "identify"}
    if not fixed:
        if not _has_file("vlm/vlm_evidence_bundle.json"):
            return {
                "action": "tool_call",
                "tool_name": "package_vlm_evidence",
                "arguments": {"case_state_path": str(state_path), "output_subdir": "vlm"},
                "stage": "package",
            }
        if _has_file("report/report.json"):
            return _final_action("report_already_generated")
        return {
            "action": "tool_call",
            "tool_name": "generate_report",
            "arguments": _report_args(),
            "stage": "report",
        }

    # 2) Ensure ADC registration exists.
    have_adc_reg = _has_adc_registration_artifact(run_art_dir)
    if adc and not have_adc_reg:
        return {
            "action": "tool_call",
            "tool_name": "register_to_reference",
            "arguments": {
                "fixed": fixed,
                "moving": adc,
                "output_subdir": "registration/ADC",
                "split_by_bvalue": False,
                "save_png_qc": True,
            },
            "stage": "register",
        }

    # 3) Ensure DWI registration exists (high-b or dwi resampled).
    have_dwi_reg = _has_dwi_registration_artifact(run_art_dir)
    if dwi and not have_dwi_reg:
        return {
            "action": "tool_call",
            "tool_name": "register_to_reference",
            "arguments": {
                "fixed": fixed,
                "moving": dwi,
                "output_subdir": "registration/DWI",
                "split_by_bvalue": True,
                "save_png_qc": True,
            },
            "stage": "register",
        }

    # 4) Segmentation.
    if not _has_file("segmentation/prostate_whole_gland_mask.nii.gz"):
        return {
            "action": "tool_call",
            "tool_name": "segment_prostate",
            "arguments": {"t2w_ref": fixed, "output_subdir": "segmentation"},
            "stage": "segment",
        }

    # 5) Features.
    if not _has_file("features/features.csv"):
        return {
            "action": "tool_call",
            "tool_name": "extract_roi_features",
            "arguments": {"output_subdir": "features", "images": []},
            "stage": "extract",
        }

    # 6) Evidence bundle.
    if not _has_file("vlm/vlm_evidence_bundle.json"):
        return {
            "action": "tool_call",
            "tool_name": "package_vlm_evidence",
            "arguments": {"case_state_path": str(state_path), "output_subdir": "vlm"},
            "stage": "package",
        }
    if _has_file("report/report.json"):
        return _final_action("report_already_generated")

    # 7) Report (finalize will also ensure this, but this unblocks the loop).
    return {
        "action": "tool_call",
        "tool_name": "generate_report",
        "arguments": _report_args(),
        "stage": "report",
    }


def _llm_policy_node(run: LangGraphRunCtx, st: LGState) -> LGState:
    step = int(st.get("step", 0))
    messages = st.get("messages", [])
    messages = _trim_messages_for_context(messages)
    approx_tokens = int(len(str(messages)) / 4)
    print(
        f"🔍 [DEBUG] Current Prompt Token Count: ~{approx_tokens} tokens "
        f"(case={run.case_id}, step={step})",
        flush=True,
    )

    _write_message_trace(
        run.tool_ctx.run_dir,
        {"ts": datetime.utcnow().isoformat() + "Z", "tag": "policy", "step": step, "messages": _summarize_messages(messages)},
    )

    _write_agent_trace(run.tool_ctx.run_dir, {"ts": datetime.utcnow().isoformat() + "Z", "tag": "reactive", "step": step, "attempt": 0, "raw_len": None, "parsed": None})
    raw = ""
    parsed_obj: Optional[Dict[str, Any]] = None
    err: Optional[str] = None

    # 1) Prefer schema-guided decoding when available.
    res = run.policy.generate(messages, use_schema=True)
    raw, parsed_obj, err = res.raw, res.parsed, res.error

    # 2) Fallback to normal generation (handles servers without guided_json).
    if parsed_obj is None:
        res = run.policy.generate(messages, use_schema=False)
        raw, parsed_obj, err = res.raw, res.parsed, res.error

    # 3) Retry once with an explicit reminder if we still don't have a valid action.
    if parsed_obj is None:
        reminder = (
            "Return ONLY a JSON object with a required 'action' field "
            "('tool_call', 'tool_calls', or 'final'). No extra keys."
        )
        res = run.policy.generate(messages, use_schema=True, extra_msg=reminder)
        raw, parsed_obj, err = res.raw, res.parsed, res.error

    # 4) Recovery: some models return structured JSON without an `action` field
    # (e.g., echoing state_summary). Attempt to salvage a valid action or retry
    # with a more explicit constraint.
    if parsed_obj is None:
        obj = _try_parse_json(raw)
        if isinstance(obj, dict):
            tool_name = obj.get("tool_name")
            arguments = obj.get("arguments")
            calls = obj.get("calls")
            if isinstance(tool_name, str) and isinstance(arguments, dict):
                parsed_obj = {
                    "action": "tool_call",
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "stage": obj.get("stage"),
                }
                err = None
            elif isinstance(calls, list) and calls and all(isinstance(c, dict) for c in calls):
                parsed_obj = {"action": "tool_calls", "calls": calls}
                err = None

    if parsed_obj is None and err and "missing/invalid 'action'" in str(err).lower():
        hard_reminder = (
            "You must choose the NEXT STEP now. Return exactly one JSON object with:\n"
            "{\"action\":\"tool_call\",\"tool_name\":\"...\",\"arguments\":{...},\"stage\":\"...\"}\n"
            "Do not summarize state. Pick a tool."
        )
        res = run.policy.generate(messages, use_schema=True, extra_msg=hard_reminder)
        raw, parsed_obj, err = res.raw, res.parsed, res.error

    if (
        err
        and parsed_obj is None
        and any(
            k in str(err).lower()
            for k in (
                "maximum context length",
                "context window exhausted",
                "max_tokens",
                "max_completion_tokens",
                "too large",
            )
        )
    ):
        for limit in (40_000, 30_000, 20_000):
            messages = _trim_messages_for_context(messages, max_chars=limit)
            res = run.policy.generate(messages, use_schema=True)
            raw, parsed_obj, err = res.raw, res.parsed, res.error
            if parsed_obj is not None:
                break

    # Enforce generate_report before allowing action=final.
    if isinstance(parsed_obj, dict) and parsed_obj.get("action") == "final":
        report_json = Path(run.tool_ctx.artifacts_dir) / "report" / "report.json"
        if not report_json.exists():
            report_args: Dict[str, Any] = {
                "case_state_path": str(run.tool_ctx.case_state_path),
                "output_subdir": "report",
            }
            if getattr(run.domain, "name", None):
                report_args["domain"] = str(run.domain.name)
            report_args = _apply_report_llm_args(run, report_args)
            parsed_obj = {
                "action": "tool_call",
                "tool_name": "generate_report",
                "arguments": report_args,
                "stage": "report",
            }
            err = (str(err) + " | forced_generate_report") if err else "forced_generate_report"

    # If we already produced a report successfully in the previous step,
    # suppress repeated generate_report loops and finalize directly.
    if (
        isinstance(parsed_obj, dict)
        and parsed_obj.get("action") == "tool_call"
        and str(parsed_obj.get("tool_name") or "") == "generate_report"
    ):
        report_json = Path(run.tool_ctx.artifacts_dir) / "report" / "report.json"
        if report_json.exists() and st.get("last_tool_name") == "generate_report" and bool(st.get("last_tool_ok")):
            parsed_obj = {"action": "final", "final_report": {"status": "ok", "note": "report_already_generated"}}
            err = (str(err) + " | dedup_generate_report") if err else "dedup_generate_report"

    # 5) Final deterministic fallback when the model keeps summarizing state.
    if parsed_obj is None:
        fb = _fallback_next_tool_call(run, st)
        if isinstance(fb, dict):
            parsed_obj = fb
            err = (str(err) + " | policy_fallback") if err else "policy_fallback"
            if run.autofix_mode == "coach":
                messages = (messages or []) + [
                    {"role": "user", "content": json.dumps({"policy_fallback": fb}, ensure_ascii=False)}
                ]

    # Alignment gate for prostate registration.
    if isinstance(parsed_obj, dict):
        gated_obj, gate_note = _alignment_gate_for_action(run, parsed_obj)
        if gated_obj is not parsed_obj:
            parsed_obj = gated_obj
            err = (str(err) + f" | {gate_note}") if err and gate_note else gate_note or err
        elif parsed_obj.get("action") == "tool_calls":
            gated_calls_obj, calls_note = _alignment_gate_for_calls(run, parsed_obj)
            if gated_calls_obj is not parsed_obj:
                parsed_obj = gated_calls_obj
                err = (str(err) + f" | {calls_note}") if err and calls_note else calls_note or err

    # Distortion gate for prostate (VLM image-read decision before lesion/report tools).
    if isinstance(parsed_obj, dict):
        dist_obj, dist_note = _distortion_gate_for_policy_action(run, parsed_obj)
        if dist_obj is not parsed_obj:
            parsed_obj = dist_obj
            err = (str(err) + f" | {dist_note}") if err and dist_note else dist_note or err

    _write_message_trace(
        run.tool_ctx.run_dir,
        {"ts": datetime.utcnow().isoformat() + "Z", "tag": "policy_response", "step": step, "model_output": raw},
    )

    _write_agent_trace(
        run.tool_ctx.run_dir,
        {"ts": datetime.utcnow().isoformat() + "Z", "tag": "reactive", "step": step, "attempt": 0, "raw_len": len(raw or ""), "parsed": parsed_obj, "error": err},
    )
    st["last_model_raw"] = raw
    st["last_model_parsed"] = parsed_obj
    st["last_error"] = err
    st["messages"] = messages
    return st


def _plan_node(run: LangGraphRunCtx, st: LGState) -> LGState:
    """
    Independent plan/reason step that produces a structured plan snapshot.
    This updates state["plan"] and appends a compact plan summary into messages.
    """
    if not _should_run_plan_reflect(run, st):
        return st
    step = int(st.get("step", 0))
    state_summary = summarize_case_state(run.tool_ctx.case_state_path)
    memory_digest = build_memory_digest(
        case_state_path=Path(run.tool_ctx.case_state_path),
        artifacts_dir=Path(run.tool_ctx.artifacts_dir),
        max_events=4,
        autofix_mode=run.autofix_mode,
    )
    state_summary_prompt = _compact_prompt_obj(state_summary, max_chars=1800)
    memory_digest_prompt = _compact_prompt_obj(memory_digest, max_chars=2200)
    tool_manifest_path = Path(run.tool_ctx.artifacts_dir) / "context" / "tool_manifest.md"
    tool_manifest_text = ""
    if tool_manifest_path.exists():
        try:
            tool_manifest_text = tool_manifest_path.read_text(encoding="utf-8")
        except Exception:
            tool_manifest_text = ""
    if tool_manifest_text:
        max_manifest_chars = 3000
        if len(tool_manifest_text) > max_manifest_chars:
            tool_manifest_text = tool_manifest_text[:max_manifest_chars].rstrip() + "\n\n[TRUNCATED]"
    skills_summary = run.skills_registry.to_prompt(set_name=None, tags=[run.domain.name], max_items=8)
    skills_summary = _truncate_text(skills_summary, max_chars=1400)
    tool_index_text = _truncate_text(json.dumps(_compact_prompt_obj(run.tool_index, max_chars=2800), ensure_ascii=False), max_chars=3000)
    plan_input_text = (
        "Task: You are given DICOM files. Use the available tools to understand the medical imaging data and produce a report.\n"
        f"case_id: {run.case_id}\n"
        f"step: {step}\n"
        f"tool_manifest: {tool_manifest_path}\n"
        + "tool_index:\n"
        + tool_index_text
        + "\n\nskills_summary:\n"
        + skills_summary
        + "\n\n"
        + ("tool_manifest_contents:\n" + tool_manifest_text + "\n\n" if tool_manifest_text else "")
        + "state_summary:\n"
        + json.dumps(state_summary_prompt, ensure_ascii=False, indent=2)
        + "\n\n"
        + "memory_digest:\n"
        + json.dumps(memory_digest_prompt, ensure_ascii=False, indent=2)
        + "\n\n"
        + "prior_plan (if any):\n"
        + _compact_plan_for_prompt(st.get("plan"))
    )

    plan_payload = {"plan_input_text": plan_input_text}
    plan_messages: List[Dict[str, Any]] = []
    raw = ""
    plan_text = ""
    try:
        res = run.planner.run(plan_payload)
        plan_text = str(res.get("plan_text") or "").strip()
        plan_messages = res.get("messages") or []
        raw = plan_text
    except Exception:
        plan_text = ""

    _write_message_trace(
        run.tool_ctx.run_dir,
        {"ts": datetime.utcnow().isoformat() + "Z", "tag": "plan", "step": step, "messages": _summarize_messages(plan_messages)},
    )
    _write_message_trace(
        run.tool_ctx.run_dir,
        {"ts": datetime.utcnow().isoformat() + "Z", "tag": "plan_response", "step": step, "model_output": raw},
    )

    st["plan"] = {"plan_text": plan_text, "tool_manifest": str(tool_manifest_path)}
    _write_plan_trace(
        run.tool_ctx.run_dir,
        {
            "ts": datetime.utcnow().isoformat() + "Z",
            "step": step,
            "plan": st.get("plan"),
        },
    )
    return st


def _reflect_node(run: LangGraphRunCtx, st: LGState) -> LGState:
    """
    Reflection step: summarize progress and suggest next action.
    Writes reflect_trace.jsonl for audit.
    """
    if not _should_run_plan_reflect(run, st):
        return st
    step = int(st.get("step", 0))
    state_summary = summarize_case_state(run.tool_ctx.case_state_path)
    memory_digest = build_memory_digest(
        case_state_path=Path(run.tool_ctx.case_state_path),
        artifacts_dir=Path(run.tool_ctx.artifacts_dir),
        max_events=4,
        autofix_mode=run.autofix_mode,
    )
    payload = {
        "case_id": run.case_id,
        "step": step,
        "plan_summary": st.get("plan"),
        "state_summary": _compact_prompt_obj(state_summary, max_chars=1800),
        "memory_digest": _compact_prompt_obj(memory_digest, max_chars=2200),
    }
    reflect_payload = {"reflect_input_text": json.dumps(payload, ensure_ascii=False)}
    reflect_messages: List[Dict[str, Any]] = []
    text = ""
    try:
        res = run.reflector.run(reflect_payload)
        text = str(res.get("reflection") or "").strip()
        reflect_messages = res.get("messages") or []
    except Exception:
        text = ""
    st["reflection"] = text
    _write_message_trace(
        run.tool_ctx.run_dir,
        {"ts": datetime.utcnow().isoformat() + "Z", "tag": "reflect", "step": step, "messages": _summarize_messages(reflect_messages)},
    )
    _write_message_trace(
        run.tool_ctx.run_dir,
        {"ts": datetime.utcnow().isoformat() + "Z", "tag": "reflect", "step": step, "reflection": text},
    )
    return st


def _route_after_policy(run: LangGraphRunCtx, st: LGState) -> str:
    if st.get("done"):
        return END
    step = int(st.get("step", 0))
    if step >= int(run.max_steps):
        return "finalize"
    parsed = st.get("last_model_parsed")
    if not isinstance(parsed, dict):
        return "finalize"
    if parsed.get("action") == "final":
        return "finalize"
    if parsed.get("action") == "tool_call":
        return "preconditions"
    if parsed.get("action") == "tool_calls":
        return "tool_exec"
    return "finalize"


def _preconditions_node(run: LangGraphRunCtx, st: LGState) -> LGState:
    """
    Preconditions + rule checks before tool execution.
    In LangGraph mode we REPLACE the next tool call with a precondition call when needed,
    and let the LLM decide the subsequent step afterward.
    """
    parsed = st.get("last_model_parsed") or {}
    if not isinstance(parsed, dict) or parsed.get("action") != "tool_call":
        return st
    tool_name = str(parsed.get("tool_name") or "")
    arguments = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
    stage = str(parsed.get("stage") or "misc")

    call = ToolCall(
        tool_name=tool_name,
        arguments=arguments,
        call_id="precondition_probe",
        case_id=run.case_id,
        stage=stage,
        requested_by="langgraph_agent:policy",
    )

    violations = validate_tool_call(call, Path(run.tool_ctx.case_state_path))
    if violations and run.autofix_mode == "coach":
        st["messages"] = (st.get("messages") or []) + [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "rule_violations": [
                            {
                                "rule_id": v.rule_id,
                                "level": v.level,
                                "tool_name": v.tool_name,
                                "message": v.message,
                                "details": v.details or {},
                            }
                            for v in violations
                        ]
                    },
                    ensure_ascii=False,
                ),
            }
        ]

    repair_fn = partial(_auto_repair_args_for_tool, domain=run.domain)
    pre = apply_preconditions(
        call,
        case_state_path=Path(run.tool_ctx.case_state_path),
        dicom_case_dir=run.dicom_case_dir,
        autofix_mode=run.autofix_mode,
        strategy="replace",
        auto_repair_fn=repair_fn,
    )

    if pre.call.tool_name != call.tool_name or pre.call.arguments != call.arguments:
        st["last_model_parsed"] = {
            "action": "tool_call",
            "tool_name": pre.call.tool_name,
            "arguments": pre.call.arguments,
            "stage": pre.call.stage,
        }
        if run.autofix_mode == "coach" and pre.notes:
            st["messages"] = (st.get("messages") or []) + [
                {"role": "user", "content": json.dumps({"preconditions_applied": pre.notes}, ensure_ascii=False)}
            ]
    return st


def _tool_exec_node(run: LangGraphRunCtx, st: LGState) -> LGState:
    step = int(st.get("step", 0))
    parsed = st.get("last_model_parsed") or {}
    # Support tool_calls by queueing pending calls and executing one at a time.
    pending = st.get("pending_calls") or []
    if isinstance(parsed, dict) and parsed.get("action") == "tool_calls":
        if not (isinstance(pending, list) and pending):
            pending = _normalize_tool_calls(parsed)
            st["pending_calls"] = pending
    if isinstance(pending, list) and pending:
        next_call = pending.pop(0)
        st["pending_calls"] = pending
        parsed = {
            "action": "tool_call",
            "tool_name": next_call.get("tool_name"),
            "arguments": next_call.get("arguments"),
            "stage": next_call.get("stage"),
        }

    if not isinstance(parsed, dict) or parsed.get("action") != "tool_call":
        return st
    tool_name = str(parsed.get("tool_name"))
    arguments = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
    stage = str(parsed.get("stage") or "misc")

    repaired_args = dict(arguments)
    autofix_applied: Dict[str, Any] = {}
    if run.autofix_mode in ("force", "coach"):
        repaired_args = _auto_repair_args_for_tool(
            tool_name,
            repaired_args,
            state_path=run.tool_ctx.case_state_path,
            ctx_case_state_path=run.tool_ctx.case_state_path,
            dicom_case_dir=run.dicom_case_dir,
            domain=run.domain,
        )
        # Best-effort diff for coach transparency
        try:
            changed = {k: {"from": arguments.get(k), "to": repaired_args.get(k)} for k in set(arguments.keys()) | set(repaired_args.keys()) if arguments.get(k) != repaired_args.get(k)}
            if changed:
                autofix_applied = {"tool_name": tool_name, "changed_arguments": changed}
        except Exception:
            autofix_applied = {}

    if tool_name == "generate_report":
        if "domain" not in repaired_args and getattr(run.domain, "name", None):
            repaired_args["domain"] = str(run.domain.name)
        if "llm_mode" not in repaired_args:
            repaired_args = _apply_report_llm_args(run, repaired_args)
        if run.llm_mode == "server":
            base_url = str(repaired_args.get("server_base_url") or (run.server_cfg.base_url if run.server_cfg else ""))
            model = str(repaired_args.get("server_model") or (run.server_cfg.model if run.server_cfg else ""))
            repaired_args["server_model"] = _sanitize_server_model(model, base_url)
    if tool_name == "detect_lesion_candidates":
        repaired_args.setdefault("threshold", float(run.lesion_threshold_default))
        repaired_args.setdefault("min_volume_cc", float(run.lesion_min_volume_cc))
        # Hard data-flow handoff: if distortion correction succeeded, force downstream lesion inputs
        # to corrected artifacts regardless of LLM-proposed stale paths.
        try:
            dist = _latest_tool_data(Path(run.tool_ctx.case_state_path), "register", "correct_prostate_distortion")
            if isinstance(dist, dict):
                forced_paths: Dict[str, str] = {}
                corr_t2 = str(dist.get("t2w_nifti") or "").strip()
                corr_adc = str(dist.get("corrected_adc_nifti") or "").strip()
                corr_highb = str(dist.get("corrected_highb_nifti") or "").strip()
                if (not corr_t2) or (not Path(corr_t2).exists()):
                    seg = _latest_tool_data(Path(run.tool_ctx.case_state_path), "segment", "segment_prostate")
                    if isinstance(seg, dict):
                        corr_t2 = str(seg.get("t2w_input_path") or "").strip()
                if corr_t2 and Path(corr_t2).exists():
                    repaired_args["t2w_nifti"] = corr_t2
                    forced_paths["t2w_nifti"] = corr_t2
                if corr_adc and Path(corr_adc).exists():
                    repaired_args["adc_nifti"] = corr_adc
                    forced_paths["adc_nifti"] = corr_adc
                if corr_highb and Path(corr_highb).exists():
                    repaired_args["highb_nifti"] = corr_highb
                    forced_paths["highb_nifti"] = corr_highb
                if forced_paths:
                    _write_agent_trace(
                        run.tool_ctx.run_dir,
                        {
                            "ts": datetime.utcnow().isoformat() + "Z",
                            "tag": "distortion_handoff",
                            "step": step,
                            "forced_inputs": forced_paths,
                        },
                    )
        except Exception:
            pass

    call = ToolCall(
        tool_name=tool_name,
        arguments=repaired_args,
        call_id=f"lg_step{step:03d}_{uuid.uuid4().hex[:6]}",
        case_id=run.case_id,
        stage=stage,
        requested_by="langgraph_agent",
    )
    _write_agent_trace(run.tool_ctx.run_dir, {"ts": datetime.utcnow().isoformat() + "Z", "tag": "tool_exec", "step": step, "attempt": 0, "parsed": {"tool_name": tool_name, "stage": stage, "arguments": repaired_args}})
    _emit_progress(
        run,
        {
            "event": "tool_start",
            "step": int(step) + 1,
            "max_steps": int(run.max_steps),
            "tool_name": tool_name,
            "stage": stage,
        },
    )

    result = run.dispatcher.dispatch(call, run.state, run.tool_ctx)
    _emit_progress(
        run,
        {
            "event": "tool_end",
            "step": int(step) + 1,
            "max_steps": int(run.max_steps),
            "tool_name": tool_name,
            "stage": stage,
            "ok": bool(result.ok),
        },
    )
    run.state.write_json(run.tool_ctx.case_state_path)
    st["last_tool_name"] = tool_name
    st["last_tool_ok"] = bool(result.ok)
    try:
        append_short_term_event(
            artifacts_dir=Path(run.tool_ctx.artifacts_dir),
            step=step,
            tool_name=tool_name,
            tool_result=result.to_dict(),
        )
    except Exception:
        pass

    # Feed observation (structured) + tool_result back to LLM
    observation = _build_observation(tool_name=tool_name, tool_result=result.to_dict())
    tool_payload: Dict[str, Any] = {"observation": observation, "tool_result": compact_for_memory(result.to_dict(), max_chars=2000)}
    if run.autofix_mode == "coach" and autofix_applied:
        tool_payload["autofix_applied"] = autofix_applied

    st["step"] = step + 1
    # Clear parsed action unless we are still draining a queued tool_calls list.
    if not (st.get("pending_calls") or []):
        st["last_model_parsed"] = None
    st["force_tool_exec"] = False
    return st


def _post_step_context_node(run: LangGraphRunCtx, st: LGState) -> LGState:
    """
    Append state_summary + memory_digest-like info each iteration.
    For now we reuse summarize_case_state only; memory_digest can be plugged later.
    """
    state_summary = summarize_case_state(run.tool_ctx.case_state_path)
    memory_digest = build_memory_digest(
        case_state_path=Path(run.tool_ctx.case_state_path),
        artifacts_dir=Path(run.tool_ctx.artifacts_dir),
        max_events=4,
        autofix_mode=run.autofix_mode,
    )
    payload = {
        "state_summary": _compact_prompt_obj(state_summary, max_chars=1800),
        "memory_digest": _compact_prompt_obj(memory_digest, max_chars=2200),
        "skills_guidance": _truncate_text(str(run.skills_guidance or ""), max_chars=1200),
    }
    st["messages"] = (st.get("messages") or []) + [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    return st


def _circuit_breaker_node(run: LangGraphRunCtx, st: LGState) -> LGState:
    """
    Explicit failure-handling branch driven by hooks:
    - If detect_lesion_candidates failed, do NOT retry; package evidence and proceed.
    - If detect_lesion_candidates succeeded, trigger lesion ROI feature extraction (once).
    """
    if st.get("last_tool_name") == "detect_lesion_candidates" and bool(st.get("last_tool_ok")):
        maybe_call, maybe_note = _build_adaptive_lesion_redetect_call(run)
        if maybe_call is not None:
            st["last_model_parsed"] = {
                "action": "tool_call",
                "tool_name": maybe_call.tool_name,
                "arguments": maybe_call.arguments,
                "stage": maybe_call.stage,
            }
            st["force_tool_exec"] = True
            st["messages"] = (st.get("messages") or []) + [
                {"role": "user", "content": json.dumps({"adaptive_lesion_threshold": maybe_note}, ensure_ascii=False)}
            ]
            return st

    res = apply_circuit_breaker(
        case_state_path=Path(run.tool_ctx.case_state_path),
        case_id=run.case_id,
        last_tool_name=st.get("last_tool_name"),
        last_tool_ok=st.get("last_tool_ok"),
    )
    def _tc_to_dict(tc: ToolCall) -> Dict[str, Any]:
        return {"tool_name": tc.tool_name, "arguments": tc.arguments, "stage": tc.stage}

    if isinstance(getattr(res, "note", None), dict) and res.note.get("fatal"):
        st["fatal_error"] = res.note
        # Force finalize on fatal error to avoid infinite retries.
        st["step"] = max(int(st.get("step", 0)), int(run.max_steps))
        st["force_tool_exec"] = False
        if res.note:
            st["messages"] = (st.get("messages") or []) + [
                {"role": "user", "content": json.dumps({"circuit_breaker": res.note}, ensure_ascii=False)}
            ]
        return st

    if isinstance(getattr(res, "calls", None), list) and res.calls:
        calls_payload = [_tc_to_dict(tc) for tc in res.calls if isinstance(tc, ToolCall)]
        if not calls_payload:
            return st
        st["last_model_parsed"] = {"action": "tool_calls", "calls": calls_payload}
    elif res.call is not None:
        st["last_model_parsed"] = {
            "action": "tool_call",
            "tool_name": res.call.tool_name,
            "arguments": res.call.arguments,
            "stage": res.call.stage,
        }
    else:
        return st
    st["force_tool_exec"] = True
    if run.autofix_mode == "coach" and res.note:
        st["messages"] = (st.get("messages") or []) + [
            {"role": "user", "content": json.dumps({"circuit_breaker": res.note}, ensure_ascii=False)}
        ]
    return st


def _route_after_circuit_breaker(run: LangGraphRunCtx, st: LGState) -> str:
    if st.get("done"):
        return END
    step = int(st.get("step", 0))
    if step >= int(run.max_steps):
        return "finalize"
    pending = st.get("pending_calls") or []
    if isinstance(pending, list) and pending:
        return "tool_exec"
    if st.get("force_tool_exec"):
        return "tool_exec"
    return "plan"


def _finalize_node(run: LangGraphRunCtx, st: LGState) -> LGState:
    """
    Produce final output artifacts.
    - Always package evidence.
    - Generate report.json/txt via generate_report (report_generation.py). Optionally run VLM if enabled.
    """
    # Ensure evidence bundle exists
    try:
        call = ToolCall(
            tool_name="package_vlm_evidence",
            arguments={"case_state_path": str(run.tool_ctx.case_state_path), "output_subdir": "vlm"},
            call_id=f"lg_auto_package_{uuid.uuid4().hex[:6]}",
            case_id=run.case_id,
            stage="package",
            requested_by="langgraph_agent:auto",
        )
        run.dispatcher.dispatch(call, run.state, run.tool_ctx)
        run.state.write_json(run.tool_ctx.case_state_path)
    except Exception:
        pass

    final_dir = run.tool_ctx.artifacts_dir / "final"
    final_json = final_dir / "final_report.json"
    final_txt = final_dir / "final_report.txt"
    final_dir.mkdir(parents=True, exist_ok=True)

    fatal = st.get("fatal_error") if isinstance(st.get("fatal_error"), dict) else None
    if fatal and fatal.get("tool_name") == "generate_report":
        note = str(fatal.get("reason") or "generate_report failed with a fatal error.")
        err = fatal.get("error")
        repeat = fatal.get("repeat_count")
        msg_lines = [
            "Report generation aborted due to fatal error.",
            f"Reason: {note}",
        ]
        if isinstance(err, dict) and err.get("message"):
            msg_lines.append(f"Error: {err.get('message')}")
        if repeat:
            msg_lines.append(f"Repeated failures: {repeat}")
        final_txt.write_text("\n".join(msg_lines) + "\n", encoding="utf-8")
        final_json.write_text(
            json.dumps(
                {
                    "case_id": run.case_id,
                    "run_id": run.run_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "final_report": {
                        "status": "failed",
                        "note": note,
                        "error": err,
                        "repeat_count": repeat,
                        "final_report_txt_path": str(final_txt),
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        run.state.set_summary(
            {
                "final_report_path": str(final_json),
                "final_report_txt_path": str(final_txt),
            }
        )
        run.state.write_json(run.tool_ctx.case_state_path)
        st["done"] = True
        st["final_report_path"] = str(final_json)
        return st

    report_ok = False
    report_data: Dict[str, Any] = {}
    report_dir = Path(run.tool_ctx.artifacts_dir) / "report"
    existing_report_json = report_dir / "report.json"
    if existing_report_json.exists():
        report_ok = True
        report_data = {
            "report_json_path": str(existing_report_json),
            "report_txt_path": str(report_dir / "report.md") if (report_dir / "report.md").exists() else None,
            "clinical_report_path": str(report_dir / "clinical_report.md") if (report_dir / "clinical_report.md").exists() else None,
            "vlm_evidence_bundle_path": str(Path(run.tool_ctx.artifacts_dir) / "vlm" / "vlm_evidence_bundle.json")
            if (Path(run.tool_ctx.artifacts_dir) / "vlm" / "vlm_evidence_bundle.json").exists()
            else None,
            "llm_structured_report_path": str(report_dir / "structured_report.json")
            if (report_dir / "structured_report.json").exists()
            else None,
        }
    else:
        report_args: Dict[str, Any] = {
            "case_state_path": str(run.tool_ctx.case_state_path),
            "output_subdir": "report",
        }
        try:
            report_args["domain"] = str(run.domain.name)
        except Exception:
            pass
        report_args = _apply_report_llm_args(run, report_args)

        try:
            call = ToolCall(
                tool_name="generate_report",
                arguments=report_args,
                call_id=f"lg_auto_report_{uuid.uuid4().hex[:6]}",
                case_id=run.case_id,
                stage="report",
                requested_by="langgraph_agent:auto",
            )
            result = run.dispatcher.dispatch(call, run.state, run.tool_ctx)
            run.state.write_json(run.tool_ctx.case_state_path)
            report_ok = bool(result.ok)
            if isinstance(result.data, dict):
                report_data = result.data
        except Exception as e:
            report_ok = False
            report_data = {"error": repr(e)}

    final_txt_source: Optional[Path] = None
    # Prefer clinical_report.txt (integrated clinical), then structured_report.txt, then report.txt.
    struct_txt: Optional[Path] = None
    struct_json = str(report_data.get("llm_structured_report_path") or "")
    if struct_json:
        try:
            struct_txt = Path(struct_json).with_suffix(".txt")
        except Exception:
            struct_txt = None
    clinical_txt = Path(str(report_data.get("clinical_report_path") or "")) if report_data.get("clinical_report_path") else None
    report_txt = Path(str(report_data.get("report_txt_path") or "")) if report_data.get("report_txt_path") else None
    for cand in (clinical_txt, struct_txt, report_txt):
        if cand is not None and str(cand) and cand.exists():
            final_txt_source = cand
            break
    if final_txt_source is not None:
        try:
            final_txt.write_text(final_txt_source.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass

    if report_ok and report_data.get("report_json_path"):
        final_json.write_text(
            json.dumps(
                {
                    "case_id": run.case_id,
                    "run_id": run.run_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "final_report": {
                        "status": "ok",
                        "report_json_path": report_data.get("report_json_path"),
                        "report_txt_path": report_data.get("report_txt_path"),
                        "clinical_report_path": report_data.get("clinical_report_path"),
                        "final_report_txt_path": str(final_txt) if final_txt.exists() else None,
                        "final_report_txt_source": str(final_txt_source) if final_txt_source else None,
                        "vlm_evidence_bundle_path": report_data.get("vlm_evidence_bundle_path"),
                        "llm_structured_report_path": report_data.get("llm_structured_report_path"),
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        final_json.write_text(
            json.dumps(
                {
                    "case_id": run.case_id,
                    "run_id": run.run_id,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "final_report": {
                        "status": "failed",
                        "note": "generate_report did not complete successfully.",
                        **({"error": report_data.get("error")} if isinstance(report_data, dict) and report_data.get("error") else {}),
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    run.state.set_summary(
        {
            "final_report_path": str(final_json),
            **({"final_report_txt_path": str(final_txt)} if final_txt.exists() else {}),
        }
    )
    run.state.write_json(run.tool_ctx.case_state_path)
    st["done"] = True
    st["final_report_path"] = str(final_json)
    return st


def run_langgraph_agent(
    *,
    case_id: str,
    dicom_case_dir: Optional[str],
    runs_root: Path,
    llm_mode: str = "server",
    plan_mode: str = "step",  # reserved
    autofix_mode: str = "force",
    max_steps: int = 12,
    finalize_with_llm: bool = False,
    max_new_tokens: int = 2048,
    server_cfg: Optional[VLLMServerConfig] = None,
    api_model: Optional[str] = None,
    api_base_url: Optional[str] = None,
    domain: Optional[DomainConfig] = None,
    lesion_threshold_default: float = 0.3,
    lesion_min_volume_cc: float = 0.1,
    lesion_threshold_adaptive: bool = True,
    lesion_threshold_candidates: Optional[List[float]] = None,
    lesion_threshold_max_retries: int = 1,
    distortion_gate_override: Optional[str] = None,
    progress_emit: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Path:
    if StateGraph is None:
        raise RuntimeError(
            "langgraph is not installed. Install dependency 'langgraph' to run run_langgraph_agent."
        )

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    domain_cfg = domain or get_domain_config("prostate")
    registry = build_registry()
    dispatcher = ToolDispatcher(registry=registry, runs_root=runs_root)
    state, tool_ctx = dispatcher.create_run(case_id=case_id, run_id=run_id)
    try:
        state.metadata = {**(state.metadata or {}), "domain": domain_cfg.name}
        state.write_json(tool_ctx.case_state_path)
    except Exception:
        pass

    # LLM selection
    if llm_mode == "server":
        cfg = server_cfg or VLLMServerConfig(max_tokens=int(max_new_tokens))
        llm = VLLMOpenAIChatAdapter(cfg)
    elif llm_mode == "openai":
        llm = OpenAIChatAdapter(OpenAIConfig(model=api_model or OpenAIConfig().model, base_url=api_base_url))
    elif llm_mode == "anthropic":
        llm = AnthropicChatAdapter(AnthropicConfig(model=api_model or AnthropicConfig().model))
    elif llm_mode == "gemini":
        llm = GeminiChatAdapter(GeminiConfig(model=api_model or GeminiConfig().model))
    else:
        # stub mode: keep compatibility with existing FakeLLM by importing lazily
        from ..loop import FakeLLM  # type: ignore

        llm = FakeLLM()

    tools_list = dispatcher.list_tools()
    tool_index = _compact_tool_index(tools_list)
    skills_path = project_root() / "config" / "skills.json"
    skills_registry = SkillRegistry.load(skills_path)
    skills_summary = skills_registry.to_prompt(set_name=None, tags=[domain_cfg.name], max_items=8)
    skills_summary = _truncate_text(skills_summary, max_chars=1600)
    skills_guidance = skills_registry.to_guidance(set_name=None, tags=[domain_cfg.name], max_items=8)
    skills_guidance = _truncate_text(skills_guidance, max_chars=1600)

    # Persist tool manifest (human + future prompt strategies)
    try:
        write_tool_manifest(tools_list=tools_list, tool_index=tool_index, ctx_dir=tool_ctx.artifacts_dir / "context")
    except Exception:
        pass

    # Initial messages (system + tool list + state)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": LANGGRAPH_SYSTEM_PROMPT}]
    tool_names = sorted([t.get("name") for t in tools_list if isinstance(t, dict) and "name" in t])
    tool_manifest_path = tool_ctx.artifacts_dir / "context" / "tool_manifest.md"
    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                {
                    "goal": "Read this case and produce a report.",
                    "case_id": case_id,
                    "inputs": {"dicom_case_dir": dicom_case_dir},
                    "tool_names": tool_names,
                    "tool_index": _compact_prompt_obj(tool_index, max_chars=3200),
                    "tool_manifest": str(tool_manifest_path),
                    "skills_summary": skills_summary,
                    "skills_guidance": skills_guidance,
                    "note": "Choose one next tool_call at a time. Use tool_result observations to decide the next step.",
                },
                ensure_ascii=False,
            ),
        }
    )
    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                {"state_summary": _compact_prompt_obj(summarize_case_state(tool_ctx.case_state_path), max_chars=1800)},
                ensure_ascii=False,
            ),
        }
    )

    planner = PlannerSubagent(llm=llm)
    reflector = ReflectorSubagent(llm=llm)
    policy = PolicySubagent(llm=llm, parse_fn=parse_model_output, guided_schema=build_guided_schema(allow_tool_calls=True))
    alignment_gate = AlignmentGateSubagent(llm=llm)
    distortion_gate = DistortionGateSubagent(llm=llm)

    run = LangGraphRunCtx(
        runs_root=runs_root,
        case_id=case_id,
        run_id=run_id,
        llm_mode=llm_mode,
        autofix_mode=autofix_mode if autofix_mode in ("force", "coach", "off") else "force",
        finalize_with_llm=bool(finalize_with_llm),
        dicom_case_dir=dicom_case_dir,
        max_steps=int(max_steps),
        dispatcher=dispatcher,
        state=state,
        tool_ctx=tool_ctx,
        tools_list=tools_list,
        tool_index=tool_index,
        skills_registry=skills_registry,
        skills_guidance=skills_guidance,
        llm=llm,
        server_cfg=server_cfg,
        api_model=api_model,
        api_base_url=api_base_url,
        planner=planner,
        reflector=reflector,
        policy=policy,
        alignment_gate=alignment_gate,
        distortion_gate=distortion_gate,
        domain=domain_cfg,
        lesion_threshold_default=float(lesion_threshold_default),
        lesion_min_volume_cc=float(lesion_min_volume_cc),
        lesion_threshold_adaptive=bool(lesion_threshold_adaptive),
        lesion_threshold_candidates=[float(x) for x in (lesion_threshold_candidates or [0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.05])],
        lesion_threshold_max_retries=int(max(0, lesion_threshold_max_retries)),
        distortion_gate_override=str(distortion_gate_override) if distortion_gate_override else None,
        progress_emit=progress_emit,
    )

    g = StateGraph(LGState)
    g.add_node("plan", lambda st: _plan_node(run, st))
    g.add_node("reflect", lambda st: _reflect_node(run, st))
    g.add_node("policy", lambda st: _llm_policy_node(run, st))
    g.add_node("preconditions", lambda st: _preconditions_node(run, st))
    g.add_node("tool_exec", lambda st: _tool_exec_node(run, st))
    g.add_node("post_ctx", lambda st: _post_step_context_node(run, st))
    g.add_node("circuit_breaker", lambda st: _circuit_breaker_node(run, st))
    g.add_node("finalize", lambda st: _finalize_node(run, st))

    g.set_entry_point("plan")
    g.add_edge("plan", "reflect")
    g.add_edge("reflect", "policy")
    g.add_conditional_edges(
        "policy", lambda st: _route_after_policy(run, st), {"preconditions": "preconditions", "tool_exec": "tool_exec", "finalize": "finalize", END: END}
    )
    g.add_edge("preconditions", "tool_exec")
    g.add_edge("tool_exec", "post_ctx")
    g.add_edge("post_ctx", "circuit_breaker")
    g.add_conditional_edges(
        "circuit_breaker",
        lambda st: _route_after_circuit_breaker(run, st),
        {"plan": "plan", "tool_exec": "tool_exec", "finalize": "finalize", END: END},
    )
    g.add_edge("finalize", END)

    app = g.compile()
    init: LGState = {"step": 0, "messages": messages, "done": False}
    # recursion_limit must exceed plan/reflect/policy/tool cycles
    app.invoke(init, {"recursion_limit": max(200, int(max_steps) * 8)})

    return tool_ctx.run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--dicom-case-dir", required=False, default=None)
    ap.add_argument("--runs-root", default=str(project_root() / "runs"))
    ap.add_argument("--llm-mode", default="server", choices=["server", "stub", "openai", "anthropic", "gemini"])
    ap.add_argument("--autofix-mode", default="force", choices=["force", "coach", "off"])
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--finalize-with-llm", action="store_true")
    ap.add_argument("--domain", default="prostate", choices=["prostate", "brain", "cardiac"], help="Domain-specific MRI logic configuration.")
    ap.add_argument("--server-base-url", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--server-model",
        default=os.environ.get("MEDGEMMA_SERVER_MODEL", "Qwen/Qwen3-VL-30B-A3B-Thinking"),
        help="Model name served by the vLLM OpenAI-compatible server.",
    )
    ap.add_argument(
        "--model-id",
        dest="server_model",
        help="Alias of --server-model (kept for convenience).",
    )
    ap.add_argument("--api-model", default="", help="Model name for llm-mode openai|anthropic|gemini.")
    ap.add_argument("--api-base-url", default="", help="Optional base URL for OpenAI-compatible gateways/proxies.")
    ap.add_argument("--lesion-threshold", type=float, default=0.3, help="Default threshold for detect_lesion_candidates.")
    ap.add_argument("--lesion-min-volume-cc", type=float, default=0.1, help="Minimum lesion component volume (cc). Smaller candidates are discarded.")
    ap.add_argument(
        "--adaptive-lesion-threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When first-pass lesion candidates are empty, rerun detect_lesion_candidates once with a lower threshold (default: enabled).",
    )
    ap.add_argument(
        "--adaptive-lesion-threshold-candidates",
        default="0.1",
        help="Candidate thresholds for one-step empty-candidate fallback retry (default: 0.1).",
    )
    ap.add_argument(
        "--adaptive-lesion-threshold-max-retries",
        type=int,
        default=1,
        help="Max adaptive reruns for detect_lesion_candidates when candidates are empty.",
    )
    args = ap.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    server_model = _sanitize_server_model(str(args.server_model), str(args.server_base_url))
    server_cfg = VLLMServerConfig(base_url=str(args.server_base_url), model=server_model, max_tokens=int(args.max_new_tokens))
    domain_cfg = get_domain_config(str(args.domain))
    run_dir = run_langgraph_agent(
        case_id=str(args.case_id),
        dicom_case_dir=str(args.dicom_case_dir) if args.dicom_case_dir else None,
        runs_root=runs_root,
        llm_mode=str(args.llm_mode),
        autofix_mode=str(args.autofix_mode),
        max_steps=int(args.max_steps),
        finalize_with_llm=bool(args.finalize_with_llm),
        max_new_tokens=int(args.max_new_tokens),
        server_cfg=server_cfg,
        api_model=str(args.api_model or "") or None,
        api_base_url=str(args.api_base_url or "") or None,
        domain=domain_cfg,
        lesion_threshold_default=float(args.lesion_threshold),
        lesion_min_volume_cc=float(args.lesion_min_volume_cc),
        lesion_threshold_adaptive=bool(args.adaptive_lesion_threshold),
        lesion_threshold_candidates=_parse_threshold_candidates(str(args.adaptive_lesion_threshold_candidates)),
        lesion_threshold_max_retries=int(args.adaptive_lesion_threshold_max_retries),
    )
    print(f"[OK] LangGraph agent run output: {run_dir}")


if __name__ == "__main__":
    main()

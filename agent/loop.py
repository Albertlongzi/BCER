from __future__ import annotations

import argparse
import base64
import json
import os
import re
import uuid
from functools import partial
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from commands.dispatcher import ToolDispatcher
from commands.registry import ToolRegistry
from commands.schemas import CaseState, ToolCall
from core.parser import parse_model_output
from llm.adapter_vllm_server import VLLMOpenAIChatAdapter, VLLMServerConfig
from llm.adapter_openai_api import OpenAIChatAdapter, OpenAIConfig
from llm.adapter_anthropic_api import AnthropicChatAdapter, AnthropicConfig
from llm.adapter_gemini_api import GeminiChatAdapter, GeminiConfig
from core.domain_config import DomainConfig, get_domain_config
from .react_failure_reflector import ReActFailureReflector
from .subagents.prompts import ONE_SHOT_SYSTEM_PROMPT, REACTIVE_SYSTEM_PROMPT, THOUGHT_PROMPT, build_reactive_system_prompt
from .hooks.preconditions import apply_preconditions
from .rules.engine import validate_tool_call
from .skills.registry import SkillRegistry
from runtime.finalize import finalize_free_text_report
from runtime.memory import append_short_term_event, build_memory_digest
from runtime.tool_manifest import write_tool_manifest
from tools.arg_models import repair_tool_args
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


def _flag_enabled(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def _register_experimental_tools(reg: ToolRegistry, *, enable_sandbox_exec: bool = True) -> None:
    enable_exp = _flag_enabled("MRI_AGENT_ENABLE_EXPERIMENTAL")
    enable_rag = True
    enable_sandbox = bool(enable_sandbox_exec)
    if enable_rag:
        from tools.rag_search import build_tool as build_rag_search_tool

        reg.register(build_rag_search_tool())
    if enable_sandbox:
        from tools.sandbox_exec import build_tool as build_sandbox_tool

        reg.register(build_sandbox_tool())


def build_registry(*, enable_sandbox_exec: bool = True) -> ToolRegistry:
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
    _register_experimental_tools(reg, enable_sandbox_exec=enable_sandbox_exec)
    return reg


def summarize_case_state(state_path: Path) -> Dict[str, Any]:
    try:
        s = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"note": "state_unreadable"}
    stage_outputs = s.get("stage_outputs", {}) or {}
    summary = {
        "case_id": s.get("case_id"),
        "run_id": s.get("run_id"),
        "stages": {},
        "completed_tools": [],
        "last_tool": None,
    }
    last_order = -1
    for stage, tools in stage_outputs.items():
        summary["stages"][stage] = {tool: len(records or []) for tool, records in (tools or {}).items()}
        for tool, records in (tools or {}).items():
            if not isinstance(records, list) or not records:
                continue
            # Find last record (by stage_order when available).
            for r in records:
                if not isinstance(r, dict):
                    continue
                order = int(r.get("stage_order") or 0)
                if order >= last_order:
                    last_order = order
                    err_obj = (r.get("data") or {}).get("error") if isinstance(r.get("data"), dict) else None
                    err_compact = None
                    if isinstance(err_obj, dict):
                        err_compact = {
                            "type": err_obj.get("type"),
                            "message": str(err_obj.get("message") or "")[:300],
                        }
                    summary["last_tool"] = {
                        "stage": stage,
                        "tool_name": tool,
                        "ok": r.get("ok"),
                        "error": err_compact,
                    }
            # Completed tools list (ok=True only).
            for r in records:
                if isinstance(r, dict) and r.get("ok") is True:
                    summary["completed_tools"].append({"stage": stage, "tool_name": tool})
                    break

    # Add a compact "latest identify + series brief" so the model can see available sequences.
    try:
        ident = _latest_tool_data(state_path, "identify", "identify_sequences") or {}
        mapping = ident.get("mapping", {}) if isinstance(ident, dict) else {}
        summary["identify_latest"] = {
            "mapping_keys": sorted(list(mapping.keys())) if isinstance(mapping, dict) else [],
            "reference_suggested": "T2w",
        }
    except Exception:
        pass

    try:
        ingest = _latest_tool_data(state_path, "ingest", "ingest_dicom_to_nifti") or {}
        if not ingest:
            ingest = _latest_tool_data(state_path, "identify", "identify_sequences") or {}
        series = ingest.get("series", []) if isinstance(ingest, dict) else []
        brief = []
        dwi_has_bvals = False
        if isinstance(series, list):
            for s0 in series:
                if not isinstance(s0, dict):
                    continue
                bvals = s0.get("b_values")
                if isinstance(bvals, list) and len(bvals) > 0:
                    dwi_has_bvals = True
                brief.append(
                    {
                        "name": s0.get("series_name"),
                        "guess": s0.get("sequence_guess"),
                        "n_dicoms": s0.get("n_dicoms"),
                        "SeriesDescription": s0.get("SeriesDescription"),
                        "b_values": s0.get("b_values"),
                    }
                )
        summary["series_brief"] = brief
        summary["dwi_has_bvals"] = dwi_has_bvals
    except Exception:
        pass
    return summary



def _compact_tool_index(tools_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a compact tool index to reduce prompt length.
    """
    out = []
    for t in tools_list:
        if not isinstance(t, dict) or "name" not in t:
            continue
        schema = t.get("input_schema") or {}
        required = schema.get("required") if isinstance(schema, dict) else None
        props = (schema.get("properties") if isinstance(schema, dict) else None) or {}
        out.append(
            {
                "name": t.get("name"),
                "desc": (str(t.get("description", ""))[:160] + ("..." if len(str(t.get("description", ""))) > 160 else "")),
                "required_args": required if isinstance(required, list) else [],
                "optional_args": sorted(list(props.keys())) if isinstance(props, dict) else [],
            }
        )
    return out


@dataclass
class FakeLLM:
    """
    Deterministic planner for MVP debugging.
    It emits a fixed tool plan suitable for the demo DICOM folder case.
    """

    step: int = 0

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        self.step += 1
        demo_case_dir = str(Path(__file__).resolve().parents[1] / "demo" / "cases" / "sub-057")
        # Very small script-like policy
        if self.step == 1:
            return json.dumps(
                {
                    "action": "tool_call",
                    "stage": "ingest",
                    "tool_name": "ingest_dicom_to_nifti",
                    "arguments": {"dicom_case_dir": demo_case_dir, "output_subdir": "ingest"},
                }
            )
        if self.step == 2:
            # The state summary will include the inventory path; but for simplicity in fake LLM,
            # we use the conventional artifact location under the current run.
            # The agent loop will rewrite this argument before dispatch if it sees a placeholder.
            return json.dumps(
                {
                    "action": "tool_call",
                    "stage": "identify",
                    "tool_name": "identify_sequences",
                    "arguments": {"series_inventory_path": "__LATEST__/ingest/series_inventory.json"},
                }
            )
        if self.step == 3:
            return json.dumps(
                {
                    "action": "final",
                    "final_report": {"status": "ok", "note": "FakeLLM stopped early. Use pipeline runner for full fake-tool chain."},
                }
            )
        return json.dumps({"action": "final", "final_report": {"status": "ok"}})


def _rewrite_placeholders(arguments: Dict[str, Any], artifacts_dir: Path) -> Dict[str, Any]:
    """
    Convenience: allow the LLM to reference __LATEST__/subdir/file relative to artifacts_dir.
    """
    out: Dict[str, Any] = {}
    for k, v in arguments.items():
        if isinstance(v, str) and v.startswith("__LATEST__/"):
            out[k] = str((artifacts_dir / v[len("__LATEST__/") :]).resolve())
        else:
            out[k] = v
    return out


def _load_sequence_mapping_for_symbolic_binder(case_state_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    try:
        ident = _latest_tool_data(case_state_path, "identify", "identify_sequences") or {}
        raw_map = ident.get("mapping") if isinstance(ident, dict) else {}
        if isinstance(raw_map, dict):
            for k, v in raw_map.items():
                ks = str(k or "").strip()
                vs = str(v or "").strip()
                if ks and vs:
                    mapping[ks] = vs
    except Exception:
        return {}
    return mapping


def _apply_symbolic_binder(
    arguments: Dict[str, Any],
    *,
    case_state_path: Path,
    dicom_case_dir: Optional[str],
) -> Dict[str, Any]:
    """
    Lightweight symbolic binder for ReAct+token mode.
    It only resolves explicit symbolic refs, and intentionally avoids
    heavyweight auto-repair, preconditions, or scope rewriting.
    """
    seq_mapping = _load_sequence_mapping_for_symbolic_binder(case_state_path)
    token_map: Dict[str, str] = {
        "@runtime.case_state_path": str(case_state_path),
        "@state.path": str(case_state_path),
        "state.path": str(case_state_path),
    }
    case_in = str(dicom_case_dir or "").strip()
    if case_in:
        token_map["@case.input"] = case_in
        token_map["case.input"] = case_in
        token_map["@case.path"] = case_in
        token_map["case.path"] = case_in

    for k, v in seq_mapping.items():
        if str(k).strip() and str(v).strip():
            token_map[f"@seq.{str(k).strip()}"] = str(v).strip()

    seq_alias = {
        "t2": "T2w",
        "t2w": "T2w",
        "adc": "ADC",
        "dwi": "DWI",
        "t1c": "T1c",
        "t1": "T1",
        "t2_brain": "T2",
        "flair": "FLAIR",
        "cine": "CINE",
    }

    def _bind_scalar(s: str) -> Any:
        raw = str(s or "").strip()
        if not raw:
            return s
        if raw in token_map:
            return token_map[raw]
        if raw.startswith("@seq."):
            seq_key = str(raw[len("@seq.") :]).strip()
            if not seq_key:
                return s
            direct = seq_mapping.get(seq_key)
            if isinstance(direct, str) and direct.strip():
                return direct
            alias_key = seq_alias.get(seq_key.lower(), seq_key)
            alias_val = seq_mapping.get(alias_key)
            if isinstance(alias_val, str) and alias_val.strip():
                return alias_val
            # Last resort: case-insensitive sequence lookup.
            low = seq_key.lower()
            for k, v in seq_mapping.items():
                if str(k).lower() == low and str(v).strip():
                    return str(v)
        return s

    def _walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        if isinstance(obj, str):
            return _bind_scalar(obj)
        return obj

    return _walk(dict(arguments or {}))


_RETRY_PATH_HINT_EXTS = (".nii", ".nii.gz", ".h5", ".json", ".csv", ".txt", ".png", ".tfm", ".pt")


def _retry_key_looks_like_input_path(key: str) -> bool:
    k = str(key or "").strip().lower()
    if not k:
        return False
    if any(tok in k for tok in ("output", "save", "write", "target_spacing", "kernel_size")):
        return False
    if k in {"fixed", "moving", "cine_path", "seg_path", "case_state_path", "dicom_case_dir"}:
        return True
    return any(tok in k for tok in ("input", "reference", "path", "nifti", "mask", "roi", "feature_table"))


def _retry_value_looks_like_path(v: Any) -> bool:
    s = str(v or "").strip()
    if not s or s.startswith("@"):
        return False
    if "/" in s or s.startswith("."):
        return True
    sl = s.lower()
    return any(sl.endswith(ext) for ext in _RETRY_PATH_HINT_EXTS)


def _normalize_and_validate_reflector_retry_args(
    *,
    current_args: Dict[str, Any],
    retry_args: Dict[str, Any],
    retry_patch: Dict[str, Any],
    run_dir: Path,
    artifacts_dir: Path,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Normalize relative path-like retry args and block obviously bad input paths.

    Returns (normalized_args, issues, normalized_keys).
    """
    out = dict(retry_args or {})
    issues: List[str] = []
    normalized_keys: List[str] = []
    patched_keys = [str(k) for k in (retry_patch or {}).keys()]
    for key in patched_keys:
        if key not in out:
            continue
        val = out.get(key)
        if not (isinstance(val, str) and _retry_key_looks_like_input_path(key) and _retry_value_looks_like_path(val)):
            continue
        raw = str(val).strip()
        p = Path(raw)
        candidates: List[Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.extend([run_dir / raw, artifacts_dir / raw])
        existing: List[Path] = []
        seen: set[str] = set()
        for c in candidates:
            cs = str(c)
            if cs in seen:
                continue
            seen.add(cs)
            try:
                if c.exists():
                    existing.append(c)
            except Exception:
                continue
        if existing:
            chosen = existing[0]
            try:
                chosen_s = str(chosen.resolve())
            except Exception:
                chosen_s = str(chosen)
            if chosen_s != raw:
                out[key] = chosen_s
                normalized_keys.append(str(key))
            continue
        # Only hard-block keys that strongly look like input file/path references.
        if any(tok in str(key).lower() for tok in ("case_state", "seg_path", "cine_path", "reference", "input", "fixed", "moving", "mask", "roi", "feature")):
            issues.append(f"{key} path does not exist after normalization: {raw}")
    return out, issues, normalized_keys



def _latest_tool_data(state_path: Path, stage: str, tool_name: str) -> Dict[str, Any]:
    """
    Return the latest stored `data` for a tool.

    NOTE:
    Historically, stages were named like "identify"/"segment"/... but current one-shot planning
    often uses numeric stage keys like "1","2","3". To be robust, if the requested stage is not
    present we fall back to searching ALL stages and returning the latest record.
    """

    def _stage_sort_key(k: str):
        # Prefer numeric stages in ascending order; non-numeric stages come after.
        try:
            return (0, int(k), "")
        except Exception:
            return (1, 10**9, str(k))

    try:
        s = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    all_stage_outputs: Dict[str, Any] = s.get("stage_outputs", {}) or {}

    # Fast path: stage exists.
    stage_outputs = (all_stage_outputs.get(stage, {}) or {}) if isinstance(all_stage_outputs, dict) else {}
    if isinstance(stage_outputs, dict):
        records = stage_outputs.get(tool_name, []) or []
        if records:
            rec = records[-1] or {}
            return rec.get("data", {}) or {}

    # Fallback: search across all stages (ordered), return the latest record found.
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


def _extract_missing_required_args(err_type: str, err_message: str) -> List[str]:
    if "schemavalidationerror" not in str(err_type or "").strip().lower():
        return []
    msg = str(err_message or "")
    keys = re.findall(r"Missing required argument:\s*([A-Za-z0-9_]+)", msg)
    out: List[str] = []
    seen: set[str] = set()
    for k in keys:
        ks = str(k or "").strip()
        if ks and ks not in seen:
            seen.add(ks)
            out.append(ks)
    return out


def _tool_error_type_message(result: Any) -> Tuple[str, str]:
    err_obj = getattr(result, "error", None)
    if err_obj is None and isinstance(result, dict):
        err_obj = result.get("error")
    if err_obj is None:
        return "", ""
    if isinstance(err_obj, dict):
        return str(err_obj.get("type") or ""), str(err_obj.get("message") or "")
    return str(getattr(err_obj, "type", "") or ""), str(getattr(err_obj, "message", "") or "")


def _pick_mapping_path(mapping: Dict[str, Any], *keys: str) -> str:
    for k in keys:
        v = mapping.get(str(k))
        if isinstance(v, str) and v.strip():
            return str(v).strip()
    # case-insensitive fallback
    low_map = {str(k).strip().lower(): v for k, v in mapping.items()}
    for k in keys:
        v = low_map.get(str(k).strip().lower())
        if isinstance(v, str) and v.strip():
            return str(v).strip()
    return ""


def _first_non_empty(*vals: Any) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return str(v).strip()
    return ""


def _build_schema_repair_patch(
    *,
    tool_name: str,
    args_current: Dict[str, Any],
    missing_keys: List[str],
    state_path: Path,
    dicom_case_dir: Optional[str],
) -> Dict[str, Any]:
    # If one required key is missing, proactively fill the full required bundle
    # for known tools to avoid repeated one-by-one SchemaValidationError loops.
    required_bundle: Dict[str, List[str]] = {
        "brats_mri_segmentation": ["t1_path", "t1c_path", "t2_path", "flair_path"],
        "resample_image": ["input_nifti"],
        "denoise_image_bm3d": ["input_nifti"],
        "register_to_reference": ["fixed", "moving"],
        "segment_cardiac_cine": ["cine_path"],
        "classify_brain_glioma_grade": ["feature_table_path"],
        "classify_cardiac_cine_disease": ["cine_path"],
    }
    target_keys = set(str(k).strip() for k in (missing_keys or []) if str(k).strip())
    for k in required_bundle.get(str(tool_name or "").strip(), []):
        target_keys.add(k)

    ident = _latest_tool_data(state_path, "identify", "identify_sequences") or {}
    mapping = ident.get("mapping") if isinstance(ident, dict) and isinstance(ident.get("mapping"), dict) else {}
    nifti_by_series = (
        ident.get("nifti_by_series")
        if isinstance(ident, dict) and isinstance(ident.get("nifti_by_series"), dict)
        else {}
    )
    reg_data = _latest_tool_data(state_path, "register", "register_to_reference") or {}
    seg_prostate = _latest_tool_data(state_path, "segment", "segment_prostate") or {}
    seg_brain = _latest_tool_data(state_path, "segment", "brats_mri_segmentation") or {}
    seg_cardiac = _latest_tool_data(state_path, "segment", "segment_cardiac_cine") or {}
    feat_data = _latest_tool_data(state_path, "extract", "extract_roi_features") or {}

    t2w = _pick_mapping_path(mapping, "T2w", "T2")
    t1 = _pick_mapping_path(mapping, "T1")
    t1c = _pick_mapping_path(mapping, "T1c")
    t2 = _pick_mapping_path(mapping, "T2", "T2w")
    flair = _pick_mapping_path(mapping, "FLAIR")
    adc = _pick_mapping_path(mapping, "ADC")
    dwi = _first_non_empty(_pick_high_b_from_nifti(nifti_by_series), _pick_mapping_path(mapping, "DWI"))
    cine = _pick_mapping_path(mapping, "CINE")
    any_nifti = _first_non_empty(adc, t2w, t2, t1c, t1, flair, cine)

    patch: Dict[str, Any] = {}
    for key in sorted(target_keys):
        if key in args_current and args_current.get(key) not in (None, "", [], {}):
            continue
        if key in {"case_state_path", "state_path"}:
            patch[key] = str(state_path)
            continue
        if key in {"dicom_case_dir", "case_path"}:
            if isinstance(dicom_case_dir, str) and dicom_case_dir.strip():
                patch[key] = str(dicom_case_dir).strip()
            continue
        if key == "input_nifti":
            v = _first_non_empty(
                str(reg_data.get("resampled_path") or ""),
                str(seg_prostate.get("t2w_input_path") or ""),
                any_nifti,
            )
            if v:
                patch[key] = v
            continue
        if key == "fixed":
            v = _first_non_empty(t2w, t2, any_nifti)
            if v:
                patch[key] = v
            continue
        if key == "moving":
            fixed_val = str(args_current.get("fixed") or patch.get("fixed") or "")
            candidates = [adc, dwi, t1c, t1, flair, any_nifti]
            v = ""
            for cand in candidates:
                if cand and cand != fixed_val:
                    v = cand
                    break
            if v:
                patch[key] = v
            continue
        if key == "t1_path" and t1:
            patch[key] = t1
            continue
        if key == "t1c_path" and t1c:
            patch[key] = t1c
            continue
        if key == "t2_path" and t2:
            patch[key] = t2
            continue
        if key == "flair_path" and flair:
            patch[key] = flair
            continue
        if key == "t2w_ref" and t2w:
            patch[key] = t2w
            continue
        if key == "cine_path" and cine:
            patch[key] = cine
            continue
        if key == "feature_table_path":
            v = str(feat_data.get("feature_table_path") or "")
            if v:
                patch[key] = v
            continue

    # A few pragmatic tool-specific fixes when the model stalls on empty args.
    tname = str(tool_name or "").strip()
    if tname == "brats_mri_segmentation":
        if t1 and "t1_path" not in patch and not args_current.get("t1_path"):
            patch["t1_path"] = t1
        if t1c and "t1c_path" not in patch and not args_current.get("t1c_path"):
            patch["t1c_path"] = t1c
        if t2 and "t2_path" not in patch and not args_current.get("t2_path"):
            patch["t2_path"] = t2
        if flair and "flair_path" not in patch and not args_current.get("flair_path"):
            patch["flair_path"] = flair
        if ("output_subdir" not in args_current) and ("output_subdir" not in patch):
            patch["output_subdir"] = "segmentation"
    elif tname in {"resample_image", "denoise_image_bm3d"}:
        if ("input_nifti" not in args_current) and ("input_nifti" not in patch) and any_nifti:
            patch["input_nifti"] = any_nifti
        if ("output_subdir" not in args_current) and ("output_subdir" not in patch):
            patch["output_subdir"] = ("resample" if tname == "resample_image" else "denoise")
    elif tname == "classify_brain_glioma_grade":
        ft = str(feat_data.get("feature_table_path") or "")
        if ("feature_table_path" not in args_current) and ("feature_table_path" not in patch) and ft:
            patch["feature_table_path"] = ft
    elif tname == "extract_roi_features":
        if (not args_current.get("images")) and ("images" not in patch):
            imgs: List[Dict[str, str]] = []
            for nm, p in (("T2w", t2w), ("ADC", adc), ("DWI", dwi), ("T1c", t1c), ("FLAIR", flair), ("CINE", cine)):
                if p:
                    imgs.append({"name": nm, "path": p})
            if imgs:
                patch["images"] = imgs
        if (not args_current.get("roi_masks")) and ("roi_masks" not in patch):
            roi_masks: List[Dict[str, str]] = []
            for nm, p in (
                ("WT", str(seg_brain.get("wt_mask_path") or "")),
                ("PROSTATE", str(seg_prostate.get("prostate_mask_path") or "")),
                ("LV", str(seg_cardiac.get("lv_mask_path") or "")),
                ("MYO", str(seg_cardiac.get("myo_mask_path") or "")),
                ("RV", str(seg_cardiac.get("rv_mask_path") or "")),
            ):
                if p:
                    roi_masks.append({"name": nm, "path": p})
            if roi_masks:
                patch["roi_masks"] = roi_masks

    return patch


def _auto_repair_args_for_tool(
    tool_name: str,
    args: Dict[str, Any],
    *,
    state_path: Path,
    ctx_case_state_path: Path,
    dicom_case_dir: Optional[str],
    domain: Optional[DomainConfig] = None,
) -> Dict[str, Any]:
    return repair_tool_args(
        tool_name,
        args,
        state_path=state_path,
        ctx_case_state_path=ctx_case_state_path,
        dicom_case_dir=dicom_case_dir,
        domain=domain,
    )


def _pick_high_b_from_nifti(nifti_by_series: Any) -> Optional[str]:
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


def _ensure_required_registrations(
    dispatcher: ToolDispatcher,
    state: CaseState,
    ctx: Any,
    *,
    case_id: str,
) -> None:
    """
    Ensure the prostate demo has the 4 required volumes in T2 space:
    - T2w (fixed)
    - ADC (moving)
    - DWI low_b + high_b (from TRACEW split)

    Ignore CALC_BVAL for MVP.
    """
    ident = _latest_tool_data(ctx.case_state_path, "identify", "identify_sequences") or {}
    mapping = ident.get("mapping", {}) if isinstance(ident, dict) else {}
    nifti_by_series = ident.get("nifti_by_series", {}) if isinstance(ident, dict) else {}
    if not isinstance(mapping, dict):
        return
    fixed = mapping.get("T2w")
    if not fixed:
        return

    # Register ADC
    adc = mapping.get("ADC")
    if adc:
        call = ToolCall(
            tool_name="register_to_reference",
            arguments={"fixed": fixed, "moving": adc, "output_subdir": "registration/ADC", "split_by_bvalue": False, "save_png_qc": True},
            call_id="auto_register_adc",
            case_id=case_id,
            stage="register",
            requested_by="agent_loop:auto",
        )
        dispatcher.dispatch(call, state, ctx)

    # Register DWI TRACEW (split into low_b/high_b by tool logic)
    dwi = mapping.get("DWI")
    dwi_high = _pick_high_b_from_nifti(nifti_by_series)
    if dwi_high:
        dwi = dwi_high
    if dwi:
        call = ToolCall(
            tool_name="register_to_reference",
            arguments={"fixed": fixed, "moving": dwi, "output_subdir": "registration/DWI", "split_by_bvalue": True, "save_png_qc": True},
            call_id="auto_register_dwi",
            case_id=case_id,
            stage="register",
            requested_by="agent_loop:auto",
        )
        dispatcher.dispatch(call, state, ctx)


def _ensure_segmentation_and_features_and_report(
    dispatcher: ToolDispatcher,
    state: CaseState,
    ctx: Any,
    *,
    case_id: str,
    llm_mode: str = "stub",
    server_cfg: Any = None,
) -> None:
    """
    After required registrations exist, ensure:
    - segmentation runs on T2w
    - features run on resampled volumes using zone mask (labels) if available
    - VLM evidence bundle is (re)generated
    """
    ident = _latest_tool_data(ctx.case_state_path, "identify", "identify_sequences") or {}
    mapping = ident.get("mapping", {}) if isinstance(ident, dict) else {}
    if not isinstance(mapping, dict) or not mapping.get("T2w"):
        return

    # Segmentation (GPU by default)
    seg_args = {
        "t2w_ref": mapping["T2w"],
        "device": "cuda",
        "output_subdir": "segmentation",
    }
    env_bundle = os.getenv("MRI_AGENT_PROSTATE_BUNDLE_DIR")
    if env_bundle:
        seg_args["bundle_dir"] = env_bundle
    seg_call = ToolCall(
        tool_name="segment_prostate",
        arguments=seg_args,
        call_id="auto_segment",
        case_id=case_id,
        stage="segment",
        requested_by="agent_loop:auto",
    )
    seg_res = dispatcher.dispatch(seg_call, state, ctx)

    # Features: prefer zonal mask to enable PZ/TZ stats
    roi_mask = None
    if seg_res.ok and isinstance(seg_res.data, dict):
        roi_mask = seg_res.data.get("zone_mask_path") or seg_res.data.get("prostate_mask_path")

    # Images: use registered NIfTIs (ADC + DWI low/high)
    reg_dir = Path(ctx.case_state_path).parent / "artifacts" / "registration"
    images = []
    seen = set()
    for p in sorted(reg_dir.glob("**/*.nii.gz")):
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        name = p.parent.name
        images.append({"name": f"{name}_{p.stem}", "path": rp})

    # Also include the T2 NIfTI written by segmentation, if present (useful for VLM + feature context).
    t2_nifti = Path(ctx.case_state_path).parent / "artifacts" / "segmentation" / "t2w_input.nii.gz"
    if t2_nifti.exists():
        rp = str(t2_nifti.resolve())
        if rp not in seen:
            images.append({"name": "T2w_t2w_input", "path": rp})
            seen.add(rp)

    if roi_mask and images:
        feat_call = ToolCall(
            tool_name="extract_roi_features",
            arguments={"roi_mask_path": str(roi_mask), "images": images, "output_subdir": "features"},
            call_id="auto_features",
            case_id=case_id,
            stage="extract",
            requested_by="agent_loop:auto",
        )
        dispatcher.dispatch(feat_call, state, ctx)

    # Evidence bundle (always): report writing is handled by agent finalization (not a tool).
    pkg_call = ToolCall(
        tool_name="package_vlm_evidence",
        arguments={"case_state_path": str(ctx.case_state_path), "output_subdir": "vlm"},
        call_id="auto_package_vlm",
        case_id=case_id,
        stage="report",
        requested_by="agent_loop:auto",
    )
    dispatcher.dispatch(pkg_call, state, ctx)


def _run_deterministic_prostate_mvp(
    dispatcher: ToolDispatcher,
    state: CaseState,
    ctx: Any,
    *,
    case_id: str,
    dicom_case_dir: Optional[str],
    llm_mode: str,
    server_cfg: Any,
) -> None:
    """
    Deterministic MVP fallback (no LLM planning):
      ingest -> identify -> register ADC+DWI -> segment -> features -> report(+VLM if available)
    This keeps the demo runnable even if the LLM server is down.
    """
    if dicom_case_dir:
        call = ToolCall(
            tool_name="ingest_dicom_to_nifti",
            arguments={"dicom_case_dir": dicom_case_dir, "require_pydicom": True, "output_subdir": "ingest"},
            call_id="det_ingest",
            case_id=case_id,
            stage="ingest",
            requested_by="agent_loop:deterministic",
        )
        dispatcher.dispatch(call, state, ctx)

    ingest = _latest_tool_data(ctx.case_state_path, "ingest", "ingest_dicom_to_nifti") or {}
    series_inventory_path = ingest.get("series_inventory_path") if isinstance(ingest, dict) else None
    if series_inventory_path:
        call = ToolCall(
            tool_name="identify_sequences",
            arguments={"series_inventory_path": str(series_inventory_path)},
            call_id="det_identify",
            case_id=case_id,
            stage="identify",
            requested_by="agent_loop:deterministic",
        )
        dispatcher.dispatch(call, state, ctx)

    _ensure_required_registrations(dispatcher, state, ctx, case_id=case_id)
    _ensure_segmentation_and_features_and_report(
        dispatcher,
        state,
        ctx,
        case_id=case_id,
        llm_mode=llm_mode,
        server_cfg=server_cfg,
    )


def _run_deterministic_cardiac_mvp(
    dispatcher: ToolDispatcher,
    state: CaseState,
    ctx: Any,
    *,
    case_id: str,
    dicom_case_dir: Optional[str],
) -> None:
    """
    Deterministic cardiac MVP fallback:
      identify -> segment_cardiac_cine -> classify_cardiac_cine_disease -> package evidence.
    """
    if dicom_case_dir:
        call = ToolCall(
            tool_name="identify_sequences",
            arguments={"dicom_case_dir": dicom_case_dir, "convert_to_nifti": True, "output_subdir": "ingest"},
            call_id="det_cardiac_identify",
            case_id=case_id,
            stage="identify",
            requested_by="agent_loop:deterministic",
        )
        dispatcher.dispatch(call, state, ctx)

    ident = _latest_tool_data(ctx.case_state_path, "identify", "identify_sequences") or {}
    mapping = ident.get("mapping", {}) if isinstance(ident, dict) else {}
    cine = mapping.get("CINE") if isinstance(mapping, dict) else None
    if (not cine) and isinstance(mapping, dict):
        for _, p in mapping.items():
            if isinstance(p, str) and p:
                cine = p
                break
    if cine:
        seg_call = ToolCall(
            tool_name="segment_cardiac_cine",
            arguments={"cine_path": cine, "output_subdir": "segmentation/cardiac_cine"},
            call_id="det_cardiac_segment",
            case_id=case_id,
            stage="segment",
            requested_by="agent_loop:deterministic",
        )
        dispatcher.dispatch(seg_call, state, ctx)

    seg_data = _latest_tool_data(ctx.case_state_path, "segment", "segment_cardiac_cine") or {}
    seg_path = seg_data.get("seg_path") if isinstance(seg_data, dict) else None
    if isinstance(seg_path, str) and seg_path:
        cls_call = ToolCall(
            tool_name="classify_cardiac_cine_disease",
            arguments={
                "seg_path": seg_path,
                "cine_path": cine,
                "output_subdir": "classification/cardiac_cine",
            },
            call_id="det_cardiac_classify",
            case_id=case_id,
            stage="report",
            requested_by="agent_loop:deterministic",
        )
        dispatcher.dispatch(cls_call, state, ctx)

    # Extract ROI features with explicit schema-compliant args to avoid fallback-chain
    # failures when deterministic mode is entered after an upstream model/tool error.
    feat_args: Dict[str, Any] = {
        "output_subdir": "features",
        "roi_interpretation": "cardiac",
        "radiomics_mode": "off",
        "images": [],
        "roi_masks": [],
    }
    if isinstance(cine, str) and cine:
        feat_args["images"].append({"name": "CINE", "path": cine})
    if isinstance(seg_data, dict):
        for key, name in (
            ("rv_mask_path", "RV"),
            ("myo_mask_path", "MYO"),
            ("lv_mask_path", "LV"),
        ):
            p = seg_data.get(key)
            if isinstance(p, str) and p:
                feat_args["roi_masks"].append({"name": name, "path": p})
        if (not feat_args["roi_masks"]) and isinstance(seg_data.get("seg_path"), str) and seg_data.get("seg_path"):
            # Last-resort fallback: use the combined label map as a single ROI.
            feat_args["roi_mask_path"] = seg_data.get("seg_path")
    feat_call = ToolCall(
        tool_name="extract_roi_features",
        arguments=feat_args,
        call_id="det_cardiac_features",
        case_id=case_id,
        stage="extract",
        requested_by="agent_loop:deterministic",
    )
    dispatcher.dispatch(feat_call, state, ctx)

    pkg_call = ToolCall(
        tool_name="package_vlm_evidence",
        arguments={"case_state_path": str(ctx.case_state_path), "output_subdir": "vlm"},
        call_id="det_cardiac_package",
        case_id=case_id,
        stage="report",
        requested_by="agent_loop:deterministic",
    )
    dispatcher.dispatch(pkg_call, state, ctx)


def run_agent_loop(
    goal: str,
    case_id: str,
    dicom_case_dir: Optional[str],
    runs_root: Path,
    llm_mode: str = "stub",
    max_steps: int = 8,
    max_retries: int = 2,
    plan_mode: str = "step",
    server_cfg: Optional[VLLMServerConfig] = None,
    api_model: Optional[str] = None,
    api_base_url: Optional[str] = None,
    finalize_with_llm: bool = False,
    enforce_mvp_pipeline: bool = False,
    autofix_mode: str = "force",  # force|coach|off
    symbolic_binder_mode: str = "off",  # off|token
    enable_preconditions: bool = True,
    enable_tool_reflection: bool = True,
    enable_failure_reflector: bool = False,
    enable_sandbox_exec: bool = True,
    domain: Optional[DomainConfig] = None,
    reactive_prompt_override: Optional[str] = None,
) -> Path:
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    domain_cfg = domain or get_domain_config("prostate")
    registry = build_registry(enable_sandbox_exec=enable_sandbox_exec)
    dispatcher = ToolDispatcher(registry=registry, runs_root=runs_root)
    state, ctx = dispatcher.create_run(case_id=case_id, run_id=run_id)
    try:
        state.metadata = {**(state.metadata or {}), "domain": domain_cfg.name}
        state.write_json(ctx.case_state_path)
    except Exception:
        pass

    repair_fn = partial(_auto_repair_args_for_tool, domain=domain_cfg)

    def _ensure_report_generated() -> Optional[str]:
        """
        Ensure report artifacts exist even when the model exits with action=final early.
        Returns report JSON path when available.
        """
        try:
            call = ToolCall(
                tool_name="generate_report",
                arguments={
                    "case_state_path": str(ctx.case_state_path),
                    "output_subdir": "report",
                    "domain": domain_cfg.name,
                },
                call_id="auto_generate_report",
                case_id=case_id,
                stage="report",
                requested_by="agent_loop:auto",
            )
            res = dispatcher.dispatch(call, state, ctx)
            if res.ok and isinstance(res.data, dict):
                p = res.data.get("report_json_path")
                if isinstance(p, str) and p:
                    return p
        except Exception:
            pass
        return None

    # Pick LLM
    if llm_mode == "stub":
        llm = FakeLLM()
    elif llm_mode == "server":
        llm = VLLMOpenAIChatAdapter(server_cfg)
    elif llm_mode == "openai":
        llm = OpenAIChatAdapter(OpenAIConfig(model=api_model or OpenAIConfig().model, base_url=api_base_url))
    elif llm_mode == "anthropic":
        llm = AnthropicChatAdapter(AnthropicConfig(model=api_model or AnthropicConfig().model))
    elif llm_mode == "gemini":
        llm = GeminiChatAdapter(GeminiConfig(model=api_model or GeminiConfig().model))
    else:
        raise ValueError("Unsupported llm_mode. Use 'stub', 'server', 'openai', 'anthropic', or 'gemini'.")

    failure_reflector: Optional[ReActFailureReflector] = None
    if enable_failure_reflector:
        failure_reflector = ReActFailureReflector(
            llm=llm,
            llm_mode=llm_mode,
            registry=registry,
        )

    tools_list = dispatcher.list_tools()
    tool_names = sorted([t.get("name") for t in tools_list if isinstance(t, dict) and "name" in t])
    tool_index = _compact_tool_index(tools_list)

    # Persist a "tool manifest" artifact for humans (and for any future prompt strategies).
    try:
        write_tool_manifest(tools_list=tools_list, tool_index=tool_index, ctx_dir=ctx.artifacts_dir / "context")
    except Exception:
        pass

    if autofix_mode not in ("force", "coach", "off"):
        autofix_mode = "force"
    if symbolic_binder_mode not in ("off", "token"):
        symbolic_binder_mode = "off"

    def _compact_for_memory(obj: Any, max_chars: int = 2000) -> Any:
        """
        Compact nested dict/list objects so we can store them as short-term memory without blowing up files/prompts.
        """
        try:
            s = json.dumps(obj, ensure_ascii=False)
            if len(s) <= max_chars:
                return obj
            # Fallback: keep head/tail slices for debugging
            return {"_truncated": True, "head": s[: max_chars // 2], "tail": s[-max_chars // 2 :]}
        except Exception:
            return {"_unserializable": True, "type": type(obj).__name__}

    def _extract_key_paths(tool_name: str, tool_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a compact, stable subset of paths/identifiers from a tool_result (truth) for memory_digest.
        """
        out: Dict[str, Any] = {}
        data = tool_result.get("data") if isinstance(tool_result, dict) else None
        if not isinstance(data, dict):
            data = {}

        if tool_name == "ingest_dicom_to_nifti":
            for k in ("series_inventory_path", "dicom_meta_path"):
                if data.get(k):
                    out[k] = data.get(k)
            if isinstance(data.get("nifti_by_series"), dict):
                out["nifti_by_series"] = data.get("nifti_by_series")

        elif tool_name == "identify_sequences":
            if isinstance(data.get("mapping"), dict):
                out["mapping"] = data.get("mapping")

        elif tool_name == "register_to_reference":
            for k in ("transform_path", "resampled_path"):
                if data.get(k):
                    out[k] = data.get(k)
            if isinstance(data.get("resampled_paths"), dict):
                out["resampled_paths"] = data.get("resampled_paths")
            if isinstance(data.get("qc_pngs"), dict):
                out["qc_pngs"] = data.get("qc_pngs")
            if isinstance(data.get("qc_metrics"), dict):
                out["qc_metrics"] = {"method": data.get("qc_metrics", {}).get("method")}

        elif tool_name == "segment_prostate":
            for k in ("prostate_mask_path", "zone_mask_path"):
                if data.get(k):
                    out[k] = data.get(k)
        elif tool_name == "segment_cardiac_cine":
            for k in ("seg_path", "rv_mask_path", "myo_mask_path", "lv_mask_path"):
                if data.get(k):
                    out[k] = data.get(k)
        elif tool_name == "classify_cardiac_cine_disease":
            for k in ("classification_path", "predicted_group"):
                if data.get(k):
                    out[k] = data.get(k)

        elif tool_name == "extract_roi_features":
            for k in ("feature_table_path", "slice_summary_path"):
                if data.get(k):
                    out[k] = data.get(k)
            if isinstance(data.get("overlay_pngs"), list):
                out["overlay_pngs"] = (data.get("overlay_pngs") or [])[:6]

        elif tool_name == "detect_lesion_candidates":
            for k in ("candidates_path", "merged_prob_path", "lesion_mask_path"):
                if data.get(k):
                    out[k] = data.get(k)

        elif tool_name == "correct_prostate_distortion":
            for k in ("out_dir", "summary_json_path", "num_pred_npz", "num_panel_png", "num_slice_png"):
                if data.get(k) is not None:
                    out[k] = data.get(k)

        elif tool_name == "package_vlm_evidence":
            for k in ("vlm_evidence_path",):
                if data.get(k):
                    out[k] = data.get(k)
        elif tool_name == "generate_report":
            for k in ("report_json_path", "report_txt_path", "clinical_report_path", "vlm_evidence_bundle_path"):
                if data.get(k):
                    out[k] = data.get(k)

        return out

    def _write_reflection(step: int, tool_name: str, tool_result: Dict[str, Any]) -> None:
        """
        Optional: ask the LLM for a brief free-text reflection AFTER each tool call.
        This is saved as an artifact and also appended to a memory jsonl file.
        It does NOT change the main JSON-only action channel.
        """
        mem_dir = ctx.artifacts_dir / "memory"
        mem_dir.mkdir(parents=True, exist_ok=True)
        mem_jsonl = mem_dir / "short_term.jsonl"
        reflection_txt = mem_dir / f"reflection_step{step:03d}.txt"

        # Always write a compact machine-readable event (even if llm_mode != server).
        try:
            append_short_term_event(
                artifacts_dir=ctx.artifacts_dir,
                step=step,
                tool_name=str(tool_name),
                tool_result=tool_result,
                max_chars=2000,
            )
        except Exception:
            # keep legacy fallback (best-effort)
            event = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "step": int(step),
                "tool_name": str(tool_name),
                "tool_result_compact": _compact_for_memory(tool_result, max_chars=2000),
                "paths": _extract_key_paths(str(tool_name), tool_result),
            }
            with mem_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

        # Free-text reflection (server mode only)
        if llm_mode != "server":
            return
        try:
            sys = (
                "You are the agent's internal reflector.\n"
                "Write 6-12 lines of free-text reflection about the tool result:\n"
                "- What changed in the case state\n"
                "- Any anomalies / missing evidence\n"
                "- What the next best tool call should be and why\n"
                "Do NOT output JSON. Do NOT output chain-of-thought. Be concise and action-oriented.\n"
            )
            payload = {
                "goal": goal,
                "step": step,
                "tool_name": tool_name,
                "tool_result": _compact_for_memory(tool_result, max_chars=4000),
            }
            refl = llm.generate([{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]).strip()
            # Strip common reasoning wrappers if present.
            refl = re.sub(r"(?is)<think>.*?</think>", "", refl).strip()
            refl = refl.replace("<think>", "").replace("</think>", "").strip()
            if refl:
                reflection_txt.write_text(refl + "\n", encoding="utf-8")
                _write_step_debug(
                    "reflection",
                    step=step,
                    attempt=0,
                    messages_obj={"tool_name": tool_name, "tool_result_compact": payload["tool_result"]},
                    raw_text=refl,
                    parsed_obj=None,
                )
        except Exception:
            return

    def _memory_digest(max_events: int = 6) -> Dict[str, Any]:
        return build_memory_digest(
            case_state_path=Path(ctx.case_state_path),
            artifacts_dir=Path(ctx.artifacts_dir),
            max_events=max_events,
            autofix_mode=autofix_mode,
        )

    def _finalize_free_text_report() -> Path:
        finalize_free_text_report(
            llm=llm,
            llm_mode=llm_mode,
            artifacts_dir=Path(ctx.artifacts_dir),
            run_dir=Path(ctx.run_dir),
            case_id=case_id,
            goal=goal,
            domain_config=domain_cfg,
        )
        # Preserve the old return contract (run_dir)
        state.set_summary(
            {
                "final_report_path": str((ctx.artifacts_dir / "final" / "final_report.json").resolve()),
                "final_report_txt_path": str((ctx.artifacts_dir / "final" / "final_report.txt").resolve()),
            }
        )
        state.write_json(ctx.case_state_path)
        return ctx.run_dir

    def _dbg_dir() -> Path:
        d = ctx.artifacts_dir / "llm_debug"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    trace_path = ctx.run_dir / "agent_trace.jsonl"

    def _write_step_debug(
        tag: str,
        *,
        step: int,
        attempt: int,
        messages_obj: List[Dict[str, Any]],
        raw_text: str,
        parsed_obj: Optional[Dict[str, Any]] = None,
        tool_result_obj: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Persist reactive-loop debug state for inspection.
        This is intentionally verbose (debug-first, not token-frugal).
        """
        d = _dbg_dir()
        prefix = f"{tag}_step{step:03d}_attempt{attempt:02d}"
        (d / f"{prefix}_messages.json").write_text(json.dumps(messages_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (d / f"{prefix}_raw.txt").write_text((raw_text or "") + "\n", encoding="utf-8")
        if parsed_obj is not None:
            (d / f"{prefix}_parsed.json").write_text(json.dumps(parsed_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if tool_result_obj is not None:
            (d / f"{prefix}_tool_result.json").write_text(json.dumps(tool_result_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        _append_jsonl(
            trace_path,
            {
                "ts": datetime.utcnow().isoformat() + "Z",
                "tag": tag,
                "step": int(step),
                "attempt": int(attempt),
                "raw_len": int(len(raw_text or "")),
                "parsed": parsed_obj,
                "tool_result_compact": (
                    {
                        "ok": tool_result_obj.get("ok"),
                        "tool_name": tool_result_obj.get("provenance", {}).get("tool_name")
                        if isinstance(tool_result_obj, dict)
                        else None,
                    }
                    if isinstance(tool_result_obj, dict)
                    else None
                ),
            },
        )

    # One-shot planning path (optional, useful for heavy/slow models)
    if plan_mode == "one_shot":
        # Keep prompt minimal to avoid max_model_len overflows.
        base_context = {
            "goal": goal,
            "case_id": case_id,
            "inputs": {"dicom_case_dir": dicom_case_dir} if dicom_case_dir else {},
            "tool_names": tool_names,
            "tool_index": tool_index,
            "skills_summary": skills_summary,
            "placeholders": {
                "__LATEST__/ingest/series_inventory.json": "Use after ingest to reference the latest inventory artifact.",
                "__LATEST__/ingest/dicom_meta.json": "Use after ingest to reference the latest dicom meta artifact.",
            },
        }

        # Pass 1: free-text plan/thought (saved, not parsed)
        thought_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": THOUGHT_PROMPT},
            {"role": "user", "content": json.dumps(base_context, ensure_ascii=False)},
        ]

        def _write_llm_debug(tag: str, *, messages_obj: List[Dict[str, Any]], raw_text: str) -> None:
            dbg = ctx.artifacts_dir / "llm_debug"
            dbg.mkdir(parents=True, exist_ok=True)
            (dbg / f"{tag}_messages.json").write_text(json.dumps(messages_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (dbg / f"{tag}_raw.txt").write_text(raw_text + "\n", encoding="utf-8")

        thought_text = ""
        if llm_mode in ("server", "openai", "anthropic", "gemini"):
            try:
                thought_text = llm.generate(thought_messages)
                _write_llm_debug("thought", messages_obj=thought_messages, raw_text=thought_text)
            except Exception:
                thought_text = ""

        # Pass 2: strict tool_calls JSON using the thought as context
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": ONE_SHOT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        **base_context,
                        "thought_plan": thought_text,
                        "instruction": "Return action=tool_calls with a short plan to accomplish the goal. Use only tool_names.",
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        # Call model with retry/repair
        text = ""
        parsed = None
        err = None
        for attempt in range(max_retries + 1):
            try:
                # If we are using a vLLM server adapter, prefer guided JSON for the one-shot plan.
                if hasattr(llm, "generate_with_schema") and llm_mode in ("server", "openai"):
                    guided = {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["tool_calls"]},
                            "calls": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "stage": {"type": "string"},
                                        "tool_name": {"type": "string"},
                                        "arguments": {"type": "object"},
                                    },
                                    "required": ["tool_name", "arguments"],
                                    "additionalProperties": False,
                                },
                                "minItems": 1,
                            },
                        },
                        "required": ["action", "calls"],
                        "additionalProperties": False,
                    }
                    text = llm.generate_with_schema(messages, guided)  # type: ignore[attr-defined]
                else:
                    text = llm.generate(messages)
            except Exception as e:
                # If the LLM server is down, keep the run going via deterministic MVP fallback.
                _write_llm_debug("plan_server_error", messages_obj=messages, raw_text=repr(e))
                if domain_cfg.name == "cardiac":
                    _run_deterministic_cardiac_mvp(
                        dispatcher,
                        state,
                        ctx,
                        case_id=case_id,
                        dicom_case_dir=dicom_case_dir,
                    )
                else:
                    _run_deterministic_prostate_mvp(
                        dispatcher,
                        state,
                        ctx,
                        case_id=case_id,
                        dicom_case_dir=dicom_case_dir,
                        llm_mode=llm_mode,
                        server_cfg=server_cfg,
                    )
                return ctx.run_dir

            _write_llm_debug(f"plan_attempt{attempt}", messages_obj=messages, raw_text=text)
            parsed, err = parse_model_output(text)
            if not err:
                break
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "error": err,
                            "repair_instructions": "Return ONLY one valid JSON object that matches the contract. Do not include any extra text.",
                            "allowed_actions": ["tool_calls", "final"],
                            "preferred_action": "tool_calls",
                        },
                        ensure_ascii=False,
                    ),
                }
            )
        if err:
            # Keep the last raw output for debugging.
            _write_llm_debug("plan_failed", messages_obj=messages, raw_text=text)
            raise RuntimeError(f"Model output invalid after retries. Last error: {err}")

        if not parsed:
            raise RuntimeError("Unexpected: parsed action is None")
        if parsed.action == "final":
            report_json_path = _ensure_report_generated()
            final_dir = ctx.artifacts_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            final_json = final_dir / "final_report.json"
            payload = parsed.payload if isinstance(parsed.payload, dict) else {"final_report": parsed.payload}
            if report_json_path:
                payload = dict(payload)
                payload.setdefault("generated_report_json_path", report_json_path)
            final_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            state.set_summary({"final_report_path": str(final_json)})
            state.write_json(ctx.case_state_path)
            return ctx.run_dir

        calls: List[Dict[str, Any]]
        if parsed.action == "tool_calls":
            calls = parsed.payload.get("calls", [])
            if not isinstance(calls, list):
                raise RuntimeError("tool_calls payload missing calls list")
        elif parsed.action == "tool_call":
            calls = [
                {
                    "tool_name": parsed.payload["tool_name"],
                    "arguments": parsed.payload["arguments"],
                    "stage": parsed.payload.get("stage", "misc"),
                }
            ]
        else:
            raise RuntimeError(f"One-shot mode expects action=tool_call/tool_calls/final, got: {parsed.action}")

        results_compact = []
        for i, c in enumerate(calls):
            args = _rewrite_placeholders(c["arguments"], ctx.artifacts_dir)
            args = repair_fn(
                c["tool_name"],
                args,
                state_path=ctx.case_state_path,
                ctx_case_state_path=ctx.case_state_path,
                dicom_case_dir=dicom_case_dir,
            )
            # Force canonical semantic stage names (avoid numeric/legacy stages from the LLM).
            tool_stage_defaults = {
                "ingest_dicom_to_nifti": "ingest",
                "identify_sequences": "identify",
                "register_to_reference": "register",
                "segment_prostate": "segment",
                "brats_mri_segmentation": "segment",
                "segment_cardiac_cine": "segment",
                "classify_cardiac_cine_disease": "report",
                "extract_roi_features": "extract",
                "correct_prostate_distortion": "register",
                "package_vlm_evidence": "report",
                "generate_report": "report",
            }
            stage = str(c.get("stage", "") or "").strip().lower()
            if stage.isdigit():
                stage = ""
            stage = stage or tool_stage_defaults.get(c["tool_name"], "misc")
            call = ToolCall(
                tool_name=c["tool_name"],
                arguments=args,
                call_id=f"llm_plan_{i:03d}",
                case_id=case_id,
                stage=stage,
                requested_by="agent_loop",
            )
            result = dispatcher.dispatch(call, state, ctx)
            if not result.ok and call.tool_name in ("ingest_dicom_to_nifti", "identify_sequences"):
                # Hard stop: without ingest/identify, later stages can't be resolved reliably.
                final_dir = ctx.artifacts_dir / "final"
                final_dir.mkdir(parents=True, exist_ok=True)
                final_json = final_dir / "final_report.json"
                final_json.write_text(
                    json.dumps(
                        {
                            "final_report": {
                                "status": "error",
                                "failed_tool": call.tool_name,
                                "call_id": call.call_id,
                                "error": result.error.to_dict() if result.error else None,
                                "warnings": result.warnings,
                            }
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                state.set_summary({"final_report_path": str(final_json)})
                state.write_json(ctx.case_state_path)
                return ctx.run_dir
            results_compact.append(
                {
                    "tool_name": call.tool_name,
                    "ok": result.ok,
                    "warnings": result.warnings,
                    "data_keys": sorted(list(result.data.keys())),
                    "artifacts": [a.to_dict() for a in result.artifacts[:8]],
                }
            )

        # Optional: enforce the fixed MVP pipeline (disabled by default in agent/reactive mode).
        if enforce_mvp_pipeline:
            if domain_cfg.name == "prostate":
                _ensure_required_registrations(dispatcher, state, ctx, case_id=case_id)
                _ensure_segmentation_and_features_and_report(
                    dispatcher,
                    state,
                    ctx,
                    case_id=case_id,
                    llm_mode=llm_mode,
                    server_cfg=server_cfg,
                )
            elif domain_cfg.name == "cardiac":
                _run_deterministic_cardiac_mvp(
                    dispatcher,
                    state,
                    ctx,
                    case_id=case_id,
                    dicom_case_dir=dicom_case_dir,
                )

        # Optionally ask model to produce a FREE-TEXT report after executing the plan.
        # This is not a tool: tools produce evidence; the LLM synthesizes the narrative.
        if finalize_with_llm and llm_mode in ("server", "openai", "anthropic", "gemini"):
            final_dir = ctx.artifacts_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)

            def _encode_png_data_url(path: str, max_bytes: int = 2_500_000) -> Optional[str]:
                p = Path(path).expanduser().resolve()
                if not p.exists():
                    return None
                try:
                    b = p.read_bytes()
                except Exception:
                    return None
                if len(b) > max_bytes:
                    return None
                return "data:image/png;base64," + base64.b64encode(b).decode("ascii")

            # Ensure evidence bundle exists; if not, auto-package it.
            evidence_path = ctx.artifacts_dir / "vlm" / "vlm_evidence_bundle.json"
            if not evidence_path.exists():
                try:
                    pkg_call = ToolCall(
                        tool_name="package_vlm_evidence",
                        arguments={"case_state_path": str(ctx.case_state_path), "output_subdir": "vlm"},
                        call_id="auto_package_before_final",
                        case_id=case_id,
                        stage="report",
                        requested_by="agent_loop:auto",
                    )
                    dispatcher.dispatch(pkg_call, state, ctx)
                except Exception:
                    pass

            evidence_bundle: Dict[str, Any] = {}
            try:
                if evidence_path.exists():
                    evidence_bundle = json.loads(evidence_path.read_text(encoding="utf-8"))
            except Exception:
                evidence_bundle = {}

            # Pick up to 4 montage images (T2w/ADC/high_b/low_b) for the VLM.
            overlay_list = (
                (evidence_bundle.get("features") or {}).get("overlay_pngs") if isinstance(evidence_bundle, dict) else None
            )
            overlays: List[Dict[str, str]] = [o for o in (overlay_list or []) if isinstance(o, dict)]
            lesion_overlay_list = (
                (evidence_bundle.get("features_lesion") or {}).get("overlay_pngs") if isinstance(evidence_bundle, dict) else None
            )
            overlays.extend([o for o in (lesion_overlay_list or []) if isinstance(o, dict)])
            # De-duplicate by path.
            seen_paths: set[str] = set()
            deduped: List[Dict[str, str]] = []
            for o in overlays:
                p = str(o.get("path") or "")
                if not p or p in seen_paths:
                    continue
                seen_paths.add(p)
                deduped.append(o)
            overlays = deduped

            def _bucket(seq: str, path: str) -> str:
                s = (seq + " " + path).lower()
                if "t2w" in s:
                    return "t2w"
                if "adc" in s:
                    return "adc"
                if "high_b" in s or "high-b" in s:
                    return "dwi_high_b"
                if "low_b" in s or "low-b" in s:
                    return "dwi_low_b"
                if "dwi" in s:
                    return "dwi_other"
                return "other"

            montages = [
                o
                for o in overlays
                if str(o.get("kind") or "") in {"roi_crop_montage_5x5", "roi_crop_montage_4x4", "full_frame_montage_4x4", "full_frame_montage_5x5"}
                and o.get("path")
            ]
            bucket_order = ["t2w", "adc", "dwi_high_b", "dwi_low_b"]
            picked: Dict[str, str] = {}
            for o in montages:
                p = str(o.get("path"))
                seq = str(o.get("sequence") or "")
                b = _bucket(seq, p)
                if b in bucket_order and b not in picked:
                    picked[b] = p
            chosen_paths: List[str] = [picked[b] for b in bucket_order if b in picked][:4]

            img_blocks: List[Dict[str, Any]] = []
            embedded_images_debug: List[Dict[str, Any]] = []
            for p in chosen_paths:
                data_url = _encode_png_data_url(p)
                if data_url:
                    img_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
                    embedded_images_debug.append({"path": p, "embedded": True})
                else:
                    embedded_images_debug.append({"path": p, "embedded": False})

            # Prompt the VLM to write a PI-RADS v2.1 *style* report without forcing a JSON schema.
            # (We do not hardcode the PI-RADS document; we enforce structure + evidence-grounding.)
            system_prompt = (domain_cfg.vlm_system_prompt or get_domain_config("prostate").vlm_system_prompt)
            user_payload = {
                "goal": goal,
                "case_id": case_id,
                "evidence_bundle_path": str(evidence_path) if evidence_path.exists() else None,
                "evidence_bundle": evidence_bundle,
                "images_included": chosen_paths,
                "embedded_images_debug": embedded_images_debug,
                "required_sections": (
                    domain_cfg.report_structure or get_domain_config("prostate").report_structure
                ),
            }

            final_messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)}] + img_blocks,
                },
            ]

            # Persist request for debugging (without huge base64 blobs).
            try:
                dbg = ctx.artifacts_dir / "vlm"
                dbg.mkdir(parents=True, exist_ok=True)
                redacted = json.loads(json.dumps(final_messages))
                for msg in redacted:
                    if not isinstance(msg, dict):
                        continue
                    c = msg.get("content")
                    if isinstance(c, list):
                        for blk in c:
                            if isinstance(blk, dict) and blk.get("type") == "image_url":
                                iu = blk.get("image_url")
                                if isinstance(iu, dict) and isinstance(iu.get("url"), str):
                                    blk["image_url"]["url"] = "<data_url omitted>"
                (dbg / "final_report_request_messages.json").write_text(
                    json.dumps(redacted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
            except Exception:
                pass

            try:
                text = llm.generate(final_messages)
            except Exception as e:
                text = ""
                _write_llm_debug("final_free_text_error", messages_obj=final_messages, raw_text=repr(e))
            _write_llm_debug("final_free_text", messages_obj=final_messages, raw_text=text)

            final_txt = final_dir / "final_report.txt"
            final_txt.write_text((text or "").strip() + "\n", encoding="utf-8")
            final_json = final_dir / "final_report.json"
            final_json.write_text(
                json.dumps(
                    {
                        "final_report": {
                            "status": "completed",
                            "run_dir": str(ctx.run_dir),
                            "evidence_bundle_path": str(evidence_path) if evidence_path.exists() else None,
                            "final_report_txt_path": str(final_txt),
                        }
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            state.set_summary({"final_report_path": str(final_json), "final_report_txt_path": str(final_txt)})
            state.write_json(ctx.case_state_path)
            return ctx.run_dir

        # Default deterministic final artifact
        report_json_path = _ensure_report_generated()
        final_dir = ctx.artifacts_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_json = final_dir / "final_report.json"
        final_json.write_text(
            json.dumps(
                {
                    "final_report": {
                        "status": "ok",
                        "note": "Executed tool_calls plan.",
                        "run_dir": str(ctx.run_dir),
                        "generated_report_json_path": report_json_path,
                    }
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        state.set_summary({"final_report_path": str(final_json)})
        state.write_json(ctx.case_state_path)
        return ctx.run_dir
    skills_path = Path(__file__).resolve().parents[1] / "config" / "skills.json"
    skills_registry = SkillRegistry.load(skills_path)
    skills_summary = skills_registry.to_prompt(set_name=None, tags=[domain_cfg.name], max_items=8)
    skills_guidance = skills_registry.to_guidance(set_name=None, tags=[domain_cfg.name], max_items=8)

    def _clip_text(text: Any, *, max_chars: int) -> str:
        s = str(text or "")
        if len(s) <= max_chars:
            return s
        return s[:max_chars].rstrip() + "\n[TRUNCATED]"

    def _compact_prompt_payload(obj: Any, *, max_chars: int) -> Any:
        compact = _compact_for_memory(obj, max_chars=max_chars)
        if isinstance(compact, (dict, list, str, int, float, bool)) or compact is None:
            return compact
        return {"value": str(compact)}

    _effective_system_prompt = reactive_prompt_override or REACTIVE_SYSTEM_PROMPT

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _effective_system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "goal": goal,
                    "case_id": case_id,
                    "inputs": {"dicom_case_dir": dicom_case_dir} if dicom_case_dir else {},
                    "tool_names": tool_names,
                    # Keep specs small-ish but still useful
                    "tool_index": _compact_prompt_payload(tool_index, max_chars=3200),
                    "tool_manifest": str(ctx.artifacts_dir / "context" / "tool_manifest.md"),
                    "skills_summary": _clip_text(skills_summary, max_chars=1600),
                    "skills_guidance": _clip_text(skills_guidance, max_chars=1600),
                    "note": "Choose one next tool_call at a time. Use tool_result observations to decide the next step.",
                },
                ensure_ascii=False,
            ),
        },
    ]

    last_parsed_obj: Optional[Dict[str, Any]] = None
    last_tool_result: Optional[Dict[str, Any]] = None
    last_tool_name: Optional[str] = None
    last_tool_ok: Optional[bool] = None
    tool_step_history: List[Dict[str, Any]] = []
    failure_reflection_decisions: List[Dict[str, Any]] = []
    lesion_failed_once: bool = False
    consecutive_fail_tool: Optional[str] = None
    consecutive_fail_count: int = 0

    for step in range(max_steps):
        # Provide state summary to keep context small
        state_summary = summarize_case_state(ctx.case_state_path)
        memory_digest = _memory_digest(max_events=4)
        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "state_summary": _compact_prompt_payload(state_summary, max_chars=1800),
                        "memory_digest": _compact_prompt_payload(memory_digest, max_chars=2200),
                        "skills_guidance": _clip_text(skills_guidance, max_chars=1200),
                    },
                    ensure_ascii=False,
                ),
            }
        )

        # LLM call with retry/repair
        text = ""
        parsed = None
        err = None

        def _trim_reactive_messages(msgs: List[Dict[str, Any]], *, keep_head: int = 2, keep_tail: int = 14, max_chars: int = 80000) -> List[Dict[str, Any]]:
            if len(msgs) <= keep_head + keep_tail:
                return msgs
            head = msgs[:keep_head]
            tail = msgs[-keep_tail:]
            trimmed = head + tail
            try:
                s = json.dumps(trimmed, ensure_ascii=False)
                if len(s) <= max_chars:
                    return trimmed
                # If still too large, reduce tail aggressively.
                for n in (10, 8, 6, 4):
                    t2 = head + msgs[-n:]
                    if len(json.dumps(t2, ensure_ascii=False)) <= max_chars:
                        return t2
            except Exception:
                pass
            return head + msgs[-4:]

        def _tool_feedback_for_llm(tool_name: str, tool_result_obj: Dict[str, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = {
                "ok": bool(tool_result_obj.get("ok")),
                "warnings": list(tool_result_obj.get("warnings") or [])[:3],
            }
            err_obj = tool_result_obj.get("error")
            if isinstance(err_obj, dict):
                out["error"] = {
                    "type": err_obj.get("type"),
                    "message": str(err_obj.get("message") or "")[:500],
                }
            facts = _extract_key_paths(tool_name, tool_result_obj)
            if facts:
                out["facts"] = _compact_prompt_payload(facts, max_chars=900)
            # Keep a tiny data preview when there are no path-like facts.
            if (not facts) and isinstance(tool_result_obj.get("data"), dict):
                data = tool_result_obj.get("data") or {}
                small: Dict[str, Any] = {}
                for k in ("predicted_group", "classification_path", "feature_table_path", "slice_summary_path", "seg_path"):
                    if k in data:
                        small[k] = data.get(k)
                if small:
                    out["data_preview"] = small
            return out

        def _llm_call_with_context_guard(base_messages: List[Dict[str, Any]]) -> str:
            def _run(msgs: List[Dict[str, Any]]) -> str:
                if hasattr(llm, "generate_with_schema") and llm_mode in ("server", "openai"):
                    guided_step = {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["tool_call", "final"]},
                            "stage": {"type": "string"},
                            "tool_name": {"type": "string"},
                            "arguments": {"type": "object"},
                            "final_report": {"type": "object"},
                        },
                        "required": ["action"],
                        "additionalProperties": True,
                    }
                    return llm.generate_with_schema(msgs, guided_step)  # type: ignore[attr-defined]
                return llm.generate(msgs)

            try:
                return _run(base_messages)
            except Exception as first_exc:
                msg = str(first_exc).lower()
                context_like = (
                    "maximum context length" in msg
                    or "context window exhausted" in msg
                    or "max_tokens" in msg
                    or "max_completion_tokens" in msg
                    or "too large" in msg
                )
                if not context_like:
                    raise
                last_exc: Exception = first_exc
                for lim in (60000, 40000, 24000):
                    try:
                        retrimmed = _trim_reactive_messages(messages, keep_head=2, keep_tail=12, max_chars=lim)
                        return _run(retrimmed)
                    except Exception as retry_exc:
                        last_exc = retry_exc
                raise last_exc

        for attempt in range(max_retries + 1):
            llm_messages = _trim_reactive_messages(messages)
            try:
                text = _llm_call_with_context_guard(llm_messages)
            except Exception as e:
                # Explicitly surface LLM server/API failures.
                _write_step_debug(
                    "reactive_server_error",
                    step=step,
                    attempt=attempt,
                    messages_obj=llm_messages,
                    raw_text=repr(e),
                    parsed_obj={"error": repr(e)},
                )
                raise RuntimeError(
                    "LLM routing failed in reactive loop "
                    f"(run_dir={ctx.run_dir}, step={step}, attempt={attempt}): {repr(e)}"
                ) from e
            parsed, err = parse_model_output(text)
            last_parsed_obj = parsed.payload if parsed else None
            parsed_obj_for_log: Optional[Dict[str, Any]] = None
            if parsed is not None:
                parsed_obj_for_log = {"action": parsed.action, **(parsed.payload if isinstance(parsed.payload, dict) else {"payload": parsed.payload})}
            _write_step_debug(
                "reactive",
                step=step,
                attempt=attempt,
                messages_obj=llm_messages,
                raw_text=text,
                parsed_obj=parsed_obj_for_log if (parsed_obj_for_log or err) else ({"error": err} if err else None),
            )
            if not err:
                break
            # Ask the model to repair its output deterministically
            messages.append({"role": "assistant", "content": _clip_text(text, max_chars=1200)})
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "error": err,
                            "repair_instructions": "Return ONLY one valid JSON object that matches the contract. Do not include any extra text.",
                            "allowed_actions": ["tool_call", "tool_calls", "final"],
                        },
                        ensure_ascii=False,
                    ),
                }
            )
        if err:
            raise RuntimeError(f"Model output invalid after retries. Last error: {err}")

        if parsed and parsed.action == "final":
            # In step-mode, treat model 'final' as a stop-signal, but produce the actual report
            # via our own finalization (free text when --finalize-with-llm).
            _write_step_debug(
                "reactive_model_final",
                step=step,
                attempt=0,
                messages_obj=messages,
                raw_text=text,
                parsed_obj={"action": "final", "final_report": parsed.payload},
            )
            report_json_path = _ensure_report_generated()
            if finalize_with_llm and llm_mode in ("server", "openai", "anthropic", "gemini"):
                return _finalize_free_text_report()
            # Fallback: persist the model JSON as-is
            final_dir = ctx.artifacts_dir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            final_json = final_dir / "final_report.json"
            final_txt = final_dir / "final_report.txt"
            payload = parsed.payload if isinstance(parsed.payload, dict) else {"final_report": parsed.payload}
            if report_json_path:
                payload = dict(payload)
                payload.setdefault("generated_report_json_path", report_json_path)
            final_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            final_txt.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            state.set_summary({"final_report_path": str(final_json)})
            state.write_json(ctx.case_state_path)
            return ctx.run_dir

        if parsed and parsed.action == "tool_calls":
            # Reactive agent mode forbids tool_calls. Ask for repair.
            messages.append({"role": "assistant", "content": _clip_text(text, max_chars=1200)})
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "error": "Reactive agent mode forbids action=tool_calls. Return exactly one action=tool_call OR action=final.",
                            "allowed_actions": ["tool_call", "final"],
                            "repair_instructions": "Return ONLY one valid JSON object. No extra text.",
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            continue

        if parsed and parsed.action == "tool_call":
            # Circuit-breaker: lesion detection is "nice to have" but can fail due to model/runtime issues.
            # If it already failed once, do NOT keep retrying it (common reactive loop failure mode).
            if lesion_failed_once and parsed.payload.get("tool_name") == "detect_lesion_candidates":
                parsed.payload["tool_name"] = "package_vlm_evidence"
                parsed.payload["stage"] = "package"
                parsed.payload["arguments"] = {"case_state_path": str(ctx.case_state_path)}
            # Avoid repeating segmentation when it already succeeded.
            if (
                last_tool_name in ("brats_mri_segmentation", "segment_prostate", "segment_cardiac_cine", "classify_cardiac_cine_disease")
                and last_tool_ok is True
                and parsed.payload.get("tool_name") == last_tool_name
            ):
                parsed.payload["tool_name"] = "package_vlm_evidence"
                parsed.payload["stage"] = "package"
                parsed.payload["arguments"] = {"case_state_path": str(ctx.case_state_path)}

            tool_name_eff = str(parsed.payload.get("tool_name") or "")
            stage_eff = str(parsed.payload.get("stage", "misc") or "misc")
            args0 = _rewrite_placeholders(parsed.payload["arguments"], ctx.artifacts_dir)
            preconditions_applied: List[Dict[str, Any]] = []
            rule_violations = []
            try:
                probe_call = ToolCall(
                    tool_name=tool_name_eff,
                    arguments=args0,
                    call_id="policy_probe",
                    case_id=case_id,
                    stage=stage_eff,
                    requested_by="agent_loop:policy",
                )
                rule_violations = validate_tool_call(probe_call, Path(ctx.case_state_path))
            except Exception:
                rule_violations = []

            # Preconditions via hooks (optional): keep runs usable even when the LLM forgets key steps.
            # NOTE: preconditions are now independent of autofix_mode so that
            # react_token can auto-run identify_sequences (to populate mappings
            # for the symbolic binder) without also auto-repairing arguments.
            if enable_preconditions:
                _pre_autofix = autofix_mode if autofix_mode != "off" else "force"
                try:
                    pre = apply_preconditions(
                        probe_call,
                        case_state_path=Path(ctx.case_state_path),
                        dicom_case_dir=dicom_case_dir,
                        autofix_mode=_pre_autofix,
                        strategy="replace",
                        auto_repair_fn=repair_fn,
                    )
                    for pre_call in pre.pre_calls:
                        dispatcher.dispatch(pre_call, state, ctx)
                    preconditions_applied = pre.notes
                    if isinstance(pre.call, ToolCall):
                        tool_name_eff = str(pre.call.tool_name or tool_name_eff)
                        stage_eff = str(pre.call.stage or stage_eff)
                        if isinstance(pre.call.arguments, dict):
                            args0 = _rewrite_placeholders(pre.call.arguments, ctx.artifacts_dir)
                except Exception:
                    preconditions_applied = []

            if symbolic_binder_mode == "token":
                try:
                    args0 = _apply_symbolic_binder(
                        args0,
                        case_state_path=Path(ctx.case_state_path),
                        dicom_case_dir=dicom_case_dir,
                    )
                except Exception:
                    pass

            repaired = repair_fn(
                tool_name_eff,
                args0,
                state_path=ctx.case_state_path,
                ctx_case_state_path=ctx.case_state_path,
                dicom_case_dir=dicom_case_dir,
            )

            def _arg_diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
                changed: Dict[str, Any] = {}
                keys = set(a.keys()) | set(b.keys())
                for k in sorted(keys):
                    av = a.get(k)
                    bv = b.get(k)
                    if av != bv:
                        changed[k] = {"from": av, "to": bv}
                return changed

            autofix_changed = _arg_diff(args0, repaired) if (isinstance(args0, dict) and isinstance(repaired, dict)) else {}
            if autofix_mode == "off":
                args_final = args0
            else:
                args_final = repaired
            call = ToolCall(
                tool_name=tool_name_eff,
                arguments=args_final,
                call_id=f"llm_call_{step:03d}",
                case_id=case_id,
                stage=stage_eff,
                requested_by="agent_loop",
            )
            result = dispatcher.dispatch(call, state, ctx)
            schema_autofix_applied: Optional[Dict[str, Any]] = None
            if symbolic_binder_mode == "token" and (not bool(result.ok)):
                err_type, err_message = _tool_error_type_message(result)
                missing_keys = _extract_missing_required_args(err_type, err_message)
                if missing_keys:
                    patch = _build_schema_repair_patch(
                        tool_name=tool_name_eff,
                        args_current=args_final if isinstance(args_final, dict) else {},
                        missing_keys=missing_keys,
                        state_path=Path(ctx.case_state_path),
                        dicom_case_dir=dicom_case_dir,
                    )
                    if patch:
                        retry_seed = dict(args_final if isinstance(args_final, dict) else {})
                        retry_seed.update(patch)
                        try:
                            retry_seed = _rewrite_placeholders(retry_seed, ctx.artifacts_dir)
                            retry_seed = _apply_symbolic_binder(
                                retry_seed,
                                case_state_path=Path(ctx.case_state_path),
                                dicom_case_dir=dicom_case_dir,
                            )
                        except Exception:
                            pass
                        retry_args = repair_fn(
                            tool_name_eff,
                            retry_seed,
                            state_path=ctx.case_state_path,
                            ctx_case_state_path=ctx.case_state_path,
                            dicom_case_dir=dicom_case_dir,
                        )
                        if isinstance(retry_args, dict) and retry_args != args_final:
                            retry_call = ToolCall(
                                tool_name=tool_name_eff,
                                arguments=retry_args,
                                call_id=f"{call.call_id}_schemafix",
                                case_id=case_id,
                                stage=stage_eff,
                                requested_by="agent_loop:schema_autofix",
                            )
                            retry_result = dispatcher.dispatch(retry_call, state, ctx)
                            schema_autofix_applied = {
                                "tool_name": tool_name_eff,
                                "trigger_error_type": err_type,
                                "trigger_error_message": err_message,
                                "missing_keys": missing_keys,
                                "patched_keys": sorted(list(patch.keys())),
                                "from_arguments": args_final,
                                "to_arguments": retry_args,
                                "retry_ok": bool(retry_result.ok),
                            }
                            call = retry_call
                            result = retry_result
            failure_reflection_feedback: Optional[Dict[str, Any]] = None
            if (
                failure_reflector is not None
                and symbolic_binder_mode == "token"
                and (not bool(result.ok))
            ):
                reflection_out = failure_reflector.reflect(
                    step=step,
                    tool_name=str(call.tool_name or ""),
                    stage=str(call.stage or stage_eff),
                    executed_args=dict(call.arguments or {}),
                    tool_result=result.to_dict(),
                    case_state_path=Path(ctx.case_state_path),
                    dicom_case_dir=dicom_case_dir,
                    step_history=tool_step_history,
                    schema_autofix_applied=schema_autofix_applied,
                    preconditions_applied=preconditions_applied,
                    rule_violations=[
                        {
                            "rule_id": v.rule_id,
                            "level": v.level,
                            "tool_name": v.tool_name,
                            "message": v.message,
                            "details": v.details or {},
                        }
                        for v in rule_violations
                    ]
                    if rule_violations
                    else None,
                )
                reflection_dec = (
                    dict(reflection_out.get("decision") or {})
                    if isinstance(reflection_out.get("decision"), dict)
                    else {}
                )
                failure_reflection_feedback = (
                    dict(reflection_out.get("feedback") or {})
                    if isinstance(reflection_out.get("feedback"), dict)
                    else None
                )
                reflection_trace = (
                    dict(reflection_out.get("trace") or {})
                    if isinstance(reflection_out.get("trace"), dict)
                    else {}
                )
                refl_action = str(reflection_dec.get("action") or "skip").strip().lower()
                if refl_action not in {"retry", "skip", "halt"}:
                    refl_action = "skip"
                    reflection_dec["action"] = "skip"

                if refl_action == "retry":
                    retry_patch = (
                        dict(reflection_dec.get("retry_arguments") or {})
                        if isinstance(reflection_dec.get("retry_arguments"), dict)
                        else {}
                    )
                    retry_ok = False
                    retry_patch_applied = False
                    retry_precheck_issues: List[str] = []
                    retry_normalized_keys: List[str] = []
                    if retry_patch:
                        retry_args = dict(call.arguments or {})
                        retry_args.update(retry_patch)
                        try:
                            retry_args = _rewrite_placeholders(retry_args, ctx.artifacts_dir)
                            retry_args = _apply_symbolic_binder(
                                retry_args,
                                case_state_path=Path(ctx.case_state_path),
                                dicom_case_dir=dicom_case_dir,
                            )
                            retry_args, retry_precheck_issues, retry_normalized_keys = _normalize_and_validate_reflector_retry_args(
                                current_args=dict(call.arguments or {}),
                                retry_args=retry_args,
                                retry_patch=retry_patch,
                                run_dir=ctx.run_dir,
                                artifacts_dir=ctx.artifacts_dir,
                            )
                        except Exception:
                            pass
                        if retry_precheck_issues:
                            reflection_dec["action"] = "skip"
                            reflection_dec["reason"] = (
                                f"reflector_retry_precheck_blocked: {'; '.join(retry_precheck_issues[:2])}"
                            )
                        elif retry_args != dict(call.arguments or {}):
                            retry_call = ToolCall(
                                tool_name=str(call.tool_name or ""),
                                arguments=retry_args,
                                call_id=f"{call.call_id}_reflectretry",
                                case_id=case_id,
                                stage=str(call.stage or stage_eff),
                                requested_by="agent_loop:failure_reflector",
                            )
                            retry_result = dispatcher.dispatch(retry_call, state, ctx)
                            call = retry_call
                            result = retry_result
                            retry_ok = bool(retry_result.ok)
                            retry_patch_applied = True
                    reflection_dec["retry_attempted"] = bool(retry_patch_applied)
                    reflection_dec["retry_ok"] = bool(retry_ok) if retry_patch_applied else False
                    if not retry_patch_applied:
                        reflection_dec["action"] = "skip"
                        reflection_dec["reason"] = str(
                            reflection_dec.get("reason")
                            or "reflector_retry_without_effective_patch"
                        )
                    if failure_reflection_feedback and isinstance(failure_reflection_feedback.get("failure_reflection"), dict):
                        fref = failure_reflection_feedback["failure_reflection"]
                        decision_obj = fref.get("decision") if isinstance(fref.get("decision"), dict) else {}
                        if isinstance(decision_obj, dict):
                            decision_obj["retry_attempted"] = bool(retry_patch_applied)
                            decision_obj["retry_ok"] = bool(retry_ok) if retry_patch_applied else False
                            if not retry_patch_applied:
                                decision_obj["action"] = "skip"
                            if (not retry_patch_applied) and retry_patch:
                                decision_obj["note"] = "Reflector proposed retry patch but it did not change arguments after token binding."
                            elif not retry_patch:
                                decision_obj["note"] = "Reflector chose retry but no corrected_args were returned."
                            elif retry_precheck_issues:
                                decision_obj["note"] = "Reflector retry patch was blocked by precheck (invalid/nonexistent input path after normalization)."
                                decision_obj["precheck_issues"] = retry_precheck_issues[:2]
                            elif retry_normalized_keys:
                                decision_obj["normalized_path_keys"] = retry_normalized_keys[:4]
                elif refl_action == "halt":
                    final_dir = ctx.artifacts_dir / "final"
                    final_dir.mkdir(parents=True, exist_ok=True)
                    final_json = final_dir / "final_report.json"
                    final_payload = {
                        "final_report": {
                            "status": "halted_by_failure_reflector",
                            "note": str(reflection_dec.get("natural_language_response") or reflection_dec.get("reason") or "hard boundary"),
                            "run_dir": str(ctx.run_dir),
                            "failing_tool": str(call.tool_name or ""),
                            "failing_stage": str(call.stage or stage_eff),
                            "failure_reflection": {
                                "action": "halt",
                                "reason": str(reflection_dec.get("reason") or ""),
                                "failure_classification": str(reflection_dec.get("failure_classification") or ""),
                            },
                        }
                    }
                    final_json.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    state.set_summary({"final_report_path": str(final_json)})
                    state.write_json(ctx.case_state_path)
                reflection_rec = {
                    "step": int(step),
                    "tool_name": str(call.tool_name or ""),
                    "stage": str(call.stage or stage_eff),
                    "action": str(reflection_dec.get("action") or refl_action or "skip"),
                    "reason": str(reflection_dec.get("reason") or ""),
                    "natural_language_response": str(reflection_dec.get("natural_language_response") or ""),
                    "failure_classification": str(reflection_dec.get("failure_classification") or ""),
                    "retry_attempted": bool(reflection_dec.get("retry_attempted")),
                    "retry_ok": bool(reflection_dec.get("retry_ok")),
                }
                failure_reflection_decisions.append(reflection_rec)
                _append_jsonl(
                    trace_path,
                    {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "tag": "failure_reflection",
                        "step": int(step),
                        "tool_name": str(call.tool_name or ""),
                        "decision": reflection_rec,
                        "trace": reflection_trace,
                    },
                )
                if refl_action == "halt":
                    return ctx.run_dir
            last_tool_result = result.to_dict()
            last_tool_name = call.tool_name
            last_tool_ok = bool(result.ok)
            tool_step_history.append(
                {
                    "node_id": f"react_step_{step:03d}_{len(tool_step_history):03d}",
                    "tool_name": str(call.tool_name or ""),
                    "status": ("DONE" if bool(result.ok) else "FAIL"),
                    "data": dict(result.data or {}) if isinstance(result.data, dict) else {},
                    "error": (result.error.to_dict() if getattr(result, "error", None) else {}),
                    "arguments": dict(call.arguments or {}),
                }
            )
            if len(tool_step_history) > 32:
                tool_step_history = tool_step_history[-32:]
            if bool(result.ok):
                consecutive_fail_tool = None
                consecutive_fail_count = 0
            else:
                if consecutive_fail_tool == call.tool_name:
                    consecutive_fail_count += 1
                else:
                    consecutive_fail_tool = call.tool_name
                    consecutive_fail_count = 1
            if call.tool_name == "detect_lesion_candidates" and not bool(last_tool_result.get("ok", False)):
                lesion_failed_once = True

            # Circuit-break repeated cardiac segmentation failures to avoid prompt bloat loops.
            if (
                domain_cfg.name == "cardiac"
                and call.tool_name == "segment_cardiac_cine"
                and (not bool(result.ok))
                and consecutive_fail_count >= 3
            ):
                _write_step_debug(
                    "segment_cardiac_circuit_break",
                    step=step,
                    attempt=0,
                    messages_obj=messages,
                    raw_text=f"segment_cardiac_cine failed consecutively {consecutive_fail_count} times; switching to deterministic fallback.",
                    parsed_obj={"tool_name": call.tool_name, "consecutive_fail_count": consecutive_fail_count},
                    tool_result_obj=last_tool_result,
                )
                _run_deterministic_cardiac_mvp(
                    dispatcher,
                    state,
                    ctx,
                    case_id=case_id,
                    dicom_case_dir=dicom_case_dir,
                )
                final_dir = ctx.artifacts_dir / "final"
                final_dir.mkdir(parents=True, exist_ok=True)
                final_json = final_dir / "final_report.json"
                final_json.write_text(
                    json.dumps(
                        {
                            "final_report": {
                                "status": "completed_with_circuit_breaker",
                                "note": "segment_cardiac_cine failed repeatedly; switched to deterministic fallback path.",
                                "run_dir": str(ctx.run_dir),
                                "failing_tool": "segment_cardiac_cine",
                                "consecutive_fail_count": int(consecutive_fail_count),
                            }
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                state.set_summary({"final_report_path": str(final_json)})
                state.write_json(ctx.case_state_path)
                return ctx.run_dir
            # Record short-term memory + optional free-text reflection (for debugging + future long-term memory).
            if enable_tool_reflection:
                try:
                    _write_reflection(step=step, tool_name=call.tool_name, tool_result=last_tool_result)
                except Exception:
                    pass
            _write_step_debug(
                "tool_exec",
                step=step,
                attempt=0,
                messages_obj=messages,
                raw_text=text,
                parsed_obj={"action": "tool_call", "tool_name": call.tool_name, "stage": call.stage, "arguments": call.arguments},
                tool_result_obj=last_tool_result,
            )
            # Feed result back (as JSON) for next step
            messages.append({"role": "assistant", "content": _clip_text(text, max_chars=1200)})
            feedback: Dict[str, Any] = {"tool_result": _tool_feedback_for_llm(call.tool_name, result.to_dict())}
            if schema_autofix_applied and not failure_reflection_feedback:
                feedback["schema_autofix_applied"] = schema_autofix_applied
            if failure_reflection_feedback:
                feedback.update(failure_reflection_feedback)
            if autofix_mode == "coach":
                if autofix_changed:
                    feedback["autofix_applied"] = {
                        "tool_name": call.tool_name,
                        "changed_arguments": autofix_changed,
                        "note": "Agent auto-repaired arguments to keep the run usable. Learn these corrections for next steps.",
                    }
                if preconditions_applied:
                    feedback["preconditions_applied"] = preconditions_applied
                if rule_violations:
                    feedback["rule_violations"] = [
                        {
                            "rule_id": v.rule_id,
                            "level": v.level,
                            "tool_name": v.tool_name,
                            "message": v.message,
                            "details": v.details or {},
                        }
                        for v in rule_violations
                    ]
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        _compact_prompt_payload(feedback, max_chars=(2200 if failure_reflection_feedback else 1600)),
                        ensure_ascii=False,
                    ),
                }
            )

    # Optional safety net: after max_steps, enforce MVP pipeline and produce report (disabled by default).
    if enforce_mvp_pipeline:
        if domain_cfg.name == "cardiac":
            _run_deterministic_cardiac_mvp(
                dispatcher,
                state,
                ctx,
                case_id=case_id,
                dicom_case_dir=dicom_case_dir,
            )
        else:
            _run_deterministic_prostate_mvp(
                dispatcher,
                state,
                ctx,
                case_id=case_id,
                dicom_case_dir=dicom_case_dir,
                llm_mode=llm_mode,
                server_cfg=server_cfg,
            )
        return ctx.run_dir

    # Debug-first: write a summary artifact to explain why we stopped.
    final_dir = ctx.artifacts_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    diag = {
        "status": "error",
        "error": f"Agent loop reached max_steps={max_steps} without final output.",
        "hint": "Inspect agent_trace.jsonl and artifacts/llm_debug/* for step-by-step behavior.",
        "last_parsed": last_parsed_obj,
        "last_tool_result": last_tool_result,
        "failure_reflection_decisions": failure_reflection_decisions[-8:],
        "run_dir": str(ctx.run_dir),
        "case_state_path": str(ctx.case_state_path),
    }
    (final_dir / "final_report.json").write_text(json.dumps({"final_report": diag}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (final_dir / "final_report.txt").write_text(json.dumps({"final_report": diag}, ensure_ascii=False) + "\n", encoding="utf-8")
    state.set_summary({"final_report_path": str(final_dir / "final_report.json")})
    state.write_json(ctx.case_state_path)
    raise RuntimeError(f"{diag['error']} (run_dir={ctx.run_dir})")


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


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal tool-calling agent loop (stub LLM).")
    p.add_argument("--goal", default="Read this case and produce a report.", help="User goal")
    p.add_argument("--case-id", default="sub-057")
    p.add_argument(
        "--dicom-case-dir",
        default=str(Path(__file__).resolve().parents[1] / "demo" / "cases" / "sub-057"),
        help="DICOM case directory (series subfolders). Provided to the LLM as input context.",
    )
    p.add_argument("--runs-root", default=str(Path(__file__).resolve().parents[1] / "runs"))
    p.add_argument("--llm-mode", default="stub", choices=["stub", "server", "openai", "anthropic", "gemini"])
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=2, help="Retries with repair prompts when model output is invalid JSON.")
    p.add_argument("--plan-mode", default="step", choices=["step", "one_shot"], help="Hint to the LLM; one_shot is useful for slow models.")
    p.add_argument("--domain", default="prostate", choices=["prostate", "brain", "cardiac"], help="Domain-specific MRI logic configuration.")
    p.add_argument(
        "--enforce-mvp-pipeline",
        action="store_true",
        default=False,
        help="Optional safety net: enforce the fixed ingest->identify->register->segment->features->report pipeline. Disabled by default to keep behavior agent-like.",
    )

    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument(
        "--finalize-with-llm",
        action="store_true",
        default=False,
        help="After tool_calls plan execution, ask the selected LLM to produce the final report text.",
    )
    p.add_argument(
        "--autofix-mode",
        default="force",
        choices=["force", "coach", "off"],
        help="How the agent handles auto-repair/preconditions: force=apply silently, coach=apply but explain back to LLM, off=no auto-repair (more autonomy, less robustness).",
    )

    # vLLM server config (OpenAI-compatible)
    p.add_argument("--server-base-url", default="http://127.0.0.1:8000")
    p.add_argument(
        "--server-model",
        default=os.environ.get("MEDGEMMA_SERVER_MODEL", "Qwen/Qwen3-VL-30B-A3B-Thinking"),
    )
    p.add_argument("--server-timeout-s", type=int, default=600)
    # Cloud API config (OpenAI/Anthropic/Gemini)
    p.add_argument("--api-model", default="", help="Model name for llm-mode openai|anthropic|gemini.")
    p.add_argument("--api-base-url", default="", help="Optional base URL for OpenAI-compatible gateways/proxies.")
    args = p.parse_args()

    server_model = _sanitize_server_model(str(args.server_model), str(args.server_base_url))
    server_cfg = VLLMServerConfig(
        base_url=str(args.server_base_url),
        model=server_model,
        timeout_s=int(args.server_timeout_s),
        max_tokens=int(args.max_new_tokens),
    )

    domain_cfg = get_domain_config(str(args.domain))
    run_dir = run_agent_loop(
        goal=args.goal,
        case_id=args.case_id,
        dicom_case_dir=str(args.dicom_case_dir) if args.dicom_case_dir else None,
        runs_root=Path(args.runs_root).expanduser().resolve(),
        llm_mode=args.llm_mode,
        max_steps=int(args.max_steps),
        max_retries=int(args.max_retries),
        plan_mode=str(args.plan_mode),
        server_cfg=server_cfg,
        api_model=str(args.api_model or "") or None,
        api_base_url=str(args.api_base_url or "") or None,
        finalize_with_llm=bool(args.finalize_with_llm),
        enforce_mvp_pipeline=bool(args.enforce_mvp_pipeline),
        autofix_mode=str(args.autofix_mode),
        domain=domain_cfg,
    )
    print(f"[OK] Agent loop run output: {run_dir}")


if __name__ == "__main__":
    main()

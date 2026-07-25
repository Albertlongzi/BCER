"""
Tool: generate_report

Purpose:
- Generate a concise, *truthful* run report from `case_state.json` + artifacts.
- This is the "human readable" surface of the pipeline; it must reflect what actually ran.

Inputs:
- case_state_path: path to run `case_state.json`
- feature_table_path (optional): CSV path
- output_subdir (optional): where to write `report.md` / `report.json` under artifacts/

Outputs:
- report_txt_path: markdown report (stored as .md)
- report_json_path: structured report JSON
"""

from __future__ import annotations

import base64
import csv
import json
import re
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec
from core.domain_config import DomainConfig, get_domain_config
from runtime.path_utils import list_with_short_paths, to_short_path
from runtime.artifact_index import build_artifact_index
from llm.adapter_vllm_server import VLLMOpenAIChatAdapter, VLLMServerConfig
from llm.adapter_openai_api import OpenAIChatAdapter, OpenAIConfig
from llm.adapter_anthropic_api import AnthropicChatAdapter, AnthropicConfig
from llm.adapter_gemini_api import GeminiChatAdapter, GeminiConfig
from .vlm_evidence import _compute_lesion_geometry


REPORT_SPEC = ToolSpec(
    name="generate_report",
    description=(
        "Generate a run report from CaseState + artifacts. "
        "If llm_mode=server|openai|anthropic|gemini, also generate a VLM-grounded structured report from a standardized evidence bundle."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "case_state_path": {"type": "string"},
            "feature_table_path": {"type": "string"},
            "output_subdir": {"type": "string"},
            "llm_mode": {"type": "string"},
            "server_base_url": {"type": "string"},
            "server_model": {"type": "string"},
            "api_model": {"type": "string"},
            "api_base_url": {"type": "string"},
            "max_tokens": {"type": "integer"},
            "temperature": {"type": "number"},
            "emit_structured_report_text": {"type": "boolean"},
            "domain": {"type": "string"},
            "impression_mode": {
                "type": "string",
                "enum": ["template", "llm"],
                "description": (
                    "Impression generator. 'template' (default) is the deterministic, "
                    "measurement-only baseline. 'llm' asks a local model for a grounded "
                    "impression and falls back to the template on any transport failure, "
                    "empty completion, ungrounded number, or unsupported clinical entity."
                ),
            },
            "impression_llm_base_url": {"type": "string"},
            "impression_llm_model": {"type": "string"},
            "impression_llm_timeout_s": {"type": "integer"},
        },
        "required": ["case_state_path"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "report_txt_path": {"type": "string"},
            "report_json_path": {"type": "string"},
            "vlm_evidence_bundle_path": {"type": "string"},
            "llm_structured_report_path": {"type": "string"},
        },
        "required": ["report_txt_path", "report_json_path", "vlm_evidence_bundle_path"],
    },
    version="0.2.0",
    tags=["report", "mvp", "template", "vlm"],
)

MIN_CLINICALLY_MEANINGFUL_LESION_CC = 0.1


def _gather_pngs(run_dir: Path) -> List[str]:
    out: List[str] = []
    for p in sorted((run_dir / "artifacts").glob("**/*.png")):
        s = str(p)
        if "/features/overlays/" in s or s.endswith("_central.png"):
            out.append(s)
    return out[:16]


def _read_features_csv_preview(path: Optional[str], max_rows: int = 64) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= max_rows:
                break
            # keep as JSON-safe primitives; cast numeric fields when possible
            out: Dict[str, Any] = {}
            for k, v in (r or {}).items():
                if str(k).startswith("rad_"):
                    continue
                if v is None:
                    out[k] = None
                    continue
                vv = str(v)
                # attempt float cast for numeric columns
                try:
                    if k in {"roi", "sequence"}:
                        out[k] = vv
                    else:
                        out[k] = float(vv) if (("." in vv) or vv.isdigit()) else vv
                except Exception:
                    out[k] = vv
            rows.append(out)
    return rows


def _find_feature_md_path(feat_path: Optional[str]) -> Optional[str]:
    if not feat_path:
        return None
    p = Path(str(feat_path)).expanduser().resolve()
    if p.suffix.lower() == ".md" and p.exists():
        return str(p)
    if p.suffix.lower() == ".csv":
        md = p.with_suffix(".md")
        if md.exists():
            return str(md)
    # fallback: check sibling features.md
    cand = p.parent / "features.md"
    if cand.exists():
        return str(cand)
    return None


def _read_slice_summary_preview(path: Optional[str], per_sequence_cap: int = 12) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    per_seq = obj.get("per_sequence") if isinstance(obj, dict) else None
    if not isinstance(per_seq, dict):
        return {}
    out: Dict[str, Any] = {}
    for name, v in per_seq.items():
        if not isinstance(v, dict):
            continue
        per_slice = v.get("per_slice", [])
        if not isinstance(per_slice, list):
            per_slice = []
        slim = per_slice[: per_sequence_cap]
        out[str(name)] = {
            "n_slices_with_roi": v.get("n_slices_with_roi"),
            "per_slice_preview": slim,
        }
    return out


def _format_section_header(section: str) -> str:
    base = str(section or "").strip()
    if not base:
        return ""
    hint = ""
    if "(" in base and base.endswith(")"):
        try:
            base, hint = base[:-1].split("(", 1)
            base = base.strip()
            hint = hint.strip()
        except Exception:
            hint = ""
    header = base.upper()
    if not header.endswith(":"):
        header += ":"
    if hint:
        header = f"{header} {hint}".strip()
    return header


def _infer_indication(domain_cfg: DomainConfig) -> str:
    for section in domain_cfg.report_structure:
        if not isinstance(section, str):
            continue
        low = section.lower()
        if "clinical indication" in low and "(" in section and section.endswith(")"):
            try:
                return section.split("(", 1)[1][:-1].strip()
            except Exception:
                continue
    return f"{domain_cfg.name.capitalize()} MRI"


def _structured_system_prompt(domain_cfg: DomainConfig) -> str:
    if domain_cfg.name == "prostate":
        return (
            "You are writing a prostate mpMRI engineering-demo report. "
            "Ground every statement in the provided evidence bundle and images. "
            "Format qualitative findings using Markdown. Use **bold** for positive findings and bullet points for clarity. "
            "Even if some images are blurry, attempt to describe high/low signal regions; only use 'indeterminate' if you truly cannot discern signal. "
            "If lesion candidates are present, fill lesion_assessment with PI-RADS scores (t2_score, dwi_score, overall_pirads) and include tile_evidence (r<row> c<col> z<z>). "
            "If no lesion candidates, leave lesion_assessment empty and set overall_pirads only if you can justify it from the images. "
            "Treat lesion candidates smaller than 0.1 cc as clinically negligible and avoid escalating PI-RADS based only on those tiny foci. "
            "Return ONLY JSON (no extra text)."
        )
    if domain_cfg.name == "cardiac":
        return (
            "You are writing a cardiac cine MRI engineering-demo report. "
            "Ground every statement in the provided evidence bundle and images. "
            "Use evidence-first reasoning: provide Top-K candidate phenotypes, supporting feature evidence, and conflict evidence. "
            "Summarize segmentation-backed observations for LV cavity, RV cavity, and myocardium only when supported by evidence. "
            "If cardiac_classification evidence exists, report the predicted group and key physiological metrics (LV/RV EF, EDV) without inventing values. "
            "If cardiac_t1_feature_analysis exists, prioritize its QC gate and feature primitives over visual intuition. "
            "If evidence is insufficient, state 'indeterminate' with limitations. "
            "Return ONLY JSON (no extra text)."
        )
    return (
        "You are writing a brain tumor MRI engineering-demo report. "
        "Ground every statement in the provided evidence bundle and images. "
        "Use the quantitative CSV findings AND qualitative signal descriptions from the images. "
        "Analyze signal characteristics on provided tiles:\n"
        "- On T1c: describe enhancement (ring vs solid).\n"
        "- On FLAIR: describe peritumoral hyperintensity (edema).\n"
        "- On T1: describe necrosis (hypointense core).\n"
        "- On T2: describe overall tumor extent (heterogeneous hyperintensity).\n"
        "You MUST fill brain_signal_characterization with those four items; if uncertain, say 'indeterminate'.\n"
        "You MUST also fill image_qc with images_reviewed (list of image paths), "
        "roi_coverage_ok, registration_alignment_ok, gross_signal_patterns, "
        "tile_evidence (with tile labels like 'rX cY zZ'), and notes.\n"
        "Format qualitative findings using Markdown. Use **bold** for positive findings and bullet points for clarity.\n"
        "Even if some images are blurry, attempt to describe high/low signal regions; only use 'indeterminate' if you truly cannot discern signal.\n"
        "Return ONLY JSON (no extra text)."
    )


def _parse_dicom_header_tags(path: Optional[str]) -> Dict[str, str]:
    """
    Parse dicom_header_txt (DICOM_TAGS_TSV) into a keyword->value dict.
    Keeps first occurrence per keyword.
    """
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {}
    tags: Dict[str, str] = {}
    try:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "\t" not in line:
                continue
            parts = line.split("\t", 3)
            if len(parts) < 4:
                continue
            keyword = parts[2].strip()
            value = parts[3].strip()
            if keyword and keyword not in tags:
                tags[keyword] = value
    except Exception:
        return {}
    return tags


def _read_csv_rows(path: Optional[str], max_rows: int = 1000) -> List[Dict[str, str]]:
    if not path:
        return []
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return []
    rows: List[Dict[str, str]] = []
    try:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, r in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append({k: str(v) for k, v in (r or {}).items()})
    except Exception:
        return []
    return rows


def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _weighted_average(rows: List[Dict[str, Any]], key: str, *, weight_key: str = "n_voxels") -> Optional[float]:
    num = 0.0
    den = 0.0
    for r in rows:
        if not isinstance(r, dict):
            continue
        v = _safe_float(r.get(key))
        w = _safe_float(r.get(weight_key))
        if v is None:
            continue
        ww = w if (w is not None and w > 0) else 1.0
        num += v * ww
        den += ww
    if den <= 0:
        return None
    return num / den


def _cardiac_t1_role(row: Dict[str, Any]) -> str:
    tag = f"{row.get('roi_source','')}_{row.get('roi','')}".upper()
    if "MYO" in tag:
        return "MYO"
    if "LV" in tag:
        return "LV"
    if "RV" in tag:
        return "RV"
    return "OTHER"


def _cardiac_t1_phase(row: Dict[str, Any]) -> str:
    tag = f"{row.get('roi_source','')}_{row.get('roi','')}".upper()
    if "_ED" in tag:
        return "ED"
    if "_ES" in tag:
        return "ES"
    return "UNK"


def _build_cardiac_t1_feature_analysis(
    *,
    features_main: Dict[str, Any],
    cardiac_classification: Dict[str, Any],
) -> Dict[str, Any]:
    rows = features_main.get("features_preview_rows") if isinstance(features_main, dict) else []
    if not isinstance(rows, list):
        rows = []
    t1_rows = [
        r
        for r in rows
        if isinstance(r, dict) and ("t1" in str(r.get("sequence", "")).lower() or "molli" in str(r.get("sequence", "")).lower())
    ]
    if not t1_rows:
        return {
            "ok": False,
            "reason": "missing_t1_rows",
            "message": "No T1/T1_MOLLI feature rows were found; cannot perform T1 evidence interpretation.",
        }

    role_rows: Dict[str, List[Dict[str, Any]]] = {"MYO": [], "LV": [], "RV": []}
    phase_map: Dict[str, Dict[str, bool]] = {"ED": {"MYO": False, "LV": False, "RV": False}, "ES": {"MYO": False, "LV": False, "RV": False}}
    min_n_vox: Optional[float] = None
    for r in t1_rows:
        role = _cardiac_t1_role(r)
        phase = _cardiac_t1_phase(r)
        if role in role_rows:
            role_rows[role].append(r)
            if phase in phase_map:
                phase_map[phase][role] = True
        nv = _safe_float(r.get("n_voxels"))
        if nv is not None:
            min_n_vox = nv if min_n_vox is None else min(min_n_vox, nv)

    def _role_summary(role: str) -> Dict[str, Optional[float]]:
        rr = role_rows.get(role, [])
        return {
            "mean": _weighted_average(rr, "mean"),
            "std": _weighted_average(rr, "std"),
            "median": _weighted_average(rr, "median"),
            "p10": _weighted_average(rr, "p10"),
            "p90": _weighted_average(rr, "p90"),
            "volume_ml": _weighted_average(rr, "volume_ml"),
            "n_voxels": _weighted_average(rr, "n_voxels"),
        }

    myo = _role_summary("MYO")
    lv = _role_summary("LV")
    rv = _role_summary("RV")
    blood_rows = role_rows.get("LV", []) + role_rows.get("RV", [])
    blood_mean = _weighted_average(blood_rows, "mean")

    myo_mean = myo.get("mean")
    myo_std = myo.get("std")
    myo_median = myo.get("median")
    myo_p10 = myo.get("p10")
    myo_p90 = myo.get("p90")
    myo_iqr = None
    if myo_p90 is not None and myo_p10 is not None:
        myo_iqr = myo_p90 - myo_p10
    hetero_cv = None
    if myo_mean not in (None, 0.0) and myo_std is not None:
        hetero_cv = myo_std / myo_mean
    focality_index = None
    if myo_median is not None and myo_p10 is not None and myo_p90 is not None:
        den = (myo_median - myo_p10)
        if abs(den) > 1e-6:
            focality_index = (myo_p90 - myo_median) / den
    myo_to_blood_ratio = None
    if myo_mean is not None and blood_mean not in (None, 0.0):
        myo_to_blood_ratio = myo_mean / blood_mean

    mask_qc_fail: List[str] = []
    if not role_rows.get("MYO"):
        mask_qc_fail.append("missing_myo_mask_features")
    if not role_rows.get("LV"):
        mask_qc_fail.append("missing_lv_cavity_mask_features")
    if not role_rows.get("RV"):
        mask_qc_fail.append("missing_rv_cavity_mask_features")
    if min_n_vox is None or min_n_vox < 200:
        mask_qc_fail.append("roi_too_small_or_fragmented")

    ed_complete = all(bool(phase_map["ED"].get(k)) for k in ("MYO", "LV", "RV"))
    es_complete = all(bool(phase_map["ES"].get(k)) for k in ("MYO", "LV", "RV"))
    if not (ed_complete or es_complete):
        mask_qc_fail.append("missing_complete_phase_triplet")

    t1_qc_fail: List[str] = []
    blood_myo_gap = None
    if myo_mean is not None and blood_mean is not None and max(abs(myo_mean), abs(blood_mean)) > 1e-6:
        blood_myo_gap = abs(blood_mean - myo_mean) / max(abs(myo_mean), abs(blood_mean))
        if blood_myo_gap < 0.03:
            t1_qc_fail.append("blood_myo_t1_not_separable")
    else:
        t1_qc_fail.append("insufficient_t1_stats_for_blood_myo_qc")

    gate_fail_reasons = list(mask_qc_fail) + list(t1_qc_fail)
    gate_pass = len(gate_fail_reasons) == 0

    metrics_obj = cardiac_classification.get("metrics") if isinstance(cardiac_classification.get("metrics"), dict) else {}
    lv_ef = _safe_float(metrics_obj.get("lv_ef_percent"))
    rv_ef = _safe_float(metrics_obj.get("rv_ef_percent"))
    lv_edvi = _safe_float(metrics_obj.get("lv_edvi_ml_m2"))
    rv_edvi = _safe_float(metrics_obj.get("rv_edvi_ml_m2"))

    diffuse_index = _clamp01((((myo_to_blood_ratio if myo_to_blood_ratio is not None else 0.82) - 0.82) / 0.16))
    hetero_index = _clamp01((((hetero_cv if hetero_cv is not None else 0.09) - 0.06) / 0.10))
    focality_tail = _clamp01((((focality_index if focality_index is not None else 1.0) - 0.9) / 0.8))
    focal_index = _clamp01(0.55 * focality_tail + 0.45 * hetero_index)
    ef_drop_index = _clamp01(((55.0 - lv_ef) / 20.0)) if lv_ef is not None else 0.5
    ef_preserved_index = _clamp01(((lv_ef - 50.0) / 20.0)) if lv_ef is not None else 0.5
    lv_dilate = _clamp01(((lv_edvi - 70.0) / 30.0)) if lv_edvi is not None else 0.5
    rv_dilate = _clamp01(((rv_edvi - 80.0) / 35.0)) if rv_edvi is not None else 0.5
    bivent_dilation_index = 0.5 * (lv_dilate + rv_dilate)
    lv_not_dilated_index = 1.0 - lv_dilate

    scores = {
        "DCM-like phenotype": _clamp01(0.40 * diffuse_index + 0.35 * ef_drop_index + 0.25 * bivent_dilation_index),
        "MI/focal scar-like phenotype": _clamp01(0.50 * focal_index + 0.25 * ef_drop_index + 0.25 * (1.0 - diffuse_index)),
        "HCM-like phenotype": _clamp01(0.40 * ef_preserved_index + 0.35 * focal_index + 0.25 * lv_not_dilated_index),
        "Infiltrative/amyloid-like pattern (T1-only hint)": _clamp01(0.65 * diffuse_index + 0.20 * (1.0 - focal_index) + 0.15 * bivent_dilation_index),
        "NOR-like pattern": _clamp01(0.45 * ef_preserved_index + 0.35 * (1.0 - focal_index) + 0.20 * (1.0 - diffuse_index)),
    }

    primitive_lookup = {
        "myo_to_blood_t1_ratio": {"value": myo_to_blood_ratio, "roi": "MYO vs LV/RV blood pools"},
        "myo_heterogeneity_cv": {"value": hetero_cv, "roi": "MYO"},
        "myo_focality_index": {"value": focality_index, "roi": "MYO"},
        "lv_ef_percent": {"value": lv_ef, "roi": "global cine geometry"},
        "biventricular_dilation_index": {"value": bivent_dilation_index, "roi": "LV/RV cavity geometry"},
        "lv_not_dilated_index": {"value": lv_not_dilated_index, "roi": "LV cavity geometry"},
    }

    candidate_profiles = {
        "DCM-like phenotype": {
            "support": [
                ("myo_to_blood_t1_ratio", diffuse_index, "diffuse myocardial T1 shift pattern"),
                ("lv_ef_percent", ef_drop_index, "LV systolic impairment trend"),
                ("biventricular_dilation_index", bivent_dilation_index, "biventricular cavity enlargement trend"),
            ],
            "conflict": [
                ("lv_ef_percent", ef_preserved_index, "preserved EF conflicts with classical DCM-like pattern"),
                ("myo_focality_index", focal_index, "strong focality conflicts with diffuse DCM-like pattern"),
            ],
        },
        "MI/focal scar-like phenotype": {
            "support": [
                ("myo_focality_index", focal_index, "focal/right-tail heterogeneity in myocardial T1"),
                ("lv_ef_percent", ef_drop_index, "EF reduction can align with focal injury burden"),
            ],
            "conflict": [
                ("myo_to_blood_t1_ratio", diffuse_index, "strong diffuse shift conflicts with isolated focal scar pattern"),
            ],
        },
        "HCM-like phenotype": {
            "support": [
                ("lv_ef_percent", ef_preserved_index, "preserved/high EF pattern"),
                ("myo_focality_index", focal_index, "regional heterogeneity can fit focal hypertrophy/fibrosis"),
                ("lv_not_dilated_index", lv_not_dilated_index, "non-dilated LV geometry trend"),
            ],
            "conflict": [
                ("lv_ef_percent", ef_drop_index, "reduced EF argues against typical HCM-like preserved function"),
                ("biventricular_dilation_index", bivent_dilation_index, "marked dilation argues against classic HCM-like geometry"),
            ],
        },
        "Infiltrative/amyloid-like pattern (T1-only hint)": {
            "support": [
                ("myo_to_blood_t1_ratio", diffuse_index, "global diffuse T1 elevation trend"),
                ("myo_focality_index", 1.0 - focal_index, "low focality supports diffuse process"),
            ],
            "conflict": [
                ("myo_focality_index", focal_index, "strong focal pattern conflicts with diffuse infiltrative process"),
            ],
        },
        "NOR-like pattern": {
            "support": [
                ("lv_ef_percent", ef_preserved_index, "preserved systolic function"),
                ("myo_focality_index", 1.0 - focal_index, "low focal heterogeneity"),
                ("myo_to_blood_t1_ratio", 1.0 - diffuse_index, "no strong diffuse shift relative to blood pool"),
            ],
            "conflict": [
                ("myo_to_blood_t1_ratio", diffuse_index, "diffuse T1 shift argues against NOR-like pattern"),
                ("myo_focality_index", focal_index, "focal heterogeneity argues against NOR-like pattern"),
            ],
        },
    }

    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_k_candidates: List[Dict[str, Any]] = []
    for rank, (label, score) in enumerate(top, start=1):
        prof = candidate_profiles.get(label, {})
        support_rows: List[Dict[str, Any]] = []
        conflict_rows: List[Dict[str, Any]] = []
        for metric_name, strength, note in prof.get("support", []):
            if float(strength) < 0.55:
                continue
            prim = primitive_lookup.get(metric_name, {})
            support_rows.append(
                {
                    "metric": metric_name,
                    "roi": prim.get("roi"),
                    "value": prim.get("value"),
                    "strength": round(float(strength), 3),
                    "note": note,
                }
            )
        for metric_name, strength, note in prof.get("conflict", []):
            if float(strength) < 0.55:
                continue
            prim = primitive_lookup.get(metric_name, {})
            conflict_rows.append(
                {
                    "metric": metric_name,
                    "roi": prim.get("roi"),
                    "value": prim.get("value"),
                    "strength": round(float(strength), 3),
                    "note": note,
                }
            )
        top_k_candidates.append(
            {
                "rank": rank,
                "label": label,
                "score": round(float(score), 3),
                "support_evidence": support_rows,
                "conflict_evidence": conflict_rows,
            }
        )

    uncertainty: List[str] = []
    if len(top_k_candidates) >= 2:
        if abs(float(top_k_candidates[0]["score"]) - float(top_k_candidates[1]["score"])) < 0.12:
            uncertainty.append("Top-1 and Top-2 candidate scores are close; disease ranking is unstable.")
    if not (ed_complete and es_complete):
        uncertainty.append("Only one complete phase triplet (ED/ES) is available for T1 ROI features.")
    if lv_ef is None or rv_ef is None:
        uncertainty.append("EF metrics are incomplete; geometry-function coupling evidence is partial.")

    next_steps: List[str] = []
    if not gate_pass:
        next_steps.extend(
            [
                "Re-check mask quality/alignment before disease inference (UNRELIABLE_MASK gate failed).",
                "Verify T1 map reconstruction and blood-vs-myo separability (possible registration/mask mismatch).",
            ]
        )
    else:
        next_steps.extend(
            [
                "If infiltrative pattern remains in Top-K, add LGE and ECV for specificity.",
                "If focal scar/MI-like pattern remains in Top-K, add LGE with segmental wall-motion correlation.",
                "For robust subtype assignment, combine with site-specific normal ranges and clinical context.",
            ]
        )

    key_metrics = [
        {"name": "MYO_GLOBAL_MEAN_T1_MS", "value": myo_mean, "unit": "ms", "roi": "MYO"},
        {"name": "MYO_GLOBAL_STD_T1_MS", "value": myo_std, "unit": "ms", "roi": "MYO"},
        {"name": "MYO_GLOBAL_IQR_T1_MS", "value": myo_iqr, "unit": "ms", "roi": "MYO"},
        {"name": "MYO_HETEROGENEITY_CV", "value": hetero_cv, "unit": "", "roi": "MYO"},
        {"name": "MYO_FOCALITY_INDEX", "value": focality_index, "unit": "", "roi": "MYO"},
        {"name": "MYO_TO_BLOOD_T1_RATIO", "value": myo_to_blood_ratio, "unit": "", "roi": "MYO vs blood"},
    ]

    return {
        "ok": True,
        "sequence": "T1_MOLLI",
        "cine_geometry_group": cardiac_classification.get("predicted_group"),
        "cine_metrics": {
            "lv_ef_percent": lv_ef,
            "rv_ef_percent": rv_ef,
            "lv_edvi_ml_m2": lv_edvi,
            "rv_edvi_ml_m2": rv_edvi,
        },
        "qc": {
            "mask_qc": {
                "ed_triplet_complete": ed_complete,
                "es_triplet_complete": es_complete,
                "min_roi_n_voxels": min_n_vox,
                "fail_reasons": mask_qc_fail,
            },
            "t1_value_qc": {
                "blood_myo_relative_gap": blood_myo_gap,
                "fail_reasons": t1_qc_fail,
            },
            "gate_pass": gate_pass,
            "gate_fail_reasons": gate_fail_reasons,
        },
        "key_metrics": key_metrics,
        "top_k_candidates": top_k_candidates if gate_pass else [],
        "uncertainty": uncertainty,
        "next_steps": next_steps,
    }


def _encode_png_data_url(path: str, max_bytes: int = 2_500_000) -> Optional[str]:
    """
    Encode a local PNG as a data URL for OpenAI-compatible multimodal messages.
    Keep a strict size cap to avoid huge requests.
    """
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


def _file_size_bytes(path: str) -> Optional[int]:
    try:
        return int(Path(path).expanduser().resolve().stat().st_size)
    except Exception:
        return None


def _llm_structured_report_schema(domain_cfg: DomainConfig) -> Dict[str, Any]:
    # Minimal, robust schema that can be grounded in our evidence bundle.
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "run_id": {"type": "string"},
            "image_qc": {
                "type": "object",
                "properties": {
                    "images_reviewed": {"type": "array", "items": {"type": "string"}},
                    "roi_coverage_ok": {"type": "boolean"},
                    "registration_alignment_ok": {"type": "boolean"},
                    "gross_signal_patterns": {"type": "string", "minLength": 1},
                     "tile_evidence": {
                         "type": "array",
                         "minItems": 1,
                         "items": {
                             "type": "object",
                             "properties": {
                                 "sequence": {"type": "string", "minLength": 1},
                                 "tile": {"type": "string", "minLength": 1},
                                 "task": {"type": "string", "enum": ["roi_coverage", "registration_alignment"]},
                                 "observation": {"type": "string", "minLength": 1},
                             },
                             "required": ["sequence", "tile", "task", "observation"],
                             "additionalProperties": False,
                         },
                     },
                    "notes": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "images_reviewed",
                    "roi_coverage_ok",
                    "registration_alignment_ok",
                    "gross_signal_patterns",
                     "tile_evidence",
                    "notes",
                ],
                "additionalProperties": False,
            },
            "narrative": {
                "type": "object",
                "properties": {
                    "findings": {"type": "string", "minLength": 1},
                    "impression": {"type": "string", "minLength": 1},
                },
                "required": ["findings", "impression"],
                "additionalProperties": False,
            },
            "brain_signal_characterization": {
                "type": "object",
                "properties": {
                    "t1c_enhancement": {"type": "string", "minLength": 1},
                    "flair_edema": {"type": "string", "minLength": 1},
                    "t1_necrosis": {"type": "string", "minLength": 1},
                    "t2_extent": {"type": "string", "minLength": 1},
                },
                "required": ["t1c_enhancement", "flair_edema", "t1_necrosis", "t2_extent"],
                "additionalProperties": False,
            },
            "quantitative_summary": {"type": "object"},
            "limitations": {"type": "array", "items": {"type": "string"}},
            "evidence_used": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["case_id", "run_id", "image_qc", "narrative", "limitations", "evidence_used"],
        "additionalProperties": True,
    }
    if domain_cfg.name == "prostate":
        # Prostate: do not require brain-specific fields, add PI-RADS slots.
        schema["properties"].pop("brain_signal_characterization", None)
        schema["properties"]["lesion_assessment"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "lesion_id": {"type": "string"},
                    "location": {"type": "string"},
                    "zone": {"type": "string"},
                    "t2_score": {"type": "integer"},
                    "dwi_score": {"type": "integer"},
                    "overall_pirads": {"type": "integer"},
                    "rationale": {"type": "string"},
                    "tile_evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["overall_pirads", "rationale"],
                "additionalProperties": True,
            },
        }
        schema["properties"]["overall_pirads"] = {"type": "integer"}
    elif domain_cfg.name == "cardiac":
        # Cardiac: keep schema generic and remove prostate/brain-specific sections.
        schema["properties"].pop("brain_signal_characterization", None)
    return schema


def _llm_image_qc_schema(domain_cfg: DomainConfig) -> Dict[str, Any]:
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "image_qc": {
                "type": "object",
                "properties": {
                    "images_reviewed": {"type": "array", "items": {"type": "string"}},
                    "roi_coverage_ok": {"type": "boolean"},
                    "registration_alignment_ok": {"type": "boolean"},
                    "gross_signal_patterns": {"type": "string", "minLength": 1},
                    "tile_evidence": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "sequence": {"type": "string", "minLength": 1},
                                "tile": {"type": "string", "minLength": 1},
                                "task": {"type": "string", "enum": ["roi_coverage", "registration_alignment"]},
                                "observation": {"type": "string", "minLength": 1},
                            },
                            "required": ["sequence", "tile", "task", "observation"],
                            "additionalProperties": False,
                        },
                    },
                    "notes": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "images_reviewed",
                    "roi_coverage_ok",
                    "registration_alignment_ok",
                    "gross_signal_patterns",
                    "tile_evidence",
                    "notes",
                ],
                "additionalProperties": False,
            },
            "limitations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["image_qc"],
        "additionalProperties": True,
    }
    if domain_cfg.name == "brain":
        schema["properties"]["brain_signal_characterization"] = {
            "type": "object",
            "properties": {
                "t1c_enhancement": {"type": "string", "minLength": 1},
                "flair_edema": {"type": "string", "minLength": 1},
                "t1_necrosis": {"type": "string", "minLength": 1},
                "t2_extent": {"type": "string", "minLength": 1},
            },
            "required": ["t1c_enhancement", "flair_edema", "t1_necrosis", "t2_extent"],
            "additionalProperties": False,
        }
        schema["required"].append("brain_signal_characterization")
    return schema

def _format_report_txt(obj: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("MRI Agent Structured Report (VLM-grounded)")
    lines.append("")
    lines.append(f"Case ID: {obj.get('case_id')}")
    lines.append(f"Run ID: {obj.get('run_id')}")
    lines.append("")
    iq = obj.get("image_qc") or {}
    if isinstance(iq, dict) and iq:
        lines.append("Image QC (VLM):")
        lines.append(f"- images_reviewed: {iq.get('images_reviewed')}")
        lines.append(f"- roi_coverage_ok: {iq.get('roi_coverage_ok')}")
        lines.append(f"- registration_alignment_ok: {iq.get('registration_alignment_ok')}")
        lines.append("Gross signal patterns:")
        lines.append(str(iq.get("gross_signal_patterns", "")).strip())
        if iq.get("tile_evidence"):
            lines.append("Tile evidence (must cite montage tile labels r<row> c<col> z<z>):")
            for ev in iq.get("tile_evidence") or []:
                if isinstance(ev, dict):
                    lines.append(f"- [{ev.get('task')}] {ev.get('sequence')} @ {ev.get('tile')}: {str(ev.get('observation','')).strip()}")
        if iq.get("notes"):
            lines.append("Notes:")
            for s in iq.get("notes") or []:
                lines.append(f"- {str(s).strip()}")
        lines.append("")
    bsc = obj.get("brain_signal_characterization")
    if isinstance(bsc, dict) and bsc:
        lines.append("Brain signal characterization:")
        lines.append(f"- T1c enhancement: {bsc.get('t1c_enhancement')}")
        lines.append(f"- FLAIR edema: {bsc.get('flair_edema')}")
        lines.append(f"- T1 necrosis: {bsc.get('t1_necrosis')}")
        lines.append(f"- T2 extent: {bsc.get('t2_extent')}")
        lines.append("")
    n = obj.get("narrative") or {}
    lines.append("Findings:")
    lines.append(str(n.get("findings", "")).strip())
    lines.append("")
    lines.append("Impression:")
    lines.append(str(n.get("impression", "")).strip())
    lines.append("")
    if obj.get("limitations"):
        lines.append("Limitations:")
        for s in obj.get("limitations") or []:
            lines.append(f"- {str(s).strip()}")
        lines.append("")
    lines.append("Evidence used:")
    for s in obj.get("evidence_used") or []:
        lines.append(f"- {str(s).strip()}")
    lines.append("")
    return "\n".join(lines)


def _normalize_llm_structured(obj: Dict[str, Any], *, evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    The model may return various JSON shapes (e.g. {"report": {...}}).
    Normalize to our canonical shape expected by _format_report_txt().
    """
    if isinstance(obj.get("report"), dict):
        obj = obj["report"]  # unwrap common pattern

    case_id = obj.get("case_id") or evidence_bundle.get("case_id")
    run_id = obj.get("run_id") or evidence_bundle.get("run_id")

    # Try common narrative fields
    narrative_findings = ""
    narrative_impression = ""
    if isinstance(obj.get("narrative"), dict):
        narrative_findings = str(obj["narrative"].get("findings") or "")
        narrative_impression = str(obj["narrative"].get("impression") or "")
    else:
        # fallback: if model provides free-form strings
        narrative_findings = str(obj.get("findings") or obj.get("overall_findings") or "")
        narrative_impression = str(obj.get("impression") or obj.get("overall_impression") or "")

    limitations = obj.get("limitations")
    if not isinstance(limitations, list):
        limitations = []

    def _is_montage_item(it: Dict[str, Any]) -> bool:
        kind = str(it.get("kind") or "")
        if kind.startswith(("roi_crop_montage", "full_frame_montage")):
            return True
        p = str(it.get("path") or "").lower()
        return "montage" in p

    evidence_used = obj.get("evidence_used")
    if not isinstance(evidence_used, list) or not evidence_used:
        evidence_used = []
        feat = evidence_bundle.get("features") or {}
        for k in ("feature_table_path", "slice_summary_path"):
            if feat.get(k):
                evidence_used.append(str(feat.get(k)))

        overlays = feat.get("overlay_pngs") or []
        if isinstance(overlays, list):
            montages = [it for it in overlays if isinstance(it, dict) and it.get("path") and _is_montage_item(it)]
            best = [it for it in overlays if isinstance(it, dict) and it.get("path")]
            for it in (montages[:4] + best[:2]):
                p = str(it.get("path"))
                if p and p not in evidence_used:
                    evidence_used.append(p)
        # include lesion overlays if available
        feat_lesion = evidence_bundle.get("features_lesion") or {}
        overlays_lesion = feat_lesion.get("overlay_pngs") or []
        if isinstance(overlays_lesion, list):
            montages = [it for it in overlays_lesion if isinstance(it, dict) and it.get("path") and _is_montage_item(it)]
            best = [it for it in overlays_lesion if isinstance(it, dict) and it.get("path")]
            for it in (montages[:3] + best[:1]):
                p = str(it.get("path"))
                if p and p not in evidence_used:
                    evidence_used.append(p)
        evidence_used = evidence_used[:8]

    def _collect_reviewed_candidates() -> List[str]:
        feat = evidence_bundle.get("features") or {}
        overlays = feat.get("overlay_pngs") or []
        paths: List[str] = []
        if isinstance(overlays, list):
            montages = [
                it
                for it in overlays
                if isinstance(it, dict)
                and (it.get("abs_path") or it.get("path"))
                and _is_montage_item(it)
            ]
            best = [it for it in overlays if isinstance(it, dict) and (it.get("abs_path") or it.get("path"))]
            for it in (montages[:4] + best[:4]):
                p = str(it.get("abs_path") or it.get("path"))
                if p and p not in paths:
                    paths.append(p)
        return paths[:8]

    # --- image_qc normalization ---
    # The model may return image_qc in different shapes. Support:
    # A) our desired flat schema:
    #    {"images_reviewed":[...],"roi_coverage_ok":bool,"registration_alignment_ok":bool,"gross_signal_patterns":str,"notes":[...]}
    # B) nested per-modality schema (observed in practice):
    #    {"t2w":{"roi_coverage":"adequate","evidence":"..."},"adc":{...},"dwi_high_b":{...},...,"registration_alignment":{"value":"good","evidence":"..."}}
    image_qc_in = obj.get("image_qc") if isinstance(obj.get("image_qc"), dict) else {}
    images_reviewed: List[str] = []
    roi_coverage_ok: Optional[bool] = None
    registration_alignment_ok: Optional[bool] = None
    gross_signal_patterns = ""
    tile_evidence: List[Dict[str, str]] = []
    notes: List[str] = []
    reviewed_candidates = _collect_reviewed_candidates()

    # Case A: flat
    if any(k in image_qc_in for k in ("images_reviewed", "roi_coverage_ok", "registration_alignment_ok", "gross_signal_patterns", "notes")):
        if isinstance(image_qc_in.get("images_reviewed"), list):
            images_reviewed = [str(x) for x in image_qc_in.get("images_reviewed") if str(x).strip()]
        if "roi_coverage_ok" in image_qc_in:
            roi_coverage_ok = bool(image_qc_in.get("roi_coverage_ok"))
        if "registration_alignment_ok" in image_qc_in:
            registration_alignment_ok = bool(image_qc_in.get("registration_alignment_ok"))
        gross_signal_patterns = str(image_qc_in.get("gross_signal_patterns") or "").strip()
        if isinstance(image_qc_in.get("tile_evidence"), list):
            for ev in image_qc_in.get("tile_evidence") or []:
                if not isinstance(ev, dict):
                    continue
                seq = str(ev.get("sequence") or "").strip()
                tile = str(ev.get("tile") or "").strip()
                task = str(ev.get("task") or "").strip()
                obs = str(ev.get("observation") or "").strip()
                if seq and tile and task and obs:
                    tile_evidence.append({"sequence": seq, "tile": tile, "task": task, "observation": obs})
        if isinstance(image_qc_in.get("notes"), list):
            notes = [str(x) for x in image_qc_in.get("notes") if str(x).strip()]
    else:
        # Case B: nested
        per_mod = []
        for key in ("t2w", "adc", "dwi_high_b", "dwi_low_b", "dwi_other"):
            v = image_qc_in.get(key)
            if not isinstance(v, dict):
                continue
            cov = str(v.get("roi_coverage") or "").strip().lower()
            ev = str(v.get("evidence") or "").strip()
            if ev:
                notes.append(f"{key}: {ev}")
            if cov:
                per_mod.append((key, cov))
        # infer ROI coverage ok if we have any modalities
        if per_mod:
            bad = [k for (k, cov) in per_mod if cov not in ("adequate", "good", "ok", "sufficient")]
            roi_coverage_ok = (len(bad) == 0)

        ra = image_qc_in.get("registration_alignment")
        if isinstance(ra, dict):
            v = str(ra.get("value") or "").strip().lower()
            ev = str(ra.get("evidence") or "").strip()
            if ev:
                notes.append(f"registration_alignment: {ev}")
            if v:
                registration_alignment_ok = v in ("good", "ok", "adequate", "aligned", "well_aligned", "well-aligned")

        # images reviewed: prefer explicit file paths, otherwise fall back to evidence_used image paths
        for s in notes:
            # lightweight heuristic: pick out PNG-like tokens mentioned in evidence strings
            if ".png" in s:
                images_reviewed.append(s)
        # keep it tidy
        images_reviewed = images_reviewed[:8]

        # gross signal patterns: we don't want hallucinated lesion reads; summarize conservatively
        gross_signal_patterns = (
            "Reviewed ROI-cropped montages for T2w/ADC/DWI to sanity-check ROI coverage and gross signal patterns. "
            "No lesion-level qualitative claims made in this MVP."
        )

    if not images_reviewed and reviewed_candidates:
        images_reviewed = reviewed_candidates
        notes.append("Images were provided to the VLM, but image_qc/images_reviewed were omitted in the response; marking QC indeterminate.")

    # If tile_evidence missing, attempt a lightweight extraction from notes/evidence strings
    # (best-effort; schema will usually force non-empty tile_evidence via guided decoding).
    if not tile_evidence:
        for s in notes:
            if "r" in s and "c" in s and "z" in s:
                tile_evidence.append(
                    {
                        "sequence": "unknown",
                        "tile": s[:64],
                        "task": "roi_coverage",
                        "observation": s,
                    }
                )
                break
    if not tile_evidence:
        tile_evidence = [
            {
                "sequence": "unknown",
                "tile": "r0 c0 z?",
                "task": "roi_coverage",
                "observation": "No explicit tile citations provided; indeterminate tile-level QC.",
            }
        ]

    if roi_coverage_ok is None:
        roi_coverage_ok = False
    if registration_alignment_ok is None:
        registration_alignment_ok = False
    if not gross_signal_patterns.strip():
        gross_signal_patterns = "Indeterminate (no image_qc provided)."

    # Deterministic fallback if the model returned empty narrative strings.
    if not narrative_findings.strip() or not narrative_impression.strip():
        card_cls = evidence_bundle.get("cardiac_classification") if isinstance(evidence_bundle.get("cardiac_classification"), dict) else {}
        card_mode = bool(card_cls) or bool((evidence_bundle.get("mapping") or {}).get("CINE"))
        if card_mode:
            pred = card_cls.get("predicted_group") if isinstance(card_cls, dict) else None
            met = card_cls.get("metrics") if isinstance(card_cls.get("metrics"), dict) else {}
            lv_ef = met.get("lv_ef_percent") if isinstance(met, dict) else None
            rv_ef = met.get("rv_ef_percent") if isinstance(met, dict) else None
            lv_edv = met.get("lv_edv_ml") if isinstance(met, dict) else None
            rv_edv = met.get("rv_edv_ml") if isinstance(met, dict) else None
            features_main = evidence_bundle.get("features_main") if isinstance(evidence_bundle.get("features_main"), dict) else {}
            rad_ok = bool(features_main.get("radiomics_summary_path")) if isinstance(features_main, dict) else False
            rows = features_main.get("features_preview_rows") if isinstance(features_main, dict) else []
            texture_feats = 0
            if isinstance(rows, list):
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    texture_feats += sum(
                        1
                        for k in r.keys()
                        if str(k).startswith("rad_") and any(t in str(k).lower() for t in ("glcm", "glrlm", "glszm", "gldm", "ngtdm"))
                    )

            if not narrative_findings.strip():
                parts = []
                if pred:
                    parts.append(f"Rule-based cardiac group is {pred}.")
                if lv_ef is not None and rv_ef is not None:
                    try:
                        parts.append(f"LV EF {float(lv_ef):.1f}% and RV EF {float(rv_ef):.1f}%.")
                    except Exception:
                        parts.append(f"LV EF {lv_ef} and RV EF {rv_ef}.")
                if lv_edv is not None and rv_edv is not None:
                    try:
                        parts.append(f"LV EDV {float(lv_edv):.1f} mL and RV EDV {float(rv_edv):.1f} mL.")
                    except Exception:
                        parts.append(f"LV EDV {lv_edv} and RV EDV {rv_edv}.")
                if rad_ok:
                    parts.append("ROI texture radiomics were extracted for cardiac masks (RV/MYO/LV, ED/ES).")
                if texture_feats > 0:
                    parts.append(f"Texture-derived feature entries available: {int(texture_feats)} (preview rows).")
                if not parts:
                    parts.append("Cardiac cine evidence available but narrative details are limited.")
                narrative_findings = " ".join(parts).strip()

            if not narrative_impression.strip():
                if pred:
                    narrative_impression = f"Findings are most compatible with {pred} in this engineering pipeline."
                else:
                    narrative_impression = "Cardiac class is indeterminate from current evidence."
        else:
            feat = evidence_bundle.get("features") or {}
            rows = feat.get("features_preview_rows") or []
            # extract a few canonical numbers (if available)
            def _get_row(seq_contains: str, roi: str, field: str) -> Optional[float]:
                if not isinstance(rows, list):
                    return None
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    if str(r.get("roi")) != roi:
                        continue
                    if seq_contains.lower() not in str(r.get("sequence", "")).lower():
                        continue
                    v = r.get(field)
                    try:
                        return float(v)
                    except Exception:
                        continue
                return None

            vol_ml = _get_row("adc", "whole_gland", "volume_ml")
            adc_pz = _get_row("adc", "peripheral_zone", "mean")
            t2_pz = _get_row("t2w", "peripheral_zone", "mean")
            dwi_hi_pz = _get_row("high_b", "peripheral_zone", "mean") or _get_row("dwi", "peripheral_zone", "mean")

            if not narrative_findings.strip():
                parts = []
                if vol_ml is not None:
                    parts.append(f"Whole-gland prostate volume (from zonal mask) ~{vol_ml:.1f} mL.")
                if t2_pz is not None:
                    parts.append(f"T2w PZ mean signal ~{t2_pz:.1f} (arbitrary units).")
                if adc_pz is not None:
                    parts.append(f"ADC PZ mean ~{adc_pz:.1f} (arbitrary units).")
                if dwi_hi_pz is not None:
                    parts.append(f"DWI high-b PZ mean ~{dwi_hi_pz:.1f} (arbitrary units).")
                parts.append("This MVP provides gland/zonal summaries only; no focal lesion localization was performed.")
                narrative_findings = " ".join(parts).strip()

            if not narrative_impression.strip():
                narrative_impression = (
                    "Indeterminate for focal lesion assessment in this MVP. "
                    "Consider radiologist review of T2/DWI/ADC with lesion-focused workflow if clinically indicated."
                )

    image_qc = obj.get("image_qc") if isinstance(obj.get("image_qc"), dict) else {}
    out = {
        "case_id": str(case_id) if case_id is not None else None,
        "run_id": str(run_id) if run_id is not None else None,
        "image_qc": {
            "images_reviewed": images_reviewed,
            "roi_coverage_ok": bool(roi_coverage_ok),
            "registration_alignment_ok": bool(registration_alignment_ok),
            "gross_signal_patterns": gross_signal_patterns,
             "tile_evidence": tile_evidence[:8],
            "notes": notes[:16],
        },
        "narrative": {"findings": narrative_findings.strip(), "impression": narrative_impression.strip()},
        "limitations": limitations,
        "evidence_used": evidence_used,
    }
    # keep any quantitative sections the model returned
    if isinstance(obj.get("quantitative_summary"), dict):
        out["quantitative_summary"] = obj.get("quantitative_summary")
    if isinstance(obj.get("brain_signal_characterization"), dict):
        out["brain_signal_characterization"] = obj.get("brain_signal_characterization")
    lesion_assessment_obj = obj.get("lesion_assessment")
    if isinstance(lesion_assessment_obj, dict):
        lesion_items: List[Any] = [lesion_assessment_obj]
    elif isinstance(lesion_assessment_obj, list):
        lesion_items = lesion_assessment_obj
    else:
        lesion_items = []
    if lesion_items:
        cleaned_la: List[Dict[str, Any]] = []
        for it in lesion_items:
            if not isinstance(it, dict):
                continue
            nested = it.get("lesion_candidates")
            if isinstance(nested, list) and nested:
                for cand in nested:
                    if not isinstance(cand, dict):
                        continue
                    row = dict(cand)
                    if not row.get("lesion_id"):
                        row["lesion_id"] = f"Lesion #{len(cleaned_la) + 1}"
                    # Inherit parent summary score when candidate-level score is absent.
                    if row.get("overall_pirads") is None and it.get("overall_pirads") is not None:
                        row["overall_pirads"] = it.get("overall_pirads")
                    cleaned_la.append(row)
                continue
            row = dict(it)
            if not row.get("lesion_id"):
                row["lesion_id"] = f"Lesion #{len(cleaned_la) + 1}"
            cleaned_la.append(row)
        out["lesion_assessment"] = cleaned_la
    if obj.get("overall_pirads") is not None:
        out["overall_pirads"] = obj.get("overall_pirads")
    return out


def _apply_structured_report_rules(
    norm: Dict[str, Any], *, evidence_bundle: Dict[str, Any], selected_images: List[Dict[str, Any]], run_dir: Path
) -> Dict[str, Any]:
    """
    Lightweight guardrails to reduce obviously-wrong structured outputs.
    In particular, avoid "no visible lesion ROI" claims when lesion ROI crops were provided.
    """
    out = dict(norm or {})
    iq = out.get("image_qc") if isinstance(out.get("image_qc"), dict) else {}
    limits = out.get("limitations") if isinstance(out.get("limitations"), list) else []
    evidence_used = out.get("evidence_used") if isinstance(out.get("evidence_used"), list) else []

    lesion_features_ok = bool((evidence_bundle.get("features_lesion") or {}).get("ok"))
    lesion_geometry = evidence_bundle.get("lesion_geometry") if isinstance(evidence_bundle.get("lesion_geometry"), dict) else {}
    lesion_geo_max_cc: Optional[float] = None
    try:
        lesions = lesion_geometry.get("lesions") if isinstance(lesion_geometry.get("lesions"), list) else []
        vols = []
        for it in lesions:
            if not isinstance(it, dict):
                continue
            try:
                vols.append(float(it.get("volume_cc") or 0.0))
            except Exception:
                continue
        if vols:
            lesion_geo_max_cc = max(vols)
    except Exception:
        lesion_geo_max_cc = None
    mapping = evidence_bundle.get("mapping") if isinstance(evidence_bundle.get("mapping"), dict) else {}
    brain_mode = bool(mapping.get("FLAIR") or mapping.get("T1c") or mapping.get("T1"))
    lesion_images_present = any(
        isinstance(it, dict)
        and str(it.get("path") or "").strip()
        and "lesion" in str(it.get("path") or "").lower()
        for it in selected_images
    )
    lesion_evidence_present = lesion_features_ok and lesion_images_present
    tiny_lesion_only = lesion_geo_max_cc is not None and lesion_geo_max_cc < float(MIN_CLINICALLY_MEANINGFUL_LESION_CC)

    # Ensure evidence_used includes all images we actually sent to the VLM.
    if isinstance(selected_images, list):
        for it in selected_images:
            if not isinstance(it, dict):
                continue
            p = str(it.get("path") or "").strip()
            if not p:
                continue
            sp = to_short_path(p, run_dir=run_dir) or p
            if sp not in evidence_used and p not in evidence_used:
                evidence_used.append(sp)
    if evidence_used:
        out["evidence_used"] = evidence_used[:12]

    # If the model didn't review any images, mark QC as unavailable and avoid claims.
    reviewed_any = isinstance(iq.get("images_reviewed"), list) and len(iq.get("images_reviewed") or []) > 0
    if not reviewed_any:
        iq["images_reviewed"] = []
        iq["roi_coverage_ok"] = False
        iq["registration_alignment_ok"] = False
        iq["gross_signal_patterns"] = "VLM image review not available (no images reviewed)."
        limits = limits + ["VLM QC unavailable because no images were reviewed by the model."]

    if brain_mode:
        bsc = out.get("brain_signal_characterization") if isinstance(out.get("brain_signal_characterization"), dict) else {}
        required = {
            "t1c_enhancement": "indeterminate",
            "flair_edema": "indeterminate",
            "t1_necrosis": "indeterminate",
            "t2_extent": "indeterminate",
        }
        for k, v in required.items():
            val = str(bsc.get(k) or "").strip()
            if not val:
                bsc[k] = v
        out["brain_signal_characterization"] = bsc
        if any(str(bsc.get(k)) == "indeterminate" for k in required.keys()):
            limits = limits + ["Brain signal characterization incomplete; some sequence-specific findings indeterminate."]

    if lesion_evidence_present:
        # Ensure images_reviewed is populated for auditability.
        reviewed = iq.get("images_reviewed") if isinstance(iq.get("images_reviewed"), list) else []
        if not reviewed:
            reviewed = []
            for it in selected_images:
                if not isinstance(it, dict):
                    continue
                p = str(it.get("path") or "").strip()
                if not p:
                    continue
                reviewed.append(to_short_path(p, run_dir=run_dir) or p)
            iq["images_reviewed"] = reviewed[:8]
        # ROI-cropped lesion images imply some ROI coverage is present.
        if not iq.get("roi_coverage_ok"):
            iq["roi_coverage_ok"] = True
        notes = iq.get("notes") if isinstance(iq.get("notes"), list) else []
        if "Lesion ROI crops were provided to the VLM." not in notes:
            notes.append("Lesion ROI crops were provided to the VLM.")
        iq["notes"] = notes[:16]

        bad_markers = (
            "no visible lesion roi",
            "lesion roi not visible",
            "roi not visible",
            "not visible in the provided image",
        )
        cleaned_limits: List[str] = []
        removed_any = False
        for s in limits:
            ss = str(s or "").strip()
            low = ss.lower()
            if any(m in low for m in bad_markers):
                removed_any = True
                continue
            cleaned_limits.append(ss)
        if removed_any:
            cleaned_limits.append(
                "Lesion ROI crops were provided, but qualitative lesion characterization remains limited in this MVP."
            )
        limits = cleaned_limits[:16]

    if tiny_lesion_only:
        # Do not hard-force PI-RADS 1. Tiny-only candidates are evidence-limited.
        # Keep lesion_assessment empty and let downstream logic decide whether this
        # should be low suspicion vs indeterminate.
        out["lesion_assessment"] = []
        out.pop("overall_pirads", None)
        narrative = out.get("narrative") if isinstance(out.get("narrative"), dict) else {}
        findings = str(narrative.get("findings") or "").strip()
        impression = str(narrative.get("impression") or "").strip()
        if not findings or "lesion" in findings.lower():
            narrative["findings"] = (
                "Only tiny focal candidates were detected; clinically meaningful lesion characterization is limited."
            )
        if not impression or "pi-rads" in impression.lower():
            narrative["impression"] = "PI-RADS indeterminate from tiny-only candidates; correlate with full image review."
        out["narrative"] = narrative
        msg = (
            f"Tiny lesion candidates (<{MIN_CLINICALLY_MEANINGFUL_LESION_CC:.1f} cc) were detected and treated as below reporting floor."
        )
        if msg not in limits:
            limits.append(msg)

    # Avoid contradiction: if lesion evidence exists, don't claim "no focal lesion".
    try:
        lesion_geo_ok = bool((evidence_bundle.get("lesion_geometry") or {}).get("ok"))
        if lesion_features_ok or lesion_geo_ok:
            narrative = out.get("narrative") if isinstance(out.get("narrative"), dict) else {}
            findings = str(narrative.get("findings") or "")
            impression = str(narrative.get("impression") or "")
            bad = ("no focal lesion", "no lesion localization", "indeterminate for focal lesion")
            if any(b in findings.lower() for b in bad):
                narrative["findings"] = (
                    "Lesion candidates were detected by the pipeline; VLM qualitative characterization is limited."
                )
            if any(b in impression.lower() for b in bad):
                narrative["impression"] = (
                    "Lesion candidates detected by the pipeline (research use only). "
                    "VLM qualitative assessment remains limited."
                )
            out["narrative"] = narrative
    except Exception:
        pass

    out["image_qc"] = iq
    out["limitations"] = limits
    # Ensure evidence_used reflects images actually provided to the VLM.
    evidence_used = out.get("evidence_used") if isinstance(out.get("evidence_used"), list) else []
    for it in selected_images:
        if not isinstance(it, dict):
            continue
        p = str(it.get("path") or "").strip()
        if not p:
            continue
        short = to_short_path(p, run_dir=run_dir) or p
        if short not in evidence_used:
            evidence_used.append(short)
    if evidence_used:
        out["evidence_used"] = evidence_used[:12]
    return out


# ---------------------------------------------------------------------------
# Reconstruction provenance
# ---------------------------------------------------------------------------


def _build_reconstruction_provenance(recon_ok: Optional[bool], recon: Any) -> Dict[str, Any]:
    """
    Distil `reconstruct_grappa`'s recorded stage output into the facts a report
    must carry about how the images were produced.

    Every field here is copied from the tool's own output block; nothing is
    inferred and nothing is recomputed.

    The one derived flag is ``retrospective``: `reconstruct_grappa` only emits
    the ``undersample`` block when *it* decimated fully-sampled input k-space
    (see `_retrospectively_undersample`), so ``undersample.applied is True``
    means, by construction, retrospective undersampling -- not a prospectively
    accelerated acquisition.
    """
    if not isinstance(recon, dict) or not recon:
        return {"ok": False, "reason": "no_reconstruct_grappa_record"}
    if recon_ok is not True:
        return {"ok": False, "reason": "reconstruct_grappa_did_not_succeed"}

    us_raw = recon.get("undersample") if isinstance(recon.get("undersample"), dict) else {}
    rc_raw = recon.get("readout_crop") if isinstance(recon.get("readout_crop"), dict) else {}

    undersample: Dict[str, Any] = {"applied": bool(us_raw.get("applied"))}
    if undersample["applied"]:
        undersample.update(
            {
                # Structural, not textual: this block exists only when the tool
                # itself decimated fully sampled k-space.
                "retrospective": True,
                "pattern": us_raw.get("pattern"),
                "factor": us_raw.get("factor"),
                "acs_lines": us_raw.get("acs_lines"),
                "ky_lines_total": us_raw.get("ky_lines_total"),
                "ky_lines_kept": us_raw.get("ky_lines_kept"),
                "sampled_fraction": us_raw.get("sampled_fraction"),
                "net_acceleration": us_raw.get("net_acceleration"),
                "tool_note": us_raw.get("note"),
            }
        )

    readout_crop: Dict[str, Any] = {"applied": bool(rc_raw.get("applied"))}
    if readout_crop["applied"]:
        readout_crop.update(
            {
                "mode": rc_raw.get("mode"),
                "axis": rc_raw.get("axis"),
                "samples_before": rc_raw.get("samples_before"),
                "samples_after": rc_raw.get("samples_after"),
                "fov_before_mm": rc_raw.get("fov_before_mm"),
                "fov_after_mm": rc_raw.get("fov_after_mm"),
                "removed_left": rc_raw.get("removed_left"),
                "removed_right": rc_raw.get("removed_right"),
                "origin_shift_mm": rc_raw.get("origin_shift_mm"),
                "outer_half_energy_fraction": rc_raw.get("outer_half_energy_fraction"),
            }
        )

    out_shape = recon.get("output_shape") if isinstance(recon.get("output_shape"), list) else None
    ns_order = recon.get("nonspatial_order") if isinstance(recon.get("nonspatial_order"), list) else None
    n_slices: Optional[int] = None
    n_phases: Optional[int] = None
    # `reconstruct_grappa` documents that nonspatial_order=[1, 0] emits a volume
    # whose 3rd axis is slices and whose 4th axis is cardiac phases.  Only claim
    # the split when that documented layout actually applies.
    if out_shape and len(out_shape) == 4 and ns_order == [1, 0]:
        try:
            n_slices = int(out_shape[2])
            n_phases = int(out_shape[3])
        except Exception:
            n_slices = None
            n_phases = None

    h5_path = recon.get("h5_path")
    return {
        "ok": True,
        "mode": recon.get("mode"),
        "source_h5_path": h5_path,
        "source_h5_name": (Path(str(h5_path)).name if h5_path else None),
        "kspace_key": recon.get("kspace_key") or recon.get("source_key"),
        "kspace_shape": recon.get("kspace_shape"),
        "n_coils": recon.get("n_coils"),
        "acs_lines_used": recon.get("acs_lines_used"),
        "kernel_size": recon.get("kernel_size"),
        "frames_total": recon.get("frames_total"),
        "grappa_applied_frames": recon.get("grappa_applied_frames"),
        "grappa_skipped_frames": recon.get("grappa_skipped_frames"),
        "grappa_failed_frames": recon.get("grappa_failed_frames"),
        "undersample": undersample,
        "readout_crop": readout_crop,
        "output_shape": out_shape,
        "n_slices": n_slices,
        "n_phases": n_phases,
        "pixel_spacing": recon.get("pixel_spacing"),
        "pixel_spacing_source": recon.get("pixel_spacing_source"),
        "reconstructed_nifti": recon.get("reconstructed_nifti"),
        "zerofilled_nifti": recon.get("zerofilled_nifti"),
    }


def _fmt_num(val: Any, digits: int = 1) -> Optional[str]:
    """Format a measured number, or return None so the caller can omit the line."""
    f = _safe_float(val)
    if f is None:
        return None
    return f"{f:.{digits}f}"


def _reconstruction_lines(prov: Dict[str, Any]) -> List[str]:
    """Human-readable acquisition/reconstruction provenance. Empty if unknown."""
    if not isinstance(prov, dict) or not prov.get("ok"):
        return []
    lines: List[str] = []

    src_bits: List[str] = []
    if prov.get("source_h5_name"):
        src_bits.append(f"`{prov['source_h5_name']}`")
    if prov.get("kspace_key"):
        src_bits.append(f"dataset key `{prov['kspace_key']}`")
    shape = prov.get("kspace_shape")
    if isinstance(shape, list) and shape:
        src_bits.append("k-space shape " + " x ".join(str(int(x)) for x in shape))
    if prov.get("n_coils") is not None:
        src_bits.append(f"{int(prov['n_coils'])} receive coils")
    if src_bits:
        lines.append("- Source: raw k-space, " + ", ".join(src_bits) + ".")

    us = prov.get("undersample") if isinstance(prov.get("undersample"), dict) else {}
    if us.get("applied"):
        head_bits: List[str] = []
        if us.get("factor") is not None:
            head_bits.append(f"R={int(us['factor'])}")
        if us.get("pattern"):
            head_bits.append(str(us["pattern"]))
        if us.get("acs_lines") is not None:
            head_bits.append(f"{int(us['acs_lines'])} ACS lines")
        head = ", ".join(head_bits) if head_bits else "undersampled"
        lines.append(
            "- RETROSPECTIVE UNDERSAMPLING: the input k-space was fully sampled and was "
            f"decimated by this pipeline ({head}) before reconstruction. "
            "This is NOT prospectively accelerated acquisition."
        )
        detail: List[str] = []
        if us.get("ky_lines_kept") is not None and us.get("ky_lines_total") is not None:
            detail.append(f"{int(us['ky_lines_kept'])} of {int(us['ky_lines_total'])} ky lines retained")
        sf = _fmt_num(us.get("sampled_fraction"), 3)
        if sf:
            detail.append(f"sampled fraction {sf}")
        na = _fmt_num(us.get("net_acceleration"), 2)
        if na:
            detail.append(f"net acceleration {na}x")
        if detail:
            lines.append("- Undersampling detail: " + "; ".join(detail) + ".")
    elif us:
        lines.append("- Undersampling: none applied; k-space reconstructed as stored.")

    recon_bits: List[str] = []
    mode = str(prov.get("mode") or "").strip()
    if mode:
        recon_bits.append(mode.upper() if mode.lower() == "grappa" else mode)
    ks = prov.get("kernel_size")
    if isinstance(ks, list) and len(ks) == 2:
        recon_bits.append(f"kernel {int(ks[0])}x{int(ks[1])}")
    if prov.get("acs_lines_used") is not None:
        recon_bits.append(f"{int(prov['acs_lines_used'])} ACS lines")
    if recon_bits:
        frag = "- Reconstruction: " + ", ".join(recon_bits)
        if prov.get("grappa_applied_frames") is not None and prov.get("frames_total") is not None:
            frag += (
                f"; applied to {int(prov['grappa_applied_frames'])} of "
                f"{int(prov['frames_total'])} frames"
            )
            skipped = prov.get("grappa_skipped_frames")
            failed = prov.get("grappa_failed_frames")
            if skipped is not None or failed is not None:
                frag += f" ({int(skipped or 0)} skipped, {int(failed or 0)} failed)"
        lines.append(frag + ".")

    rc = prov.get("readout_crop") if isinstance(prov.get("readout_crop"), dict) else {}
    if rc.get("applied"):
        bits: List[str] = []
        if rc.get("samples_before") is not None and rc.get("samples_after") is not None:
            bits.append(f"{int(rc['samples_before'])} -> {int(rc['samples_after'])} readout samples")
        fb = _fmt_num(rc.get("fov_before_mm"))
        fa = _fmt_num(rc.get("fov_after_mm"))
        if fb and fa:
            bits.append(f"readout FOV {fb} -> {fa} mm")
        if rc.get("removed_left") is not None and rc.get("removed_right") is not None:
            bits.append(f"{int(rc['removed_left'])} / {int(rc['removed_right'])} samples removed left / right")
        os_mm = _fmt_num(rc.get("origin_shift_mm"))
        if os_mm:
            bits.append(f"origin shifted {os_mm} mm")
        if bits:
            lines.append("- Readout crop: " + "; ".join(bits) + ".")
        oh = _fmt_num(rc.get("outer_half_energy_fraction"), 3)
        if oh:
            lines.append(f"- Fraction of readout energy in the discarded outer halves: {oh}.")
    elif rc:
        lines.append("- Readout crop: none applied.")

    geom: List[str] = []
    out_shape = prov.get("output_shape")
    if isinstance(out_shape, list) and out_shape:
        geom.append("reconstructed volume " + " x ".join(str(int(x)) for x in out_shape))
    ps = prov.get("pixel_spacing")
    if isinstance(ps, list) and len(ps) == 3:
        try:
            spacing = f"voxel {float(ps[0]):.4f} x {float(ps[1]):.4f} x {float(ps[2]):.4f} mm"
            # Whether the spacing was read from the file or asserted by the
            # caller changes how much weight the volumes can carry.
            if prov.get("pixel_spacing_source"):
                spacing += f" (spacing source: {prov['pixel_spacing_source']})"
            geom.append(spacing)
        except Exception:
            pass
    if geom:
        lines.append("- Geometry: " + "; ".join(geom) + ".")
    return lines


# ---------------------------------------------------------------------------
# Cardiac measurements -> FINDINGS
# ---------------------------------------------------------------------------


def _cardiac_measurements(
    cls_info: Dict[str, Any],
    recon_prov: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Collect the cardiac quantities the pipeline actually measured.

    Returns a list of {label, value, unit, digits, source} records.  Values that
    the tools did not produce are simply absent -- nothing is estimated, nothing
    is recomputed from masks here, and nothing is padded with a placeholder.
    """
    metrics = cls_info.get("metrics") if isinstance(cls_info.get("metrics"), dict) else {}
    out: List[Dict[str, Any]] = []

    def add(label: str, key: str, unit: str, digits: int = 1, source: str = "classify_cardiac_cine_disease") -> None:
        val = _safe_float(metrics.get(key))
        if val is None:
            return
        out.append({"label": label, "value": val, "unit": unit, "digits": digits, "source": source})

    add("LV end-diastolic volume", "lv_edv_ml", "mL")
    add("LV end-systolic volume", "lv_esv_ml", "mL")
    add("LV ejection fraction", "lv_ef_percent", "%")
    add("RV end-diastolic volume", "rv_edv_ml", "mL")
    add("RV end-systolic volume", "rv_esv_ml", "mL")
    add("RV ejection fraction", "rv_ef_percent", "%")
    add("Myocardial volume at end-diastole", "myo_ed_volume_ml", "mL")
    add("LV myocardial mass", "lv_mass_g", "g")
    add("Maximum myocardial wall thickness", "max_myo_thickness_mm", "mm")
    add("LV EDV indexed to BSA", "lv_edvi_ml_m2", "mL/m^2")
    add("RV EDV indexed to BSA", "rv_edvi_ml_m2", "mL/m^2")
    add("LV mass indexed to BSA", "lv_mass_index_g_m2", "g/m^2")
    add("Body surface area", "bsa_m2", "m^2", digits=2)

    if isinstance(recon_prov, dict) and recon_prov.get("ok"):
        if recon_prov.get("n_slices") is not None:
            out.append(
                {
                    "label": "Short-axis slices reconstructed",
                    "value": float(recon_prov["n_slices"]),
                    "unit": "",
                    "digits": 0,
                    "source": "reconstruct_grappa",
                }
            )
        if recon_prov.get("n_phases") is not None:
            out.append(
                {
                    "label": "Cardiac phases reconstructed",
                    "value": float(recon_prov["n_phases"]),
                    "unit": "",
                    "digits": 0,
                    "source": "reconstruct_grappa",
                }
            )
    return out


def _cardiac_findings_lines(
    *,
    cls_info: Dict[str, Any],
    seg_info: Dict[str, Any],
    recon_prov: Dict[str, Any],
    n_segmented_frames: Optional[int],
) -> List[str]:
    """
    FINDINGS as measurements.  File paths are deliberately excluded -- they are
    recorded under PROVENANCE / ARTIFACTS instead.
    """
    metrics = cls_info.get("metrics") if isinstance(cls_info.get("metrics"), dict) else {}
    phases = cls_info.get("phase_indices") if isinstance(cls_info.get("phase_indices"), dict) else {}
    lines: List[str] = []

    seg_ok = bool(seg_info.get("ok")) and bool(seg_info.get("cardiac_seg_path_abs") or seg_info.get("cardiac_seg_path"))
    if not seg_ok and not metrics:
        lines.append("- No cardiac cine segmentation or ventricular measurements are available from this run.")
        return lines

    lv_edv = _fmt_num(metrics.get("lv_edv_ml"))
    lv_esv = _fmt_num(metrics.get("lv_esv_ml"))
    lv_ef = _fmt_num(metrics.get("lv_ef_percent"))
    lv_bits = [b for b in (
        f"EDV {lv_edv} mL" if lv_edv else None,
        f"ESV {lv_esv} mL" if lv_esv else None,
        f"EF {lv_ef}%" if lv_ef else None,
    ) if b]
    if lv_bits:
        lines.append("- Left ventricle: " + ", ".join(lv_bits) + ".")

    rv_edv = _fmt_num(metrics.get("rv_edv_ml"))
    rv_esv = _fmt_num(metrics.get("rv_esv_ml"))
    rv_ef = _fmt_num(metrics.get("rv_ef_percent"))
    rv_bits = [b for b in (
        f"EDV {rv_edv} mL" if rv_edv else None,
        f"ESV {rv_esv} mL" if rv_esv else None,
        f"EF {rv_ef}%" if rv_ef else None,
    ) if b]
    if rv_bits:
        lines.append("- Right ventricle: " + ", ".join(rv_bits) + ".")

    myo_vol = _fmt_num(metrics.get("myo_ed_volume_ml"))
    lv_mass = _fmt_num(metrics.get("lv_mass_g"))
    max_thk = _fmt_num(metrics.get("max_myo_thickness_mm"))
    myo_bits = [b for b in (
        f"end-diastolic volume {myo_vol} mL" if myo_vol else None,
        f"LV mass {lv_mass} g" if lv_mass else None,
        f"maximum wall thickness {max_thk} mm" if max_thk else None,
    ) if b]
    if myo_bits:
        lines.append("- Myocardium: " + ", ".join(myo_bits) + ".")

    idx_bits = [b for b in (
        f"LV EDVI {_fmt_num(metrics.get('lv_edvi_ml_m2'))} mL/m^2" if _fmt_num(metrics.get("lv_edvi_ml_m2")) else None,
        f"RV EDVI {_fmt_num(metrics.get('rv_edvi_ml_m2'))} mL/m^2" if _fmt_num(metrics.get("rv_edvi_ml_m2")) else None,
        f"LV mass index {_fmt_num(metrics.get('lv_mass_index_g_m2'))} g/m^2" if _fmt_num(metrics.get("lv_mass_index_g_m2")) else None,
    ) if b]
    if idx_bits:
        lines.append("- Body-surface-indexed values: " + ", ".join(idx_bits) + ".")
    elif metrics and "bsa_m2" in metrics and _safe_float(metrics.get("bsa_m2")) is None:
        lines.append(
            "- Body-surface-indexed values (LV/RV EDVI, LV mass index) not reported: "
            "height and weight were not available to the pipeline."
        )

    ed_1 = phases.get("ed_frame_1based")
    es_1 = phases.get("es_frame_1based")
    if ed_1 is not None or es_1 is not None:
        total = None
        if isinstance(recon_prov, dict) and recon_prov.get("ok") and recon_prov.get("n_phases") is not None:
            total = int(recon_prov["n_phases"])
        elif n_segmented_frames:
            total = int(n_segmented_frames)
        parts = []
        if ed_1 is not None:
            parts.append(f"end-diastole at frame {int(ed_1)}" + (f" of {total}" if total else ""))
        if es_1 is not None:
            parts.append(f"end-systole at frame {int(es_1)}" + (f" of {total}" if total else ""))
        src = str(phases.get("source") or "").strip()
        how = {
            "lv_volume_extrema": "selected as the frames of maximum and minimum segmented LV blood-pool volume",
        }.get(src, (f"phase selection source: {src}" if src else ""))
        line = "- Cardiac phases: " + ", ".join(parts)
        if how:
            line += f" ({how})"
        lines.append(line + ".")

    cov_bits: List[str] = []
    if isinstance(recon_prov, dict) and recon_prov.get("ok"):
        if recon_prov.get("n_slices") is not None and recon_prov.get("n_phases") is not None:
            cov_bits.append(
                f"{int(recon_prov['n_slices'])} short-axis slices x {int(recon_prov['n_phases'])} cardiac phases"
            )
        ps = recon_prov.get("pixel_spacing")
        if isinstance(ps, list) and len(ps) == 3:
            try:
                cov_bits.append(
                    f"in-plane {float(ps[0]):.2f} x {float(ps[1]):.2f} mm, slice thickness {float(ps[2]):.1f} mm"
                )
            except Exception:
                pass
    if n_segmented_frames:
        cov_bits.append(f"segmentation run on {int(n_segmented_frames)} cine frames")
    if cov_bits:
        lines.append("- Coverage: " + "; ".join(cov_bits) + ".")

    cproxy = metrics.get("local_contraction_proxy") if isinstance(metrics.get("local_contraction_proxy"), dict) else {}
    valid = cproxy.get("valid_slices")
    abn = cproxy.get("abnormal_slices")
    if valid is not None and abn is not None:
        mean_ratio = _fmt_num(cproxy.get("abnormal_ratio_mean"), 3)
        txt = (
            f"- Regional contraction proxy: myocardial area change from end-diastole to end-systole was "
            f"measurable on {int(valid)} short-axis slices; {int(abn)} of these changed by less than 10%"
        )
        if mean_ratio:
            txt += f" (mean fractional area change {mean_ratio})"
        lines.append(txt + ".")

    predicted_group = cls_info.get("predicted_group")
    if predicted_group:
        lines.append(f"- Rule-based disease class (non-diagnostic): {predicted_group}.")
    rule_trace = cls_info.get("rule_trace") if isinstance(cls_info.get("rule_trace"), list) else []
    for rule in rule_trace[:3]:
        if str(rule or "").strip():
            lines.append(f"- Classifier rule applied: {str(rule).strip()}")
    gt = cls_info.get("ground_truth_group")
    if gt:
        lines.append(f"- Reference label recorded with the dataset: {gt}.")
    return lines


# ---------------------------------------------------------------------------
# Optional LLM impression (default OFF)
# ---------------------------------------------------------------------------

# Entities a cine-morphometry pipeline cannot assess.  If an LLM impression
# mentions one, the impression is discarded and the deterministic template is
# used instead.
# Words an impression may use even though they carry no measurement: pure
# connectives, articles, and the handful of comparative terms needed to read a
# number back as a sentence. Deliberately small -- anything clinical must come
# from the measurements themselves.
_IMPRESSION_FUNCTION_WORDS: frozenset = frozenset("""
a an and are as at be been both but by for from has have in into is it its no nor not
of on or that the there these this to was were which with within without
also overall respectively while whereas however therefore thus
show shows shown demonstrate demonstrates indicate indicates measure measured measures
remain remains report reported reports record recorded records give gives given
value values study case run analysis section impression finding findings
normal preserved reduced increased decreased elevated low high larger smaller
above below over under between approximately about each per both all
""".split())


# Shared-prefix length that treats ventricle/ventricular and myocardium/
# myocardial as the same term without letting unrelated words through.
_IMPRESSION_STEM_LEN = 6

# Standard expansions of abbreviations that appear in the measurements. These
# introduce no new clinical entity -- they are the long form of a term the
# pipeline already reported -- so the vocabulary whitelist accepts them. Keep
# this list short, explicit and auditable; it is part of the safety guard.
_IMPRESSION_ABBREVIATION_EXPANSIONS: Dict[str, Tuple[str, ...]] = {
    "ef": ("ejection", "fraction"),
    "edv": ("end", "diastolic", "volume"),
    "esv": ("end", "systolic", "volume"),
    "edvi": ("end", "diastolic", "volume", "index", "indexed"),
    "lv": ("left", "ventricle", "ventricular"),
    "rv": ("right", "ventricle", "ventricular"),
    "myo": ("myocardium", "myocardial"),
    "nor": ("group", "class", "category"),
    "bsa": ("body", "surface", "area"),
}

_FILENAME_SUFFIXES: Tuple[str, ...] = (".nii", ".nii.gz", ".dcm", ".h5", ".hdf5", ".mha", ".nrrd", ".png", ".json")


def _looks_like_filename(value: str) -> bool:
    """True when a "description" is really just the file the data came from."""
    low = str(value or "").strip().lower()
    return bool(low) and (low.endswith(_FILENAME_SUFFIXES) or "/" in low)


def _impression_content_words(text: str) -> List[str]:
    """Lowercased alphabetic words of >=3 characters, function words removed."""
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", str(text or "").lower())
    return [w.strip("-'") for w in words if w.strip("-'") not in _IMPRESSION_FUNCTION_WORDS]


def _contains_term(haystack: str, term: str) -> bool:
    """Substring search that will not fire on a term embedded in another word.

    Plain ``in`` made "scar" match "di-scar-ded"; that failed safe (it rejected a
    valid impression) but it also made the guard look stricter than it is.
    """
    return re.search(r"(?<![a-z])" + re.escape(term), str(haystack or "").lower()) is not None


def _impression_unsupported_words(content: str, measurement_lines: Sequence[str]) -> List[str]:
    """Content words in *content* that do not appear in the supplied measurements.

    This is the whitelist that makes the guard sound: the model may rephrase and
    join what it was given, but it cannot introduce a clinical entity, an
    attribute, or a patient fact that the pipeline never produced.
    """
    supplied_text = "\n".join(measurement_lines)
    allowed = set(_impression_content_words(supplied_text))
    # Admit the long form of any abbreviation actually present.
    low_supplied = supplied_text.lower()
    for abbrev, expansion in _IMPRESSION_ABBREVIATION_EXPANSIONS.items():
        if re.search(r"(?<![a-z])" + abbrev + r"(?![a-z])", low_supplied):
            allowed.update(expansion)
    seen: List[str] = []
    for word in _impression_content_words(content):
        if word in allowed or word in seen:
            continue
        # Tolerate inflections and the adjective forms clinical prose needs:
        # ventricle/ventricular, myocardium/myocardial, mass/masses. A shared
        # six-character stem is enough for those and still rejects the words the
        # exploit introduced (signal, chambers, unremarkable, characteristics),
        # none of which stem-match anything in the measurements.
        if any(word.rstrip("s") == a.rstrip("s") for a in allowed):
            continue
        if len(word) >= _IMPRESSION_STEM_LEN and any(
            len(a) >= _IMPRESSION_STEM_LEN and word[:_IMPRESSION_STEM_LEN] == a[:_IMPRESSION_STEM_LEN]
            for a in allowed
        ):
            continue
        seen.append(word)
    return seen


_IMPRESSION_BANNED_SUBSTRINGS: Tuple[str, ...] = (
    "late gadolinium",
    "lge",
    "scar",
    "fibrosis",
    "infarct",
    "ischemi",
    "ischaemi",
    "perfusion",
    "edema",
    "oedema",
    "valve",
    "valvular",
    "regurgitat",
    "stenosis",
    "thrombus",
    "pericardial",
    "effusion",
    "aneurysm",
    "myocarditis",
    "amyloid",
    "sarcoid",
    "recommend",
    "biopsy",
    "should undergo",
    "consistent with a diagnosis",
)

_NUMBER_RE = None


def _extract_numbers(text: str) -> List[float]:
    import re

    global _NUMBER_RE
    if _NUMBER_RE is None:
        _NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
    out: List[float] = []
    for tok in _NUMBER_RE.findall(str(text or "")):
        try:
            out.append(float(tok))
        except Exception:
            continue
    return out


def _impression_numbers_grounded(text: str, allowed: List[float]) -> Tuple[bool, Optional[float]]:
    """
    Every number in an LLM impression must be a number the pipeline measured
    (optionally rounded to integer or one decimal).  Returns (ok, offending).
    """
    expanded: List[float] = []
    for a in allowed:
        try:
            expanded.extend([float(a), float(round(a)), float(round(a, 1))])
        except Exception:
            continue
    for x in _extract_numbers(text):
        if not any(abs(x - a) <= 0.05 for a in expanded):
            return False, x
    return True, None


def _ollama_no_think_chat(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout_s: int,
    num_predict: int,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call Ollama's native /api/chat with thinking disabled.

    Reasoning models served by Ollama spend the whole completion budget on the
    chain of thought when thinking is enabled, and `message.content` comes back
    empty.  `"think": false` suppresses the trace and returns the answer
    directly.  The reasoning trace is never used as an impression.

    Returns (content, error).
    """
    import urllib.error
    import urllib.request

    url = str(base_url or "").strip().rstrip("/")
    for suffix in ("/v1/chat/completions", "/api/chat", "/v1"):
        if url.lower().endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
            break
    endpoint = f"{url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": int(num_predict)},
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - transport
        return None, f"http_{exc.code}"
    except Exception as exc:  # pragma: no cover - transport
        return None, f"transport_error:{type(exc).__name__}"
    msg = body.get("message") if isinstance(body.get("message"), dict) else {}
    content = str(msg.get("content") or "").strip()
    if not content:
        return None, "empty_completion"
    return content, None


def _build_llm_impression(
    *,
    base_url: str,
    model: str,
    timeout_s: int,
    measurement_lines: List[str],
    allowed_numbers: List[float],
    max_chars: int = 1200,
) -> Dict[str, Any]:
    """
    Ask an LLM for an impression grounded ONLY in the measurements the pipeline
    produced, then verify it before allowing it anywhere near the report.

    Verification (all must pass, else the caller keeps the template):
      * non-empty completion,
      * length within `max_chars`,
      * every number present in the measured set,
      * no mention of an entity this pipeline cannot assess.
    """
    if not measurement_lines:
        return {"ok": False, "reason": "no_measurements"}
    system = (
        "You are drafting the IMPRESSION section of a cardiac MRI research report. "
        "You are given ONLY the measurements a segmentation-and-morphometry pipeline produced. "
        "Rules, without exception: "
        "(1) use only the numbers given, and quote them exactly as given; "
        "(2) do not introduce any finding, structure, or disease the measurements do not cover -- "
        "in particular say nothing about scar, fibrosis, infarction, perfusion, oedema, valves, "
        "pericardium, or tissue characterisation, because none were imaged; "
        "(3) you may restate the pipeline's own rule-based classification and the rule text verbatim; "
        "(4) make no recommendations and no diagnosis; "
        "(5) reply with 2 to 4 plain sentences, no headings, no bullet points, no preamble."
    )
    user = "Measurements from this study:\n" + "\n".join(measurement_lines) + "\n\nWrite the impression."
    content, err = _ollama_no_think_chat(
        base_url=base_url,
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        timeout_s=timeout_s,
        num_predict=400,
    )
    if content is None:
        return {"ok": False, "reason": err or "no_content"}
    if len(content) > max_chars:
        return {"ok": False, "reason": "too_long"}
    ok_nums, offender = _impression_numbers_grounded(content, allowed_numbers)
    if not ok_nums:
        return {"ok": False, "reason": f"ungrounded_number:{offender}"}
    low = content.lower()
    supplied = "\n".join(measurement_lines).lower()
    for bad in _IMPRESSION_BANNED_SUBSTRINGS:
        if _contains_term(low, bad) and not _contains_term(supplied, bad):
            return {"ok": False, "reason": f"unsupported_entity:{bad}"}
    # The banned list is a blacklist and a blacklist cannot enumerate every
    # unsupported clinical claim. Review demonstrated the hole: "the myocardium
    # demonstrates normal signal characteristics ... within normal limits for
    # this patient's age" carries no banned term and no ungrounded number, yet
    # asserts tissue characterisation this pipeline never performs and an age it
    # never had. So the real gate is a vocabulary whitelist: every content word
    # in the impression must already appear in the measurements it was given.
    stray = _impression_unsupported_words(content, measurement_lines)
    if stray:
        return {"ok": False, "reason": "unsupported_vocabulary:" + ",".join(stray[:5])}
    return {"ok": True, "text": content, "model": model}


def generate_report(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    case_state_path = Path(args["case_state_path"]).expanduser().resolve()
    feature_table_path = args.get("feature_table_path")

    out_subdir = args.get("output_subdir", "report")
    out_dir = ctx.artifacts_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    state = json.loads(case_state_path.read_text(encoding="utf-8"))
    stage_outputs = state.get("stage_outputs", {}) or {}
    run_dir = case_state_path.parent

    # Domain resolution, most trusted first:
    #   1. explicit `domain` argument (what the planner/executor passes),
    #   2. `metadata.domain` recorded on the CaseState by the run that produced it,
    #   3. the historical default.
    # (2) matters: the domain decides which domain-specific blocks the report is
    # allowed to emit, and guessing it is how a cardiac case ended up carrying
    # PI-RADS metadata.
    domain_name = str(args.get("domain") or "").strip().lower()
    state_meta = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    state_domain = str(state_meta.get("domain") or "").strip().lower()
    if domain_name:
        domain_source = "argument"
    elif state_domain:
        domain_source = "case_state.metadata.domain"
    else:
        domain_source = "default"
    domain_cfg = get_domain_config(domain_name or state_domain or "prostate")

    vlm_dir = ctx.artifacts_dir / "vlm"
    vlm_dir.mkdir(parents=True, exist_ok=True)

    def _stage_sort_key(k: str):
        try:
            return (0, int(k), "")
        except Exception:
            return (1, 10**9, str(k))

    def best_record(tool_name: str) -> Tuple[Optional[bool], Dict[str, Any]]:
        """
        Prefer a successful record if any exists; otherwise fall back to the latest record.
        This avoids "LLM planned call failed" masking a later auto-repair success.
        """
        if not isinstance(stage_outputs, dict):
            return None, {}

        latest_ok: Optional[bool] = None
        latest_data: Dict[str, Any] = {}
        best_ok: Optional[bool] = None
        best_data: Dict[str, Any] = {}

        for st in sorted(stage_outputs.keys(), key=_stage_sort_key):
            tools = stage_outputs.get(st, {}) or {}
            if not isinstance(tools, dict):
                continue
            records = tools.get(tool_name, []) or []
            if not isinstance(records, list) or not records:
                continue
            rec = records[-1] or {}
            latest_ok = rec.get("ok")
            latest_data = rec.get("data", {}) or {}
            if rec.get("ok") is True:
                best_ok = True
                best_data = latest_data

        if best_ok is True:
            return True, best_data
        return latest_ok, latest_data

    ingest_ok, ingest = best_record("ingest_dicom_to_nifti")
    ident_ok, ident = best_record("identify_sequences")
    reg_ok, reg = best_record("register_to_reference")
    seg_ok, seg = best_record("segment_prostate")
    card_seg_ok, card_seg = best_record("segment_cardiac_cine")
    card_cls_ok, card_cls = best_record("classify_cardiac_cine_disease")
    brain_seg_ok, brain_seg = best_record("brats_mri_segmentation")
    feat_ok, feat = best_record("extract_roi_features")
    lesion_ok, lesion = best_record("detect_lesion_candidates")
    recon_ok, recon = best_record("reconstruct_grappa")
    seg_data = seg if bool(seg_ok) else (card_seg if isinstance(card_seg, dict) else {})
    seg_any_ok = bool(seg_ok) or bool(card_seg_ok)
    recon_provenance = _build_reconstruction_provenance(recon_ok, recon)

    mapping = (ident.get("mapping", {}) if isinstance(ident, dict) else {}) or {}
    series = (ingest.get("series", []) if isinstance(ingest, dict) else []) or []
    if not series and isinstance(ident, dict):
        series = ident.get("series", []) or []
    present = sorted(set((s.get("sequence_guess") for s in series if isinstance(s, dict) and s.get("sequence_guess"))))
    # Heuristic: consider ingest OK if identify produced NIfTI outputs or ingest artifacts exist.
    try:
        run_art_dir = case_state_path.parent / "artifacts"
        nifti_dir = run_art_dir / "ingest" / "nifti"
        have_nifti = nifti_dir.exists() and any(nifti_dir.glob("*.nii*"))
        if not ingest_ok and ident_ok:
            ingest_ok = bool(have_nifti or (isinstance(ident.get("nifti_by_series"), dict) and ident.get("nifti_by_series")))
        if not ingest_ok and have_nifti:
            ingest_ok = True
    except Exception:
        pass
    if not present and isinstance(mapping, dict) and mapping:
        present = sorted(list(mapping.keys()))

    feature_path = feature_table_path or (feat.get("feature_table_path") if isinstance(feat, dict) else None)
    slice_path = (feat.get("slice_summary_path") if isinstance(feat, dict) else None)
    overlay_list = (feat.get("overlay_pngs") if isinstance(feat, dict) else [])

    run_art_dir = case_state_path.parent / "artifacts"

    def _discover_main_feature_path() -> Optional[str]:
        cand = run_art_dir / "features" / "features.csv"
        if cand.exists():
            return str(cand)
        try:
            cands: List[Path] = []
            for p in run_art_dir.rglob("features.csv"):
                parts = [str(x) for x in p.parts]
                if "features_lesion" in parts:
                    continue
                if p.parent.name != "features":
                    continue
                cands.append(p)
            if not cands:
                return None
            cands.sort(key=lambda x: len(str(x)))
            return str(cands[0])
        except Exception:
            return None

    def _feature_set_from_paths(
        feat_path: Optional[str], slc_path: Optional[str], overlays: Any, *, ok_hint: bool
    ) -> Dict[str, Any]:
        overlay_items = overlays if isinstance(overlays, list) else []
        md_path = _find_feature_md_path(feat_path)
        return {
            "ok": bool(ok_hint or feat_path),
            "feature_table_path": to_short_path(feat_path, run_dir=run_dir),
            "feature_table_path_abs": feat_path,
            "feature_table_md_path": to_short_path(md_path, run_dir=run_dir),
            "feature_table_md_path_abs": md_path,
            "slice_summary_path": to_short_path(slc_path, run_dir=run_dir),
            "slice_summary_path_abs": slc_path,
            "overlay_pngs": list_with_short_paths(overlay_items, run_dir=run_dir),
            "features_preview_rows": _read_features_csv_preview(feat_path, max_rows=64),
            "slice_summary_preview": _read_slice_summary_preview(slc_path, per_sequence_cap=12),
        }

    def _infer_overlay_meta(path: str) -> Dict[str, str]:
        low = path.lower()
        kind = ""
        if "full_frame_montage_4x4" in low:
            kind = "full_frame_montage_4x4"
        elif "full_frame_montage_5x5" in low:
            kind = "full_frame_montage_5x5"
        elif "full_frame_montage" in low:
            kind = "full_frame_montage"
        elif "full_frame_best" in low:
            kind = "full_frame_best_slice"
        elif "roi_crop_montage_4x4" in low:
            kind = "roi_crop_montage_4x4"
        elif "roi_crop_montage_5x5" in low:
            kind = "roi_crop_montage_5x5"
        elif "roi_crop_montage" in low:
            kind = "roi_crop_montage"
        elif "roi_crop_best" in low:
            kind = "roi_crop_best_slice"
        elif low.endswith("_tiles.json"):
            if "full_frame_montage" in low:
                kind = "full_frame_montage_4x4_tiles"
            else:
                kind = "roi_crop_montage_4x4_tiles"
        seq = ""
        if "adc" in low:
            seq = "ADC"
        elif "t2" in low:
            seq = "T2w"
        elif "high_b" in low or "high-b" in low:
            seq = "DWI_high_b"
        elif "low_b" in low or "low-b" in low:
            seq = "DWI_low_b"
        elif "dwi" in low:
            seq = "DWI"
        roi_src = ""
        if "lesion" in low:
            roi_src = "lesion"
        elif "prostate" in low:
            roi_src = "prostate"
        meta: Dict[str, str] = {}
        if kind:
            meta["kind"] = kind
        if seq:
            meta["sequence"] = seq
        if roi_src:
            meta["roi_source"] = roi_src
        return meta

    def _feature_set_from_subdir(subdir: str) -> Dict[str, Any]:
        base = run_art_dir / subdir
        feat_p = base / "features.csv"
        slc_p = base / "slice_summary.json"
        rad_p = base / "radiomics_summary.json"
        overlays_dir = base / "overlays"
        feat_s = str(feat_p) if feat_p.exists() else None
        slc_s = str(slc_p) if slc_p.exists() else None
        rad_s = str(rad_p) if rad_p.exists() else None
        ovs: List[Dict[str, Any]] = []
        if overlays_dir.exists():
            for p in sorted(overlays_dir.glob("*")):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in {".png", ".json"}:
                    continue
                meta = _infer_overlay_meta(str(p))
                item: Dict[str, Any] = {"path": str(p)}
                item.update(meta)
                ovs.append(item)
        out = _feature_set_from_paths(feat_s, slc_s, ovs, ok_hint=bool(feat_s))
        if rad_s:
            out["radiomics_summary_path"] = to_short_path(rad_s, run_dir=run_dir)
            out["radiomics_summary_path_abs"] = rad_s
        return out

    features_main = _feature_set_from_subdir("features")
    features_lesion = _feature_set_from_subdir("features_lesion")

    last_is_lesion = isinstance(feature_path, str) and "features_lesion" in feature_path.replace("\\", "/")
    if not features_main.get("ok"):
        main_disc = _discover_main_feature_path()
        if main_disc:
            main_slc = str(Path(main_disc).with_name("slice_summary.json"))
            features_main = _feature_set_from_paths(main_disc, main_slc, [], ok_hint=True)
        elif not last_is_lesion:
            features_main = _feature_set_from_paths(feature_path, slice_path, overlay_list, ok_hint=bool(feat_ok))

    if not features_lesion.get("ok") and last_is_lesion:
        features_lesion = _feature_set_from_paths(feature_path, slice_path, overlay_list, ok_hint=bool(feat_ok))

    features_primary = features_main if features_main.get("ok") else features_lesion
    main_feature_path = features_main.get("feature_table_path_abs")
    lesion_feature_path = features_lesion.get("feature_table_path_abs")
    overlay_list_main = features_main.get("overlay_pngs") or []
    overlay_list_lesion = features_lesion.get("overlay_pngs") or []
    short_feature_path = features_primary.get("feature_table_path")
    short_slice_path = features_primary.get("slice_summary_path")
    short_overlays = features_primary.get("overlay_pngs") or []
    features_preview = features_primary.get("features_preview_rows") or []
    slice_preview = features_primary.get("slice_summary_preview") or {}
    cardiac_t1_feature_analysis = (
        _build_cardiac_t1_feature_analysis(features_main=features_main, cardiac_classification=card_cls if isinstance(card_cls, dict) else {})
        if domain_cfg.name == "cardiac"
        else None
    )

    lesion_mask_path = lesion.get("lesion_mask_path") if isinstance(lesion, dict) else None
    candidates_path = lesion.get("candidates_path") if isinstance(lesion, dict) else None
    prostate_mask_path = seg_data.get("prostate_mask_path") if isinstance(seg_data, dict) else None
    zone_mask_path = seg_data.get("zone_mask_path") if isinstance(seg_data, dict) else None
    lesion_geometry = _compute_lesion_geometry(
        lesion_mask_path=lesion_mask_path, prostate_mask_path=prostate_mask_path, zone_mask_path=zone_mask_path
    )

    # Standardized VLM evidence bundle (include *content previews*, not just paths)
    # --- Degradation context (Track 2: missing modality awareness) ---
    degradation_context = args.get("degradation_context")
    if not isinstance(degradation_context, dict):
        degradation_context = None

    evidence_bundle = {
        "case_id": state.get("case_id"),
        "run_id": state.get("run_id"),
        "degradation_context": degradation_context,
        "mapping": mapping,
        "pipeline_status": {
            "extract_roi_features_ok": bool(feat_ok),
            "extract_roi_features_main_ok": bool(features_main.get("ok")),
            "extract_roi_features_lesion_ok": bool(features_lesion.get("ok")),
            "segment_prostate_ok": bool(seg_ok),
            "segment_cardiac_cine_ok": bool(card_seg_ok),
            "classify_cardiac_cine_disease_ok": bool(card_cls_ok),
            "reconstruct_grappa_ok": bool(recon_ok),
            "sequences_present": present,
        },
        # Back-compat: `features` points to the primary (prefer main) set.
        "features": features_primary,
        "features_main": features_main,
        "features_lesion": features_lesion,
        "lesion_candidates": {
            "ok": bool(lesion_ok),
            "candidates_path": to_short_path(candidates_path, run_dir=run_dir),
            "candidates_path_abs": candidates_path,
            "lesion_mask_path": to_short_path(lesion_mask_path, run_dir=run_dir),
            "lesion_mask_path_abs": lesion_mask_path,
        },
        "lesion_geometry": lesion_geometry,
        "segmentation": {
            "ok": bool(seg_any_ok),
            "prostate_mask_path": to_short_path((seg_data.get("prostate_mask_path") if isinstance(seg_data, dict) else None), run_dir=run_dir),
            "prostate_mask_path_abs": (seg_data.get("prostate_mask_path") if isinstance(seg_data, dict) else None),
            "zone_mask_path": to_short_path((seg_data.get("zone_mask_path") if isinstance(seg_data, dict) else None), run_dir=run_dir),
            "zone_mask_path_abs": (seg_data.get("zone_mask_path") if isinstance(seg_data, dict) else None),
            "cardiac_seg_path": to_short_path((seg_data.get("seg_path") if isinstance(seg_data, dict) else None), run_dir=run_dir),
            "cardiac_seg_path_abs": (seg_data.get("seg_path") if isinstance(seg_data, dict) else None),
            "rv_mask_path": to_short_path((seg_data.get("rv_mask_path") if isinstance(seg_data, dict) else None), run_dir=run_dir),
            "rv_mask_path_abs": (seg_data.get("rv_mask_path") if isinstance(seg_data, dict) else None),
            "myo_mask_path": to_short_path((seg_data.get("myo_mask_path") if isinstance(seg_data, dict) else None), run_dir=run_dir),
            "myo_mask_path_abs": (seg_data.get("myo_mask_path") if isinstance(seg_data, dict) else None),
            "lv_mask_path": to_short_path((seg_data.get("lv_mask_path") if isinstance(seg_data, dict) else None), run_dir=run_dir),
            "lv_mask_path_abs": (seg_data.get("lv_mask_path") if isinstance(seg_data, dict) else None),
        },
        "cardiac_classification": {
            "ok": bool(card_cls_ok),
            "classification_path": to_short_path((card_cls.get("classification_path") if isinstance(card_cls, dict) else None), run_dir=run_dir),
            "classification_path_abs": (card_cls.get("classification_path") if isinstance(card_cls, dict) else None),
            "predicted_group": (card_cls.get("predicted_group") if isinstance(card_cls, dict) else None),
            "ground_truth_group": (card_cls.get("ground_truth_group") if isinstance(card_cls, dict) else None),
            "ground_truth_match": (card_cls.get("ground_truth_match") if isinstance(card_cls, dict) else None),
            "needs_vlm_review": (card_cls.get("needs_vlm_review") if isinstance(card_cls, dict) else None),
            "metrics": (card_cls.get("metrics") if isinstance(card_cls, dict) else None),
            "phase_indices": (card_cls.get("phase_indices") if isinstance(card_cls, dict) else None),
            "rule_trace": (card_cls.get("rule_trace") if isinstance(card_cls, dict) else None),
        },
        "cardiac_t1_feature_analysis": cardiac_t1_feature_analysis,
        "reconstruction": recon_provenance,
        "qc_pngs": [to_short_path(p, run_dir=run_dir) for p in _gather_pngs(run_dir)],
        "notes": [
            "Ground all statements in evidence. If evidence is insufficient, say 'indeterminate' and list limitations.",
            "Do not invent lesions or numeric measurements not present in features.csv (or features.md) / slice_summary.json.",
            (
                "Prefer cardiac_t1_feature_analysis for T1 evidence-first interpretation; avoid non-evidence visual guesswork."
                if domain_cfg.name == "cardiac"
                else "Prefer features_main for prostate volume; features_lesion describes lesion ROIs."
            ),
        ],
    }

    # Inject degradation advisory into evidence notes for the VLM.
    if degradation_context:
        missing = degradation_context.get("missing_modalities") or []
        omitted = degradation_context.get("omitted_tools") or []
        if missing or omitted:
            advisory = (
                "LIMITED EXAM: This case has missing modalities. "
                f"Missing: {', '.join(missing)}. "
                f"Omitted pipeline steps: {', '.join(omitted)}. "
                "The report MUST explicitly state these limitations in the Limitations section. "
                "Do NOT synthesize findings for modalities that were not acquired."
            )
            evidence_bundle["notes"].append(advisory)
            evidence_bundle["degradation_advisory"] = advisory

    artifact_index_path, artifact_index = build_artifact_index(case_state_path=case_state_path, artifacts_dir=ctx.artifacts_dir)
    evidence_bundle["artifact_index_path"] = to_short_path(str(artifact_index_path), run_dir=run_dir)
    evidence_bundle["artifact_index"] = {"n_artifacts": len(artifact_index.get("artifacts") or [])}

    vlm_evidence_bundle_path = vlm_dir / "vlm_evidence_bundle.json"
    vlm_evidence_bundle_path.write_text(json.dumps(evidence_bundle, indent=2) + "\n", encoding="utf-8")

    # Stable CaseContext (separate from CaseState/execution log). This is meant to be a durable,
    # predictable "entry point" for downstream tools and multi-case workflows.
    context_dir = ctx.artifacts_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    case_context_path = context_dir / "case_context.json"

    # Minimal measurement summary from features preview
    meas = {}
    try:
        rows = evidence_bundle.get("features_main", {}).get("features_preview_rows") or evidence_bundle.get(
            "features", {}
        ).get("features_preview_rows") or []
        if isinstance(rows, list):
            # prostate volume from any whole_gland row
            for r in rows:
                if isinstance(r, dict) and r.get("roi") == "whole_gland" and r.get("volume_ml") is not None:
                    meas["prostate_volume_ml"] = float(r["volume_ml"])
                    break
    except Exception:
        pass

    case_context = {
        "case_id": state.get("case_id"),
        "run_id": state.get("run_id"),
        "generated_at": str(state.get("created_at") or ""),
        "reference": {"suggested": "T2w"},
        "mapping": mapping,
        "stage_meta": state.get("stage_meta", {}) if isinstance(state, dict) else {},
        "artifacts": {
            "segmentation": evidence_bundle.get("segmentation", {}),
            "cardiac_classification": evidence_bundle.get("cardiac_classification", {}),
            "features": evidence_bundle.get("features", {}),
            "features_main": evidence_bundle.get("features_main", {}),
            "features_lesion": evidence_bundle.get("features_lesion", {}),
            "lesion_candidates": evidence_bundle.get("lesion_candidates", {}),
            "lesion_geometry": evidence_bundle.get("lesion_geometry", {}),
            "cardiac_t1_feature_analysis": evidence_bundle.get("cardiac_t1_feature_analysis", {}),
            "qc_pngs": evidence_bundle.get("qc_pngs"),
            "vlm_evidence_bundle_path": to_short_path(str(vlm_evidence_bundle_path), run_dir=run_dir),
            "vlm_evidence_bundle_path_abs": str(vlm_evidence_bundle_path),
            "artifact_index_path": evidence_bundle.get("artifact_index_path"),
        },
        "measurements_summary": meas,
        "tool_versions": {"generate_report": REPORT_SPEC.version},
    }
    case_context_path.write_text(json.dumps(case_context, indent=2) + "\n", encoding="utf-8")

    # ---- Clinical-style report (lightweight template) ----
    series_list = series if isinstance(series, list) else []
    t2_series = next((s for s in series_list if isinstance(s, dict) and s.get("sequence_guess") == "T2w"), None)
    t2_tags = _parse_dicom_header_tags(t2_series.get("dicom_header_txt") if isinstance(t2_series, dict) else None)
    # Fallback: use dicom_meta.json if series headers are unavailable (ingest skipped).
    if not t2_tags:
        dicom_meta_path = None
        if isinstance(ingest, dict):
            dicom_meta_path = ingest.get("dicom_meta_path")
        if not dicom_meta_path and isinstance(ident, dict):
            dicom_meta_path = ident.get("dicom_meta_path")
        if dicom_meta_path and Path(str(dicom_meta_path)).exists():
            try:
                meta_obj = json.loads(Path(str(dicom_meta_path)).read_text(encoding="utf-8"))
                series_meta = meta_obj.get("series_meta", {}) if isinstance(meta_obj, dict) else {}
                t2_src = None
                if isinstance(mapping, dict):
                    t2_src = mapping.get("T2w") or mapping.get("t2w")
                best = None
                if t2_src:
                    t2_src = str(t2_src)
                    for name, item in series_meta.items():
                        if not isinstance(item, dict):
                            continue
                        sdir = str(item.get("series_dir") or "")
                        if sdir == t2_src or (t2_src and sdir and sdir.endswith(Path(t2_src).name)):
                            best = item
                            break
                if best is None:
                    for name, item in series_meta.items():
                        if "t2" in str(name).lower():
                            best = item if isinstance(item, dict) else None
                            break
                if isinstance(best, dict):
                    header = best.get("header") if isinstance(best.get("header"), dict) else {}
                    if header:
                        t2_tags = {str(k): str(v) for k, v in header.items() if v is not None}
                    elif best.get("dicom_header_txt"):
                        t2_tags = _parse_dicom_header_tags(best.get("dicom_header_txt"))
            except Exception:
                t2_tags = t2_tags or {}
    study_desc = (
        t2_tags.get("ReasonForStudy")
        or t2_tags.get("ClinicalTrialProtocolID")
        or t2_tags.get("ClinicalTrialProtocolName")
        or t2_tags.get("StudyDescription")
        or t2_tags.get("SeriesDescription")
        or t2_tags.get("BodyPartExamined")
    )
    manufacturer = t2_tags.get("Manufacturer") or ""
    field_strength = t2_tags.get("MagneticFieldStrength") or ""
    model = t2_tags.get("ManufacturerModelName") or ""
    slice_thickness = t2_tags.get("SliceThickness") or ""

    seq_descs: List[str] = []
    dwi_bvals: List[str] = []
    for s in series_list:
        if not isinstance(s, dict):
            continue
        guess = str(s.get("sequence_guess") or s.get("series_name") or "series")
        desc = str(s.get("SeriesDescription") or s.get("series_name") or guess)
        # TECHNICAL FACTORS is narrative prose. When the only "description" we
        # have is the file the volume came from (the NIfTI path case), naming the
        # sequence is the fact; the filename belongs in PROVENANCE.
        seq_descs.append(guess if _looks_like_filename(desc) or desc == guess else f"{guess}: {desc}")
        if str(s.get("sequence_guess")) == "DWI":
            bvals = s.get("b_values")
            if isinstance(bvals, list) and bvals:
                dwi_bvals = [str(b) for b in bvals]
    if not seq_descs:
        if isinstance(mapping, dict) and mapping:
            seq_descs = [str(k) for k in mapping.keys()]
        else:
            try:
                dicom_meta_path = None
                if isinstance(ingest, dict):
                    dicom_meta_path = ingest.get("dicom_meta_path")
                if not dicom_meta_path and isinstance(ident, dict):
                    dicom_meta_path = ident.get("dicom_meta_path")
                if dicom_meta_path and Path(str(dicom_meta_path)).exists():
                    meta_obj = json.loads(Path(str(dicom_meta_path)).read_text(encoding="utf-8"))
                    series_meta = meta_obj.get("series_meta", {}) if isinstance(meta_obj, dict) else {}
                    for name, item in series_meta.items():
                        if not isinstance(item, dict):
                            continue
                        header = item.get("header") if isinstance(item.get("header"), dict) else {}
                        desc = header.get("SeriesDescription") or header.get("ProtocolName") or name
                        seq_descs.append(f"{name}: {desc}")
                        diff = item.get("diffusion") if isinstance(item.get("diffusion"), dict) else {}
                        bvals = diff.get("b_values")
                        if isinstance(bvals, list) and bvals:
                            dwi_bvals = [str(b) for b in bvals]
            except Exception:
                pass

    # Volume summaries
    prostate_vol_ml: Optional[float] = None
    brain_wt_vol_ml: Optional[float] = None
    brain_tc_vol_ml: Optional[float] = None
    brain_et_vol_ml: Optional[float] = None
    lesion_vol_ml: Optional[float] = None
    tz_vol_ml: Optional[float] = None
    pz_vol_ml: Optional[float] = None
    def _get_roi_vol(rows: List[Dict[str, str]], name: str) -> Optional[float]:
        for r in rows:
            roi = str(r.get("roi") or "").strip()
            src = str(r.get("roi_source") or "").strip()
            if roi.lower() == name.lower() or src.lower() == name.lower():
                try:
                    return float(r.get("volume_ml") or 0.0)
                except Exception:
                    return None
        return None

    main_rows: List[Dict[str, str]] = []
    lesion_rows: List[Dict[str, str]] = []
    try:
        main_rows = _read_csv_rows(main_feature_path)
    except Exception:
        main_rows = []

    try:
        lesion_rows = _read_csv_rows(lesion_feature_path)
    except Exception:
        lesion_rows = []

    try:
        if main_rows:
            brain_wt_vol_ml = _get_roi_vol(main_rows, "Whole_Tumor")
            brain_tc_vol_ml = _get_roi_vol(main_rows, "Tumor_Core")
            brain_et_vol_ml = _get_roi_vol(main_rows, "Enhancing_Tumor")
        if domain_cfg.name != "brain":
            # Prefer T2w for prostate volume, fallback to ADC/high_b
            for seq in ("T2w_t2w_input", "T2w", "ADC", "high_b"):
                for r in main_rows:
                    if (
                        r.get("roi") == "whole_gland"
                        and r.get("roi_source") != "lesion"
                        and r.get("sequence") == seq
                    ):
                        prostate_vol_ml = float(r.get("volume_ml") or 0.0)
                        break
                if prostate_vol_ml is not None:
                    break
            if prostate_vol_ml is None:
                for r in main_rows:
                    if r.get("roi") == "whole_gland" and r.get("roi_source") != "lesion":
                        prostate_vol_ml = float(r.get("volume_ml") or 0.0)
                        break

        def _is_lesion_feature_row(r: Dict[str, str]) -> bool:
            roi = str(r.get("roi") or "").strip().lower()
            src = str(r.get("roi_source") or "").strip().lower()
            # Backward-compatible legacy row: roi_source=lesion, roi=whole_gland
            if src == "lesion" and roi == "whole_gland":
                return True
            # Current per-lesion-component rows: roi_source=lesion_component_*, roi=lesion
            if roi == "lesion":
                return True
            if src.startswith("lesion_component"):
                return True
            return False

        for seq in ("T2w_t2w_input", "T2w", "ADC", "high_b"):
            for r in lesion_rows:
                if _is_lesion_feature_row(r) and r.get("sequence") == seq:
                    lesion_vol_ml = float(r.get("volume_ml") or 0.0)
                    break
            if lesion_vol_ml is not None:
                break
        if lesion_vol_ml is None:
            for r in lesion_rows:
                if _is_lesion_feature_row(r):
                    lesion_vol_ml = float(r.get("volume_ml") or 0.0)
                    break

        for seq in ("T2w_t2w_input", "T2w", "ADC", "high_b"):
            for r in main_rows:
                if (
                    r.get("roi_source") != "lesion"
                    and r.get("roi") == "central_gland"
                    and r.get("sequence") == seq
                ):
                    tz_vol_ml = float(r.get("volume_ml") or 0.0)
                    break
            if tz_vol_ml is not None:
                break
        for seq in ("T2w_t2w_input", "T2w", "ADC", "high_b"):
            for r in main_rows:
                if (
                    r.get("roi_source") != "lesion"
                    and r.get("roi") == "peripheral_zone"
                    and r.get("sequence") == seq
                ):
                    pz_vol_ml = float(r.get("volume_ml") or 0.0)
                    break
            if pz_vol_ml is not None:
                break
    except Exception:
        tz_vol_ml = None
        pz_vol_ml = None

    lesion_geo_best: Optional[Dict[str, Any]] = None
    try:
        if isinstance(lesion_geometry, dict) and lesion_geometry.get("ok"):
            lesions = lesion_geometry.get("lesions")
            if isinstance(lesions, list) and lesions:
                lesions_sorted = sorted(
                    [l for l in lesions if isinstance(l, dict)],
                    key=lambda x: float(x.get("volume_cc") or 0.0),
                    reverse=True,
                )
                if lesions_sorted:
                    lesion_geo_best = lesions_sorted[0]
    except Exception:
        lesion_geo_best = None

    # Ignore tiny lesion geometry that is below the clinical relevance floor.
    if lesion_geo_best is not None:
        try:
            if float(lesion_geo_best.get("volume_cc") or 0.0) < float(MIN_CLINICALLY_MEANINGFUL_LESION_CC):
                lesion_geo_best = None
        except Exception:
            lesion_geo_best = None

    # Lesion signal summaries (T2/ADC/DWI) for clinical narrative and PI-RADS heuristics.
    lesion_t2_mean: Optional[float] = None
    lesion_adc_mean: Optional[float] = None
    lesion_high_b_mean: Optional[float] = None
    ref_t2_mean: Optional[float] = None
    ref_adc_mean: Optional[float] = None
    ref_high_b_mean: Optional[float] = None
    ref_roi_used: Optional[str] = None

    def _as_float(val: Any) -> Optional[float]:
        try:
            return float(val)
        except Exception:
            return None

    def _seq_has(seq: str, keys: Tuple[str, ...]) -> bool:
        s = str(seq or "").lower()
        return any(k in s for k in keys)

    def _mean_from(
        rows: List[Dict[str, str]],
        *,
        roi: Optional[str],
        roi_source: Optional[str],
        seq_keys: Tuple[str, ...],
    ) -> Optional[float]:
        for r in rows or []:
            if roi and str(r.get("roi") or "") != roi:
                continue
            if roi_source is not None and str(r.get("roi_source") or "") != roi_source:
                continue
            seq = str(r.get("sequence") or "")
            if seq_keys and not _seq_has(seq, seq_keys):
                continue
            mean_val = _as_float(r.get("mean_intensity") or r.get("mean") or r.get("mean_signal"))
            if mean_val is not None:
                return mean_val
        return None

    def _mean_dwi(rows: List[Dict[str, str]], *, roi: Optional[str], roi_source: Optional[str]) -> Optional[float]:
        val = _mean_from(rows, roi=roi, roi_source=roi_source, seq_keys=("high_b", "high-b", "dwi_high", "b1400", "highb"))
        if val is None:
            val = _mean_from(rows, roi=roi, roi_source=roi_source, seq_keys=("dwi",))
        return val

    def _is_lesion_metric_row(r: Dict[str, str]) -> bool:
        roi = str(r.get("roi") or "").strip().lower()
        src = str(r.get("roi_source") or "").strip().lower()
        if src == "lesion" and roi == "whole_gland":
            return True
        if roi == "lesion":
            return True
        if src.startswith("lesion_component"):
            return True
        return False

    lesion_component_hint: Optional[str] = None
    if lesion_geo_best is not None:
        try:
            comp_id = lesion_geo_best.get("component_id")
            if comp_id is not None:
                lesion_component_hint = f"lesion_component_{int(comp_id)}"
        except Exception:
            lesion_component_hint = None

    def _mean_from_lesion_rows(seq_keys: Tuple[str, ...]) -> Optional[float]:
        def _scan(prefer_component: bool) -> Optional[float]:
            for r in lesion_rows or []:
                if not _is_lesion_metric_row(r):
                    continue
                src = str(r.get("roi_source") or "").strip()
                if prefer_component and lesion_component_hint and src != lesion_component_hint:
                    continue
                seq = str(r.get("sequence") or "")
                if seq_keys and not _seq_has(seq, seq_keys):
                    continue
                mean_val = _as_float(r.get("mean_intensity") or r.get("mean") or r.get("mean_signal"))
                if mean_val is not None:
                    return mean_val
            return None

        if lesion_component_hint:
            preferred = _scan(prefer_component=True)
            if preferred is not None:
                return preferred
        return _scan(prefer_component=False)

    # Lesion means from lesion ROI (legacy + lesion_component_* formats).
    lesion_t2_mean = _mean_from_lesion_rows(("t2w",))
    lesion_adc_mean = _mean_from_lesion_rows(("adc",))
    lesion_high_b_mean = _mean_from_lesion_rows(("high_b", "high-b", "dwi_high", "b1400", "highb", "dwi"))

    # Pick a reference ROI based on lesion zone, with fallbacks.
    zone_label = str(lesion_geo_best.get("zone") or "") if lesion_geo_best else ""
    zone_upper = zone_label.upper()
    if "PZ" in zone_upper:
        ref_roi_used = "peripheral_zone"
    elif "TZ" in zone_upper or "CG" in zone_upper:
        ref_roi_used = "central_gland"
    else:
        ref_roi_used = None

    if ref_roi_used:
        ref_t2_mean = _mean_from(main_rows, roi=ref_roi_used, roi_source=None, seq_keys=("t2w",))
        ref_adc_mean = _mean_from(main_rows, roi=ref_roi_used, roi_source=None, seq_keys=("adc",))
        ref_high_b_mean = _mean_dwi(main_rows, roi=ref_roi_used, roi_source=None)
    if ref_t2_mean is None:
        ref_t2_mean = _mean_from(main_rows, roi="whole_gland", roi_source=None, seq_keys=("t2w",))
    if ref_adc_mean is None:
        ref_adc_mean = _mean_from(main_rows, roi="whole_gland", roi_source=None, seq_keys=("adc",))
    if ref_high_b_mean is None:
        ref_high_b_mean = _mean_dwi(main_rows, roi="whole_gland", roi_source=None)

    # Lesion candidates summary
    lesion_candidates_n: Optional[int] = None
    lesion_max_prob: Optional[float] = None
    lesion_threshold: Optional[float] = None
    try:
        cand_path = None
        if isinstance(lesion, dict):
            cand_path = lesion.get("candidates_path")
        if not cand_path:
            cand = (case_state_path.parent / "artifacts" / "lesion" / "candidates.json")
            if cand.exists():
                cand_path = str(cand)
        if cand_path and Path(cand_path).exists():
            cobj = json.loads(Path(cand_path).read_text(encoding="utf-8"))
            cands = cobj.get("candidates") if isinstance(cobj, dict) else None
            if isinstance(cobj, dict):
                try:
                    lesion_threshold = float(cobj.get("threshold"))
                except Exception:
                    lesion_threshold = None
            if isinstance(cands, list):
                lesion_candidates_n = int(len(cands))
                if cands:
                    try:
                        lesion_max_prob = float(cands[0].get("max_prob"))
                    except Exception:
                        lesion_max_prob = None
    except Exception:
        pass

    def _safe_float(val: Any) -> Optional[float]:
        try:
            return float(val)
        except Exception:
            return None

    def _ratio(val: Optional[float], ref: Optional[float]) -> Optional[float]:
        if val is None or ref is None:
            return None
        try:
            if float(ref) == 0.0:
                return None
            return float(val) / float(ref)
        except Exception:
            return None

    def _ratio_deviation(ratio: Optional[float]) -> Optional[float]:
        if ratio is None:
            return None
        try:
            return abs(float(ratio) - 1.0)
        except Exception:
            return None

    def _deviation_level(ratio: Optional[float]) -> int:
        """
        Direction-agnostic abnormality level from lesion/reference ratio.
        0: minimal (<10%), 1: mild (>=10%), 2: moderate (>=20%), 3: marked (>=35%)
        """
        dev = _ratio_deviation(ratio)
        if dev is None:
            return 0
        if dev >= 0.35:
            return 3
        if dev >= 0.20:
            return 2
        if dev >= 0.10:
            return 1
        return 0

    def _deviation_desc(label: str, val: Optional[float], ref: Optional[float]) -> str:
        ratio = _ratio(val, ref)
        if ratio is None:
            return f"{label}: not available"
        dev = _ratio_deviation(ratio)
        if dev is None:
            return f"{label}: not available"
        direction = "higher" if ratio > 1.0 else ("lower" if ratio < 1.0 else "equal")
        return (
            f"{label}: {direction} than reference "
            f"(lesion {val:.1f} vs ref {ref:.1f}, ratio {ratio:.2f}, deviation {dev*100:.1f}%)"
        )

    def _signal_desc(
        *,
        label: str,
        val: Optional[float],
        ref: Optional[float],
        low_ratio: float,
        high_ratio: float,
        marked_low: Optional[float] = None,
        marked_high: Optional[float] = None,
        higher_is_worse: bool = False,
    ) -> str:
        ratio = _ratio(val, ref)
        if ratio is None:
            return f"{label}: not available"
        if higher_is_worse:
            if marked_high is not None and ratio >= marked_high:
                status = "markedly high"
            elif ratio >= high_ratio:
                status = "mildly high"
            elif ratio <= low_ratio:
                status = "low"
            else:
                status = "within expected range"
        else:
            if marked_low is not None and ratio <= marked_low:
                status = "markedly low"
            elif ratio <= low_ratio:
                status = "mildly low"
            elif ratio >= high_ratio:
                status = "high"
            else:
                status = "within expected range"
        return f"{label}: {status} (lesion {val:.1f} vs ref {ref:.1f}, ratio {ratio:.2f})"

    def _score_diffusion(
        *,
        lesion_adc: Optional[float],
        ref_adc: Optional[float],
        lesion_dwi: Optional[float],
        ref_dwi: Optional[float],
        size_mm: Optional[float],
        has_lesion: bool,
    ) -> Tuple[int, str]:
        if not has_lesion:
            return 1, "No focal diffusion abnormality identified."
        adc_ratio = _ratio(lesion_adc, ref_adc)
        dwi_ratio = _ratio(lesion_dwi, ref_dwi)
        if adc_ratio is None and dwi_ratio is None:
            return 3, "Diffusion metrics unavailable; score forced to 3."
        adc_lvl = _deviation_level(adc_ratio)
        dwi_lvl = _deviation_level(dwi_ratio)
        max_lvl = max(adc_lvl, dwi_lvl)
        min_lvl = min(adc_lvl, dwi_lvl)
        if max_lvl == 0:
            score = 2
        elif max_lvl >= 3 or (max_lvl >= 2 and min_lvl >= 1):
            score = 4
        else:
            score = 3
        if score == 4 and size_mm is not None and size_mm >= 15.0:
            score = 5
        desc_parts = [
            _deviation_desc("ADC", lesion_adc, ref_adc),
            _deviation_desc("DWI", lesion_dwi, ref_dwi),
            "Scoring uses deviation magnitude (direction-agnostic) due cross-site intensity scaling variability.",
        ]
        return score, "; ".join(desc_parts)

    def _score_t2(
        *,
        lesion_t2: Optional[float],
        ref_t2: Optional[float],
        size_mm: Optional[float],
        has_lesion: bool,
    ) -> Tuple[int, str]:
        if not has_lesion:
            return 1, "No focal T2 abnormality identified."
        t2_ratio = _ratio(lesion_t2, ref_t2)
        if t2_ratio is None:
            return 3, "T2 metrics unavailable; score forced to 3."
        t2_lvl = _deviation_level(t2_ratio)
        if t2_lvl >= 2:
            score = 4
        elif t2_lvl >= 1:
            score = 3
        else:
            score = 2
        if score == 4 and size_mm is not None and size_mm >= 15.0:
            score = 5
        desc = _deviation_desc("T2", lesion_t2, ref_t2) + "; Scoring uses deviation magnitude from zonal reference."
        return score, desc

    def _pirads_category(score: Optional[int]) -> str:
        mapping = {
            1: "Clinically significant cancer is highly unlikely",
            2: "Clinically significant cancer is unlikely",
            3: "Clinically significant cancer is equivocal",
            4: "Clinically significant cancer is likely",
            5: "Clinically significant cancer is highly likely",
        }
        return mapping.get(int(score or 0), "Assessment category not specified")

    def _compute_prostate_dims_cm(mask_path: Optional[str]) -> Optional[Tuple[float, float, float]]:
        if not mask_path:
            return None
        p = Path(str(mask_path)).expanduser().resolve()
        if not p.exists():
            return None
        try:
            import SimpleITK as sitk  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return None
        try:
            img = sitk.ReadImage(str(p))
            arr = sitk.GetArrayViewFromImage(img)
            if int(np.count_nonzero(arr)) == 0:
                return None
            zyx = np.argwhere(arr > 0)
            zmin, ymin, xmin = zyx.min(axis=0).tolist()
            zmax, ymax, xmax = zyx.max(axis=0).tolist()
            spacing = img.GetSpacing()  # x,y,z in mm
            dim_x_mm = (xmax - xmin + 1) * float(spacing[0])
            dim_y_mm = (ymax - ymin + 1) * float(spacing[1])
            dim_z_mm = (zmax - zmin + 1) * float(spacing[2])
            return (dim_x_mm / 10.0, dim_y_mm / 10.0, dim_z_mm / 10.0)
        except Exception:
            return None

    # Prostate dimensions (image axes) from mask if available.
    prostate_dims_cm = _compute_prostate_dims_cm(prostate_mask_path)

    # Lesion size estimation for PI-RADS size thresholding.
    lesion_size_mm: Optional[float] = None
    lesion_size_cm: Optional[float] = None
    lesion_size_note = ""
    if lesion_geo_best and isinstance(lesion_geo_best.get("bbox_mm"), list):
        try:
            bbox_mm = [float(x) for x in lesion_geo_best.get("bbox_mm") or []]
            if bbox_mm:
                lesion_size_mm = max(bbox_mm)
                lesion_size_cm = lesion_size_mm / 10.0
        except Exception:
            lesion_size_mm = None
            lesion_size_cm = None
    if lesion_size_cm is None and lesion_vol_ml is not None:
        try:
            lesion_size_cm = 2.0 * ((3.0 * float(lesion_vol_ml)) / (4.0 * math.pi)) ** (1.0 / 3.0)
            lesion_size_mm = lesion_size_cm * 10.0
            lesion_size_note = "approx (from volume)"
        except Exception:
            lesion_size_mm = None
            lesion_size_cm = None
            lesion_size_note = ""

    if lesion_vol_ml is not None and lesion_vol_ml < float(MIN_CLINICALLY_MEANINGFUL_LESION_CC):
        lesion_vol_ml = None

    has_lesion = bool(
        (lesion_geo_best is not None)
        or (lesion_vol_ml is not None and lesion_vol_ml >= float(MIN_CLINICALLY_MEANINGFUL_LESION_CC))
    )
    adc_available = bool((isinstance(mapping, dict) and mapping.get("ADC")) or ("ADC" in present))
    segmentation_usable = bool(
        seg_ok
        and prostate_mask_path
        and Path(str(prostate_mask_path)).expanduser().exists()
    )
    lesion_geo_ok = bool(isinstance(lesion_geometry, dict) and lesion_geometry.get("ok"))
    lesion_geo_note = ""
    if isinstance(lesion_geometry, dict):
        lesion_geo_note = str(lesion_geometry.get("note") or "").strip().lower()
    lesion_geo_items = (
        lesion_geometry.get("lesions")
        if isinstance(lesion_geometry, dict) and isinstance(lesion_geometry.get("lesions"), list)
        else []
    )
    tiny_only = bool((not has_lesion) and lesion_geo_items)

    # Explicitly separate "tool didn't detect" from "pipeline couldn't assess".
    if has_lesion:
        lesion_tool_status = "detected"
    elif tiny_only:
        lesion_tool_status = "tiny_only"
    elif not adc_available or not segmentation_usable:
        lesion_tool_status = "not_assessable"
    elif lesion_geo_note == "lesion_mask_empty" or (lesion_candidates_n == 0 and lesion_geo_ok):
        lesion_tool_status = "not_detected"
    elif lesion_geo_note in {
        "missing_lesion_or_prostate_mask",
        "lesion_or_prostate_mask_missing_on_disk",
        "prostate_mask_empty",
    }:
        lesion_tool_status = "not_assessable"
    else:
        lesion_tool_status = "indeterminate"

    if has_lesion:
        evidence_tier = "sufficient"
    elif lesion_tool_status == "not_detected" and adc_available and segmentation_usable and feat_ok:
        evidence_tier = "sufficient"
    elif adc_available and (segmentation_usable or feat_ok):
        evidence_tier = "partial"
    else:
        evidence_tier = "limited"

    zone_for_score = "indeterminate"
    if lesion_geo_best:
        zlab = str(lesion_geo_best.get("zone") or "").upper()
        if "PZ" in zlab:
            zone_for_score = "PZ"
        elif "TZ" in zlab or "CG" in zlab:
            zone_for_score = "TZ/CG"

    diff_score, diff_desc = _score_diffusion(
        lesion_adc=lesion_adc_mean,
        ref_adc=ref_adc_mean,
        lesion_dwi=lesion_high_b_mean,
        ref_dwi=ref_high_b_mean,
        size_mm=lesion_size_mm,
        has_lesion=has_lesion,
    )
    t2_score, t2_desc = _score_t2(
        lesion_t2=lesion_t2_mean,
        ref_t2=ref_t2_mean,
        size_mm=lesion_size_mm,
        has_lesion=has_lesion,
    )
    if zone_for_score == "PZ":
        overall_score = diff_score
        overall_basis = "PZ (DWI-dominant)"
    elif zone_for_score == "TZ/CG":
        overall_score = t2_score
        overall_basis = "TZ (T2-dominant)"
        if t2_score == 3 and diff_score >= 4:
            overall_score = 4
            overall_basis = "TZ (T2=3 upgraded by DWI>=4)"
    else:
        overall_score = max(diff_score, t2_score)
        overall_basis = "Zone indeterminate; used higher of DWI/ADC vs T2 scores"
    overall_category = _pirads_category(overall_score)

    def _coerce_pirads(val: Any) -> Optional[int]:
        try:
            v = int(val)
        except Exception:
            return None
        return v if 1 <= v <= 5 else None

    def _collect_vlm_assessments(obj: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(obj, dict):
            return []
        items = obj.get("lesion_assessment")
        if not isinstance(items, list):
            return []
        out: List[Dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            overall = _coerce_pirads(it.get("overall_pirads"))
            tiles = it.get("tile_evidence") if isinstance(it.get("tile_evidence"), list) else []
            has_tiles = any(isinstance(t, str) and ("r" in t and "c" in t) for t in tiles)
            if overall is None or not has_tiles:
                continue
            out.append(dict(it))
        return out

    vlm_overall_score: Optional[int] = None
    vlm_overall_category: Optional[str] = None

    def _brain_quant_table(rows: List[Dict[str, str]]) -> Optional[str]:
        if not rows:
            return None
        roi_order = [
            ("Tumor_Core", "Tumor Core"),
            ("Whole_Tumor", "Whole Tumor"),
            ("Enhancing_Tumor", "Enhancing Tumor"),
        ]

        def _pick(roi_key: str, seq_hint: Optional[str], field: str) -> Optional[float]:
            for r in rows:
                roi = str(r.get("roi") or "").strip()
                src = str(r.get("roi_source") or "").strip()
                seq = str(r.get("sequence") or "").strip().lower()
                if roi.lower() != roi_key.lower() and src.lower() != roi_key.lower():
                    continue
                if seq_hint and seq_hint.lower() not in seq:
                    continue
                val = r.get(field)
                out = _safe_float(val)
                if out is not None:
                    return out
            return None

        table_lines = [
            "| ROI | Volume (cm³) | Mean Intensity (T1c) |",
            "| :--- | ---: | ---: |",
        ]
        any_row = False
        for roi_key, label in roi_order:
            vol = _pick(roi_key, "t1c", "volume_ml") or _pick(roi_key, None, "volume_ml")
            mean_t1c = _pick(roi_key, "t1c", "mean") or _pick(roi_key, "t1c", "mean_intensity")
            vol_str = f"{vol:.1f}" if vol is not None else "N/A"
            mean_str = f"{mean_t1c:.1f}" if mean_t1c is not None else "N/A"
            table_lines.append(f"| {label} | {vol_str} | {mean_str} |")
            any_row = True
        return "\n".join(table_lines) if any_row else None

    brain_quant_table_md = _brain_quant_table(main_rows)

    norm_struct: Optional[Dict[str, Any]] = None
    selected_images_for_rules: List[Dict[str, Any]] = []
    structured_path = out_dir / "structured_report.json"
    if structured_path.exists():
        try:
            obj = json.loads(structured_path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                try:
                    sel_path = vlm_dir / "vlm_selected_images.json"
                    if sel_path.exists():
                        sel_obj = json.loads(sel_path.read_text(encoding="utf-8"))
                        if isinstance(sel_obj, list):
                            selected_images_for_rules = [it for it in sel_obj if isinstance(it, dict)]
                except Exception:
                    selected_images_for_rules = []
                norm_struct = _normalize_llm_structured(obj, evidence_bundle=evidence_bundle)
                norm_struct = _apply_structured_report_rules(
                    norm_struct, evidence_bundle=evidence_bundle, selected_images=selected_images_for_rules, run_dir=run_dir
                )
        except Exception:
            norm_struct = None

    vlm_assessments = _collect_vlm_assessments(norm_struct)
    vlm_report_overall = _coerce_pirads(norm_struct.get("overall_pirads")) if isinstance(norm_struct, dict) else None
    if vlm_assessments:
        scores = [_coerce_pirads(it.get("overall_pirads")) for it in vlm_assessments]
        scores = [s for s in scores if s is not None]
        if scores:
            vlm_overall_score = max(scores)
            vlm_overall_category = _pirads_category(vlm_overall_score)
    if vlm_report_overall is not None:
        if vlm_overall_score is None:
            vlm_overall_score = vlm_report_overall
            vlm_overall_category = _pirads_category(vlm_overall_score)
        else:
            vlm_overall_score = max(vlm_overall_score, vlm_report_overall)
            vlm_overall_category = _pirads_category(vlm_overall_score)

    rule_overall_score = overall_score
    rule_overall_category = overall_category
    final_overall_score: Optional[int] = None
    final_overall_category = "Indeterminate"
    final_overall_range: Optional[str] = None
    final_assessment_note: Optional[str] = None

    tool_overall_floor: Optional[int] = None
    if has_lesion:
        lesion_vol_for_floor = _safe_float(lesion_vol_ml)
        if lesion_vol_for_floor is None and lesion_geo_best is not None:
            lesion_vol_for_floor = _safe_float(lesion_geo_best.get("volume_cc"))
        p = _safe_float(lesion_max_prob)
        s = _safe_float(lesion_size_mm)
        if (
            p is not None
            and p >= 0.90
            and ((lesion_vol_for_floor is not None and lesion_vol_for_floor >= 2.0) or (s is not None and s >= 15.0))
        ):
            tool_overall_floor = 5
        elif (
            (
                p is not None
                and p >= 0.70
                and ((lesion_vol_for_floor is not None and lesion_vol_for_floor >= 1.0) or (s is not None and s >= 10.0))
            )
            or (
                lesion_vol_for_floor is not None
                and lesion_vol_for_floor >= 2.0
                and s is not None
                and s >= 12.0
            )
        ):
            tool_overall_floor = 4
        else:
            tool_overall_floor = 3

    if has_lesion:
        candidates = [int(rule_overall_score)]
        if vlm_overall_score is not None:
            candidates.append(int(vlm_overall_score))
        if tool_overall_floor is not None:
            candidates.append(int(tool_overall_floor))
        final_overall_score = max(candidates) if candidates else int(rule_overall_score)
        final_overall_category = _pirads_category(final_overall_score)
        if tool_overall_floor is not None and final_overall_score == tool_overall_floor and tool_overall_floor > int(rule_overall_score):
            final_assessment_note = (
                f"PI-RADS floor applied from tool evidence "
                f"(max_prob={lesion_max_prob if lesion_max_prob is not None else 'NA'}, "
                f"volume_cc={lesion_vol_ml if lesion_vol_ml is not None else 'NA'}, "
                f"size_mm={lesion_size_mm if lesion_size_mm is not None else 'NA'})."
            )
    else:
        if vlm_overall_score is not None:
            final_overall_score = vlm_overall_score
            final_overall_category = _pirads_category(vlm_overall_score)
        elif evidence_tier == "sufficient" and lesion_tool_status == "not_detected":
            # No hard force to PI-RADS 1: treat as low-suspicion range.
            final_overall_score = 2
            final_overall_category = _pirads_category(2)
            final_overall_range = "1-2"
            final_assessment_note = "Tooling did not detect a lesion under sufficient inputs; low suspicion but not absolute exclusion."
        else:
            # Evidence-limited branch: keep indeterminate instead of hard-coding PI-RADS 1.
            final_overall_score = None
            final_overall_category = "Indeterminate due to insufficient lesion evidence"
            final_overall_range = "1-3"
            if lesion_tool_status == "not_assessable":
                blockers: List[str] = []
                if not adc_available:
                    blockers.append("ADC unavailable")
                if not segmentation_usable:
                    blockers.append("segmentation unavailable")
                if lesion_geo_note in {
                    "missing_lesion_or_prostate_mask",
                    "lesion_or_prostate_mask_missing_on_disk",
                    "prostate_mask_empty",
                }:
                    blockers.append("lesion candidate geometry unavailable")
                if not feat_ok:
                    blockers.append("ROI features unavailable")
                if blockers:
                    detail = "; ".join(blockers)
                    if (not adc_available) or (not segmentation_usable):
                        final_assessment_note = f"Pipeline could not reliably assess lesions ({detail})."
                    else:
                        final_assessment_note = f"No robust lesion-level evidence was available for confident PI-RADS assignment ({detail})."
                else:
                    final_assessment_note = "No robust lesion-level evidence was available for confident PI-RADS assignment."
            elif lesion_tool_status == "tiny_only":
                final_assessment_note = "Only tiny lesion candidates were detected; clinically meaningful PI-RADS assignment is uncertain."
            else:
                final_assessment_note = "No robust lesion-level evidence was available for confident PI-RADS assignment."

    def _final_pirads_text() -> str:
        if final_overall_score is not None:
            return f"{final_overall_score}/5 ({final_overall_category})"
        if final_overall_range:
            return f"indeterminate (suggested PI-RADS {final_overall_range})"
        return "indeterminate"

    def _final_pirads_impression_text() -> str:
        if final_overall_score is not None:
            return f"PI-RADS {final_overall_score}"
        if final_overall_range:
            return f"PI-RADS indeterminate (suggested {final_overall_range})"
        return "PI-RADS indeterminate"

    clinical_lines: List[str] = []
    impression_meta: Dict[str, Any] = {"source": "template", "requested": "template"}
    cardiac_template_impression: List[str] = []
    brain_mode = domain_cfg.name == "brain"
    cardiac_mode = domain_cfg.name == "cardiac"
    if not brain_mode:
        # If tumor ROI volumes exist or brain mapping present, treat as brain report even if domain missing.
        if any(v is not None for v in (brain_wt_vol_ml, brain_tc_vol_ml, brain_et_vol_ml)):
            brain_mode = True
        elif isinstance(mapping, dict) and any(mapping.get(k) for k in ("FLAIR", "T1c", "T1")):
            brain_mode = True

    if cardiac_mode:
        indication = study_desc or "Cardiac cine MRI evaluation"
        tech_bits: List[str] = []
        if manufacturer or field_strength or model:
            tech_bits.append(
                " ".join([x for x in [manufacturer, f"{field_strength}T" if field_strength else "", model] if x])
            )
        if seq_descs:
            tech_bits.append("Sequences: " + "; ".join(seq_descs))
        tech_text = " ".join(tech_bits) if tech_bits else "Not provided."
        seg_info = evidence_bundle.get("segmentation", {}) if isinstance(evidence_bundle, dict) else {}
        cls_info = evidence_bundle.get("cardiac_classification", {}) if isinstance(evidence_bundle, dict) else {}
        cardiac_seg = seg_info.get("cardiac_seg_path") if isinstance(seg_info, dict) else None
        rv_mask = seg_info.get("rv_mask_path") if isinstance(seg_info, dict) else None
        myo_mask = seg_info.get("myo_mask_path") if isinstance(seg_info, dict) else None
        lv_mask = seg_info.get("lv_mask_path") if isinstance(seg_info, dict) else None
        predicted_group = cls_info.get("predicted_group") if isinstance(cls_info, dict) else None
        needs_vlm_review = bool(cls_info.get("needs_vlm_review")) if isinstance(cls_info, dict) else False
        cls_metrics = cls_info.get("metrics") if isinstance(cls_info.get("metrics"), dict) else {}
        features_main = evidence_bundle.get("features_main", {}) if isinstance(evidence_bundle, dict) else {}
        feat_ok = bool(features_main.get("ok")) if isinstance(features_main, dict) else False
        feat_table = features_main.get("feature_table_path") if isinstance(features_main, dict) else None
        t1_analysis = evidence_bundle.get("cardiac_t1_feature_analysis") if isinstance(evidence_bundle, dict) else None
        card_case_results = card_seg.get("case_results") if isinstance(card_seg, dict) else None
        n_segmented_frames = len(card_case_results) if isinstance(card_case_results, list) and card_case_results else None
        vlm_findings = ""
        vlm_impression = ""
        if isinstance(norm_struct, dict):
            narrative = norm_struct.get("narrative") if isinstance(norm_struct.get("narrative"), dict) else {}
            vlm_findings = str(narrative.get("findings") or "").strip()
            vlm_impression = str(narrative.get("impression") or "").strip()

        clinical_lines.append("# MRI CARDIAC CINE REPORT")
        clinical_lines.append("")
        clinical_lines.append("**CLINICAL INDICATION:**")
        clinical_lines.append(str(indication))
        clinical_lines.append("")
        clinical_lines.append("**TECHNICAL FACTORS:**")
        clinical_lines.append(str(tech_text))
        clinical_lines.append("")
        recon_lines = _reconstruction_lines(recon_provenance)
        if recon_lines:
            clinical_lines.append("**ACQUISITION AND RECONSTRUCTION:**")
            clinical_lines.extend(recon_lines)
            clinical_lines.append("")
        clinical_lines.append("**FINDINGS:**")
        # Measurements only.  Artifact paths live under PROVENANCE / ARTIFACTS.
        cardiac_findings = _cardiac_findings_lines(
            cls_info=cls_info if isinstance(cls_info, dict) else {},
            seg_info=seg_info if isinstance(seg_info, dict) else {},
            recon_prov=recon_provenance,
            n_segmented_frames=n_segmented_frames,
        )
        clinical_lines.extend(cardiac_findings)
        if isinstance(t1_analysis, dict) and bool(t1_analysis.get("ok")):
            qc = t1_analysis.get("qc") if isinstance(t1_analysis.get("qc"), dict) else {}
            gate_pass = bool(qc.get("gate_pass")) if isinstance(qc, dict) else False
            gate_reasons = qc.get("gate_fail_reasons") if isinstance(qc, dict) else []
            clinical_lines.append(f"- T1 evidence QC gate: {'PASS' if gate_pass else 'FAIL'}")
            if isinstance(gate_reasons, list) and gate_reasons:
                clinical_lines.append("- T1 QC issues: " + "; ".join(str(x) for x in gate_reasons))
            key_metrics = t1_analysis.get("key_metrics") if isinstance(t1_analysis.get("key_metrics"), list) else []
            if key_metrics:
                clinical_lines.append("- T1 key feature primitives (max 6):")
                for m in key_metrics[:6]:
                    if not isinstance(m, dict):
                        continue
                    name = str(m.get("name") or "")
                    val = m.get("value")
                    unit = str(m.get("unit") or "")
                    roi = str(m.get("roi") or "")
                    if isinstance(val, (int, float)):
                        vtxt = f"{float(val):.3f}" if unit == "" else f"{float(val):.2f}"
                    else:
                        vtxt = str(val)
                    unit_txt = f" {unit}".rstrip()
                    clinical_lines.append(f"- {name} ({roi}): {vtxt}{unit_txt}")
            topk = t1_analysis.get("top_k_candidates") if isinstance(t1_analysis.get("top_k_candidates"), list) else []
            if topk:
                clinical_lines.append("- T1 evidence-first disease candidates (Top-K, non-diagnostic):")
                for c in topk[:3]:
                    if not isinstance(c, dict):
                        continue
                    rank = c.get("rank")
                    label = str(c.get("label") or "unknown")
                    score = _safe_float(c.get("score"))
                    score_txt = f"{score:.3f}" if score is not None else "NA"
                    clinical_lines.append(f"- Top-{rank}: {label} (score={score_txt})")
                    support = c.get("support_evidence") if isinstance(c.get("support_evidence"), list) else []
                    if support:
                        for ev in support[:2]:
                            if not isinstance(ev, dict):
                                continue
                            clinical_lines.append(
                                f"-   support: {ev.get('metric')}={ev.get('value')} ({ev.get('note')})"
                            )
                    conflict = c.get("conflict_evidence") if isinstance(c.get("conflict_evidence"), list) else []
                    if conflict:
                        for ev in conflict[:2]:
                            if not isinstance(ev, dict):
                                continue
                            clinical_lines.append(
                                f"-   conflict: {ev.get('metric')}={ev.get('value')} ({ev.get('note')})"
                            )
            uncertainty = t1_analysis.get("uncertainty") if isinstance(t1_analysis.get("uncertainty"), list) else []
            if uncertainty:
                clinical_lines.append("- Uncertainty: " + " | ".join(str(x) for x in uncertainty))
            nxt = t1_analysis.get("next_steps") if isinstance(t1_analysis.get("next_steps"), list) else []
            if nxt:
                clinical_lines.append("- Next-step suggestions: " + " | ".join(str(x) for x in nxt[:3]))
        if vlm_findings:
            clinical_lines.append("- VLM interpretation: " + vlm_findings)
        clinical_lines.append("")

        # --- Deterministic impression (always computed; the baseline) ---------
        template_impression: List[str] = []
        n = 0

        def _imp(text: str) -> None:
            nonlocal n
            n += 1
            template_impression.append(f"{n}. {text}")

        _lv_ef_s = _fmt_num(cls_metrics.get("lv_ef_percent"))
        _rv_ef_s = _fmt_num(cls_metrics.get("rv_ef_percent"))
        _lv_edv_s = _fmt_num(cls_metrics.get("lv_edv_ml"))
        _rv_edv_s = _fmt_num(cls_metrics.get("rv_edv_ml"))
        _lv_mass_s = _fmt_num(cls_metrics.get("lv_mass_g"))
        if _lv_ef_s or _rv_ef_s:
            fn_bits = [b for b in (
                f"LV EF {_lv_ef_s}%" if _lv_ef_s else None,
                f"RV EF {_rv_ef_s}%" if _rv_ef_s else None,
            ) if b]
            _imp("Biventricular systolic function measured from the cine segmentation: " + ", ".join(fn_bits) + ".")
        if _lv_edv_s or _rv_edv_s or _lv_mass_s:
            size_bits = [b for b in (
                f"LV EDV {_lv_edv_s} mL" if _lv_edv_s else None,
                f"RV EDV {_rv_edv_s} mL" if _rv_edv_s else None,
                f"LV mass {_lv_mass_s} g" if _lv_mass_s else None,
            ) if b]
            _imp("Chamber size and mass: " + ", ".join(size_bits) + ".")
        if predicted_group:
            _rule_txt = ""
            _rt = cls_info.get("rule_trace") if isinstance(cls_info.get("rule_trace"), list) else []
            if _rt and str(_rt[0] or "").strip():
                _rule_txt = f" ({str(_rt[0]).strip().rstrip('.')})"
            _imp(f"Rule-based classifier assigned group {predicted_group}{_rule_txt}. This is a rule output, not a diagnosis.")
        if isinstance(t1_analysis, dict) and bool(t1_analysis.get("ok")):
            qc = t1_analysis.get("qc") if isinstance(t1_analysis.get("qc"), dict) else {}
            if not bool(qc.get("gate_pass")):
                _imp("T1 feature QC gate failed; T1 disease inference is withheld until mask/alignment issues are fixed.")
            else:
                topk = t1_analysis.get("top_k_candidates") if isinstance(t1_analysis.get("top_k_candidates"), list) else []
                if topk and isinstance(topk[0], dict):
                    _imp(f"T1 evidence-first Top-1 candidate: {topk[0].get('label')} (score={topk[0].get('score')}).")
        if needs_vlm_review:
            _imp("Classification is borderline/indeterminate by rules; VLM-assisted review is suggested.")
        if vlm_impression:
            _imp(f"VLM summary: {vlm_impression}")
        _us = recon_provenance.get("undersample") if isinstance(recon_provenance, dict) else None
        if isinstance(_us, dict) and _us.get("applied") and _us.get("retrospective"):
            _fac = _us.get("factor")
            _imp(
                "All measurements derive from images reconstructed after RETROSPECTIVE "
                + (f"R={int(_fac)} " if _fac is not None else "")
                + "undersampling of fully sampled k-space; the acquisition was not prospectively accelerated."
            )
        if not template_impression:
            if isinstance(cardiac_seg, str) and cardiac_seg:
                _imp("Cardiac cine segmentation completed, but no ventricular measurements were produced; no impression can be stated.")
            else:
                _imp("No cardiac cine segmentation or measurements are available; no impression can be stated.")

        # --- Optional LLM impression (default OFF) ----------------------------
        # Enabled only by an explicit `impression_mode="llm"`.  Any transport
        # failure, empty completion, ungrounded number, or unsupported clinical
        # entity falls back to the deterministic template above.
        impression_mode = str(args.get("impression_mode") or "template").strip().lower()
        impression_meta = {"source": "template", "requested": impression_mode}
        cardiac_template_impression = list(template_impression)
        final_impression = list(template_impression)
        if impression_mode == "llm":
            _measures = _cardiac_measurements(cls_info if isinstance(cls_info, dict) else {}, recon_provenance)
            _m_lines: List[str] = []
            _allowed: List[float] = []
            for m in _measures:
                _v = f"{float(m['value']):.{int(m['digits'])}f}"
                _unit = f" {m['unit']}" if m["unit"] else ""
                _m_lines.append(f"- {m['label']}: {_v}{_unit}")
                _allowed.append(float(m["value"]))
            if predicted_group:
                _m_lines.append(f"- Rule-based classification group: {predicted_group}")
            _rt = cls_info.get("rule_trace") if isinstance(cls_info.get("rule_trace"), list) else []
            for _r in _rt[:2]:
                if str(_r or "").strip():
                    _m_lines.append(f"- Classifier rule text: {str(_r).strip()}")
            _base = str(
                args.get("impression_llm_base_url")
                or args.get("server_base_url")
                or "http://127.0.0.1:11434"
            )
            _model = str(args.get("impression_llm_model") or args.get("server_model") or "qwen3.6:27b")
            _timeout = int(args.get("impression_llm_timeout_s") or 60)
            impression_meta.update({"base_url": _base, "model": _model})
            _res = _build_llm_impression(
                base_url=_base,
                model=_model,
                timeout_s=_timeout,
                measurement_lines=_m_lines,
                allowed_numbers=_allowed,
            )
            if _res.get("ok"):
                impression_meta["source"] = "llm"
                final_impression = [str(_res["text"]).strip()]
                _us_note = ""
                if isinstance(_us, dict) and _us.get("applied") and _us.get("retrospective"):
                    _fac = _us.get("factor")
                    _us_note = (
                        " Measurements derive from images reconstructed after RETROSPECTIVE "
                        + (f"R={int(_fac)} " if _fac is not None else "")
                        + "undersampling of fully sampled k-space."
                    )
                final_impression.append(
                    f"(Impression drafted by `{_model}` from the measurements above and "
                    f"checked against them; the deterministic template is retained in report.json.{_us_note})"
                )
            else:
                impression_meta["fallback_reason"] = _res.get("reason")

        clinical_lines.append("**IMPRESSION:**")
        clinical_lines.extend(final_impression)

        # --- Provenance / artifacts (paths belong here, not in FINDINGS) ------
        prov_lines: List[str] = []
        if isinstance(cardiac_seg, str) and cardiac_seg:
            prov_lines.append(f"- Cardiac cine segmentation: `{cardiac_seg}`")
        else:
            prov_lines.append("- Cardiac cine segmentation: not produced.")
        if isinstance(rv_mask, str) and rv_mask:
            prov_lines.append(f"- RV mask: `{rv_mask}`")
        if isinstance(myo_mask, str) and myo_mask:
            prov_lines.append(f"- Myocardium mask: `{myo_mask}`")
        if isinstance(lv_mask, str) and lv_mask:
            prov_lines.append(f"- LV mask: `{lv_mask}`")
        if isinstance(cls_info, dict) and cls_info.get("classification_path"):
            prov_lines.append(f"- Classification JSON: `{cls_info.get('classification_path')}`")
        if feat_ok and isinstance(feat_table, str) and feat_table:
            prov_lines.append(f"- ROI feature table: `{feat_table}`")
        if isinstance(recon_provenance, dict) and recon_provenance.get("ok"):
            if recon_provenance.get("source_h5_path"):
                prov_lines.append(f"- Source k-space file: `{recon_provenance['source_h5_path']}`")
            if recon_provenance.get("reconstructed_nifti"):
                prov_lines.append(
                    f"- Reconstructed image: `{to_short_path(str(recon_provenance['reconstructed_nifti']), run_dir=run_dir)}`"
                )
            if recon_provenance.get("zerofilled_nifti"):
                prov_lines.append(
                    f"- Zero-filled reference image: `{to_short_path(str(recon_provenance['zerofilled_nifti']), run_dir=run_dir)}`"
                )
        if prov_lines:
            clinical_lines.append("")
            clinical_lines.append("**PROVENANCE / ARTIFACTS:**")
            clinical_lines.extend(prov_lines)
        clinical_lines.append("")
        clinical_lines.append("**LIMITATIONS:**")
        clinical_lines.append(
            "- Engineering demo output. Volumes, ejection fractions, mass and wall thickness are derived "
            "from an automatic segmentation and a rule-based classifier; none of it is clinically validated."
        )
    elif not brain_mode:
        indication = study_desc or "Not provided"
        comparison = "None."
        tech_bits: List[str] = []
        if manufacturer or field_strength or model:
            tech_bits.append(
                " ".join([x for x in [manufacturer, f"{field_strength}T" if field_strength else "", model] if x])
            )
        if seq_descs:
            tech_bits.append("Sequences: " + "; ".join(seq_descs))
        if dwi_bvals:
            tech_bits.append("DWI b-values: " + ", ".join(dwi_bvals))
        if slice_thickness:
            tech_bits.append(f"T2w slice thickness: {slice_thickness} mm")
        tech_text = " ".join(tech_bits) if tech_bits else "Not provided."

        imaging_med = "Not provided."

        dims_str = "Not calculated"
        if prostate_dims_cm:
            dims_str = f"{prostate_dims_cm[0]:.2f} x {prostate_dims_cm[1]:.2f} x {prostate_dims_cm[2]:.2f} cm (image axes)"
        vol_str = f"{prostate_vol_ml:.1f} cc" if prostate_vol_ml is not None else "Not calculated"
        psa_density_str = "Not calculated"

        quality_bits: List[str] = []
        # --- Track 2: degradation context advisory ---
        if degradation_context:
            _missing_mods = degradation_context.get("missing_modalities") or []
            _omitted_tools = degradation_context.get("omitted_tools") or []
            if _missing_mods:
                quality_bits.append(f"LIMITED EXAM — missing modalities: {', '.join(_missing_mods)}")
            if _omitted_tools:
                quality_bits.append(f"Omitted pipeline steps: {', '.join(_omitted_tools)}")
        if not seg_ok:
            quality_bits.append("Segmentation unavailable")
        if not feat_ok:
            quality_bits.append("ROI features unavailable")
        if not adc_available:
            quality_bits.append("ADC missing; lesion quantification fallback to qualitative T2w/DWI review")
        if not segmentation_usable:
            quality_bits.append("Prostate mask unavailable/invalid for lesion-candidate validation")
        if not dwi_bvals:
            quality_bits.append("DWI b-values not provided; diffusion interpretation limited")
        quality_bits.append(f"Evidence tier: {evidence_tier}")
        quality_bits.append(f"Lesion tool status: {lesion_tool_status}")
        if final_assessment_note:
            quality_bits.append(final_assessment_note)
        if lesion_candidates_n is not None:
            if lesion_candidates_n == 0:
                thresh_note = f" at threshold {lesion_threshold}" if lesion_threshold is not None else ""
                quality_bits.append(f"Tool-generated lesion candidates: 0 candidates{thresh_note} (candidates.json)")
            else:
                quality_bits.append(f"Tool-generated lesion candidates: {lesion_candidates_n} (research use only)")
                if not has_lesion:
                    quality_bits.append(
                        f"All candidates are below clinical reporting floor ({MIN_CLINICALLY_MEANINGFUL_LESION_CC:.1f} cc)"
                    )
        if isinstance(norm_struct, dict):
            iq = norm_struct.get("image_qc") if isinstance(norm_struct.get("image_qc"), dict) else {}
            reviewed = iq.get("images_reviewed") if isinstance(iq.get("images_reviewed"), list) else []
            if reviewed:
                quality_bits.append(
                    f"VLM QC: roi_coverage_ok={bool(iq.get('roi_coverage_ok'))}, "
                    f"registration_alignment_ok={bool(iq.get('registration_alignment_ok'))}"
                )
        quality_str = "; ".join(quality_bits) if quality_bits else "Diagnostic."

        clinical_lines.append("# MRI PROSTATE RADIOLOGY REPORT")
        clinical_lines.append("")
        clinical_lines.append("**CLINICAL INDICATION:**")
        clinical_lines.append(indication)
        clinical_lines.append("")
        clinical_lines.append("**COMPARISON:**")
        clinical_lines.append(comparison)
        clinical_lines.append("")
        clinical_lines.append("**TECHNICAL FACTORS:**")
        clinical_lines.append(tech_text)
        clinical_lines.append("")
        clinical_lines.append("**IMAGING MEDICATION:**")
        clinical_lines.append(imaging_med)
        clinical_lines.append("")
        clinical_lines.append("---")
        clinical_lines.append("")
        clinical_lines.append("**FINDINGS:**")
        clinical_lines.append("")
        clinical_lines.append("**General Prostate:**")
        clinical_lines.append(f"* **Dimensions:** {dims_str}")
        clinical_lines.append(f"* **Volume:** {vol_str}")
        clinical_lines.append(f"* **PSA Density:** {psa_density_str}")
        clinical_lines.append(f"* **Quality/Limitations:** {quality_str}")
        clinical_lines.append("")
        clinical_lines.append("**Transitional Zone (TZ):**")
        if zone_for_score == "TZ/CG" and has_lesion:
            clinical_lines.append("Dominant lesion candidate in TZ/CG; see Lesion section below.")
        else:
            clinical_lines.append("No focal TZ lesion candidate identified by pipeline.")
        clinical_lines.append("")
        clinical_lines.append("**Peripheral Zone (PZ):**")
        if zone_for_score == "PZ" and has_lesion:
            clinical_lines.append("Dominant lesion candidate in PZ; see Lesion section below.")
        else:
            clinical_lines.append("No focal PZ lesion candidate identified by pipeline.")
        clinical_lines.append("")
        clinical_lines.append("**Focal Lesion Evaluation (PI-RADS v2):**")
        if has_lesion:
            zone = lesion_geo_best.get("zone") if lesion_geo_best else None
            quad = lesion_geo_best.get("axial_quadrant_image") if lesion_geo_best else None
            clock = lesion_geo_best.get("clock_face_image") if lesion_geo_best else None
            seg_part = lesion_geo_best.get("cc_segment") if lesion_geo_best else None
            z_pct = lesion_geo_best.get("z_percent_from_inferior") if lesion_geo_best else None
            loc_parts = [p for p in [zone, seg_part, clock, quad] if p]
            if isinstance(z_pct, (int, float)):
                loc_parts.append(f"z~{float(z_pct):.1f}%")
            location_str = "; ".join([str(p) for p in loc_parts]) if loc_parts else "Not provided"

            size_parts: List[str] = []
            if lesion_size_cm is not None:
                size_core = f"{lesion_size_cm:.2f} cm"
                if lesion_size_note:
                    size_core += f" ({lesion_size_note})"
                size_parts.append(size_core)
            if lesion_geo_best and lesion_geo_best.get("volume_cc") is not None:
                size_parts.append(f"{float(lesion_geo_best.get('volume_cc') or 0.0):.2f} cc")
            elif lesion_vol_ml is not None:
                size_parts.append(f"{lesion_vol_ml:.2f} cc")
            size_str = " / ".join(size_parts) if size_parts else "Not available"

            margin_str = "Not characterized"
            if lesion_geo_best and isinstance(lesion_geo_best.get("capsule_contact"), bool):
                margin_str = "Abuts capsule" if lesion_geo_best.get("capsule_contact") else "No capsule contact on mask"

            clinical_lines.append("> **Lesion #1**")
            clinical_lines.append(f"> * **Location:** {location_str}")
            clinical_lines.append(f"> * **Size:** {size_str}")
            clinical_lines.append(f"> * **Margin:** {margin_str}")
            vlm_first = vlm_assessments[0] if vlm_assessments else {}
            vlm_t2 = _coerce_pirads(vlm_first.get("t2_score")) if isinstance(vlm_first, dict) else None
            vlm_dwi = _coerce_pirads(vlm_first.get("dwi_score")) if isinstance(vlm_first, dict) else None
            t2_note = f" (VLM {vlm_t2}/5)" if vlm_t2 is not None else ""
            dwi_note = f" (VLM {vlm_dwi}/5)" if vlm_dwi is not None else ""
            clinical_lines.append(f"> * **T2 Characteristics:** {t2_desc}; **Score:** {t2_score}/5{t2_note}")
            clinical_lines.append(f"> * **DWI/ADC Characteristics:** {diff_desc}; **Score:** {diff_score}/5{dwi_note}")
            clinical_lines.append("> * **Perfusion (DCE):** Not available")
            overall_bits: List[str] = []
            if zone_for_score == "indeterminate":
                overall_bits.append(overall_basis)
            if vlm_overall_score is not None and vlm_overall_score != rule_overall_score:
                overall_bits.append(f"VLM override; rule-based {rule_overall_score}/5")
            overall_extra = f"; {'; '.join(overall_bits)}" if overall_bits else ""
            clinical_lines.append(
                f"> * **Overall PI-RADS Score:** {_final_pirads_text()}{overall_extra}"
            )
        else:
            clinical_lines.append(
                f"No suspicious lesion identified. Overall PI-RADS Score: {_final_pirads_text()}."
            )
        clinical_lines.append("")
        clinical_lines.append("**VLM Interpretation:**")
        if isinstance(norm_struct, dict):
            narrative = norm_struct.get("narrative") if isinstance(norm_struct.get("narrative"), dict) else {}
            findings_txt = str(narrative.get("findings") or "").strip()
            impression_txt = str(narrative.get("impression") or "").strip()
            if findings_txt:
                clinical_lines.append(f"- Findings: {findings_txt}")
            if impression_txt:
                clinical_lines.append(f"- Impression: {impression_txt}")
            if vlm_assessments:
                for it in vlm_assessments:
                    if not isinstance(it, dict):
                        continue
                    pid = str(it.get("lesion_id") or "Lesion")
                    loc = str(it.get("location") or "")
                    score = _coerce_pirads(it.get("overall_pirads"))
                    rationale = str(it.get("rationale") or "").strip()
                    tiles = it.get("tile_evidence") if isinstance(it.get("tile_evidence"), list) else []
                    tile_txt = ", ".join([str(t) for t in tiles[:4]]) if tiles else "Not provided"
                    parts = [pid]
                    if loc:
                        parts.append(loc)
                    if score is not None:
                        parts.append(f"PI-RADS {score}")
                    if rationale:
                        parts.append(rationale)
                    clinical_lines.append(f"- VLM lesion assessment: " + "; ".join(parts) + f". Tiles: {tile_txt}")
            if not findings_txt and not impression_txt and not vlm_assessments:
                clinical_lines.append("- Not available.")
        else:
            clinical_lines.append("- Not available.")
        clinical_lines.append("")
        clinical_lines.append("**Extra-Prostatic Findings:**")
        clinical_lines.append("* **Seminal Vesicles:** Not provided")
        clinical_lines.append("* **Urinary Bladder:** Not provided")
        clinical_lines.append("* **Pelvic Lymph Nodes:** Not provided")
        clinical_lines.append("* **Pelvic Bowel:** Not provided")
        clinical_lines.append("* **Pelvic Vasculature:** Not provided")
        clinical_lines.append("* **Bones:** Not provided")
        clinical_lines.append("* **Abdominal/Pelvic Wall:** Not provided")
        clinical_lines.append("")
        clinical_lines.append("---")
        clinical_lines.append("")
        clinical_lines.append("**IMPRESSION:**")
        if has_lesion:
            zone = lesion_geo_best.get("zone") if lesion_geo_best else "zone indeterminate"
            quad = lesion_geo_best.get("axial_quadrant_image") if lesion_geo_best else "location indeterminate"
            clinical_lines.append(
                f"1. Dominant lesion candidate in {zone} ({quad}) - **{_final_pirads_impression_text()}**."
            )
        else:
            if final_assessment_note:
                clinical_lines.append(f"1. No suspicious lesion identified by tool pipeline - **{_final_pirads_impression_text()}**.")
                clinical_lines.append(f"2. {final_assessment_note}")
                clinical_lines.append("3. No additional extraprostatic findings reported by pipeline.")
            else:
                clinical_lines.append(f"1. No suspicious lesion identified - **{_final_pirads_impression_text()}**.")
                clinical_lines.append("2. No additional extraprostatic findings reported by pipeline.")
        if has_lesion:
            clinical_lines.append("2. No additional extraprostatic findings reported by pipeline.")
        clinical_lines.append("")
        clinical_lines.append("*Standardized reporting guidelines follow recommendations by ACR-ESUR PIRADS v2.*")
    else:
        clinical_lines.append("## CLINICAL INDICATION")
        clinical_lines.append("Brain MRI (Tumor Protocol)")
        clinical_lines.append("")
        clinical_lines.append("## TECHNICAL FACTORS")
        if seq_descs:
            clinical_lines.append("Sequences: " + "; ".join(seq_descs))
        else:
            clinical_lines.append("Not provided.")
        clinical_lines.append("")
        clinical_lines.append("## FINDINGS")
        clinical_lines.append("**Quantitative Metrics:**")
        if brain_quant_table_md:
            clinical_lines.extend(brain_quant_table_md.splitlines())
        else:
            clinical_lines.append("_Not available._")
        clinical_lines.append("")
        clinical_lines.append("**Qualitative Assessment:**")
        qual_lines: List[str] = []
        if brain_seg_ok and isinstance(brain_seg, dict):
            qual_lines.append("- Tumor segmentation masks generated (TC/WT/ET).")
        else:
            qual_lines.append("- Tumor segmentation masks not available.")
        bsc = norm_struct.get("brain_signal_characterization") if isinstance(norm_struct, dict) else {}
        if isinstance(bsc, dict):
            def _bsc_val(*keys: str) -> Optional[str]:
                for k in keys:
                    v = str(bsc.get(k) or "").strip()
                    if v:
                        return v
                return None

            t1c = _bsc_val("t1c_enhancement", "T1c")
            flair = _bsc_val("flair_edema", "FLAIR")
            t1 = _bsc_val("t1_necrosis", "T1")
            t2 = _bsc_val("t2_extent", "T2")
            if t1c:
                qual_lines.append(f"- **T1c enhancement:** {t1c}")
            if flair:
                qual_lines.append(f"- **FLAIR edema:** {flair}")
            if t1:
                qual_lines.append(f"- **T1 necrosis:** {t1}")
            if t2:
                qual_lines.append(f"- **T2 extent:** {t2}")

        if isinstance(norm_struct, dict):
            narrative = norm_struct.get("narrative") if isinstance(norm_struct.get("narrative"), dict) else {}
            findings_txt = str(narrative.get("findings") or "").strip()
            if findings_txt:
                for ln in findings_txt.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    if ln.startswith(("-", "*")):
                        qual_lines.append(ln)
                    else:
                        qual_lines.append(f"- {ln}")
        if qual_lines:
            clinical_lines.extend(qual_lines)
        else:
            clinical_lines.append("_Not available._")
        clinical_lines.append("")
        clinical_lines.append("## IMPRESSION")
        if brain_seg_ok:
            clinical_lines.append("1. Brain tumor segmentation performed.")
        else:
            clinical_lines.append("1. Brain tumor segmentation not available.")

    # Merge VLM QC into the clinical report when available and consistent (brain-only template).
    if brain_mode:
        try:
            norm_qc = norm_struct
            if norm_qc is None and structured_path.exists():
                obj = json.loads(structured_path.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    sel_imgs = list(selected_images_for_rules)
                    norm_qc = _normalize_llm_structured(obj, evidence_bundle=evidence_bundle)
                    norm_qc = _apply_structured_report_rules(
                        norm_qc, evidence_bundle=evidence_bundle, selected_images=sel_imgs, run_dir=run_dir
                    )
            if isinstance(norm_qc, dict):
                iq = norm_qc.get("image_qc") if isinstance(norm_qc.get("image_qc"), dict) else {}
                reviewed = iq.get("images_reviewed") if isinstance(iq.get("images_reviewed"), list) else []
                clinical_lines.append("")
                clinical_lines.append("## IMAGE QUALITY CONTROL (VLM)")
                if reviewed:
                    clinical_lines.append(f"- images_reviewed: {reviewed}")
                    clinical_lines.append(f"- roi_coverage_ok: {bool(iq.get('roi_coverage_ok'))}")
                    clinical_lines.append(f"- registration_alignment_ok: {bool(iq.get('registration_alignment_ok'))}")
                    gsp = str(iq.get("gross_signal_patterns") or "").strip()
                    if gsp:
                        clinical_lines.append(f"- gross_signal_patterns: {gsp}")
                else:
                    clinical_lines.append("- VLM QC unavailable (no images reviewed).")
        except Exception:
            pass

    clinical_report_path = out_dir / "clinical_report.md"
    clinical_report_path.write_text("\n".join(clinical_lines) + "\n", encoding="utf-8")

    lines: List[str] = []
    lines.append("## MRI Agent Run Report")
    lines.append(f"**Case ID:** {state.get('case_id')}")
    lines.append(f"**Run ID:** {state.get('run_id')}")
    lines.append("")
    lines.append(
        "Sequences present (heuristic): " + ", ".join(present) if present else "Sequences present: unknown"
    )
    lines.append("")
    lines.append("### Selected Mapping")
    for k, v in mapping.items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")
    # Technical factors (from DICOM header when available).
    try:
        series_list = series if isinstance(series, list) else []
        t2_series = next((s for s in series_list if isinstance(s, dict) and s.get("sequence_guess") == "T2w"), None)
        t2_tags = _parse_dicom_header_tags(t2_series.get("dicom_header_txt") if isinstance(t2_series, dict) else None)
        manufacturer = t2_tags.get("Manufacturer") or ""
        field_strength = t2_tags.get("MagneticFieldStrength") or ""
        model = t2_tags.get("ManufacturerModelName") or ""
        slice_thickness = t2_tags.get("SliceThickness") or ""
        tech_bits: List[str] = []
        if manufacturer or field_strength or model:
            tech_bits.append(" ".join([x for x in [manufacturer, f"{field_strength}T" if field_strength else "", model] if x]))
        if slice_thickness:
            tech_bits.append(f"T2w slice thickness: {slice_thickness} mm")
        if tech_bits:
            lines.append("### Technical Factors")
            lines.append("; ".join(tech_bits))
            lines.append("")
    except Exception:
        pass
    lines.append("### Stage Status")
    # Only stages this domain actually has; see the domain-scoped `stage_status`
    # built for report.json below.
    _stage_pairs: List[Tuple[str, Any]] = [
        ("ingest_dicom_to_nifti", ingest_ok),
        ("identify_sequences", ident_ok),
    ]
    if domain_cfg.name == "cardiac":
        _stage_pairs += [
            ("reconstruct_grappa", recon_ok),
            ("segment_cardiac_cine", card_seg_ok),
            ("classify_cardiac_cine_disease", card_cls_ok),
            ("extract_roi_features", feat_ok),
        ]
    elif domain_cfg.name == "brain":
        _stage_pairs += [
            ("register_to_reference", reg_ok),
            ("brats_mri_segmentation", brain_seg_ok),
            ("extract_roi_features", feat_ok),
        ]
    else:
        _stage_pairs += [
            ("register_to_reference", reg_ok),
            ("segment_prostate", seg_ok),
            ("extract_roi_features", feat_ok),
        ]
    for _sname, _sok in _stage_pairs:
        lines.append(f"- {_sname}: {'OK' if _sok else 'missing/failed'}")
    lines.append("")

    # Domain-independent: recorded whenever the case actually came from k-space.
    _recon_md_lines = _reconstruction_lines(recon_provenance)
    if _recon_md_lines:
        lines.append("### Acquisition / Reconstruction")
        lines.extend(_recon_md_lines)
        lines.append("")

    lines.append("### Key Outputs")
    def _shorten_path(val: Optional[str]) -> Optional[str]:
        if not val:
            return None
        return to_short_path(str(val), run_dir=run_dir)

    def _shorten_list(items: Any) -> List[str]:
        if not isinstance(items, list):
            return []
        out: List[str] = []
        for it in items:
            if isinstance(it, str):
                out.append(to_short_path(it, run_dir=run_dir))
        return out

    if isinstance(reg, dict) and reg.get("qc_pngs"):
        items = _shorten_list(reg.get("qc_pngs"))
        lines.append(f"- registration qc_pngs: {', '.join(items) if items else reg.get('qc_pngs')}")
    if isinstance(reg, dict) and reg.get("resampled_paths"):
        resampled = reg.get("resampled_paths")
        if isinstance(resampled, dict):
            short_map = {k: _shorten_path(v) for k, v in resampled.items()}
            lines.append(f"- registration resampled_paths: {short_map}")
        else:
            lines.append(f"- registration resampled_paths: {resampled}")
    if isinstance(seg, dict) and seg.get("prostate_mask_path"):
        lines.append(f"- prostate_mask_path: {_shorten_path(seg.get('prostate_mask_path'))}")
    if isinstance(seg, dict) and seg.get("zone_mask_path"):
        lines.append(f"- zone_mask_path: {_shorten_path(seg.get('zone_mask_path'))}")
    if feature_table_path:
        lines.append(f"- feature_table_path: {_shorten_path(feature_table_path)}")
    if isinstance(feat, dict) and feat.get("feature_table_md_path"):
        lines.append(f"- feature_table_md_path: {_shorten_path(feat.get('feature_table_md_path'))}")
    elif isinstance(feat, dict) and feat.get("feature_table_path"):
        lines.append(f"- feature_table_path: {_shorten_path(feat.get('feature_table_path'))}")
    if isinstance(feat, dict) and feat.get("slice_summary_path"):
        lines.append(f"- slice_summary_path: {_shorten_path(feat.get('slice_summary_path'))}")
    if isinstance(feat, dict) and feat.get("overlay_pngs"):
        items = _shorten_list(feat.get("overlay_pngs"))
        lines.append(f"- overlay_pngs: {', '.join(items) if items else feat.get('overlay_pngs')}")
    lines.append(f"- vlm_evidence_bundle_path: {_shorten_path(str(vlm_evidence_bundle_path))}")

    lines.append("")
    lines.append("### Notes / Limitations")
    lines.append("- This is an engineering demo; outputs are not clinically validated.")

    report_txt_path = out_dir / "report.md"
    report_txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Stage status is domain-scoped: a cardiac run has no `segment_prostate`
    # stage, and reporting one as "missing/failed" is a false statement about
    # the pipeline rather than a fact about the case.
    stage_status: Dict[str, Any] = {
        "ingest_dicom_to_nifti": ingest_ok,
        "identify_sequences": ident_ok,
    }
    if domain_cfg.name == "cardiac":
        stage_status.update(
            {
                "reconstruct_grappa": recon_ok,
                "segment_cardiac_cine": card_seg_ok,
                "classify_cardiac_cine_disease": card_cls_ok,
                "extract_roi_features": feat_ok,
            }
        )
    elif domain_cfg.name == "brain":
        stage_status.update(
            {
                "register_to_reference": reg_ok,
                "brats_mri_segmentation": brain_seg_ok,
                "extract_roi_features": feat_ok,
            }
        )
    else:
        stage_status.update(
            {
                "register_to_reference": reg_ok,
                "segment_prostate": seg_ok,
                "extract_roi_features": feat_ok,
            }
        )

    report_json: Dict[str, Any] = {
        "case_id": state.get("case_id"),
        "run_id": state.get("run_id"),
        "domain": domain_cfg.name,
        "domain_source": domain_source,
        "sequences_present": present,
        "mapping": mapping,
        "feature_table_path": evidence_bundle.get("features", {}).get("feature_table_path"),
        "vlm_evidence_bundle_path": str(vlm_evidence_bundle_path),
        "stage_status": stage_status,
    }

    # Domain-specific blocks.  A cardiac report must not carry PI-RADS
    # structures and a prostate report must not carry cardiac ones.
    if domain_cfg.name == "prostate":
        report_json["lesion_assessment_meta"] = {
            "evidence_tier": evidence_tier,
            "lesion_tool_status": lesion_tool_status,
            "adc_available": adc_available,
            "segmentation_usable": segmentation_usable,
            "lesion_geometry_note": lesion_geo_note or None,
            "lesion_candidates_n": lesion_candidates_n,
            "lesion_max_prob": lesion_max_prob,
            "final_overall_pirads": final_overall_score,
            "final_overall_pirads_range": final_overall_range,
        }
    elif domain_cfg.name == "cardiac":
        _card_ev = evidence_bundle.get("cardiac_classification") if isinstance(evidence_bundle, dict) else {}
        _card_ev = _card_ev if isinstance(_card_ev, dict) else {}
        report_json["cardiac_assessment"] = {
            "classification_ok": bool(card_cls_ok),
            "predicted_group": _card_ev.get("predicted_group"),
            "ground_truth_group": _card_ev.get("ground_truth_group"),
            "needs_vlm_review": _card_ev.get("needs_vlm_review"),
            "metrics": _card_ev.get("metrics"),
            "phase_indices": _card_ev.get("phase_indices"),
            "rule_trace": _card_ev.get("rule_trace"),
            "segmentation_ok": bool(card_seg_ok),
        }
        report_json["cardiac_t1_feature_analysis"] = evidence_bundle.get("cardiac_t1_feature_analysis")
        report_json["impression"] = {
            **impression_meta,
            "template_lines": cardiac_template_impression,
        }
    elif domain_cfg.name == "brain":
        report_json["brain_assessment"] = {
            "segmentation_ok": bool(brain_seg_ok),
            "wt_volume_ml": brain_wt_vol_ml,
            "tc_volume_ml": brain_tc_vol_ml,
            "et_volume_ml": brain_et_vol_ml,
        }

    # Reconstruction provenance is domain-independent: it is recorded whenever
    # the case actually came from raw k-space.
    if isinstance(recon_provenance, dict) and recon_provenance.get("ok"):
        report_json["reconstruction"] = recon_provenance

    report_json["limitations"] = []

    report_json_path = out_dir / "report.json"
    report_json_path.write_text(json.dumps(report_json, indent=2) + "\n", encoding="utf-8")

    emit_structured_report_text = bool(args.get("emit_structured_report_text", False))
    llm_structured_report_path: Optional[Path] = None
    llm_structured_report_error: Optional[str] = None
    llm_mode = str(args.get("llm_mode") or "").strip()
    if not llm_mode and domain_cfg.name == "brain":
        # Default to server mode for brain so the VLM step runs by default.
        llm_mode = "server"
    adapter: Optional[Any] = None
    max_tokens = int(args.get("max_tokens") or 900)
    temperature = float(args.get("temperature") or 0.0)
    if llm_mode == "server":
        base_url = str(args.get("server_base_url") or "http://127.0.0.1:8000")
        model = str(
            args.get("server_model")
            or os.environ.get("MEDGEMMA_SERVER_MODEL")
            or "Qwen/Qwen3-VL-30B-A3B-Thinking"
        )
        base_url_l = base_url.strip().lower()
        model_l = model.strip().lower()
        is_local = ("127.0.0.1" in base_url_l) or ("localhost" in base_url_l)
        if model_l.startswith("gpt-") or model_l.startswith("o1") or model_l.startswith("o3"):
            # Avoid OpenAI model names when using a local vLLM server.
            if is_local:
                model = os.environ.get("MEDGEMMA_SERVER_MODEL") or "Qwen/Qwen3-VL-30B-A3B-Thinking"
        # Common alias from earlier experiments; vLLM will 404 if it doesn't match /v1/models.
        if model.strip().lower() in ("qwen-vl", "qwen_vl", "qwenvl", "vl", "qwen"):
            model = os.environ.get("MEDGEMMA_SERVER_MODEL", "Qwen/Qwen3-VL-30B-A3B-Thinking")
        adapter = VLLMOpenAIChatAdapter(
            VLLMServerConfig(base_url=base_url, model=model, max_tokens=max_tokens, temperature=temperature)
        )
    elif llm_mode == "openai":
        api_model = str(args.get("api_model") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini")
        api_base_url = str(args.get("api_base_url") or os.environ.get("OPENAI_BASE_URL") or "").strip() or None
        adapter = OpenAIChatAdapter(OpenAIConfig(model=api_model, base_url=api_base_url, max_tokens=max_tokens, temperature=temperature))
    elif llm_mode == "anthropic":
        api_model = str(args.get("api_model") or os.environ.get("ANTHROPIC_MODEL") or "claude-3-7-sonnet-20250219")
        adapter = AnthropicChatAdapter(AnthropicConfig(model=api_model, max_tokens=max_tokens, temperature=temperature))
    elif llm_mode == "gemini":
        api_model = str(args.get("api_model") or os.environ.get("GEMINI_MODEL") or "gemini-1.5-pro")
        adapter = GeminiChatAdapter(GeminiConfig(model=api_model, max_tokens=max_tokens, temperature=temperature))

    if adapter is not None:
        try:
            guided = _llm_structured_report_schema(domain_cfg)

            # Embed a few key images (prefer montage) as data URLs so the VLM can actually see them.
            def _seq_bucket(p: str) -> str:
                s = p.lower()
                if domain_cfg.name == "brain":
                    if "t1ce" in s or "t1c" in s or "t1gd" in s:
                        return "t1c"
                    if "flair" in s:
                        return "flair"
                    if "t2" in s:
                        return "t2"
                    if "t1" in s:
                        return "t1"
                    return "other"
                if "t2w" in s:
                    return "t2w"
                if "high_b" in s or "high-b" in s:
                    return "dwi_high_b"
                if "low_b" in s or "low-b" in s:
                    return "dwi_low_b"
                if "adc" in s:
                    return "adc"
                if "dwi" in s:
                    return "dwi_other"
                return "other"

            def _overlay_kind(item: Dict[str, Any]) -> str:
                kind = str(item.get("kind") or "").strip()
                if kind:
                    return kind
                p = str(item.get("path") or "").lower()
                if not p:
                    return ""
                if "full_frame_montage_4x4" in p:
                    return "full_frame_montage_4x4"
                if "full_frame_montage_5x5" in p:
                    return "full_frame_montage_5x5"
                if "full_frame_montage" in p:
                    return "full_frame_montage"
                if "full_frame_best" in p:
                    return "full_frame_best_slice"
                if "roi_crop_montage_4x4" in p:
                    return "roi_crop_montage_4x4"
                if "roi_crop_montage_5x5" in p:
                    return "roi_crop_montage_5x5"
                if "roi_crop_montage" in p:
                    return "roi_crop_montage"
                if "roi_crop_best" in p:
                    return "roi_crop_best_slice"
                if p.endswith("_tiles.json"):
                    if "full_frame_montage" in p:
                        return "full_frame_montage_4x4_tiles"
                    return "roi_crop_montage_4x4_tiles"
                return ""

            selected_images_debug: List[Dict[str, Any]] = []
            imgs_for_vlm: List[str] = []
            def _normalize_overlay_items(raw: Any, *, source: str) -> List[Dict[str, Any]]:
                items: List[Dict[str, Any]] = []
                if not isinstance(raw, list):
                    return items
                for it in raw:
                    if not isinstance(it, dict):
                        continue
                    path = str(it.get("abs_path") or it.get("path") or "").strip()
                    if not path:
                        continue
                    item = dict(it)
                    item["path"] = path
                    if not item.get("kind"):
                        item["kind"] = _overlay_kind(item)
                    if not item.get("roi_source"):
                        low = path.lower()
                        if "lesion" in low:
                            item["roi_source"] = "lesion"
                        elif "prostate" in low:
                            item["roi_source"] = "prostate"
                    item["source"] = source
                    items.append(item)
                return items

            overlay_candidates: List[Dict[str, Any]] = []
            overlay_candidates.extend(_normalize_overlay_items(overlay_list, source="case_state"))
            overlay_candidates.extend(_normalize_overlay_items(features_main.get("overlay_pngs"), source="features_main"))
            overlay_candidates.extend(_normalize_overlay_items(features_lesion.get("overlay_pngs"), source="features_lesion"))
            # De-duplicate by path while preserving order.
            seen_paths: set[str] = set()
            deduped: List[Dict[str, Any]] = []
            for it in overlay_candidates:
                p = str(it.get("path") or "")
                if not p or p in seen_paths:
                    continue
                seen_paths.add(p)
                deduped.append(it)
            overlay_candidates = deduped

            bucket_order = ["t1c", "t1", "t2", "flair"] if domain_cfg.name == "brain" else ["t2w", "adc", "dwi_high_b", "dwi_low_b"]
            preferred_kinds = []
            if domain_cfg.name == "brain":
                preferred_kinds = ["full_frame_montage_4x4", "roi_crop_montage_4x4", "full_frame_montage_5x5", "roi_crop_montage_5x5"]
            else:
                preferred_kinds = ["full_frame_montage_4x4", "roi_crop_montage_4x4", "full_frame_montage_5x5", "roi_crop_montage_5x5"]

            def _pick_montages(items: List[Dict[str, Any]], *, roi_filter: Optional[str], reason: str) -> None:
                picked_by_bucket: Dict[str, Dict[str, Any]] = {}
                for kind in preferred_kinds:
                    for it in items:
                        if roi_filter and str(it.get("roi_source") or "") != roi_filter:
                            continue
                        if _overlay_kind(it) != kind:
                            continue
                        p = str(it.get("path") or "")
                        b = _seq_bucket(p)
                        if b in bucket_order and b not in picked_by_bucket:
                            picked_by_bucket[b] = it
                for b in bucket_order:
                    if b in picked_by_bucket:
                        it = picked_by_bucket[b]
                        p = str(it.get("path") or "")
                        if p and p not in imgs_for_vlm:
                            imgs_for_vlm.append(p)
                            selected_images_debug.append(
                                {
                                    "path": p,
                                    "kind": _overlay_kind(it),
                                    "bucket": b,
                                    "roi_source": it.get("roi_source"),
                                    "why": reason,
                                }
                            )

            # 1) Prefer lesion overlays for prostate (best lesion localization).
            if domain_cfg.name == "prostate":
                _pick_montages(
                    overlay_candidates,
                    roi_filter="lesion",
                    reason="Lesion montages provide ROI-localized evidence for PI-RADS characterization.",
                )
                _pick_montages(
                    overlay_candidates,
                    roi_filter="prostate",
                    reason="Whole-gland montages provide context for zonal anatomy and QC.",
                )
            else:
                _pick_montages(
                    overlay_candidates,
                    roi_filter=None,
                    reason="Montage provides consistent visual context across ROI slices.",
                )

            # 2) If still short, add best-slice images as fallback.
            best_kinds = {"full_frame_best_slice", "roi_crop_best_slice"}
            for it in overlay_candidates:
                if _overlay_kind(it) not in best_kinds:
                    continue
                p = str(it.get("path") or "")
                b = _seq_bucket(p)
                if p and b in bucket_order and p not in imgs_for_vlm:
                    imgs_for_vlm.append(p)
                    selected_images_debug.append(
                        {
                            "path": p,
                            "kind": _overlay_kind(it),
                            "bucket": b,
                            "roi_source": it.get("roi_source"),
                            "why": "Best-slice fallback for qualitative signal assessment.",
                        }
                    )

            max_images = 8 if domain_cfg.name == "brain" else 6
            imgs_for_vlm = imgs_for_vlm[:max_images]
            img_blocks = []
            embedded_images_debug: List[Dict[str, Any]] = []
            for p in imgs_for_vlm:
                data_url = _encode_png_data_url(p)
                if data_url:
                    img_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
                    embedded_images_debug.append(
                        {"path": str(p), "embedded": True, "bytes": _file_size_bytes(str(p))}
                    )
                else:
                    embedded_images_debug.append(
                        {"path": str(p), "embedded": False, "bytes": _file_size_bytes(str(p))}
                    )

            base_instructions = [
                "Use features_preview_rows and slice_summary_preview for quantitative statements.",
                "Use provided images to fill image_qc (ROI coverage, registration alignment, gross signal patterns).",
                "Images may be full-frame or ROI-cropped and often include ROI outlines plus z-index labels; use them when available.",
                "ROIs correspond to segmented masks; focus descriptions within the ROI and cite tile labels (rX cY zZ) when possible.",
                "Do not hallucinate lesions. If uncertain, mark indeterminate and add to limitations.",
                "If evidence is insufficient for a claim, mark it indeterminate and add to limitations.",
            ]
            if domain_cfg.name == "prostate":
                extra_instructions = [
                    "Prefer evidence_bundle.features_main for prostate volume and zonal summaries.",
                    "Use evidence_bundle.features_lesion for lesion ROI summaries when present.",
                    "If evidence_bundle.lesion_geometry.ok is true, summarize the dominant lesion's volume and location from that structure.",
                    "Selected images may include lesion overlays (roi_source=lesion) and whole-gland overlays (roi_source=prostate); use lesion overlays for PI-RADS characterization and whole-gland overlays for zonal context.",
                ]
            else:
                extra_instructions = [
                    "If tumor masks are present (TC/WT/ET), summarize their presence and any available quantitative info.",
                    "If no tumor masks or quantitative metrics are available, state that explicitly.",
                ]
            user_content = [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "evidence_bundle_path": str(vlm_evidence_bundle_path),
                            "evidence_bundle": evidence_bundle,
                            "selected_images": selected_images_debug,
                            "instructions": base_instructions + extra_instructions,
                        },
                        ensure_ascii=False,
                    ),
                }
            ] + img_blocks

            messages = [
                {"role": "system", "content": _structured_system_prompt(domain_cfg)},
                {"role": "user", "content": user_content},
            ]

            # Persist the EXACT multimodal request we sent (for debugging "did the VLM see images?")
            try:
                (vlm_dir / "vlm_request_messages.json").write_text(
                    json.dumps(messages, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                # Also write a redacted view so humans can inspect structure without huge base64 blobs.
                try:
                    redacted = json.loads(json.dumps(messages))
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
                    (vlm_dir / "vlm_request_messages_redacted.json").write_text(
                        json.dumps(redacted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                    )
                except Exception:
                    pass
                (vlm_dir / "vlm_embedded_images.json").write_text(
                    json.dumps(embedded_images_debug, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                (vlm_dir / "vlm_selected_images.json").write_text(
                    json.dumps(selected_images_debug, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
            except Exception:
                pass

            def _merge_chunk_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
                merged: Dict[str, Any] = {
                    "case_id": state.get("case_id"),
                    "run_id": state.get("run_id"),
                    "image_qc": {
                        "images_reviewed": [],
                        "roi_coverage_ok": True,
                        "registration_alignment_ok": True,
                        "gross_signal_patterns": "",
                        "tile_evidence": [],
                        "notes": [],
                    },
                    "limitations": [],
                }
                if domain_cfg.name == "brain":
                    merged["brain_signal_characterization"] = {
                        "t1c_enhancement": "indeterminate",
                        "flair_edema": "indeterminate",
                        "t1_necrosis": "indeterminate",
                        "t2_extent": "indeterminate",
                    }

                def _is_indeterminate(val: Optional[str]) -> bool:
                    return not val or str(val).strip().lower() == "indeterminate"

                images_reviewed: List[str] = []
                tile_evidence: List[Dict[str, str]] = []
                notes: List[str] = []
                gross_parts: List[str] = []
                roi_flags: List[bool] = []
                reg_flags: List[bool] = []
                limitations: List[str] = []

                for r in results:
                    if not isinstance(r, dict):
                        continue
                    iq = r.get("image_qc") if isinstance(r.get("image_qc"), dict) else {}
                    imgs = iq.get("images_reviewed") if isinstance(iq.get("images_reviewed"), list) else []
                    for s in imgs:
                        s = str(s)
                        if s and s not in images_reviewed:
                            images_reviewed.append(s)
                    if "roi_coverage_ok" in iq:
                        roi_flags.append(bool(iq.get("roi_coverage_ok")))
                    if "registration_alignment_ok" in iq:
                        reg_flags.append(bool(iq.get("registration_alignment_ok")))
                    gsp = str(iq.get("gross_signal_patterns") or "").strip()
                    if gsp:
                        gross_parts.append(gsp)
                    tev = iq.get("tile_evidence") if isinstance(iq.get("tile_evidence"), list) else []
                    for ev in tev:
                        if isinstance(ev, dict):
                            tile_evidence.append(
                                {
                                    "sequence": str(ev.get("sequence") or ""),
                                    "tile": str(ev.get("tile") or ""),
                                    "task": str(ev.get("task") or ""),
                                    "observation": str(ev.get("observation") or ""),
                                }
                            )
                    nlist = iq.get("notes") if isinstance(iq.get("notes"), list) else []
                    for n in nlist:
                        n = str(n).strip()
                        if n and n not in notes:
                            notes.append(n)
                    lims = r.get("limitations") if isinstance(r.get("limitations"), list) else []
                    for lim in lims:
                        lim = str(lim).strip()
                        if lim and lim not in limitations:
                            limitations.append(lim)

                    if domain_cfg.name == "brain":
                        bsc = r.get("brain_signal_characterization") if isinstance(r.get("brain_signal_characterization"), dict) else {}
                        for key in ("t1c_enhancement", "flair_edema", "t1_necrosis", "t2_extent"):
                            val = str(bsc.get(key) or "").strip()
                            if not _is_indeterminate(val) and _is_indeterminate(merged["brain_signal_characterization"].get(key)):
                                merged["brain_signal_characterization"][key] = val

                merged["image_qc"]["images_reviewed"] = images_reviewed[:8]
                if roi_flags:
                    merged["image_qc"]["roi_coverage_ok"] = all(roi_flags)
                else:
                    merged["image_qc"]["roi_coverage_ok"] = False
                if reg_flags:
                    merged["image_qc"]["registration_alignment_ok"] = all(reg_flags)
                else:
                    merged["image_qc"]["registration_alignment_ok"] = False
                merged["image_qc"]["gross_signal_patterns"] = " | ".join(gross_parts) if gross_parts else "indeterminate"
                merged["image_qc"]["tile_evidence"] = tile_evidence[:8] if tile_evidence else [
                    {
                        "sequence": "unknown",
                        "tile": "r0 c0 z?",
                        "task": "roi_coverage",
                        "observation": "No explicit tile citations provided; indeterminate tile-level QC.",
                    }
                ]
                merged["image_qc"]["notes"] = notes[:16]
                if limitations:
                    merged["limitations"] = limitations
                return merged

            if domain_cfg.name == "brain" and len(imgs_for_vlm) > 2:
                guided_chunk = _llm_image_qc_schema(domain_cfg)
                chunk_results: List[Dict[str, Any]] = []
                chunk_size = 2
                for idx in range(0, len(imgs_for_vlm), chunk_size):
                    chunk_paths = imgs_for_vlm[idx : idx + chunk_size]
                    chunk_blocks: List[Dict[str, Any]] = []
                    chunk_embedded: List[Dict[str, Any]] = []
                    for p in chunk_paths:
                        data_url = _encode_png_data_url(p)
                        if data_url:
                            chunk_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
                            chunk_embedded.append({"path": str(p), "embedded": True, "bytes": _file_size_bytes(str(p))})
                        else:
                            chunk_embedded.append({"path": str(p), "embedded": False, "bytes": _file_size_bytes(str(p))})

                    chunk_user_content = [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "evidence_bundle_path": str(vlm_evidence_bundle_path),
                                    "evidence_bundle": evidence_bundle,
                                    "selected_images": [it for it in selected_images_debug if it.get("path") in chunk_paths],
                                    "instructions": base_instructions
                                    + extra_instructions
                                    + [
                                        f"Chunk {idx // chunk_size + 1} of {math.ceil(len(imgs_for_vlm) / chunk_size)}: only assess the provided images.",
                                        "Include provided image paths in image_qc.images_reviewed.",
                                    ],
                                },
                                ensure_ascii=False,
                            ),
                        }
                    ] + chunk_blocks
                    chunk_messages = [
                        {"role": "system", "content": _structured_system_prompt(domain_cfg)},
                        {"role": "user", "content": chunk_user_content},
                    ]

                    try:
                        (vlm_dir / f"vlm_request_messages_chunk{idx // chunk_size + 1}.json").write_text(
                            json.dumps(chunk_messages, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                        )
                    except Exception:
                        pass

                    raw = adapter.generate_with_schema(chunk_messages, guided_json=guided_chunk)
                    (vlm_dir / f"llm_structured_report_raw_chunk{idx // chunk_size + 1}.txt").write_text(
                        raw + "\n", encoding="utf-8"
                    )
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        obj = None
                    if isinstance(obj, dict):
                        (vlm_dir / f"llm_structured_report_raw_chunk{idx // chunk_size + 1}.json").write_text(
                            json.dumps(obj, indent=2) + "\n", encoding="utf-8"
                        )
                        chunk_results.append(obj)

                merged = _merge_chunk_results(chunk_results)
                raw = json.dumps(merged, indent=2)
                (vlm_dir / "llm_structured_report_raw.txt").write_text(raw + "\n", encoding="utf-8")
                obj = merged
            else:
                raw = adapter.generate_with_schema(messages, guided_json=guided)
                (vlm_dir / "llm_structured_report_raw.txt").write_text(raw + "\n", encoding="utf-8")
                try:
                    obj = json.loads(raw)
                except Exception:
                    obj = None

            if isinstance(obj, dict):
                # Write raw JSON for debugging
                (vlm_dir / "llm_structured_report_raw.json").write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
                norm = _normalize_llm_structured(obj, evidence_bundle=evidence_bundle)
                norm = _apply_structured_report_rules(
                    norm, evidence_bundle=evidence_bundle, selected_images=selected_images_debug, run_dir=run_dir
                )
                llm_structured_report_path = out_dir / "structured_report.json"
                llm_structured_report_path.write_text(json.dumps(norm, indent=2) + "\n", encoding="utf-8")
                if emit_structured_report_text:
                    (out_dir / "structured_report.txt").write_text(_format_report_txt(norm), encoding="utf-8")
        except Exception as e:
            llm_structured_report_error = repr(e)
            (vlm_dir / "llm_structured_report_error.txt").write_text(llm_structured_report_error + "\n", encoding="utf-8")

    # If a structured report JSON already exists (e.g. from a previous run),
    # normalize it and optionally write structured_report.txt.
    existing_struct = out_dir / "structured_report.json"
    if existing_struct.exists():
        try:
            existing_obj = json.loads(existing_struct.read_text(encoding="utf-8"))
            if isinstance(existing_obj, dict):
                norm2 = _normalize_llm_structured(existing_obj, evidence_bundle=evidence_bundle)
                selected_images_for_rules: List[Dict[str, Any]] = []
                try:
                    sel_path = vlm_dir / "vlm_selected_images.json"
                    if sel_path.exists():
                        sel_obj = json.loads(sel_path.read_text(encoding="utf-8"))
                        if isinstance(sel_obj, list):
                            selected_images_for_rules = [it for it in sel_obj if isinstance(it, dict)]
                except Exception:
                    selected_images_for_rules = []
                norm2 = _apply_structured_report_rules(
                    norm2, evidence_bundle=evidence_bundle, selected_images=selected_images_for_rules, run_dir=run_dir
                )
                existing_struct.write_text(json.dumps(norm2, indent=2) + "\n", encoding="utf-8")
                if emit_structured_report_text:
                    (out_dir / "structured_report.txt").write_text(_format_report_txt(norm2), encoding="utf-8")
        except Exception:
            pass
    if not emit_structured_report_text:
        try:
            sr_txt = out_dir / "structured_report.txt"
            if sr_txt.exists():
                sr_txt.unlink()
        except Exception:
            pass

    artifacts = [
        ArtifactRef(path=str(report_txt_path), kind="text", description="Report text"),
        ArtifactRef(path=str(report_json_path), kind="json", description="Report JSON"),
        ArtifactRef(path=str(clinical_report_path), kind="text", description="Clinical-style report text"),
        ArtifactRef(path=str(vlm_evidence_bundle_path), kind="json", description="VLM evidence bundle"),
        ArtifactRef(path=str(case_context_path), kind="json", description="CaseContext (stable run context)"),
    ]
    if llm_structured_report_path is not None:
        artifacts.append(
            ArtifactRef(path=str(llm_structured_report_path), kind="json", description="VLM structured report JSON")
        )
        if emit_structured_report_text:
            artifacts.append(
                ArtifactRef(
                    path=str(llm_structured_report_path.with_suffix(".txt")),
                    kind="text",
                    description="VLM structured report text",
                )
            )

    sources = [ArtifactRef(path=str(case_state_path), kind="json", description="CaseState input")]
    if feature_table_path:
        sources.append(ArtifactRef(path=str(feature_table_path), kind="csv", description="Feature table input"))
    feat_md = evidence_bundle.get("features", {}).get("feature_table_md_path")
    if feat_md:
        sources.append(ArtifactRef(path=str(feat_md), kind="md", description="Feature table input (markdown)"))

    return {
        "data": (
            {
                "report_txt_path": str(report_txt_path),
                "report_json_path": str(report_json_path),
                "clinical_report_path": str(clinical_report_path),
                "vlm_evidence_bundle_path": str(vlm_evidence_bundle_path),
                **(
                    {"llm_structured_report_path": str(llm_structured_report_path)}
                    if llm_structured_report_path is not None
                    else {}
                ),
                **(
                    {"llm_structured_report_error": llm_structured_report_error}
                    if llm_structured_report_error is not None and llm_structured_report_path is None
                    else {}
                ),
            }
        ),
        "artifacts": artifacts,
        "warnings": (
            [f"VLM structured report failed: {llm_structured_report_error}"]
            if llm_structured_report_error is not None and llm_structured_report_path is None
            else []
        ),
        "source_artifacts": sources,
        "generated_artifacts": artifacts,
    }


def build_tool() -> Tool:
    return Tool(spec=REPORT_SPEC, func=generate_report)

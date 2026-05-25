from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from core.paths import project_root
from .template_schema import PlanTemplate


_TEMPLATE_FILE_BY_KEY: Dict[Tuple[str, str], str] = {
    ("prostate", "full_pipeline"): "prostate_full_pipeline.json",
    ("brain", "full_pipeline"): "brain_full_pipeline.json",
    ("cardiac", "full_pipeline"): "cardiac_full_pipeline.json",
    ("*", "qa"): "qa.json",
    ("*", "custom_analysis"): "custom_analysis.json",
    ("*", "denoise"): "denoise.json",
    ("*", "super_resolution"): "super_resolution.json",
    ("*", "raw_recon"): "raw_recon.json",
}

# noToken-specific template variants are intentionally sparse:
# only tasks/domains with explicit noToken-compatible contracts should exist.
_TEMPLATE_FILE_BY_KEY_NOTOKEN: Dict[Tuple[str, str], str] = {
    ("prostate", "full_pipeline"): "prostate_full_pipeline_notoken.json",
    ("brain", "full_pipeline"): "brain_full_pipeline_notoken.json",
    ("*", "raw_recon"): "raw_recon_notoken.json",
}


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
        "rawrecon": "raw_recon",
        "raw-recon": "raw_recon",
        "raw_reconstruction": "raw_recon",
        "reconstruct_raw": "raw_recon",
        "metadata": "qa",
        "read_metadata": "qa",
        "question_answer": "qa",
        "question-answer": "qa",
        "custom": "custom_analysis",
        "code_interpreter": "custom_analysis",
        "code-interpreter": "custom_analysis",
        "sandbox": "custom_analysis",
    }
    return aliases.get(s, s)


def _resolve_template_key(*, domain: str, request_type: str) -> Tuple[str, str]:
    dom = str(domain or "").strip().lower()
    req = _normalize_request_type(request_type)
    if not req:
        req = "full_pipeline"

    if req in {"qa", "custom_analysis", "denoise", "super_resolution", "raw_recon"}:
        return "*", req

    if req in {"full_pipeline", "register", "segment", "lesion", "report", "classify"}:
        return dom, "full_pipeline"

    return dom, "full_pipeline"


def _templates_dir() -> Path:
    return project_root() / "agent" / "plans" / "templates"


@lru_cache(maxsize=64)
def _load_template_cached(path_str: str) -> Tuple[PlanTemplate, str]:
    path = Path(path_str)
    raw_bytes = path.read_bytes()
    template_hash = hashlib.sha256(raw_bytes).hexdigest()
    payload = json.loads(raw_bytes.decode("utf-8"))
    tmpl = PlanTemplate.model_validate(payload)
    return tmpl, template_hash


def load_plan_template(*, domain: str, request_type: str, variant: str = "default") -> PlanTemplate:
    key = _resolve_template_key(domain=domain, request_type=request_type)
    v = str(variant or "default").strip().lower() or "default"
    if v in {"default", "full"}:
        fname = _TEMPLATE_FILE_BY_KEY.get(key)
    elif v in {"notoken", "no_token"}:
        fname = _TEMPLATE_FILE_BY_KEY_NOTOKEN.get(key)
        if not fname:
            raise FileNotFoundError(
                f"Unsupported noToken template for domain={domain!r}, request_type={request_type!r} (key={key})"
            )
    else:
        raise ValueError(f"Unsupported template variant: {variant!r}")
    if not fname:
        raise ValueError(f"No plan template mapping for domain={domain!r}, request_type={request_type!r}")

    path = _templates_dir() / fname
    if not path.exists():
        raise FileNotFoundError(f"Plan template file not found: {path}")

    tmpl, template_hash = _load_template_cached(str(path))
    return tmpl.model_copy(
        deep=True,
        update={
            "source_path": str(path),
            "template_hash": template_hash,
        },
    )


def list_supported_request_types() -> List[str]:
    out = {"full_pipeline", "register", "segment", "lesion", "report", "classify"}
    out.update({req for (_, req) in _TEMPLATE_FILE_BY_KEY if req != "full_pipeline"})
    return sorted(out)


def template_file_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for (domain, req), fname in _TEMPLATE_FILE_BY_KEY.items():
        mapping[f"{domain}:{req}"] = fname
    for (domain, req), fname in _TEMPLATE_FILE_BY_KEY_NOTOKEN.items():
        mapping[f"notoken::{domain}:{req}"] = fname
    return mapping

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .bootstrap import ensure_import_paths
from .dummy_tools import build_dummy_tools

ensure_import_paths()

from commands.registry import ToolRegistry  # noqa: E402


_BASE_DOMAINS: Sequence[str] = ("prostate", "brain", "cardiac")

_DOMAIN_CAPABILITY_REQUIREMENTS: Dict[str, Dict[str, List[str]]] = {
    "prostate": {
        "full_pipeline": [
            "identify_sequences",
            "register_to_reference",
            "segment_prostate",
            "generate_report",
        ],
        "segment": ["identify_sequences", "segment_prostate"],
        "register": ["identify_sequences", "register_to_reference"],
        "lesion": ["identify_sequences", "segment_prostate", "detect_lesion_candidates"],
        "report": ["generate_report"],
        "qa": ["identify_sequences", "rag_search"],
        "custom_analysis": ["identify_sequences", "sandbox_exec"],
    },
    "brain": {
        "full_pipeline": ["identify_sequences", "brats_mri_segmentation", "generate_report"],
        "segment": ["identify_sequences", "brats_mri_segmentation"],
        "classify": ["identify_sequences", "brats_mri_segmentation", "extract_roi_features", "classify_brain_glioma_grade"],
        "report": ["generate_report"],
        "qa": ["identify_sequences", "rag_search"],
        "custom_analysis": ["identify_sequences", "sandbox_exec"],
    },
    "cardiac": {
        "full_pipeline": [
            "identify_sequences",
            "segment_cardiac_cine",
            "classify_cardiac_cine_disease",
            "generate_report",
        ],
        "segment": ["identify_sequences", "segment_cardiac_cine"],
        "classify": ["segment_cardiac_cine", "classify_cardiac_cine_disease"],
        "report": ["generate_report"],
        "qa": ["identify_sequences", "rag_search"],
        "custom_analysis": ["identify_sequences", "sandbox_exec"],
    },
}


def _infer_tool_domains(name: str) -> List[str]:
    n = str(name or "").strip()
    low = n.lower()
    if not n:
        return []
    if low.startswith("dummy_"):
        return ["demo"]

    if n in {"identify_sequences", "extract_roi_features", "package_vlm_evidence", "generate_report"}:
        return list(_BASE_DOMAINS)
    if n in {"rag_search", "search_repo"}:
        return list(_BASE_DOMAINS)
    if n in {"sandbox_exec"}:
        return list(_BASE_DOMAINS)
    if n in {"register_to_reference", "alignment_qc", "materialize_registration"}:
        return ["prostate", "brain"]
    if n in {"segment_prostate", "detect_lesion_candidates", "correct_prostate_distortion"}:
        return ["prostate"]
    if n in {"brats_mri_segmentation", "classify_brain_glioma_grade"}:
        return ["brain"]
    if n in {"segment_cardiac_cine", "classify_cardiac_cine_disease"}:
        return ["cardiac"]

    # Generic fallback for unknown tools.
    if "cardiac" in low:
        return ["cardiac"]
    if "brain" in low or "brats" in low or "tumor" in low:
        return ["brain"]
    if "prostate" in low or "lesion" in low:
        return ["prostate"]
    return []


def _infer_tool_capabilities(name: str) -> List[str]:
    n = str(name or "").strip().lower()
    if not n:
        return []
    if n.startswith("dummy_load"):
        return ["identify"]
    if n.startswith("dummy_segment"):
        return ["segment"]
    if n.startswith("dummy_generate_report"):
        return ["report"]
    out: List[str] = []
    if "ingest" in n or "identify" in n:
        out.append("identify")
    if "register" in n or "alignment" in n or "materialize" in n:
        out.append("register")
    if "segment" in n:
        out.append("segment")
    if "classif" in n:
        out.append("classify")
    if "lesion" in n or "candidate" in n:
        out.append("lesion")
    if "feature" in n:
        out.append("extract")
    if "rag" in n or "search_repo" in n:
        out.append("qa")
    if "sandbox" in n:
        out.append("custom_analysis")
    if "report" in n or "evidence" in n or "package_vlm" in n:
        out.append("report")
    return sorted(set(out))


def build_shell_registry(*, dry_run: bool = False, include_core: bool | None = None) -> ToolRegistry:
    """
    Build the tool registry for shell execution.

    - dry_run=True defaults to dummy-only tools for lightweight end-to-end demos.
    - include_core=True forces registering real BCER core tools too.
    """
    if include_core is None:
        include_core = not dry_run

    if include_core:
        from agent.loop import build_registry as build_core_registry  # noqa: E402

        reg = build_core_registry()
    else:
        reg = ToolRegistry()

    if dry_run:
        for tool in build_dummy_tools():
            try:
                reg.register(tool)
            except Exception:
                # Ignore duplicate names if someone enables both modes manually.
                pass

    return reg


def list_tool_specs(registry: ToolRegistry) -> List[Dict[str, Any]]:
    return list(registry.list_specs())


def list_tool_names(registry: ToolRegistry) -> List[str]:
    out: List[str] = []
    for spec in registry.list_specs():
        if isinstance(spec, dict) and spec.get("name"):
            out.append(str(spec["name"]))
    return sorted(out)


def list_tool_metadata(registry: ToolRegistry) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for spec in registry.list_specs():
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name") or "").strip()
        if not name:
            continue
        out.append(
            {
                "name": name,
                "description": str(spec.get("description") or ""),
                "domains": _infer_tool_domains(name),
                "capabilities": _infer_tool_capabilities(name),
                "required_args": list((spec.get("input_schema") or {}).get("required") or []),
                "tags": list(spec.get("tags") or []),
            }
        )
    return sorted(out, key=lambda x: str(x.get("name") or ""))


def discover_domain_catalog(registry: ToolRegistry) -> Dict[str, Dict[str, Any]]:
    names = set(list_tool_names(registry))
    meta = list_tool_metadata(registry)

    tools_by_domain: Dict[str, List[str]] = {}
    for rec in meta:
        nm = str(rec.get("name") or "")
        for dom in rec.get("domains") or []:
            tools_by_domain.setdefault(str(dom), []).append(nm)

    out: Dict[str, Dict[str, Any]] = {}
    for domain, cap_map in _DOMAIN_CAPABILITY_REQUIREMENTS.items():
        caps: List[str] = []
        for cap, req_tools in cap_map.items():
            if all(t in names for t in req_tools):
                caps.append(cap)
        domain_tools = sorted(set(tools_by_domain.get(domain) or []))
        if caps or domain_tools:
            out[domain] = {
                "capabilities": sorted(set(caps)),
                "tool_count": len(domain_tools),
                "tools": domain_tools,
            }

    if "demo" in tools_by_domain:
        demo_tools = sorted(set(tools_by_domain["demo"]))
        out["demo"] = {
            "capabilities": ["full_pipeline", "segment", "report"],
            "tool_count": len(demo_tools),
            "tools": demo_tools,
        }

    return out

#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agent.langgraph.loop import (
    _build_brain_plan_nodes,
    _build_cardiac_plan_nodes,
    _build_custom_analysis_plan_nodes,
    _build_denoise_plan_nodes,
    _build_prostate_plan_nodes,
    _build_qa_plan_nodes,
    _build_raw_recon_plan_nodes,
    _build_super_resolution_plan_nodes,
)
from core.paths import project_root
from core.plan_dag import PlanNode


def _dump_nodes(nodes: List[PlanNode]) -> List[Dict[str, Any]]:
    return [n.model_dump(mode="json") for n in nodes]


def _write_template(
    *,
    out_dir: Path,
    filename: str,
    template_id: str,
    domain: str,
    request_type: str,
    notes: List[str],
    nodes: List[PlanNode],
) -> None:
    payload = {
        "template_id": template_id,
        "template_version": "1.0.0",
        "domain": domain,
        "request_type": request_type,
        "notes": [str(x) for x in (notes or []) if str(x).strip()],
        "nodes": _dump_nodes(nodes),
    }
    path = out_dir / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    out_dir = project_root() / "agent" / "plans" / "templates"
    out_dir.mkdir(parents=True, exist_ok=True)

    prostate_nodes, prostate_notes = _build_prostate_plan_nodes(
        intent={"wants_report": True, "wants_features": True, "wants_lesion": False},
        modalities={"T2w": True, "ADC": True, "DWI": True},
        optional_overrides={"extract_roi_features": False, "detect_lesion_candidates": False},
    )
    _write_template(
        out_dir=out_dir,
        filename="prostate_full_pipeline.json",
        template_id="prostate_full_pipeline",
        domain="prostate",
        request_type="full_pipeline",
        notes=prostate_notes,
        nodes=prostate_nodes,
    )

    brain_nodes, brain_notes = _build_brain_plan_nodes(
        intent={"wants_report": True, "wants_features": False, "wants_lesion": False},
        optional_overrides={"extract_roi_features": False},
    )
    _write_template(
        out_dir=out_dir,
        filename="brain_full_pipeline.json",
        template_id="brain_full_pipeline",
        domain="brain",
        request_type="full_pipeline",
        notes=brain_notes,
        nodes=brain_nodes,
    )

    cardiac_nodes, cardiac_notes = _build_cardiac_plan_nodes(
        intent={"wants_report": True, "wants_features": False, "wants_classification": True},
        optional_overrides={"extract_roi_features": False},
        case_ref=None,
        request_type="full_pipeline",
    )
    _write_template(
        out_dir=out_dir,
        filename="cardiac_full_pipeline.json",
        template_id="cardiac_full_pipeline",
        domain="cardiac",
        request_type="full_pipeline",
        notes=cardiac_notes,
        nodes=cardiac_nodes,
    )

    qa_nodes, qa_notes = _build_qa_plan_nodes(
        goal="metadata query",
        domain_name="prostate",
        llm_mode="stub",
        server_cfg=None,
        api_model=None,
        api_base_url=None,
    )
    _write_template(
        out_dir=out_dir,
        filename="qa.json",
        template_id="qa",
        domain="*",
        request_type="qa",
        notes=qa_notes,
        nodes=qa_nodes,
    )

    custom_nodes, custom_notes = _build_custom_analysis_plan_nodes(goal="run custom analysis", domain_name="prostate")
    _write_template(
        out_dir=out_dir,
        filename="custom_analysis.json",
        template_id="custom_analysis",
        domain="*",
        request_type="custom_analysis",
        notes=custom_notes,
        nodes=custom_nodes,
    )

    denoise_nodes, denoise_notes, _ = _build_denoise_plan_nodes(
        goal="run denoise workflow",
        case_scan={"nifti_files": [], "dicom_files": 1},
    )
    _write_template(
        out_dir=out_dir,
        filename="denoise.json",
        template_id="denoise",
        domain="*",
        request_type="denoise",
        notes=denoise_notes,
        nodes=denoise_nodes,
    )

    super_nodes, super_notes, _ = _build_super_resolution_plan_nodes(
        goal="run super-resolution workflow",
        case_scan={"nifti_files": [], "dicom_files": 1},
    )
    _write_template(
        out_dir=out_dir,
        filename="super_resolution.json",
        template_id="super_resolution",
        domain="*",
        request_type="super_resolution",
        notes=super_notes,
        nodes=super_nodes,
    )

    raw_nodes, raw_notes, _ = _build_raw_recon_plan_nodes(
        goal="/tmp/input.h5",
        domain_name="cardiac",
        case_ref=Path("."),
    )
    _write_template(
        out_dir=out_dir,
        filename="raw_recon.json",
        template_id="raw_recon",
        domain="cardiac",
        request_type="raw_recon",
        notes=raw_notes,
        nodes=raw_nodes,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .bootstrap import ensure_import_paths

ensure_import_paths()

from commands.registry import Tool  # noqa: E402
from commands.schemas import ArtifactRef, ToolContext, ToolSpec  # noqa: E402


DUMMY_LOAD_SPEC = ToolSpec(
    name="dummy_load_case",
    description="Dummy tool: register and summarize a case path.",
    input_schema={
        "type": "object",
        "properties": {
            "case_path": {"type": "string"},
            "output_subdir": {"type": "string"},
        },
        "required": ["case_path"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "case_path": {"type": "string"},
            "case_manifest_path": {"type": "string"},
        },
        "required": ["case_path", "case_manifest_path"],
    },
    version="0.1.0",
    tags=["dummy", "demo"],
)


DUMMY_SEG_SPEC = ToolSpec(
    name="dummy_segment",
    description="Dummy tool: simulate segmentation output.",
    input_schema={
        "type": "object",
        "properties": {
            "case_path": {"type": "string"},
            "anatomy": {"type": "string"},
            "output_subdir": {"type": "string"},
        },
        "required": ["case_path"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "mask_path": {"type": "string"},
            "dice_hint": {"type": "number"},
        },
        "required": ["mask_path"],
    },
    version="0.1.0",
    tags=["dummy", "demo"],
)


DUMMY_REPORT_SPEC = ToolSpec(
    name="dummy_generate_report",
    description="Dummy tool: simulate report markdown output.",
    input_schema={
        "type": "object",
        "properties": {
            "case_id": {"type": "string"},
            "mask_path": {"type": "string"},
            "output_subdir": {"type": "string"},
        },
        "required": ["case_id"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "report_txt_path": {"type": "string"},
            "report_json_path": {"type": "string"},
        },
        "required": ["report_txt_path", "report_json_path"],
    },
    version="0.1.0",
    tags=["dummy", "demo"],
)


def _dummy_load_case(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    case_path = str(args.get("case_path") or "")
    out_subdir = str(args.get("output_subdir") or "dummy/load")
    out_dir = ctx.artifacts_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "case_path": case_path,
        "case_id": ctx.case_id,
        "note": "dummy load case",
    }
    manifest_path = out_dir / "case_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    artifacts = [
        ArtifactRef(path=str(manifest_path), kind="json", description="Dummy case manifest"),
    ]
    return {
        "data": {
            "case_path": case_path,
            "case_manifest_path": str(manifest_path),
        },
        "artifacts": artifacts,
        "generated_artifacts": artifacts,
        "warnings": [],
    }


def _dummy_segment(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    out_subdir = str(args.get("output_subdir") or "dummy/segmentation")
    out_dir = ctx.artifacts_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    anatomy = str(args.get("anatomy") or "unknown")
    mask_path = out_dir / "dummy_mask.nii.gz"
    mask_path.write_text("dummy mask bytes\n", encoding="utf-8")

    qc_path = out_dir / "seg_qc.json"
    qc_path.write_text(json.dumps({"dice_hint": 0.91, "anatomy": anatomy}, indent=2) + "\n", encoding="utf-8")

    artifacts = [
        ArtifactRef(path=str(mask_path), kind="nifti", description="Dummy segmentation mask"),
        ArtifactRef(path=str(qc_path), kind="json", description="Dummy segmentation qc"),
    ]
    return {
        "data": {
            "mask_path": str(mask_path),
            "dice_hint": 0.91,
        },
        "artifacts": artifacts,
        "generated_artifacts": artifacts,
        "warnings": [],
    }


def _dummy_generate_report(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    out_subdir = str(args.get("output_subdir") or "dummy/report")
    out_dir = ctx.artifacts_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    case_id = str(args.get("case_id") or ctx.case_id)
    mask_path = str(args.get("mask_path") or "")

    report_md = out_dir / "report.md"
    report_json = out_dir / "report.json"

    lines = [
        "# Dummy Report",
        "",
        f"- case_id: `{case_id}`",
        f"- mask: `{mask_path}`",
        "- note: dry-run pipeline executed.",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "case_id": case_id,
        "mask_path": mask_path,
        "status": "ok",
    }
    report_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    artifacts = [
        ArtifactRef(path=str(report_md), kind="text", description="Dummy markdown report"),
        ArtifactRef(path=str(report_json), kind="json", description="Dummy structured report"),
    ]
    return {
        "data": {
            "report_txt_path": str(report_md),
            "report_json_path": str(report_json),
        },
        "artifacts": artifacts,
        "generated_artifacts": artifacts,
        "warnings": [],
    }


def build_dummy_tools() -> List[Tool]:
    return [
        Tool(spec=DUMMY_LOAD_SPEC, func=_dummy_load_case),
        Tool(spec=DUMMY_SEG_SPEC, func=_dummy_segment),
        Tool(spec=DUMMY_REPORT_SPEC, func=_dummy_generate_report),
    ]

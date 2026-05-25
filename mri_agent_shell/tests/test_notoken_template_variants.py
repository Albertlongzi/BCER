from __future__ import annotations

import pytest

from agent.langgraph.loop import _inject_explicit_input_aliases
from agent.plans.template_loader import load_plan_template
from core.plan_dag import PlanNode


def test_load_plan_template_notoken_variant_for_brain_segment_uses_notoken_file() -> None:
    tmpl = load_plan_template(domain="brain", request_type="segment", variant="notoken")
    assert tmpl.template_id == "brain_full_pipeline_notoken"
    assert str(tmpl.source_path).endswith("brain_full_pipeline_notoken.json")


def test_load_plan_template_notoken_variant_unsupported_for_denoise() -> None:
    with pytest.raises(FileNotFoundError):
        load_plan_template(domain="brain", request_type="denoise", variant="notoken")


def test_inject_explicit_input_aliases_replaces_placeholder_and_reports_missing() -> None:
    nodes = [
        PlanNode(
            node_id="seg_001",
            tool_name="brats_mri_segmentation",
            stage="segment",
            arguments={
                "t1c_path": "${alias:t1c_nifti}",
                "t1_path": "${alias:t1_nifti}",
                "t2_path": "${alias:t2_nifti}",
                "flair_path": "${alias:flair_nifti}",
            },
            required=True,
            depends_on=[],
        )
    ]
    out, notes, missing = _inject_explicit_input_aliases(
        nodes=nodes,
        explicit_input_aliases={
            "t1c_nifti": "/tmp/T1c.nii.gz",
            "t1_nifti": "/tmp/T1.nii.gz",
            "t2_nifti": "/tmp/T2.nii.gz",
        },
    )
    args = dict(out[0].arguments or {})
    assert args["t1c_path"] == "/tmp/T1c.nii.gz"
    assert args["flair_path"] == "${alias:flair_nifti}"
    assert "flair_nifti" in missing
    assert any("Injected explicit input aliases" in n for n in notes)
    assert any("Missing explicit input aliases" in n for n in notes)

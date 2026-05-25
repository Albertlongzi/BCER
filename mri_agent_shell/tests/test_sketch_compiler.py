from __future__ import annotations

from agent.plans.sketch_compiler import compile_constrained_plan_sketch
from agent.plans.sketch_schema import ConstrainedPlanSketch
from core.plan_dag import PlanNode


def _mk_blueprints() -> list[PlanNode]:
    return [
        PlanNode(
            node_id="identify_sequences_001",
            tool_name="identify_sequences",
            stage="identify",
            arguments={"dicom_case_dir": "@case.input"},
            required=True,
            depends_on=[],
        ),
        PlanNode(
            node_id="segment_brats_030",
            tool_name="brats_mri_segmentation",
            stage="segment",
            arguments={
                "t1_path": "T1",
                "t1c_path": "T1c",
                "t2_path": "T2",
                "flair_path": "FLAIR",
            },
            required=True,
            depends_on=["identify_sequences_001"],
        ),
        PlanNode(
            node_id="package_evidence_060",
            tool_name="package_vlm_evidence",
            stage="package",
            arguments={"case_state_path": "@runtime.case_state_path"},
            required=True,
            depends_on=["segment_brats_030"],
        ),
        PlanNode(
            node_id="generate_report_070",
            tool_name="generate_report",
            stage="report",
            arguments={"case_state_path": "@runtime.case_state_path", "domain": "brain"},
            required=True,
            depends_on=["package_evidence_060"],
        ),
    ]


def test_sketch_compiler_overlays_inputs_and_autofills_required_closure() -> None:
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "long_brain_report",
            "domain": "brain",
            "steps": [
                {
                    "step_id": "id1",
                    "tool": "identify_sequences",
                    "depends_on": [],
                    "inputs": {"dicom_case_dir": "@case.input"},
                    "goal": "map sequences",
                    "optional": False,
                },
                {
                    "step_id": "seg1",
                    "tool": "brats_mri_segmentation",
                    "depends_on": ["id1"],
                    "inputs": {"t1c_path": "@seq.T1c", "flair_path": "@seq.FLAIR"},
                    "goal": "segment tumor",
                    "optional": False,
                },
                {
                    "step_id": "rep1",
                    "tool": "generate_report",
                    "depends_on": ["seg1"],
                    "inputs": {"case_state_path": "@runtime.case_state_path", "domain": "brain"},
                    "goal": "report",
                    "optional": False,
                },
            ],
            "final_targets": ["rep1"],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=_mk_blueprints(),
        allowed_tools=[
            "identify_sequences",
            "brats_mri_segmentation",
            "package_vlm_evidence",
            "generate_report",
        ],
    )
    assert res.ok is True, res.diagnostics
    by_tool = {n.tool_name: n for n in res.nodes}
    assert "package_vlm_evidence" in by_tool  # compiler autofill for report closure
    seg = by_tool["brats_mri_segmentation"]
    assert seg.arguments["t1c_path"] == "@seq.T1c"
    assert seg.arguments["flair_path"] == "@seq.FLAIR"
    rep = by_tool["generate_report"]
    assert "package_evidence_060" in rep.depends_on  # blueprint dep preserved
    assert any(
        isinstance(x, dict) and x.get("tool") == "package_vlm_evidence"
        for x in (res.diagnostics.get("autofilled_nodes") or [])
    )


def test_sketch_compiler_rejects_illegal_tool() -> None:
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "bad",
            "domain": "brain",
            "steps": [
                {
                    "step_id": "x",
                    "tool": "sandbox_exec",
                    "depends_on": [],
                    "inputs": {},
                    "goal": "illegal",
                    "optional": False,
                }
            ],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=_mk_blueprints(),
        allowed_tools=["identify_sequences", "brats_mri_segmentation"],
    )
    assert res.ok is False
    errs = " | ".join(str(x) for x in (res.diagnostics.get("errors") or []))
    assert "Illegal tool" in errs


def test_sketch_compiler_rejects_dependency_cycle() -> None:
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "cycle",
            "domain": "brain",
            "steps": [
                {
                    "step_id": "a",
                    "tool": "identify_sequences",
                    "depends_on": ["b"],
                    "inputs": {},
                    "goal": "a",
                    "optional": False,
                },
                {
                    "step_id": "b",
                    "tool": "brats_mri_segmentation",
                    "depends_on": ["a"],
                    "inputs": {},
                    "goal": "b",
                    "optional": False,
                },
            ],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=_mk_blueprints(),
        allowed_tools=["identify_sequences", "brats_mri_segmentation"],
    )
    assert res.ok is False
    errs = " | ".join(str(x) for x in (res.diagnostics.get("errors") or []))
    assert "cycle" in errs.lower()


def test_sketch_compiler_rewrites_sketch_node_refs_to_compiled_node_ids() -> None:
    blueprints = [
        PlanNode(
            node_id="identify_sequences_001",
            tool_name="identify_sequences",
            stage="identify",
            arguments={"dicom_case_dir": "@case.input"},
            required=True,
            depends_on=[],
        ),
        PlanNode(
            node_id="segment_prostate_030",
            tool_name="segment_prostate",
            stage="segment",
            arguments={"t2w_ref": "T2w"},
            required=True,
            depends_on=["identify_sequences_001"],
        ),
        PlanNode(
            node_id="detect_lesions_050",
            tool_name="detect_lesion_candidates",
            stage="lesion",
            arguments={"output_subdir": "lesion"},
            required=True,
            depends_on=["segment_prostate_030"],
        ),
    ]
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "long_prostate_full",
            "domain": "prostate",
            "steps": [
                {
                    "step_id": "id1",
                    "tool": "identify_sequences",
                    "depends_on": [],
                    "inputs": {"dicom_case_dir": "@case.input"},
                    "goal": "identify",
                    "optional": False,
                },
                {
                    "step_id": "seg1",
                    "tool": "segment_prostate",
                    "depends_on": ["id1"],
                    "inputs": {"t2w_ref": "@seq.T2w"},
                    "goal": "segment",
                    "optional": False,
                },
                {
                    "step_id": "les1",
                    "tool": "detect_lesion_candidates",
                    "depends_on": ["seg1"],
                    "inputs": {
                        "t2w_nifti": "@node.seg1.t2w_input_path",
                        "prostate_mask_nifti": "@node.seg1.prostate_mask_path",
                    },
                    "goal": "lesion detect",
                    "optional": False,
                },
            ],
            "final_targets": ["les1"],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=blueprints,
        allowed_tools=["identify_sequences", "segment_prostate", "detect_lesion_candidates"],
    )
    assert res.ok is True, res.diagnostics
    lesion = next(n for n in res.nodes if n.tool_name == "detect_lesion_candidates")
    assert lesion.arguments["t2w_nifti"] == "@node.segment_prostate_030.t2w_input_path"
    assert lesion.arguments["prostate_mask_nifti"] == "@node.segment_prostate_030.prostate_mask_path"
    stats = res.diagnostics.get("rewrite_stats") or {}
    assert int(stats.get("node_ref_rewrites") or 0) >= 2


def test_sketch_compiler_canonicalizes_case_ref_token_for_raw_recon_file_case() -> None:
    blueprints = [
        PlanNode(
            node_id="reconstruct_grappa_010",
            tool_name="reconstruct_grappa",
            stage="reconstruct",
            arguments={"h5_path": "", "output_subdir": "grappa"},
            required=True,
            depends_on=[],
        )
    ]
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "short_recon_grappa",
            "domain": "cardiac",
            "steps": [
                {
                    "step_id": "recon1",
                    "tool": "reconstruct_grappa",
                    "depends_on": [],
                    "inputs": {"h5_path": "@case_ref"},
                    "goal": "recon",
                    "optional": False,
                }
            ],
            "final_targets": ["recon1"],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=blueprints,
        allowed_tools=["reconstruct_grappa"],
        case_ref_is_file=True,
    )
    assert res.ok is True, res.diagnostics
    node = res.nodes[0]
    assert node.arguments["h5_path"] == "@case.file"
    stats = res.diagnostics.get("rewrite_stats") or {}
    assert int(stats.get("case_ref_token_rewrites") or 0) >= 1


def test_sketch_compiler_canonicalizes_output_field_alias_and_drops_unknown_override() -> None:
    blueprints = [
        PlanNode(
            node_id="identify_sequences_001",
            tool_name="identify_sequences",
            stage="identify",
            arguments={"dicom_case_dir": "@case.input"},
            required=True,
            depends_on=[],
        ),
        PlanNode(
            node_id="segment_cine_020",
            tool_name="segment_cardiac_cine",
            stage="segment",
            arguments={"cine_path": "CINE", "output_subdir": "seg/cardiac"},
            required=True,
            depends_on=["identify_sequences_001"],
        ),
        PlanNode(
            node_id="denoise_bm3d_010",
            tool_name="denoise_image_bm3d",
            stage="denoise",
            arguments={"input_nifti": "@seq.T2", "output_subdir": "denoise"},
            required=True,
            depends_on=[],
        ),
        PlanNode(
            node_id="compare_denoise_020",
            tool_name="compare_nifti_slices",
            stage="qa",
            arguments={
                "image_a": "@seq.T2",
                "image_b": "@node.denoise_bm3d_010.denoised_nifti",
                "output_subdir": "compare",
            },
            required=False,
            depends_on=["denoise_bm3d_010"],
        ),
        PlanNode(
            node_id="extract_features_040",
            tool_name="extract_roi_features",
            stage="extract",
            arguments={"output_subdir": "features", "roi_interpretation": "cardiac"},
            required=True,
            depends_on=["segment_cine_020"],
        ),
    ]
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "mix",
            "domain": "cardiac",
            "steps": [
                {
                    "step_id": "id1",
                    "tool": "identify_sequences",
                    "depends_on": [],
                    "inputs": {"dicom_case_dir": "@case.input"},
                    "goal": "identify",
                    "optional": False,
                },
                {
                    "step_id": "seg1",
                    "tool": "segment_cardiac_cine",
                    "depends_on": ["id1"],
                    "inputs": {"cine_path": "@node.id1.cine_path"},
                    "goal": "segment cine",
                    "optional": False,
                },
                {
                    "step_id": "den1",
                    "tool": "denoise_image_bm3d",
                    "depends_on": [],
                    "inputs": {"input_nifti": "@seq.T2"},
                    "goal": "denoise",
                    "optional": False,
                },
                {
                    "step_id": "cmp1",
                    "tool": "compare_nifti_slices",
                    "depends_on": ["den1"],
                    "inputs": {
                        "image_a": "@seq.T2",
                        "image_b": "@node.den1.output_nifti",
                    },
                    "goal": "qa compare",
                    "optional": False,
                },
                {
                    "step_id": "feat1",
                    "tool": "extract_roi_features",
                    "depends_on": ["seg1"],
                    "inputs": {
                        "images": "@node.seg1.cine_path",
                        "roi_masks": "@node.seg1.segmentation_path",
                    },
                    "goal": "features",
                    "optional": False,
                },
            ],
            "final_targets": ["feat1", "cmp1"],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=blueprints,
        allowed_tools=[
            "identify_sequences",
            "segment_cardiac_cine",
            "denoise_image_bm3d",
            "compare_nifti_slices",
            "extract_roi_features",
        ],
    )
    assert res.ok is True, res.diagnostics
    by_id = {n.node_id: n for n in res.nodes}
    seg = by_id["segment_cine_020"]
    # identify_sequences.cine_path -> @seq.CINE canonicalization
    assert seg.arguments["cine_path"] == "@seq.CINE"
    cmp_node = by_id["compare_denoise_020"]
    # output_nifti alias -> denoised_nifti
    assert cmp_node.arguments["image_b"] == "@node.denoise_bm3d_010.denoised_nifti"
    feat = by_id["extract_features_040"]
    # unknown segment_cardiac_cine.cine_path override dropped to preserve compiler/autorepair path
    assert "images" not in feat.arguments
    # segmentation_path alias -> seg_path
    assert feat.arguments["roi_masks"] == "@node.segment_cine_020.seg_path"
    stats = res.diagnostics.get("rewrite_stats") or {}
    assert int(stats.get("node_output_field_alias_rewrites") or 0) >= 2
    assert int(stats.get("unknown_node_output_field_drops") or 0) >= 1


def test_sketch_compiler_preserves_blueprint_optional_semantics() -> None:
    blueprints = [
        PlanNode(
            node_id="denoise_bm3d_010",
            tool_name="denoise_image_bm3d",
            stage="denoise",
            arguments={"input_nifti": "@seq.T2", "output_subdir": "denoise"},
            required=True,
            depends_on=[],
        ),
        PlanNode(
            node_id="compare_denoise_020",
            tool_name="compare_nifti_slices",
            stage="qa",
            arguments={"image_a": "@seq.T2", "image_b": "@node.denoise_bm3d_010.denoised_nifti"},
            required=False,
            depends_on=["denoise_bm3d_010"],
        ),
    ]
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "short_denoise",
            "domain": "brain",
            "steps": [
                {
                    "step_id": "den1",
                    "tool": "denoise_image_bm3d",
                    "depends_on": [],
                    "inputs": {"input_nifti": "@seq.T2"},
                    "goal": "denoise",
                    "optional": False,
                },
                {
                    "step_id": "cmp1",
                    "tool": "compare_nifti_slices",
                    "depends_on": ["den1"],
                    "inputs": {"image_b": "@node.den1.output_nifti"},
                    "goal": "compare",
                    "optional": False,  # LLM forgot to mark optional
                },
            ],
            "final_targets": ["den1"],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=blueprints,
        allowed_tools=["denoise_image_bm3d", "compare_nifti_slices"],
    )
    assert res.ok is True, res.diagnostics
    by_id = {n.node_id: n for n in res.nodes}
    assert by_id["denoise_bm3d_010"].required is True
    assert by_id["compare_denoise_020"].required is False
    warns = " | ".join(str(x) for x in (res.diagnostics.get("warnings") or []))
    assert "preserving blueprint required=False" in warns


def test_sketch_compiler_canonicalizes_superres_spacing_and_drops_extra_reference() -> None:
    blueprints = [
        PlanNode(
            node_id="resample_sr_010",
            tool_name="resample_image",
            stage="super_resolution",
            arguments={
                "input_nifti": "@auto",
                "target_spacing": [0.5, 0.5, 0.5],
                "interpolation": "bspline",
                "output_subdir": "resample",
            },
            required=True,
            depends_on=[],
        )
    ]
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "short_superres",
            "domain": "brain",
            "steps": [
                {
                    "step_id": "sr1",
                    "tool": "resample_image",
                    "depends_on": [],
                    "inputs": {
                        "input_nifti": "@seq.T2w",
                        "target_spacing": "[1,1,1]",
                        "reference_nifti": "@seq.T1c",
                    },
                    "goal": "resample",
                    "optional": False,
                }
            ],
            "final_targets": ["sr1"],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=blueprints,
        allowed_tools=["resample_image"],
    )
    assert res.ok is True, res.diagnostics
    node = res.nodes[0]
    assert node.arguments["target_spacing"] == [1.0, 1.0, 1.0]
    assert "reference_nifti" not in node.arguments
    stats = res.diagnostics.get("rewrite_stats") or {}
    assert int(stats.get("resample_target_spacing_string_to_array") or 0) >= 1
    assert int(stats.get("resample_reference_override_drops") or 0) >= 1


def test_sketch_compiler_canonicalizes_superres_spacing_comma_string() -> None:
    blueprints = [
        PlanNode(
            node_id="resample_sr_010",
            tool_name="resample_image",
            stage="super_resolution",
            arguments={
                "input_nifti": "@auto",
                "target_spacing": [0.5, 0.5, 0.5],
            },
            required=True,
            depends_on=[],
        )
    ]
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "short_superres",
            "domain": "brain",
            "steps": [
                {
                    "step_id": "sr1",
                    "tool": "resample_image",
                    "depends_on": [],
                    "inputs": {
                        "input_nifti": "@seq.T1",
                        "target_spacing": "1.0,1.0,1.0",
                    },
                    "goal": "resample",
                    "optional": False,
                }
            ],
            "final_targets": ["sr1"],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=blueprints,
        allowed_tools=["resample_image"],
    )
    assert res.ok is True, res.diagnostics
    assert res.nodes[0].arguments["target_spacing"] == [1.0, 1.0, 1.0]


def test_sketch_compiler_drops_resample_output_nifti_runtime_case_state_composite() -> None:
    blueprints = [
        PlanNode(
            node_id="resample_sr_010",
            tool_name="resample_image",
            stage="super_resolution",
            arguments={
                "input_nifti": "@auto",
                "target_spacing": [1.0, 1.0, 1.0],
            },
            required=True,
            depends_on=[],
        )
    ]
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "short_superres",
            "domain": "prostate",
            "steps": [
                {
                    "step_id": "sr1",
                    "tool": "resample_image",
                    "depends_on": [],
                    "inputs": {
                        "input_nifti": "@seq.T2w",
                        "output_nifti": "@runtime.case_state_path/sr_resampled/out.nii.gz",
                    },
                    "goal": "resample t2w",
                    "optional": False,
                }
            ],
            "final_targets": ["sr1"],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=blueprints,
        allowed_tools=["resample_image"],
    )
    assert res.ok is True, res.diagnostics
    assert "output_nifti" not in (res.nodes[0].arguments or {})
    stats = res.diagnostics.get("rewrite_stats") or {}
    assert int(stats.get("resample_output_nifti_runtime_case_state_drops") or 0) >= 1


def test_sketch_schema_ignores_extra_top_level_fields() -> None:
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "short_recon_grappa",
            "domain": "cardiac",
            "request_type": "raw_recon",  # harmless extra from some models
            "steps": [
                {
                    "step_id": "recon1",
                    "tool": "reconstruct_grappa",
                    "depends_on": [],
                    "inputs": {"h5_path": "@case.file"},
                    "goal": "recon",
                    "optional": False,
                }
            ],
        }
    )
    assert sketch.task == "short_recon_grappa"
    assert sketch.domain == "cardiac"
    assert len(sketch.steps) == 1


def test_sketch_compiler_canonicalizes_segment_prostate_output_field_aliases() -> None:
    blueprints = [
        PlanNode(
            node_id="segment_prostate_030",
            tool_name="segment_prostate",
            stage="segment",
            arguments={"t2w_ref": "@seq.T2w"},
            required=True,
            depends_on=[],
        ),
        PlanNode(
            node_id="extract_features_040",
            tool_name="extract_roi_features",
            stage="extract",
            arguments={"output_subdir": "features"},
            required=True,
            depends_on=["segment_prostate_030"],
        ),
    ]
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "long_prostate_full",
            "domain": "prostate",
            "steps": [
                {
                    "step_id": "seg1",
                    "tool": "segment_prostate",
                    "depends_on": [],
                    "inputs": {"t2w_ref": "@seq.T2w"},
                    "goal": "segment",
                    "optional": False,
                },
                {
                    "step_id": "feat1",
                    "tool": "extract_roi_features",
                    "depends_on": ["seg1"],
                    "inputs": {
                        "context_mask_path": "@node.seg1.whole_gland_mask_path",
                        "roi_masks": {"zones": "@node.seg1.zonal_mask_path"},
                    },
                    "goal": "features",
                    "optional": False,
                },
            ],
            "final_targets": ["feat1"],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=blueprints,
        allowed_tools=["segment_prostate", "extract_roi_features"],
    )
    assert res.ok is True, res.diagnostics
    feat = next(n for n in res.nodes if n.tool_name == "extract_roi_features")
    assert feat.arguments["context_mask_path"] == "@node.segment_prostate_030.prostate_mask_path"
    assert feat.arguments["roi_masks"]["zones"] == "@node.segment_prostate_030.zone_mask_path"
    stats = res.diagnostics.get("rewrite_stats") or {}
    assert int(stats.get("node_output_field_alias_rewrites") or 0) >= 2


def test_sketch_compiler_drops_self_referential_output_override() -> None:
    blueprints = [
        PlanNode(
            node_id="denoise_bm3d_010",
            tool_name="denoise_image_bm3d",
            stage="denoise",
            arguments={"input_nifti": "@seq.T2", "output_subdir": "denoise"},
            required=True,
            depends_on=[],
        )
    ]
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "short_denoise",
            "domain": "brain",
            "steps": [
                {
                    "step_id": "den1",
                    "tool": "denoise_image_bm3d",
                    "depends_on": [],
                    "inputs": {
                        "input_nifti": "@seq.T2",
                        "output_nifti": "@node.den1.output_nifti",
                    },
                    "goal": "denoise",
                    "optional": False,
                }
            ],
            "final_targets": ["den1"],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=blueprints,
        allowed_tools=["denoise_image_bm3d"],
    )
    assert res.ok is True, res.diagnostics
    node = res.nodes[0]
    assert "output_nifti" not in (node.arguments or {})
    stats = res.diagnostics.get("rewrite_stats") or {}
    assert int(stats.get("self_output_token_override_drops") or 0) >= 1


def test_sketch_compiler_drops_optional_duplicate_step_exceeding_blueprint_multiplicity() -> None:
    blueprints = [
        PlanNode(
            node_id="reconstruct_grappa_010",
            tool_name="reconstruct_grappa",
            stage="reconstruct",
            arguments={"h5_path": "@case.file", "output_subdir": "recon"},
            required=True,
            depends_on=[],
        ),
        PlanNode(
            node_id="qa_snapshot_020",
            tool_name="generate_qa_snapshot",
            stage="qa",
            arguments={"input_nifti": "@node.reconstruct_grappa_010.reconstructed_nifti"},
            required=False,
            depends_on=["reconstruct_grappa_010"],
        ),
    ]
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "short_recon_grappa",
            "domain": "cardiac",
            "steps": [
                {
                    "step_id": "recon1",
                    "tool": "reconstruct_grappa",
                    "depends_on": [],
                    "inputs": {"h5_path": "@case.file"},
                    "goal": "primary recon",
                    "optional": False,
                },
                {
                    "step_id": "recon_alt",
                    "tool": "reconstruct_grappa",
                    "depends_on": [],
                    "inputs": {"h5_path": "@case.file", "kernel_size": [4, 5]},
                    "goal": "alternate recon",
                    "optional": True,
                },
                {
                    "step_id": "qa1",
                    "tool": "generate_qa_snapshot",
                    "depends_on": ["recon1"],
                    "inputs": {"input_nifti": "@node.recon1.output_nifti"},
                    "goal": "qa",
                    "optional": False,
                },
            ],
            "final_targets": ["recon1", "qa1"],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=blueprints,
        allowed_tools=["reconstruct_grappa", "generate_qa_snapshot"],
        case_ref_is_file=True,
    )
    assert res.ok is True, res.diagnostics
    assert len([n for n in res.nodes if n.tool_name == "reconstruct_grappa"]) == 1
    dropped = res.diagnostics.get("dropped_optional_steps") or []
    assert any(isinstance(x, dict) and x.get("sketch_step_id") == "recon_alt" for x in dropped)
    warns = " | ".join(str(x) for x in (res.diagnostics.get("warnings") or []))
    assert "exceeds blueprint multiplicity" in warns


def test_sketch_compiler_drops_required_to_optional_sketch_dependency_to_avoid_skip_block() -> None:
    blueprints = [
        PlanNode(
            node_id="denoise_bm3d_010",
            tool_name="denoise_image_bm3d",
            stage="denoise",
            arguments={"input_nifti": "@seq.T2"},
            required=True,
            depends_on=[],
        ),
        PlanNode(
            node_id="compare_denoise_020",
            tool_name="compare_nifti_slices",
            stage="qa",
            arguments={"image_a": "@seq.T2", "image_b": "@node.denoise_bm3d_010.denoised_nifti"},
            required=False,
            depends_on=["denoise_bm3d_010"],
        ),
        PlanNode(
            node_id="package_evidence_060",
            tool_name="package_vlm_evidence",
            stage="package",
            arguments={"case_state_path": "@runtime.case_state_path"},
            required=True,
            depends_on=["denoise_bm3d_010"],
        ),
    ]
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "demo",
            "domain": "brain",
            "steps": [
                {
                    "step_id": "den1",
                    "tool": "denoise_image_bm3d",
                    "depends_on": [],
                    "inputs": {"input_nifti": "@seq.T2"},
                    "goal": "denoise",
                    "optional": False,
                },
                {
                    "step_id": "qa1",
                    "tool": "compare_nifti_slices",
                    "depends_on": ["den1"],
                    "inputs": {"image_b": "@node.den1.output_nifti"},
                    "goal": "qa",
                    "optional": True,
                },
                {
                    "step_id": "pkg1",
                    "tool": "package_vlm_evidence",
                    "depends_on": ["den1", "qa1"],  # risky: required depends on optional
                    "inputs": {"case_state_path": "@runtime.case_state_path"},
                    "goal": "package",
                    "optional": False,
                },
            ],
            "final_targets": ["pkg1"],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=blueprints,
        allowed_tools=["denoise_image_bm3d", "compare_nifti_slices", "package_vlm_evidence"],
    )
    assert res.ok is True, res.diagnostics
    by_id = {n.node_id: n for n in res.nodes}
    pkg = by_id["package_evidence_060"]
    assert "compare_denoise_020" not in (pkg.depends_on or [])
    assert "denoise_bm3d_010" in (pkg.depends_on or [])
    warns = " | ".join(str(x) for x in (res.diagnostics.get("warnings") or []))
    assert "dependency node is optional" in warns


def test_sketch_compiler_normalizes_extract_roi_features_radiomics_bool() -> None:
    blueprints = [
        PlanNode(
            node_id="extract_features_040",
            tool_name="extract_roi_features",
            stage="extract",
            arguments={"images": ["@seq.T2"], "roi_mask_path": "@seq.T2"},
            required=True,
            depends_on=[],
        )
    ]
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "long_prostate_full",
            "domain": "prostate",
            "steps": [
                {
                    "step_id": "feat1",
                    "tool": "extract_roi_features",
                    "depends_on": [],
                    "inputs": {"radiomics": True},
                    "goal": "features",
                    "optional": False,
                }
            ],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=blueprints,
        allowed_tools=["extract_roi_features"],
    )
    assert res.ok is True, res.diagnostics
    feat = res.nodes[0]
    assert feat.arguments["radiomics"] == {"enabled": True}
    stats = res.diagnostics.get("rewrite_stats") or {}
    assert int(stats.get("extract_radiomics_bool_to_object") or 0) >= 1


def test_sketch_compiler_drops_invalid_nested_node_output_refs_in_extract_images() -> None:
    blueprints = [
        PlanNode(
            node_id="reconstruct_grappa_005",
            tool_name="reconstruct_grappa",
            stage="reconstruct",
            arguments={"h5_path": "@case.file"},
            required=True,
            depends_on=[],
        ),
        PlanNode(
            node_id="extract_features_040",
            tool_name="extract_roi_features",
            stage="extract",
            arguments={"images": ["@node.reconstruct_grappa_005.reconstructed_nifti"]},
            required=True,
            depends_on=["reconstruct_grappa_005"],
        ),
    ]
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "long_cardiac_full",
            "domain": "cardiac",
            "steps": [
                {
                    "step_id": "rec1",
                    "tool": "reconstruct_grappa",
                    "depends_on": [],
                    "inputs": {"h5_path": "@case.file"},
                    "goal": "recon",
                    "optional": False,
                },
                {
                    "step_id": "feat1",
                    "tool": "extract_roi_features",
                    "depends_on": ["rec1"],
                    "inputs": {"images": ["@node.rec1.cine_nifti_path"]},
                    "goal": "features",
                    "optional": False,
                },
            ],
            "final_targets": ["feat1"],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=blueprints,
        allowed_tools=["reconstruct_grappa", "extract_roi_features"],
        case_ref_is_file=True,
    )
    assert res.ok is True, res.diagnostics
    feat = next(n for n in res.nodes if n.tool_name == "extract_roi_features")
    assert feat.arguments["images"] == ["@node.reconstruct_grappa_005.reconstructed_nifti"]
    stats = res.diagnostics.get("rewrite_stats") or {}
    assert int(stats.get("unknown_nested_node_output_field_drops") or 0) >= 1


def test_sketch_compiler_drops_invalid_cardiac_classify_ed_es_overrides_to_allow_tool_repair() -> None:
    blueprints = [
        PlanNode(
            node_id="segment_cine_020",
            tool_name="segment_cardiac_cine",
            stage="segment",
            arguments={"cine_path": "@seq.CINE", "output_subdir": "segmentation/cardiac_cine"},
            required=True,
            depends_on=[],
        ),
        PlanNode(
            node_id="classify_cine_030",
            tool_name="classify_cardiac_cine_disease",
            stage="report",
            arguments={
                "cine_path": "@seq.CINE",
                "seg_path": "@node.segment_cine_020.seg_path",
                "seg_dir": "@node.segment_cine_020.pred_dir",
                "output_subdir": "classification/cardiac_cine",
            },
            required=True,
            depends_on=["segment_cine_020"],
        ),
    ]
    sketch = ConstrainedPlanSketch.model_validate(
        {
            "task": "long_cardiac_full",
            "domain": "cardiac",
            "steps": [
                {
                    "step_id": "seg1",
                    "tool": "segment_cardiac_cine",
                    "depends_on": [],
                    "inputs": {"cine_path": "@seq.CINE"},
                    "goal": "segment",
                    "optional": False,
                },
                {
                    "step_id": "cls1",
                    "tool": "classify_cardiac_cine_disease",
                    "depends_on": ["seg1"],
                    "inputs": {
                        "ed_seg_path": "@node.seg1.ed_segmentation_path",
                        "es_seg_path": "@node.seg1.es_seg_path",
                    },
                    "goal": "classify",
                    "optional": False,
                },
            ],
            "final_targets": ["cls1"],
        }
    )
    res = compile_constrained_plan_sketch(
        sketch=sketch,
        blueprints=blueprints,
        allowed_tools=["segment_cardiac_cine", "classify_cardiac_cine_disease"],
    )
    assert res.ok is True, res.diagnostics
    cls = next(n for n in res.nodes if n.tool_name == "classify_cardiac_cine_disease")
    assert "ed_seg_path" not in (cls.arguments or {})
    assert "es_seg_path" not in (cls.arguments or {})
    assert cls.arguments.get("seg_path") == "@node.segment_cine_020.seg_path"
    stats = res.diagnostics.get("rewrite_stats") or {}
    assert int(stats.get("unknown_node_output_field_drops") or 0) >= 2

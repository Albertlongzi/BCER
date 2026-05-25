"""Tests for BindingPolicy and its integration with Cerebellum.

Tests cover:
1. @auto bug fix verification (DICOM case should NOT get directory path)
2. noToken arm disables symbolic refs resolution
3. noToken arm does NOT implicitly autowire node outputs
4. Policy matrix snapshot to prevent regression / drift
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from mri_agent_shell.runtime.binding_policy import (
    POLICY_DETERMINISTIC_ONLY,
    POLICY_FULL,
    POLICY_NO_REFLECTOR,
    POLICY_NO_TOKEN,
    BindingPolicy,
    degrade_dag_tokens,
)
from mri_agent_shell.runtime.cerebellum import _ScopedBinder

from core.plan_dag import PlanNode


# ── helpers ───────────────────────────────────────────────────────────


@dataclass
class _FakeDAGNode:
    node_id: str
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class _FakeDAG:
    nodes: List[_FakeDAGNode]


def _mk_denoise_dag() -> _FakeDAG:
    """Minimal denoise DAG skeleton matching denoise.json template."""
    return _FakeDAG(nodes=[
        _FakeDAGNode(
            node_id="identify_sequences_001",
            tool_name="identify_sequences",
            arguments={"dicom_case_dir": "@case.input", "convert_to_nifti": True},
        ),
        _FakeDAGNode(
            node_id="denoise_bm3d_010",
            tool_name="denoise_image_bm3d",
            arguments={"input_nifti": "@auto", "output_subdir": "denoise"},
        ),
        _FakeDAGNode(
            node_id="compare_denoise_020",
            tool_name="compare_nifti_slices",
            arguments={
                "image_a": "@node.denoise_bm3d_010.input_nifti",
                "image_b": "@node.denoise_bm3d_010.denoised_nifti",
            },
        ),
    ])


def _mk_prostate_full_dag() -> _FakeDAG:
    """Minimal prostate_full_pipeline DAG skeleton."""
    return _FakeDAG(nodes=[
        _FakeDAGNode(
            node_id="identify_sequences_001",
            tool_name="identify_sequences",
            arguments={"dicom_case_dir": "@case.input"},
        ),
        _FakeDAGNode(
            node_id="register_adc_010",
            tool_name="register_to_reference",
            arguments={"fixed": "T2w", "moving": "ADC"},
        ),
        _FakeDAGNode(
            node_id="segment_prostate_030",
            tool_name="segment_prostate",
            arguments={"t2w_ref": "T2w"},
        ),
        _FakeDAGNode(
            node_id="detect_lesions_050",
            tool_name="detect_lesion_candidates",
            arguments={
                "t2w_nifti": "@node.segment_prostate_030.t2w_input_path",
                "adc_nifti": "@node.register_adc_010.resampled_path",
            },
        ),
        _FakeDAGNode(
            node_id="package_evidence_060",
            tool_name="package_vlm_evidence",
            arguments={"case_state_path": "@runtime.case_state_path"},
        ),
        _FakeDAGNode(
            node_id="generate_report_070",
            tool_name="generate_report",
            arguments={"case_state_path": "@runtime.case_state_path", "domain": "prostate"},
        ),
    ])


# ── 1. test_notoken_no_auto_bug_on_dicom_case ────────────────────────


def test_notoken_no_auto_bug_on_dicom_case() -> None:
    """Verify degrade_dag_tokens does NOT replace @auto with a directory path.

    The old _pre_resolve_dag_tokens resolved @auto to str(case_path) when
    no NIfTI files were found (prostate DICOM case), which then blocked
    _normalize_step_args from auto-resolving.  The new approach clears @auto
    to "" so that type normalisation can infer the input.
    """
    dag = _mk_denoise_dag()
    degrade_dag_tokens(dag, policy=POLICY_NO_TOKEN)

    denoise_node = dag.nodes[1]
    assert denoise_node.tool_name == "denoise_image_bm3d"

    # @auto should be converted to "" (not a directory path)
    assert denoise_node.arguments["input_nifti"] == "", (
        f"@auto was resolved to {denoise_node.arguments['input_nifti']!r} "
        "instead of being cleared to empty string"
    )


def test_degrade_preserves_runtime_tokens() -> None:
    """@runtime.* tokens must be preserved — they are infrastructure, not symbolic binding."""
    dag = _mk_prostate_full_dag()
    degrade_dag_tokens(dag, policy=POLICY_NO_TOKEN)

    pkg_node = dag.nodes[4]
    assert pkg_node.tool_name == "package_vlm_evidence"
    assert pkg_node.arguments["case_state_path"] == "@runtime.case_state_path"


def test_degrade_clears_node_tokens() -> None:
    """@node.* tokens must be cleared in noToken — no symbolic cross-node binding."""
    dag = _mk_prostate_full_dag()
    degrade_dag_tokens(dag, policy=POLICY_NO_TOKEN)

    detect_node = dag.nodes[3]
    assert detect_node.tool_name == "detect_lesion_candidates"
    assert detect_node.arguments["t2w_nifti"] == ""
    assert detect_node.arguments["adc_nifti"] == ""


def test_degrade_preserves_case_input() -> None:
    """@case.input stays as-is for type normalisation in identify_sequences."""
    dag = _mk_denoise_dag()
    degrade_dag_tokens(dag, policy=POLICY_NO_TOKEN)

    id_node = dag.nodes[0]
    assert id_node.arguments["dicom_case_dir"] == "@case.input"


def test_degrade_strips_seq_prefix() -> None:
    """@seq.T2w should be stripped to plain 'T2w'."""
    dag = _FakeDAG(nodes=[
        _FakeDAGNode(
            node_id="seg_001",
            tool_name="segment_prostate",
            arguments={"t2w_ref": "@seq.T2w"},
        ),
    ])
    degrade_dag_tokens(dag, policy=POLICY_NO_TOKEN)
    assert dag.nodes[0].arguments["t2w_ref"] == "T2w"


def test_degrade_noop_for_full_policy() -> None:
    """Full policy should not modify any tokens."""
    dag = _mk_denoise_dag()
    original_args = {n.node_id: dict(n.arguments) for n in dag.nodes}
    degrade_dag_tokens(dag, policy=POLICY_FULL)
    for n in dag.nodes:
        assert n.arguments == original_args[n.node_id]


# ── 2. test_notoken_disables_symbolic_refs_resolution ─────────────────


def test_resolve_runtime_refs_only_skips_node_tokens() -> None:
    """resolve_runtime_refs_only must NOT resolve @node.* or @seq.* tokens."""
    with tempfile.TemporaryDirectory() as td:
        case_root = Path(td) / "case"
        case_root.mkdir()

        binder = _ScopedBinder(case_root=case_root)
        binder.refs["runtime.case_state_path"] = "/tmp/state.json"
        binder.refs["case.input"] = str(case_root)
        binder.refs["seq.T2w"] = "/data/T2w.nii.gz"
        binder.node_outputs["seg_001"] = {"mask_path": "/data/mask.nii.gz"}

        args = {
            "state": "@runtime.case_state_path",
            "case": "@case.input",
            "seq": "@seq.T2w",
            "node_ref": "@node.seg_001.mask_path",
            "auto": "@auto",
        }

        resolved = binder.resolve_runtime_refs_only(args)

        # Infrastructure tokens resolved
        assert resolved["state"] == "/tmp/state.json"
        assert resolved["case"] == str(case_root)

        # Symbolic tokens NOT resolved
        assert resolved["seq"] == "@seq.T2w"
        assert resolved["node_ref"] == "@node.seg_001.mask_path"
        assert resolved["auto"] == "@auto"


def test_resolve_runtime_refs_only_allows_case_file_token() -> None:
    """noToken infrastructure resolution should include @case.file for raw-file cases."""
    with tempfile.TemporaryDirectory() as td:
        case_root = Path(td) / "case"
        case_root.mkdir()
        case_file = case_root / "cine.h5"
        case_file.write_bytes(b"fake")

        binder = _ScopedBinder(case_root=case_root)
        binder.refs["case.input"] = str(case_root)
        binder.refs["case.file"] = str(case_file)
        binder.refs["runtime.case_state_path"] = "/tmp/state.json"

        args = {
            "h5_path": "@case.file",
            "case_dir": "@case.input",
            "state": "@runtime.case_state_path",
            "seq": "@seq.CINE",
        }
        resolved = binder.resolve_runtime_refs_only(args)
        assert resolved["h5_path"] == str(case_file)
        assert resolved["case_dir"] == str(case_root)
        assert resolved["state"] == "/tmp/state.json"
        assert resolved["seq"] == "@seq.CINE"


def test_resolve_refs_full_resolves_everything() -> None:
    """Full resolve_refs must resolve @node.*, @seq.*, @case.*, @runtime.*."""
    with tempfile.TemporaryDirectory() as td:
        case_root = Path(td) / "case"
        case_root.mkdir()

        binder = _ScopedBinder(case_root=case_root)
        binder.refs["runtime.case_state_path"] = "/tmp/state.json"
        binder.refs["case.input"] = str(case_root)
        binder.refs["seq.T2w"] = "/data/T2w.nii.gz"
        binder.node_outputs["seg_001"] = {"mask_path": "/data/mask.nii.gz"}

        args = {
            "state": "@runtime.case_state_path",
            "case": "@case.input",
            "seq": "@seq.T2w",
            "node_ref": "@node.seg_001.mask_path",
        }

        resolved = binder.resolve_refs(args)

        assert resolved["state"] == "/tmp/state.json"
        assert resolved["case"] == str(case_root)
        assert resolved["seq"] == "/data/T2w.nii.gz"
        assert resolved["node_ref"] == "/data/mask.nii.gz"


# ── 3. test_notoken_does_not_implicitly_autowire_node_outputs ─────────


def test_normalize_step_args_notoken_skips_nifti_autowire() -> None:
    """Under noToken policy, _normalize_step_args must NOT auto-resolve
    empty input_nifti from seq_paths for denoise/resample tools.
    
    This verifies the root cause of the old bug is fixed: even if
    seq_paths has valid entries, the noToken arm should not use them
    because implicit_node_autowire_enabled=False.
    """
    with tempfile.TemporaryDirectory() as td:
        case_root = Path(td) / "case"
        case_root.mkdir()
        nifti = case_root / "T2w.nii.gz"
        nifti.write_bytes(b"fake")

        binder = _ScopedBinder(case_root=case_root)
        binder.seq_paths["T2w"] = str(nifti)

        # Import Cerebellum lazily to avoid heavy dependencies in unit test
        from mri_agent_shell.runtime.cerebellum import Cerebellum
        from mri_agent_shell.runtime.session import ModelConfig, SessionState

        session = SessionState(
            workspace_path=str(td),
            runs_root=str(Path(td) / "runs"),
            model_config=ModelConfig(provider="stub", llm="stub"),
            dry_run=True,
        )
        registry = {}  # dummy

        # Full policy: should autowire
        cereb_full = Cerebellum(
            session=session, registry=registry,
            binding_policy=POLICY_FULL,
        )
        result_full = cereb_full._normalize_step_args(
            tool_name="denoise_image_bm3d",
            args={"input_nifti": ""},  # empty = should be auto-resolved
            binder=binder,
            policy=POLICY_FULL,
        )
        assert result_full["input_nifti"] == str(nifti.resolve()), (
            "Full policy should autowire empty input_nifti from seq_paths"
        )

        # noToken policy: should NOT autowire
        cereb_notoken = Cerebellum(
            session=session, registry=registry,
            binding_policy=POLICY_NO_TOKEN,
        )
        result_notoken = cereb_notoken._normalize_step_args(
            tool_name="denoise_image_bm3d",
            args={"input_nifti": ""},
            binder=binder,
            policy=POLICY_NO_TOKEN,
        )
        assert result_notoken["input_nifti"] == "", (
            "noToken policy should NOT autowire input_nifti"
        )


def test_normalize_step_args_notoken_skips_seq_autocomplete() -> None:
    """Under noToken policy, plain sequence names like 'T2w' should NOT
    be resolved via seq_paths for segment_prostate / register_to_reference.
    """
    with tempfile.TemporaryDirectory() as td:
        case_root = Path(td) / "case"
        case_root.mkdir()
        nifti = case_root / "T2w.nii.gz"
        nifti.write_bytes(b"fake")

        binder = _ScopedBinder(case_root=case_root)
        binder.seq_paths["T2w"] = str(nifti)

        from mri_agent_shell.runtime.cerebellum import Cerebellum
        from mri_agent_shell.runtime.session import ModelConfig, SessionState

        session = SessionState(
            workspace_path=str(td),
            runs_root=str(Path(td) / "runs"),
            model_config=ModelConfig(provider="stub", llm="stub"),
            dry_run=True,
        )

        cereb = Cerebellum(
            session=session, registry={},
            binding_policy=POLICY_NO_TOKEN,
        )

        # segment_prostate with plain "T2w"
        result = cereb._normalize_step_args(
            tool_name="segment_prostate",
            args={"t2w_ref": "T2w"},
            binder=binder,
            policy=POLICY_NO_TOKEN,
        )
        # Should remain "T2w" (not resolved to nifti path)
        assert result["t2w_ref"] == "T2w", (
            f"noToken should not resolve 'T2w' but got {result['t2w_ref']!r}"
        )


# ── 3b. repair_tool_args semantic-binding suppressors (noToken) ──────


def test_repair_tool_args_notoken_suppresses_sequence_mapping_resolve() -> None:
    """repair_tool_args should keep type/schema repair, but not resolve
    plain sequence tokens ("T2w"/"ADC") into concrete paths when
    suppress_sequence_resolve=True.
    """
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        case_state_path = run_dir / "case_state.json"

        t2w = run_dir / "T2w.nii.gz"
        adc = run_dir / "ADC.nii.gz"
        t2w.write_bytes(b"fake")
        adc.write_bytes(b"fake")

        state = {
            "stage_outputs": {
                "identify": {
                    "identify_sequences": [
                        {"data": {"mapping": {"T2w": str(t2w), "ADC": str(adc)}}}
                    ]
                }
            }
        }
        case_state_path.write_text(json.dumps(state), encoding="utf-8")

        from tools.arg_models import repair_tool_args

        repaired_full = repair_tool_args(
            "register_to_reference",
            {"fixed": "T2w", "moving": "ADC"},
            state_path=case_state_path,
            ctx_case_state_path=case_state_path,
            dicom_case_dir=None,
        )
        assert repaired_full["fixed"] == str(t2w)
        assert repaired_full["moving"] == str(adc)

        repaired_notoken = repair_tool_args(
            "register_to_reference",
            {"fixed": "T2w", "moving": "ADC"},
            state_path=case_state_path,
            ctx_case_state_path=case_state_path,
            dicom_case_dir=None,
            suppress_sequence_resolve=True,
        )
        assert repaired_notoken["fixed"] == "T2w"
        assert repaired_notoken["moving"] == "ADC"


def test_repair_tool_args_notoken_suppresses_repair_node_output_autowire() -> None:
    """repair_tool_args should not back-fill t2w_ref from artifacts/ingest when
    suppress_node_output_autowire=True (even if sequence resolve is also off).
    """
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / "run"
        ingested_dir = run_dir / "artifacts" / "ingested" / "nifti"
        ingested_dir.mkdir(parents=True, exist_ok=True)
        t2w_ingested = ingested_dir / "series_t2w.nii.gz"
        t2w_ingested.write_bytes(b"fake")

        case_state_path = run_dir / "case_state.json"
        case_state_path.write_text(json.dumps({"stage_outputs": {}}), encoding="utf-8")

        from tools.arg_models import repair_tool_args

        repaired_with_autowire = repair_tool_args(
            "segment_prostate",
            {"t2w_ref": "T2w"},
            state_path=case_state_path,
            ctx_case_state_path=case_state_path,
            dicom_case_dir=None,
            suppress_sequence_resolve=True,  # isolate artifact fallback path
            suppress_node_output_autowire=False,
        )
        assert repaired_with_autowire["t2w_ref"] == str(t2w_ingested.resolve())

        repaired_no_autowire = repair_tool_args(
            "segment_prostate",
            {"t2w_ref": "T2w"},
            state_path=case_state_path,
            ctx_case_state_path=case_state_path,
            dicom_case_dir=None,
            suppress_sequence_resolve=True,
            suppress_node_output_autowire=True,
        )
        assert repaired_no_autowire["t2w_ref"] == "T2w"


# ── 4. test_full_vs_notoken_policy_matrix_snapshot ────────────────────


def test_policy_matrix_snapshot() -> None:
    """Snapshot test ensuring the canonical policy definitions don't drift.
    
    If any mechanism flag changes, this test will fail — forcing explicit
    acknowledgment and documentation of the change.
    """
    assert POLICY_FULL.to_dict() == {
        "symbolic_bind_enabled": True,
        "implicit_seq_autocomplete_enabled": True,
        "implicit_node_autowire_enabled": True,
        "tool_arg_type_repair_enabled": True,
        "suppress_sequence_resolve": False,
        "suppress_node_output_autowire": False,
        "error_reflection_enabled": True,
        "scope_guard_enabled": True,
        "provenance_logging_enabled": True,
    }

    assert POLICY_NO_TOKEN.to_dict() == {
        "symbolic_bind_enabled": False,
        "implicit_seq_autocomplete_enabled": False,
        "implicit_node_autowire_enabled": False,
        "tool_arg_type_repair_enabled": True,
        "suppress_sequence_resolve": True,
        "suppress_node_output_autowire": True,
        "error_reflection_enabled": True,
        "scope_guard_enabled": True,
        "provenance_logging_enabled": True,
    }

    assert POLICY_NO_REFLECTOR.to_dict() == {
        "symbolic_bind_enabled": True,
        "implicit_seq_autocomplete_enabled": True,
        "implicit_node_autowire_enabled": True,
        "tool_arg_type_repair_enabled": True,
        "suppress_sequence_resolve": False,
        "suppress_node_output_autowire": False,
        "error_reflection_enabled": False,
        "scope_guard_enabled": True,
        "provenance_logging_enabled": True,
    }

    assert POLICY_DETERMINISTIC_ONLY.to_dict() == {
        "symbolic_bind_enabled": True,
        "implicit_seq_autocomplete_enabled": True,
        "implicit_node_autowire_enabled": True,
        "tool_arg_type_repair_enabled": True,
        "suppress_sequence_resolve": False,
        "suppress_node_output_autowire": False,
        "error_reflection_enabled": False,
        "scope_guard_enabled": True,
        "provenance_logging_enabled": True,
    }


def test_arm_policy_mapping_completeness() -> None:
    """All benchmark arms must have a defined policy mapping."""
    expected_arms = [
        "bcr_full", "bcr_sketch", "bcr_no_token", "bcr_no_reflector",
        "bcr_deterministic_only", "static_pipeline",
        "react", "react_token", "pure_react",
    ]
    for arm in expected_arms:
        policy = BindingPolicy.for_arm(arm)
        assert isinstance(policy, BindingPolicy), f"No policy defined for arm {arm!r}"


# ── DAG degradation examples (per-template) ──────────────────────────


def test_degrade_denoise_template() -> None:
    """Verify denoise template degradation matches spec:
    - @case.input → kept (type normalisation)
    - @auto → "" (cleared)
    - @node.* → "" (no symbolic binding)
    """
    dag = _mk_denoise_dag()
    degrade_dag_tokens(dag, policy=POLICY_NO_TOKEN)

    assert dag.nodes[0].arguments["dicom_case_dir"] == "@case.input"
    assert dag.nodes[1].arguments["input_nifti"] == ""
    assert dag.nodes[2].arguments["image_a"] == ""
    assert dag.nodes[2].arguments["image_b"] == ""


def test_degrade_prostate_full_template() -> None:
    """Verify prostate_full_pipeline template degradation:
    - @case.input → kept
    - plain "T2w"/"ADC" → kept (these are plain strings, not tokens)
    - @node.* → "" (no symbolic binding)
    - @runtime.* → kept (infrastructure)
    """
    dag = _mk_prostate_full_dag()
    degrade_dag_tokens(dag, policy=POLICY_NO_TOKEN)

    # identify_sequences: @case.input kept
    assert dag.nodes[0].arguments["dicom_case_dir"] == "@case.input"

    # register: plain strings unchanged
    assert dag.nodes[1].arguments["fixed"] == "T2w"
    assert dag.nodes[1].arguments["moving"] == "ADC"

    # segment: plain string unchanged
    assert dag.nodes[2].arguments["t2w_ref"] == "T2w"

    # detect_lesions: @node.* cleared
    assert dag.nodes[3].arguments["t2w_nifti"] == ""
    assert dag.nodes[3].arguments["adc_nifti"] == ""

    # package/report: @runtime.* kept
    assert dag.nodes[4].arguments["case_state_path"] == "@runtime.case_state_path"
    assert dag.nodes[5].arguments["case_state_path"] == "@runtime.case_state_path"

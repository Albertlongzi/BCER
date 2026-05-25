from __future__ import annotations

import tempfile
from pathlib import Path

from mri_agent_shell.runtime.cerebellum import Cerebellum, _ScopedBinder
from mri_agent_shell.runtime.patch_spec import StructuredRepairContext, ToolSchemaInfo
from mri_agent_shell.runtime.session import ModelConfig, SessionState
from mri_agent_shell.tool_registry import build_shell_registry

from core.plan_dag import PlanNode


def _mk_session(workspace: Path) -> SessionState:
    return SessionState(
        workspace_path=str(workspace),
        runs_root=str(workspace / "runs"),
        model_config=ModelConfig(provider="stub", llm="stub"),
        dry_run=True,
    )


def _mk_reflection_halt_payload() -> dict:
    return {
        "action": "halt",
        "reason": "reflector_disabled: RuntimeError",
        "retry_arguments": {},
        "natural_language_response": "disabled",
    }


def _mk_required_fail_inputs(case_dir: Path) -> tuple[PlanNode, dict, _ScopedBinder]:
    node = PlanNode(
        node_id="node_001",
        tool_name="generate_report",
        stage="report",
        arguments={"case_state_path": "@state.path"},
        required=True,
    )
    rec = {
        "status": "FAIL",
        "arguments": {"case_state_path": "@state.path"},
        "error": {
            "type": "RuntimeError",
            "message": "unresolved reference in path argument 'case_state_path' for generate_report: @state.path",
        },
    }
    binder = _ScopedBinder(case_root=case_dir)
    return node, rec, binder


def test_no_reflector_mode_disables_deterministic_fallback_override() -> None:
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        case_dir = ws / "case_a"
        case_dir.mkdir(parents=True, exist_ok=True)

        session = _mk_session(ws)
        session.set_case_input(str(case_dir))
        registry = build_shell_registry(dry_run=True, include_core=False)
        cereb = Cerebellum(
            session=session,
            registry=registry,
            failure_reflect_fn=lambda _payload: _mk_reflection_halt_payload(),
            allow_deterministic_fallback_on_nonretry=False,
        )
        cereb._deterministic_retry_suggestion = lambda **_kwargs: {"case_state_path": "@runtime.case_state_path"}  # type: ignore[assignment]

        node, rec, binder = _mk_required_fail_inputs(case_dir)
        verdict = cereb._reflect_on_required_failure(
            node=node,
            rec=rec,
            binder=binder,
            scope_domain="prostate",
            case_id="case_a",
            run_id="run_x",
            step_results=[],
        )
        assert verdict["action"] == "halt"


def test_default_mode_keeps_deterministic_fallback_override() -> None:
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        case_dir = ws / "case_a"
        case_dir.mkdir(parents=True, exist_ok=True)

        session = _mk_session(ws)
        session.set_case_input(str(case_dir))
        registry = build_shell_registry(dry_run=True, include_core=False)
        cereb = Cerebellum(
            session=session,
            registry=registry,
            failure_reflect_fn=lambda _payload: _mk_reflection_halt_payload(),
            allow_deterministic_fallback_on_nonretry=True,
        )
        cereb._deterministic_retry_suggestion = lambda **_kwargs: {"case_state_path": "@runtime.case_state_path"}  # type: ignore[assignment]

        node, rec, binder = _mk_required_fail_inputs(case_dir)
        verdict = cereb._reflect_on_required_failure(
            node=node,
            rec=rec,
            binder=binder,
            scope_domain="prostate",
            case_id="case_a",
            run_id="run_x",
            step_results=[],
        )
        assert verdict["action"] == "retry"
        assert verdict["retry_arguments"] == {"case_state_path": "@runtime.case_state_path"}


def test_schema_missing_required_is_classified_as_fixable_runtime() -> None:
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        case_dir = ws / "case_a"
        case_dir.mkdir(parents=True, exist_ok=True)

        session = _mk_session(ws)
        session.set_case_input(str(case_dir))
        registry = build_shell_registry(dry_run=True, include_core=False)
        cereb = Cerebellum(session=session, registry=registry)
        cls = cereb._classify_reflection_limit(
            err_type="SchemaValidationError",
            err_msg="Missing required argument: input_nifti",
            deterministic_retry={},
        )
        assert cls == "fixable_runtime"


def test_identify_sequences_omission_can_be_recovered_without_schema_required_keys() -> None:
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        case_dir = ws / "case_a"
        case_dir.mkdir(parents=True, exist_ok=True)

        session = _mk_session(ws)
        session.set_case_input(str(case_dir))
        registry = build_shell_registry(dry_run=True, include_core=False)
        cereb = Cerebellum(session=session, registry=registry)

        ctx = StructuredRepairContext(
            failing_tool="identify_sequences",
            failing_args={"convert_to_nifti": True},
            error_type="ValueError",
            error_message="identify_sequences requires dicom_case_dir or series_inventory_path",
            failure_classification="unknown_runtime",
            tool_schema=ToolSchemaInfo(
                name="identify_sequences",
                required_keys=[],
                optional_keys=["dicom_case_dir", "series_inventory_path", "convert_to_nifti"],
                input_schema={},
            ),
            available_tokens={"@case.input": str(case_dir)},
            available_modalities=[],
            last_artifacts={},
            allowed_actions=["retry", "skip", "halt"],
            insertable_tools=[],
        )
        out = cereb._recover_missing_required_args(repair_ctx=ctx)
        assert out.get("dicom_case_dir") == "@case.input"

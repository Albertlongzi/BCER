from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from mri_agent_shell.runtime.cerebellum import Cerebellum, ScopeViolation, _CaseScopeGuard, _default_external_model_roots
from mri_agent_shell.runtime.session import ModelConfig, SessionState
from mri_agent_shell.tool_registry import build_shell_registry

from commands.registry import Tool, ToolRegistry
from commands.schemas import ToolContext, ToolSpec
from core.plan_dag import AgentPlanDAG, CaseScope, DagPolicy, PlanNode


class CerebellumScopeGuardTests(unittest.TestCase):
    def _mk_session(self, workspace: Path) -> SessionState:
        return SessionState(
            workspace_path=str(workspace),
            runs_root=str(workspace / "runs"),
            model_config=ModelConfig(provider="stub", llm="stub"),
            dry_run=True,
        )

    def test_cross_domain_absolute_path_injection_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            prostate_case = ws / "sub-prostate"
            brain_case = ws / "Brats18_CBICA_AAM_1"
            prostate_case.mkdir(parents=True, exist_ok=True)
            brain_case.mkdir(parents=True, exist_ok=True)

            session = self._mk_session(ws)
            session.set_case_input(str(prostate_case))
            registry = build_shell_registry(dry_run=True, include_core=False)
            cereb = Cerebellum(session=session, registry=registry)

            dag = AgentPlanDAG(
                plan_id="scope_block_001",
                goal="attempt cross-domain path injection",
                case_scope=CaseScope(
                    domain="prostate",
                    case_id="sub-prostate",
                    case_ref=str(prostate_case.resolve()),
                    workspace_root=str(ws),
                    runs_root=str(ws / "runs"),
                ),
                policy=DagPolicy(skip_optional_by_default=False),
                nodes=[
                    PlanNode(
                        node_id="load_case",
                        tool_name="dummy_load_case",
                        stage="ingest",
                        arguments={
                            "case_path": str(brain_case.resolve()),  # outside current case scope
                            "output_subdir": "dummy/load",
                        },
                        required=True,
                    )
                ],
            )

            out = cereb.execute_dag(dag, emit=lambda _msg: None)
            self.assertFalse(bool(out.get("ok")))

            trace_path = Path(str(out.get("trace_path") or ""))
            self.assertTrue(trace_path.exists())
            events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            fails = [e for e in events if e.get("tool_name") == "dummy_load_case" and e.get("status") == "FAIL"]
            self.assertTrue(fails)
            self.assertEqual(str((fails[-1].get("error") or {}).get("type") or ""), "ScopeViolation")

    def test_normalization_ignores_stale_session_path_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-prostate"
            stale_dir = ws / "brain-old"
            case_dir.mkdir(parents=True, exist_ok=True)
            stale_dir.mkdir(parents=True, exist_ok=True)

            t2_case = case_dir / "T2w.nii.gz"
            adc_case = case_dir / "ADC.nii.gz"
            t2_case.write_text("case-t2\n", encoding="utf-8")
            adc_case.write_text("case-adc\n", encoding="utf-8")
            stale_t2 = stale_dir / "T2w.nii.gz"
            stale_t2.write_text("stale-t2\n", encoding="utf-8")

            session = self._mk_session(ws)
            session.set_case_input(str(case_dir))
            # Stale cross-domain cache: should never win.
            session.path_keys["seq.T2w"] = str(stale_t2.resolve())
            session.path_keys["seq.ADC"] = str(stale_t2.resolve())

            registry = build_shell_registry(dry_run=True, include_core=False)
            cereb = Cerebellum(session=session, registry=registry)

            out = cereb._normalize_step_args(
                tool_name="register_to_reference",
                args={"fixed": "T2w", "moving": "ADC", "output_subdir": "registration"},
            )
            self.assertEqual(str(out.get("fixed")), str(t2_case.resolve()))
            self.assertEqual(str(out.get("moving")), str(adc_case.resolve()))

    def test_required_false_nodes_are_natively_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "case_a"
            case_dir.mkdir(parents=True, exist_ok=True)

            session = self._mk_session(ws)
            session.set_case_input(str(case_dir))
            registry = build_shell_registry(dry_run=True, include_core=False)
            cereb = Cerebellum(session=session, registry=registry)

            dag = AgentPlanDAG(
                plan_id="optional_skip_001",
                goal="validate optional skip semantics",
                case_scope=CaseScope(
                    domain="demo",
                    case_id="case_a",
                    case_ref=str(case_dir.resolve()),
                    workspace_root=str(ws),
                    runs_root=str(ws / "runs"),
                ),
                policy=DagPolicy(skip_optional_by_default=True),
                nodes=[
                    PlanNode(
                        node_id="load",
                        tool_name="dummy_load_case",
                        stage="ingest",
                        arguments={"case_path": "@case.input", "output_subdir": "dummy/load"},
                        required=True,
                    ),
                    PlanNode(
                        node_id="seg_optional",
                        tool_name="dummy_segment",
                        stage="segment",
                        arguments={"case_path": "@case.input", "output_subdir": "dummy/segmentation"},
                        required=False,
                        depends_on=["load"],
                    ),
                ],
            )

            out = cereb.execute_dag(dag, emit=lambda _msg: None)
            self.assertTrue(bool(out.get("ok")))

            trace_path = Path(str(out.get("trace_path") or ""))
            events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            skipped = [
                e
                for e in events
                if e.get("tool_name") == "dummy_segment" and str(e.get("status")) == "SKIPPED_OPTIONAL"
            ]
            self.assertTrue(skipped)

    def test_default_external_model_roots_allow_brats_bundle_cache(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "case_a"
            run_dir = ws / "runs" / "case_a" / "run_x"
            case_dir.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)

            guard = _CaseScopeGuard(
                case_root=case_dir.resolve(),
                run_root=run_dir.resolve(),
                external_roots=_default_external_model_roots(),
            )
            bundle_root = Path("~/.cache/torch/hub/bundle/brats_mri_segmentation").expanduser().resolve()
            guard.validate_args(
                tool_name="brats_mri_segmentation",
                args={"bundle_root": str(bundle_root)},
            )

    def test_external_path_outside_allowlist_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "case_a"
            run_dir = ws / "runs" / "case_a" / "run_x"
            case_dir.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)

            allow = [Path("~/.cache/torch/hub/bundle").expanduser().resolve()]
            guard = _CaseScopeGuard(
                case_root=case_dir.resolve(),
                run_root=run_dir.resolve(),
                external_roots=allow,
            )
            with self.assertRaises(ScopeViolation):
                guard.validate_args(
                    tool_name="brats_mri_segmentation",
                    args={"bundle_root": str((ws / "outside_bundle").resolve())},
                )

    def test_symlink_path_inside_case_scope_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "acdc_multiseq_patient061_ed"
            external_root = ws / "acdc_test_group_4_per_group"
            run_dir = ws / "runs" / "case_a" / "run_x"
            case_dir.mkdir(parents=True, exist_ok=True)
            external_root.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)

            external_cine = external_root / "patient061_4d.nii.gz"
            external_cine.write_text("cine\n", encoding="utf-8")
            linked_cine = case_dir / "patient061_4d.nii.gz"
            os.symlink(str(external_cine), str(linked_cine))

            guard = _CaseScopeGuard(
                case_root=case_dir.resolve(),
                run_root=run_dir.resolve(),
                external_roots=[],
            )
            # Must not raise; symlink path itself is inside case scope.
            guard.validate_args(
                tool_name="segment_cardiac_cine",
                args={"cine_path": str(linked_cine)},
            )

    def test_patient_info_cfg_under_symlink_target_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "acdc_multiseq_patient061_ed"
            external_root = ws / "acdc_test_group_4_per_group"
            run_dir = ws / "runs" / "case_a" / "run_x"
            case_dir.mkdir(parents=True, exist_ok=True)
            external_root.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)

            external_cine = external_root / "patient061_4d.nii.gz"
            external_cine.write_text("cine\n", encoding="utf-8")
            external_info = external_root / "Info.cfg"
            external_info.write_text("ED:1\n", encoding="utf-8")
            linked_cine = case_dir / "patient061_4d.nii.gz"
            os.symlink(str(external_cine), str(linked_cine))

            guard = _CaseScopeGuard(
                case_root=case_dir.resolve(),
                run_root=run_dir.resolve(),
                external_roots=[],
            )
            # patient_info_path may be resolved to the physical target path by arg repair.
            # This should still be allowed when it is under a symlink-targeted case directory.
            guard.validate_args(
                tool_name="classify_cardiac_cine_disease",
                args={
                    "cine_path": str(linked_cine),
                    "patient_info_path": str(external_info.resolve()),
                },
            )

    def test_reflect_node_default_noop_can_run_in_dag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "case_a"
            case_dir.mkdir(parents=True, exist_ok=True)

            session = self._mk_session(ws)
            session.set_case_input(str(case_dir))
            registry = build_shell_registry(dry_run=True, include_core=False)
            cereb = Cerebellum(session=session, registry=registry)

            dag = AgentPlanDAG(
                plan_id="reflect_node_001",
                goal="validate reflect-node scaffolding",
                case_scope=CaseScope(
                    domain="demo",
                    case_id="case_a",
                    case_ref=str(case_dir.resolve()),
                    workspace_root=str(ws),
                    runs_root=str(ws / "runs"),
                ),
                policy=DagPolicy(skip_optional_by_default=False),
                nodes=[
                    PlanNode(
                        node_id="reflect_005",
                        node_type="reflect",
                        stage="misc",
                        arguments={"reason": "pre-check"},
                        required=True,
                    ),
                    PlanNode(
                        node_id="load_010",
                        tool_name="dummy_load_case",
                        stage="ingest",
                        arguments={"case_path": "@case.input", "output_subdir": "dummy/load"},
                        required=True,
                        depends_on=["reflect_005"],
                    ),
                ],
            )
            out = cereb.execute_dag(dag, emit=lambda _msg: None)
            self.assertTrue(bool(out.get("ok")))

            trace_path = Path(str(out.get("trace_path") or ""))
            events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            control_done = [
                e
                for e in events
                if e.get("event_type") == "control_node"
                and e.get("tool_name") == "__reflect__"
                and str(e.get("status")) == "DONE"
            ]
            self.assertTrue(control_done)

    def test_gate_node_can_halt_execution_with_custom_reflector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "case_a"
            case_dir.mkdir(parents=True, exist_ok=True)

            session = self._mk_session(ws)
            session.set_case_input(str(case_dir))
            registry = build_shell_registry(dry_run=True, include_core=False)
            cereb = Cerebellum(
                session=session,
                registry=registry,
                reflect_fn=lambda _payload: {"decision": "halt", "reason": "qc_gate_failed"},
            )

            dag = AgentPlanDAG(
                plan_id="gate_node_001",
                goal="validate gate halt behavior",
                case_scope=CaseScope(
                    domain="demo",
                    case_id="case_a",
                    case_ref=str(case_dir.resolve()),
                    workspace_root=str(ws),
                    runs_root=str(ws / "runs"),
                ),
                policy=DagPolicy(skip_optional_by_default=False),
                nodes=[
                    PlanNode(
                        node_id="gate_005",
                        node_type="gate",
                        stage="misc",
                        arguments={},
                        required=True,
                    ),
                    PlanNode(
                        node_id="load_010",
                        tool_name="dummy_load_case",
                        stage="ingest",
                        arguments={"case_path": "@case.input", "output_subdir": "dummy/load"},
                        required=True,
                        depends_on=["gate_005"],
                    ),
                ],
            )
            out = cereb.execute_dag(dag, emit=lambda _msg: None)
            self.assertFalse(bool(out.get("ok")))

            trace_path = Path(str(out.get("trace_path") or ""))
            events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            gate_fail = [
                e
                for e in events
                if e.get("event_type") == "control_node"
                and e.get("tool_name") == "__gate__"
                and str(e.get("status")) == "FAIL"
            ]
            self.assertTrue(gate_fail)
            self.assertEqual(str((gate_fail[-1].get("error") or {}).get("type") or ""), "ReflectHalt")

    def test_required_failure_can_retry_via_reflection_engine(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "case_a"
            outside_dir = ws / "outside_case"
            case_dir.mkdir(parents=True, exist_ok=True)
            outside_dir.mkdir(parents=True, exist_ok=True)

            session = self._mk_session(ws)
            session.set_case_input(str(case_dir))
            registry = build_shell_registry(dry_run=True, include_core=False)
            cereb = Cerebellum(
                session=session,
                registry=registry,
                failure_reflect_fn=lambda _payload: {
                    "action": "retry",
                    "reason": "recover_scope",
                    "retry_arguments": {"case_path": "@case.input", "output_subdir": "dummy/load"},
                    "natural_language_response": "I retried with in-scope case path and recovered.",
                },
            )

            dag = AgentPlanDAG(
                plan_id="reflect_retry_001",
                goal="recover a required failure via reflection retry",
                case_scope=CaseScope(
                    domain="demo",
                    case_id="case_a",
                    case_ref=str(case_dir.resolve()),
                    workspace_root=str(ws),
                    runs_root=str(ws / "runs"),
                ),
                policy=DagPolicy(skip_optional_by_default=False),
                nodes=[
                    PlanNode(
                        node_id="load_010",
                        tool_name="dummy_load_case",
                        stage="ingest",
                        arguments={"case_path": str(outside_dir.resolve()), "output_subdir": "dummy/load"},
                        required=True,
                    ),
                ],
            )
            out = cereb.execute_dag(dag, emit=lambda _msg: None)
            self.assertTrue(bool(out.get("ok")))
            decisions = out.get("reflection_decisions") or []
            self.assertTrue(decisions)
            self.assertEqual(str(decisions[-1].get("action") or ""), "retry")
            self.assertIn("recovered", str(out.get("natural_language_response") or "").lower())

    def test_default_reflector_auto_repairs_unresolved_path_token(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "case_a"
            case_dir.mkdir(parents=True, exist_ok=True)

            session = self._mk_session(ws)
            session.set_case_input(str(case_dir))
            registry = build_shell_registry(dry_run=True, include_core=False)
            cereb = Cerebellum(session=session, registry=registry)

            dag = AgentPlanDAG(
                plan_id="reflect_retry_state_path_001",
                goal="auto-repair unresolved @state.path token",
                case_scope=CaseScope(
                    domain="demo",
                    case_id="case_a",
                    case_ref=str(case_dir.resolve()),
                    workspace_root=str(ws),
                    runs_root=str(ws / "runs"),
                ),
                policy=DagPolicy(skip_optional_by_default=False),
                nodes=[
                    PlanNode(
                        node_id="load_010",
                        tool_name="dummy_load_case",
                        stage="ingest",
                        arguments={"case_path": "@bogus.path", "output_subdir": "dummy/load"},
                        required=True,
                    ),
                ],
            )
            out = cereb.execute_dag(dag, emit=lambda _msg: None)
            self.assertTrue(bool(out.get("ok")))
            decisions = out.get("reflection_decisions") or []
            self.assertTrue(decisions)
            self.assertEqual(str(decisions[-1].get("action") or ""), "retry")
            self.assertIn("path/token", str(out.get("natural_language_response") or "").lower())

    def test_reflector_hard_limit_schema_explains_backend_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            session = self._mk_session(ws)
            registry = build_shell_registry(dry_run=True, include_core=False)
            cereb = Cerebellum(session=session, registry=registry)

            out = cereb._default_failure_reflect_fn(
                {
                    "tool_name": "rag_search",
                    "error": {
                        "type": "SchemaValidationError",
                        "message": "Tool 'rag_search' is not allowed for domain 'prostate'.",
                    },
                    "deterministic_retry_suggestion": {},
                }
            )
            self.assertEqual(str(out.get("action") or ""), "halt")
            msg = str(out.get("natural_language_response") or "").lower()
            self.assertIn("do not have permission to modify system schema configurations", msg)

    def test_reflector_hard_limit_scope_explains_guard_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            session = self._mk_session(ws)
            registry = build_shell_registry(dry_run=True, include_core=False)
            cereb = Cerebellum(session=session, registry=registry)

            out = cereb._default_failure_reflect_fn(
                {
                    "tool_name": "segment_cardiac_cine",
                    "error": {
                        "type": "ScopeViolation",
                        "message": "path argument 'cine_path' escaped case scope for segment_cardiac_cine: /tmp/outside.nii.gz",
                    },
                    "deterministic_retry_suggestion": {},
                }
            )
            self.assertEqual(str(out.get("action") or ""), "halt")
            msg = str(out.get("natural_language_response") or "").lower()
            self.assertIn("cannot bypass case scope guard restrictions", msg)

    def test_successful_rag_step_bubbles_answer_to_run_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "case_a"
            case_dir.mkdir(parents=True, exist_ok=True)

            session = self._mk_session(ws)
            session.set_case_input(str(case_dir))

            def _rag_tool(_args: dict, _ctx: ToolContext) -> dict:
                return {"data": {"answer": "TE is 90ms from DWI metadata."}}

            reg = ToolRegistry()
            reg.register(
                Tool(
                    spec=ToolSpec(
                        name="rag_search",
                        description="test rag tool",
                        input_schema={"type": "object", "properties": {}},
                        output_schema={
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                        },
                    ),
                    func=_rag_tool,
                )
            )

            cereb = Cerebellum(session=session, registry=reg)
            dag = AgentPlanDAG(
                plan_id="qa_answer_bubble_001",
                goal="read TE",
                case_scope=CaseScope(
                    domain="prostate",
                    case_id="case_a",
                    case_ref=str(case_dir.resolve()),
                    workspace_root=str(ws),
                    runs_root=str(ws / "runs"),
                ),
                policy=DagPolicy(skip_optional_by_default=False),
                nodes=[
                    PlanNode(
                        node_id="rag_search_020",
                        tool_name="rag_search",
                        stage="qa",
                        arguments={},
                        required=True,
                    )
                ],
            )
            out = cereb.execute_dag(dag, emit=lambda _msg: None)
            self.assertTrue(bool(out.get("ok")))
            self.assertEqual(str(out.get("natural_language_response") or ""), "TE is 90ms from DWI metadata.")
            self.assertEqual(str(out.get("final_answer") or ""), "TE is 90ms from DWI metadata.")


if __name__ == "__main__":
    unittest.main()

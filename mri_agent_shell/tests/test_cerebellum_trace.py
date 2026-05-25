from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mri_agent_shell.agent.brain import Brain
from mri_agent_shell.runtime.cerebellum import Cerebellum
from mri_agent_shell.runtime.session import ModelConfig, SessionState
from mri_agent_shell.tool_registry import build_shell_registry


class CerebellumTraceTests(unittest.TestCase):
    def test_identify_arg_normalization_prefers_session_case_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "demo_case"
            case_dir.mkdir(parents=True, exist_ok=True)

            session = SessionState(
                workspace_path=str(ws),
                runs_root=str(ws / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=True,
            )
            session.set_case_input(str(case_dir))
            registry = build_shell_registry(dry_run=True, include_core=False)
            cereb = Cerebellum(session=session, registry=registry)

            out = cereb._normalize_step_args(
                tool_name="identify_sequences",
                args={"dicom_case_dir": "@case.input", "convert_to_nifti": True},
            )
            self.assertEqual(str(out.get("dicom_case_dir")), str(case_dir))

    def test_segment_prostate_token_resolves_to_case_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-019_2"
            case_dir.mkdir(parents=True, exist_ok=True)
            t2 = case_dir / "T2w.nii.gz"
            t2.write_text("dummy\n", encoding="utf-8")

            session = SessionState(
                workspace_path=str(ws),
                runs_root=str(ws / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=True,
            )
            session.set_case_input(str(case_dir))
            registry = build_shell_registry(dry_run=True, include_core=False)
            cereb = Cerebellum(session=session, registry=registry)

            out = cereb._normalize_step_args(
                tool_name="segment_prostate",
                args={"t2w_ref": "T2w", "output_subdir": "segmentation"},
            )
            self.assertEqual(str(out.get("t2w_ref")), str(t2.resolve()))

    def test_register_prefers_nifti_file_inside_modality_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-019_2"
            t2_dir = case_dir / "T2w"
            adc_dir = case_dir / "ADC"
            t2_dir.mkdir(parents=True, exist_ok=True)
            adc_dir.mkdir(parents=True, exist_ok=True)
            t2 = t2_dir / "image.nii.gz"
            adc = adc_dir / "image.nii.gz"
            t2.write_text("dummy\n", encoding="utf-8")
            adc.write_text("dummy\n", encoding="utf-8")

            session = SessionState(
                workspace_path=str(ws),
                runs_root=str(ws / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=True,
            )
            session.set_case_input(str(case_dir))
            registry = build_shell_registry(dry_run=True, include_core=False)
            cereb = Cerebellum(session=session, registry=registry)

            out = cereb._normalize_step_args(
                tool_name="register_to_reference",
                args={"fixed": "T2w", "moving": "ADC", "output_subdir": "registration"},
            )
            self.assertEqual(str(out.get("fixed")), str(t2.resolve()))
            self.assertEqual(str(out.get("moving")), str(adc.resolve()))

    def test_dry_run_produces_trace_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "demo_case"
            case_dir.mkdir(parents=True, exist_ok=True)

            session = SessionState(
                workspace_path=str(ws),
                runs_root=str(ws / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=True,
            )
            session.set_case_input(str(case_dir))

            registry = build_shell_registry(dry_run=True, include_core=False)
            brain = Brain(tool_specs=registry.list_specs())
            req_res = brain.from_request("process this prostate case and generate a report", session)
            if req_res.kind == "template" and req_res.template is not None:
                tpl = req_res.template.to_payload()
                req = dict(tpl.get("REQUIRED") or {})
                req["domain"] = "prostate"
                req["case_ref"] = "case.input"
                tpl["REQUIRED"] = req
                req_res = brain.from_template(tpl, session=session)
            self.assertEqual(req_res.kind, "plan")

            cereb = Cerebellum(session=session, registry=registry)
            out = cereb.execute_plan(req_res.plan.model_dump(), emit=lambda _msg: None)

            report_path = Path(str(out.get("report_path") or ""))
            trace_path = Path(str(out.get("trace_path") or ""))
            self.assertTrue(report_path.exists())
            self.assertTrue(trace_path.exists())

            lines = trace_path.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 3)

    def test_dependency_blocked_step_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "demo_case"
            case_dir.mkdir(parents=True, exist_ok=True)

            session = SessionState(
                workspace_path=str(ws),
                runs_root=str(ws / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=True,
            )
            session.set_case_input(str(case_dir))
            registry = build_shell_registry(dry_run=True, include_core=False)
            cereb = Cerebellum(session=session, registry=registry)

            plan = {
                "goal": "dependency test",
                "domain": "prostate",
                "request_type": "full_pipeline",
                "case_ref": "case.input",
                "case_id": "demo_case",
                "steps": [
                    {
                        "tool_name": "dummy_generate_report",
                        "stage": "report",
                        "required": True,
                        "depends_on": ["dummy_segment"],
                        "arguments": {"case_id": "demo_case", "output_subdir": "dummy/report"},
                    }
                ],
            }

            out = cereb.execute_plan(plan, emit=lambda _msg: None)
            self.assertFalse(bool(out.get("ok")))
            trace_path = Path(str(out.get("trace_path") or ""))
            self.assertTrue(trace_path.exists())
            events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            blocked = [e for e in events if str(e.get("status")) == "BLOCKED"]
            self.assertTrue(blocked)


if __name__ == "__main__":
    unittest.main()

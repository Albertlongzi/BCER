from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mri_agent_shell.runtime.session import ModelConfig, SessionState
from mri_agent_shell.shell.repl import AgentShell


class ShellInputFlowTests(unittest.TestCase):
    def test_help_without_colon_shows_shell_help(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            lines: list[str] = []

            shell.handle_line("help", emit=lines.append)
            text = "\n".join(lines)
            self.assertIn("Commands:", text)
            self.assertNotIn("Request is underspecified", text)

    def test_nl_request_observe_then_plan_without_manual_template_fill(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-019_2"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("dummy\n", encoding="utf-8")

            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            session.set_case_input(str(case_dir))
            lines: list[str] = []

            shell.handle_line("help me process this case and generate a report", emit=lines.append)
            text = "\n".join(lines)
            self.assertIn("inspect_case", text)
            self.assertIsNotNone(session.current_plan)

    def test_nl_cardiac_missing_cine_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "acdc_like_case"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("dummy\n", encoding="utf-8")

            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            lines: list[str] = []
            shell.handle_line(f":case load {str(case_dir)}", emit=lines.append)

            shell.handle_line("please do full cardiac pipeline and classify", emit=lines.append)
            text = "\n".join(lines)
            self.assertIn("cine", text.lower())
            self.assertNotIn("blocked: missing required modality", text.lower())
            self.assertIsNone(session.current_plan)

    def test_nl_response_strips_think_tags_and_formats_prompt_response(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "case_x"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("dummy\n", encoding="utf-8")

            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            session.set_case_input(str(case_dir))
            lines: list[str] = []

            mocked_dag = {
                "plan_id": "dag_test_001",
                "case_scope": {"case_ref": str(case_dir), "case_id": "case_x", "domain": "prostate"},
                "nodes": [],
                "planner_status": "ready",
                "natural_language_response": "<think>hidden reasoning</think>Final answer for user.",
            }
            with patch.object(AgentShell, "_plan_dag_from_langgraph", return_value=(mocked_dag, "")):
                shell.handle_line("please process this prostate case", emit=lines.append)

            text = "\n".join(lines)
            self.assertIn("User Prompt:", text)
            self.assertIn("Agent Response:", text)
            self.assertIn("Final answer for user.", text)
            self.assertNotIn("hidden reasoning", text)
            self.assertNotIn("<think>", text)

    def test_inline_fields_make_plan_ready(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            lines: list[str] = []

            shell.handle_line("process this case and generate a report", emit=lines.append)
            self.assertIsNotNone(session.current_template)
            self.assertIsNone(session.current_plan)

            shell.handle_line("domain: brain, request_type: segment", emit=lines.append)
            text = "\n".join(lines)
            self.assertIn("Required fields filled. Type ':run' to execute.", text)
            self.assertIsNotNone(session.current_plan)
            case_ref = str((session.current_template or {}).get("REQUIRED", {}).get("case_ref", ""))
            self.assertIn("Brats18_CBICA_AAM_1", case_ref)

    def test_script_paste_block_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            script = ws / "flow.txt"
            script.write_text(
                "\n".join(
                    [
                        ":paste",
                        "domain: cardiac",
                        "request_type: segment",
                        ":end",
                        ":exit",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            session = SessionState(
                workspace_path=str(ws),
                runs_root=str(ws / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            lines: list[str] = []
            rc = shell.run_script(script, emit=lines.append)
            self.assertEqual(rc, 0)
            self.assertIsNotNone(session.current_plan)
            self.assertIn("Required fields filled. Type ':run' to execute.", "\n".join(lines))

    def test_run_warns_when_case_input_key_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            lines: list[str] = []
            session.current_plan = {
                "domain": "brain",
                "steps": [
                    {
                        "tool_name": "identify_sequences",
                        "stage": "identify",
                        "required": True,
                        "arguments": {"dicom_case_dir": "@case.input", "convert_to_nifti": True, "output_subdir": "ingest"},
                    }
                ],
            }
            session.path_keys.pop("case.input", None)
            session.case_inputs.pop("case_input_path", None)
            shell.handle_line(":run", emit=lines.append)
            text = "\n".join(lines)
            self.assertIn("Plan is ready but `case.input` is unresolved.", text)

    def test_run_prints_agent_response_when_final_answer_exists(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "case_q"
            case_dir.mkdir(parents=True, exist_ok=True)

            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            lines: list[str] = []

            session.current_plan = {
                "plan_id": "dag_qa_001",
                "goal": "read TE",
                "case_scope": {
                    "domain": "prostate",
                    "case_id": "case_q",
                    "case_ref": str(case_dir),
                    "workspace_root": str(ws),
                    "runs_root": str(ws / "runs"),
                },
                "policy": {"skip_optional_by_default": False},
                "nodes": [],
            }

            with patch(
                "mri_agent_shell.shell.repl.Cerebellum.execute_dag",
                return_value={
                    "ok": True,
                    "natural_language_response": "",
                    "final_answer": "<think>hidden chain of thought</think>The TE is 90 ms.",
                },
            ):
                shell.handle_line(":run", emit=lines.append)

            text = "\n".join(lines)
            self.assertIn("Agent Response:", text)
            self.assertIn("The TE is 90 ms.", text)
            self.assertNotIn("<think>", text)
            self.assertNotIn("hidden chain of thought", text)

    def test_run_renders_structured_response_and_hides_raw_json_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "case_q2"
            case_dir.mkdir(parents=True, exist_ok=True)

            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            lines: list[str] = []

            session.current_plan = {
                "plan_id": "dag_qa_002",
                "goal": "read TE",
                "case_scope": {
                    "domain": "prostate",
                    "case_id": "case_q2",
                    "case_ref": str(case_dir),
                    "workspace_root": str(ws),
                    "runs_root": str(ws / "runs"),
                },
                "policy": {"skip_optional_by_default": False},
                "nodes": [],
            }

            with patch(
                "mri_agent_shell.shell.repl.Cerebellum.execute_dag",
                return_value={
                    "ok": True,
                    "trace_path": str(ws / "logs" / "case_q2" / "tool_trace.jsonl"),
                    "natural_language_response": (
                        "<thought_process>\n"
                        "I checked DICOM evidence lines for TE.\n"
                        "</thought_process>\n"
                        "<final_answer>\n"
                        "EchoTime (TE) for T2w is 132 ms.\n"
                        "</final_answer>"
                    ),
                    "final_answer": "",
                },
            ):
                shell.handle_line(":run", emit=lines.append)

            text = "\n".join(lines)
            self.assertIn("Agent Response:", text)
            self.assertIn("I checked DICOM evidence lines for TE.", text)
            self.assertIn("EchoTime (TE) for T2w is 132 ms.", text)
            self.assertIn("Run completed successfully", text)
            self.assertNotIn("<thought_process>", text)
            self.assertNotIn("<final_answer>", text)
            self.assertNotIn("\"ok\": true", text)

    def test_domain_switch_rebinds_auto_demo_case_and_case_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            lines: list[str] = []

            shell.handle_line("domain: brain", emit=lines.append)
            shell.handle_line("domain: prostate", emit=lines.append)
            text = "\n".join(lines)
            self.assertIn("Detected domain 'prostate'. Auto-loading default case: sub-019_2", text)

            tpl = session.current_template or {}
            req = dict(tpl.get("REQUIRED") or {})
            opt = dict(tpl.get("OPTIONAL") or {})
            self.assertIn("sub-019_2", str(req.get("case_ref") or ""))
            self.assertEqual(str(opt.get("case_id") or ""), "sub-019_2")
            self.assertEqual(str(session.case_id or ""), "sub-019_2")

    def test_manual_case_load_is_flushed_on_domain_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            custom_case = ws / "my_manual_case"
            custom_case.mkdir(parents=True, exist_ok=True)

            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            lines: list[str] = []

            shell.handle_line(f":case load {custom_case}", emit=lines.append)
            shell.handle_line("domain: prostate", emit=lines.append)
            text = "\n".join(lines)
            self.assertIn("Manual scope change detected: cleared prior case binding.", text)
            self.assertIn("Auto-loading default case: sub-019_2", text)

            tpl = session.current_template or {}
            req = dict(tpl.get("REQUIRED") or {})
            self.assertIn("sub-019_2", str(req.get("case_ref") or ""))
            self.assertEqual(str(session.case_id or ""), "sub-019_2")

    def test_domain_change_with_explicit_case_ref_rebinds_to_user_case(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            custom_case = ws / "my_manual_case"
            custom_case.mkdir(parents=True, exist_ok=True)

            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            lines: list[str] = []

            shell.handle_line("domain: prostate", emit=lines.append)
            shell.handle_line(
                f"domain: prostate, request_type: segment, case_ref: {custom_case}",
                emit=lines.append,
            )
            tpl = session.current_template or {}
            req = dict(tpl.get("REQUIRED") or {})
            opt = dict(tpl.get("OPTIONAL") or {})
            self.assertEqual(str(req.get("case_ref") or ""), str(custom_case.resolve()))
            self.assertEqual(str(opt.get("case_id") or ""), "my_manual_case")
            self.assertEqual(str(session.case_id or ""), "my_manual_case")

    def test_case_load_invalid_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            lines: list[str] = []

            shell.handle_line(":case load /definitely/not/exist", emit=lines.append)
            text = "\n".join(lines)
            self.assertIn("Invalid case path:", text)
            self.assertNotIn("Case loaded:", text)

    def test_case_load_with_trailing_inline_patch_keeps_path_clean(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-019_2"
            case_dir.mkdir(parents=True, exist_ok=True)

            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            lines: list[str] = []

            shell.handle_line(
                f":case load {case_dir} domain: prostate, request_type: segment",
                emit=lines.append,
            )
            text = "\n".join(lines)
            self.assertIn("Case loaded:", text)
            self.assertEqual(str(session.case_inputs.get("case_input_path") or ""), str(case_dir.resolve()))
            tpl = session.current_template or {}
            req = dict(tpl.get("REQUIRED") or {})
            self.assertEqual(str(req.get("domain") or ""), "prostate")


if __name__ == "__main__":
    unittest.main()

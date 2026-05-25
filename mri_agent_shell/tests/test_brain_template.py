from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mri_agent_shell.agent.brain import Brain
from mri_agent_shell.runtime.session import ModelConfig, SessionState


class BrainTemplateTests(unittest.TestCase):
    def test_underspecified_request_returns_template(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            brain = Brain(tool_specs=[])
            res = brain.from_request("process this medical case and generate a report", session)
            self.assertEqual(res.kind, "template")
            self.assertIsNotNone(res.template)
            self.assertIn("REQUIRED", res.template.to_payload())
            payload = res.template.to_payload()
            self.assertIsInstance(payload["REQUIRED"].get("domain"), dict)
            self.assertIn("choices", payload["REQUIRED"]["domain"])

    def test_template_to_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            case_dir = Path(td) / "case_a"
            case_dir.mkdir(parents=True, exist_ok=True)

            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            session.set_case_input(str(case_dir))

            brain = Brain(tool_specs=[])
            tpl = {
                "REQUIRED": {
                    "domain": "prostate",
                    "request_type": "full_pipeline",
                    "case_ref": "case.input",
                },
                "OPTIONAL": {
                    "case_id": "case_a",
                },
            }
            res = brain.from_template(tpl, session=session)
            self.assertEqual(res.kind, "plan")
            self.assertIsNotNone(res.plan)
            self.assertGreaterEqual(len(res.plan.steps), 3)

    def test_required_flags_for_lesion_and_full_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            case_dir = Path(td) / "case_a"
            case_dir.mkdir(parents=True, exist_ok=True)

            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            session.set_case_input(str(case_dir))
            brain = Brain(tool_specs=[])

            lesion_tpl = {
                "REQUIRED": {
                    "domain": "prostate",
                    "request_type": "lesion",
                    "case_ref": "case.input",
                },
                "OPTIONAL": {"case_id": "case_a"},
            }
            lesion_res = brain.from_template(lesion_tpl, session=session)
            self.assertEqual(lesion_res.kind, "plan")
            lesion_steps = {s.tool_name: s for s in (lesion_res.plan.steps if lesion_res.plan else [])}
            self.assertTrue(lesion_steps["detect_lesion_candidates"].required)
            self.assertTrue(lesion_steps["extract_roi_features"].required)
            self.assertIn("segment_prostate", lesion_steps["detect_lesion_candidates"].depends_on)

            brain_tpl = {
                "REQUIRED": {
                    "domain": "brain",
                    "request_type": "full_pipeline",
                    "case_ref": "case.input",
                },
                "OPTIONAL": {"case_id": "case_a"},
            }
            brain_res = brain.from_template(brain_tpl, session=session)
            self.assertEqual(brain_res.kind, "plan")
            brain_steps = {s.tool_name: s for s in (brain_res.plan.steps if brain_res.plan else [])}
            self.assertTrue(brain_steps["extract_roi_features"].required)
            self.assertIn("brats_mri_segmentation", brain_steps["extract_roi_features"].depends_on)


if __name__ == "__main__":
    unittest.main()

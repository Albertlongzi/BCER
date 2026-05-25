from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from agent.langgraph.loop import plan_agent_dag


class _FakeSketchLLM:
    def __init__(self, raw: str) -> None:
        self._raw = raw

    def generate_with_schema(self, messages, guided_json):  # noqa: ANN001
        return self._raw

    def generate(self, messages):  # noqa: ANN001
        # Used only by blocked natural-language fallback path.
        return "planner blocked"


def _mk_brain_case(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    for name in ("T1.nii.gz", "T1c.nii.gz", "T2.nii.gz", "FLAIR.nii.gz"):
        (case_dir / name).write_text("x\n", encoding="utf-8")


def test_plan_agent_dag_sketch_mode_compiles_and_persists_artifacts() -> None:
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        case_dir = ws / "Brats18_case"
        _mk_brain_case(case_dir)

        sketch_raw = json.dumps(
            {
                "task": "short_segment_brain",
                "domain": "brain",
                "steps": [
                    {
                        "step_id": "id1",
                        "tool": "identify_sequences",
                        "depends_on": [],
                        "inputs": {"dicom_case_dir": "@case.input"},
                        "goal": "identify sequences",
                        "optional": False,
                    },
                    {
                        "step_id": "seg1",
                        "tool": "brats_mri_segmentation",
                        "depends_on": ["id1"],
                        "inputs": {
                            "t1_path": "@seq.T1",
                            "t1c_path": "@seq.T1c",
                            "t2_path": "@seq.T2",
                            "flair_path": "@seq.FLAIR",
                        },
                        "goal": "segment brain tumor",
                        "optional": False,
                    },
                ],
                "final_targets": ["seg1"],
                "planner_notes": "coarse segment sketch",
            }
        )

        with patch("agent.langgraph.loop._build_planner_llm", return_value=_FakeSketchLLM(sketch_raw)):
            dag = plan_agent_dag(
                goal="Segment the brain tumor and return masks only.",
                domain="brain",
                case_ref=str(case_dir),
                request_type="segment",
                llm_mode="server",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
                planning_mode="sketch",
                planner_task_id="short_segment_brain",
            )

        assert str(dag.planner_status) == "ready"
        assert dag.nodes
        assert dag.planner_metadata.get("planning_mode") == "sketch"
        assert isinstance(dag.planner_artifacts, dict) and dag.planner_artifacts
        for key in ("raw_output", "parsed_json", "compile_diagnostics", "compiled_dag"):
            assert key in dag.planner_artifacts, dag.planner_artifacts
            assert Path(str(dag.planner_artifacts[key])).exists()
        assert any("Planner planning_mode: sketch" in str(n) for n in dag.notes)
        assert any("Sketch compiler diagnostics:" in str(n) for n in dag.notes)


def test_plan_agent_dag_sketch_mode_strict_compile_failure_no_template_fallback() -> None:
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        case_dir = ws / "Brats18_case"
        _mk_brain_case(case_dir)

        bad_sketch_raw = json.dumps(
            {
                "task": "short_segment_brain",
                "domain": "brain",
                "steps": [
                    {
                        "step_id": "x1",
                        "tool": "sandbox_exec",
                        "depends_on": [],
                        "inputs": {},
                        "goal": "illegal tool",
                        "optional": False,
                    }
                ],
                "final_targets": ["x1"],
            }
        )

        with patch("agent.langgraph.loop._build_planner_llm", return_value=_FakeSketchLLM(bad_sketch_raw)):
            dag = plan_agent_dag(
                goal="Segment the brain tumor only.",
                domain="brain",
                case_ref=str(case_dir),
                request_type="segment",
                llm_mode="server",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
                planning_mode="sketch",
                planner_task_id="short_segment_brain",
            )

        assert str(dag.planner_status) == "blocked"
        assert dag.nodes == []
        assert isinstance(dag.planner_artifacts, dict) and "compile_diagnostics" in dag.planner_artifacts
        diag = json.loads(Path(str(dag.planner_artifacts["compile_diagnostics"])).read_text(encoding="utf-8"))
        assert diag.get("compile_ok") is False
        errs = " | ".join(str(x) for x in (diag.get("errors") or []))
        assert "Illegal tool" in errs


from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import os

from agent.langgraph import loop as planner_loop
from agent.langgraph.loop import plan_agent_dag
from agent.plans.template_loader import load_plan_template
from commands.dispatcher import ToolDispatcher
from core.plan_dag import AgentPlanDAG, CaseScope


class LangGraphPlannerDagTests(unittest.TestCase):
    @staticmethod
    def _tool_dep_signature(nodes) -> list[tuple[str, tuple[str, ...]]]:
        return [(str(n.tool_name), tuple(str(x) for x in (n.depends_on or []))) for n in nodes]

    def _assert_acyclic(self, nodes) -> None:
        by_id = {str(n.node_id): n for n in nodes}
        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(nid: str) -> None:
            if nid in visited:
                return
            if nid in visiting:
                raise AssertionError(f"Cycle detected at node: {nid}")
            visiting.add(nid)
            node = by_id.get(nid)
            if node is not None:
                for dep in (node.depends_on or []):
                    dep_s = str(dep)
                    self.assertIn(dep_s, by_id, f"Missing dependency node: {dep_s}")
                    dfs(dep_s)
            visiting.remove(nid)
            visited.add(nid)

        for node_id in by_id:
            dfs(node_id)

    def test_plan_agent_dag_is_planning_only_no_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "case_prostate"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "ADC.nii.gz").write_text("adc\n", encoding="utf-8")
            (case_dir / "DWI_b1400.nii.gz").write_text("dwi\n", encoding="utf-8")

            with patch.object(ToolDispatcher, "create_run", autospec=True) as mk_run, patch.object(
                ToolDispatcher, "dispatch", autospec=True
            ) as mk_dispatch:
                dag = plan_agent_dag(
                    goal="Generate a prostate MRI report with lesion review.",
                    domain="prostate",
                    case_ref=str(case_dir),
                    llm_mode="stub",
                    workspace_root=str(ws),
                    runs_root=str(ws / "runs"),
                )

            self.assertIsInstance(dag, AgentPlanDAG)
            self.assertEqual(mk_run.call_count, 0)
            self.assertEqual(mk_dispatch.call_count, 0)
            self.assertTrue(any("planning-only mode" in n for n in dag.notes))
            self.assertTrue(bool(str(dag.template_id or "").strip()))
            self.assertTrue(bool(str(dag.template_version or "").strip()))
            self.assertTrue(bool(str(dag.template_hash or "").strip()))

    def test_prostate_report_plan_marks_lesion_required_in_full_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-019_2"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "ADC.nii.gz").write_text("adc\n", encoding="utf-8")
            (case_dir / "DWI_b1400.nii.gz").write_text("dwi\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="Generate a prostate report from this case.",
                domain="prostate",
                case_ref=str(case_dir),
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            nodes = {n.tool_name: n for n in dag.nodes}

            self.assertIn("detect_lesion_candidates", nodes)
            self.assertTrue(bool(nodes["detect_lesion_candidates"].required))
            self.assertIn("generate_report", nodes)
            self.assertTrue(bool(nodes["generate_report"].required))

    def test_segmentation_only_goal_omits_report_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-seg-only"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "ADC.nii.gz").write_text("adc\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="Segment prostate only and output mask only; no report.",
                domain="prostate",
                case_ref=str(case_dir),
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertNotIn("generate_report", tool_names)
            self.assertNotIn("package_vlm_evidence", tool_names)

    def test_negated_report_phrase_prunes_report_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-seg-no-report"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "ADC.nii.gz").write_text("adc\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="Just do a segmentation. I do not need a report.",
                domain="prostate",
                case_ref=str(case_dir),
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertNotIn("generate_report", tool_names)
            self.assertNotIn("package_vlm_evidence", tool_names)

    def test_negated_report_goal_overrides_planner_report_mentions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-negated-overrides"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "ADC.nii.gz").write_text("adc\n", encoding="utf-8")

            mocked_plan = {"plan_text": "\n".join(["1. identify_sequences", "2. segment_prostate", "3. generate_report"])}
            with patch("agent.langgraph.loop.PlannerSubagent.run", return_value=mocked_plan):
                dag = plan_agent_dag(
                    goal="Segment only. Do not generate report.",
                    domain="prostate",
                    case_ref=str(case_dir),
                    llm_mode="stub",
                    workspace_root=str(ws),
                    runs_root=str(ws / "runs"),
                )
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertNotIn("generate_report", tool_names)
            self.assertNotIn("package_vlm_evidence", tool_names)

    def test_register_request_type_prunes_brain_segmentation_and_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "Brats18_CBICA_AAM_1"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T1c.nii.gz").write_text("t1c\n", encoding="utf-8")
            (case_dir / "T1.nii.gz").write_text("t1\n", encoding="utf-8")
            (case_dir / "T2.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "FLAIR.nii.gz").write_text("flair\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="no i want to register brain",
                domain="brain",
                case_ref=str(case_dir),
                request_type="register",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertIn("identify_sequences", tool_names)
            self.assertIn("register_to_reference", tool_names)
            self.assertNotIn("brats_mri_segmentation", tool_names)
            self.assertNotIn("extract_roi_features", tool_names)
            self.assertNotIn("generate_report", tool_names)
            reg_nodes = [n for n in dag.nodes if n.tool_name == "register_to_reference"]
            self.assertTrue(reg_nodes)
            self.assertTrue(all(bool(n.required) for n in reg_nodes))

    def test_registration_override_swaps_brain_register_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "Brats18_CBICA_AAM_1"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T1c.nii.gz").write_text("t1c\n", encoding="utf-8")
            (case_dir / "T2.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "FLAIR.nii.gz").write_text("flair\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="could you register T1C to T2 and Flair to T2 instead?",
                domain="brain",
                case_ref=str(case_dir),
                request_type="register",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            regs = [n for n in dag.nodes if n.tool_name == "register_to_reference"]
            self.assertEqual(len(regs), 2)
            reg_args = [{k: str(v) for k, v in (r.arguments or {}).items()} for r in regs]
            combos = {(ra.get("moving"), ra.get("fixed")) for ra in reg_args}
            self.assertIn(("@seq.T1c", "@seq.T2"), combos)
            self.assertIn(("@seq.FLAIR", "@seq.T2"), combos)
            labels = [str(r.label or "") for r in regs]
            self.assertTrue(any("Register T1c to T2" in lb for lb in labels))
            self.assertTrue(any("Register FLAIR to T2" in lb for lb in labels))
            self.assertTrue(any(str((r.arguments or {}).get("output_subdir") or "").endswith("/T1c") for r in regs))
            self.assertTrue(any(str((r.arguments or {}).get("output_subdir") or "").endswith("/FLAIR") for r in regs))
            self.assertTrue(all(bool(r.required) for r in regs))

    def test_blocked_plan_has_natural_response_and_no_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "acdc_like_case"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T1.nii.gz").write_text("t1\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="please do full cardiac pipeline and classify",
                domain="cardiac",
                case_ref=str(case_dir),
                request_type="full_pipeline",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            self.assertEqual(str(dag.planner_status), "blocked")
            self.assertTrue(bool(str(dag.natural_language_response or "").strip()))
            self.assertIn("CINE", str(dag.natural_language_response))
            self.assertEqual(len(dag.nodes), 0)
            self.assertTrue(any("missing modality: CINE" in str(x) for x in (dag.blocking_reasons or [])))

    def test_cardiac_patient061_4d_directory_evidence_unblocks_and_binds_cine_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "acdc_multiseq_patient061_ed"
            case_dir.mkdir(parents=True, exist_ok=True)
            cine_file = case_dir / "patient061_4d.nii.gz"
            cine_file.write_text("cine\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="please do full cardiac pipeline and classify",
                domain="cardiac",
                case_ref=str(case_dir),
                request_type="full_pipeline",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            self.assertEqual(str(dag.planner_status), "ready")
            self.assertFalse(any("missing modality: CINE" in str(x) for x in (dag.blocking_reasons or [])))
            seg_nodes = [n for n in dag.nodes if n.tool_name == "segment_cardiac_cine"]
            self.assertTrue(seg_nodes)
            self.assertEqual(str((seg_nodes[0].arguments or {}).get("cine_path") or ""), str(cine_file))
            self.assertTrue(any("Directory evidence" in str(n) and "CINE" in str(n) for n in (dag.notes or [])))

    def test_cardiac_symlink_candidate_keeps_lexical_case_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "acdc_multiseq_patient061_ed"
            external_root = ws / "acdc_test_group_4_per_group"
            case_dir.mkdir(parents=True, exist_ok=True)
            external_root.mkdir(parents=True, exist_ok=True)
            external_cine = external_root / "patient061_4d.nii.gz"
            external_cine.write_text("cine\n", encoding="utf-8")
            linked_cine = case_dir / "patient061_4d.nii.gz"
            os.symlink(str(external_cine), str(linked_cine))

            dag = plan_agent_dag(
                goal="please do full cardiac pipeline and classify",
                domain="cardiac",
                case_ref=str(case_dir),
                request_type="full_pipeline",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            seg_nodes = [n for n in dag.nodes if n.tool_name == "segment_cardiac_cine"]
            self.assertTrue(seg_nodes)
            self.assertEqual(str((seg_nodes[0].arguments or {}).get("cine_path") or ""), str(linked_cine))

    def test_metadata_qa_intent_builds_minimal_identify_then_rag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-meta-qa"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="I only want to read the dicom header information of prostate mri image.",
                domain="prostate",
                case_ref=str(case_dir),
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            self.assertEqual(str(dag.requested_request_type), "qa")
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertEqual(tool_names, ["identify_sequences", "rag_search"])
            ident_nodes = [n for n in dag.nodes if n.tool_name == "identify_sequences"]
            self.assertTrue(ident_nodes)
            ident_args = dict(ident_nodes[0].arguments or {})
            self.assertTrue(bool(ident_args.get("deep_dump")))
            self.assertTrue(bool(ident_args.get("require_pydicom")))
            qa_nodes = [n for n in dag.nodes if n.tool_name == "rag_search"]
            self.assertTrue(qa_nodes)
            self.assertTrue(bool(qa_nodes[0].required))
            self.assertIn("question", dict(qa_nodes[0].arguments or {}))
            self.assertEqual(
                str((qa_nodes[0].arguments or {}).get("case_state_path") or ""),
                "@runtime.case_state_path",
            )

    def test_custom_analysis_intent_builds_identify_then_sandbox_exec(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-custom-analysis"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal="run custom analysis with code interpreter on this case",
                domain="prostate",
                case_ref=str(case_dir),
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            self.assertEqual(str(dag.requested_request_type), "custom_analysis")
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertEqual(tool_names, ["identify_sequences", "sandbox_exec"])
            sand = [n for n in dag.nodes if n.tool_name == "sandbox_exec"]
            self.assertTrue(sand)
            self.assertTrue(bool(sand[0].required))
            self.assertIn("cmd", dict(sand[0].arguments or {}))

    def test_planner_optional_phrase_can_override_feature_required(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-optional"
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "T2w.nii.gz").write_text("t2\n", encoding="utf-8")
            (case_dir / "ADC.nii.gz").write_text("adc\n", encoding="utf-8")

            mocked_plan = {
                "plan_text": "\n".join(
                    [
                        "1. identify_sequences",
                        "2. register_to_reference (ADC)",
                        "3. segment_prostate",
                        "4. extract_roi_features (optional if segmentation is enough)",
                        "5. generate_report",
                    ]
                )
            }
            with patch("agent.langgraph.loop.PlannerSubagent.run", return_value=mocked_plan):
                dag = plan_agent_dag(
                    goal="Generate a prostate report.",
                    domain="prostate",
                    case_ref=str(case_dir),
                    llm_mode="stub",
                    workspace_root=str(ws),
                    runs_root=str(ws / "runs"),
                )
            feat_nodes = [n for n in dag.nodes if n.tool_name == "extract_roi_features"]
            self.assertTrue(feat_nodes)
            self.assertFalse(bool(feat_nodes[0].required))

    def test_denoise_request_type_builds_bm3d_flow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-denoise"
            case_dir.mkdir(parents=True, exist_ok=True)
            noisy = case_dir / "noisy_b1400.nii.gz"
            noisy.write_text("dummy\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal=f"Run denoise-only workflow and generate before/after PNG for {str(noisy)}",
                domain="prostate",
                case_ref=str(case_dir),
                request_type="denoise",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            self.assertEqual(str(dag.planner_status), "ready")
            self.assertEqual(str(dag.requested_request_type), "denoise")
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertEqual(tool_names, ["denoise_image_bm3d", "compare_nifti_slices"])
            denoise = dag.nodes[0]
            self.assertEqual(str((denoise.arguments or {}).get("input_nifti") or ""), str(noisy))
            self.assertTrue(bool(denoise.required))
            compare = dag.nodes[1]
            self.assertTrue(bool(compare.required))

    def test_super_resolution_request_type_builds_resample_flow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "sub-superres"
            case_dir.mkdir(parents=True, exist_ok=True)
            src = case_dir / "denoised.nii.gz"
            src.write_text("dummy\n", encoding="utf-8")

            dag = plan_agent_dag(
                goal=f"Do super-resolution 2x on {str(src)} and output PNG comparison",
                domain="prostate",
                case_ref=str(case_dir),
                request_type="super_resolution",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            self.assertEqual(str(dag.planner_status), "ready")
            self.assertEqual(str(dag.requested_request_type), "super_resolution")
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertEqual(tool_names, ["resample_image", "compare_nifti_slices"])
            resample = dag.nodes[0]
            self.assertEqual(str((resample.arguments or {}).get("input_nifti") or ""), str(src))
            self.assertIn("target_spacing", dict(resample.arguments or {}))
            self.assertTrue(bool(dag.nodes[1].required))

    def test_raw_recon_request_type_builds_recon_to_classify(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "cardiac_raw"
            case_dir.mkdir(parents=True, exist_ok=True)
            h5p = case_dir / "cine_sax.h5"
            h5p.write_bytes(b"dummy")

            dag = plan_agent_dag(
                goal=f"Run raw GRAPPA reconstruction then cardiac segmentation/classification on {str(h5p)}",
                domain="cardiac",
                case_ref=str(case_dir),
                request_type="raw_recon",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            self.assertEqual(str(dag.planner_status), "ready")
            self.assertEqual(str(dag.requested_request_type), "raw_recon")
            tool_names = [n.tool_name for n in dag.nodes]
            self.assertEqual(
                tool_names,
                ["reconstruct_grappa", "generate_qa_snapshot"],
            )
            recon = dag.nodes[0]
            qa = dag.nodes[1]
            self.assertEqual(str((recon.arguments or {}).get("h5_path") or ""), str(h5p))
            self.assertEqual(
                str((qa.arguments or {}).get("input_nifti") or ""),
                "@node.reconstruct_grappa_010.reconstructed_nifti",
            )
            self.assertTrue(all(bool(n.required) for n in dag.nodes))

    def test_raw_recon_without_h5_blocks_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            case_dir = ws / "cardiac_raw_empty"
            case_dir.mkdir(parents=True, exist_ok=True)

            dag = plan_agent_dag(
                goal="Run raw GRAPPA reconstruction and classify.",
                domain="cardiac",
                case_ref=str(case_dir),
                request_type="raw_recon",
                llm_mode="stub",
                workspace_root=str(ws),
                runs_root=str(ws / "runs"),
            )
            self.assertEqual(str(dag.planner_status), "blocked")
            self.assertEqual(len(dag.nodes), 0)
            self.assertTrue(any(".h5" in str(x).lower() for x in (dag.blocking_reasons or [])))

    def test_template_skeleton_matches_legacy_builder_structure(self) -> None:
        legacy_cases: list[tuple[str, str, list]] = []

        prostate_nodes, _ = planner_loop._build_prostate_plan_nodes(
            intent={"wants_report": True, "wants_features": False, "wants_lesion": False},
            modalities={"T2w": True, "ADC": True, "DWI": True},
            optional_overrides={"extract_roi_features": False, "detect_lesion_candidates": False},
        )
        legacy_cases.append(("prostate", "full_pipeline", prostate_nodes))

        brain_nodes, _ = planner_loop._build_brain_plan_nodes(
            intent={"wants_report": True, "wants_features": False, "wants_lesion": False},
            optional_overrides={"extract_roi_features": False},
        )
        legacy_cases.append(("brain", "full_pipeline", brain_nodes))

        cardiac_nodes, _ = planner_loop._build_cardiac_plan_nodes(
            intent={"wants_report": True, "wants_features": False, "wants_classification": True},
            optional_overrides={"extract_roi_features": False},
            case_ref=None,
            request_type="full_pipeline",
        )
        legacy_cases.append(("cardiac", "full_pipeline", cardiac_nodes))

        qa_nodes, _ = planner_loop._build_qa_plan_nodes(
            goal="metadata query",
            domain_name="prostate",
            llm_mode="stub",
            server_cfg=None,
            api_model=None,
            api_base_url=None,
        )
        legacy_cases.append(("prostate", "qa", qa_nodes))

        custom_nodes, _ = planner_loop._build_custom_analysis_plan_nodes(
            goal="run custom analysis",
            domain_name="prostate",
        )
        legacy_cases.append(("prostate", "custom_analysis", custom_nodes))

        denoise_nodes, _, denoise_blocking = planner_loop._build_denoise_plan_nodes(
            goal="run denoise workflow",
            case_scan={"nifti_files": [], "dicom_files": 1},
        )
        self.assertFalse(denoise_blocking)
        legacy_cases.append(("prostate", "denoise", denoise_nodes))

        super_nodes, _, super_blocking = planner_loop._build_super_resolution_plan_nodes(
            goal="run super-resolution workflow",
            case_scan={"nifti_files": [], "dicom_files": 1},
        )
        self.assertFalse(super_blocking)
        legacy_cases.append(("prostate", "super_resolution", super_nodes))

        raw_nodes, _, raw_blocking = planner_loop._build_raw_recon_plan_nodes(
            goal="/tmp/input.h5",
            domain_name="cardiac",
            case_ref=Path("/"),
        )
        self.assertFalse(raw_blocking)
        legacy_cases.append(("cardiac", "raw_recon", raw_nodes))

        for domain, request_type, legacy_nodes in legacy_cases:
            template = load_plan_template(domain=domain, request_type=request_type)
            self.assertEqual(
                self._tool_dep_signature(template.nodes),
                self._tool_dep_signature(legacy_nodes),
                f"template mismatch for {domain}/{request_type}",
            )

    def test_template_round_trip_dependencies_valid(self) -> None:
        template_cases = [
            ("prostate", "full_pipeline"),
            ("brain", "full_pipeline"),
            ("cardiac", "full_pipeline"),
            ("prostate", "qa"),
            ("prostate", "custom_analysis"),
            ("prostate", "denoise"),
            ("prostate", "super_resolution"),
            ("cardiac", "raw_recon"),
        ]

        for domain, request_type in template_cases:
            template = load_plan_template(domain=domain, request_type=request_type)
            dag = AgentPlanDAG(
                plan_id=f"template_{domain}_{request_type}",
                template_id=str(template.template_id),
                template_version=str(template.template_version),
                template_hash=str(template.template_hash),
                goal="",
                case_scope=CaseScope(
                    domain=domain,  # type: ignore[arg-type]
                    case_id="case_template",
                    case_ref="/tmp/case_template",
                    workspace_root="/tmp/workspace",
                    runs_root="/tmp/workspace/runs",
                ),
                nodes=[n.model_copy(deep=True) for n in template.nodes],
                notes=[f"template_source={template.source_path}"],
            )
            self.assertGreater(len(dag.nodes), 0)
            self._assert_acyclic(dag.nodes)


if __name__ == "__main__":
    unittest.main()

"""Report content contracts: measurements, domain scoping, recon provenance.

The defects these cover, all observed in real output:

1. The cardiac FINDINGS section was a list of absolute file paths plus four
   numbers.  Paths are not findings.
2. ``report.json`` carried ``lesion_assessment_meta`` -- PI-RADS-shaped, with
   ``adc_available: false`` and ``segmentation_usable: false`` -- on a cardiac
   case that has no prostate and no ADC.
3. Nothing in the report said how the images were produced.  The demo decimates
   fully sampled k-space to R=4 to exercise GRAPPA; a reader who is not told
   that will read it as a prospectively accelerated acquisition.
4. The impression restated the pipeline ("segmentation was successfully
   produced") rather than the study.

The fixture numbers are copied verbatim from a real run,
``graph-cardiac-20260725064122-ed803098`` (case ``cmr_p003_cine_sax``).
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from commands.schemas import ToolContext
from tools.report_generation import (
    _build_llm_impression,
    _build_reconstruction_provenance,
    _cardiac_findings_lines,
    _impression_numbers_grounded,
    _reconstruction_lines,
    generate_report,
)
import tools.report_generation as report_generation


# --- Real numbers from graph-cardiac-20260725064122-ed803098 ----------------

REAL_METRICS: Dict[str, Any] = {
    "voxel_volume_ml": 0.02184429265248957,
    "lv_edv_ml": 132.72592215652662,
    "lv_esv_ml": 42.42161633113474,
    "lv_ef_percent": 68.03818301514154,
    "rv_edv_ml": 165.09916386751618,
    "rv_esv_ml": 72.02063287525812,
    "rv_ef_percent": 56.37734850489547,
    "myo_ed_volume_ml": 86.00098017285144,
    "lv_mass_g": 90.30102918149402,
    "max_myo_thickness_mm": 11.212425185682635,
    "height_cm": None,
    "weight_kg": None,
    "bsa_m2": None,
    "lv_edvi_ml_m2": None,
    "rv_edvi_ml_m2": None,
    "lv_mass_index_g_m2": None,
    "local_contraction_proxy": {
        "valid_slices": 10,
        "abnormal_slices": 6,
        "abnormal_ratio_mean": 0.19534425554338702,
        "abnormal_slices_threshold_for_several": 4,
        "is_several_segments_abnormal": False,
    },
}

REAL_RECON_DATA: Dict[str, Any] = {
    "reconstructed_nifti": "/art/02_reconstruct-grappa/recon/reconstructed_cine.nii.gz",
    "zerofilled_nifti": "/art/02_reconstruct-grappa/zerofill/zerofilled_cine.nii.gz",
    "h5_path": "/data/MultiCoil/Center005_Siemens_30T_Vida_P003_cine_sax.h5",
    "mode": "grappa",
    "source_key": "kspace",
    "kspace_key": "kspace",
    "kspace_shape": [12, 14, 10, 448, 162],
    "n_coils": 10,
    "acs_lines_used": 24,
    "kernel_size": [5, 5],
    "frames_total": 168,
    "grappa_applied_frames": 168,
    "grappa_skipped_frames": 0,
    "grappa_failed_frames": 0,
    "output_shape": [224, 162, 14, 12],
    "nonspatial_axes": [0, 1],
    "nonspatial_order": [1, 0],
    "pixel_spacing": [1.517857, 1.798942, 8.0],
    "pixel_spacing_source": "argument",
    "undersample": {
        "applied": True,
        "pattern": "uniform_ky_plus_acs",
        "factor": 4,
        "acs_lines": 24,
        "ky_lines_total": 162,
        "ky_lines_kept": 59,
        "sampled_fraction": 0.364198,
        "net_acceleration": 2.7458,
        "note": (
            "input k-space was fully sampled and was decimated by this tool "
            "before reconstruction; it is NOT prospectively accelerated data"
        ),
    },
    "readout_crop": {
        "applied": True,
        "mode": "fov_mm",
        "axis": 0,
        "samples_before": 448,
        "samples_after": 224,
        "readout_spacing_mm": 1.517857,
        "fov_before_mm": 679.9999,
        "fov_after_mm": 340.0,
        "origin_shift_mm": 170.0,
        "outer_half_energy_fraction": 0.11996,
        "requested_fov_mm": 340.0,
        "removed_left": 112,
        "removed_right": 112,
    },
}


def _record(tool: str, data: Dict[str, Any], order: int) -> Dict[str, Any]:
    return {
        "call_id": f"{tool}-001",
        "ok": True,
        "consumable": True,
        "data": data,
        "stage_order": order,
    }


def _cardiac_case_state(
    *,
    domain: Optional[str] = "cardiac",
    with_recon: bool = True,
    undersample_applied: bool = True,
) -> Dict[str, Any]:
    recon = json.loads(json.dumps(REAL_RECON_DATA))
    if not undersample_applied:
        # A prospectively accelerated / as-acquired reconstruction: the tool
        # never decimated anything, so it emits only {"applied": False}.
        recon["undersample"] = {"applied": False}
    seg_dir = "/art/04_segment-cardiac-cine"
    seg_data = {
        "seg_path": f"{seg_dir}/nnunet_io/pred/reconstructed_cine_f01.nii.gz",
        "rv_mask_path": f"{seg_dir}/masks/reconstructed_cine_f01_rv_mask.nii.gz",
        "myo_mask_path": f"{seg_dir}/masks/reconstructed_cine_f01_myo_mask.nii.gz",
        "lv_mask_path": f"{seg_dir}/masks/reconstructed_cine_f01_lv_mask.nii.gz",
        "case_results": [
            {"case_id": f"reconstructed_cine_f{i:02d}"} for i in range(1, 13)
        ],
        "note": "Cardiac cine segmentation labels use ACDC convention: 1=RV, 2=MYO, 3=LV.",
    }
    cls_data = {
        "classification_path": "/art/05_classify-cardiac-cine-disease/cardiac_cine_classification.json",
        "predicted_group": "NOR",
        "ground_truth_group": None,
        "ground_truth_match": False,
        "needs_vlm_review": False,
        "metrics": REAL_METRICS,
        "phase_indices": {
            "ed_index_0based": 0,
            "es_index_0based": 6,
            "ed_frame_1based": 1,
            "es_frame_1based": 7,
            "source": "lv_volume_extrema",
        },
        "rule_trace": ["Rule: preserved biventricular EF, no dilation, no hypertrophy -> NOR."],
    }
    stage_outputs: Dict[str, Any] = {
        "identify": {
            "identify_sequences": [
                _record(
                    "identify_sequences",
                    {"mapping": {"CINE": recon["reconstructed_nifti"]}, "series": []},
                    2,
                )
            ]
        },
        "segment": {"segment_cardiac_cine": [_record("segment_cardiac_cine", seg_data, 4)]},
        "classify": {
            "classify_cardiac_cine_disease": [
                _record("classify_cardiac_cine_disease", cls_data, 5)
            ]
        },
    }
    if with_recon:
        stage_outputs["reconstruct"] = {
            "reconstruct_grappa": [_record("reconstruct_grappa", recon, 1)]
        }
    state: Dict[str, Any] = {
        "case_id": "cmr_p003_cine_sax",
        "run_id": "graph-cardiac-test",
        "stage_outputs": stage_outputs,
    }
    if domain is not None:
        state["metadata"] = {"domain": domain}
    return state


def _prostate_case_state() -> Dict[str, Any]:
    return {
        "case_id": "prostate_case_001",
        "run_id": "graph-prostate-test",
        "metadata": {"domain": "prostate"},
        "stage_outputs": {
            "identify": {
                "identify_sequences": [
                    _record(
                        "identify_sequences",
                        {
                            "mapping": {
                                "T2w": "/art/t2w.nii.gz",
                                "ADC": "/art/adc.nii.gz",
                                "DWI": "/art/dwi.nii.gz",
                            },
                            "series": [],
                        },
                        1,
                    )
                ]
            },
            "segment": {
                "segment_prostate": [
                    _record(
                        "segment_prostate",
                        {"prostate_mask_path": "/art/prostate_mask.nii.gz"},
                        2,
                    )
                ]
            },
        },
    }


class _RunMixin:
    def _run(self, state: Dict[str, Any], **args: Any) -> Dict[str, Any]:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: None)
        run_dir = tmp / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        cs_path = run_dir / "case_state.json"
        cs_path.write_text(json.dumps(state), encoding="utf-8")
        artifacts = tmp / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        ctx = ToolContext(
            case_id=str(state.get("case_id")),
            run_id=str(state.get("run_id")),
            run_dir=run_dir,
            artifacts_dir=artifacts,
            case_state_path=cs_path,
        )
        call_args: Dict[str, Any] = {"case_state_path": str(cs_path)}
        call_args.update(args)
        generate_report(call_args, ctx)
        return {
            "clinical": (artifacts / "report" / "clinical_report.md").read_text(encoding="utf-8"),
            "report_md": (artifacts / "report" / "report.md").read_text(encoding="utf-8"),
            "report_json": json.loads((artifacts / "report" / "report.json").read_text(encoding="utf-8")),
        }

    @staticmethod
    def _section(text: str, header: str) -> str:
        """Return the body of a ``**HEADER:**`` block."""
        lines = text.splitlines()
        try:
            start = lines.index(f"**{header}:**")
        except ValueError:
            return ""
        body: List[str] = []
        for line in lines[start + 1 :]:
            if line.startswith("**") and line.endswith(":**"):
                break
            body.append(line)
        return "\n".join(body).strip()


class CardiacFindingsAreMeasurementsTests(_RunMixin, unittest.TestCase):
    """FINDINGS must state what was measured, not where files landed."""

    def test_findings_contain_no_file_paths(self) -> None:
        out = self._run(_cardiac_case_state())
        findings = self._section(out["clinical"], "FINDINGS")
        self.assertTrue(findings, "cardiac report has no FINDINGS section")
        for marker in ("/art/", ".nii.gz", ".json", "nnunet_io", "_mask"):
            self.assertNotIn(
                marker, findings, f"FINDINGS still contains path-like text {marker!r}:\n{findings}"
            )
        # Any absolute path, whatever its shape.
        hits = re.findall(r"(?:^|[\s`(])/[A-Za-z0-9_.\-]+/\S*", findings)
        self.assertEqual(hits, [], f"FINDINGS still contains absolute paths: {hits}")

    def test_paths_moved_to_provenance_section(self) -> None:
        out = self._run(_cardiac_case_state())
        prov = self._section(out["clinical"], "PROVENANCE / ARTIFACTS")
        self.assertIn("reconstructed_cine_f01_lv_mask.nii.gz", prov)
        self.assertIn("Center005_Siemens_30T_Vida_P003_cine_sax.h5", prov)

    def test_findings_report_the_measured_quantities(self) -> None:
        out = self._run(_cardiac_case_state())
        findings = self._section(out["clinical"], "FINDINGS")
        for expected in (
            "EDV 132.7 mL",
            "ESV 42.4 mL",
            "EF 68.0%",
            "EDV 165.1 mL",
            "ESV 72.0 mL",
            "EF 56.4%",
            "LV mass 90.3 g",
            "maximum wall thickness 11.2 mm",
            "end-diastolic volume 86.0 mL",
            "end-diastole at frame 1 of 12",
            "end-systole at frame 7 of 12",
            "maximum and minimum segmented LV blood-pool volume",
            "14 short-axis slices x 12 cardiac phases",
            "slice thickness 8.0 mm",
            "measurable on 10 short-axis slices",
            "NOR",
        ):
            self.assertIn(expected, findings, f"missing measurement {expected!r}:\n{findings}")

    def test_absent_measurements_are_not_invented(self) -> None:
        """height/weight/BSA are null in the real run; nothing may be padded."""
        out = self._run(_cardiac_case_state())
        findings = self._section(out["clinical"], "FINDINGS")
        self.assertIn("not reported: height and weight were not available", findings)
        for forbidden in ("EDVI 0", "mL/m^2", "g/m^2"):
            self.assertNotIn(
                f"{forbidden} ", findings.replace("LV/RV EDVI, LV mass index", "")
            )

    def test_findings_degrade_honestly_without_measurements(self) -> None:
        state = _cardiac_case_state()
        del state["stage_outputs"]["classify"]
        del state["stage_outputs"]["segment"]
        out = self._run(state)
        findings = self._section(out["clinical"], "FINDINGS")
        self.assertIn("No cardiac cine segmentation or ventricular measurements", findings)
        for token in ("132.7", "68.0", "NOR"):
            self.assertNotIn(token, findings)

    def test_impression_describes_the_study_not_the_pipeline(self) -> None:
        out = self._run(_cardiac_case_state())
        impression = self._section(out["clinical"], "IMPRESSION")
        self.assertNotIn("segmentation was successfully produced", impression)
        self.assertIn("LV EF 68.0%", impression)
        self.assertIn("RV EF 56.4%", impression)
        self.assertIn("NOR", impression)


class DomainScopingTests(_RunMixin, unittest.TestCase):
    """A cardiac report must not carry prostate structures, and vice versa."""

    PROSTATE_ONLY = ("lesion_assessment_meta",)
    CARDIAC_ONLY = ("cardiac_assessment", "cardiac_t1_feature_analysis")

    def test_cardiac_report_json_has_no_prostate_blocks(self) -> None:
        out = self._run(_cardiac_case_state())
        rj = out["report_json"]
        for key in self.PROSTATE_ONLY:
            self.assertNotIn(key, rj, f"cardiac report.json still carries {key!r}")
        blob = json.dumps(rj).lower()
        for token in ("pirads", "pi-rads", "lesion", "prostate", "adc_available"):
            self.assertNotIn(token, blob, f"cardiac report.json mentions {token!r}")
        self.assertEqual(rj.get("domain"), "cardiac")

    def test_prostate_report_json_keeps_its_validated_block(self) -> None:
        out = self._run(_prostate_case_state())
        rj = out["report_json"]
        self.assertIn("lesion_assessment_meta", rj)
        meta = rj["lesion_assessment_meta"]
        for key in (
            "evidence_tier",
            "lesion_tool_status",
            "adc_available",
            "segmentation_usable",
            "final_overall_pirads",
            "final_overall_pirads_range",
        ):
            self.assertIn(key, meta)
        for key in self.CARDIAC_ONLY:
            self.assertNotIn(key, rj, f"prostate report.json carries cardiac block {key!r}")
        self.assertEqual(rj.get("domain"), "prostate")

    def test_domain_read_from_case_state_metadata_when_not_passed(self) -> None:
        """`metadata.domain` is a real field; it beats guessing from tool names."""
        out = self._run(_cardiac_case_state())  # no domain= argument
        rj = out["report_json"]
        self.assertEqual(rj.get("domain"), "cardiac")
        self.assertEqual(rj.get("domain_source"), "case_state.metadata.domain")
        self.assertNotIn("lesion_assessment_meta", rj)

    def test_explicit_domain_argument_wins(self) -> None:
        out = self._run(_cardiac_case_state(), domain="cardiac")
        self.assertEqual(out["report_json"].get("domain_source"), "argument")

    def test_missing_metadata_domain_falls_back_to_prostate_default(self) -> None:
        out = self._run(_cardiac_case_state(domain=None))
        rj = out["report_json"]
        self.assertEqual(rj.get("domain"), "prostate")
        self.assertEqual(rj.get("domain_source"), "default")

    def test_stage_status_is_domain_scoped(self) -> None:
        cardiac = self._run(_cardiac_case_state())["report_json"]["stage_status"]
        self.assertNotIn("segment_prostate", cardiac)
        self.assertIs(cardiac["segment_cardiac_cine"], True)
        self.assertIs(cardiac["classify_cardiac_cine_disease"], True)
        self.assertIs(cardiac["reconstruct_grappa"], True)
        prostate = self._run(_prostate_case_state())["report_json"]["stage_status"]
        self.assertIn("segment_prostate", prostate)
        self.assertNotIn("segment_cardiac_cine", prostate)


class ReconstructionProvenanceTests(_RunMixin, unittest.TestCase):
    """The report must record how the images were produced."""

    def test_provenance_block_carries_the_real_numbers(self) -> None:
        prov = _build_reconstruction_provenance(True, REAL_RECON_DATA)
        self.assertTrue(prov["ok"])
        self.assertEqual(prov["source_h5_name"], "Center005_Siemens_30T_Vida_P003_cine_sax.h5")
        self.assertEqual(prov["undersample"]["factor"], 4)
        self.assertEqual(prov["undersample"]["acs_lines"], 24)
        self.assertIs(prov["undersample"]["retrospective"], True)
        self.assertEqual(prov["readout_crop"]["samples_before"], 448)
        self.assertEqual(prov["readout_crop"]["samples_after"], 224)
        self.assertEqual(prov["n_slices"], 14)
        self.assertEqual(prov["n_phases"], 12)

    def test_report_states_retrospective_undersampling(self) -> None:
        out = self._run(_cardiac_case_state())
        acq = self._section(out["clinical"], "ACQUISITION AND RECONSTRUCTION")
        self.assertIn("RETROSPECTIVE UNDERSAMPLING", acq)
        self.assertIn("R=4", acq)
        self.assertIn("24 ACS lines", acq)
        self.assertIn("59 of 162 ky lines retained", acq)
        self.assertIn("NOT prospectively accelerated", acq)
        self.assertIn("GRAPPA", acq)
        self.assertIn("448 -> 224 readout samples", acq)
        self.assertIn("680.0 -> 340.0 mm", acq)
        self.assertIn("Center005_Siemens_30T_Vida_P003_cine_sax.h5", acq)

    def test_impression_repeats_the_retrospective_caveat(self) -> None:
        out = self._run(_cardiac_case_state())
        impression = self._section(out["clinical"], "IMPRESSION")
        self.assertIn("RETROSPECTIVE", impression)
        self.assertIn("not prospectively accelerated", impression)

    def test_as_acquired_reconstruction_makes_no_retrospective_claim(self) -> None:
        out = self._run(_cardiac_case_state(undersample_applied=False))
        text = out["clinical"]
        self.assertNotIn("RETROSPECTIVE", text)
        self.assertIn("k-space reconstructed as stored", text)
        us = out["report_json"]["reconstruction"]["undersample"]
        self.assertIs(us["applied"], False)
        self.assertNotIn("retrospective", us)

    def test_no_recon_stage_means_no_reconstruction_claims(self) -> None:
        out = self._run(_cardiac_case_state(with_recon=False))
        self.assertNotIn("ACQUISITION AND RECONSTRUCTION", out["clinical"])
        self.assertNotIn("RETROSPECTIVE", out["clinical"])
        self.assertNotIn("reconstruction", out["report_json"])
        # Slice/phase counts came from the recon block; they must not be guessed.
        findings = self._section(out["clinical"], "FINDINGS")
        self.assertNotIn("short-axis slices x", findings)

    def test_slice_phase_split_only_claimed_for_documented_layout(self) -> None:
        data = json.loads(json.dumps(REAL_RECON_DATA))
        data["nonspatial_order"] = [0, 1]  # not the documented (slices, frames) layout
        prov = _build_reconstruction_provenance(True, data)
        self.assertIsNone(prov["n_slices"])
        self.assertIsNone(prov["n_phases"])

    def test_failed_recon_record_yields_no_provenance(self) -> None:
        self.assertFalse(_build_reconstruction_provenance(False, REAL_RECON_DATA)["ok"])
        self.assertFalse(_build_reconstruction_provenance(None, {})["ok"])
        self.assertEqual(_reconstruction_lines({"ok": False}), [])


class LlmImpressionGuardTests(_RunMixin, unittest.TestCase):
    """The LLM impression is off by default and never bypasses the guards."""

    MEASUREMENTS = [
        "- LV ejection fraction: 68.0 %",
        "- RV ejection fraction: 56.4 %",
        "- LV end-diastolic volume: 132.7 mL",
    ]
    ALLOWED = [68.03818301514154, 56.37734850489547, 132.72592215652662]

    def test_default_is_the_deterministic_template(self) -> None:
        out = self._run(_cardiac_case_state())
        self.assertEqual(out["report_json"]["impression"]["source"], "template")
        self.assertEqual(out["report_json"]["impression"]["requested"], "template")

    def test_numeric_guard_accepts_measured_and_rounded_values(self) -> None:
        ok, bad = _impression_numbers_grounded(
            "LV EF is 68.0% (68%) and LV EDV is 132.7 mL.", self.ALLOWED
        )
        self.assertTrue(ok, f"rejected a measured value: {bad}")

    def test_numeric_guard_rejects_a_number_nobody_measured(self) -> None:
        ok, bad = _impression_numbers_grounded("LV EF is 68.0% and LV mass is 210.0 g.", self.ALLOWED)
        self.assertFalse(ok)
        self.assertAlmostEqual(bad, 210.0)

    def _patched(self, reply: Optional[str], err: Optional[str] = None) -> Any:
        def fake(**kwargs: Any):
            return reply, err

        return fake

    def test_transport_failure_falls_back_to_template(self) -> None:
        report_generation._ollama_no_think_chat = self._patched(None, "transport_error:URLError")
        self.addCleanup(self._restore)
        out = self._run(_cardiac_case_state(), impression_mode="llm")
        imp = out["report_json"]["impression"]
        self.assertEqual(imp["source"], "template")
        self.assertEqual(imp["fallback_reason"], "transport_error:URLError")
        self.assertIn("LV EF 68.0%", self._section(out["clinical"], "IMPRESSION"))

    def test_empty_completion_falls_back_to_template(self) -> None:
        report_generation._ollama_no_think_chat = self._patched(None, "empty_completion")
        self.addCleanup(self._restore)
        out = self._run(_cardiac_case_state(), impression_mode="llm")
        self.assertEqual(out["report_json"]["impression"]["source"], "template")
        self.assertEqual(out["report_json"]["impression"]["fallback_reason"], "empty_completion")

    def test_ungrounded_number_falls_back_to_template(self) -> None:
        report_generation._ollama_no_think_chat = self._patched(
            "LV EF is 68.0% with an LV mass of 210.0 g."
        )
        self.addCleanup(self._restore)
        out = self._run(_cardiac_case_state(), impression_mode="llm")
        imp = out["report_json"]["impression"]
        self.assertEqual(imp["source"], "template")
        self.assertTrue(str(imp["fallback_reason"]).startswith("ungrounded_number"))
        self.assertNotIn("210.0", out["clinical"])

    def test_unsupported_clinical_entity_falls_back_to_template(self) -> None:
        report_generation._ollama_no_think_chat = self._patched(
            "LV EF is 68.0%. There is no late gadolinium enhancement and no myocardial scar."
        )
        self.addCleanup(self._restore)
        out = self._run(_cardiac_case_state(), impression_mode="llm")
        imp = out["report_json"]["impression"]
        self.assertEqual(imp["source"], "template")
        self.assertTrue(str(imp["fallback_reason"]).startswith("unsupported_entity"))
        self.assertNotIn("gadolinium", out["clinical"])

    def test_grounded_completion_is_used_and_attributed(self) -> None:
        report_generation._ollama_no_think_chat = self._patched(
            "The left ventricular ejection fraction is 68.0% and the right "
            "ventricular ejection fraction is 56.4%. The rule-based group is NOR."
        )
        self.addCleanup(self._restore)
        out = self._run(
            _cardiac_case_state(), impression_mode="llm", impression_llm_model="qwen3.6:27b"
        )
        imp = out["report_json"]["impression"]
        self.assertEqual(imp["source"], "llm")
        self.assertEqual(imp["model"], "qwen3.6:27b")
        # The deterministic template is preserved alongside it.
        self.assertTrue(any("LV EF 68.0%" in line for line in imp["template_lines"]))
        text = self._section(out["clinical"], "IMPRESSION")
        self.assertIn("ejection fraction is 68.0%", text)
        self.assertIn("qwen3.6:27b", text)
        self.assertIn("RETROSPECTIVE", text)

    def test_llm_never_runs_without_measurements(self) -> None:
        calls: List[int] = []

        def fake(**kwargs: Any):
            calls.append(1)
            return "should not be used", None

        report_generation._ollama_no_think_chat = fake
        self.addCleanup(self._restore)
        res = _build_llm_impression(
            base_url="http://127.0.0.1:1",
            model="m",
            timeout_s=1,
            measurement_lines=[],
            allowed_numbers=[],
        )
        self.assertFalse(res["ok"])
        self.assertEqual(res["reason"], "no_measurements")
        self.assertEqual(calls, [])

    @staticmethod
    def _restore() -> None:
        report_generation._ollama_no_think_chat = _ORIGINAL_CHAT


_ORIGINAL_CHAT = report_generation._ollama_no_think_chat


class CardiacFindingsUnitTests(unittest.TestCase):
    """Direct checks on the findings builder, independent of file plumbing."""

    def test_missing_metric_is_omitted_not_zero_filled(self) -> None:
        cls_info = {
            "predicted_group": "NOR",
            "metrics": {"lv_ef_percent": 68.03818301514154},
            "phase_indices": {},
        }
        lines = _cardiac_findings_lines(
            cls_info=cls_info,
            seg_info={"ok": True, "cardiac_seg_path": "/x/seg.nii.gz"},
            recon_prov={"ok": False},
            n_segmented_frames=None,
        )
        joined = "\n".join(lines)
        self.assertIn("EF 68.0%", joined)
        self.assertNotIn("EDV", joined)
        self.assertNotIn("mass", joined)
        self.assertNotIn("0.0 mL", joined)
        self.assertNotIn("Cardiac phases", joined)


if __name__ == "__main__":
    unittest.main()

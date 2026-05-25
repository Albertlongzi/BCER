from __future__ import annotations

from pathlib import Path

from benchmark.benchmark_runner import (
    FaultInjectionSpecV2,
    FaultInjectorV2,
    _normalize_case_row,
    _compute_err,
    _compute_safe_halt,
    _compute_tcr,
    _fault_profile_status,
)


def test_fault_injector_token_mutation_applies_once() -> None:
    injector = FaultInjectorV2(
        spec=FaultInjectionSpecV2(
            enabled=True,
            fault="token_mutation",
            seed=7,
            profile={
                "token_pairs": [
                    ["@runtime.case_state_path", "@state.path"],
                ]
            },
        )
    )

    args = {"case_state_path": "@runtime.case_state_path"}
    out, mutated = injector.maybe_mutate_arguments(
        tool_name="generate_report",
        arguments=args,
        source="cerebellum.pre_guard",
    )

    assert mutated is True
    assert out["case_state_path"] == "@state.path"
    assert injector.applied is True
    assert len(injector.events) == 1

    out2, mutated2 = injector.maybe_mutate_arguments(
        tool_name="generate_report",
        arguments=out,
        source="cerebellum.pre_guard",
    )
    assert mutated2 is False
    assert out2["case_state_path"] == "@state.path"


def test_fault_injector_path_mutation_scope_escape() -> None:
    injector = FaultInjectorV2(
        spec=FaultInjectionSpecV2(
            enabled=True,
            fault="path_mutation",
            seed=13,
            profile={"modes": ["scope_escape"]},
        )
    )

    args = {"dicom_case_dir": "/tmp/in_scope/case"}
    out, mutated = injector.maybe_mutate_arguments(
        tool_name="identify_sequences",
        arguments=args,
        source="cerebellum.pre_guard",
    )

    assert mutated is True
    assert str(out["dicom_case_dir"]).startswith("/tmp/benchmark_scope_escape/")
    assert injector.applied is True


def test_fault_injector_path_mutation_defaults_to_missing() -> None:
    injector = FaultInjectorV2(
        spec=FaultInjectionSpecV2(
            enabled=True,
            fault="path_mutation",
            seed=13,
            profile={"modes": ["missing", "scope_escape"]},
        )
    )

    args = {"dicom_case_dir": "/tmp/in_scope/case"}
    out, mutated = injector.maybe_mutate_arguments(
        tool_name="identify_sequences",
        arguments=args,
        source="cerebellum.pre_guard",
    )

    assert mutated is True
    assert str(out["dicom_case_dir"]).startswith("/tmp/in_scope/case/")
    assert "__benchmark_missing__" in str(out["dicom_case_dir"])
    assert str(out["dicom_case_dir"]).endswith(".missing")
    assert "/tmp/benchmark_scope_escape/" not in str(out["dicom_case_dir"])
    assert injector.applied is True


def test_fault_injector_token_mutation_case_state_alias_fallback() -> None:
    injector = FaultInjectorV2(
        spec=FaultInjectionSpecV2(
            enabled=True,
            fault="token_mutation",
            seed=3,
            profile={},
        )
    )

    args = {"case_state_path": "/tmp/run/case_state.json"}
    out, mutated = injector.maybe_mutate_arguments(
        tool_name="generate_report",
        arguments=args,
        source="cerebellum.pre_guard",
    )

    assert mutated is True
    assert out["case_state_path"] == "@state.path"
    assert injector.applied is True


def test_compute_err_not_applicable_is_excluded() -> None:
    injector = FaultInjectorV2(spec=FaultInjectionSpecV2(enabled=False, fault="semantic_swap", seed=0, profile={}))
    err = _compute_err(
        fault="semantic_swap",
        injector=injector,
        success=False,
        not_applicable=True,
    )
    assert err["eligible"] is False
    assert err["value"] is None
    assert err["not_applicable"] is True


def test_compute_safe_halt_for_scope_violation() -> None:
    injector = FaultInjectorV2(spec=FaultInjectionSpecV2(enabled=True, fault="scope_violation", seed=0, profile={}))
    injector.applied = True
    safe = _compute_safe_halt(
        fault="scope_violation",
        injector=injector,
        success=False,
        actual_status="failure",
        observed_error_types=["ScopeViolation"],
        reflection_decisions=[{"action": "halt", "reason": "hard_limit_scope"}],
        not_applicable=False,
    )
    assert safe["eligible"] is True
    assert safe["pass"] is True
    assert safe["value"] == 1.0


def test_fault_profile_status_marks_not_applicable_when_disabled() -> None:
    contract = {
        "fault_profiles": {
            "semantic_swap": {"enabled": False},
        }
    }
    status = _fault_profile_status(contract=contract, fault="semantic_swap")
    assert status["not_applicable"] is True
    assert status["not_applicable_reason"] == "fault_profile_disabled"
    assert status["profile_enabled"] is False


def test_compute_tcr_full_completion(tmp_path: Path) -> None:
    report_path = tmp_path / "artifacts" / "report" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{}\n", encoding="utf-8")

    case_state = {
        "stage_outputs": {
            "identify": {
                "identify_sequences": [
                    {"ok": True, "data": {"mapping": {"T2w": "/a/b/c.nii.gz"}}}
                ]
            },
            "report": {
                "generate_report": [
                    {
                        "ok": True,
                        "data": {
                            "report_json_path": str(report_path),
                        },
                    }
                ]
            },
        }
    }

    contract = {
        "required_stage_success": ["identify_sequences", "generate_report"],
        "required_artifacts": [
            {
                "id": "report_json",
                "tool": "generate_report",
                "data_key": "report_json_path",
                "must_exist": True,
            }
        ],
    }

    tcr = _compute_tcr(contract=contract, case_state=case_state, run_dir=tmp_path)
    assert tcr["pass"] is True
    assert tcr["ratio"] == 1.0
    assert tcr["completed"] == tcr["total"]


def test_normalize_case_row_preserves_input_aliases() -> None:
    row = _normalize_case_row(
        {
            "case_id": "brain_case_001",
            "domain": "brain",
            "case_root": "/tmp/case",
            "input_format": "nifti",
            "modalities": {"T1": True},
            "supports_tasks": ["short_segment_brain"],
            "input_aliases": {"t1_nifti": "/tmp/case/T1.nii.gz"},
        }
    )
    assert row["input_aliases"] == {"t1_nifti": "/tmp/case/T1.nii.gz"}



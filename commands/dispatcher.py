from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .registry import ToolRegistry
from .schemas import (
    ArtifactRef,
    CaseState,
    ExecutionLogEntry,
    ToolCall,
    ToolContext,
    ToolError,
    ToolResult,
    environment_snapshot,
    normalize_artifacts,
    now_iso,
    sha256_json,
)
from .tool_runtime import (
    ToolRuntimeConfig,
    load_tool_runtime_config,
    run_tool_subprocess,
    should_dispatch_subprocess,
)


class SchemaValidationError(ValueError):
    pass


def _is_type(value: Any, json_type: str) -> bool:
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "object":
        return isinstance(value, dict)
    if json_type == "array":
        return isinstance(value, list)
    return True  # unknown type -> skip


def validate_args_minimal(schema: Dict[str, Any], args: Dict[str, Any]) -> None:
    """
    Minimal JSON-schema-ish validation.
    Supports: type=object, required, properties, per-property type.
    """
    if not schema:
        return
    if schema.get("type") and schema.get("type") != "object":
        return
    if not isinstance(args, dict):
        raise SchemaValidationError("Tool arguments must be a JSON object (dict).")

    required = schema.get("required", [])
    for key in required:
        if key not in args:
            raise SchemaValidationError(f"Missing required argument: {key}")

    props = schema.get("properties", {})
    for key, prop_schema in props.items():
        if key not in args:
            continue
        expected_type = prop_schema.get("type")
        if expected_type and not _is_type(args[key], expected_type):
            raise SchemaValidationError(
                f"Argument '{key}' expected type '{expected_type}', got '{type(args[key]).__name__}'"
            )


def validate_output_minimal(payload: Any) -> None:
    """
    Very-minimal tool output validation.
    Enforces:
    - tool returns either ToolResult or a dict payload
    - payload['data'] is a dict (if payload dict)
    - payload['artifacts'|'source_artifacts'|'generated_artifacts'] are lists if present (if payload dict)
    - payload['warnings'] is a list if present (if payload dict)
    """
    if isinstance(payload, ToolResult):
        if not isinstance(payload.data, dict):
            raise SchemaValidationError("ToolResult.data must be a dict")
        if not isinstance(payload.artifacts, list):
            raise SchemaValidationError("ToolResult.artifacts must be a list")
        if not isinstance(payload.warnings, list):
            raise SchemaValidationError("ToolResult.warnings must be a list")
        return

    if not isinstance(payload, dict):
        raise SchemaValidationError(f"Tool must return dict payload or ToolResult, got: {type(payload).__name__}")

    if "data" in payload and not isinstance(payload["data"], dict):
        raise SchemaValidationError("Tool payload 'data' must be a dict")
    for k in ("artifacts", "source_artifacts", "generated_artifacts"):
        if k in payload and not isinstance(payload[k], list):
            raise SchemaValidationError(f"Tool payload '{k}' must be a list")
    if "warnings" in payload and not isinstance(payload["warnings"], list):
        raise SchemaValidationError("Tool payload 'warnings' must be a list")


class ToolDispatcher:
    def __init__(
        self,
        registry: ToolRegistry,
        runs_root: Path,
        *,
        runtime_config: Optional[ToolRuntimeConfig] = None,
    ) -> None:
        self.registry = registry
        self.runs_root = runs_root
        self.runtime_config = runtime_config or load_tool_runtime_config()

    def list_tools(self) -> List[Dict[str, Any]]:
        return self.registry.list_specs()

    def create_run(self, case_id: str, run_id: str) -> Tuple[CaseState, ToolContext]:
        run_dir = self.runs_root / case_id / run_id
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        case_state_path = run_dir / "case_state.json"

        state = CaseState(case_id=case_id, run_id=run_id)
        state.write_json(case_state_path)

        ctx = ToolContext(
            case_id=case_id,
            run_id=run_id,
            run_dir=run_dir,
            artifacts_dir=artifacts_dir,
            case_state_path=case_state_path,
        )
        return state, ctx

    def _append_log(self, run_dir: Path, entry: ExecutionLogEntry) -> None:
        log_path = run_dir / "execution_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(entry.to_jsonl() + "\n")

    def dispatch(self, call: ToolCall, state: CaseState, ctx: ToolContext) -> ToolResult:
        tool = self.registry.get(call.tool_name)
        resolved_name = tool.spec.name
        domain = None
        try:
            if isinstance(state.metadata, dict):
                domain = str(state.metadata.get("domain") or "").strip().lower() or None
        except Exception:
            domain = None

        def _domain_allows(tool_name: str, domain_name: Optional[str]) -> bool:
            if not domain_name:
                return True
            utility_tools = {
                "rag_search",
                "search_repo",
                "sandbox_exec",
                "compare_nifti_slices",
                "generate_qa_snapshot",
            }
            if tool_name in utility_tools:
                return True
            brain_tools = {
                "ingest_dicom_to_nifti",
                "identify_sequences",
                "register_to_reference",
                "alignment_qc",
                "materialize_registration",
                "brats_mri_segmentation",
                "classify_brain_glioma_grade",
                "denoise_image_bm3d",
                "resample_image",
                "extract_roi_features",
                "package_vlm_evidence",
                "generate_report",
            }
            cardiac_tools = {
                "ingest_dicom_to_nifti",
                "identify_sequences",
                "reconstruct_grappa",
                "segment_cardiac_cine",
                "classify_cardiac_cine_disease",
                "denoise_image_bm3d",
                "resample_image",
                "extract_roi_features",
                "package_vlm_evidence",
                "generate_report",
            }
            prostate_tools = {
                "ingest_dicom_to_nifti",
                "identify_sequences",
                "register_to_reference",
                "alignment_qc",
                "materialize_registration",
                "segment_prostate",
                "denoise_image_bm3d",
                "resample_image",
                "extract_roi_features",
                "detect_lesion_candidates",
                "correct_prostate_distortion",
                "package_vlm_evidence",
                "generate_report",
            }
            if domain_name == "brain":
                return tool_name in brain_tools
            if domain_name == "cardiac":
                return tool_name in cardiac_tools
            if domain_name == "prostate":
                return tool_name in prostate_tools
            return True

        if not _domain_allows(resolved_name, domain):
            raise SchemaValidationError(
                f"Tool '{call.tool_name}' is not allowed for domain '{domain}'."
            )
        raw_args = dict(call.arguments or {})
        input_schema = tool.spec.input_schema or {}
        removed_args: List[str] = []
        if isinstance(input_schema, dict):
            props = input_schema.get("properties")
            if isinstance(props, dict) and props:
                filtered = {k: v for k, v in raw_args.items() if k in props}
                removed_args = sorted(set(raw_args.keys()) - set(filtered.keys()))
                raw_args = filtered
        input_hash = sha256_json(raw_args)
        start = time.time()

        def _norm_path(val: Any) -> str:
            try:
                return str(Path(str(val)).expanduser().resolve())
            except Exception:
                return str(val or "")

        def _norm_subdir(val: Any) -> str:
            s = str(val or "").replace("\\", "/").strip()
            if not s:
                return ""
            if "/artifacts/" in s:
                s = s.rsplit("/artifacts/", 1)[-1]
            s = s.lstrip("./")
            if s.startswith("artifacts/"):
                s = s[len("artifacts/") :]
            return s.strip("/")

        def _norm_bool(val: Any) -> Optional[bool]:
            if isinstance(val, bool):
                return val
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return bool(val)
            low = str(val).strip().lower()
            if low in ("1", "true", "yes", "on"):
                return True
            if low in ("0", "false", "no", "off"):
                return False
            return None

        def _norm_method(val: Any) -> str:
            s = str(val or "").strip().lower()
            return s or "identity"

        def _infer_reg_modality(token: Any) -> Optional[str]:
            low = str(token or "").strip().lower()
            if not low:
                return None
            if any(k in low for k in ("high_b", "highb", "low_b", "mid_b", "dwi", "tracew", "trace")):
                return "dwi"
            if "adc" in low:
                return "adc"
            return None

        def _registration_intent_modality() -> Optional[str]:
            # Prefer call_id intent first (e.g., register_dwi_020), then explicit output_subdir,
            # then moving path fallback.
            for token in (call.call_id, raw_args.get("output_subdir"), raw_args.get("moving")):
                m = _infer_reg_modality(token)
                if m:
                    return m
            return None

        def _find_existing_registration() -> Optional[Dict[str, Any]]:
            try:
                moving = _norm_path(raw_args.get("moving"))
                fixed = _norm_path(raw_args.get("fixed"))
                if not moving or not fixed:
                    return None
                target_subdir = _norm_subdir(raw_args.get("output_subdir"))
                target_intent = _registration_intent_modality()
                # Reuse only when output intent is explicit.
                if not target_subdir:
                    return None
                target_split = _norm_bool(raw_args.get("split_by_bvalue"))
                target_method = _norm_method(raw_args.get("method"))
                stage_outputs = state.stage_outputs or {}
                for _stage, tools in stage_outputs.items():
                    if not isinstance(tools, dict):
                        continue
                    recs = tools.get("register_to_reference")
                    if not isinstance(recs, list):
                        continue
                    for r in recs:
                        if not isinstance(r, dict) or not r.get("ok"):
                            continue
                        data = r.get("data") if isinstance(r.get("data"), dict) else None
                        if not isinstance(data, dict):
                            continue
                        prev_moving = _norm_path(data.get("moving"))
                        prev_fixed = _norm_path(data.get("fixed"))
                        if not (prev_moving and prev_fixed and prev_moving == moving and prev_fixed == fixed):
                            continue

                        prev_subdir = _norm_subdir(data.get("output_subdir"))
                        if prev_subdir != target_subdir:
                            continue

                        prev_intent = _infer_reg_modality(prev_subdir) or _infer_reg_modality(data.get("moving"))
                        if target_intent and prev_intent and target_intent != prev_intent:
                            continue

                        prev_split = _norm_bool(data.get("split_by_bvalue"))
                        if target_split is not None and prev_split is not None and target_split != prev_split:
                            continue

                        prev_method = _norm_method(
                            data.get("method")
                            or (
                                (data.get("qc_metrics") or {}).get("method")
                                if isinstance(data.get("qc_metrics"), dict)
                                else None
                            )
                        )
                        if target_method and prev_method and target_method != prev_method:
                            continue

                        return data
            except Exception:
                return None
            return None

        source_artifacts: List[ArtifactRef] = []
        generated_artifacts: List[ArtifactRef] = []
        warnings: List[str] = []
        error_dict: Optional[Dict[str, Any]] = None

        try:
            existing_registration = None
            if resolved_name == "register_to_reference":
                existing_registration = _find_existing_registration()

            if existing_registration is not None:
                validate_args_minimal(tool.spec.output_schema or {}, existing_registration)
                runtime_ms = int((time.time() - start) * 1000)
                provenance = {
                    "timestamp": now_iso(),
                    "tool_name": tool.spec.name,
                    "tool_version": tool.spec.version,
                    "runtime_ms": runtime_ms,
                    "input_hash": input_hash,
                    "output_hash": sha256_json(existing_registration),
                    "environment": environment_snapshot(),
                }
                warnings = ["duplicate registration skipped; reused existing result (strict intent/output match)"]
                if removed_args:
                    warnings.append(f"dispatcher dropped unsupported arguments: {', '.join(removed_args)}")
                result = ToolResult(
                    ok=True,
                    data=existing_registration,
                    artifacts=[],
                    warnings=warnings,
                    error=None,
                    provenance=provenance,
                )
            else:
                validate_args_minimal(input_schema, raw_args)
                if should_dispatch_subprocess(resolved_name, self.runtime_config):
                    payload = run_tool_subprocess(
                        tool_name=resolved_name,
                        args=raw_args,
                        ctx=ctx,
                        cfg=self.runtime_config,
                    )
                else:
                    payload = tool.func(raw_args, ctx)
                validate_output_minimal(payload)

                # Tools may return ToolResult (preferred future direction) OR dict payload (current MVP).
                if isinstance(payload, ToolResult):
                    data = payload.data
                    tool_ok = payload.ok
                    tool_error_msg = str(payload.error) if payload.error else None
                    norm_artifacts = normalize_artifacts(payload.artifacts, default_kind="unknown")
                    warnings = list(payload.warnings or [])
                    # ToolResult doesn't carry source/generated yet; treat artifacts as generated by default.
                    source_artifacts = []
                    generated_artifacts = list(norm_artifacts)
                else:
                    # Respect the tool's own "ok" field when present;
                    # previously ok was always True if no exception was raised,
                    # silently swallowing error-dict returns like
                    #   {"ok": False, "error": "Input NIfTI not found: ..."}.
                    tool_ok = payload.get("ok", True) if isinstance(payload, dict) else True
                    tool_error_msg = str(payload.get("error") or "") if (not tool_ok and isinstance(payload, dict)) else None
                    data = payload.get("data", {})
                    norm_artifacts = normalize_artifacts(payload.get("artifacts", []), default_kind="unknown")
                    warnings = payload.get("warnings", [])
                    source_artifacts = normalize_artifacts(payload.get("source_artifacts", []), default_kind="ref")
                    generated_artifacts = normalize_artifacts(payload.get("generated_artifacts", []), default_kind="unknown")

                # If the tool signalled failure, convert to a proper error result
                # so cerebellum / reflector can see it and attempt recovery.
                if not tool_ok:
                    raise RuntimeError(
                        tool_error_msg or f"Tool {tool.spec.name} returned ok=False"
                    )

                if removed_args:
                    warnings = list(warnings or [])
                    warnings.append(f"dispatcher dropped unsupported arguments: {', '.join(removed_args)}")

                # Optional: validate tool output against declared output schema (minimal)
                validate_args_minimal(tool.spec.output_schema or {}, data)

                output_hash = sha256_json(data)
                runtime_ms = int((time.time() - start) * 1000)

                provenance = {
                    "timestamp": now_iso(),
                    "tool_name": tool.spec.name,
                    "tool_version": tool.spec.version,
                    "runtime_ms": runtime_ms,
                    "input_hash": input_hash,
                    "output_hash": output_hash,
                    "environment": environment_snapshot(),
                }

                result = ToolResult(
                    ok=True,
                    data=data,
                    artifacts=norm_artifacts,
                    warnings=warnings,
                    error=None,
                    provenance=provenance,
                )

        except Exception as e:
            runtime_ms = int((time.time() - start) * 1000)
            stack = traceback.format_exc()
            terr = ToolError(type=type(e).__name__, message=str(e), stack=stack)
            error_dict = terr.to_dict()

            provenance = {
                "timestamp": now_iso(),
                "tool_name": tool.spec.name,
                "tool_version": tool.spec.version,
                "runtime_ms": runtime_ms,
                "input_hash": input_hash,
                "output_hash": "",
                "environment": environment_snapshot(),
            }

            result = ToolResult(
                ok=False,
                data={},
                artifacts=[],
                warnings=[],
                error=terr,
                provenance=provenance,
            )

        def _normalize_stage(stage_raw: str) -> tuple[str, int]:
            s = (stage_raw or "misc").strip()
            # numeric legacy stages
            num_map = {
                "0": "misc",
                "1": "ingest",
                "2": "identify",
                "3": "register",
                "4": "segment",
                "5": "extract",
                "6": "report",
            }
            if s.isdigit() and s in num_map:
                s = num_map[s]
            s = s.lower()
            order = {
                "ingest": 1,
                "identify": 2,
                "register": 3,
                "segment": 4,
                "extract": 5,
                "report": 6,
                "final": 7,
                "misc": 99,
            }.get(s, 99)
            return s, order

        # Update state (write-through), organized by semantic stage_name (+ stage_order)
        stage_name, stage_order = _normalize_stage(call.stage or "misc")
        stage_data = result.data if result.ok else {"error": error_dict or {}}
        state.add_stage_record(
            stage=stage_name,
            tool_name=call.tool_name,
            call_id=call.call_id,
            ok=result.ok,
            data=stage_data,
            stage_order=stage_order,
        )
        if result.artifacts:
            state.add_artifacts(result.artifacts)
        # Update summary with completed tools list for quick UI/agent access.
        try:
            completed: List[Dict[str, Any]] = []
            for st, tools in (state.stage_outputs or {}).items():
                if not isinstance(tools, dict):
                    continue
                for tname, records in tools.items():
                    if not isinstance(records, list):
                        continue
                    if any(isinstance(r, dict) and r.get("ok") is True for r in records):
                        completed.append({"stage": st, "tool_name": tname})
            state.summary = {**(state.summary or {}), "completed_tools": completed}
        except Exception:
            pass
        state.write_json(ctx.case_state_path)

        # Log (append-only JSONL)
        output_hash = result.provenance.get("output_hash") or sha256_json(result.data)
        runtime_ms = int(result.provenance.get("runtime_ms") or 0)

        entry = ExecutionLogEntry(
            timestamp=result.provenance.get("timestamp", now_iso()),
            tool_name=call.tool_name,
            tool_version=tool.spec.version,
            call_id=call.call_id,
            case_id=call.case_id,
            stage=stage_name,
            requested_by=call.requested_by,
            arguments=raw_args,
            ok=result.ok,
            warnings=result.warnings,
            error=result.error.to_dict() if result.error else None,
            runtime_ms=runtime_ms,
            input_hash=input_hash,
            output_hash=output_hash,
            environment=result.provenance.get("environment", environment_snapshot()),
            source_artifacts=source_artifacts,
            generated_artifacts=generated_artifacts or result.artifacts,
        )
        self._append_log(ctx.run_dir, entry)

        return result

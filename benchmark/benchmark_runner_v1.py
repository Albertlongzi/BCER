from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# Suppress verbose ITK/GDCM C++ warnings (e.g., empty-series directory scans) during benchmark runs.
os.environ["ITK_GLOBAL_DEFAULT_WARNING_LEVEL"] = "0"

if __package__ in {None, ""}:
    # parents[1] is the repo root (parents[2] is whatever directory the clone
    # happens to sit in). Insert at the FRONT so script-mode execution resolves
    # the in-repo agent/commands/core packages ahead of any editable install
    # that reaches sys.path via an easy-install.pth.
    _repo_root = str(Path(__file__).resolve().parents[1])
    if _repo_root in sys.path:
        sys.path.remove(_repo_root)
    sys.path.insert(0, _repo_root)

from agent.langgraph.loop import plan_agent_dag, run_langgraph_agent
from agent.loop import run_agent_loop
from commands.dispatcher import ToolDispatcher
from commands.schemas import ToolCall
from core.domain_config import get_domain_config
from core.paths import project_root
from core.plan_dag import AgentPlanDAG, CaseScope, DagPolicy, PlanNode
from llm.adapter_vllm_server import VLLMServerConfig
from mri_agent_shell.runtime.cerebellum import Cerebellum
from mri_agent_shell.runtime.binding_policy import BindingPolicy, degrade_dag_tokens
from mri_agent_shell.runtime.session import ModelConfig, SessionState, normalize_provider
from mri_agent_shell.tool_registry import build_shell_registry


EXPECTED_ABLATIONS = (
    "static_pipeline",
    "pure_react",
    "bcr_no_reflector",
    "bcr_full",
    "bcr_sketch",
    "bcr_no_token",
    "bcr_deterministic_only",
)
DEFAULT_SERVER_MODEL = "Qwen/Qwen3-VL-30B-A3B-Thinking"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = str(line).strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict):
                out.append(obj)
    except Exception:
        return out
    return out


def _sanitize_case_id(raw: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(raw or "").strip())


def _short_text(text: Any, *, max_chars: int = 220) -> str:
    s = str(text or "").strip().replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


@dataclass
class BackendConfig:
    backend_id: str
    provider: str
    model: str
    base_url: str
    api_key: str


@dataclass
class FaultInjectionSpec:
    enabled: bool = False
    stage: str = "pre_dispatch"
    tool_name: str = ""
    match: str = "first_call"
    mutation_type: str = ""
    mutation_key: str = ""
    mutation_value: Any = None
    expected_fault_class: str = ""


@dataclass
class FaultInjector:
    spec: FaultInjectionSpec
    applied: bool = False
    target_seen_calls: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)

    def maybe_mutate_arguments(
        self,
        *,
        tool_name: str,
        arguments: Dict[str, Any],
        source: str,
    ) -> Tuple[Dict[str, Any], bool]:
        args = dict(arguments or {})
        if not self.spec.enabled:
            return args, False
        if str(self.spec.stage).strip().lower() != "pre_dispatch":
            return args, False

        target_tool = str(self.spec.tool_name or "").strip()
        if target_tool and str(tool_name or "").strip() != target_tool:
            return args, False

        self.target_seen_calls += 1

        match_rule = str(self.spec.match or "first_call").strip().lower()
        if match_rule == "first_call" and self.target_seen_calls != 1:
            return args, False
        if match_rule == "first_call" and self.applied:
            return args, False

        if str(self.spec.mutation_type or "").strip() != "set_argument":
            return args, False

        key = str(self.spec.mutation_key or "").strip()
        if not key:
            return args, False

        before = args.get(key)
        args[key] = self.spec.mutation_value
        self.applied = True
        self.events.append(
            {
                "ts": _utc_now_iso(),
                "source": source,
                "tool_name": str(tool_name),
                "mutation": {
                    "type": "set_argument",
                    "key": key,
                    "from": before,
                    "to": self.spec.mutation_value,
                },
            }
        )
        return args, True

    def maybe_mutate_tool_call(self, call: ToolCall, *, source: str = "dispatcher.pre_dispatch") -> ToolCall:
        new_args, mutated = self.maybe_mutate_arguments(
            tool_name=str(call.tool_name),
            arguments=dict(call.arguments or {}),
            source=source,
        )
        if not mutated:
            return call
        return ToolCall(
            tool_name=call.tool_name,
            arguments=new_args,
            call_id=call.call_id,
            case_id=call.case_id,
            stage=call.stage,
            requested_by=call.requested_by,
        )


@contextmanager
def patched_dispatch(injector: FaultInjector) -> Iterator[None]:
    if not injector.spec.enabled:
        yield
        return

    original_dispatch = ToolDispatcher.dispatch

    def _wrapped_dispatch(self: ToolDispatcher, call: ToolCall, state: Any, ctx: Any) -> Any:
        patched_call = injector.maybe_mutate_tool_call(call)
        return original_dispatch(self, patched_call, state, ctx)

    ToolDispatcher.dispatch = _wrapped_dispatch  # type: ignore[assignment]
    try:
        yield
    finally:
        ToolDispatcher.dispatch = original_dispatch  # type: ignore[assignment]


class InjectingCerebellum(Cerebellum):
    def __init__(self, *, fault_injector: FaultInjector, **kwargs: Any) -> None:
        self._fault_injector = fault_injector
        super().__init__(**kwargs)

    def _execute_node(
        self,
        *,
        node: PlanNode,
        state: Any,
        ctx: Any,
        scope_domain: str,
        guard: Any,
        binder: Any,
        case_id: str,
        run_id: str,
        trace_path: Path,
        max_attempts: int,
        emit: Callable[[str], None],
    ) -> Tuple[bool, Dict[str, Any]]:
        # For token faults, inject before symbolic resolution so mutations can
        # target @node/@seq references (reflector-relevant), instead of only
        # post-bind absolute paths seen at pre-guard time.
        try:
            fault_name = str(getattr(getattr(self._fault_injector, "spec", None), "fault", "") or "").strip().lower()
        except Exception:
            fault_name = ""
        if fault_name == "token_mutation":
            try:
                original_node_args = dict(getattr(node, "arguments", {}) or {})
                mutated_args, mutated = self._fault_injector.maybe_mutate_arguments(
                    tool_name=str(getattr(node, "tool_name", "") or ""),
                    arguments=original_node_args,
                    source="cerebellum.pre_resolve",
                )
                if mutated:
                    try:
                        node = node.model_copy(update={"arguments": dict(mutated_args)}, deep=True)
                    except Exception:
                        # Best effort for non-pydantic node objects.
                        setattr(node, "arguments", dict(mutated_args))
            except Exception:
                pass

        original_validate_args = getattr(guard, "validate_args")

        def _wrapped_validate_args(*, tool_name: str, args: Dict[str, Any]) -> None:
            # Inject at pre-guard time so mutations survive _normalize_step_args rewrites.
            # Mutate in place; cerebellum will reuse this same dict for dispatcher call.
            mutated_args, mutated = self._fault_injector.maybe_mutate_arguments(
                tool_name=str(tool_name or ""),
                arguments=dict(args or {}),
                source="cerebellum.pre_guard",
            )
            if mutated:
                args.clear()
                args.update(mutated_args)
            original_validate_args(tool_name=tool_name, args=args)

        setattr(guard, "validate_args", _wrapped_validate_args)
        try:
            return super()._execute_node(
                node=node,
                state=state,
                ctx=ctx,
                scope_domain=scope_domain,
                guard=guard,
                binder=binder,
                case_id=case_id,
                run_id=run_id,
                trace_path=trace_path,
                max_attempts=max_attempts,
                emit=emit,
            )
        finally:
            setattr(guard, "validate_args", original_validate_args)


def _fault_spec_from_case(case_obj: Dict[str, Any]) -> FaultInjectionSpec:
    fi = case_obj.get("fault_injection") if isinstance(case_obj.get("fault_injection"), dict) else {}
    target = fi.get("target") if isinstance(fi.get("target"), dict) else {}
    mutation = fi.get("mutation") if isinstance(fi.get("mutation"), dict) else {}
    return FaultInjectionSpec(
        enabled=bool(fi.get("enabled")),
        stage=str(fi.get("stage") or "pre_dispatch"),
        tool_name=str(target.get("tool_name") or "").strip(),
        match=str(target.get("match") or "first_call").strip(),
        mutation_type=str(mutation.get("type") or "").strip(),
        mutation_key=str(mutation.get("key") or "").strip(),
        mutation_value=mutation.get("value"),
        expected_fault_class=str(fi.get("expected_fault_class") or "").strip().lower(),
    )


def _find_qwen_backend(suite: Dict[str, Any], *, server_base_url: str, server_model_override: str) -> BackendConfig:
    backends = suite.get("llm_backends") if isinstance(suite.get("llm_backends"), list) else []
    chosen: Optional[Dict[str, Any]] = None
    for row in backends:
        if not isinstance(row, dict):
            continue
        if str(row.get("id") or "").strip() == "qwen_local_default":
            chosen = row
            break
    if chosen is None:
        raise ValueError("benchmark_suite missing llm_backends.qwen_local_default")

    provider = str(chosen.get("provider") or "").strip().lower() or "openai_compatible_server"
    if provider != "openai_compatible_server":
        raise ValueError(
            "qwen_local_default must use provider=openai_compatible_server for this runner. "
            f"Got: {provider}"
        )

    model = str(server_model_override or chosen.get("model") or "qwen-local").strip()
    base_url = str(server_base_url or os.environ.get("MRI_AGENT_SHELL_SERVER_BASE_URL") or "http://127.0.0.1:8000/v1").strip()
    api_key = str(os.environ.get("MRI_AGENT_SHELL_SERVER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY").strip()

    return BackendConfig(
        backend_id="qwen_local_default",
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=(api_key if api_key else "EMPTY"),
    )


def _model_config_from_backend(backend: BackendConfig, *, max_tokens: int) -> ModelConfig:
    provider = normalize_provider(str(backend.provider or "openai_compatible_server"))
    base_url = str(backend.base_url or "").strip()
    return ModelConfig(
        provider=provider,
        llm=backend.model,
        vlm=backend.model,
        server_base_url=(base_url if provider == "openai_compatible_server" else ""),
        api_base_url=(base_url if provider != "openai_compatible_server" else ""),
        api_key=backend.api_key,
        max_tokens=int(max_tokens),
        temperature=0.0,
    )


def _server_cfg_from_backend(backend: BackendConfig, *, max_tokens: int) -> Optional[VLLMServerConfig]:
    provider = normalize_provider(str(backend.provider or "openai_compatible_server"))
    if provider != "openai_compatible_server":
        return None
    return VLLMServerConfig(
        base_url=backend.base_url,
        model=backend.model,
        api_key=backend.api_key,
        max_tokens=int(max_tokens),
        temperature=0.0,
    )


def _llm_mode_from_backend(backend: BackendConfig) -> str:
    provider = normalize_provider(str(backend.provider or "openai_compatible_server"))
    if provider == "openai_compatible_server":
        return "server"
    if provider == "openai_official":
        return "openai"
    if provider == "gemini":
        return "gemini"
    if provider == "anthropic":
        return "anthropic"
    raise ValueError(f"Unsupported backend provider for benchmark runner: {provider}")


def _llm_invoke_kwargs_from_backend(backend: BackendConfig, *, max_tokens: int) -> Dict[str, Any]:
    llm_mode = _llm_mode_from_backend(backend)
    out: Dict[str, Any] = {
        "llm_mode": llm_mode,
        "server_cfg": None,
        "api_model": None,
        "api_base_url": None,
    }
    if llm_mode == "server":
        out["server_cfg"] = _server_cfg_from_backend(backend, max_tokens=max_tokens)
        return out
    out["api_model"] = str(backend.model or "").strip() or None
    base_url = str(backend.base_url or "").strip()
    if base_url:
        out["api_base_url"] = base_url
    return out


def _disabled_reflector(payload: Dict[str, Any]) -> Dict[str, Any]:
    err = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    err_type = str(err.get("type") or "RuntimeError")
    err_msg = str(err.get("message") or "required node failed")
    tool_name = str(payload.get("tool_name") or "unknown_tool")
    return {
        "action": "halt",
        "reason": f"reflector_disabled: {err_type}: {err_msg}",
        "retry_arguments": {},
        "natural_language_response": (
            f"Reflector disabled for this ablation. Required step '{tool_name}' failed: {err_msg}."
        ),
    }


def _deterministic_only_reflector(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Tier-1 (deterministic) reflector only — no LLM call (Tier-2 disabled).

    Mirrors the deterministic retry logic from Cerebellum._default_failure_reflect_fn
    but never falls through to the LLM reflection branch.
    """
    err = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    err_type = str(err.get("type") or "RuntimeError")
    err_msg = str(err.get("message") or "required node failed")
    tool_name = str(payload.get("tool_name") or "unknown_tool")

    deterministic_retry = payload.get("deterministic_retry_suggestion")
    deterministic_retry = (
        dict(deterministic_retry)
        if isinstance(deterministic_retry, dict) and deterministic_retry
        else {}
    )

    # Hard limits are never retryable even with deterministic fix
    et = str(err_type or "").strip().lower()
    msg = str(err_msg or "").strip().lower()
    if et == "schemavalidationerror" or "not allowed for domain" in msg:
        return {
            "action": "halt",
            "reason": f"hard_limit_schema: {err_type}: {err_msg}",
            "retry_arguments": {},
            "natural_language_response": (
                f"Schema or domain violation for '{tool_name}': {err_msg}. "
                "Cannot fix at runtime."
            ),
        }
    if et == "scopeviolation" and "unresolved reference in path argument" not in msg:
        return {
            "action": "halt",
            "reason": f"hard_limit_scope: {err_type}: {err_msg}",
            "retry_arguments": {},
            "natural_language_response": (
                f"Scope violation for '{tool_name}': {err_msg}. "
                "Cannot fix at runtime."
            ),
        }

    # If deterministic fix available, use it (Tier 1)
    if deterministic_retry:
        return {
            "action": "retry",
            "reason": f"deterministic_fix: {err_type}: {err_msg}",
            "retry_arguments": deterministic_retry,
            "natural_language_response": (
                f"Deterministic fix applied for '{tool_name}'. Retrying with corrected arguments."
            ),
        }

    # No deterministic fix and no LLM (Tier 2 disabled) → halt
    return {
        "action": "halt",
        "reason": f"tier2_disabled_no_deterministic_fix: {err_type}: {err_msg}",
        "retry_arguments": {},
        "natural_language_response": (
            f"Tier-1 deterministic reflector could not fix '{tool_name}': {err_msg}. "
            "LLM reflection (Tier-2) is disabled for this arm."
        ),
    }


def _build_static_nodes(domain: str) -> List[PlanNode]:
    dom = str(domain or "").strip().lower()
    if dom == "prostate":
        return [
            PlanNode(
                node_id="identify_sequences_001",
                tool_name="identify_sequences",
                stage="identify",
                arguments={"dicom_case_dir": "@case.input", "convert_to_nifti": True, "output_subdir": "ingest"},
                required=True,
                depends_on=[],
                label="Identify and map prostate sequences",
            ),
            PlanNode(
                node_id="segment_prostate_030",
                tool_name="segment_prostate",
                stage="segment",
                arguments={"t2w_ref": "T2w", "output_subdir": "segmentation"},
                required=True,
                depends_on=["identify_sequences_001"],
                label="Segment whole-gland prostate",
            ),
            PlanNode(
                node_id="package_evidence_060",
                tool_name="package_vlm_evidence",
                stage="package",
                arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "vlm"},
                required=True,
                depends_on=["segment_prostate_030"],
                label="Package report evidence",
            ),
            PlanNode(
                node_id="generate_report_070",
                tool_name="generate_report",
                stage="report",
                arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "report", "domain": "prostate"},
                required=True,
                depends_on=["package_evidence_060"],
                label="Generate report",
            ),
        ]

    if dom == "brain":
        return [
            PlanNode(
                node_id="identify_sequences_001",
                tool_name="identify_sequences",
                stage="identify",
                arguments={"dicom_case_dir": "@case.input", "convert_to_nifti": True, "output_subdir": "ingest"},
                required=True,
                depends_on=[],
                label="Identify and map brain sequences",
            ),
            PlanNode(
                node_id="brats_segmentation_020",
                tool_name="brats_mri_segmentation",
                stage="segment",
                arguments={
                    "t1c_path": "T1c",
                    "t1_path": "T1",
                    "t2_path": "T2",
                    "flair_path": "FLAIR",
                    "output_subdir": "segmentation",
                },
                required=True,
                depends_on=["identify_sequences_001"],
                label="Segment tumor subregions",
            ),
            PlanNode(
                node_id="package_evidence_060",
                tool_name="package_vlm_evidence",
                stage="package",
                arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "vlm"},
                required=True,
                depends_on=["brats_segmentation_020"],
                label="Package report evidence",
            ),
            PlanNode(
                node_id="generate_report_070",
                tool_name="generate_report",
                stage="report",
                arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "report", "domain": "brain"},
                required=True,
                depends_on=["package_evidence_060"],
                label="Generate report",
            ),
        ]

    if dom == "cardiac":
        return [
            PlanNode(
                node_id="identify_sequences_001",
                tool_name="identify_sequences",
                stage="identify",
                arguments={"dicom_case_dir": "@case.input", "convert_to_nifti": True, "output_subdir": "ingest"},
                required=True,
                depends_on=[],
                label="Identify and map cardiac cine sequences",
            ),
            PlanNode(
                node_id="segment_cardiac_020",
                tool_name="segment_cardiac_cine",
                stage="segment",
                arguments={"cine_path": "CINE", "output_subdir": "segmentation"},
                required=True,
                depends_on=["identify_sequences_001"],
                label="Segment cardiac cine",
            ),
            PlanNode(
                node_id="classify_cardiac_030",
                tool_name="classify_cardiac_cine_disease",
                stage="classify",
                arguments={"cine_path": "CINE", "output_subdir": "classification"},
                required=True,
                depends_on=["segment_cardiac_020"],
                label="Classify cardiac disease",
            ),
            PlanNode(
                node_id="package_evidence_060",
                tool_name="package_vlm_evidence",
                stage="package",
                arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "vlm"},
                required=True,
                depends_on=["classify_cardiac_030"],
                label="Package report evidence",
            ),
            PlanNode(
                node_id="generate_report_070",
                tool_name="generate_report",
                stage="report",
                arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "report", "domain": "cardiac"},
                required=True,
                depends_on=["package_evidence_060"],
                label="Generate report",
            ),
        ]

    raise ValueError(f"Unsupported domain for static pipeline: {domain}")


def _build_static_dag(
    *,
    case_obj: Dict[str, Any],
    case_id: str,
    workspace_root: Path,
    runs_root: Path,
) -> AgentPlanDAG:
    domain = str(case_obj.get("domain") or "").strip().lower()
    case_ref = str(case_obj.get("case_ref") or "").strip()
    if not case_ref:
        raise ValueError("Missing case_ref in benchmark case")

    return AgentPlanDAG(
        plan_id=f"static_{case_id}_{uuid.uuid4().hex[:8]}",
        goal=str(case_obj.get("prompt") or "").strip(),
        case_scope=CaseScope(
            domain=domain,  # type: ignore[arg-type]
            case_id=case_id,
            case_ref=str(Path(case_ref).expanduser().resolve()),
            workspace_root=str(workspace_root.resolve()),
            runs_root=str(runs_root.resolve()),
            allow_external_model_roots=[],
        ),
        planner_status="ready",
        natural_language_response="Static pipeline plan generated by benchmark runner.",
        requested_request_type="full_pipeline",
        policy=DagPolicy(skip_optional_by_default=False, stop_on_required_failure=True, max_attempts_default=2),
        nodes=_build_static_nodes(domain),
        notes=["Ablation=static_pipeline", "Planner bypassed (deterministic DAG)."],
    )


def _new_session(
    *,
    workspace_root: Path,
    runs_root: Path,
    model_config: ModelConfig,
    case_ref: str,
    case_id: str,
) -> SessionState:
    session = SessionState(
        workspace_path=str(workspace_root),
        runs_root=str(runs_root),
        model_config=model_config,
        dry_run=False,
    )
    session.set_case_id(case_id)
    case_ref_path = Path(str(case_ref or "")).expanduser().resolve()
    # SessionState currently binds case.input to a directory root; keep file refs
    # available separately for raw-recon plans that point h5_path explicitly.
    session_input = case_ref_path.parent if case_ref_path.is_file() else case_ref_path
    session.set_case_input(str(session_input))
    if case_ref_path.is_file():
        session.case_inputs["case_file_path"] = str(case_ref_path)
        session.path_keys["case.file"] = str(case_ref_path)
    return session


def _run_cerebellum_mode(
    *,
    case_obj: Dict[str, Any],
    case_id: str,
    ablation_mode: str,
    backend: BackendConfig,
    workspace_root: Path,
    runs_root: Path,
    max_new_tokens: int,
    injector: FaultInjector,
) -> Dict[str, Any]:
    case_ref = str(case_obj.get("case_ref") or "").strip()
    domain = str(case_obj.get("domain") or "").strip().lower()
    model_cfg = _model_config_from_backend(backend, max_tokens=max_new_tokens)
    llm_kwargs = _llm_invoke_kwargs_from_backend(backend, max_tokens=max_new_tokens)
    session = _new_session(
        workspace_root=workspace_root,
        runs_root=runs_root,
        model_config=model_cfg,
        case_ref=case_ref,
        case_id=case_id,
    )
    registry = build_shell_registry(dry_run=False, include_core=True)
    explicit_aliases = (
        case_obj.get("planner_input_aliases")
        if isinstance(case_obj.get("planner_input_aliases"), dict)
        else {}
    )
    explicit_alias_roots: List[str] = []
    if isinstance(explicit_aliases, dict) and explicit_aliases:
        seen_roots: set[str] = set()
        for _k, raw_val in explicit_aliases.items():
            vals = raw_val if isinstance(raw_val, list) else [raw_val]
            for raw in vals:
                s = str(raw or "").strip()
                if not s:
                    continue
                try:
                    p = Path(s).expanduser()
                    root_lex = str((p.parent if p.suffix else p))
                    root_abs = str(Path(os.path.abspath(root_lex)))
                    root_res = str(Path(root_lex).resolve())
                except Exception:
                    continue
                for root in (root_lex, root_abs, root_res):
                    rr = str(root or "").strip()
                    if not rr or rr in seen_roots:
                        continue
                    seen_roots.add(rr)
                    explicit_alias_roots.append(rr)

    # ── Ablation policy ──────────────────────────────────────────────
    policy = BindingPolicy.for_arm(ablation_mode)

    cereb_cls: Any = InjectingCerebellum if injector.spec.enabled else Cerebellum
    cereb_kwargs: Dict[str, Any] = {
        "session": session,
        "registry": registry,
        "binding_policy": policy,
    }
    if injector.spec.enabled:
        cereb_kwargs["fault_injector"] = injector

    if ablation_mode in {"static_pipeline", "bcr_no_reflector"}:
        cereb_kwargs["failure_reflect_fn"] = _disabled_reflector

    if ablation_mode == "bcr_no_reflector":
        # Keep this ablation strict: no reflector and no deterministic
        # fallback override on required failures.
        cereb_kwargs["allow_deterministic_fallback_on_nonretry"] = False

    if ablation_mode == "bcr_deterministic_only":
        cereb_kwargs["failure_reflect_fn"] = _deterministic_only_reflector

    cerebellum = cereb_cls(**cereb_kwargs)

    if ablation_mode == "static_pipeline":
        dag = _build_static_dag(
            case_obj=case_obj,
            case_id=case_id,
            workspace_root=workspace_root,
            runs_root=runs_root,
        )
    else:
        template_variant = "notoken" if ablation_mode == "bcr_no_token" else "default"
        planning_mode = "sketch" if ablation_mode == "bcr_sketch" else "template"
        dag = plan_agent_dag(
            goal=str(case_obj.get("prompt") or "").strip(),
            domain=domain,
            case_ref=case_ref,
            case_id=case_id,
            request_type=(str(case_obj.get("request_type") or "").strip() or None),
            llm_mode=str(llm_kwargs.get("llm_mode") or "server"),
            max_new_tokens=int(max_new_tokens),
            server_cfg=llm_kwargs.get("server_cfg"),
            api_model=llm_kwargs.get("api_model"),
            api_base_url=llm_kwargs.get("api_base_url"),
            workspace_root=str(workspace_root),
            runs_root=str(runs_root),
            allow_external_model_roots=explicit_alias_roots,
            template_variant=template_variant,
            explicit_input_aliases=(
                case_obj.get("planner_input_aliases")
                if isinstance(case_obj.get("planner_input_aliases"), dict)
                else None
            ),
            planning_mode=planning_mode,
            planner_task_id=(str(case_obj.get("task_id") or "").strip() or None),
        )

    # Apply DAG-level token degradation based on policy (replaces the
    # old _pre_resolve_dag_tokens single-line hack).
    dag = degrade_dag_tokens(dag, policy=policy)

    with patched_dispatch(injector):
        out = cerebellum.execute_dag(dag, emit=lambda _msg: None)

    return {
        "run_result": out,
        "run_dir": str(out.get("run_dir") or ""),
        "trace_path": str(out.get("trace_path") or ""),
        "reflection_decisions": out.get("reflection_decisions") if isinstance(out.get("reflection_decisions"), list) else [],
        "planner_status": str(getattr(dag, "planner_status", "ready")),
        "planner_artifacts": (
            dag.planner_artifacts if isinstance(getattr(dag, "planner_artifacts", None), dict) else {}
        ),
        "planner_metadata": (
            dag.planner_metadata if isinstance(getattr(dag, "planner_metadata", None), dict) else {}
        ),
    }


def _run_pure_react_mode(
    *,
    case_obj: Dict[str, Any],
    case_id: str,
    backend: BackendConfig,
    runs_root: Path,
    max_new_tokens: int,
    max_steps: int,
    max_retries: int,
    injector: FaultInjector,
) -> Dict[str, Any]:
    case_ref = str(case_obj.get("case_ref") or "").strip()
    domain = str(case_obj.get("domain") or "").strip().lower()
    llm_kwargs = _llm_invoke_kwargs_from_backend(backend, max_tokens=max_new_tokens)

    with patched_dispatch(injector):
        run_dir = run_agent_loop(
            goal=str(case_obj.get("prompt") or "").strip(),
            case_id=case_id,
            dicom_case_dir=case_ref,
            runs_root=Path(runs_root),
            llm_mode=str(llm_kwargs.get("llm_mode") or "server"),
            max_steps=int(max_steps),
            max_retries=int(max_retries),
            plan_mode="step",
            server_cfg=llm_kwargs.get("server_cfg"),
            api_model=llm_kwargs.get("api_model"),
            api_base_url=llm_kwargs.get("api_base_url"),
            finalize_with_llm=False,
            enforce_mvp_pipeline=False,
            autofix_mode="off",
            symbolic_binder_mode="off",
            enable_preconditions=False,
            enable_tool_reflection=False,
            domain=get_domain_config(domain),
        )

    run_dir_path = Path(run_dir)
    return {
        "run_result": {"ok": None},
        "run_dir": str(run_dir_path),
        "trace_path": str(run_dir_path / "agent_trace.jsonl"),
        "reflection_decisions": [],
        "planner_status": "reactive",
    }


def _has_tool_success(case_state: Dict[str, Any], tool_name: str) -> bool:
    stage_outputs = case_state.get("stage_outputs") if isinstance(case_state.get("stage_outputs"), dict) else {}
    for _stage, tools in stage_outputs.items():
        if not isinstance(tools, dict):
            continue
        recs = tools.get(tool_name)
        if not isinstance(recs, list):
            continue
        if any(isinstance(r, dict) and (r.get("ok") is True) for r in recs):
            return True
    return False


def _collect_error_types(
    *,
    execution_rows: List[Dict[str, Any]],
    trace_rows: List[Dict[str, Any]],
    raised_error: Optional[Dict[str, Any]],
) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()

    def _add(err_type: str) -> None:
        et = str(err_type or "").strip()
        if not et or et in seen:
            return
        seen.add(et)
        out.append(et)

    for row in execution_rows:
        err = row.get("error") if isinstance(row.get("error"), dict) else {}
        _add(str(err.get("type") or ""))

    for row in trace_rows:
        err = row.get("error") if isinstance(row.get("error"), dict) else {}
        _add(str(err.get("type") or ""))

    if isinstance(raised_error, dict):
        _add(str(raised_error.get("type") or ""))

    return out


def _collect_run_artifacts(run_dir: str, trace_path: str) -> Dict[str, Any]:
    run_dir_path = Path(str(run_dir or "")).expanduser()
    case_state_path = run_dir_path / "case_state.json"
    execution_log_path = run_dir_path / "execution_log.jsonl"
    trace_path_obj = Path(str(trace_path or "")).expanduser() if str(trace_path or "").strip() else Path("")

    case_state = _read_json(case_state_path)
    execution_rows = _read_jsonl(execution_log_path)
    trace_rows = _read_jsonl(trace_path_obj) if str(trace_path_obj) and trace_path_obj.exists() else []

    return {
        "run_dir": str(run_dir_path),
        "case_state_path": str(case_state_path),
        "execution_log_path": str(execution_log_path),
        "trace_path": str(trace_path_obj) if str(trace_path_obj) else "",
        "case_state": case_state,
        "execution_rows": execution_rows,
        "trace_rows": trace_rows,
        "generate_report_ok": _has_tool_success(case_state, "generate_report"),
    }


def _extract_run_dir_hint(*, message: str, traceback_text: str) -> str:
    text = "\n".join([str(message or ""), str(traceback_text or "")])
    pattern = re.compile(r"(/[^\s'\"\\]+/runs/[^\s'\"\\]+/\d{8}_\d{6}_[0-9a-fA-F]+)")
    m = pattern.search(text)
    if not m:
        return ""
    return str(m.group(1) or "").strip()


def _collect_success_tools(*, execution_rows: List[Dict[str, Any]], trace_rows: List[Dict[str, Any]]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()

    for row in execution_rows:
        if not isinstance(row, dict):
            continue
        if row.get("ok") is not True:
            continue
        name = str(row.get("tool_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)

    if ordered:
        return ordered

    for row in trace_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("event_type") or "") != "tool_call":
            continue
        if str(row.get("status") or "").strip().upper() != "DONE":
            continue
        name = str(row.get("tool_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)

    return ordered


def _failure_point(
    *,
    actual_status: str,
    raised_error: Optional[Dict[str, Any]],
    execution_rows: List[Dict[str, Any]],
    trace_rows: List[Dict[str, Any]],
) -> str:
    if str(actual_status).strip().lower() == "success":
        return "-"

    if isinstance(raised_error, dict) and (raised_error.get("type") or raised_error.get("message")):
        et = str(raised_error.get("type") or "RuntimeError").strip()
        msg = _short_text(raised_error.get("message") or "")
        if msg:
            return f"LLM routing - {et}: {msg}"
        return f"LLM routing - {et}"

    fails = [
        row
        for row in execution_rows
        if isinstance(row, dict) and (row.get("ok") is False)
    ]
    if fails:
        row = fails[-1]
        err = row.get("error") if isinstance(row.get("error"), dict) else {}
        et = str(err.get("type") or "RuntimeError").strip()
        msg = _short_text(err.get("message") or "")
        tool = str(row.get("tool_name") or "unknown_tool").strip()
        return f"{tool} - {et}: {msg}" if msg else f"{tool} - {et}"

    trace_fails = [
        row
        for row in trace_rows
        if isinstance(row, dict) and str(row.get("status") or "").strip().upper() == "FAIL"
    ]
    if trace_fails:
        row = trace_fails[-1]
        err = row.get("error") if isinstance(row.get("error"), dict) else {}
        et = str(err.get("type") or "RuntimeError").strip()
        msg = _short_text(err.get("message") or "")
        tool = str(row.get("tool_name") or "unknown_tool").strip()
        return f"{tool} - {et}: {msg}" if msg else f"{tool} - {et}"

    return "Unknown failure (no explicit error record found)"


def _derive_actual_status(
    *,
    ablation_mode: str,
    run_payload: Dict[str, Any],
    artifacts: Dict[str, Any],
    raised_error: Optional[Dict[str, Any]],
) -> str:
    if raised_error:
        return "failure"

    if ablation_mode in {"static_pipeline", "bcr_no_reflector", "bcr_full", "bcr_sketch", "bcr_no_token", "bcr_deterministic_only"}:
        run_result = run_payload.get("run_result") if isinstance(run_payload.get("run_result"), dict) else {}
        planner_status = str(run_payload.get("planner_status") or "ready").strip().lower()
        run_ok = bool(run_result.get("ok"))
        if planner_status == "blocked":
            return "failure"
        return "success" if run_ok else "failure"

    return "success" if bool(artifacts.get("generate_report_ok")) else "failure"


def _compute_tcr(
    *,
    case_obj: Dict[str, Any],
    actual_status: str,
    observed_error_types: List[str],
    raised_error: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    def _norm_error(v: Any) -> str:
        s = str(v or "").strip().lower()
        return re.sub(r"[^a-z0-9_]+", "", s)

    def _error_match(expected: str, observed: List[str], raised: Optional[Dict[str, Any]]) -> bool:
        exp = _norm_error(expected)
        if not exp:
            return False
        candidates: List[str] = []
        candidates.extend([_norm_error(x) for x in observed if str(x or "").strip()])
        if isinstance(raised, dict):
            candidates.append(_norm_error(raised.get("type")))
            msg = str(raised.get("message") or "")
            if msg:
                # Keep a normalized message token so expected types embedded in the message can be matched.
                candidates.append(_norm_error(msg))
        for c in candidates:
            if not c:
                continue
            if c == exp or c.endswith(exp) or exp.endswith(c) or exp in c:
                return True
        return False

    exp = (
        case_obj.get("expectations", {}).get("track_1", {})
        if isinstance(case_obj.get("expectations"), dict)
        else {}
    )
    expected_status = str(exp.get("expected_status") or "success").strip().lower()
    expected_error_type = str(exp.get("expected_error_type") or "").strip()
    tier = str(case_obj.get("tier") or "").strip().lower()

    hit = str(actual_status).strip().lower() == expected_status
    if hit and expected_status == "failure":
        # Hard-failure must fail for the expected reason (e.g., ScopeViolation), not any random crash.
        if tier == "hard_failure":
            hit = bool(expected_error_type) and _error_match(expected_error_type, observed_error_types, raised_error)
        elif expected_error_type:
            hit = _error_match(expected_error_type, observed_error_types, raised_error)

    return {
        "expected_status": expected_status,
        "expected_error_type": expected_error_type,
        "actual_status": str(actual_status).strip().lower(),
        "pass": bool(hit),
    }


def _compute_err(
    *,
    case_obj: Dict[str, Any],
    ablation_mode: str,
    artifacts: Dict[str, Any],
    actual_status: str,
    injector: FaultInjector,
    reflection_decisions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    exp_track = (
        case_obj.get("expectations", {}).get("track_1", {})
        if isinstance(case_obj.get("expectations"), dict)
        else {}
    )
    min_retries = int(exp_track.get("min_reflection_retries") or 0)
    max_retries_raw = exp_track.get("max_reflection_retries")
    max_retries = int(max_retries_raw) if max_retries_raw is not None else None

    fi = injector.spec
    target_tool = str(fi.tool_name or "").strip()
    execution_rows = artifacts.get("execution_rows") if isinstance(artifacts.get("execution_rows"), list) else []
    trace_rows = artifacts.get("trace_rows") if isinstance(artifacts.get("trace_rows"), list) else []

    target_fail_rows = [
        r
        for r in execution_rows
        if isinstance(r, dict)
        and str(r.get("tool_name") or "") == target_tool
        and (r.get("ok") is False)
    ]
    target_fail_trace = [
        r
        for r in trace_rows
        if isinstance(r, dict)
        and str(r.get("tool_name") or "") == target_tool
        and str(r.get("status") or "").strip().upper() == "FAIL"
    ]
    target_ok_rows = [
        r
        for r in execution_rows
        if isinstance(r, dict)
        and str(r.get("tool_name") or "") == target_tool
        and (r.get("ok") is True)
    ]

    retry_decisions = [
        d for d in reflection_decisions if isinstance(d, dict) and str(d.get("action") or "").strip().lower() == "retry"
    ]

    reflector_enabled = ablation_mode in {"bcr_full", "bcr_sketch"}
    reflector_caught = bool(retry_decisions) if reflector_enabled else False

    retry_count = len(retry_decisions)
    within_bounds = retry_count >= min_retries and (max_retries is None or retry_count <= max_retries)

    eligible = bool(fi.enabled) and str(fi.expected_fault_class or "") == "recoverable"
    recovered = str(actual_status).strip().lower() == "success" and bool(target_ok_rows)

    err_pass: Optional[bool]
    if not eligible:
        err_pass = None
    else:
        err_pass = bool(
            injector.applied
            and (target_fail_rows or target_fail_trace)
            and reflector_caught
            and recovered
            and within_bounds
        )

    return {
        "eligible": eligible,
        "fault_enabled": bool(fi.enabled),
        "fault_expected_class": str(fi.expected_fault_class or ""),
        "fault_applied": bool(injector.applied),
        "fault_error_observed": bool(target_fail_rows or target_fail_trace),
        "reflector_caught": bool(reflector_caught),
        "retry_count": int(retry_count),
        "retry_within_bounds": bool(within_bounds),
        "target_tool_recovered": bool(recovered),
        "pass": err_pass,
    }


def _blank_bucket() -> Dict[str, Any]:
    return {
        "runs": 0,
        "tcr_pass": 0,
        "tcr_rate": 0.0,
        "err_eligible_runs": 0,
        "err_pass": 0,
        "err_rate": None,
    }


def _update_bucket(bucket: Dict[str, Any], run_rec: Dict[str, Any]) -> None:
    bucket["runs"] += 1
    if bool(run_rec.get("tcr", {}).get("pass")):
        bucket["tcr_pass"] += 1

    err = run_rec.get("err") if isinstance(run_rec.get("err"), dict) else {}
    if bool(err.get("eligible")):
        bucket["err_eligible_runs"] += 1
        if err.get("pass") is True:
            bucket["err_pass"] += 1


def _finalize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    runs = int(bucket.get("runs") or 0)
    tcr_pass = int(bucket.get("tcr_pass") or 0)
    err_eligible = int(bucket.get("err_eligible_runs") or 0)
    err_pass = int(bucket.get("err_pass") or 0)

    out = dict(bucket)
    out["tcr_rate"] = (float(tcr_pass) / float(runs)) if runs > 0 else 0.0
    out["err_rate"] = (float(err_pass) / float(err_eligible)) if err_eligible > 0 else None
    return out


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_ablation: Dict[str, Dict[str, Any]] = {}
    by_dataset: Dict[str, Dict[str, Any]] = {}

    for rec in records:
        mode = str(rec.get("ablation_mode") or "")
        dataset = str(rec.get("dataset") or "")

        if mode not in by_ablation:
            by_ablation[mode] = _blank_bucket()
            by_ablation[mode]["datasets"] = {}
        if dataset not in by_dataset:
            by_dataset[dataset] = _blank_bucket()
            by_dataset[dataset]["ablations"] = {}

        _update_bucket(by_ablation[mode], rec)
        _update_bucket(by_dataset[dataset], rec)

        mode_ds = by_ablation[mode]["datasets"]
        if dataset not in mode_ds:
            mode_ds[dataset] = _blank_bucket()
        _update_bucket(mode_ds[dataset], rec)

        ds_mode = by_dataset[dataset]["ablations"]
        if mode not in ds_mode:
            ds_mode[mode] = _blank_bucket()
        _update_bucket(ds_mode[mode], rec)

    for mode, bucket in by_ablation.items():
        datasets = bucket.pop("datasets")
        by_ablation[mode] = _finalize_bucket(bucket)
        by_ablation[mode]["datasets"] = {k: _finalize_bucket(v) for k, v in sorted(datasets.items())}

    for dataset, bucket in by_dataset.items():
        ablations = bucket.pop("ablations")
        by_dataset[dataset] = _finalize_bucket(bucket)
        by_dataset[dataset]["ablations"] = {k: _finalize_bucket(v) for k, v in sorted(ablations.items())}

    return {
        "by_ablation": {k: by_ablation[k] for k in sorted(by_ablation.keys())},
        "by_dataset": {k: by_dataset[k] for k in sorted(by_dataset.keys())},
    }


def _pick_track1_cases(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = suite.get("cases") if isinstance(suite.get("cases"), list) else []
    out: List[Dict[str, Any]] = []
    for c in rows:
        if not isinstance(c, dict):
            continue
        tracks = c.get("tracks") if isinstance(c.get("tracks"), list) else []
        if "track_1" in [str(x) for x in tracks]:
            out.append(c)
    return out


def _pick_ablation_order(suite: Dict[str, Any]) -> List[str]:
    rows = suite.get("ablation_modes") if isinstance(suite.get("ablation_modes"), list) else []
    ordered: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        mode_id = str(row.get("id") or "").strip()
        if mode_id in EXPECTED_ABLATIONS:
            ordered.append(mode_id)
    if not ordered:
        return list(EXPECTED_ABLATIONS)

    # Preserve suite order but ensure all required modes exist exactly once.
    dedup: List[str] = []
    seen: set[str] = set()
    for m in ordered:
        if m in seen:
            continue
        dedup.append(m)
        seen.add(m)
    for m in EXPECTED_ABLATIONS:
        if m not in seen:
            dedup.append(m)
    return dedup


def run_track1_benchmark(
    *,
    suite_path: Path,
    output_path: Path,
    workspace_root: Path,
    runs_root: Path,
    server_base_url: str,
    server_model_override: str,
    max_new_tokens: int,
    max_steps: int,
    max_retries: int,
    case_filter: List[str],
) -> Dict[str, Any]:
    suite = _read_json(suite_path)
    if not suite:
        raise ValueError(f"Failed to parse benchmark suite: {suite_path}")

    backend = _find_qwen_backend(
        suite,
        server_base_url=server_base_url,
        server_model_override=server_model_override,
    )

    cases = _pick_track1_cases(suite)
    if case_filter:
        keep = set([str(x).strip() for x in case_filter if str(x).strip()])
        cases = [c for c in cases if str(c.get("id") or "") in keep]
    if not cases:
        raise ValueError("No Track 1 cases selected.")

    ablation_order = _pick_ablation_order(suite)

    records: List[Dict[str, Any]] = []
    for case_obj in cases:
        case_raw_id = str(case_obj.get("id") or "case")
        case_ref = str(case_obj.get("case_ref") or "").strip()
        if not case_ref:
            raise ValueError(f"Case {case_raw_id} missing case_ref")

        for mode in ablation_order:
            benchmark_case_id = _sanitize_case_id(f"{case_raw_id}__{mode}")
            injector = FaultInjector(spec=_fault_spec_from_case(case_obj))

            run_payload: Dict[str, Any] = {}
            raised_error: Optional[Dict[str, Any]] = None
            err_tb = ""

            try:
                if mode in {"static_pipeline", "bcr_no_reflector", "bcr_full", "bcr_sketch"}:
                    run_payload = _run_cerebellum_mode(
                        case_obj=case_obj,
                        case_id=benchmark_case_id,
                        ablation_mode=mode,
                        backend=backend,
                        workspace_root=workspace_root,
                        runs_root=runs_root,
                        max_new_tokens=max_new_tokens,
                        injector=injector,
                    )
                elif mode == "pure_react":
                    run_payload = _run_pure_react_mode(
                        case_obj=case_obj,
                        case_id=benchmark_case_id,
                        backend=backend,
                        runs_root=runs_root,
                        max_new_tokens=max_new_tokens,
                        max_steps=max_steps,
                        max_retries=max_retries,
                        injector=injector,
                    )
                else:
                    raise ValueError(f"Unsupported ablation_mode: {mode}")
            except Exception as e:
                err_tb = traceback.format_exc(limit=60)
                run_dir_hint = _extract_run_dir_hint(message=str(e), traceback_text=err_tb)
                if run_dir_hint:
                    run_payload["run_dir"] = run_dir_hint
                    if mode == "pure_react":
                        run_payload["trace_path"] = str(Path(run_dir_hint) / "agent_trace.jsonl")
                raised_error = {
                    "type": type(e).__name__,
                    "message": str(e),
                    "traceback": err_tb,
                }

            run_dir = str(run_payload.get("run_dir") or "")
            trace_path = str(run_payload.get("trace_path") or "")
            artifacts = _collect_run_artifacts(run_dir=run_dir, trace_path=trace_path)

            observed_errors = _collect_error_types(
                execution_rows=artifacts.get("execution_rows") if isinstance(artifacts.get("execution_rows"), list) else [],
                trace_rows=artifacts.get("trace_rows") if isinstance(artifacts.get("trace_rows"), list) else [],
                raised_error=raised_error,
            )

            actual_status = _derive_actual_status(
                ablation_mode=mode,
                run_payload=run_payload,
                artifacts=artifacts,
                raised_error=raised_error,
            )

            reflection_decisions = run_payload.get("reflection_decisions") if isinstance(run_payload.get("reflection_decisions"), list) else []
            tcr = _compute_tcr(
                case_obj=case_obj,
                actual_status=actual_status,
                observed_error_types=observed_errors,
                raised_error=raised_error,
            )
            err = _compute_err(
                case_obj=case_obj,
                ablation_mode=mode,
                artifacts=artifacts,
                actual_status=actual_status,
                injector=injector,
                reflection_decisions=reflection_decisions,
            )

            rec = {
                "timestamp": _utc_now_iso(),
                "case_id": case_raw_id,
                "benchmark_case_id": benchmark_case_id,
                "dataset": str(case_obj.get("dataset") or ""),
                "domain": str(case_obj.get("domain") or ""),
                "tier": str(case_obj.get("tier") or ""),
                "ablation_mode": mode,
                "llm_backend": backend.backend_id,
                "planner_status": str(run_payload.get("planner_status") or ""),
                "tcr": tcr,
                "err": err,
                "fault_injection": {
                    "enabled": bool(injector.spec.enabled),
                    "target_tool": str(injector.spec.tool_name or ""),
                    "mutation": {
                        "type": str(injector.spec.mutation_type or ""),
                        "key": str(injector.spec.mutation_key or ""),
                        "value": injector.spec.mutation_value,
                    },
                    "applied": bool(injector.applied),
                    "events": list(injector.events),
                },
                "reflection_decisions": reflection_decisions,
                "run_artifacts": {
                    "run_dir": artifacts.get("run_dir"),
                    "case_state_path": artifacts.get("case_state_path"),
                    "execution_log_path": artifacts.get("execution_log_path"),
                    "trace_path": artifacts.get("trace_path"),
                },
                "observed_error_types": observed_errors,
                "raised_error": raised_error,
            }
            records.append(rec)
            execution_rows = artifacts.get("execution_rows") if isinstance(artifacts.get("execution_rows"), list) else []
            trace_rows = artifacts.get("trace_rows") if isinstance(artifacts.get("trace_rows"), list) else []
            success_tools = _collect_success_tools(execution_rows=execution_rows, trace_rows=trace_rows)
            failure_desc = _failure_point(
                actual_status=str(tcr.get("actual_status") or ""),
                raised_error=raised_error,
                execution_rows=execution_rows,
                trace_rows=trace_rows,
            )
            prompt_preview = _short_text(case_obj.get("prompt") or "", max_chars=100)
            tcr_label = "PASS" if bool(tcr.get("pass")) else "FAIL"
            err_pass = err.get("pass")
            err_label = "NA" if err_pass is None else ("PASS" if bool(err_pass) else "FAIL")
            success_path = " -> ".join(success_tools) if success_tools else "-"
            rec["run_monitor"] = {
                "prompt_preview": prompt_preview,
                "success_tools": success_tools,
                "success_path": success_path,
                "failure_point": failure_desc,
                "tcr_label": tcr_label,
                "err_label": err_label,
            }

            print(f"\n[{mode}] {case_raw_id}")
            print(f"  Prompt: {prompt_preview}")
            print(f"  Success: [{success_path}]")
            if str(tcr.get("actual_status") or "").strip().lower() == "failure":
                print(f"  Failed at: {failure_desc}")
            else:
                print("  Failed at: -")
            print(f"  TCR: {tcr_label} (actual={tcr.get('actual_status')}, expected={tcr.get('expected_status')})")
            print(f"  ERR: {err_label}")

    aggregate = _aggregate(records)
    result_payload = {
        "suite_id": str(suite.get("suite_id") or ""),
        "suite_version": str(suite.get("version") or ""),
        "generated_at": _utc_now_iso(),
        "track": "track_1",
        "forced_llm_backend": {
            "id": backend.backend_id,
            "provider": backend.provider,
            "model": backend.model,
            "base_url": backend.base_url,
        },
        "runner_config": {
            "suite_path": str(suite_path),
            "workspace_root": str(workspace_root),
            "runs_root": str(runs_root),
            "max_new_tokens": int(max_new_tokens),
            "max_steps": int(max_steps),
            "max_retries": int(max_retries),
            "ablations": ablation_order,
            "case_filter": list(case_filter),
        },
        "results": records,
        "aggregates": aggregate,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_payload


def build_arg_parser() -> argparse.ArgumentParser:
    root = project_root()
    ap = argparse.ArgumentParser(description="Run Track 1 benchmark ablations (BCER robustness study)")
    ap.add_argument(
        "--suite",
        type=str,
        default=str(root / "benchmark" / "benchmark_suite.json"),
        help="Path to benchmark_suite.json",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=str(root / "benchmark" / "benchmark_results_track1.json"),
        help="Output JSON path",
    )
    ap.add_argument(
        "--workspace-root",
        type=str,
        default=str(root),
        help="Workspace root passed into SessionState",
    )
    ap.add_argument(
        "--runs-root",
        type=str,
        default=str(root / "runs" / "benchmark_track1"),
        help="Runs root for benchmark execution artifacts",
    )
    ap.add_argument(
        "--server-base-url",
        type=str,
        default=str(os.environ.get("MRI_AGENT_SHELL_SERVER_BASE_URL") or "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible server base URL for local Qwen backend",
    )
    ap.add_argument(
        "--server-model",
        type=str,
        default=DEFAULT_SERVER_MODEL,
        help="Model override for qwen_local_default",
    )
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument(
        "--case",
        action="append",
        default=[],
        help="Optional case id filter; can be passed multiple times",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    suite_path = Path(args.suite).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()

    payload = run_track1_benchmark(
        suite_path=suite_path,
        output_path=output_path,
        workspace_root=workspace_root,
        runs_root=runs_root,
        server_base_url=str(args.server_base_url),
        server_model_override=str(args.server_model),
        max_new_tokens=int(args.max_new_tokens),
        max_steps=int(args.max_steps),
        max_retries=int(args.max_retries),
        case_filter=[str(x) for x in (args.case or [])],
    )

    agg = payload.get("aggregates") if isinstance(payload.get("aggregates"), dict) else {}
    by_mode = agg.get("by_ablation") if isinstance(agg.get("by_ablation"), dict) else {}
    print("\nTrack 1 benchmark finished.")
    print(f"Output: {output_path}")
    for mode in sorted(by_mode.keys()):
        row = by_mode.get(mode) if isinstance(by_mode.get(mode), dict) else {}
        tcr_rate = float(row.get("tcr_rate") or 0.0)
        err_rate = row.get("err_rate")
        err_txt = "NA" if err_rate is None else f"{float(err_rate):.3f}"
        print(
            f"- {mode}: TCR={tcr_rate:.3f} ({int(row.get('tcr_pass') or 0)}/{int(row.get('runs') or 0)}), "
            f"ERR={err_txt} ({int(row.get('err_pass') or 0)}/{int(row.get('err_eligible_runs') or 0)})"
        )


if __name__ == "__main__":
    main()

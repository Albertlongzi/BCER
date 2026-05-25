from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


DomainName = Literal["prostate", "brain", "cardiac", "demo"]


def build_scope_id(*, domain: str, case_id: str, case_ref: str) -> str:
    raw = f"{str(domain).strip().lower()}|{str(case_id).strip()}|{str(case_ref).strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class CaseScope(BaseModel):
    scope_id: str = ""
    domain: DomainName
    case_id: str
    case_ref: str
    workspace_root: str
    runs_root: str
    allow_external_model_roots: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _populate_scope_id(self) -> "CaseScope":
        if not str(self.scope_id or "").strip():
            self.scope_id = build_scope_id(
                domain=self.domain,
                case_id=self.case_id,
                case_ref=self.case_ref,
            )
        return self


class DagPolicy(BaseModel):
    skip_optional_by_default: bool = True
    stop_on_required_failure: bool = True
    max_attempts_default: int = 2


class PlanNode(BaseModel):
    node_id: str
    tool_name: str = ""
    node_type: Literal["tool", "reflect", "gate"] = "tool"
    stage: str = "misc"
    arguments: Dict[str, Any] = Field(default_factory=dict)
    required: bool = True
    depends_on: List[str] = Field(default_factory=list)
    label: Optional[str] = None
    max_attempts: Optional[int] = None

    @model_validator(mode="after")
    def _ensure_node_id(self) -> "PlanNode":
        if not str(self.node_id or "").strip():
            raise ValueError("PlanNode.node_id must be non-empty")
        self.node_type = str(self.node_type or "tool").strip().lower() or "tool"  # type: ignore[assignment]
        if self.node_type == "tool" and not str(self.tool_name or "").strip():
            raise ValueError("PlanNode.tool_name must be non-empty for node_type=tool")
        if self.node_type in {"reflect", "gate"} and not str(self.tool_name or "").strip():
            self.tool_name = f"__{self.node_type}__"
        if not isinstance(self.arguments, dict):
            raise ValueError("PlanNode.arguments must be a dict")
        self.depends_on = [str(x).strip() for x in (self.depends_on or []) if str(x).strip()]
        return self


class AgentPlanDAG(BaseModel):
    version: Literal["1.0"] = "1.0"
    plan_id: str
    template_id: str = ""
    template_version: str = ""
    template_hash: str = ""
    goal: str = ""
    case_scope: CaseScope
    planner_status: Literal["ready", "blocked", "clarify"] = "ready"
    natural_language_response: str = ""
    blocking_reasons: List[str] = Field(default_factory=list)
    requested_request_type: str = ""
    policy: DagPolicy = Field(default_factory=DagPolicy)
    nodes: List[PlanNode] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    planner_metadata: Dict[str, Any] = Field(default_factory=dict)
    planner_artifacts: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_nodes(self) -> "AgentPlanDAG":
        if not str(self.plan_id or "").strip():
            raise ValueError("AgentPlanDAG.plan_id must be non-empty")
        seen = set()
        for node in self.nodes:
            if node.node_id in seen:
                raise ValueError(f"Duplicate node_id in DAG: {node.node_id}")
            seen.add(node.node_id)
        return self


def _node_id_from_legacy(tool_name: str, idx: int) -> str:
    base = str(tool_name or "node").strip().lower().replace(" ", "_")
    return f"{base}_{idx:03d}"


def legacy_plan_to_dag(
    *,
    legacy_plan: Dict[str, Any],
    domain: str,
    case_id: str,
    case_ref: str,
    workspace_root: str,
    runs_root: str,
    plan_id: Optional[str] = None,
) -> AgentPlanDAG:
    plan_obj = dict(legacy_plan or {})
    steps = plan_obj.get("steps")
    if not isinstance(steps, list):
        steps = []

    nodes: List[PlanNode] = []
    by_tool_to_ids: Dict[str, List[str]] = {}
    resolved_ids = set()
    for idx, raw in enumerate(steps):
        if not isinstance(raw, dict):
            continue
        tool_name = str(raw.get("tool_name") or "").strip()
        if not tool_name:
            continue
        node_id = str(raw.get("node_id") or "").strip() or _node_id_from_legacy(tool_name, idx)
        depends_on: List[str] = []
        raw_deps = raw.get("depends_on")
        if isinstance(raw_deps, list):
            for dep in raw_deps:
                dep_s = str(dep or "").strip()
                if not dep_s:
                    continue
                if dep_s in resolved_ids:
                    depends_on.append(dep_s)
                    continue
                # Legacy plans use tool names in depends_on. Map to latest seen node of that tool.
                dep_ids = by_tool_to_ids.get(dep_s) or []
                if dep_ids:
                    depends_on.append(dep_ids[-1])
                else:
                    depends_on.append(dep_s)
        node = PlanNode(
            node_id=node_id,
            tool_name=tool_name,
            stage=str(raw.get("stage") or "misc"),
            arguments=dict(raw.get("arguments") or {}),
            required=bool(raw.get("required", True)),
            depends_on=depends_on,
            label=str(raw.get("label") or raw.get("description") or "").strip() or None,
            max_attempts=(
                int(raw.get("max_attempts"))
                if raw.get("max_attempts") is not None and str(raw.get("max_attempts")).strip()
                else None
            ),
        )
        nodes.append(node)
        resolved_ids.add(node_id)
        by_tool_to_ids.setdefault(tool_name, []).append(node_id)

    dag = AgentPlanDAG(
        plan_id=str(plan_id or plan_obj.get("plan_id") or plan_obj.get("run_id") or f"legacy_{case_id}"),
        goal=str(plan_obj.get("goal") or ""),
        case_scope=CaseScope(
            domain=str(domain or "prostate").strip().lower(),  # type: ignore[arg-type]
            case_id=str(case_id or "").strip(),
            case_ref=str(case_ref or "").strip(),
            workspace_root=str(workspace_root or "").strip(),
            runs_root=str(runs_root or "").strip(),
        ),
        policy=DagPolicy(),
        nodes=nodes,
        notes=[str(x) for x in (plan_obj.get("notes") or []) if str(x).strip()],
    )
    return dag

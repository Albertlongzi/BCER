from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field, ConfigDict, model_validator


class SketchStep(BaseModel):
    """One constrained tool-level step emitted by the sketch planner."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    tool: str
    depends_on: List[str] = Field(default_factory=list)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    goal: str = ""
    optional: bool = False
    checks: List[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _validate_fields(self) -> "SketchStep":
        self.step_id = str(self.step_id or "").strip()
        self.tool = str(self.tool or "").strip()
        self.goal = str(self.goal or "").strip()
        self.notes = str(self.notes or "").strip()
        self.depends_on = [str(x).strip() for x in (self.depends_on or []) if str(x).strip()]
        self.checks = [str(x).strip() for x in (self.checks or []) if str(x).strip()]
        if not self.step_id:
            raise ValueError("SketchStep.step_id must be non-empty")
        if not self.tool:
            raise ValueError("SketchStep.tool must be non-empty")
        if not isinstance(self.inputs, dict):
            raise ValueError("SketchStep.inputs must be an object")
        return self


class ConstrainedPlanSketch(BaseModel):
    """Constrained coarse DAG produced by the LLM sketch planner."""

    # Top-level planner outputs from different models may include harmless extra
    # fields (e.g., request_type). Ignore them instead of hard-failing compile.
    model_config = ConfigDict(extra="ignore")

    task: str
    domain: str
    steps: List[SketchStep] = Field(default_factory=list)
    final_targets: List[str] = Field(default_factory=list)
    planner_notes: str = ""

    @model_validator(mode="after")
    def _validate_fields(self) -> "ConstrainedPlanSketch":
        self.task = str(self.task or "").strip()
        self.domain = str(self.domain or "").strip().lower()
        self.planner_notes = str(self.planner_notes or "").strip()
        self.final_targets = [str(x).strip() for x in (self.final_targets or []) if str(x).strip()]
        if not self.task:
            raise ValueError("ConstrainedPlanSketch.task must be non-empty")
        if not self.domain:
            raise ValueError("ConstrainedPlanSketch.domain must be non-empty")
        if not isinstance(self.steps, list) or not self.steps:
            raise ValueError("ConstrainedPlanSketch.steps must be a non-empty list")

        seen: set[str] = set()
        for step in self.steps:
            if step.step_id in seen:
                raise ValueError(f"Duplicate sketch step_id: {step.step_id}")
            seen.add(step.step_id)
        return self


def constrained_plan_sketch_guided_json_schema() -> Dict[str, Any]:
    """JSON schema for guided decoding / best-effort JSON enforcement."""

    return ConstrainedPlanSketch.model_json_schema()

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, model_validator

from core.plan_dag import PlanNode


class PlanTemplate(BaseModel):
    template_id: str
    template_version: str
    domain: str
    request_type: str
    notes: List[str] = Field(default_factory=list)
    nodes: List[PlanNode] = Field(default_factory=list)

    source_path: str = ""
    template_hash: str = ""

    @model_validator(mode="after")
    def _validate_core_fields(self) -> "PlanTemplate":
        self.template_id = str(self.template_id or "").strip()
        self.template_version = str(self.template_version or "").strip()
        self.domain = str(self.domain or "").strip().lower()
        self.request_type = str(self.request_type or "").strip().lower()
        if not self.template_id:
            raise ValueError("PlanTemplate.template_id must be non-empty")
        if not self.template_version:
            raise ValueError("PlanTemplate.template_version must be non-empty")
        if not self.domain:
            raise ValueError("PlanTemplate.domain must be non-empty")
        if not self.request_type:
            raise ValueError("PlanTemplate.request_type must be non-empty")
        return self

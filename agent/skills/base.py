from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    tools: List[str]
    preconditions: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    checks: List[str] = field(default_factory=list)
    guidance: Dict[str, Any] = field(default_factory=dict)


class Skill:
    """Lightweight placeholder interface for skill packs."""

    spec: SkillSpec

    def run(self, ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

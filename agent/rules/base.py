from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    level: str  # hard|soft

    def validate(self, payload: Dict[str, Any]) -> None:
        return None

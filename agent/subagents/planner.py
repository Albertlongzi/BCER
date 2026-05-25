from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .base import Subagent
from .prompts import PLAN_SYSTEM_PROMPT


@dataclass
class PlannerSubagent(Subagent):
    llm: Any

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": payload.get("plan_input_text", "")},
        ]
        text = ""
        try:
            text = str(self.llm.generate(messages)).strip()
        except Exception:
            text = ""
        return {"plan_text": text, "messages": messages}

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .base import Subagent
from .prompts import REFLECT_SYSTEM_PROMPT


@dataclass
class ReflectorSubagent(Subagent):
    llm: Any

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": REFLECT_SYSTEM_PROMPT},
            {"role": "user", "content": payload.get("reflect_input_text", "")},
        ]
        text = ""
        try:
            text = str(self.llm.generate(messages)).strip()
        except Exception:
            text = ""
        return {"reflection": text, "messages": messages}

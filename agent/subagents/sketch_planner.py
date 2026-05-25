from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .base import Subagent


@dataclass
class SketchPlannerSubagent(Subagent):
    llm: Any
    system_prompt: str

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": str(self.system_prompt or "")},
            {"role": "user", "content": payload.get("plan_input_text", "")},
        ]
        text = ""
        err: str | None = None
        guided_json = payload.get("guided_json_schema") if isinstance(payload.get("guided_json_schema"), dict) else None
        errors: List[str] = []
        used_schema = False

        if guided_json is not None and hasattr(self.llm, "generate_with_schema"):
            used_schema = True
            try:
                text = str(self.llm.generate_with_schema(messages, guided_json)).strip()
            except Exception as e:  # noqa: BLE001
                errors.append(f"generate_with_schema failed: {repr(e)}")
                text = ""
            if not text:
                # Some providers/models return empty content under guided JSON even when
                # the request succeeds. Fall back to plain generation so planner artifacts
                # capture a real model response instead of a silent empty string.
                try:
                    text = str(self.llm.generate(messages)).strip()
                    if text:
                        errors.append("generate_with_schema returned empty; used plain generate fallback")
                except Exception as e:  # noqa: BLE001
                    errors.append(f"generate fallback failed: {repr(e)}")
                    text = ""
        else:
            try:
                text = str(self.llm.generate(messages)).strip()
            except Exception as e:  # noqa: BLE001
                errors.append(f"generate failed: {repr(e)}")
                text = ""

        if errors:
            err = " | ".join(errors)
        elif used_schema and not text:
            err = "generate_with_schema returned empty and no fallback content"
        return {"sketch_text": text, "messages": messages, "error": err}

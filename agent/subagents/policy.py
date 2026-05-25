from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base import Subagent


def build_guided_schema(*, allow_tool_calls: bool = True) -> Dict[str, Any]:
    one_of = [
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["tool_call"]},
                "tool_name": {"type": "string"},
                "arguments": {"type": "object"},
                "stage": {"type": "string"},
            },
            "required": ["action", "tool_name", "arguments"],
        },
        {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["final"]},
                "final_report": {},
            },
            "required": ["action"],
        },
    ]
    if allow_tool_calls:
        one_of.insert(
            1,
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["tool_calls"]},
                    "calls": {"type": "array"},
                },
                "required": ["action", "calls"],
            },
        )
    return {"type": "object", "oneOf": one_of}


@dataclass
class PolicyResult:
    raw: str
    parsed: Optional[Dict[str, Any]]
    error: Optional[str]


class PolicySubagent(Subagent):
    def __init__(
        self,
        llm: Any,
        parse_fn,
        guided_schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.llm = llm
        self.parse_fn = parse_fn
        self.guided_schema = guided_schema

    def generate(
        self,
        messages: List[Dict[str, Any]],
        *,
        use_schema: bool = True,
        extra_msg: Optional[str] = None,
    ) -> PolicyResult:
        cur_messages = messages
        if extra_msg:
            cur_messages = (cur_messages or []) + [{"role": "user", "content": extra_msg}]
        try:
            if use_schema and self.guided_schema and hasattr(self.llm, "generate_with_schema"):
                out = str(self.llm.generate_with_schema(cur_messages, self.guided_schema)).strip()
            else:
                out = str(self.llm.generate(cur_messages)).strip()
            act, perr = self.parse_fn(out)
            if act is None:
                return PolicyResult(raw=out, parsed=None, error=perr or "parse_error")
            return PolicyResult(raw=out, parsed={"action": act.action, **act.payload}, error=None)
        except Exception as e:
            return PolicyResult(raw="", parsed=None, error=repr(e))

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("PolicySubagent.run is not used; call generate().")

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ParsedAction:
    action: str  # "tool_call" | "final"
    payload: Dict[str, Any]


def extract_outermost_json_object(text: str) -> Optional[str]:
    """
    Extract the first *balanced* JSON object substring from text.
    Robust against leading/trailing chatter.

    Notes:
    - Handles nested braces.
    - Attempts to ignore braces inside JSON strings.
    """
    if not text:
        return None

    s = text
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return None


def try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON parsing:
    - direct json.loads(text)
    - else extract balanced {...} and parse
    """
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    frag = extract_outermost_json_object(text)
    if not frag:
        return None
    try:
        obj = json.loads(frag)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def validate_tool_action(obj: Dict[str, Any]) -> ParsedAction:
    """
    Validate model output into a minimal action contract.

    Supported actions:
    - tool_call: {"action":"tool_call","tool_name":str,"arguments":dict, "stage"?:str}
    - tool_calls: {"action":"tool_calls","calls":[{tool_call}, ...]}
    - final: {"action":"final","final_report":<any>}
    """
    action = obj.get("action")
    if action in ("tools/call", "call_tool"):
        action = "tool_call"
    if action not in ("tool_call", "tool_calls", "final"):
        raise ValueError("Missing/invalid 'action'. Expected 'tool_call', 'tool_calls', or 'final'.")

    if action == "tool_call":
        tool_name = obj.get("tool_name")
        arguments = obj.get("arguments")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tool_call requires non-empty 'tool_name' (string).")
        if not isinstance(arguments, dict):
            raise ValueError("tool_call requires 'arguments' (object/dict).")
        stage = obj.get("stage", "misc")
        if stage is not None and not isinstance(stage, str):
            raise ValueError("'stage' must be a string if provided.")
        return ParsedAction(action="tool_call", payload={"tool_name": tool_name, "arguments": arguments, "stage": stage or "misc"})

    if action == "tool_calls":
        calls = obj.get("calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError("tool_calls requires non-empty 'calls' (array).")
        norm_calls = []
        for i, c in enumerate(calls):
            if not isinstance(c, dict):
                raise ValueError(f"calls[{i}] must be an object/dict.")
            tool_name = c.get("tool_name")
            arguments = c.get("arguments")
            stage = c.get("stage", "misc")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError(f"calls[{i}].tool_name must be a non-empty string.")
            if not isinstance(arguments, dict):
                raise ValueError(f"calls[{i}].arguments must be an object/dict.")
            if stage is not None and not isinstance(stage, str):
                raise ValueError(f"calls[{i}].stage must be a string if provided.")
            norm_calls.append({"tool_name": tool_name, "arguments": arguments, "stage": stage or "misc"})
        return ParsedAction(action="tool_calls", payload={"calls": norm_calls})

    # final
    return ParsedAction(action="final", payload={"final_report": obj.get("final_report", obj.get("report", obj))})


def parse_model_output(text: str) -> Tuple[Optional[ParsedAction], Optional[str]]:
    """
    Parse + validate. Returns (ParsedAction|None, error_message|None).
    """
    obj = try_parse_json(text)
    if obj is None:
        return None, "Could not parse any JSON object from model output."
    try:
        act = validate_tool_action(obj)
        return act, None
    except Exception as e:
        return None, f"JSON parsed but failed validation: {e}"


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class HookContext:
    tool_name: str
    stage: str
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


def pre_tool(ctx: HookContext) -> HookContext:
    return ctx


def post_tool(ctx: HookContext, result: Dict[str, Any]) -> Dict[str, Any]:
    return result

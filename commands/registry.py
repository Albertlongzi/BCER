from __future__ import annotations

from dataclasses import dataclass
import difflib
from typing import Any, Callable, Dict, List, Optional

from .schemas import ToolContext, ToolSpec


ToolCallable = Callable[[Dict[str, Any], ToolContext], Dict[str, Any]]


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    func: ToolCallable


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._aliases: Dict[str, str] = {
            "package_evidence": "package_vlm_evidence",
            "package evidence": "package_vlm_evidence",
            "packageevidence": "package_vlm_evidence",
        }

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def list_specs(self) -> List[Dict[str, Any]]:
        return [t.spec.to_dict() for t in self._tools.values()]

    def _normalize(self, name: str) -> str:
        return "".join(ch.lower() for ch in str(name or "") if ch.isalnum())

    def _resolve_name(self, name: str) -> Optional[str]:
        if not name:
            return None
        if name in self._tools:
            return name
        alias = self._aliases.get(name)
        if alias in self._tools:
            return alias
        norm = self._normalize(name)
        if not norm:
            return None
        # Match against normalized tool names.
        norm_to_tool: Dict[str, str] = {self._normalize(k): k for k in self._tools.keys()}
        if norm in norm_to_tool:
            return norm_to_tool[norm]
        # Fuzzy match as a last resort.
        candidates = difflib.get_close_matches(norm, list(norm_to_tool.keys()), n=1, cutoff=0.86)
        if candidates:
            return norm_to_tool[candidates[0]]
        return None

    def get(self, name: str) -> Tool:
        resolved = self._resolve_name(name)
        if not resolved or resolved not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[resolved]

    def has(self, name: str) -> bool:
        return name in self._tools

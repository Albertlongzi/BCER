from __future__ import annotations

from typing import Any, Dict


class Subagent:
    """Minimal interface for planner/reflector/policy subagents."""

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

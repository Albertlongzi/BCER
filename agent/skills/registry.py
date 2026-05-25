from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import SkillSpec


@dataclass
class SkillRegistry:
    skills: Dict[str, SkillSpec]
    skill_sets: Dict[str, List[str]]

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "SkillRegistry":
        skills: Dict[str, SkillSpec] = {}
        for item in data.get("skills", []) or []:
            if not isinstance(item, dict) or "name" not in item:
                continue
            skills[item["name"]] = SkillSpec(
                name=str(item.get("name")),
                description=str(item.get("description", "")),
                tools=list(item.get("tools", []) or []),
                preconditions=list(item.get("preconditions", []) or []),
                outputs=list(item.get("outputs", []) or []),
                tags=list(item.get("tags", []) or []),
                inputs=list(item.get("inputs", []) or []),
                checks=list(item.get("checks", []) or []),
                guidance=item.get("guidance", {}) if isinstance(item.get("guidance", {}), dict) else {},
            )
        skill_sets = data.get("skill_sets", {}) or {}
        if not isinstance(skill_sets, dict):
            skill_sets = {}
        return cls(skills=skills, skill_sets=skill_sets)

    @classmethod
    def load(cls, path: Path) -> "SkillRegistry":
        if not path.exists():
            return cls(skills={}, skill_sets={})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        return cls.from_json(data if isinstance(data, dict) else {})

    def list_specs(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in self.skills.values():
            out.append(
                {
                    "name": s.name,
                    "description": s.description,
                    "tools": s.tools,
                    "preconditions": s.preconditions,
                    "outputs": s.outputs,
                    "tags": s.tags,
                    "inputs": s.inputs,
                    "checks": s.checks,
                    "guidance": s.guidance,
                }
            )
        return out

    def resolve_set(self, set_name: Optional[str]) -> List[SkillSpec]:
        if not set_name:
            return list(self.skills.values())
        names = self.skill_sets.get(set_name, []) or []
        return [self.skills[n] for n in names if n in self.skills]

    def to_prompt(
        self,
        *,
        set_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_items: int = 12,
    ) -> str:
        skills = self.resolve_set(set_name)
        if tags:
            tags_norm = {t.strip().lower() for t in tags if isinstance(t, str) and t.strip()}
            if tags_norm:
                skills = [s for s in skills if any(str(tag).lower() in tags_norm for tag in s.tags)]
        if not skills:
            return "(none)"
        lines = []
        for s in skills[:max_items]:
            tools = ", ".join(s.tools[:8])
            pre = "; ".join(s.preconditions[:3])
            out = "; ".join(s.outputs[:3])
            tags = ", ".join(s.tags[:4])
            inputs = ", ".join(s.inputs[:3])
            checks = "; ".join(s.checks[:2])
            line = f"- {s.name}: {s.description} | tools=[{tools}]"
            if inputs:
                line += f" | inputs={inputs}"
            if pre:
                line += f" | pre={pre}"
            if checks:
                line += f" | checks={checks}"
            if out:
                line += f" | outputs={out}"
            if tags:
                line += f" | tags={tags}"
            lines.append(line)
        if len(skills) > max_items:
            lines.append("- [TRUNCATED]")
        return "\n".join(lines)

    def to_guidance(
        self,
        *,
        set_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_items: int = 8,
    ) -> List[Dict[str, Any]]:
        skills = self.resolve_set(set_name)
        if tags:
            tags_norm = {t.strip().lower() for t in tags if isinstance(t, str) and t.strip()}
            if tags_norm:
                skills = [s for s in skills if any(str(tag).lower() in tags_norm for tag in s.tags)]
        out: List[Dict[str, Any]] = []
        for s in skills:
            if not s.inputs and not s.checks:
                continue
            out.append(
                {
                    "name": s.name,
                    "inputs": list(s.inputs[:8]),
                    "checks": list(s.checks[:8]),
                }
            )
        if len(out) > max_items:
            out = out[:max_items] + [{"truncated": True, "total": len(out)}]
        return out

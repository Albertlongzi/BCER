from __future__ import annotations

import json
import re
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Subagent
from .prompts import ALIGNMENT_GATE_SYSTEM_PROMPT


def _encode_png_as_data_url(path: Path) -> Optional[str]:
    try:
        b = path.read_bytes()
    except Exception:
        return None
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to extract JSON object from mixed text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


@dataclass
class AlignmentGateResult:
    raw: str
    parsed: Optional[Dict[str, Any]]
    error: Optional[str]


class AlignmentGateSubagent(Subagent):
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def decide(
        self,
        *,
        payload: Dict[str, Any],
        image_paths: List[Dict[str, Any]],
        use_schema: bool = False,
    ) -> AlignmentGateResult:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": ALIGNMENT_GATE_SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})

        # Attach images (if any) as multimodal blocks.
        for img in image_paths:
            p = Path(str(img.get("path") or "")).expanduser()
            if not p.exists():
                continue
            label = str(img.get("label") or img.get("name") or "image")
            data_url = _encode_png_as_data_url(p)
            if not data_url:
                continue
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Image: {label}"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            )

        try:
            if use_schema and hasattr(self.llm, "generate_with_schema"):
                # Best-effort: ask for JSON object but avoid strict schema to keep compatibility.
                raw = str(self.llm.generate_with_schema(messages, {"type": "object"})).strip()
            else:
                raw = str(self.llm.generate(messages)).strip()
        except Exception as e:
            return AlignmentGateResult(raw="", parsed=None, error=repr(e))

        parsed = _extract_json(raw)
        if parsed is None:
            return AlignmentGateResult(raw=raw, parsed=None, error="parse_error")
        return AlignmentGateResult(raw=raw, parsed=parsed, error=None)

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("AlignmentGateSubagent.run is not used; call decide().")

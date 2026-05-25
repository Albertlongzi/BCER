from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import Subagent
from .prompts import DISTORTION_GATE_SYSTEM_PROMPT


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
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


@dataclass
class DistortionGateResult:
    raw: str
    parsed: Optional[Dict[str, Any]]
    error: Optional[str]


class DistortionGateSubagent(Subagent):
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def decide(
        self,
        *,
        payload: Dict[str, Any],
        image_paths: List[Dict[str, Any]],
        use_schema: bool = False,
    ) -> DistortionGateResult:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": DISTORTION_GATE_SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})

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
                raw = str(self.llm.generate_with_schema(messages, {"type": "object"})).strip()
            else:
                raw = str(self.llm.generate(messages)).strip()
        except Exception as e:
            return DistortionGateResult(raw="", parsed=None, error=repr(e))

        parsed = _extract_json(raw)
        if parsed is None:
            return DistortionGateResult(raw=raw, parsed=None, error="parse_error")
        return DistortionGateResult(raw=raw, parsed=parsed, error=None)

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("DistortionGateSubagent.run is not used; call decide().")

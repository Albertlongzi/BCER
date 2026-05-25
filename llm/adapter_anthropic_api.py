from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AnthropicConfig:
    model: str = os.environ.get("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
    api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    timeout_s: int = 600
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.9


def _split_data_url(url: str) -> Tuple[str, str]:
    # Returns (media_type, b64_data)
    if not url.startswith("data:"):
        raise ValueError("Anthropic adapter only supports data: URLs or local files for images.")
    header, b64 = url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Only base64 data URLs are supported.")
    media = header.split(";", 1)[0].replace("data:", "") or "image/png"
    return media, b64


def _image_block_from_url(url: str) -> Dict[str, Any]:
    if url.startswith("data:"):
        media_type, b64 = _split_data_url(url)
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}
    if url.startswith("file://"):
        path = Path(url[len("file://") :])
    else:
        path = Path(url)
    if not path.exists():
        raise FileNotFoundError(f"Image path not found: {path}")
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}


def _normalize_content(content: Any) -> List[Dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        out: List[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                out.append({"type": "text", "text": str(block.get("text", ""))})
            elif btype == "image_url":
                image_url = block.get("image_url")
                if isinstance(image_url, dict):
                    url = str(image_url.get("url", ""))
                else:
                    url = str(image_url or "")
                if url:
                    out.append(_image_block_from_url(url))
        return out
    return [{"type": "text", "text": str(content)}]


def _collect_system(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    system_parts: List[str] = []
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = str(m.get("role", "user")).lower()
        if role == "system":
            for block in _normalize_content(m.get("content")):
                if block.get("type") == "text":
                    system_parts.append(str(block.get("text", "")))
            continue
        content = _normalize_content(m.get("content"))
        out.append({"role": "assistant" if role == "assistant" else "user", "content": content})
    system = "\n".join([t for t in system_parts if t.strip()]).strip()
    return system, out


class AnthropicChatAdapter:
    """Anthropic Messages API adapter (Claude)."""

    def __init__(self, cfg: Optional[AnthropicConfig] = None) -> None:
        self.cfg = cfg or AnthropicConfig()

    def _client(self):
        try:
            from anthropic import Anthropic  # type: ignore
        except Exception as e:
            raise RuntimeError("anthropic package not installed. Install: pip install anthropic") from e
        return Anthropic(api_key=self.cfg.api_key)

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        return self._request(messages=messages)

    def generate_with_schema(self, messages: List[Dict[str, Any]], guided_json: Dict[str, Any]) -> str:
        # Claude does not support guided JSON; fall back to standard generation.
        return self._request(messages=messages)

    def _request(self, *, messages: List[Dict[str, Any]]) -> str:
        client = self._client()
        system, conv = _collect_system(messages)
        try:
            resp = client.messages.create(
                model=self.cfg.model,
                system=system or None,
                messages=conv or [{"role": "user", "content": [{"type": "text", "text": ""}]}],
                max_tokens=int(self.cfg.max_tokens),
                temperature=float(self.cfg.temperature),
                top_p=float(self.cfg.top_p),
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic API request failed: {repr(e)}") from e

        try:
            parts = resp.content or []
            texts = [p.text for p in parts if getattr(p, "type", None) == "text"]
            return "\n".join(texts).strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected Anthropic response format: {repr(e)}") from e

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class OpenAIConfig:
    model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    api_key: str = os.environ.get("OPENAI_API_KEY", "")
    base_url: Optional[str] = os.environ.get("OPENAI_BASE_URL") or None
    timeout_s: int = 600
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.9
    # Reduce repetitive malformed tool-call outputs in reactive loops.
    presence_penalty: float = 1.10


class OpenAIChatAdapter:
    """OpenAI API adapter using official SDK (OpenAI-format messages)."""

    def __init__(self, cfg: Optional[OpenAIConfig] = None) -> None:
        self.cfg = cfg or OpenAIConfig()

    def _client(self):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("openai package not installed. Install: pip install openai") from e
        return OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url, timeout=self.cfg.timeout_s)

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        return self._request(messages=messages, guided_json=None)

    def generate_with_schema(self, messages: List[Dict[str, Any]], guided_json: Dict[str, Any]) -> str:
        return self._request(messages=messages, guided_json=guided_json)

    def _request(self, *, messages: List[Dict[str, Any]], guided_json: Optional[Dict[str, Any]]) -> str:
        client = self._client()
        model_l = str(self.cfg.model or "").strip().lower()
        is_gpt5_family = model_l.startswith("gpt-5")

        kwargs: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
        }
        if is_gpt5_family:
            # GPT-5/gpt-5-mini are reasoning models that consume internal thinking tokens
            # before producing visible output. max_tokens=2048 gets exhausted by reasoning alone,
            # leaving the content field empty. Use at least 16384 to ensure output is produced.
            kwargs["max_completion_tokens"] = max(int(self.cfg.max_tokens), 16384)
        else:
            kwargs["max_tokens"] = int(self.cfg.max_tokens)
            kwargs["temperature"] = float(self.cfg.temperature)
            kwargs["top_p"] = float(self.cfg.top_p)
            kwargs["presence_penalty"] = float(self.cfg.presence_penalty)
        if guided_json is not None:
            # Best-effort JSON forcing; OpenAI will ignore unsupported schemas.
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise RuntimeError(f"OpenAI API request failed: {repr(e)}") from e
        try:
            return str(resp.choices[0].message.content or "").strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected OpenAI response format: {repr(e)}") from e

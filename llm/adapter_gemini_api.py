from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from collections import deque
from io import BytesIO
from typing import Any, Dict, List, Optional
import time


@dataclass(frozen=True)
class GeminiConfig:
    model: str = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
    api_key: str = os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.9
    # For thinking/reasoning models (gemini-2.5-x), set thinking_budget tokens.
    # -1 = dynamic (model decides), 0 = disabled, >0 = fixed budget.
    thinking_budget: int = int(os.environ.get("GEMINI_THINKING_BUDGET", "-1"))


def _data_url_to_image(url: str):
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    if not url.startswith("data:"):
        return None
    header, b64 = url.split(",", 1)
    if ";base64" not in header:
        return None
    raw = base64.b64decode(b64)
    return Image.open(BytesIO(raw)).convert("RGB")


def _normalize_parts(content: Any) -> List[Any]:
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        out: List[Any] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                out.append(str(block.get("text", "")))
            elif btype == "image_url":
                image_url = block.get("image_url")
                if isinstance(image_url, dict):
                    url = str(image_url.get("url", ""))
                else:
                    url = str(image_url or "")
                if url.startswith("data:"):
                    img = _data_url_to_image(url)
                    if img is not None:
                        out.append(img)
        return out
    return [str(content)]


def _build_history(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    system_parts: List[str] = []
    history: List[Dict[str, Any]] = []
    for m in messages:
        role = str(m.get("role", "user")).lower()
        if role == "system":
            for part in _normalize_parts(m.get("content")):
                if isinstance(part, str) and part.strip():
                    system_parts.append(part)
            continue
        parts = _normalize_parts(m.get("content"))
        if role not in ("user", "assistant"):
            role = "user"
        role = "model" if role == "assistant" else "user"
        history.append({"role": role, "parts": parts})

    if system_parts:
        prefix = "\n".join(system_parts).strip()
        if history and history[0]["role"] == "user":
            history[0]["parts"] = [prefix] + history[0].get("parts", [])
        else:
            history.insert(0, {"role": "user", "parts": [prefix]})
    return history


class GeminiChatAdapter:
    """Google Gemini API adapter (google-generativeai)."""

    def __init__(self, cfg: Optional[GeminiConfig] = None) -> None:
        self.cfg = cfg or GeminiConfig()
        # Simple client-side throttling to avoid RPM errors.
        self._min_delay_s = float(os.environ.get("GEMINI_MIN_DELAY_S", "0") or 0)
        self._max_rpm = int(os.environ.get("GEMINI_MAX_RPM", "0") or 0)
        self._req_times = deque()  # timestamps (monotonic) of recent requests

    def _apply_rate_limit(self) -> None:
        now = time.monotonic()
        if self._min_delay_s > 0 and self._req_times:
            gap = now - self._req_times[-1]
            if gap < self._min_delay_s:
                time.sleep(self._min_delay_s - gap)
                now = time.monotonic()
        if self._max_rpm > 0:
            window = 60.0
            while self._req_times and (now - self._req_times[0]) > window:
                self._req_times.popleft()
            if len(self._req_times) >= self._max_rpm:
                sleep_for = window - (now - self._req_times[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
                    now = time.monotonic()
                    while self._req_times and (now - self._req_times[0]) > window:
                        self._req_times.popleft()
        self._req_times.append(time.monotonic())

    @staticmethod
    def _is_thinking_model(model: str) -> bool:
        """Return True for Gemini models that may consume hidden reasoning/thinking tokens."""
        m = model.strip().lower()
        # Gemini 2.5 models and explicit "-thinking" variants support/consume thinking tokens.
        # Gemini 3 preview models have also shown truncated visible JSON output at low budgets,
        # consistent with hidden reasoning token consumption, so treat them conservatively too.
        return (
            m.startswith("gemini-2.5")
            or m.startswith("gemini-3")
            or "-thinking" in m
        )

    def _model(self):
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:
            raise RuntimeError("google-generativeai package not installed. Install: pip install google-generativeai") from e
        genai.configure(api_key=self.cfg.api_key)
        is_thinking = self._is_thinking_model(self.cfg.model)
        # Thinking/reasoning models (gemini-2.5-x) consume thinking tokens before
        # the visible output, so max_output_tokens must be large enough to cover both.
        # Use at least 16384 for thinking models (analogous to gpt-5 max_completion_tokens).
        effective_max_tokens = max(int(self.cfg.max_tokens), 16384) if is_thinking else int(self.cfg.max_tokens)
        generation_config: Dict[str, Any] = {
            "max_output_tokens": effective_max_tokens,
        }
        if not is_thinking:
            # Sampling params are ignored by thinking models (temperature is fixed).
            generation_config["temperature"] = float(self.cfg.temperature)
            generation_config["top_p"] = float(self.cfg.top_p)

        if is_thinking and self.cfg.thinking_budget != 0:
            # thinking_config was added in newer SDK versions; check at runtime before using it.
            # GenerationConfig (v0.8.x) raises ValueError before GenerativeModel can catch it
            # when unknown fields are present in the dict, so we probe the dataclass directly.
            _gc_fields = set(getattr(genai.types.GenerationConfig, "__dataclass_fields__", {}).keys())
            if "thinking_config" in _gc_fields:
                generation_config["thinking_config"] = {"thinking_budget": self.cfg.thinking_budget}
        return genai.GenerativeModel(self.cfg.model, generation_config=generation_config)

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        return self._request(messages=messages)

    def generate_with_schema(self, messages: List[Dict[str, Any]], guided_json: Dict[str, Any]) -> str:
        # Gemini SDK does not support guided JSON; fall back to standard generation.
        return self._request(messages=messages)

    def _request(self, *, messages: List[Dict[str, Any]]) -> str:
        self._apply_rate_limit()
        model = self._model()
        history = _build_history(messages)
        if not history:
            resp = model.generate_content("")
            return str(getattr(resp, "text", "")).strip()
        if history[-1]["role"] != "user":
            history.append({"role": "user", "parts": ["Continue."]})
        last = history[-1]
        chat = model.start_chat(history=history[:-1])
        resp = chat.send_message(last.get("parts", []))
        return str(getattr(resp, "text", "")).strip()

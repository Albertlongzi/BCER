from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class VLLMServerConfig:
    """
    Minimal OpenAI-compatible vLLM chat/completions config.
    """

    base_url: str = "http://127.0.0.1:8000"
    model: str = os.environ.get("MEDGEMMA_SERVER_MODEL", "Qwen/Qwen3-VL-30B-A3B-Thinking")
    api_key: str = "EMPTY"
    timeout_s: int = 600

    # Higher default to avoid truncation in free-text reports.
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.9
    # Guardrails for runaway repetition in tool-calling loops.
    repetition_penalty: float = 1.05


def _normalize_server_base_url(base_url: str) -> str:
    """
    Accept either host root or common OpenAI paths:
    - http://host:port
    - http://host:port/v1
    - http://host:port/v1/chat/completions
    """
    url = str(base_url or "").strip()
    if not url:
        return "http://127.0.0.1:8000"
    url = url.rstrip("/")
    low = url.lower()
    for suffix in ("/v1/chat/completions", "/chat/completions", "/v1/completions", "/v1"):
        if low.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url.rstrip("/")


class VLLMOpenAIChatAdapter:
    """
    Calls an OpenAI-compatible vLLM server:
      POST {base_url}/v1/chat/completions
    """

    def __init__(self, cfg: Optional[VLLMServerConfig] = None) -> None:
        self.cfg = cfg or VLLMServerConfig()

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        return self._request(messages=messages, guided_json=None)

    def generate_with_schema(self, messages: List[Dict[str, Any]], guided_json: Dict[str, Any]) -> str:
        """
        Use vLLM guided JSON decoding (when supported by the server).
        """

        return self._request(messages=messages, guided_json=guided_json)

    def _request(self, *, messages: List[Dict[str, Any]], guided_json: Optional[Dict[str, Any]]) -> str:
        base_url = _normalize_server_base_url(self.cfg.base_url)
        url = base_url + "/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": int(self.cfg.max_tokens),
            "temperature": float(self.cfg.temperature),
            "top_p": float(self.cfg.top_p),
            "repetition_penalty": float(self.cfg.repetition_penalty),
        }

        # Best-effort JSON enforcement on OpenAI-compatible endpoint.
        # vLLM supports guided decoding via extra_body in many versions.
        if guided_json is not None:
            payload["response_format"] = {"type": "json_object"}
            payload["extra_body"] = {"guided_json": guided_json}

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
            method="POST",
        )

        def _do_request(cur_payload: Dict[str, Any]) -> str:
            cur_req = urllib.request.Request(
                url,
                data=json.dumps(cur_payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.cfg.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(cur_req, timeout=self.cfg.timeout_s) as resp:
                return resp.read().decode("utf-8")

        try:
            body = _do_request(payload)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
            # Friendly guardrail: if max_tokens is too large for the remaining context, shrink and retry once.
            # vLLM typically returns:
            #   "'max_tokens' ... too large: 4096. ... maximum context length is 32768 tokens and your request has 31715 input tokens (4096 > 32768 - 31715)."
            if int(getattr(e, "code", 0) or 0) == 400:
                m = re.search(r"maximum context length is\s+(\d+)\s+tokens.*?request has\s+(\d+)\s+input tokens", body, re.IGNORECASE)
                if m:
                    max_ctx = int(m.group(1))
                    in_tok = int(m.group(2))
                    # Do not force a high minimum (e.g., 64), because some requests may only have
                    # a tiny remaining budget (e.g., 8-32 tokens) near the context limit.
                    remaining = max(0, max_ctx - in_tok)
                    cur_max = int(payload.get("max_tokens") or 0)
                    new_max = min(cur_max, remaining)
                    if new_max >= 1 and new_max < cur_max:
                        payload2 = dict(payload)
                        payload2["max_tokens"] = int(new_max)
                        try:
                            body = _do_request(payload2)
                        except Exception:
                            # Fall back to the original error
                            raise RuntimeError(f"vLLM server HTTPError: {e.code} {e.reason} url={url}\n{body}") from e
                    elif remaining <= 0:
                        raise RuntimeError(
                            "vLLM context window exhausted before completion tokens. "
                            f"url={url} max_ctx={max_ctx} input_tokens={in_tok} "
                            "Please reduce prompt/context size."
                        ) from e
                    else:
                        raise RuntimeError(f"vLLM server HTTPError: {e.code} {e.reason} url={url}\n{body}") from e
                else:
                    raise RuntimeError(f"vLLM server HTTPError: {e.code} {e.reason} url={url}\n{body}") from e
            else:
                raise RuntimeError(f"vLLM server HTTPError: {e.code} {e.reason} url={url}\n{body}") from e
        except Exception as e:
            raise RuntimeError(f"vLLM server request failed: {repr(e)} (url={url})") from e

        try:
            obj = json.loads(body)
            return str(obj["choices"][0]["message"]["content"]).strip()
        except Exception as e:
            raise RuntimeError(f"Unexpected vLLM response format. Raw body:\n{body}") from e

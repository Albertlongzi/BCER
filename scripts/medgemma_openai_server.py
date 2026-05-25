#!/usr/bin/env python3
"""
OpenAI-compatible chat/completions server for MedGemma (Transformers).

Why:
- vLLM's OpenAI server may not support MedGemma VLM reliably.
- BCER already speaks OpenAI chat format; we keep that interface stable.

Supported:
- POST /v1/chat/completions  (subset of OpenAI schema)
- GET  /v1/models
- GET  /health

Multimodal:
- OpenAI-style content blocks: [{"type":"text","text":...},{"type":"image_url","image_url":{"url":"data:..."}}]
  We decode images and pass them to MedGemma via Transformers chat template API.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def _now_ts() -> int:
    return int(time.time())


def _read_data_url(url: str) -> bytes:
    # Expected: data:<mime>;base64,<payload>
    if not url.startswith("data:"):
        raise ValueError("Only data: URLs are supported for image_url (offline server).")
    try:
        header, b64 = url.split(",", 1)
    except ValueError as e:
        raise ValueError("Invalid data URL (missing comma).") from e
    if ";base64" not in header:
        raise ValueError("Only base64 data URLs are supported.")
    return base64.b64decode(b64)


def _load_image_from_url(url: str) -> Image.Image:
    # Offline-first: support data URLs and local file paths.
    if url.startswith("data:"):
        raw = _read_data_url(url)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    if url.startswith("file://"):
        path = url[len("file://") :]
        return Image.open(path).convert("RGB")
    if url.startswith("/") or url.startswith("./") or url.startswith("../"):
        return Image.open(url).convert("RGB")
    raise ValueError(f"Unsupported image_url url scheme: {url[:32]}...")


def _extract_openai_blocks(content: Any) -> List[Dict[str, Any]]:
    """
    Normalize OpenAI 'content' into a list of blocks.
    - If content is a string -> [{"type":"text","text":...}]
    - If content is a list -> return as-is (validated later)
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    raise TypeError(f"Unsupported message.content type: {type(content)}")


def _merge_system_into_first_user(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Some chat templates do not have a dedicated 'system' role.
    To be robust, we prepend system text into the first user text block.
    """
    system_texts: List[str] = []
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = str(m.get("role", "")).lower()
        if role == "system":
            blocks = _extract_openai_blocks(m.get("content"))
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "text":
                    system_texts.append(str(b.get("text", "")))
            continue
        out.append(m)

    if not system_texts:
        return out

    prefix = "\n".join([t for t in system_texts if t.strip()]).strip()
    if not prefix:
        return out

    # Find first user message and prefix its first text block.
    for m in out:
        if str(m.get("role", "")).lower() != "user":
            continue
        blocks = _extract_openai_blocks(m.get("content"))
        if not blocks:
            blocks = [{"type": "text", "text": prefix}]
        else:
            # Insert prefix as a leading text block to avoid modifying block structure.
            blocks = [{"type": "text", "text": prefix}] + blocks
        m["content"] = blocks
        break
    return out


def _normalize_roles_for_gemma(messages_tf: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    MedGemma/Gemma chat templates often require strict alternation:
      user/assistant/user/assistant/...

    OpenAI-compatible clients (like BCER) may legally send:
    - multiple consecutive user messages
    - tool/function roles
    - system roles

    This function:
    - maps any non-(user|assistant) role to 'user' by embedding it as text
    - collapses consecutive messages with the same role by concatenating their content blocks
    """
    norm: List[Dict[str, Any]] = []

    def _role(m: Dict[str, Any]) -> str:
        return str(m.get("role", "user")).lower()

    for m in messages_tf:
        role0 = _role(m)
        content0 = m.get("content")
        blocks = content0 if isinstance(content0, list) else [{"type": "text", "text": str(content0 or "")}]

        if role0 not in ("user", "assistant"):
            # Convert to user with a role tag so information isn't lost.
            blocks = [{"type": "text", "text": f"[{role0}]"}] + blocks
            role0 = "user"

        if not norm:
            norm.append({"role": role0, "content": list(blocks)})
            continue

        if norm[-1]["role"] == role0:
            # Merge consecutive same-role messages
            norm[-1]["content"] = list(norm[-1].get("content", [])) + [{"type": "text", "text": "\n"}] + list(blocks)
        else:
            norm.append({"role": role0, "content": list(blocks)})

    # Ensure conversation starts with user (some templates require it)
    if norm and norm[0]["role"] != "user":
        norm[0]["content"] = [{"type": "text", "text": "[assistant_preamble]"}] + list(norm[0].get("content", []))
        norm[0]["role"] = "user"

    if not norm:
        norm = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    return norm


def _to_transformers_messages(openai_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """
    Convert OpenAI messages (with image_url blocks) into Transformers-style messages for MedGemma:
      {"role":"user","content":[{"type":"image","image":PIL}, {"type":"text","text":...}]}

    Returns:
      - messages_tf: messages with "image" blocks containing PIL images
      - all_images: in-order list of PIL images used (debug/inspection)
    """
    msgs = _merge_system_into_first_user([dict(m) for m in openai_messages])
    all_images: List[Image.Image] = []
    out: List[Dict[str, Any]] = []

    for m in msgs:
        role = str(m.get("role", "user"))
        blocks = _extract_openai_blocks(m.get("content"))
        tf_blocks: List[Dict[str, Any]] = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                tf_blocks.append({"type": "text", "text": str(b.get("text", ""))})
            elif t == "image":
                # Non-OpenAI style, but we support it if upstream already passes PIL-less image blocks.
                raise ValueError("Unsupported block type 'image' without image payload. Use image_url.")
            elif t == "image_url":
                iu = b.get("image_url")
                if isinstance(iu, dict):
                    url = str(iu.get("url", ""))
                else:
                    url = str(iu or "")
                img = _load_image_from_url(url)
                all_images.append(img)
                tf_blocks.append({"type": "image", "image": img})
            else:
                # Ignore unknown block types to stay compatible with tool-call content etc.
                pass
        if not tf_blocks:
            # Ensure non-empty content for chat template stability
            tf_blocks = [{"type": "text", "text": ""}]
        out.append({"role": role, "content": tf_blocks})

    out = _normalize_roles_for_gemma(out)
    return out, all_images


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    # Remove ```json ... ``` or ``` ... ``` wrapper if present
    if s.startswith("```"):
        # find last fence
        end = s.rfind("```")
        if end > 0:
            inner = s.split("\n", 1)[1] if "\n" in s else ""
            inner = inner[: end - 3] if end >= 3 else inner
            return inner.strip()
    return s


class MedGemmaServer:
    def __init__(self, model_id: str, *, dtype: str = "bfloat16", device_map: str = "auto") -> None:
        self.model_id = model_id

        torch_dtype = torch.bfloat16 if dtype.lower() in ("bf16", "bfloat16") else torch.float16

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)

        # best-effort device for inputs (model may be sharded)
        try:
            self.input_device = next(self.model.parameters()).device
        except Exception:
            self.input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype

    def chat_completions(
        self,
        *,
        openai_messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> str:
        messages_tf, _images = _to_transformers_messages(openai_messages)

        inputs = self.processor.apply_chat_template(
            messages_tf,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.input_device, dtype=self.torch_dtype)
        input_len = int(inputs["input_ids"].shape[-1])

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
            )
        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return _strip_code_fences(decoded)


def build_app(server: MedGemmaServer) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "model": server.model_id}

    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        return {"object": "list", "data": [{"id": server.model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request) -> JSONResponse:
        payload = await req.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid JSON body.")

        model = payload.get("model") or server.model_id
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="Field 'messages' must be a list.")

        # OpenAI uses max_tokens; our internal naming is max_new_tokens.
        max_tokens = payload.get("max_tokens", 2048)
        temperature = payload.get("temperature", 0.0)
        top_p = payload.get("top_p", 0.9)
        # When temperature == 0, sample makes little sense; keep deterministic.
        do_sample = bool(payload.get("do_sample", False)) and float(temperature) > 0.0

        try:
            text = server.chat_completions(
                openai_messages=messages,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                do_sample=do_sample,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {repr(e)}") from e

        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": _now_ts(),
            "model": str(model),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }
        return JSONResponse(content=resp)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local snapshot dir.")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "bfloat16"), choices=["bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--device-map", default=os.environ.get("DEVICE_MAP", "auto"))
    args = parser.parse_args()

    print(f"[INFO] loading model: {args.model}")
    server = MedGemmaServer(args.model, dtype=str(args.dtype), device_map=str(args.device_map))
    app = build_app(server)

    import uvicorn

    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()


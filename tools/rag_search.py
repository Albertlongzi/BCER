from __future__ import annotations

import json
import os
import re
import heapq
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec
from core.paths import project_root
from llm.adapter_anthropic_api import AnthropicChatAdapter, AnthropicConfig
from llm.adapter_gemini_api import GeminiChatAdapter, GeminiConfig
from llm.adapter_openai_api import OpenAIChatAdapter, OpenAIConfig
from llm.adapter_vllm_server import VLLMOpenAIChatAdapter, VLLMServerConfig
from runtime.rag_local import search_repo


RAG_SPEC = ToolSpec(
    name="rag_search",
    description=(
        "Case-local QA over generated JSON/TXT artifacts (identify/ingest outputs). "
        "Answers targeted metadata questions such as DICOM header fields."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "User question to answer from local case artifacts."},
            "query": {"type": "string", "description": "Backward-compatible alias of question."},
            "case_state_path": {"type": "string", "description": "Case state JSON path."},
            "root": {"type": "string", "description": "Legacy fallback root for repo-wide search."},
            "max_results": {"type": "integer", "default": 16},
            "max_chars_per_file": {"type": "integer", "default": 4000},
            "output_subdir": {"type": "string", "default": "qa"},
            "llm_mode": {"type": "string", "description": "server|openai|anthropic|gemini"},
            "server_base_url": {"type": "string"},
            "server_model": {"type": "string"},
            "api_model": {"type": "string"},
            "api_base_url": {"type": "string"},
            "max_tokens": {"type": "integer", "default": 512},
            "temperature": {"type": "number", "default": 0.0},
        },
        "required": [],
    },
    output_schema={
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "final_answer": {"type": "string"},
            "evidence_files": {"type": "array"},
            "evidence_snippets": {"type": "array"},
            "results_path": {"type": "string"},
        },
        "required": ["question", "answer", "evidence_files", "evidence_snippets"],
    },
    version="0.2.0",
    tags=["rag", "qa", "metadata"],
)


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _looks_textual(path: Path) -> bool:
    return path.suffix.lower() in {".json", ".txt", ".md", ".cfg", ".yaml", ".yml"}


def _collect_artifacts_text_files(case_state_path: Path, *, limit: int = 256) -> List[Path]:
    out: List[Path] = []
    artifacts_dir = case_state_path.parent / "artifacts"
    if not artifacts_dir.exists():
        return out
    try:
        for p in artifacts_dir.rglob("*"):
            if len(out) >= limit:
                break
            if not p.is_file():
                continue
            if not _looks_textual(p):
                continue
            out.append(p.resolve())
    except Exception:
        return out
    return out


def _extract_referenced_text_paths(
    *,
    source_path: Path,
    text: str,
    case_state_path: Path,
    max_refs: int = 128,
) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    s = str(text or "")
    if not s:
        return out

    refs = re.findall(r"([A-Za-z0-9_./\\-]+\.(?:json|txt|cfg|md|yaml|yml))", s, flags=re.IGNORECASE)
    roots = [
        source_path.parent,
        case_state_path.parent,
        case_state_path.parent / "artifacts",
    ]

    for raw in refs:
        if len(out) >= max_refs:
            break
        cand_raw = str(raw or "").strip().strip("'\"")
        if not cand_raw:
            continue
        p = Path(cand_raw).expanduser()
        candidates: List[Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            for r in roots:
                candidates.append(r / p)
        for cand in candidates:
            try:
                cp = cand.resolve()
            except Exception:
                continue
            if not cp.exists() or not cp.is_file() or (not _looks_textual(cp)):
                continue
            key = str(cp)
            if key in seen:
                continue
            seen.add(key)
            out.append(cp)
            break
    return out


def _expand_textual_evidence_paths(
    *,
    seed_paths: List[Path],
    case_state_path: Path,
    max_paths: int = 512,
    max_depth: int = 3,
) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()

    queue: List[Tuple[Path, int]] = []
    for p in seed_paths:
        if len(queue) >= max_paths:
            break
        queue.append((p, 0))

    i = 0
    while i < len(queue) and len(out) < max_paths:
        cur, depth = queue[i]
        i += 1
        try:
            rp = cur.resolve()
        except Exception:
            continue
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        if not rp.exists() or (not rp.is_file()) or (not _looks_textual(rp)):
            continue
        out.append(rp)

        if depth >= max_depth:
            continue
        try:
            txt = rp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        refs = _extract_referenced_text_paths(source_path=rp, text=txt, case_state_path=case_state_path)
        for ref in refs:
            if len(queue) >= max_paths:
                break
            queue.append((ref, depth + 1))

    return out


def _collect_text_candidate_paths(case_state_path: Path) -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()

    def _push(path_like: Any) -> None:
        raw = str(path_like or "").strip()
        if not raw:
            return
        p = Path(raw).expanduser()
        try:
            if not p.exists() or (not p.is_file()):
                return
            if not _looks_textual(p):
                return
            key = str(p.resolve())
            if key in seen:
                return
            seen.add(key)
            out.append(p.resolve())
        except Exception:
            return

    state = _safe_read_json(case_state_path)
    for rec in state.get("artifacts_index") or []:
        if isinstance(rec, dict):
            _push(rec.get("path"))
    stage_outputs = state.get("stage_outputs") or {}
    if isinstance(stage_outputs, dict):
        for _, tools in stage_outputs.items():
            if not isinstance(tools, dict):
                continue
            for _, records in tools.items():
                if not isinstance(records, list):
                    continue
                for r in records:
                    if not isinstance(r, dict):
                        continue
                    data = r.get("data") or {}
                    if not isinstance(data, dict):
                        continue
                    for k, v in data.items():
                        if isinstance(v, str) and (k.endswith("_path") or v.lower().endswith((".json", ".txt", ".cfg", ".md"))):
                            _push(v)
    _push(case_state_path)
    for p in _collect_artifacts_text_files(case_state_path, limit=256):
        _push(p)
    return _expand_textual_evidence_paths(seed_paths=out, case_state_path=case_state_path)


def _tokenize(text: str) -> List[str]:
    toks = [t for t in re.split(r"[^a-z0-9]+", str(text or "").lower()) if t]
    stop = {"the", "and", "for", "from", "with", "what", "which", "show", "read", "only", "want", "image", "mri"}
    return [t for t in toks if t not in stop]


def _metadata_hint_tokens(question: str) -> List[str]:
    q = str(question or "").lower()
    hints: List[str] = []
    if any(x in q for x in ("te", "echo", "echo time")):
        hints.extend(["te", "echo", "echotime"])
    if any(x in q for x in ("tr", "repetition", "repetition time")):
        hints.extend(["tr", "repetition", "repetitiontime"])
    if "dicom" in q or "header" in q:
        hints.extend(["dicom", "header"])
    return sorted(set(hints))


def _snippets_from_file(path: Path, *, query_tokens: List[str], max_hits: int = 8) -> List[Dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []
    if not lines:
        return []

    if not query_tokens:
        out: List[Dict[str, Any]] = []
        for idx, line in enumerate(lines, start=1):
            if not str(line or "").strip():
                continue
            out.append({"path": str(path), "line_no": idx, "line": line.strip()[:300]})
            if len(out) >= max_hits:
                break
        return out

    ranked: List[Tuple[int, int, str]] = []
    for idx, line in enumerate(lines, start=1):
        low = line.lower()
        score = _line_match_score(low, query_tokens)
        if score <= 0:
            continue
        ranked.append((score, idx, str(line).strip()[:300]))

    ranked.sort(key=lambda x: (-int(x[0]), int(x[1])))
    return [{"path": str(path), "line_no": idx, "line": line} for score, idx, line in ranked[: max(1, int(max_hits))]]


def _line_matches_token(low_line: str, token: str) -> bool:
    tok = str(token or "").strip().lower()
    if not tok:
        return False
    if tok == "te":
        return (
            "echotime" in low_line
            or "echo time" in low_line
            or "(0018,0081)" in low_line
            or bool(re.search(r"(^|[^a-z0-9])te([^a-z0-9]|$)", low_line))
        )
    if tok == "tr":
        return (
            "repetitiontime" in low_line
            or "repetition time" in low_line
            or "(0018,0080)" in low_line
            or bool(re.search(r"(^|[^a-z0-9])tr([^a-z0-9]|$)", low_line))
        )
    if len(tok) <= 2:
        return bool(re.search(rf"(^|[^a-z0-9]){re.escape(tok)}([^a-z0-9]|$)", low_line))
    return tok in low_line


def _line_match_score(low_line: str, query_tokens: List[str]) -> int:
    score = 0
    for token in query_tokens:
        tok = str(token or "").strip().lower()
        if not tok or (not _line_matches_token(low_line, tok)):
            continue
        if tok in {"echotime", "repetitiontime"}:
            score += 120
        elif tok in {"echo", "repetition"}:
            score += 60
        elif tok in {"te", "tr"}:
            score += 45
        elif tok in {"header", "dicom"}:
            score += 12
        else:
            score += max(4, min(24, len(tok) * 2))

    # Boost explicit DICOM tag lines for TE/TR.
    if "(0018,0081)" in low_line or "echotime" in low_line:
        score += 80
    if "(0018,0080)" in low_line or "repetitiontime" in low_line:
        score += 80
    return score


def _snippets_from_large_file(
    path: Path,
    *,
    query_tokens: List[str],
    max_hits: int = 8,
    max_lines: int = 120000,
) -> List[Dict[str, Any]]:
    if not query_tokens:
        out: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for idx, line in enumerate(f, start=1):
                    if idx > max_lines:
                        break
                    if not str(line or "").strip():
                        continue
                    out.append({"path": str(path), "line_no": idx, "line": str(line).strip()[:300]})
                    if len(out) >= max_hits:
                        break
        except Exception:
            return []
        return out

    # Keep only highest-scoring matching lines from large files.
    heap: List[Tuple[int, int, str]] = []
    keep_n = max(int(max_hits) * 8, int(max_hits))
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, start=1):
                if idx > max_lines:
                    break
                low = line.lower()
                score = _line_match_score(low, query_tokens)
                if score <= 0:
                    continue
                cand = (int(score), -int(idx), str(line).strip()[:300])
                if len(heap) < keep_n:
                    heapq.heappush(heap, cand)
                    continue
                if cand > heap[0]:
                    heapq.heapreplace(heap, cand)
    except Exception:
        return []

    ranked = sorted(heap, key=lambda x: (-int(x[0]), -int(x[1])))
    out = [{"path": str(path), "line_no": int(-neg_idx), "line": line} for score, neg_idx, line in ranked[: max(1, int(max_hits))]]
    return out


def _evidence_path_priority(path: Path) -> int:
    s = str(path).lower()
    n = path.name.lower()
    score = 0
    if "header" in n:
        score += 100
    if "dicom_txt" in s:
        score += 80
    if "dicom_headers_index" in n:
        score += 50
    if "dicom_meta" in n:
        score += 20
    if "case_state" in n:
        score -= 10
    return score


def _build_evidence(question: str, paths: List[Path], *, max_results: int, max_chars_per_file: int) -> List[Dict[str, Any]]:
    tokens = sorted(set(_tokenize(question) + _metadata_hint_tokens(question)))
    snippets: List[Dict[str, Any]] = []
    size_skip_limit = max(512, int(max_chars_per_file) * 3)
    large_stream_limit = max(20000, int(max_chars_per_file) * 32)
    ordered_paths = sorted(paths, key=lambda p: (_evidence_path_priority(p), str(p)), reverse=True)
    for p in ordered_paths:
        if len(snippets) >= max_results:
            break
        try:
            fsize = int(p.stat().st_size)
            if fsize > large_stream_limit:
                continue
        except Exception:
            continue
        if fsize > size_skip_limit:
            hits = _snippets_from_large_file(p, query_tokens=tokens, max_hits=6)
        else:
            hits = _snippets_from_file(p, query_tokens=tokens, max_hits=6)
        for h in hits:
            snippets.append(h)
            if len(snippets) >= max_results:
                break
    return snippets


def _build_llm_adapter(args: Dict[str, Any]) -> Optional[Any]:
    mode = str(args.get("llm_mode") or "").strip().lower()
    max_tokens = int(args.get("max_tokens") or 512)
    temperature = float(args.get("temperature") or 0.0)
    try:
        if mode == "server":
            base_url = str(args.get("server_base_url") or "http://127.0.0.1:8000/v1").strip()
            model = str(args.get("server_model") or os.environ.get("MEDGEMMA_SERVER_MODEL") or "Qwen/Qwen3-VL-30B-A3B-Thinking")
            return VLLMOpenAIChatAdapter(
                VLLMServerConfig(base_url=base_url, model=model, max_tokens=max_tokens, temperature=temperature)
            )
        if mode == "openai":
            model = str(args.get("api_model") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini")
            base = str(args.get("api_base_url") or os.environ.get("OPENAI_BASE_URL") or "").strip() or None
            return OpenAIChatAdapter(OpenAIConfig(model=model, base_url=base, max_tokens=max_tokens, temperature=temperature))
        if mode == "anthropic":
            model = str(args.get("api_model") or os.environ.get("ANTHROPIC_MODEL") or "claude-3-7-sonnet-20250219")
            return AnthropicChatAdapter(AnthropicConfig(model=model, max_tokens=max_tokens, temperature=temperature))
        if mode == "gemini":
            model = str(args.get("api_model") or os.environ.get("GEMINI_MODEL") or "gemini-1.5-pro")
            return GeminiChatAdapter(GeminiConfig(model=model, max_tokens=max_tokens, temperature=temperature))
    except Exception:
        return None
    return None


def _llm_answer(question: str, evidence: List[Dict[str, Any]], *, adapter: Any) -> str:
    evidence_txt = "\n".join(
        [f"[{Path(str(x.get('path') or '')).name}:{x.get('line_no')}] {str(x.get('line') or '')}" for x in evidence[:24]]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You answer MRI metadata questions using only provided evidence lines. "
                "If uncertain, say uncertain and what is missing.\n\n"
                "You MUST format your output exactly like this:\n"
                "<thought_process>\n"
                "Your step-by-step reasoning here\n"
                "</thought_process>\n"
                "<final_answer>\n"
                "Your concise final answer here\n"
                "</final_answer>"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Evidence:\n{evidence_txt}\n\n"
                "Return a concise answer and cite evidence as [file:line]. "
                "Follow the required XML tag format exactly."
            ),
        },
    ]
    raw = str(adapter.generate(messages)).strip()
    return _ensure_structured_answer(
        raw,
        default_thought="I extracted relevant evidence lines and synthesized a concise answer.",
    )


def _heuristic_answer(question: str, evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return _format_structured_answer(
            thought="No matching local evidence lines were found for the query tokens.",
            final="I could not find matching local metadata lines for this question.",
        )
    lines = [f"[{Path(str(x.get('path') or '')).name}:{x.get('line_no')}] {str(x.get('line') or '')}" for x in evidence[:6]]
    return _format_structured_answer(
        thought="I selected the highest-scoring local metadata lines relevant to the question.",
        final="I found these relevant metadata lines:\n" + "\n".join(lines),
    )


def _extract_tag_block(text: str, tag: str) -> str:
    s = str(text or "")
    m = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", s, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return ""
    return str(m.group(1) or "").strip()


def _format_structured_answer(*, thought: str, final: str) -> str:
    return (
        "<thought_process>\n"
        f"{str(thought or '').strip()}\n"
        "</thought_process>\n"
        "<final_answer>\n"
        f"{str(final or '').strip()}\n"
        "</final_answer>"
    )


def _ensure_structured_answer(text: str, *, default_thought: str) -> str:
    thought = _extract_tag_block(text, "thought_process")
    final = _extract_tag_block(text, "final_answer")
    if thought and final:
        return _format_structured_answer(thought=thought, final=final)
    body = str(text or "").strip()
    if not body:
        body = "I am uncertain based on the available evidence."
    return _format_structured_answer(thought=default_thought, final=body)


def _resolve_root(root_arg: Optional[str]) -> Path:
    if root_arg:
        return Path(root_arg).expanduser().resolve()
    return project_root()


def rag_search_tool(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    question = str(args.get("question") or args.get("query") or "").strip()
    if not question:
        raise ValueError("rag_search requires non-empty question or query")

    case_state_path = Path(str(args.get("case_state_path") or ctx.case_state_path)).expanduser()
    max_results = int(args.get("max_results") or 16)
    max_chars_per_file = int(args.get("max_chars_per_file") or 4000)
    out_subdir = str(args.get("output_subdir") or "qa")

    warnings: List[str] = []
    evidence_files = _collect_text_candidate_paths(case_state_path) if case_state_path.exists() else []
    evidence = _build_evidence(
        question,
        evidence_files,
        max_results=max_results,
        max_chars_per_file=max_chars_per_file,
    )

    # Legacy fallback: repo search if case-local evidence is empty.
    if not evidence:
        root = _resolve_root(args.get("root"))
        legacy_hits = search_repo(question, root=root, max_results=max_results, exts={".py", ".md", ".json", ".txt", ".cfg"})
        for h in legacy_hits:
            evidence.append(
                {
                    "path": str(h.get("path") or ""),
                    "line_no": int(h.get("line_no") or 0),
                    "line": str(h.get("line") or ""),
                }
            )
        warnings.append("No strong case-local evidence; used legacy repository search fallback.")

    adapter = _build_llm_adapter(args)
    if adapter is not None and evidence:
        try:
            answer = _llm_answer(question, evidence, adapter=adapter)
        except Exception:
            answer = _heuristic_answer(question, evidence)
            warnings.append("LLM answering failed; returned heuristic answer.")
    else:
        answer = _heuristic_answer(question, evidence)
        if adapter is None:
            warnings.append("No LLM adapter configured for rag_search; returned heuristic answer.")

    out_dir = ctx.artifacts_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "rag_answer.json"
    payload = {
        "question": question,
        "answer": answer,
        "evidence_files": [str(p) for p in evidence_files],
        "evidence_snippets": evidence,
    }
    results_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    artifacts = [
        ArtifactRef(
            path=str(results_path),
            kind="json",
            description="RAG QA answer and evidence",
            media_type="application/json",
        )
    ]
    data = {
        "question": question,
        "answer": answer,
        "final_answer": answer,
        "evidence_files": [str(p) for p in evidence_files],
        "evidence_snippets": evidence,
        "results_path": str(results_path),
    }
    return {
        "data": data,
        "artifacts": artifacts,
        "generated_artifacts": artifacts,
        "warnings": warnings,
    }


def build_tool() -> Tool:
    return Tool(spec=RAG_SPEC, func=rag_search_tool)

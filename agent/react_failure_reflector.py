from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mri_agent_shell.runtime.patch_spec import (
    PatchSpec,
    STRUCTURED_REFLECTOR_SYSTEM_PROMPT,
    StructuredRepairContext,
    build_reflector_user_prompt,
    build_structured_repair_context,
)


_UNRESOLVED_REF_RE = re.compile(r"unresolved reference in path argument '([^']+)'[^:]*: (\S+)")
_EMBEDDED_JSON_RE = re.compile(r"Evidence:\s*(\{.*\})\s*$", re.DOTALL)
_PATHLIKE_EXTS = (
    ".nii",
    ".nii.gz",
    ".h5",
    ".json",
    ".csv",
    ".txt",
    ".png",
    ".tfm",
    ".pt",
)


def _clip_text(text: Any, *, max_chars: int = 320) -> str:
    s = str(text or "")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 12].rstrip() + " [TRUNCATED]"


def _json_sig(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return repr(obj)


def _dedupe_keep_last(items: List[Dict[str, Any]], *, key_fn) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out_rev: List[Dict[str, Any]] = []
    for item in reversed(items):
        sig = str(key_fn(item))
        if sig in seen:
            continue
        seen.add(sig)
        out_rev.append(item)
    return list(reversed(out_rev))


@dataclass
class ReActFailureReflector:
    """ReAct-side adapter for the BCER structured failure reflector core."""

    llm: Any
    llm_mode: str
    registry: Any
    recent_tool_window: int = 8
    recent_reflection_window: int = 6
    token_preview_limit: int = 8
    _recent_reflections: List[Dict[str, Any]] = field(default_factory=list)

    def _load_sequence_mapping(self, case_state_path: Path) -> Dict[str, str]:
        try:
            state = json.loads(case_state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        stage_outputs = state.get("stage_outputs", {})
        if not isinstance(stage_outputs, dict):
            return {}

        latest_mapping: Dict[str, str] = {}
        for _stage, tools in stage_outputs.items():
            if not isinstance(tools, dict):
                continue
            recs = tools.get("identify_sequences")
            if not isinstance(recs, list) or not recs:
                continue
            rec = recs[-1] if isinstance(recs[-1], dict) else {}
            data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
            mapping = data.get("mapping") if isinstance(data.get("mapping"), dict) else {}
            for k, v in mapping.items():
                ks = str(k or "").strip()
                vs = str(v or "").strip()
                if ks and vs:
                    latest_mapping[ks] = vs
        return latest_mapping

    def _build_binding_context(self, *, case_state_path: Path, dicom_case_dir: Optional[str]) -> Dict[str, Any]:
        seq_mapping = self._load_sequence_mapping(case_state_path)
        token_bindings: Dict[str, str] = {
            "@runtime.case_state_path": str(case_state_path),
            "@state.path": str(case_state_path),
            "state.path": str(case_state_path),
        }
        case_in = str(dicom_case_dir or "").strip()
        if case_in:
            token_bindings["@case.input"] = case_in
            token_bindings["@case.path"] = case_in
            token_bindings["case.path"] = case_in
            token_bindings["case.input"] = case_in
        for k, v in seq_mapping.items():
            if str(k).strip() and str(v).strip():
                token_bindings[f"@seq.{str(k).strip()}"] = str(v).strip()
        return {
            "available_valid_tokens": sorted(token_bindings.keys()),
            "token_bindings": token_bindings,
            "session_token_bindings": {},
            "available_modalities": sorted([str(k) for k in seq_mapping.keys() if str(k).strip()]),
        }

    def _classify_limit(self, *, err_type: str, err_msg: str, deterministic_retry: Dict[str, Any]) -> str:
        et = str(err_type or "").strip().lower()
        msg = str(err_msg or "").strip().lower()
        if deterministic_retry:
            return "fixable_runtime"
        if et == "schemavalidationerror":
            if any(sig in msg for sig in ("missing required argument", "missing required", "required property")):
                return "fixable_runtime"
            if "not allowed for domain" in msg:
                return "hard_limit_schema"
            return "unknown_runtime"
        if "not allowed for domain" in msg:
            return "hard_limit_schema"
        if et == "scopeviolation":
            if "unresolved reference in path argument" in msg:
                return "fixable_runtime"
            return "hard_limit_scope"
        return "unknown_runtime"

    def _deterministic_retry_suggestion(
        self,
        *,
        tool_name: str,
        last_args: Dict[str, Any],
        error: Dict[str, Any],
        binding_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        out = dict(last_args or {})
        changed = False

        aliases = {
            "@state.path": "@runtime.case_state_path",
            "state.path": "@runtime.case_state_path",
            "@case.path": "@case.input",
            "case.path": "@case.input",
            "@case.root": "@case.input",
            "case.root": "@case.input",
        }

        def _map_aliases(obj: Any) -> Any:
            nonlocal changed
            if isinstance(obj, dict):
                return {k: _map_aliases(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_map_aliases(v) for v in obj]
            if isinstance(obj, str) and obj in aliases:
                changed = True
                return aliases[obj]
            return obj

        out = _map_aliases(out)

        err_msg = str((error or {}).get("message") or "")
        m = _UNRESOLVED_REF_RE.search(err_msg)
        unresolved_key = str(m.group(1) or "").strip().lower() if m else ""
        unresolved_ref = str(m.group(2) or "").strip() if m else ""

        valid_tokens = set(
            str(x or "").strip()
            for x in (binding_context.get("available_valid_tokens") or [])
            if str(x or "").strip()
        )

        def _can_use(tok: str) -> bool:
            return tok in valid_tokens or tok in {"@runtime.case_state_path", "@case.input"}

        def _strip_seq_prefix(tok: str) -> str:
            t = str(tok or "").strip()
            if t.startswith("@seq."):
                return t[len("@seq.") :].strip()
            if t.lower().startswith("seq."):
                return t[4:].strip()
            return ""

        if unresolved_key in {"case_state_path", "state_path"} and _can_use("@runtime.case_state_path"):
            out[unresolved_key] = "@runtime.case_state_path"
            changed = True
        elif unresolved_key in {"case_path", "dicom_case_dir"} and _can_use("@case.input"):
            out[unresolved_key] = "@case.input"
            changed = True
        elif unresolved_ref in {"@state.path", "state.path"} and _can_use("@runtime.case_state_path"):
            if unresolved_key:
                out[unresolved_key] = "@runtime.case_state_path"
                changed = True
        else:
            seq_fallback = _strip_seq_prefix(unresolved_ref)
            if seq_fallback:
                if unresolved_key:
                    out[unresolved_key] = seq_fallback
                    changed = True
                else:
                    for k, v in list(out.items()):
                        if str(v or "").strip() == unresolved_ref:
                            out[k] = seq_fallback
                            changed = True

        if str(tool_name or "").strip() == "rag_search":
            raw = str(out.get("case_state_path") or "").strip()
            if (not raw or raw in {"@state.path", "state.path"}) and _can_use("@runtime.case_state_path"):
                out["case_state_path"] = "@runtime.case_state_path"
                changed = True

        return out if changed else {}

    def _history_window(self, step_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        trimmed = [h for h in (step_history or []) if isinstance(h, dict)][-max(1, self.recent_tool_window) :]
        return _dedupe_keep_last(
            trimmed,
            key_fn=lambda h: _json_sig(
                {
                    "tool_name": h.get("tool_name"),
                    "status": h.get("status"),
                    "error_type": ((h.get("error") or {}) if isinstance(h.get("error"), dict) else {}).get("type"),
                    "error_message": ((h.get("error") or {}) if isinstance(h.get("error"), dict) else {}).get("message"),
                }
            ),
        )

    def _history_for_repair_ctx(self, step_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, h in enumerate(self._history_window(step_history)):
            rec = {
                "node_id": str(h.get("node_id") or f"react_hist_{idx:03d}"),
                "tool_name": str(h.get("tool_name") or ""),
                "status": str(h.get("status") or ""),
                "data": dict(h.get("data") or {}) if isinstance(h.get("data"), dict) else {},
            }
            out.append(rec)
        return out

    def _is_path_like_key(self, key: str) -> bool:
        k = str(key or "").strip().lower()
        if not k:
            return False
        if k in {"fixed", "moving", "cine_path", "seg_path", "input_nifti", "reference_nifti", "feature_table_path"}:
            return True
        return any(tok in k for tok in ("path", "file", "dir", "nifti", "mask", "csv", "json", "bundle", "weights"))

    def _compact_value_preview(self, value: Any, *, max_chars: int = 180) -> Any:
        if isinstance(value, str):
            return _clip_text(value, max_chars=max_chars)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            out: List[Any] = []
            for item in value[:3]:
                out.append(self._compact_value_preview(item, max_chars=max_chars))
            return out
        if isinstance(value, dict):
            out: Dict[str, Any] = {}
            for i, (k, v) in enumerate(value.items()):
                if i >= 4:
                    break
                out[str(k)] = self._compact_value_preview(v, max_chars=max_chars)
            return out
        return _clip_text(repr(value), max_chars=max_chars)

    def _looks_like_path_value(self, value: Any) -> bool:
        s = str(value or "").strip()
        if not s or s.startswith("@"):
            return False
        if "/" in s or s.startswith("."):
            return True
        sl = s.lower()
        return any(sl.endswith(ext) for ext in _PATHLIKE_EXTS)

    def _path_candidate_strings(self, raw_value: str, *, run_dir: Optional[Path]) -> List[str]:
        s = str(raw_value or "").strip()
        if not s:
            return []
        candidates: List[Path] = []
        p = Path(s)
        if p.is_absolute():
            candidates.append(p)
        elif run_dir is not None:
            candidates.append((run_dir / s))
            candidates.append((run_dir / "artifacts" / s))
        # Keep exact relative value for transparency even when unresolved.
        out: List[str] = [s]
        seen = {s}
        for c in candidates:
            try:
                cs = str(c.resolve())
            except Exception:
                cs = str(c)
            if cs not in seen:
                out.append(cs)
                seen.add(cs)
        return out[:4]

    def _extract_upstream_value_context(
        self,
        step_history: List[Dict[str, Any]],
        *,
        run_dir: Optional[Path],
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for h in self._history_window(step_history):
            if str(h.get("status") or "").upper() != "DONE":
                continue
            node_id = str(h.get("node_id") or "")
            if not node_id:
                continue
            tool_name = str(h.get("tool_name") or "")
            data = h.get("data") if isinstance(h.get("data"), dict) else {}
            args = h.get("arguments") if isinstance(h.get("arguments"), dict) else {}

            data_preview: Dict[str, Any] = {}
            data_path_candidates: Dict[str, List[str]] = {}
            for k, v in data.items():
                ks = str(k or "")
                if self._is_path_like_key(ks):
                    data_preview[ks] = self._compact_value_preview(v)
                    if isinstance(v, str) and self._looks_like_path_value(v):
                        cands = self._path_candidate_strings(v, run_dir=run_dir)
                        if cands:
                            data_path_candidates[ks] = cands
            if not data_preview:
                for k, v in data.items():
                    ks = str(k or "")
                    if isinstance(v, (str, int, float, bool)) and ks in {
                        "predicted_group",
                        "series_inventory_path",
                        "mapping",
                        "reconstructed_nifti",
                    }:
                        data_preview[ks] = self._compact_value_preview(v)

            args_preview: Dict[str, Any] = {}
            args_path_candidates: Dict[str, List[str]] = {}
            for k, v in args.items():
                ks = str(k or "")
                if self._is_path_like_key(ks):
                    args_preview[ks] = self._compact_value_preview(v)
                    if isinstance(v, str) and self._looks_like_path_value(v):
                        cands = self._path_candidate_strings(v, run_dir=run_dir)
                        if cands:
                            args_path_candidates[ks] = cands

            out[node_id] = {
                "tool_name": tool_name,
                "data_preview": data_preview,
                "arguments_preview": args_preview,
            }
            if data_path_candidates:
                out[node_id]["data_path_candidates"] = data_path_candidates
            if args_path_candidates:
                out[node_id]["arguments_path_candidates"] = args_path_candidates
        return out

    def _hard_limit_verdict(self, *, tool_name: str, err_type: str, err_msg: str, limit_class: str) -> Dict[str, Any]:
        if limit_class == "hard_limit_schema":
            nl = (
                f"Hard boundary for '{tool_name}': {err_type}: {err_msg}. "
                "Tool/domain schema policy blocks this call. Stop and adjust code/config, not prompts."
            )
        elif limit_class == "hard_limit_scope":
            nl = (
                f"Hard boundary for '{tool_name}': {err_type}: {err_msg}. "
                "Case scope guard blocked the path. Stop and adjust runtime configuration or inputs."
            )
        else:
            nl = f"Required tool '{tool_name}' failed: {err_msg}"
        return {
            "action": "halt",
            "reason": f"{limit_class}: {err_type}: {err_msg}",
            "natural_language_response": nl,
            "retry_arguments": {},
        }

    def _llm_reflect(
        self,
        *,
        repair_ctx: StructuredRepairContext,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if self.llm is None or self.llm_mode not in {"server", "openai", "anthropic", "gemini"}:
            return None, ""
        allowed_arg_keys = None
        if repair_ctx.tool_schema is not None:
            props = repair_ctx.tool_schema.input_schema.get("properties", {})
            if isinstance(props, dict):
                allowed_arg_keys = set(props.keys())
        msgs = [
            {"role": "system", "content": STRUCTURED_REFLECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": build_reflector_user_prompt(repair_ctx)},
        ]
        raw = ""
        try:
            raw = str(self.llm.generate(msgs)).strip()
            patch = PatchSpec.parse_and_validate(
                raw,
                allowed_arg_keys=allowed_arg_keys,
                allowed_actions=("retry", "skip", "halt"),
            )
            return patch.to_legacy_verdict(), raw
        except Exception:
            return None, raw

    def _extract_semantic_evidence(self, err: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        direct = err.get("semantic_lint_evidence")
        if isinstance(direct, dict):
            return direct
        msg = str(err.get("message") or "")
        m = _EMBEDDED_JSON_RE.search(msg)
        if not m:
            return None
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _build_feedback(
        self,
        *,
        step: int,
        tool_name: str,
        stage: str,
        err: Dict[str, Any],
        limit_class: str,
        verdict: Dict[str, Any],
        repair_ctx: Optional[StructuredRepairContext],
        deterministic_retry: Dict[str, Any],
        step_history: List[Dict[str, Any]],
        schema_autofix_applied: Optional[Dict[str, Any]],
        preconditions_applied: Optional[List[Dict[str, Any]]],
        rule_violations: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        semantic_evidence = self._extract_semantic_evidence(err)
        candidates: List[Dict[str, Any]] = []
        if deterministic_retry:
            candidates.append(
                {
                    "kind": "deterministic_retry",
                    "corrected_args": dict(deterministic_retry),
                    "reason": "token/path alias or unresolved reference fix",
                }
            )
        if semantic_evidence and isinstance(semantic_evidence.get("candidate_repairs"), list):
            for c in semantic_evidence.get("candidate_repairs") or []:
                if isinstance(c, dict):
                    candidates.append(dict(c))
        if schema_autofix_applied:
            candidates.append(
                {
                    "kind": "schema_autofix_attempt",
                    "missing_keys": list(schema_autofix_applied.get("missing_keys") or []),
                    "patched_keys": list(schema_autofix_applied.get("patched_keys") or []),
                    "retry_ok": bool(schema_autofix_applied.get("retry_ok")),
                }
            )
        candidates = _dedupe_keep_last(
            [c for c in candidates if isinstance(c, dict)],
            key_fn=lambda x: _json_sig(x),
        )

        token_preview: Dict[str, str] = {}
        if repair_ctx is not None:
            items = list((repair_ctx.available_tokens or {}).items())
            items = sorted(items, key=lambda kv: (0 if str(kv[0]).startswith("@runtime.") else 1, str(kv[0])))
            for k, v in items[: max(1, self.token_preview_limit)]:
                token_preview[str(k)] = _clip_text(v, max_chars=120)

        recent_tools: List[Dict[str, Any]] = []
        for h in self._history_window(step_history):
            err_h = h.get("error") if isinstance(h.get("error"), dict) else {}
            recent_tools.append(
                {
                    "tool_name": h.get("tool_name"),
                    "status": h.get("status"),
                    "error_type": err_h.get("type"),
                    "error_message": _clip_text(err_h.get("message"), max_chars=120) if err_h else "",
                }
            )

        evidence_blocks: List[Dict[str, Any]] = []
        if semantic_evidence:
            evidence_blocks.append(
                {
                    "kind": "semantic_lint_evidence",
                    "evidence_type": semantic_evidence.get("evidence_type"),
                    "pair": semantic_evidence.get("pair"),
                    "detector_basis": semantic_evidence.get("detector_basis"),
                }
            )
        if rule_violations:
            evidence_blocks.append(
                {
                    "kind": "rule_violations",
                    "count": len(rule_violations),
                    "items": [
                        {
                            "tool_name": r.get("tool_name"),
                            "rule_id": r.get("rule_id"),
                            "level": r.get("level"),
                            "message": _clip_text(r.get("message"), max_chars=140),
                        }
                        for r in (rule_violations or [])[:3]
                        if isinstance(r, dict)
                    ],
                }
            )
        if preconditions_applied:
            evidence_blocks.append(
                {
                    "kind": "preconditions_applied",
                    "count": len(preconditions_applied),
                    "items": [dict(x) for x in (preconditions_applied or [])[:3] if isinstance(x, dict)],
                }
            )
        if schema_autofix_applied:
            evidence_blocks.append(
                {
                    "kind": "schema_autofix",
                    "trigger_error_type": schema_autofix_applied.get("trigger_error_type"),
                    "trigger_error_message": _clip_text(schema_autofix_applied.get("trigger_error_message"), max_chars=140),
                    "missing_keys": list(schema_autofix_applied.get("missing_keys") or []),
                    "retry_ok": bool(schema_autofix_applied.get("retry_ok")),
                }
            )

        current = {
            "step": int(step),
            "tool_name": str(tool_name),
            "error_type": str(err.get("type") or ""),
            "action": str(verdict.get("action") or ""),
            "reason": _clip_text(verdict.get("reason"), max_chars=180),
        }
        self._recent_reflections.append(current)
        self._recent_reflections = _dedupe_keep_last(
            self._recent_reflections[- max(1, self.recent_reflection_window * 2) :],
            key_fn=lambda x: _json_sig(x),
        )[- max(1, self.recent_reflection_window) :]

        return {
            "failure_reflection": {
                "trigger": {"step": int(step), "tool_name": str(tool_name), "stage": str(stage or "")},
                "decision": {
                    "action": str(verdict.get("action") or "skip"),
                    "reason": _clip_text(verdict.get("reason"), max_chars=220),
                    "natural_language_response": _clip_text(verdict.get("natural_language_response"), max_chars=220),
                },
                "error": {
                    "type": str(err.get("type") or ""),
                    "message": _clip_text(err.get("message"), max_chars=320),
                    "classification": str(limit_class or "unknown_runtime"),
                },
                "candidate_repair_space": candidates[:4],
                "evidence": evidence_blocks[:4],
                "upstream_context": {
                    "available_modalities": list((repair_ctx.available_modalities if repair_ctx else []) or [])[:8],
                    "available_tokens_preview": token_preview,
                    "recent_tool_window": recent_tools[- max(1, self.recent_tool_window) :],
                },
                "recent_failure_window": list(self._recent_reflections),
                "note": "action=skip means no automatic retry; choose a different tool call or arguments next.",
            }
        }

    def reflect(
        self,
        *,
        step: int,
        tool_name: str,
        stage: str,
        executed_args: Dict[str, Any],
        tool_result: Dict[str, Any],
        case_state_path: Path,
        dicom_case_dir: Optional[str],
        step_history: List[Dict[str, Any]],
        schema_autofix_applied: Optional[Dict[str, Any]] = None,
        preconditions_applied: Optional[List[Dict[str, Any]]] = None,
        rule_violations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        err = tool_result.get("error") if isinstance(tool_result.get("error"), dict) else {}
        err_type = str(err.get("type") or "RuntimeError")
        err_msg = str(err.get("message") or "tool failed")

        binding_context = self._build_binding_context(
            case_state_path=case_state_path,
            dicom_case_dir=dicom_case_dir,
        )
        deterministic_retry = self._deterministic_retry_suggestion(
            tool_name=tool_name,
            last_args=dict(executed_args or {}),
            error=err,
            binding_context=binding_context,
        )
        limit_class = self._classify_limit(
            err_type=err_type,
            err_msg=err_msg,
            deterministic_retry=deterministic_retry,
        )

        repair_ctx: Optional[StructuredRepairContext] = None
        try:
            repair_ctx = build_structured_repair_context(
                tool_name=str(tool_name),
                last_arguments=dict(executed_args or {}),
                error=err,
                failure_classification=limit_class,
                binding_context=binding_context,
                deterministic_retry=(deterministic_retry if deterministic_retry else None),
                registry=self.registry,
                step_results=self._history_for_repair_ctx(step_history),
            )
            upstream_value_ctx = self._extract_upstream_value_context(
                step_history,
                run_dir=case_state_path.parent if isinstance(case_state_path, Path) else None,
            )
            if upstream_value_ctx:
                for node_id, rec in upstream_value_ctx.items():
                    if node_id not in repair_ctx.last_artifacts:
                        repair_ctx.last_artifacts[node_id] = {
                            "tool_name": rec.get("tool_name"),
                            "data_keys": [],
                        }
                    if isinstance(repair_ctx.last_artifacts.get(node_id), dict):
                        repair_ctx.last_artifacts[node_id]["data_preview"] = rec.get("data_preview") or {}
                        if rec.get("arguments_preview"):
                            repair_ctx.last_artifacts[node_id]["arguments_preview"] = rec.get("arguments_preview")
                        if rec.get("data_path_candidates"):
                            repair_ctx.last_artifacts[node_id]["data_path_candidates"] = rec.get("data_path_candidates")
                        if rec.get("arguments_path_candidates"):
                            repair_ctx.last_artifacts[node_id]["arguments_path_candidates"] = rec.get("arguments_path_candidates")
                extra = dict(repair_ctx.extra or {})
                extra["upstream_artifact_values"] = upstream_value_ctx
                repair_ctx.extra = extra
        except Exception:
            repair_ctx = None

        raw_llm = ""
        used_llm = False
        if limit_class in {"hard_limit_schema", "hard_limit_scope"}:
            verdict = self._hard_limit_verdict(
                tool_name=tool_name,
                err_type=err_type,
                err_msg=err_msg,
                limit_class=limit_class,
            )
        else:
            # ReAct semantics: if the reflector cannot confidently fix,
            # prefer "skip" (no automatic retry) so the brain can revise.
            if deterministic_retry:
                fallback = {
                    "action": "retry",
                    "reason": f"deterministic_fix: {err_type}: {err_msg}",
                    "retry_arguments": deterministic_retry,
                    "natural_language_response": (
                        f"Deterministic path/token fix prepared for '{tool_name}'. Auto-retrying once."
                    ),
                }
            else:
                fallback = {
                    "action": "skip",
                    "reason": f"reflector_no_safe_patch: {err_type}: {err_msg}",
                    "retry_arguments": {},
                    "natural_language_response": (
                        f"No safe one-step retry patch found for '{tool_name}'. Choose different arguments or another tool."
                    ),
                }
            verdict = dict(fallback)
            if repair_ctx is not None:
                llm_verdict, raw_llm = self._llm_reflect(repair_ctx=repair_ctx)
                used_llm = bool(raw_llm or llm_verdict)
                if isinstance(llm_verdict, dict) and llm_verdict:
                    verdict = dict(llm_verdict)
            action = str(verdict.get("action") or "skip").strip().lower()
            if action not in {"retry", "skip", "halt"}:
                verdict = dict(fallback)
            if (
                str(verdict.get("action") or "").strip().lower() == "retry"
                and not isinstance(verdict.get("retry_arguments"), dict)
            ):
                verdict["retry_arguments"] = {}
            if (
                str(verdict.get("action") or "").strip().lower() == "retry"
                and (not dict(verdict.get("retry_arguments") or {}))
                and deterministic_retry
            ):
                verdict["retry_arguments"] = dict(deterministic_retry)

        verdict["failure_classification"] = limit_class
        feedback = self._build_feedback(
            step=step,
            tool_name=tool_name,
            stage=stage,
            err=err,
            limit_class=limit_class,
            verdict=verdict,
            repair_ctx=repair_ctx,
            deterministic_retry=deterministic_retry,
            step_history=step_history,
            schema_autofix_applied=schema_autofix_applied,
            preconditions_applied=preconditions_applied,
            rule_violations=rule_violations,
        )
        return {
            "decision": verdict,
            "feedback": feedback,
            "trace": {
                "used_llm": bool(used_llm),
                "raw_model_output": _clip_text(raw_llm, max_chars=1200) if raw_llm else "",
                "failure_classification": limit_class,
                "deterministic_retry_suggestion": dict(deterministic_retry),
            },
        }

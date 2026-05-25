from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from jsonschema import ValidationError, validate

from ..bootstrap import ensure_import_paths
from ..io.report_writer import ensure_report
from ..runtime.events import append_event, event_record, summarize_args, summarize_outputs
from ..runtime.session import SessionState

ensure_import_paths()

from commands.dispatcher import ToolDispatcher, validate_args_minimal  # noqa: E402
from commands.schemas import CaseState, ToolCall, ToolContext  # noqa: E402
from core.domain_config import get_domain_config  # noqa: E402
from core.paths import project_root  # noqa: E402
from core.plan_dag import AgentPlanDAG, PlanNode, legacy_plan_to_dag  # noqa: E402
from tools.arg_models import RepairConfig, repair_tool_args  # noqa: E402
from mri_agent_shell.runtime.binding_policy import BindingPolicy, POLICY_FULL  # noqa: E402
from mri_agent_shell.runtime.patch_spec import (  # noqa: E402
    ALLOWED_ACTIONS,
    PatchSpec,
    StructuredRepairContext,
    STRUCTURED_REFLECTOR_SYSTEM_PROMPT,
    build_reflector_user_prompt,
    build_structured_repair_context,
    build_tool_schema_info,
    collect_last_successful_artifacts,
)


class ScopeViolation(RuntimeError):
    pass


def _safe_resolve(path_like: str) -> Path:
    return Path(str(path_like)).expanduser().resolve()


def _safe_abspath(path_like: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path_like))))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except Exception:
        return False


def _deep_merge_retry_args(base: Any, patch: Any) -> Any:
    """Overlay retry patch onto prior args without clobbering nested path structures."""
    if isinstance(base, dict) and isinstance(patch, dict):
        out: Dict[str, Any] = dict(base)
        for k, v in patch.items():
            key = str(k)
            if key in out:
                out[key] = _deep_merge_retry_args(out[key], v)
            else:
                out[key] = v
        return out
    return patch


def _guess_case_ref_from_session(session: SessionState) -> str:
    return str(session.case_inputs.get("case_input_path") or session.path_keys.get("case.input") or "").strip()


def _default_external_model_roots() -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()

    def _push(path_like: Any) -> None:
        raw = str(path_like or "").strip()
        if not raw:
            return
        try:
            p = _safe_resolve(raw)
        except Exception:
            return
        key = str(p)
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    _push(Path("~/.cache/torch/hub/bundle").expanduser())
    _push(Path("~/.cache/torch/hub").expanduser())

    try:
        cfg = RepairConfig.from_env(project_root())
        _push(cfg.model_registry_path)
        _push(cfg.prostate_bundle_dir)
        _push(cfg.lesion_weights_dir)
        _push(cfg.brain_bundle_dir)
        _push(cfg.cardiac_cmr_reverse_root)
        _push(cfg.cardiac_results_folder)
        _push(cfg.cardiac_nnunet_python)
        _push(Path(cfg.cardiac_nnunet_python).expanduser().resolve().parent)
        _push(cfg.distortion_repo_root)
        _push(cfg.distortion_diff_ckpt)
        _push(cfg.distortion_cnn_ckpt)
        for p in (cfg.distortion_test_roots or []):
            _push(p)
    except Exception:
        pass
    return out


@dataclass
class _ScopedBinder:
    case_root: Path
    refs: Dict[str, str] = field(default_factory=dict)
    seq_paths: Dict[str, str] = field(default_factory=dict)
    node_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def set_ref(self, key: str, value: str) -> None:
        k = str(key or "").strip()
        v = str(value or "").strip()
        if not k or not v:
            return
        self.refs[k] = v

    def seed_runtime(self, ctx: ToolContext, *, case_file: Optional[Path] = None) -> None:
        self.set_ref("runtime.run_dir", str(ctx.run_dir))
        self.set_ref("runtime.artifacts_dir", str(ctx.artifacts_dir))
        self.set_ref("runtime.case_state_path", str(ctx.case_state_path))
        self.set_ref("case.input", str(self.case_root))
        if case_file is not None:
            self.set_ref("case.file", str(case_file))

    def resolve_refs(self, obj: Any) -> Any:
        token_aliases = {
            "state.path": "runtime.case_state_path",
            "case.path": "case.input",
        }

        def _resolve(v: Any) -> Any:
            if isinstance(v, dict):
                return {k: _resolve(x) for k, x in v.items()}
            if isinstance(v, list):
                return [_resolve(x) for x in v]
            if isinstance(v, str) and v.startswith("@"):
                token = v[1:]
                if token.startswith("node."):
                    parts = token.split(".")
                    if len(parts) >= 3:
                        node_id = parts[1]
                        key_path = parts[2:]
                        cur: Any = self.node_outputs.get(node_id)
                        for seg in key_path:
                            if isinstance(cur, dict) and seg in cur:
                                cur = cur.get(seg)
                            else:
                                cur = None
                                break
                        if cur is not None:
                            return cur
                mapped = self.refs.get(token)
                if mapped is not None and str(mapped).strip():
                    return mapped
                alias = token_aliases.get(token)
                if alias:
                    alias_val = self.refs.get(alias)
                    if alias_val is not None and str(alias_val).strip():
                        return alias_val
                return v
            return v

        return _resolve(obj)

    def resolve_runtime_refs_only(self, obj: Any) -> Any:
        """Resolve only infrastructure tokens (``@runtime.*``, ``@case.input``, ``@case.file``).

        Used by the noToken arm: infrastructure tokens (run_dir, artifacts_dir,
        case_state_path) are not "symbolic binding" — they are bookkeeping the
        executor always needs.  Everything else is left as-is.
        """
        _allowed_prefixes = ("runtime.", "case.input", "case.file")

        def _resolve_rt(v: Any) -> Any:
            if isinstance(v, dict):
                return {k: _resolve_rt(x) for k, x in v.items()}
            if isinstance(v, list):
                return [_resolve_rt(x) for x in v]
            if isinstance(v, str) and v.startswith("@"):
                token = v[1:]
                if any(token.startswith(pfx) or token == pfx for pfx in _allowed_prefixes):
                    mapped = self.refs.get(token)
                    if mapped is not None and str(mapped).strip():
                        return mapped
                return v  # leave all other @-tokens unresolved
            return v

        return _resolve_rt(obj)

    def _normalize_candidate_path(self, p: Path, *, prefer_file: bool) -> str:
        try:
            pp = p.expanduser()
            if pp.exists() and pp.is_file():
                return str(_safe_abspath(str(pp)))
            if pp.exists() and pp.is_dir() and prefer_file:
                picked = self._pick_nifti_from_dir(pp)
                if picked is not None:
                    return str(_safe_abspath(str(picked)))
            if pp.exists() and pp.is_dir():
                return str(_safe_abspath(str(pp)))
            return str(pp)
        except Exception:
            return str(p)

    def _pick_nifti_from_dir(self, path: Path) -> Optional[Path]:
        try:
            if not path.exists() or not path.is_dir():
                return None
            direct = sorted([x for x in path.iterdir() if x.is_file() and x.name.lower().endswith((".nii", ".nii.gz"))])
            if direct:
                return direct[0]
            deep = sorted([x for x in path.rglob("*") if x.is_file() and x.name.lower().endswith((".nii", ".nii.gz"))])
            if deep:
                return deep[0]
        except Exception:
            return None
        return None

    def resolve_sequence_or_case_path(
        self,
        raw: str,
        *,
        preferred_tokens: Optional[List[str]] = None,
        prefer_file: bool = False,
    ) -> Optional[str]:
        s = str(raw or "").strip()
        if not s:
            return None
        # Keep strict unresolved-token behavior for generic refs, but allow @seq.<token>
        # to degrade into fuzzy sequence lookup (e.g., @seq.T2w -> T2w).
        if s.startswith("@seq."):
            s = str(s[len("@seq.") :]).strip()
            if not s:
                return None
        elif s.startswith("@"):
            return None
        elif s.lower().startswith("seq."):
            s = str(s[4:]).strip()
            if not s:
                return None

        try:
            p = Path(s).expanduser()
            if p.is_absolute():
                return self._normalize_candidate_path(p, prefer_file=prefer_file)
        except Exception:
            pass

        aliases = {
            "t2": "T2w",
            "t2w": "T2w",
            "t1c": "T1c",
            "t1": "T1",
            "flair": "FLAIR",
            "adc": "ADC",
            "dwi": "DWI",
            "cine": "CINE",
        }

        candidates: List[str] = []
        if preferred_tokens:
            candidates.extend(preferred_tokens)
        candidates.extend([s, aliases.get(s.lower(), s)])

        seen = set()
        for name in candidates:
            n = str(name or "").strip()
            if not n or n in seen:
                continue
            seen.add(n)
            v = str(self.seq_paths.get(n) or "").strip()
            if not v:
                continue
            try:
                p = Path(v).expanduser()
                if p.exists() or p.is_absolute():
                    resolved = self._normalize_candidate_path(p, prefer_file=prefer_file)
                    if resolved:
                        return resolved
            except Exception:
                return v

        case_dir = self.case_root
        direct_candidates = [
            case_dir / s,
            case_dir / f"{s}.nii.gz",
            case_dir / f"{s}.nii",
            case_dir / s.lower(),
            case_dir / f"{s.lower()}.nii.gz",
            case_dir / f"{s.lower()}.nii",
        ]
        for cand in direct_candidates:
            if cand.exists():
                resolved = self._normalize_candidate_path(cand, prefer_file=prefer_file)
                if resolved:
                    return resolved

        low = s.lower()
        hint_map = {
            "t2w": ("t2", "t2w"),
            "t2": ("t2", "t2w"),
            "adc": ("adc",),
            "dwi": ("dwi", "trace", "high_b", "high-b", "b800", "b1000", "b1400", "b1500", "b2000"),
            "t1c": ("t1c", "t1ce", "t1gd"),
            "t1": ("t1",),
            "flair": ("flair",),
            "cine": ("cine", "bssfp", "ssfp", "sax"),
        }
        hints = hint_map.get(low, (low,))
        try:
            for p in sorted(case_dir.rglob("*")):
                n = p.name.lower()
                if not any(h in n for h in hints):
                    continue
                if p.is_file() and n.endswith((".nii", ".nii.gz", ".dcm")):
                    return str(_safe_abspath(str(p)))
                if p.is_dir():
                    resolved = self._normalize_candidate_path(p, prefer_file=prefer_file)
                    if resolved:
                        return resolved
        except Exception:
            return None
        return None

    def learn_from_tool(self, *, tool_name: str, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return

        for k, v in data.items():
            if isinstance(v, str) and k.endswith("_path"):
                self.set_ref(f"{tool_name}.{k[:-5]}", v)

        if tool_name == "identify_sequences":
            mapping = data.get("mapping") if isinstance(data.get("mapping"), dict) else {}
            for name, path in mapping.items():
                if isinstance(path, str) and path:
                    self.seq_paths[str(name)] = str(path)
                    self.set_ref(f"seq.{name}", str(path))
            if isinstance(mapping.get("T2"), str) and not str(mapping.get("T2w") or "").strip():
                self.seq_paths["T2w"] = str(mapping.get("T2"))
                self.set_ref("seq.T2w", str(mapping.get("T2")))
            if isinstance(mapping.get("T1"), str) and not str(mapping.get("T1c") or "").strip():
                self.seq_paths["T1c"] = str(mapping.get("T1"))
                self.set_ref("seq.T1c", str(mapping.get("T1")))

    def learn_from_node(self, *, node_id: str, data: Dict[str, Any]) -> None:
        self.node_outputs[str(node_id)] = dict(data or {})


@dataclass
class _CaseScopeGuard:
    case_root: Path
    run_root: Path
    external_roots: List[Path] = field(default_factory=list)
    _case_symlink_target_roots: Optional[List[Path]] = field(default=None, init=False, repr=False)

    _EXTERNAL_ARG_KEYS = {
        "bundle_dir",
        "bundle_root",
        "weights_dir",
        "nnunet_python",
        "results_folder",
        "cmr_reverse_root",
        "project_root",
        "repo_root",
        "test_root",
        "python_exec",
        "ckpt",
        "cnn_ckpt",
        "checkpoint",
        "model_registry",
        "model_registry_path",
        "server_base_url",
        "api_base_url",
    }

    _NON_PATH_ARG_KEYS = {
        "output_subdir",
        "method",
        "domain",
        "llm_mode",
        "api_model",
        "temperature",
        "max_tokens",
        "device",
        "note",
    }

    _MUST_BE_ABS_KEYS = {
        "dicom_case_dir",
        "series_inventory_path",
        "case_state_path",
        "case_path",
        "fixed",
        "moving",
        "t2w_ref",
        "t1c_path",
        "t1_path",
        "t2_path",
        "flair_path",
        "cine_path",
        "seg_path",
        "ed_seg_path",
        "es_seg_path",
        "patient_info_path",
    }

    def _is_path_candidate(self, key: str, value: str) -> bool:
        k = str(key or "").strip().lower()
        v = str(value or "").strip()
        if not v or k in self._NON_PATH_ARG_KEYS:
            return False
        if k in self._EXTERNAL_ARG_KEYS:
            return True
        if k in self._MUST_BE_ABS_KEYS:
            return True
        if k.endswith("_path") or k.endswith("_dir") or k.endswith("_root") or k.endswith("_ckpt"):
            if k == "output_subdir":
                return False
            return True
        if v.startswith(("/", "~", "./", "../")):
            return True
        if v.lower().endswith((
            ".nii",
            ".nii.gz",
            ".dcm",
            ".json",
            ".txt",
            ".md",
            ".png",
            ".jpg",
            ".jpeg",
            ".tfm",
            ".pt",
            ".pth",
            ".ckpt",
        )):
            return True
        return False

    def _walk_path_args(self, obj: Any, key_hint: str = "") -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                out.extend(self._walk_path_args(v, key_hint=str(k)))
            return out
        if isinstance(obj, list):
            for v in obj:
                out.extend(self._walk_path_args(v, key_hint=key_hint))
            return out
        if isinstance(obj, str) and self._is_path_candidate(key_hint, obj):
            out.append((key_hint, obj))
        return out

    def _allow_external(self, resolved: Path) -> bool:
        for root in self.external_roots:
            if _is_relative_to(resolved, root):
                return True
        return False

    def _collect_case_symlink_target_roots(self) -> List[Path]:
        cached = self._case_symlink_target_roots
        if isinstance(cached, list):
            return cached
        out: List[Path] = []
        seen: set[str] = set()

        def _push(p: Path) -> None:
            try:
                rp = p.resolve()
            except Exception:
                return
            key = str(rp)
            if key in seen:
                return
            seen.add(key)
            out.append(rp)

        try:
            for p in self.case_root.rglob("*"):
                if not p.is_symlink():
                    continue
                try:
                    target = p.resolve()
                except Exception:
                    continue
                if target.is_dir():
                    _push(target)
                else:
                    _push(target.parent)
        except Exception:
            pass
        self._case_symlink_target_roots = out
        return out

    def _allow_case_symlink_target(self, resolved: Path) -> bool:
        for root in self._collect_case_symlink_target_roots():
            if _is_relative_to(resolved, root):
                return True
        return False

    def validate_args(self, *, tool_name: str, args: Dict[str, Any]) -> None:
        for key, raw in self._walk_path_args(args):
            value = str(raw or "").strip()
            key_l = str(key or "").strip().lower()
            if not value:
                continue
            if value.startswith("@"):
                raise ScopeViolation(f"unresolved reference in path argument '{key}' for {tool_name}: {value}")

            if key_l in self._EXTERNAL_ARG_KEYS:
                try:
                    p = Path(value).expanduser()
                    if p.is_absolute() and not self._allow_external(p.resolve()):
                        # External keys are allowed, but if a root allowlist is provided, enforce it.
                        if self.external_roots:
                            raise ScopeViolation(
                                f"external path '{p}' for '{key}' is outside allow_external_model_roots"
                            )
                except ScopeViolation:
                    raise
                except Exception:
                    pass
                continue

            p = Path(value).expanduser()
            if key_l in self._MUST_BE_ABS_KEYS and (not p.is_absolute()):
                raise ScopeViolation(f"path argument '{key}' must be absolute for {tool_name}: {value}")

            if not p.is_absolute():
                # Relative path-like values are allowed only if they are clearly runtime-local outputs.
                continue

            # Symlink-aware lexical scope check: if the user-facing path itself is
            # inside case/run scope, allow it even if the symlink target resolves
            # outside (common for shared dataset links).
            try:
                lexical = Path(os.path.abspath(str(p)))
            except Exception:
                lexical = p
            if _is_relative_to(lexical, self.case_root):
                continue
            if _is_relative_to(lexical, self.run_root):
                continue

            resolved = p.resolve()
            if _is_relative_to(resolved, self.case_root):
                continue
            if _is_relative_to(resolved, self.run_root):
                continue
            # Allow explicit benchmark/planner contract inputs when the DAG
            # scope declares a path root allowlist (historically named
            # allow_external_model_roots, but reused for explicit noToken
            # input aliases as well).
            if self._allow_external(resolved):
                continue
            if self._allow_case_symlink_target(resolved):
                continue
            raise ScopeViolation(
                f"path argument '{key}' escaped case scope for {tool_name}: {resolved}"
            )


class Cerebellum:
    def __init__(
        self,
        *,
        session: SessionState,
        registry: Any,
        max_attempts: int = 2,
        reflect_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        failure_reflect_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        allow_deterministic_fallback_on_nonretry: bool = True,
        binding_policy: Optional[BindingPolicy] = None,
    ) -> None:
        self.session = session
        self.registry = registry
        self.max_attempts = max(1, int(max_attempts))
        self._last_scope_key = ""
        self._reflect_fn = reflect_fn or self._default_reflect_fn
        self._failure_reflect_fn = failure_reflect_fn or self._default_failure_reflect_fn
        self._failure_retry_limit = 1
        self._allow_deterministic_fallback_on_nonretry = bool(allow_deterministic_fallback_on_nonretry)
        self._reflection_llm: Any = None
        self._reflection_llm_ready = False
        self.binding_policy: BindingPolicy = binding_policy or POLICY_FULL
        self._online_semantic_lint_enabled = str(
            os.getenv("MRI_AGENT_ONLINE_SEMANTIC_LINT", "")
        ).strip().lower() in {"1", "true", "yes", "on"}

        self.dispatcher = ToolDispatcher(
            registry=self.registry,
            runs_root=Path(self.session.runs_root).expanduser().resolve(),
        )

    def _path_arg_audit_enabled(self, *, case_id: str, tool_name: str) -> bool:
        raw = str(os.getenv("MRI_AGENT_PATH_ARG_AUDIT", "")).strip().lower()
        if raw not in {"1", "true", "yes", "on"}:
            return False
        filt = str(os.getenv("MRI_AGENT_PATH_ARG_AUDIT_FILTER", "")).strip().lower()
        if not filt:
            return True
        needles = [x.strip() for x in re.split(r"[,;]", filt) if x.strip()]
        hay = f"{case_id} {tool_name}".lower()
        return any(n in hay for n in needles)

    @staticmethod
    def _is_path_like_arg_key(key: str) -> bool:
        k = str(key or "").strip().lower()
        if not k:
            return False
        if k in {
            "fixed", "moving", "images", "roi_masks", "image_a", "image_b",
            "t2w_ref", "t2w_nifti", "adc_nifti", "highb_nifti",
        }:
            return True
        return any(tok in k for tok in ("path", "dir", "file", "nifti", "mask", "image", "ref", "bundle", "weights", "root"))

    @staticmethod
    def _audit_value_repr(v: Any) -> Any:
        if isinstance(v, str):
            return (v if len(v) <= 200 else (v[:197] + "..."))
        if isinstance(v, (int, float, bool)) or v is None:
            return v
        try:
            s = json.dumps(v, ensure_ascii=False, sort_keys=True)
        except Exception:
            s = repr(v)
        return s if len(s) <= 240 else (s[:237] + "...")

    def _emit_path_arg_audit(
        self,
        *,
        node_id: str,
        tool_name: str,
        case_id: str,
        run_id: str,
        attempt: int,
        trace_path: Path,
        stage_args: Dict[str, Dict[str, Any]],
        emit: Callable[[str], None],
    ) -> None:
        stage_order = ["original", "after_resolve", "after_normalize1", "after_repair", "after_normalize2", "final"]
        keys: set[str] = set()
        for stage in stage_order:
            ad = stage_args.get(stage) or {}
            if not isinstance(ad, dict):
                continue
            for k in ad.keys():
                if self._is_path_like_arg_key(str(k)):
                    keys.add(str(k))

        audits: List[Dict[str, Any]] = []
        for key in sorted(keys):
            chain: Dict[str, Any] = {}
            changed_by: List[str] = []
            prev_s: Optional[str] = None
            for stage in stage_order:
                raw_v = (stage_args.get(stage) or {}).get(key)
                chain[stage] = self._audit_value_repr(raw_v)
                try:
                    cur_s = json.dumps(raw_v, ensure_ascii=False, sort_keys=True)
                except Exception:
                    cur_s = repr(raw_v)
                if prev_s is not None and cur_s != prev_s:
                    changed_by.append(stage)
                prev_s = cur_s
            audits.append({"key": key, "changed_by": changed_by, "chain": chain})

        if not audits:
            return

        rec = event_record(
            event_type="arg_path_audit",
            tool_name=tool_name,
            status="INFO",
            case_id=case_id,
            run_id=run_id,
            attempt=attempt,
        )
        rec["node_id"] = str(node_id)
        rec["path_arg_audits"] = audits
        append_event(trace_path, rec)

        for item in audits:
            chain = item.get("chain") or {}
            changed_by = item.get("changed_by") or []
            emit(
                "ARG_AUDIT "
                f"{node_id} {tool_name} {item['key']} "
                f"changed_by={','.join(changed_by) if changed_by else 'none'} "
                f"original={chain.get('original')!r} "
                f"-> after_resolve={chain.get('after_resolve')!r} "
                f"-> after_normalize1={chain.get('after_normalize1')!r} "
                f"-> after_repair={chain.get('after_repair')!r} "
                f"-> after_normalize2={chain.get('after_normalize2')!r} "
                f"-> final={chain.get('final')!r}"
            )

    @staticmethod
    def _basename_hint(path_like: Any) -> str:
        try:
            return Path(str(path_like or "")).name.lower()
        except Exception:
            return str(path_like or "").strip().lower()

    def _detect_semantic_lint_violation(
        self,
        *,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """High-confidence semantic inconsistency detector (experiment mode).

        This detector intentionally emits evidence + candidate swap space and does
        not auto-correct. The existing reflector decides whether to retry with a
        swap patch.
        """
        if not self._online_semantic_lint_enabled:
            return None

        t = str(tool_name or "").strip()
        a = dict(args or {})

        if t == "register_to_reference":
            fixed = str(a.get("fixed") or "").strip()
            moving = str(a.get("moving") or "").strip()
            if fixed and moving:
                f = self._basename_hint(fixed)
                m = self._basename_hint(moving)
                moving_like = any(k in f for k in ("adc", "dwi", "trace", "highb", "high-b", "b800", "b1000", "b1400", "b1500", "b2000"))
                fixed_like = ("t2" in m) or ("t2w" in m)
                if moving_like and fixed_like:
                    evidence = {
                        "evidence_type": "registration_role_swap",
                        "tool_name": t,
                        "pair": ["fixed", "moving"],
                        "current_map": {"fixed": fixed, "moving": moving},
                        "detector_basis": {
                            "fixed_basename": f,
                            "moving_basename": m,
                            "hint": "fixed looks diffusion-like while moving looks T2-like",
                        },
                        "candidate_repairs": [
                            {
                                "op": "swap_pair",
                                "pair": ["fixed", "moving"],
                                "confidence": "high",
                                "reason": "Registration role semantics likely reversed by modality-role mismatch.",
                            }
                        ],
                    }
                    return evidence

        if t == "brats_mri_segmentation":
            t1c = str(a.get("t1c_path") or "").strip()
            flair = str(a.get("flair_path") or "").strip()
            if t1c and flair:
                t1c_b = self._basename_hint(t1c)
                flair_b = self._basename_hint(flair)
                t1c_looks_flair = "flair" in t1c_b
                flair_looks_t1c = any(k in flair_b for k in ("t1ce", "t1c", "t1gd", "t1_ce"))
                if t1c_looks_flair and flair_looks_t1c:
                    evidence = {
                        "evidence_type": "modality_role_swap",
                        "tool_name": t,
                        "pair": ["t1c_path", "flair_path"],
                        "current_map": {"t1c_path": t1c, "flair_path": flair},
                        "detector_basis": {
                            "t1c_basename": t1c_b,
                            "flair_basename": flair_b,
                            "hint": "t1c_path filename looks FLAIR while flair_path filename looks T1c/T1ce",
                        },
                        "candidate_repairs": [
                            {
                                "op": "swap_pair",
                                "pair": ["t1c_path", "flair_path"],
                                "confidence": "high",
                                "reason": "BraTS modality-role mismatch strongly suggests swapped T1c/FLAIR inputs.",
                            }
                        ],
                    }
                    return evidence

        return None

    def _semantic_lint_error_from_evidence(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        pair = evidence.get("pair") if isinstance(evidence.get("pair"), list) else []
        pair_str = "<->".join([str(x) for x in pair]) if pair else "unknown_pair"
        msg = (
            "High-confidence semantic inconsistency detected after successful tool execution. "
            f"Likely swapped arguments: {pair_str}. "
            f"Evidence: {json.dumps(evidence, ensure_ascii=False, sort_keys=True)}"
        )
        return {
            "type": "SemanticLintViolation",
            "message": msg,
            "semantic_lint_evidence": evidence,
        }

    def _default_reflect_fn(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
        decision = str(args.get("decision") or "continue").strip().lower() if isinstance(args, dict) else "continue"
        if decision not in {"continue", "halt"}:
            decision = "continue"
        reason = str(args.get("reason") or "no-op control node") if isinstance(args, dict) else "no-op control node"
        return {"decision": decision, "reason": reason}

    def _build_reflection_llm(self) -> Any:
        if self._reflection_llm_ready:
            return self._reflection_llm
        self._reflection_llm_ready = True
        cfg = self.session.model_config
        provider = str(getattr(cfg, "provider", "") or "").strip().lower()
        try:
            if provider == "openai_compatible_server":
                from llm.adapter_vllm_server import VLLMOpenAIChatAdapter, VLLMServerConfig  # noqa: E402

                self._reflection_llm = VLLMOpenAIChatAdapter(
                    VLLMServerConfig(
                        base_url=str(getattr(cfg, "server_base_url", "") or "http://127.0.0.1:8000/v1"),
                        model=str(getattr(cfg, "llm", "") or ""),
                        api_key=str(getattr(cfg, "effective_api_key", lambda: "")() or "EMPTY"),
                        max_tokens=int(min(512, int(getattr(cfg, "max_tokens", 512) or 512))),
                        temperature=float(getattr(cfg, "temperature", 0.0) or 0.0),
                    )
                )
                return self._reflection_llm
            if provider == "openai_official":
                from llm.adapter_openai_api import OpenAIChatAdapter, OpenAIConfig  # noqa: E402

                self._reflection_llm = OpenAIChatAdapter(
                    OpenAIConfig(
                        model=str(getattr(cfg, "llm", "") or ""),
                        api_key=str(getattr(cfg, "effective_api_key", lambda: "")() or ""),
                        base_url=(str(getattr(cfg, "api_base_url", "") or "").strip() or None),
                        max_tokens=int(min(512, int(getattr(cfg, "max_tokens", 512) or 512))),
                        temperature=float(getattr(cfg, "temperature", 0.0) or 0.0),
                    )
                )
                return self._reflection_llm
            if provider == "anthropic":
                from llm.adapter_anthropic_api import AnthropicChatAdapter, AnthropicConfig  # noqa: E402

                self._reflection_llm = AnthropicChatAdapter(
                    AnthropicConfig(
                        model=str(getattr(cfg, "llm", "") or ""),
                        api_key=str(getattr(cfg, "effective_api_key", lambda: "")() or ""),
                        max_tokens=int(min(512, int(getattr(cfg, "max_tokens", 512) or 512))),
                        temperature=float(getattr(cfg, "temperature", 0.0) or 0.0),
                    )
                )
                return self._reflection_llm
            if provider == "gemini":
                from llm.adapter_gemini_api import GeminiChatAdapter, GeminiConfig  # noqa: E402

                self._reflection_llm = GeminiChatAdapter(
                    GeminiConfig(
                        model=str(getattr(cfg, "llm", "") or ""),
                        api_key=str(getattr(cfg, "effective_api_key", lambda: "")() or ""),
                        max_tokens=int(min(512, int(getattr(cfg, "max_tokens", 512) or 512))),
                        temperature=float(getattr(cfg, "temperature", 0.0) or 0.0),
                    )
                )
                return self._reflection_llm
        except Exception:
            self._reflection_llm = None
            return None
        self._reflection_llm = None
        return None

    def _parse_reflection_json(self, text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if not raw:
            return {}
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                obj = json.loads(str(m.group(0)))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return {}

    def _collect_reflection_binding_context(self, *, binder: _ScopedBinder) -> Dict[str, Any]:
        token_bindings: Dict[str, str] = {}
        for k, v in (binder.refs or {}).items():
            kk = str(k or "").strip()
            vv = str(v or "").strip()
            if kk and vv:
                token_bindings[f"@{kk}"] = vv

        session_token_bindings: Dict[str, str] = {}
        for k, v in (self.session.path_keys or {}).items():
            kk = str(k or "").strip()
            vv = str(v or "").strip()
            if kk and vv:
                session_token_bindings[f"@{kk}"] = vv

        return {
            "available_valid_tokens": sorted(token_bindings.keys()),
            "token_bindings": token_bindings,
            "session_path_keys": sorted(session_token_bindings.keys()),
            "session_token_bindings": session_token_bindings,
            "available_modalities": sorted([str(k) for k in (binder.seq_paths or {}).keys() if str(k).strip()]),
        }

    def _apply_ref_token_aliases(self, obj: Any) -> Tuple[Any, bool]:
        aliases = {
            "@state.path": "@runtime.case_state_path",
            "state.path": "@runtime.case_state_path",
            "@case.path": "@case.input",
            "case.path": "@case.input",
            "@case.root": "@case.input",
            "case.root": "@case.input",
        }
        changed = False

        def _map(v: Any) -> Any:
            nonlocal changed
            if isinstance(v, dict):
                return {k: _map(x) for k, x in v.items()}
            if isinstance(v, list):
                return [_map(x) for x in v]
            if isinstance(v, str) and v in aliases:
                changed = True
                return aliases[v]
            return v

        return _map(obj), changed

    def _recover_missing_required_args(
        self,
        *,
        repair_ctx: StructuredRepairContext,
    ) -> Dict[str, Any]:
        """Heuristic Tier-2 fallback for argument omission.

        This is intentionally kept out of deterministic Tier-1 flow so
        ``bcr_deterministic_only`` can still serve as a strict ablation arm.
        """
        tool_schema = repair_ctx.tool_schema
        if tool_schema is None:
            return {}

        out = dict(repair_ctx.failing_args or {})
        required = [str(k).strip() for k in (tool_schema.required_keys or []) if str(k).strip()]
        missing = [k for k in required if k not in out]
        changed = False

        tokens = {str(k): str(v) for k, v in (repair_ctx.available_tokens or {}).items() if str(k).strip()}
        modalities = {str(x).strip() for x in (repair_ctx.available_modalities or []) if str(x).strip()}

        # identify_sequences commonly accepts (dicom_case_dir OR series_inventory_path)
        # via runtime logic, so omission may not surface in schema.required_keys.
        if str(repair_ctx.failing_tool or "").strip() == "identify_sequences":
            has_dicom = bool(str(out.get("dicom_case_dir") or "").strip())
            has_inventory = bool(str(out.get("series_inventory_path") or "").strip())
            if (not has_dicom) and (not has_inventory):
                out["dicom_case_dir"] = "@case.input"
                changed = True

        if not missing and (not changed):
            return {}

        def _pick_seq_token(*prefs: str) -> str:
            for pref in prefs:
                p = str(pref or "").strip()
                if not p:
                    continue
                direct = f"@seq.{p}"
                if direct in tokens:
                    return direct
            for pref in prefs:
                p = str(pref or "").strip().lower()
                if not p:
                    continue
                for tk in tokens:
                    if tk.lower() == f"@seq.{p}":
                        return tk
            for pref in prefs:
                p = str(pref or "").strip()
                if p and p in modalities:
                    return p
            return ""

        def _pick_any_existing_path() -> str:
            for v in out.values():
                s = str(v or "").strip()
                if not s:
                    continue
                if ("/" in s) or s.startswith("@"):
                    return s
            for tk in sorted(tokens.keys()):
                if tk.startswith("@seq."):
                    return tk
            return ""

        for key in missing:
            k = str(key).strip()
            low = k.lower()
            val = ""
            if low in {"case_state_path", "state_path"}:
                val = "@runtime.case_state_path"
            elif low in {"case_path", "dicom_case_dir"}:
                val = "@case.input"
            elif low in {"fixed"}:
                val = _pick_seq_token("T2w", "T2", "T1c", "T1", "FLAIR", "CINE")
            elif low in {"moving"}:
                val = _pick_seq_token("ADC", "DWI", "T2w", "T2", "FLAIR", "CINE")
            elif low in {"t2w_ref", "t2w_nifti", "t2_path"}:
                val = _pick_seq_token("T2w", "T2")
            elif low in {"adc_nifti"}:
                val = _pick_seq_token("ADC")
            elif low in {"highb_nifti"}:
                val = _pick_seq_token("DWI")
            elif low in {"t1c_path"}:
                val = _pick_seq_token("T1c", "T1")
            elif low in {"t1_path"}:
                val = _pick_seq_token("T1")
            elif low in {"flair_path"}:
                val = _pick_seq_token("FLAIR")
            elif low in {"cine_path"}:
                val = _pick_seq_token("CINE")
            elif low in {"h5_path"}:
                val = "@case.file" if "@case.file" in tokens else ""
            elif low == "input_nifti":
                val = _pick_any_existing_path()
            elif low.endswith("_path") or low.endswith("_nifti"):
                val = _pick_any_existing_path()

            if val and (k not in out):
                out[k] = val
                changed = True

        return out if changed else {}

    def _deterministic_retry_suggestion(
        self,
        *,
        node: PlanNode,
        last_args: Dict[str, Any],
        error: Dict[str, Any],
        binding_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        out = dict(last_args or {})
        changed = False

        out, alias_changed = self._apply_ref_token_aliases(out)
        changed = changed or alias_changed

        msg = str((error or {}).get("message") or "")
        m = re.search(r"unresolved reference in path argument '([^']+)'[^:]*: (\S+)", msg)
        unresolved_key = str(m.group(1) or "").strip().lower() if m else ""
        unresolved_ref = str(m.group(2) or "").strip() if m else ""

        valid_tokens = set(
            str(x or "").strip() for x in (binding_context.get("available_valid_tokens") or []) if str(x or "").strip()
        )

        def _can_use(tok: str) -> bool:
            if tok in valid_tokens:
                return True
            return tok in {"@runtime.case_state_path", "@case.input", "@case.file"}

        def _strip_seq_prefix(tok: str) -> str:
            t = str(tok or "").strip()
            if t.startswith("@seq."):
                return str(t[len("@seq.") :]).strip()
            if t.lower().startswith("seq."):
                return str(t[4:]).strip()
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
            # ScopeViolation unresolved @seq.<X>: retry with bare token <X>
            # so downstream resolver can use sequence map + fuzzy rglob fallback.
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

        if str(node.tool_name or "").strip() == "rag_search":
            raw = str(out.get("case_state_path") or "").strip()
            if (not raw or raw in {"@state.path", "state.path"}) and _can_use("@runtime.case_state_path"):
                out["case_state_path"] = "@runtime.case_state_path"
                changed = True

        return out if changed else {}

    def _classify_reflection_limit(self, *, err_type: str, err_msg: str, deterministic_retry: Dict[str, Any]) -> str:
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

    def _hard_limit_response(
        self,
        *,
        tool_name: str,
        err_type: str,
        err_msg: str,
        limit_class: str,
    ) -> Dict[str, Any]:
        if limit_class == "hard_limit_schema":
            nl = (
                f"I encountered a {err_type} for '{tool_name}': {err_msg}. "
                "As a runtime execution agent, I do not have permission to modify system schema configurations. "
                "Please have a developer add this tool to the domain whitelist or update the backend schema rules."
            )
            return {
                "action": "halt",
                "reason": f"hard_limit_schema: {err_type}: {err_msg}",
                "retry_arguments": {},
                "natural_language_response": nl,
            }
        if limit_class == "hard_limit_scope":
            nl = (
                f"I encountered a {err_type} for '{tool_name}': {err_msg}. "
                "As a runtime execution agent, I cannot bypass case scope guard restrictions. "
                "Please adjust case paths or policy configuration in code."
            )
            return {
                "action": "halt",
                "reason": f"hard_limit_scope: {err_type}: {err_msg}",
                "retry_arguments": {},
                "natural_language_response": nl,
            }
        return {
            "action": "halt",
            "reason": f"{err_type}: {err_msg}",
            "retry_arguments": {},
            "natural_language_response": (
                f"I cannot continue because required step '{tool_name}' failed: {err_msg}."
            ),
        }

    def _default_failure_reflect_fn(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Tier-2 reflector using structured repair context + PatchSpec validation."""
        err = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        err_type = str((err or {}).get("type") or "RuntimeError")
        err_msg = str((err or {}).get("message") or "required node failed")
        tool_name = str(payload.get("tool_name") or "unknown_tool")
        deterministic_retry = payload.get("deterministic_retry_suggestion")
        deterministic_retry = (
            dict(deterministic_retry)
            if isinstance(deterministic_retry, dict) and deterministic_retry
            else {}
        )
        limit_class = self._classify_reflection_limit(
            err_type=err_type,
            err_msg=err_msg,
            deterministic_retry=deterministic_retry,
        )
        if limit_class in {"hard_limit_schema", "hard_limit_scope"}:
            return self._hard_limit_response(
                tool_name=tool_name,
                err_type=err_type,
                err_msg=err_msg,
                limit_class=limit_class,
            )
        fallback = {
            "action": "halt",
            "reason": f"{err_type}: {err_msg}",
            "retry_arguments": {},
            "natural_language_response": (
                f"I cannot continue because required step '{tool_name}' failed: {err_msg}."
            ),
        }
        if deterministic_retry:
            fallback = {
                "action": "retry",
                "reason": f"auto_fix_unresolved_path: {err_type}: {err_msg}",
                "retry_arguments": deterministic_retry,
                "natural_language_response": (
                    f"I corrected a path/token binding for required step '{tool_name}' and will retry."
                ),
            }

        # --- Build structured repair context if available ---
        repair_ctx: Optional[StructuredRepairContext] = payload.get("_structured_repair_context")
        msg_low = str(err_msg or "").strip().lower()
        missing_required_signal = (
            ("required property" in msg_low)
            or ("missing required" in msg_low)
            or ("missing argument" in msg_low)
            or ("requires " in msg_low)
        )

        # Tier-2 heuristic for argument omission. Keep this path only in the
        # default reflector so deterministic-only ablations remain strict.
        if (not deterministic_retry) and missing_required_signal and repair_ctx is not None:
            heuristic_retry = self._recover_missing_required_args(repair_ctx=repair_ctx)
            if heuristic_retry:
                return {
                    "action": "retry",
                    "reason": f"heuristic_restore_required_args: {err_type}: {err_msg}",
                    "retry_arguments": heuristic_retry,
                    "natural_language_response": (
                        f"I restored missing required arguments for '{tool_name}' and will retry."
                    ),
                }

        llm = self._build_reflection_llm()
        if llm is None:
            return fallback

        # --- Determine allowed arg keys for whitelist filtering ---
        allowed_arg_keys = None
        if repair_ctx is not None and repair_ctx.tool_schema is not None:
            schema_props = repair_ctx.tool_schema.input_schema.get("properties", {})
            if isinstance(schema_props, dict):
                allowed_arg_keys = set(schema_props.keys())

        # --- Build prompt from structured context (or legacy fallback) ---
        if repair_ctx is not None:
            msgs = [
                {"role": "system", "content": STRUCTURED_REFLECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": build_reflector_user_prompt(repair_ctx)},
            ]
        else:
            # Legacy prompt for backward compatibility
            msgs = [
                {
                    "role": "system",
                    "content": STRUCTURED_REFLECTOR_SYSTEM_PROMPT,
                },
                {"role": "user", "content": json.dumps({
                    "failure_classification": limit_class,
                    "failure": payload,
                }, ensure_ascii=False)},
            ]

        try:
            raw = str(llm.generate(msgs)).strip()
            patch = PatchSpec.parse_and_validate(
                raw,
                allowed_arg_keys=allowed_arg_keys,
                allowed_actions=("retry", "skip", "halt"),
            )
            return patch.to_legacy_verdict()
        except Exception:
            return fallback

    def _reflect_on_required_failure(
        self,
        *,
        node: PlanNode,
        rec: Dict[str, Any],
        binder: _ScopedBinder,
        scope_domain: str,
        case_id: str,
        run_id: str,
        step_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        last_arguments = (
            dict(rec.get("arguments") or {})
            if isinstance(rec.get("arguments"), dict)
            else dict(node.arguments or {})
        )
        err = dict(rec.get("error") or {}) if isinstance(rec.get("error"), dict) else {}
        binding_context = self._collect_reflection_binding_context(binder=binder)
        deterministic_retry = self._deterministic_retry_suggestion(
            node=node,
            last_args=last_arguments,
            error=err,
            binding_context=binding_context,
        )

        # --- Build structured repair context ---
        limit_class = self._classify_reflection_limit(
            err_type=str(err.get("type") or "RuntimeError"),
            err_msg=str(err.get("message") or ""),
            deterministic_retry=deterministic_retry,
        )
        repair_ctx: Optional[StructuredRepairContext] = None
        try:
            repair_ctx = build_structured_repair_context(
                tool_name=str(node.tool_name),
                last_arguments=last_arguments,
                error=err,
                failure_classification=limit_class,
                binding_context=binding_context,
                deterministic_retry=deterministic_retry if deterministic_retry else None,
                registry=self.registry,
                step_results=step_results or [],
            )
        except Exception:
            repair_ctx = None

        payload = {
            "node_id": str(node.node_id),
            "tool_name": str(node.tool_name),
            "stage": str(node.stage or ""),
            "scope_domain": str(scope_domain),
            "case_id": str(case_id),
            "run_id": str(run_id),
            "error": err,
            "last_arguments": last_arguments,
            "status": str(rec.get("status") or "FAIL"),
            "binding_context": binding_context,
            "deterministic_retry_suggestion": deterministic_retry,
            "_structured_repair_context": repair_ctx,
        }
        verdict = self._failure_reflect_fn(payload)
        if not isinstance(verdict, dict):
            verdict = {}
        action = str(verdict.get("action") or "halt").strip().lower()
        if action not in {"halt", "skip", "retry"}:
            action = "halt"
        reason = str(verdict.get("reason") or "").strip()
        nl = str(verdict.get("natural_language_response") or "").strip()
        retry_args = verdict.get("retry_arguments")
        if not isinstance(retry_args, dict):
            retry_args = {}
        retry_args, _ = self._apply_ref_token_aliases(dict(retry_args))
        if action == "retry" and not retry_args and deterministic_retry:
            retry_args = dict(deterministic_retry)
        if action in {"halt", "skip"} and deterministic_retry and self._allow_deterministic_fallback_on_nonretry:
            action = "retry"
            retry_args = dict(deterministic_retry)
            if not reason:
                reason = "Auto-corrected unresolved path/token binding for retry."
            if not nl:
                nl = "I corrected a path/token binding issue and retried the failed required step."
        return {
            "action": action,
            "reason": reason,
            "natural_language_response": nl,
            "retry_arguments": dict(retry_args),
        }

    def execute_plan(
        self,
        plan: Dict[str, Any],
        *,
        emit: Callable[[str], None] = print,
    ) -> Dict[str, Any]:
        plan_obj = dict(plan or {})
        domain = str(plan_obj.get("domain") or "prostate").strip().lower() or "prostate"
        self._sync_case_from_plan(plan_obj)

        case_id = str(
            plan_obj.get("case_id")
            or self.session.case_id
            or f"case_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        ).strip()
        self.session.case_id = case_id

        case_ref = str(plan_obj.get("case_ref") or _guess_case_ref_from_session(self.session) or "").strip()
        if case_ref in {"", "case.input", "@case.input", "case_input", "-"}:
            case_ref = _guess_case_ref_from_session(self.session)
        if not case_ref:
            raise ValueError("Cannot execute plan without case_ref bound to a real directory")
        p = Path(case_ref).expanduser()
        if not p.is_absolute():
            p = Path(self.session.workspace_path) / p
        case_ref = str(_safe_abspath(str(p)))

        dag = legacy_plan_to_dag(
            legacy_plan=plan_obj,
            domain=domain,
            case_id=case_id,
            case_ref=case_ref,
            workspace_root=str(self.session.workspace_path),
            runs_root=str(self.session.runs_root),
            plan_id=str(plan_obj.get("plan_id") or plan_obj.get("run_id") or ""),
        )
        return self.execute_dag(dag, emit=emit)

    def execute_dag(
        self,
        dag: AgentPlanDAG | Dict[str, Any],
        *,
        emit: Callable[[str], None] = print,
    ) -> Dict[str, Any]:
        dag_obj = AgentPlanDAG.model_validate(dag)
        scope = dag_obj.case_scope

        case_ref_path = _safe_abspath(scope.case_ref)
        case_file: Optional[Path] = None
        if case_ref_path.exists() and case_ref_path.is_file():
            case_file = case_ref_path
            case_root = case_ref_path.parent
        else:
            case_root = case_ref_path
        if not case_root.exists() or not case_root.is_dir():
            raise ValueError(f"case_ref is not a directory: {case_ref_path}")

        self._reset_session_scope(scope_key=f"{scope.domain}|{case_root}", case_root=case_root, case_id=scope.case_id)
        if case_file is not None:
            self.session.case_inputs["case_file_path"] = str(case_file)
            self.session.path_keys["case.file"] = str(case_file)

        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        state, ctx = self.dispatcher.create_run(case_id=scope.case_id, run_id=run_id)

        try:
            state.metadata = {**(state.metadata or {}), "domain": str(scope.domain)}
            state.write_json(ctx.case_state_path)
        except Exception:
            pass

        trace_path = Path(self.session.workspace_path) / "logs" / scope.case_id / "tool_trace.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        binder = _ScopedBinder(case_root=case_root)
        binder.seed_runtime(ctx, case_file=case_file)

        guard = _CaseScopeGuard(
            case_root=case_root,
            run_root=Path(ctx.run_dir).resolve(),
            external_roots=[
                *[
                    _safe_resolve(x)
                    for x in (scope.allow_external_model_roots or [])
                    if str(x).strip()
                ],
                *_default_external_model_roots(),
            ],
        )

        step_results: List[Dict[str, Any]] = []
        reflection_decisions: List[Dict[str, Any]] = []
        node_status: Dict[str, str] = {}
        node_by_id = {n.node_id: n for n in dag_obj.nodes}
        overall_ok = True
        halt = False
        agent_response = ""

        while len(node_status) < len(dag_obj.nodes):
            progressed = False

            for node in dag_obj.nodes:
                node_id = str(node.node_id)
                if node_id in node_status:
                    continue

                if (not bool(node.required)) and bool(dag_obj.policy.skip_optional_by_default):
                    rec = self._skipped_step_record(
                        node=node,
                        reason="optional node skipped by policy (required=False)",
                        status="SKIPPED_OPTIONAL",
                        case_id=scope.case_id,
                        run_id=run_id,
                        trace_path=trace_path,
                        emit=emit,
                    )
                    step_results.append(rec)
                    node_status[node_id] = "SKIPPED_OPTIONAL"
                    progressed = True
                    continue

                deps = [d for d in node.depends_on if str(d).strip()]
                unresolved = [d for d in deps if d not in node_status]
                if unresolved:
                    continue

                bad_deps = [d for d in deps if node_status.get(d) != "DONE"]
                if bad_deps:
                    bad_desc = ", ".join([f"{d}:{node_status.get(d)}" for d in bad_deps])
                    status = "BLOCKED" if node.required else "SKIPPED_OPTIONAL"
                    rec = self._skipped_step_record(
                        node=node,
                        reason=f"blocked by dependencies: {bad_desc}",
                        status=status,
                        case_id=scope.case_id,
                        run_id=run_id,
                        trace_path=trace_path,
                        emit=emit,
                    )
                    step_results.append(rec)
                    node_status[node_id] = status
                    progressed = True
                    if node.required:
                        overall_ok = False
                        if bool(dag_obj.policy.stop_on_required_failure):
                            halt = True
                            break
                    continue

                ok, rec = self._execute_node(
                    node=node,
                    state=state,
                    ctx=ctx,
                    scope_domain=str(scope.domain),
                    guard=guard,
                    binder=binder,
                    case_id=scope.case_id,
                    run_id=run_id,
                    trace_path=trace_path,
                    max_attempts=(node.max_attempts or dag_obj.policy.max_attempts_default or self.max_attempts),
                    emit=emit,
                )
                step_results.append(rec)
                node_status[node_id] = str(rec.get("status") or ("DONE" if ok else "FAIL"))
                if ok:
                    step_answer = self._extract_natural_response_from_step(rec)
                    if step_answer:
                        agent_response = step_answer
                progressed = True
                if not ok and node.required:
                    reflection = self._reflect_on_required_failure(
                        node=node,
                        rec=rec,
                        binder=binder,
                        scope_domain=str(scope.domain),
                        case_id=scope.case_id,
                        run_id=run_id,
                        step_results=step_results,
                    )
                    action = str(reflection.get("action") or "halt").strip().lower()
                    reason = str(reflection.get("reason") or "").strip()
                    nl = str(reflection.get("natural_language_response") or "").strip()
                    reflection_rec = {
                        "node_id": node_id,
                        "tool_name": str(node.tool_name),
                        "action": action,
                        "reason": reason,
                        "natural_language_response": nl,
                    }
                    reflection_decisions.append(reflection_rec)
                    append_event(
                        trace_path,
                        event_record(
                            event_type="reflection",
                            tool_name=str(node.tool_name),
                            status=action.upper(),
                            case_id=scope.case_id,
                            run_id=run_id,
                            error=({"type": "ReflectionDecision", "message": reason} if reason else None),
                            outputs={"natural_language_response": nl} if nl else None,
                        ),
                    )
                    if nl:
                        agent_response = nl
                        emit(f"REFLECT {node.node_id} {node.tool_name}: {nl}")

                    if action == "retry":
                        retry_args = reflection.get("retry_arguments")
                        if not isinstance(retry_args, dict):
                            retry_args = {}
                        base_args = dict(node.arguments or {})
                        merged_retry_args = (
                            _deep_merge_retry_args(base_args, retry_args)
                            if retry_args
                            else base_args
                        )
                        retry_node = node.model_copy(
                            update={
                                "required": True,
                                "arguments": merged_retry_args,
                            }
                        )
                        ok_retry, rec_retry = self._execute_node(
                            node=retry_node,
                            state=state,
                            ctx=ctx,
                            scope_domain=str(scope.domain),
                            guard=guard,
                            binder=binder,
                            case_id=scope.case_id,
                            run_id=run_id,
                            trace_path=trace_path,
                            max_attempts=self._failure_retry_limit,
                            emit=emit,
                        )
                        rec_retry["reflection_action"] = "retry"
                        step_results.append(rec_retry)
                        node_status[node_id] = str(rec_retry.get("status") or ("DONE" if ok_retry else "FAIL"))
                        if ok_retry:
                            retry_answer = self._extract_natural_response_from_step(rec_retry)
                            if retry_answer:
                                agent_response = retry_answer
                        if ok_retry:
                            continue
                        overall_ok = False
                        if bool(dag_obj.policy.stop_on_required_failure):
                            halt = True
                            break
                        continue

                    if action == "skip":
                        node_status[node_id] = "SKIPPED_BY_REFLECTION"
                        if isinstance(step_results[-1], dict):
                            step_results[-1]["status"] = "SKIPPED_BY_REFLECTION"
                            step_results[-1]["reflection_action"] = "skip"
                        overall_ok = False
                        continue

                    overall_ok = False
                    if bool(dag_obj.policy.stop_on_required_failure):
                        halt = True
                        break

            if halt:
                break

            if not progressed:
                # Cycle/invalid dependency references: mark remaining nodes blocked.
                for node in dag_obj.nodes:
                    node_id = str(node.node_id)
                    if node_id in node_status:
                        continue
                    missing = [d for d in node.depends_on if d not in node_by_id and d not in node_status]
                    waiting = [d for d in node.depends_on if d in node_by_id and d not in node_status]
                    reason_bits = []
                    if missing:
                        reason_bits.append(f"unknown deps={missing}")
                    if waiting:
                        reason_bits.append(f"cyclic/unreachable deps={waiting}")
                    reason = "; ".join(reason_bits) or "unreachable dependency state"
                    status = "BLOCKED" if node.required else "SKIPPED_OPTIONAL"
                    rec = self._skipped_step_record(
                        node=node,
                        reason=reason,
                        status=status,
                        case_id=scope.case_id,
                        run_id=run_id,
                        trace_path=trace_path,
                        emit=emit,
                    )
                    step_results.append(rec)
                    node_status[node_id] = status
                    if node.required:
                        overall_ok = False
                break

        report_path = ensure_report(
            workspace_path=Path(self.session.workspace_path),
            case_id=scope.case_id,
            goal=dag_obj.goal,
            run_dir=ctx.run_dir,
            step_results=step_results,
        )

        final_event = event_record(
            event_type="session_end",
            tool_name="<session>",
            status=("DONE" if overall_ok else "FAIL"),
            case_id=scope.case_id,
            run_id=run_id,
            outputs={
                "report_path": str(report_path),
                "run_dir": str(ctx.run_dir),
                "scope_id": str(scope.scope_id),
            },
        )
        append_event(trace_path, final_event)

        self.session.last_run = {
            "ok": overall_ok,
            "case_id": scope.case_id,
            "run_id": run_id,
            "run_dir": str(ctx.run_dir),
            "trace_path": str(trace_path),
            "report_path": str(report_path),
            "domain": str(scope.domain),
            "scope_id": str(scope.scope_id),
            "n_steps": len(step_results),
            "reflection_decisions": reflection_decisions,
            "natural_language_response": agent_response,
            "final_answer": agent_response,
        }
        self.session.set_path_key("report.md", str(report_path))
        self.session.set_path_key("runtime.last_run_dir", str(ctx.run_dir))
        self.session.set_path_key("runtime.last_trace", str(trace_path))

        return dict(self.session.last_run)

    def _reset_session_scope(self, *, scope_key: str, case_root: Path, case_id: str) -> None:
        if self._last_scope_key != scope_key:
            self.session.path_keys = {}
            self._last_scope_key = scope_key
        self.session.case_inputs["case_input_path"] = str(case_root)
        self.session.path_keys["case.input"] = str(case_root)
        self.session.case_id = str(case_id)

    def _skipped_step_record(
        self,
        *,
        node: PlanNode,
        reason: str,
        status: str,
        case_id: str,
        run_id: str,
        trace_path: Path,
        emit: Callable[[str], None],
    ) -> Dict[str, Any]:
        args = dict(node.arguments or {})
        args_summary = summarize_args(args)
        err = {"type": "Skipped", "message": reason}
        rec = event_record(
            event_type="tool_call",
            tool_name=str(node.tool_name),
            status=status,
            case_id=case_id,
            run_id=run_id,
            args_summary=args_summary,
            error=err,
        )
        rec["node_id"] = str(node.node_id)
        rec["label"] = str(node.label or "")
        append_event(trace_path, rec)
        emit(f"SKIP {node.node_id} {node.tool_name}: {reason}")
        return {
            "node_id": str(node.node_id),
            "label": str(node.label or ""),
            "tool_name": str(node.tool_name),
            "status": status,
            "required": bool(node.required),
            "arguments": dict(args),
            "duration_s": None,
            "error": err,
            "outputs": {},
            "data": {},
        }

    def _execute_node(
        self,
        *,
        node: PlanNode,
        state: CaseState,
        ctx: ToolContext,
        scope_domain: str,
        guard: _CaseScopeGuard,
        binder: _ScopedBinder,
        case_id: str,
        run_id: str,
        trace_path: Path,
        max_attempts: int,
        emit: Callable[[str], None],
    ) -> Tuple[bool, Dict[str, Any]]:
        node_type = str(getattr(node, "node_type", "tool") or "tool").strip().lower() or "tool"
        if node_type in {"reflect", "gate"}:
            return self._execute_control_node(
                node=node,
                node_type=node_type,
                state=state,
                ctx=ctx,
                scope_domain=scope_domain,
                case_id=case_id,
                run_id=run_id,
                trace_path=trace_path,
                emit=emit,
            )

        tool_name = str(node.tool_name)
        stage = str(node.stage or "misc")
        required = bool(node.required)
        current_args = dict(node.arguments or {})
        attempts = max(1, int(max_attempts or 1))
        last_error: Optional[Dict[str, Any]] = None

        bp = self.binding_policy

        for attempt in range(1, attempts + 1):
            args_original = dict(current_args)

            # Step 1: symbolic token resolution (controlled by policy)
            if bp.symbolic_bind_enabled:
                resolved_args = binder.resolve_refs(current_args)
            else:
                # Even when symbolic binding is off, we still resolve
                # @runtime.* tokens (infrastructure, not semantic binding)
                resolved_args = binder.resolve_runtime_refs_only(current_args)
            args_after_resolve = dict(resolved_args)

            # Step 2: deterministic tool-arg normalisation (controlled by policy)
            resolved_args = self._normalize_step_args(
                tool_name=tool_name, args=resolved_args, binder=binder,
                policy=bp,
            )
            args_after_normalize1 = dict(resolved_args)

            # Step 3: Pydantic schema/type repair (controlled by policy)
            if bp.tool_arg_type_repair_enabled:
                repaired_args = repair_tool_args(
                    tool_name,
                    dict(resolved_args),
                    state_path=ctx.case_state_path,
                    ctx_case_state_path=ctx.case_state_path,
                    dicom_case_dir=self.session.case_inputs.get("case_input_path"),
                    domain=get_domain_config(scope_domain),
                    suppress_sequence_resolve=bool(getattr(bp, "suppress_sequence_resolve", False)),
                    suppress_node_output_autowire=bool(getattr(bp, "suppress_node_output_autowire", False)),
                )
            else:
                repaired_args = dict(resolved_args)
            args_after_repair = dict(repaired_args)

            # Step 4: second normalisation pass (same policy)
            repaired_args = self._normalize_step_args(
                tool_name=tool_name, args=repaired_args, binder=binder,
                policy=bp,
            )
            args_after_normalize2 = dict(repaired_args)
            if self._path_arg_audit_enabled(case_id=case_id, tool_name=tool_name):
                self._emit_path_arg_audit(
                    node_id=str(node.node_id),
                    tool_name=tool_name,
                    case_id=case_id,
                    run_id=run_id,
                    attempt=attempt,
                    trace_path=trace_path,
                    stage_args={
                        "original": args_original,
                        "after_resolve": args_after_resolve,
                        "after_normalize1": args_after_normalize1,
                        "after_repair": args_after_repair,
                        "after_normalize2": args_after_normalize2,
                        "final": dict(repaired_args),
                    },
                    emit=emit,
                )

            try:
                guard.validate_args(tool_name=tool_name, args=repaired_args)
            except ScopeViolation as e:
                err = {"type": "ScopeViolation", "message": str(e)}
                args_summary = summarize_args(repaired_args)
                fail = event_record(
                    event_type="tool_call",
                    tool_name=tool_name,
                    status="FAIL",
                    case_id=case_id,
                    run_id=run_id,
                    args_summary=args_summary,
                    error=err,
                    attempt=attempt,
                )
                fail["node_id"] = str(node.node_id)
                fail["label"] = str(node.label or "")
                append_event(trace_path, fail)
                emit(f"FAIL {node.node_id} {tool_name}: {err['message']}")
                return False, {
                    "node_id": str(node.node_id),
                    "label": str(node.label or ""),
                    "tool_name": tool_name,
                    "status": "FAIL",
                    "required": required,
                    "arguments": dict(repaired_args),
                    "duration_s": None,
                    "error": err,
                    "outputs": {},
                    "data": {},
                }

            args_summary = summarize_args(repaired_args)
            start = event_record(
                event_type="tool_call",
                tool_name=tool_name,
                status="RUNNING",
                case_id=case_id,
                run_id=run_id,
                args_summary=args_summary,
                attempt=attempt,
            )
            start["node_id"] = str(node.node_id)
            start["label"] = str(node.label or "")
            append_event(trace_path, start)
            emit(f"RUN {node.node_id} {tool_name}({args_summary})")

            t0 = time.time()
            try:
                self._validate_tool_args(tool_name=tool_name, args=repaired_args)
                call = ToolCall(
                    tool_name=tool_name,
                    arguments=repaired_args,
                    call_id=f"dag_{node.node_id}_try{attempt}",
                    case_id=case_id,
                    stage=stage,
                    requested_by="mri_agent_shell:cerebellum",
                )
                result = self.dispatcher.dispatch(call, state, ctx)
                dur = time.time() - t0

                if result.ok:
                    semantic_evidence = self._detect_semantic_lint_violation(
                        tool_name=tool_name,
                        args=repaired_args,
                    )
                    if semantic_evidence:
                        err = self._semantic_lint_error_from_evidence(semantic_evidence)
                        last_error = err
                        fail = event_record(
                            event_type="tool_call",
                            tool_name=tool_name,
                            status="FAIL",
                            case_id=case_id,
                            run_id=run_id,
                            args_summary=args_summary,
                            duration_s=dur,
                            error=err,
                            attempt=attempt,
                        )
                        fail["node_id"] = str(node.node_id)
                        fail["label"] = str(node.label or "")
                        fail["semantic_lint_evidence"] = semantic_evidence
                        append_event(trace_path, fail)
                        emit(
                            f"FAIL {node.node_id} {tool_name}: "
                            f"SemanticLintViolation: {semantic_evidence.get('evidence_type')}"
                        )
                        return False, {
                            "node_id": str(node.node_id),
                            "label": str(node.label or ""),
                            "tool_name": tool_name,
                            "status": "FAIL",
                            "required": required,
                            "arguments": dict(repaired_args),
                            "duration_s": round(dur, 4),
                            "error": err,
                            "semantic_lint_evidence": semantic_evidence,
                            "outputs": {},
                            "data": {},
                        }

                    outputs = summarize_outputs(result.data)
                    self._learn_from_result(tool_name=tool_name, data=result.data, binder=binder)
                    binder.learn_from_node(node_id=str(node.node_id), data=result.data)
                    end = event_record(
                        event_type="tool_call",
                        tool_name=tool_name,
                        status="DONE",
                        case_id=case_id,
                        run_id=run_id,
                        args_summary=args_summary,
                        duration_s=dur,
                        outputs=outputs,
                        attempt=attempt,
                    )
                    end["node_id"] = str(node.node_id)
                    end["label"] = str(node.label or "")
                    append_event(trace_path, end)
                    emit(f"DONE {node.node_id} {tool_name}: {dur:.2f}s")
                    return True, {
                        "node_id": str(node.node_id),
                        "label": str(node.label or ""),
                        "tool_name": tool_name,
                        "status": "DONE",
                        "required": required,
                        "arguments": dict(repaired_args),
                        "duration_s": round(dur, 4),
                        "outputs": outputs,
                        "data": result.data,
                    }

                err = result.error.to_dict() if result.error else {"type": "RuntimeError", "message": "unknown error"}
                last_error = err
                fail = event_record(
                    event_type="tool_call",
                    tool_name=tool_name,
                    status="FAIL",
                    case_id=case_id,
                    run_id=run_id,
                    args_summary=args_summary,
                    duration_s=dur,
                    error=err,
                    attempt=attempt,
                )
                fail["node_id"] = str(node.node_id)
                fail["label"] = str(node.label or "")
                append_event(trace_path, fail)
                emit(f"FAIL {node.node_id} {tool_name}: {err.get('type')}: {err.get('message')}")

                if attempt < attempts and self._is_retryable(err):
                    current_args = self._repair_args_after_error(tool_name=tool_name, args=current_args, err=err, binder=binder)
                    retry = event_record(
                        event_type="retry",
                        tool_name=tool_name,
                        status="RETRYING",
                        case_id=case_id,
                        run_id=run_id,
                        args_summary=args_summary,
                        error=err,
                        attempt=attempt,
                    )
                    retry["node_id"] = str(node.node_id)
                    retry["label"] = str(node.label or "")
                    append_event(trace_path, retry)
                    continue

                return False, {
                    "node_id": str(node.node_id),
                    "label": str(node.label or ""),
                    "tool_name": tool_name,
                    "status": "FAIL",
                    "required": required,
                    "arguments": dict(repaired_args),
                    "duration_s": round(dur, 4),
                    "error": err,
                    "outputs": {},
                    "data": {},
                }

            except Exception as e:
                dur = time.time() - t0
                err = {"type": type(e).__name__, "message": str(e)}
                last_error = err
                fail = event_record(
                    event_type="tool_call",
                    tool_name=tool_name,
                    status="FAIL",
                    case_id=case_id,
                    run_id=run_id,
                    args_summary=args_summary,
                    duration_s=dur,
                    error=err,
                    attempt=attempt,
                )
                fail["node_id"] = str(node.node_id)
                fail["label"] = str(node.label or "")
                append_event(trace_path, fail)
                emit(f"FAIL {node.node_id} {tool_name}: {err['type']}: {err['message']}")

                if attempt < attempts and self._is_retryable(err):
                    current_args = self._repair_args_after_error(tool_name=tool_name, args=current_args, err=err, binder=binder)
                    retry = event_record(
                        event_type="retry",
                        tool_name=tool_name,
                        status="RETRYING",
                        case_id=case_id,
                        run_id=run_id,
                        args_summary=args_summary,
                        error=err,
                        attempt=attempt,
                    )
                    retry["node_id"] = str(node.node_id)
                    retry["label"] = str(node.label or "")
                    append_event(trace_path, retry)
                    continue

                return False, {
                    "node_id": str(node.node_id),
                    "label": str(node.label or ""),
                    "tool_name": tool_name,
                    "status": "FAIL",
                    "required": required,
                    "arguments": dict(repaired_args),
                    "duration_s": round(dur, 4),
                    "error": err,
                    "outputs": {},
                    "data": {},
                }

        return False, {
            "node_id": str(node.node_id),
            "label": str(node.label or ""),
            "tool_name": tool_name,
            "status": "FAIL",
            "required": required,
            "arguments": dict(current_args),
            "duration_s": None,
            "error": last_error or {"type": "RuntimeError", "message": "exhausted retries"},
            "outputs": {},
            "data": {},
        }

    def _execute_control_node(
        self,
        *,
        node: PlanNode,
        node_type: str,
        state: CaseState,
        ctx: ToolContext,
        scope_domain: str,
        case_id: str,
        run_id: str,
        trace_path: Path,
        emit: Callable[[str], None],
    ) -> Tuple[bool, Dict[str, Any]]:
        del state  # reserved for future control-node policies
        tool_name = str(node.tool_name or f"__{node_type}__")
        args = dict(node.arguments or {})
        args_summary = summarize_args(args)
        start = event_record(
            event_type="control_node",
            tool_name=tool_name,
            status="RUNNING",
            case_id=case_id,
            run_id=run_id,
            args_summary=args_summary,
        )
        start["node_id"] = str(node.node_id)
        start["label"] = str(node.label or "")
        append_event(trace_path, start)
        emit(f"RUN {node.node_id} {tool_name}({args_summary}) [node_type={node_type}]")

        t0 = time.time()
        payload = {
            "node_id": str(node.node_id),
            "node_type": node_type,
            "tool_name": tool_name,
            "label": str(node.label or ""),
            "arguments": args,
            "scope_domain": str(scope_domain),
            "case_id": str(case_id),
            "run_id": str(run_id),
            "state_path": str(ctx.case_state_path),
            "run_dir": str(ctx.run_dir),
        }
        try:
            verdict = self._reflect_fn(payload)
            decision = str((verdict or {}).get("decision") or "continue").strip().lower()
            if decision not in {"continue", "halt"}:
                raise ValueError(f"unsupported control decision: {decision}")
            reason = str((verdict or {}).get("reason") or "").strip()
            dur = time.time() - t0

            if decision == "halt":
                err = {"type": "ReflectHalt", "message": reason or "control node requested halt"}
                fail = event_record(
                    event_type="control_node",
                    tool_name=tool_name,
                    status="FAIL",
                    case_id=case_id,
                    run_id=run_id,
                    args_summary=args_summary,
                    duration_s=dur,
                    error=err,
                )
                fail["node_id"] = str(node.node_id)
                fail["label"] = str(node.label or "")
                append_event(trace_path, fail)
                emit(f"FAIL {node.node_id} {tool_name}: {err['message']}")
                return False, {
                    "node_id": str(node.node_id),
                    "label": str(node.label or ""),
                    "tool_name": tool_name,
                    "status": "FAIL",
                    "required": bool(node.required),
                    "arguments": dict(args),
                    "duration_s": round(dur, 4),
                    "error": err,
                    "outputs": {},
                    "data": {"decision": decision, **({"reason": reason} if reason else {})},
                }

            outputs = {"decision": decision, **({"reason": reason} if reason else {})}
            done = event_record(
                event_type="control_node",
                tool_name=tool_name,
                status="DONE",
                case_id=case_id,
                run_id=run_id,
                args_summary=args_summary,
                duration_s=dur,
                outputs=outputs,
            )
            done["node_id"] = str(node.node_id)
            done["label"] = str(node.label or "")
            append_event(trace_path, done)
            emit(f"DONE {node.node_id} {tool_name}: {dur:.2f}s")
            return True, {
                "node_id": str(node.node_id),
                "label": str(node.label or ""),
                "tool_name": tool_name,
                "status": "DONE",
                "required": bool(node.required),
                "arguments": dict(args),
                "duration_s": round(dur, 4),
                "outputs": outputs,
                "data": outputs,
            }
        except Exception as e:
            dur = time.time() - t0
            err = {"type": type(e).__name__, "message": str(e)}
            fail = event_record(
                event_type="control_node",
                tool_name=tool_name,
                status="FAIL",
                case_id=case_id,
                run_id=run_id,
                args_summary=args_summary,
                duration_s=dur,
                error=err,
            )
            fail["node_id"] = str(node.node_id)
            fail["label"] = str(node.label or "")
            append_event(trace_path, fail)
            emit(f"FAIL {node.node_id} {tool_name}: {err['type']}: {err['message']}")
            return False, {
                "node_id": str(node.node_id),
                "label": str(node.label or ""),
                "tool_name": tool_name,
                "status": "FAIL",
                "required": bool(node.required),
                "arguments": dict(args),
                "duration_s": round(dur, 4),
                "error": err,
                "outputs": {},
                "data": {},
            }

    def _sync_case_from_plan(self, plan: Dict[str, Any]) -> None:
        if not isinstance(plan, dict):
            return
        raw_ref = str(plan.get("case_ref") or "").strip()
        if not raw_ref:
            return

        if raw_ref in {"case.input", "@case.input"}:
            bound = str(self.session.case_inputs.get("case_input_path") or "").strip()
            if bound:
                self.session.path_keys["case.input"] = bound
            return

        p = Path(raw_ref).expanduser()
        if not p.is_absolute():
            p = Path(self.session.workspace_path) / p
        try:
            case_ref = str(_safe_abspath(str(p)))
        except Exception:
            case_ref = str(p)

        self.session.case_inputs["case_input_path"] = case_ref
        self.session.path_keys["case.input"] = case_ref
        if not str(self.session.case_id or "").strip():
            self.session.case_id = Path(case_ref).name

    def _extract_natural_response_from_step(self, rec: Dict[str, Any]) -> str:
        if not isinstance(rec, dict):
            return ""
        data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
        for key in ("final_answer", "natural_language_response", "answer"):
            v = str(data.get(key) or "").strip()
            if v:
                return v
        return ""

    def _normalize_step_args(
        self,
        *,
        tool_name: str,
        args: Dict[str, Any],
        binder: Optional[_ScopedBinder] = None,
        policy: Optional[BindingPolicy] = None,
    ) -> Dict[str, Any]:
        out = dict(args or {})
        bp = policy or self.binding_policy
        bind = binder
        if bind is None:
            case_ref = _guess_case_ref_from_session(self.session)
            if not case_ref:
                return out
            try:
                bind = _ScopedBinder(case_root=_safe_abspath(case_ref))
            except Exception:
                bind = _ScopedBinder(case_root=Path(case_ref).expanduser())

        # ----- identify_sequences: always allowed (type normalisation) -----
        if tool_name == "identify_sequences":
            raw = out.get("dicom_case_dir")
            if isinstance(raw, str):
                case_input = str(bind.case_root)
                if raw.startswith("@") or raw in {"case.input", "@case.input"}:
                    out["dicom_case_dir"] = case_input
                else:
                    try:
                        p = Path(raw).expanduser()
                        if not p.is_absolute():
                            out["dicom_case_dir"] = case_input
                    except Exception:
                        out["dicom_case_dir"] = case_input
            return out

        # ------------------------------------------------------------------
        # denoise_image_bm3d / compare_nifti_slices: auto-resolve
        # unresolved @-tokens to the first available NIfTI from the
        # prior ingest step, sequence paths, or the case directory.
        # POLICY: requires implicit_node_autowire_enabled
        # ------------------------------------------------------------------
        if tool_name in ("denoise_image_bm3d", "compare_nifti_slices", "resample_image", "generate_qa_snapshot"):
            if not bp.implicit_node_autowire_enabled:
                return out  # noToken: skip implicit NIfTI guessing
            _nifti_keys = (
                ("input_nifti",) if tool_name in ("denoise_image_bm3d", "resample_image", "generate_qa_snapshot")
                else ("image_a", "image_b")
            )
            for key in _nifti_keys:
                raw = str(out.get(key) or "").strip()
                # Only auto-resolve tokens that resolve_refs could not handle
                # (i.e. still start with @ but are NOT valid @node.* results).
                if not raw or (raw.startswith("@") and not raw.startswith("@node.")):
                    nifti: Optional[str] = None
                    # 1. Try sequence paths populated by identify_sequences
                    if bind.seq_paths:
                        for _name, _path in bind.seq_paths.items():
                            _sp = str(_path or "").strip()
                            if _sp:
                                try:
                                    if Path(_sp).exists():
                                        nifti = str(_safe_abspath(_sp))
                                        break
                                except Exception:
                                    pass
                    # 2. Try artifacts/ingest/nifti directory
                    if not nifti:
                        _arts = str(bind.refs.get("runtime.artifacts_dir") or "").strip()
                        if _arts:
                            for _sub in ("ingest/nifti", "ingest"):
                                _found = bind._pick_nifti_from_dir(Path(_arts) / _sub)
                                if _found is not None:
                                    nifti = str(_safe_abspath(str(_found)))
                                    break
                    # 3. Fall back to the case root
                    if not nifti:
                        _found = bind._pick_nifti_from_dir(bind.case_root)
                        if _found is not None:
                            nifti = str(_safe_abspath(str(_found)))
                    if nifti:
                        out[key] = nifti
            return out

        # POLICY: sequence-name autocomplete requires implicit_seq_autocomplete_enabled
        if tool_name == "segment_prostate":
            if not bp.implicit_seq_autocomplete_enabled:
                return out
            tok = str(out.get("t2w_ref") or "").strip()
            if tok:
                resolved = bind.resolve_sequence_or_case_path(tok, preferred_tokens=["T2w", "T2"], prefer_file=True)
                if resolved:
                    out["t2w_ref"] = resolved
            return out

        # POLICY: sequence-name autocomplete requires implicit_seq_autocomplete_enabled
        if tool_name == "register_to_reference":
            if not bp.implicit_seq_autocomplete_enabled:
                return out
            def _prefs_for_register_token(raw_tok: str, *, is_fixed: bool) -> List[str]:
                s = str(raw_tok or "").strip()
                low = s.lower()
                if low.startswith("@seq."):
                    low = low[len("@seq.") :]
                if low.startswith("seq."):
                    low = low[len("seq.") :]
                if low in {"t2w", "t2"}:
                    return ["T2w", "T2"] if is_fixed else ["T2", "T2w"]
                if low in {"t1c", "t1ce", "t1gd", "t1_gd", "t1_ce"}:
                    return ["T1c", "T1"] if is_fixed else ["T1c", "T1"]
                if low in {"t1"}:
                    return ["T1"]
                if "adc" in low:
                    return ["ADC"]
                if any(k in low for k in ("dwi", "trace", "high_b", "high-b", "b800", "b1000", "b1400", "b1500", "b2000")):
                    return ["DWI"]
                if "flair" in low:
                    return ["FLAIR"]
                if any(k in low for k in ("cine", "bssfp", "ssfp", "sax", "shortaxis", "short_axis")):
                    return ["CINE"]
                return [str(raw_tok or "").strip()]

            fixed = str(out.get("fixed") or "").strip()
            moving = str(out.get("moving") or "").strip()
            if fixed:
                fixed_res = bind.resolve_sequence_or_case_path(
                    fixed,
                    preferred_tokens=_prefs_for_register_token(fixed, is_fixed=True),
                    prefer_file=True,
                )
                if fixed_res:
                    out["fixed"] = fixed_res
            if moving:
                move_res = bind.resolve_sequence_or_case_path(
                    moving,
                    preferred_tokens=_prefs_for_register_token(moving, is_fixed=False),
                    prefer_file=True,
                )
                if move_res:
                    out["moving"] = move_res
            return out

        # POLICY: sequence-name autocomplete requires implicit_seq_autocomplete_enabled
        if tool_name == "brats_mri_segmentation":
            if not bp.implicit_seq_autocomplete_enabled:
                return out
            for key, prefs in (
                ("t1c_path", ["T1c", "T1"]),
                ("t1_path", ["T1"]),
                ("t2_path", ["T2"]),
                ("flair_path", ["FLAIR"]),
            ):
                raw = str(out.get(key) or "").strip()
                if not raw:
                    continue
                resolved = bind.resolve_sequence_or_case_path(raw, preferred_tokens=prefs, prefer_file=True)
                if resolved:
                    out[key] = resolved
            return out

        # POLICY: sequence-name autocomplete requires implicit_seq_autocomplete_enabled
        if tool_name == "segment_cardiac_cine":
            if not bp.implicit_seq_autocomplete_enabled:
                return out
            raw = str(out.get("cine_path") or "").strip()
            if raw:
                resolved = bind.resolve_sequence_or_case_path(raw, preferred_tokens=["CINE"], prefer_file=True)
                if resolved:
                    out["cine_path"] = resolved
            return out

        if tool_name == "dummy_load_case":
            raw = str(out.get("case_path") or "").strip()
            if raw in {"case.input", "@case.input"}:
                out["case_path"] = str(bind.case_root)

        return out

    def _repair_args_after_error(
        self,
        *,
        tool_name: str,
        args: Dict[str, Any],
        err: Dict[str, Any],
        binder: Optional[_ScopedBinder] = None,
    ) -> Dict[str, Any]:
        out = dict(args or {})
        msg = str((err or {}).get("message") or "").lower()
        et = str((err or {}).get("type") or "").lower()
        if (
            ("not found" not in msg)
            and ("filenotfound" not in et)
            and ("no dicom series ids found" not in msg)
            and ("no dicom files" not in msg)
        ):
            return out
        return self._normalize_step_args(tool_name=tool_name, args=out, binder=binder, policy=self.binding_policy)

    def _validate_tool_args(self, *, tool_name: str, args: Dict[str, Any]) -> None:
        tool = self.registry.get(tool_name)
        schema = tool.spec.input_schema or {}
        if not schema:
            return
        try:
            validate(instance=args, schema=schema)
        except ValidationError:
            validate_args_minimal(schema, args)

    def _is_retryable(self, err: Dict[str, Any]) -> bool:
        et = str((err or {}).get("type") or "").lower()
        msg = str((err or {}).get("message") or "").lower()
        retry_signals = (
            "timeout",
            "tempor",
            "connection",
            "http",
            "rate limit",
            "429",
            "503",
            "filenotfound",
            "not found",
            "no dicom files",
            "no dicom series ids found",
        )
        return any(tok in et or tok in msg for tok in retry_signals)

    def _learn_from_result(self, *, tool_name: str, data: Dict[str, Any], binder: _ScopedBinder) -> None:
        if not isinstance(data, dict):
            return

        binder.learn_from_tool(tool_name=tool_name, data=data)

        # Session cache is no longer used for execution binding, but keeping these keys
        # helps shell UX/introspection and backward compatibility.
        for k, v in data.items():
            if isinstance(v, str) and k.endswith("_path"):
                self.session.set_path_key(f"{tool_name}.{k[:-5]}", v)

        if tool_name == "identify_sequences":
            mapping = data.get("mapping") if isinstance(data.get("mapping"), dict) else {}
            for name, path in mapping.items():
                if isinstance(path, str) and path:
                    self.session.set_path_key(f"seq.{name}", path)
            if isinstance(mapping.get("T2"), str) and not str(mapping.get("T2w") or "").strip():
                self.session.set_path_key("seq.T2w", str(mapping.get("T2")))
            if isinstance(mapping.get("T1"), str) and not str(mapping.get("T1c") or "").strip():
                self.session.set_path_key("seq.T1c", str(mapping.get("T1")))

        if tool_name == "segment_prostate":
            if isinstance(data.get("prostate_mask_path"), str):
                self.session.set_path_key("seg.prostate_mask", str(data["prostate_mask_path"]))
            if isinstance(data.get("zone_mask_path"), str):
                self.session.set_path_key("seg.zone_mask", str(data["zone_mask_path"]))

        if tool_name == "brats_mri_segmentation":
            for k, alias in (
                ("tc_mask_path", "seg.tc_mask"),
                ("wt_mask_path", "seg.wt_mask"),
                ("et_mask_path", "seg.et_mask"),
            ):
                if isinstance(data.get(k), str):
                    self.session.set_path_key(alias, str(data[k]))

        if tool_name == "segment_cardiac_cine":
            if isinstance(data.get("seg_path"), str):
                self.session.set_path_key("seg.cardiac_mask", str(data["seg_path"]))

        if tool_name == "dummy_segment":
            if isinstance(data.get("mask_path"), str):
                self.session.set_path_key("dummy_segment.mask", str(data["mask_path"]))

        if tool_name in {"generate_report", "dummy_generate_report"}:
            if isinstance(data.get("report_txt_path"), str):
                self.session.set_path_key("report.md", str(data["report_txt_path"]))
            if isinstance(data.get("report_json_path"), str):
                self.session.set_path_key("report.json", str(data["report_json_path"]))

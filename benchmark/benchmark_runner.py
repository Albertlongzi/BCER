from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    here = Path(__file__).resolve()
    sys.path.append(str(here.parents[1]))
    sys.path.append(str(here.parents[2]))

from agent.loop import run_agent_loop
from agent.subagents.prompts import build_reactive_system_prompt
from benchmark import benchmark_runner as bench_v1
from commands.schemas import ToolCall
from core.domain_config import get_domain_config
from core.paths import project_root


PAPER_ARM_CHOICES = ("bcer", "bcer_sketch", "react", "react_token", "react_token_reflector")
ARM_ALIASES = {"bcer": "bcer_sketch"}
ARM_CHOICES = PAPER_ARM_CHOICES
FAULT_CHOICES = (
    "none",
    # Tier-1: deterministic reflector should handle these
    "token_mutation",
    "path_mutation",
    # Tier-2-recoverable: requires LLM reflection to fix
    "argument_omission",
    "semantic_swap",
    "space_mismatch",
    # Tier-2-nonrecoverable: reflector MUST halt/block
    "missing_modality",
    "scope_violation",
    # Crash resilience (not reflector-related)
    "timeout",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_arm(arm: str) -> str:
    return ARM_ALIASES.get(str(arm or "").strip(), str(arm or "").strip())


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = str(line or "").strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _tasks_obj(path: Path) -> Dict[str, Any]:
    raw = _read_json(path)
    if isinstance(raw.get("tasks"), dict):
        return raw["tasks"]
    return raw


def _normalize_case_row(row: Dict[str, Any]) -> Dict[str, Any]:
    case_root = str(row.get("case_root") or row.get("case_ref") or "").strip()
    domain = str(row.get("domain") or "").strip().lower()
    case_id = str(row.get("case_id") or Path(case_root).name or "case").strip()
    modalities = row.get("modalities") if isinstance(row.get("modalities"), dict) else {}
    supports = row.get("supports_tasks") if isinstance(row.get("supports_tasks"), list) else []
    input_aliases = (
        row.get("input_aliases")
        if isinstance(row.get("input_aliases"), dict)
        else (row.get("canonical_input_aliases") if isinstance(row.get("canonical_input_aliases"), dict) else {})
    )
    return {
        "case_id": case_id,
        "domain": domain,
        "case_root": case_root,
        "modalities": modalities,
        "supports_tasks": [str(x) for x in supports],
        "input_format": str(row.get("input_format") or ""),
        "input_aliases": dict(input_aliases or {}),
        "notes": row.get("notes"),
    }


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _render_goal(template: str, *, task_id: str, case: Dict[str, Any], request_type: str) -> str:
    ctx = _SafeFormatDict(
        {
            "task_id": task_id,
            "request_type": request_type,
            "case_id": case.get("case_id"),
            "domain": case.get("domain"),
            "case_root": case.get("case_root"),
            "input_format": case.get("input_format"),
            "modalities": json.dumps(case.get("modalities") or {}, ensure_ascii=False),
            "timestamp": _utc_now_iso(),
        }
    )
    out = str(template or "").format_map(ctx).strip()
    return out


def _task_domain_ok(contract: Dict[str, Any], case_domain: str) -> bool:
    raw = contract.get("domain")
    if isinstance(raw, str):
        return str(raw).strip().lower() == case_domain
    if isinstance(raw, list):
        return case_domain in [str(x).strip().lower() for x in raw if str(x).strip()]
    return True


@dataclass
class FaultInjectionSpecV2:
    enabled: bool = False
    fault: str = "none"
    seed: int = 0
    profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FaultInjectorV2:
    spec: FaultInjectionSpecV2
    applied: bool = False
    target_seen_calls: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)

    _PATH_HINT_KEYS = {
        "dicom_case_dir",
        "series_inventory_path",
        "case_state_path",
        "state_path",
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
        "input_nifti",
        "output_nifti",
        "h5_path",
    }
    _LOW_IMPACT_PATH_KEYS = {
        "series_inventory_path",
        "case_state_path",
        "state_path",
        "case_path",
    }

    def _record(
        self,
        *,
        source: str,
        tool_name: str,
        key: str,
        before: Any,
        after: Any,
        mode: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        rec = {
            "ts": _utc_now_iso(),
            "source": source,
            "tool_name": tool_name,
            "fault": self.spec.fault,
            "mode": mode,
            "key": key,
            "from": before,
            "to": after,
        }
        if isinstance(extra, dict) and extra:
            rec.update(extra)
        self.events.append(rec)

    def _token_pairs(self) -> List[Tuple[str, str]]:
        raw = self.spec.profile.get("token_pairs") if isinstance(self.spec.profile, dict) else None
        out: List[Tuple[str, str]] = []
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                src = str(item[0] or "").strip()
                dst = str(item[1] or "").strip()
                if src and dst and src != dst:
                    out.append((src, dst))
        if not out:
            out = [
                ("@runtime.case_state_path", "@state.path"),
                ("@seq.T2w", "@seq.T2"),
                ("@case.input", "@case.root"),
            ]
        return out

    def _token_difficulty(self) -> str:
        prof = self.spec.profile if isinstance(self.spec.profile, dict) else {}
        raw = str(prof.get("difficulty") or os.environ.get("MRI_AGENT_FAULT_TOKEN_DIFFICULTY") or "").strip().lower()
        return raw if raw in {"easy", "hard", "hard_v2"} else "easy"

    def _is_path_candidate(self, key: str, value: str) -> bool:
        k = str(key or "").strip().lower()
        v = str(value or "").strip()
        if not v:
            return False
        if k in self._PATH_HINT_KEYS:
            return True
        if k.endswith("_path") or k.endswith("_dir") or k.endswith("_root"):
            return True
        if v.startswith("/") or v.startswith("~/"):
            return True
        if "/" in v or "\\" in v:
            return True
        if v.lower().endswith((".nii", ".nii.gz", ".h5", ".json", ".txt", ".csv", ".png", ".tfm")):
            return True
        return False

    def _mutate_token(self, *, args: Dict[str, Any], tool_name: str, source: str) -> Tuple[Dict[str, Any], bool]:
        new_args = dict(args)
        pairs = self._token_pairs()
        difficulty = self._token_difficulty()

        if difficulty in {"hard", "hard_v2"}:
            # Prefer corruptions that deterministic alias-fixers do NOT know,
            # but that a reflector can recover from using available token lists
            # and upstream node outputs. In practice, @node.* and @seq.* refs
            # provide the best signal for LLM-assisted recovery.
            candidates: List[Tuple[int, str, str, str]] = []
            for key in sorted(new_args.keys()):
                val = new_args.get(key)
                if not isinstance(val, str) or not val.startswith("@"):
                    continue
                mutated = ""
                score = -100
                if val.startswith("@node."):
                    parts = val.split(".")
                    if len(parts) >= 3:
                        # Corrupt only the output field, keeping node_id intact.
                        suffix = "__tokhardv2" if difficulty == "hard_v2" else "__tokhard"
                        parts[-1] = parts[-1] + suffix
                        mutated = ".".join(parts)
                        score = 500 if difficulty == "hard_v2" else 300
                elif val.startswith("@seq."):
                    head, _, tail = val.partition("@seq.")
                    tail = str(tail or "").strip()
                    if tail:
                        suffix = "__TOKHARDV2" if difficulty == "hard_v2" else "__TOKHARD"
                        mutated = f"{head}@seq.{tail}{suffix}"
                        score = 300 if difficulty == "hard_v2" else 220
                elif val.startswith("@runtime.") or val.startswith("@case.") or val.startswith("@state."):
                    # Low priority in hard mode because deterministic repair
                    # already has explicit aliases for many of these.
                    if difficulty == "hard":
                        if val.startswith("@runtime.case_state_path"):
                            mutated = val.replace("@runtime.case_state_path", "@runtime.case_state_json", 1)
                        elif val.startswith("@state.path"):
                            mutated = val.replace("@state.path", "@state.case_state_json", 1)
                        elif val.startswith("@case.input"):
                            mutated = val.replace("@case.input", "@case.root", 1)
                        if mutated:
                            score = 10
                    else:
                        # hard_v2 tries to avoid these deterministic-friendly aliases entirely
                        continue
                else:
                    mutated = f"{val}__tokhardv2" if difficulty == "hard_v2" else f"{val}__tokhard"
                    score = 30 if difficulty == "hard_v2" else 50
                if mutated and mutated != val:
                    # De-prioritize low-impact bookkeeping keys so token-hard
                    # faults hit task-critical references more often.
                    low_key = str(key or "").strip().lower()
                    if low_key in {"case_state_path", "state_path"}:
                        score -= 100
                    candidates.append((score, str(key), val, mutated))
            if candidates:
                candidates.sort(key=lambda x: (-x[0], x[1]))
                _, key, before, after = candidates[0]
                new_args[key] = after
                self._record(
                        source=source,
                        tool_name=tool_name,
                        key=key,
                        before=before,
                        after=after,
                        mode="token_hard_corrupt",
                        extra={"difficulty": difficulty},
                    )
                return new_args, True

        # 1) Exact token-pair replacement (profile-driven).
        for key in sorted(new_args.keys()):
            val = new_args.get(key)
            if not isinstance(val, str):
                continue
            for src, dst in pairs:
                if val == src:
                    new_args[key] = dst
                    self._record(source=source, tool_name=tool_name, key=key, before=val, after=dst, mode="token_pair")
                    return new_args, True

        # 2) Substring replacement for embedded symbolic refs.
        for key in sorted(new_args.keys()):
            val = new_args.get(key)
            if not isinstance(val, str):
                continue
            for src, dst in pairs:
                if src and src in val:
                    mutated = val.replace(src, dst, 1)
                    if mutated != val:
                        new_args[key] = mutated
                        self._record(source=source, tool_name=tool_name, key=key, before=val, after=mutated, mode="token_pair_substr")
                        return new_args, True

        # 3) Key-aware alias corruption, to ensure mutation is applicable on
        # real planner args even when symbolic pairs are absent.
        for key in sorted(new_args.keys()):
            val = new_args.get(key)
            if not isinstance(val, str) or not str(val).strip():
                continue
            low_key = str(key).strip().lower()
            if low_key in {"case_state_path", "state_path"}:
                mutated = "@state.path"
                if val != mutated:
                    new_args[key] = mutated
                    self._record(source=source, tool_name=tool_name, key=key, before=val, after=mutated, mode="token_alias_state")
                    return new_args, True
            if low_key in {"dicom_case_dir", "case_path"}:
                mutated = "@case.root"
                if val != mutated:
                    new_args[key] = mutated
                    self._record(source=source, tool_name=tool_name, key=key, before=val, after=mutated, mode="token_alias_case")
                    return new_args, True

        # 4) Mutate existing symbolic tokens directly.
        for key in sorted(new_args.keys()):
            val = new_args.get(key)
            if not isinstance(val, str):
                continue
            if val.startswith("@"):
                mutated = f"{val}_typo"
                new_args[key] = mutated
                self._record(source=source, tool_name=tool_name, key=key, before=val, after=mutated, mode="token_typo")
                return new_args, True

        # 5) Bare modality-token corruption.
        token_typos = {
            "T2w": "T2w_typo",
            "ADC": "ADC_typo",
            "DWI": "DWI_typo",
            "T1": "T1_typo",
            "T1c": "T1c_typo",
            "T2": "T2_typo",
            "FLAIR": "FLAIR_typo",
            "CINE": "CINE_typo",
        }
        for key in sorted(new_args.keys()):
            val = new_args.get(key)
            if not isinstance(val, str):
                continue
            if val in token_typos:
                mutated = token_typos[val]
                new_args[key] = mutated
                self._record(source=source, tool_name=tool_name, key=key, before=val, after=mutated, mode="token_alias")
                return new_args, True

        # 6) Last-resort mutation for path-like fields so the run is marked
        # as injected instead of silently becoming a no-op combo.
        for key in sorted(new_args.keys()):
            val = new_args.get(key)
            if not isinstance(val, str):
                continue
            if not self._is_path_candidate(key, val):
                continue
            p = Path(val).expanduser()
            name = p.name or f"{tool_name}_{key}"
            if name.endswith(".nii.gz"):
                mutated_name = name[:-7] + "_token.nii.gz"
            else:
                mutated_name = name + ".token"
            mutated = str(p.with_name(mutated_name)) if name else str(val) + ".token"
            if mutated != val:
                new_args[key] = mutated
                self._record(source=source, tool_name=tool_name, key=key, before=val, after=mutated, mode="token_path_suffix")
                return new_args, True

        return new_args, False

    def _modality_tag(self, value: str) -> str:
        stem = Path(str(value or "")).name.lower()
        tags = ("t1c", "flair", "t1", "t2w", "t2", "adc", "dwi", "highb", "tracew", "cine", "seg")
        for tag in tags:
            if tag in stem:
                return tag
        return ""

    def _semantic_inroot_path_candidate(self, *, args: Dict[str, Any], pick_key: str, before: str) -> str:
        before_s = str(before or "")
        before_path = Path(before_s).expanduser()
        before_ext = "".join(before_path.suffixes)
        before_tag = self._modality_tag(before_s)

        best: Tuple[int, str] = (-10**9, "")

        for k in sorted(args.keys()):
            if k == pick_key:
                continue
            val = args.get(k)
            if not isinstance(val, str):
                continue
            cand = str(val).strip()
            if (not cand) or (cand == before_s):
                continue
            if not self._is_path_candidate(k, cand):
                continue
            cand_p = Path(cand).expanduser()
            if not cand_p.is_absolute():
                continue
            if not cand_p.exists():
                continue

            score = 0
            if {pick_key, k} == {"fixed", "moving"}:
                score += 100
            if before_ext and "".join(cand_p.suffixes) == before_ext:
                score += 20
            cand_tag = self._modality_tag(cand)
            if before_tag and cand_tag and cand_tag != before_tag:
                score += 40
            if score > best[0]:
                best = (score, cand)

        if best[1]:
            return best[1]

        if before_path.is_absolute() and before_path.parent.exists():
            before_name = before_path.name
            for sib in sorted(before_path.parent.glob("*")):
                if not sib.is_file():
                    continue
                if sib.name == before_name:
                    continue
                if before_ext and not sib.name.endswith(before_ext):
                    continue
                cand = str(sib)
                if self._modality_tag(cand) and self._modality_tag(cand) != before_tag:
                    return cand

        return ""

    def _build_missing_path(self, *, before: str, tool_name: str, pick_key: str) -> str:
        p = Path(before).expanduser()
        if p.is_absolute():
            file_like = bool(p.suffix) or p.name.lower().endswith(".nii.gz")
            if file_like:
                return str(p.parent / "__benchmark_missing__" / (p.name + ".missing"))
            return str(p / "__benchmark_missing__" / f"{tool_name}_{pick_key}.missing")

        rp = Path(before)
        file_like = bool(rp.suffix) or rp.name.lower().endswith(".nii.gz")
        if file_like:
            return str(rp.parent / "__benchmark_missing__" / (rp.name + ".missing"))
        return str(rp / "__benchmark_missing__" / f"{tool_name}_{pick_key}.missing")

    def _mutate_path(self, *, args: Dict[str, Any], tool_name: str, source: str) -> Tuple[Dict[str, Any], bool]:
        new_args = dict(args)
        path_keys = [
            k
            for k in sorted(new_args.keys())
            if isinstance(new_args.get(k), str)
            and self._is_path_candidate(k, str(new_args.get(k)))
            and (str(k).strip().lower() not in self._LOW_IMPACT_PATH_KEYS)
        ]
        if not path_keys:
            return new_args, False

        pick_key = path_keys[0]
        before = str(new_args.get(pick_key) or "")
        modes = self.spec.profile.get("modes") if isinstance(self.spec.profile, dict) else None
        mode_list = [str(x).strip().lower() for x in (modes or []) if str(x).strip()]
        if not mode_list:
            mode_list = ["in_root_semantic", "missing", "scope_escape"]

        force_mode = ""
        if isinstance(self.spec.profile, dict):
            force_mode = str(self.spec.profile.get("force_mode") or "").strip().lower()

        # Prefer ``missing`` mode so the injected path does NOT exist on disk,
        # guaranteeing a tool error that the reflector must handle.
        # ``in_root_semantic`` (existing file, wrong modality) is kept as an
        # option but only used when explicitly forced via ``force_mode``,
        # because it causes *silent corruption* (tool succeeds with wrong
        # data) which the reflector is never invoked for — eliminating the
        # arm-differentiation signal in the ablation study.
        if force_mode and force_mode in mode_list:
            mode = force_mode
        elif "missing" in mode_list:
            mode = "missing"
        elif "in_root_semantic" in mode_list:
            mode = "in_root_semantic"
        else:
            rng = random.Random(self.spec.seed + self.target_seen_calls)
            mode = mode_list[rng.randrange(len(mode_list))]

        base_name = Path(before).name if before else f"{tool_name}_{pick_key}.nii.gz"
        if mode == "scope_escape":
            mutated = str(Path("/tmp/benchmark_scope_escape") / str(tool_name) / str(base_name))
        elif mode == "in_root_semantic":
            candidate = self._semantic_inroot_path_candidate(args=new_args, pick_key=pick_key, before=before)
            if candidate:
                mutated = candidate
            else:
                mutated = self._build_missing_path(before=before, tool_name=tool_name, pick_key=pick_key)
                mode = "missing_fallback"
        else:
            mutated = self._build_missing_path(before=before, tool_name=tool_name, pick_key=pick_key)

        new_args[pick_key] = mutated
        self._record(source=source, tool_name=tool_name, key=pick_key, before=before, after=mutated, mode=f"path_{mode}")
        return new_args, True

    def _mutate_space(self, *, args: Dict[str, Any], tool_name: str, source: str) -> Tuple[Dict[str, Any], bool]:
        new_args = dict(args)
        profile = self.spec.profile if isinstance(self.spec.profile, dict) else {}
        # Default 100× so the extreme spacing is more likely to trigger a
        # validation error inside tools that do range-checking (e.g.
        # SimpleITK resample).  The previous 4× was too mild — most tools
        # silently accept it, producing degraded-but-"successful" output
        # that the reflector was never invoked for.
        scale_factor = 100.0
        try:
            scale_factor = float(profile.get("scale_factor") or 100.0)
        except Exception:
            scale_factor = 100.0
        raw_space_keys = profile.get("space_keys")
        space_keys = [str(k).strip() for k in raw_space_keys] if isinstance(raw_space_keys, list) else []
        if not space_keys:
            space_keys = ["target_spacing", "pixel_spacing"]

        for key in space_keys:
            val = new_args.get(key)
            if isinstance(val, list) and val:
                try:
                    mutated_vals = [float(x) * scale_factor for x in val]
                    new_args[key] = mutated_vals
                    self._record(
                        source=source,
                        tool_name=tool_name,
                        key=key,
                        before=val,
                        after=mutated_vals,
                        mode="space_scale",
                    )
                    return new_args, True
                except Exception:
                    continue

        allow_interp = bool(profile.get("mutate_interpolation", True))
        interp_tools = {"register_to_reference", "resample_image"}
        if allow_interp and (tool_name in interp_tools) and isinstance(new_args.get("interpolation"), str):
            before = str(new_args.get("interpolation"))
            after = "nearest"
            if before != after:
                new_args["interpolation"] = after
                self._record(source=source, tool_name=tool_name, key="interpolation", before=before, after=after, mode="space_interp")
                return new_args, True

        return new_args, False

    # -- Tier-2-recoverable faults -----------------------------------------------

    _SWAP_PAIRS: ClassVar[Dict[str, List[Tuple[str, str]]]] = {
        "register_to_reference": [("fixed", "moving")],
        "detect_lesion_candidates": [("adc_nifti", "highb_nifti")],
        "brats_mri_segmentation": [("t1c_path", "flair_path"), ("t1_path", "t2_path")],
        "extract_roi_features": [("seg_path", "ed_seg_path")],
    }
    _SWAP_PAIR_IMPACT: ClassVar[Dict[frozenset[str], int]] = {
        frozenset({"fixed", "moving"}): 100,
        frozenset({"t1c_path", "flair_path"}): 95,
        frozenset({"adc_nifti", "highb_nifti"}): 90,
        frozenset({"t1_path", "t2_path"}): 75,
        frozenset({"seg_path", "ed_seg_path"}): 70,
    }

    def _mutate_argument_omission(
        self, *, args: Dict[str, Any], tool_name: str, source: str,
    ) -> Tuple[Dict[str, Any], bool]:
        """Drop a required key from the arguments dict.

        Profile can specify ``omit_keys`` list per tool; otherwise uses a
        heuristic that prefers path-like keys that are not the first key.
        """
        new_args = dict(args)
        profile_keys = (
            self.spec.profile.get("omit_keys")
            if isinstance(self.spec.profile, dict)
            else None
        )
        candidates: List[str] = []
        if isinstance(profile_keys, list) and profile_keys:
            candidates = [str(k) for k in profile_keys if str(k) in new_args]
        if not candidates:
            # prefer path-like required keys (skip case_state_path which is auto-filled)
            candidates = [
                k for k in sorted(new_args.keys())
                if isinstance(new_args.get(k), str)
                and k not in {"case_state_path", "state_path"}
                and self._is_path_candidate(k, str(new_args.get(k)))
            ]
        if not candidates:
            candidates = [
                k for k in sorted(new_args.keys())
                if k not in {"case_state_path", "state_path"}
            ]
        if not candidates:
            return new_args, False

        rng = random.Random(self.spec.seed + self.target_seen_calls)
        pick = candidates[rng.randrange(len(candidates))]
        before = new_args.pop(pick)
        self._record(
            source=source, tool_name=tool_name, key=pick,
            before=before, after="<OMITTED>", mode="argument_omission",
        )
        return new_args, True

    def _mutate_semantic_swap(
        self, *, args: Dict[str, Any], tool_name: str, source: str,
    ) -> Tuple[Dict[str, Any], bool]:
        """Swap two semantically-paired keys (e.g. fixed<->moving).

        Profile can specify ``swap_pairs`` list; otherwise uses built-in
        ``_SWAP_PAIRS`` lookup keyed by tool_name.
        """
        new_args = dict(args)
        raw_pairs = (
            self.spec.profile.get("swap_pairs")
            if isinstance(self.spec.profile, dict)
            else None
        )
        pairs: List[Tuple[str, str]] = []
        if isinstance(raw_pairs, list):
            for item in raw_pairs:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    a, b = str(item[0]).strip(), str(item[1]).strip()
                    if a and b and a != b:
                        pairs.append((a, b))
        if not pairs:
            pairs = list(self._SWAP_PAIRS.get(tool_name, []))

        min_impact = 70
        if isinstance(self.spec.profile, dict):
            try:
                min_impact = int(self.spec.profile.get("min_impact") or 70)
            except Exception:
                min_impact = 70

        ranked: List[Tuple[int, str, str]] = []
        for a, b in pairs:
            if a not in new_args or b not in new_args:
                continue
            before_a = new_args.get(a)
            before_b = new_args.get(b)
            if before_a == before_b:
                continue
            impact = int(self._SWAP_PAIR_IMPACT.get(frozenset({a, b}), 10))
            ranked.append((impact, a, b))

        if not ranked:
            return new_args, False

        ranked.sort(key=lambda x: (-x[0], x[1], x[2]))
        impact, a, b = ranked[0]
        if impact < min_impact:
            return new_args, False

        before_a, before_b = new_args[a], new_args[b]
        new_args[a], new_args[b] = before_b, before_a
        self._record(
            source=source,
            tool_name=tool_name,
            key=f"{a}<->{b}",
            before=f"{a}={before_a}, {b}={before_b}",
            after=f"{a}={before_b}, {b}={before_a}",
            mode="semantic_swap",
            extra={
                "swap_pair": [a, b],
                "before_map": {a: before_a, b: before_b},
                "after_map": {a: before_b, b: before_a},
                "impact": impact,
            },
        )
        return new_args, True

    # -- Tier-2-nonrecoverable faults -------------------------------------------

    def _mutate_missing_modality(
        self, *, args: Dict[str, Any], tool_name: str, source: str,
    ) -> Tuple[Dict[str, Any], bool]:
        """Replace a required modality path with a non-existent sentinel.

        Unlike ``path_mutation`` (which breaks existing paths), this simulates
        the scenario where a required imaging modality was never acquired.
        The reflector MUST halt because the data genuinely does not exist.
        """
        new_args = dict(args)
        modality_keys = [
            k for k in sorted(new_args.keys())
            if isinstance(new_args.get(k), str)
            and any(
                tag in k.lower()
                for tag in ("t2w", "adc", "dwi", "highb", "flair", "t1c", "t1", "cine")
            )
        ]
        if not modality_keys:
            modality_keys = [
                k for k in sorted(new_args.keys())
                if isinstance(new_args.get(k), str)
                and self._is_path_candidate(k, str(new_args.get(k)))
                and k not in {"case_state_path", "state_path"}
            ]
        if not modality_keys:
            return new_args, False

        rng = random.Random(self.spec.seed + self.target_seen_calls)
        pick = modality_keys[rng.randrange(len(modality_keys))]
        before = new_args[pick]
        sentinel = f"/nonexistent/missing_modality/{tool_name}/{pick}.nii.gz"
        new_args[pick] = sentinel
        self._record(
            source=source, tool_name=tool_name, key=pick,
            before=before, after=sentinel, mode="missing_modality",
        )
        return new_args, True

    def _mutate_scope_violation(
        self, *, args: Dict[str, Any], tool_name: str, source: str,
    ) -> Tuple[Dict[str, Any], bool]:
        """Inject an out-of-scope path that MUST trigger ScopeViolation.

        The reflector is expected to halt because scope-guard violations are
        non-recoverable (the agent cannot change policy).
        """
        new_args = dict(args)
        path_keys = [
            k for k in sorted(new_args.keys())
            if isinstance(new_args.get(k), str)
            and self._is_path_candidate(k, str(new_args.get(k)))
        ]
        if not path_keys:
            return new_args, False

        pick = path_keys[0]
        before = new_args[pick]
        # Use /etc/passwd as an unambiguous out-of-scope path
        new_args[pick] = "/etc/passwd"
        self._record(
            source=source, tool_name=tool_name, key=pick,
            before=before, after="/etc/passwd", mode="scope_violation",
        )
        return new_args, True

    # ---------------------------------------------------------------------------

    def maybe_mutate_arguments(
        self,
        *,
        tool_name: str,
        arguments: Dict[str, Any],
        source: str,
    ) -> Tuple[Dict[str, Any], bool]:
        args = dict(arguments or {})
        if not self.spec.enabled:
            return args, False
        if self.applied and self.spec.fault != "timeout":
            return args, False

        self.target_seen_calls += 1
        fault = str(self.spec.fault or "none").strip().lower()

        if fault == "timeout":
            self.applied = True
            self._record(
                source=source,
                tool_name=tool_name,
                key="<timeout>",
                before="dispatch",
                after="timeout",
                mode="timeout_injected",
            )
            raise TimeoutError(f"Injected timeout fault at {source} ({tool_name}).")

        if fault == "token_mutation":
            out, ok = self._mutate_token(args=args, tool_name=tool_name, source=source)
            if ok:
                self.applied = True
            return out, ok

        if fault == "path_mutation":
            out, ok = self._mutate_path(args=args, tool_name=tool_name, source=source)
            if ok:
                self.applied = True
            return out, ok

        if fault == "space_mismatch":
            out, ok = self._mutate_space(args=args, tool_name=tool_name, source=source)
            if ok:
                self.applied = True
            return out, ok

        if fault == "argument_omission":
            out, ok = self._mutate_argument_omission(args=args, tool_name=tool_name, source=source)
            if ok:
                self.applied = True
            return out, ok

        if fault == "semantic_swap":
            out, ok = self._mutate_semantic_swap(args=args, tool_name=tool_name, source=source)
            if ok:
                self.applied = True
            return out, ok

        if fault == "missing_modality":
            out, ok = self._mutate_missing_modality(args=args, tool_name=tool_name, source=source)
            if ok:
                self.applied = True
            return out, ok

        if fault == "scope_violation":
            out, ok = self._mutate_scope_violation(args=args, tool_name=tool_name, source=source)
            if ok:
                self.applied = True
            return out, ok

        return args, False

    def maybe_mutate_tool_call(self, call: ToolCall, *, source: str = "dispatcher.pre_dispatch") -> ToolCall:
        new_args, changed = self.maybe_mutate_arguments(
            tool_name=str(call.tool_name),
            arguments=dict(call.arguments or {}),
            source=source,
        )
        if not changed:
            return call
        return ToolCall(
            tool_name=call.tool_name,
            arguments=new_args,
            call_id=call.call_id,
            case_id=call.case_id,
            stage=call.stage,
            requested_by=call.requested_by,
        )


def _collect_run_artifacts_safe(*, run_dir: str, trace_path: str) -> Dict[str, Any]:
    if not str(run_dir or "").strip():
        return {
            "run_dir": "",
            "case_state_path": "",
            "execution_log_path": "",
            "trace_path": str(trace_path or ""),
            "case_state": {},
            "execution_rows": [],
            "trace_rows": [],
            "generate_report_ok": False,
        }
    return bench_v1._collect_run_artifacts(run_dir=run_dir, trace_path=trace_path)


def _stage_sort_key(k: str) -> Tuple[int, int, str]:
    try:
        return (0, int(k), "")
    except Exception:
        return (1, 10**9, str(k))


def _tool_success(case_state: Dict[str, Any], tool_name: str) -> bool:
    stage_outputs = case_state.get("stage_outputs") if isinstance(case_state.get("stage_outputs"), dict) else {}
    for _stage, tools in stage_outputs.items():
        if not isinstance(tools, dict):
            continue
        recs = tools.get(tool_name)
        if not isinstance(recs, list):
            continue
        if any(isinstance(r, dict) and (r.get("ok") is True) for r in recs):
            return True
    return False


def _latest_tool_data(case_state: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    stage_outputs = case_state.get("stage_outputs") if isinstance(case_state.get("stage_outputs"), dict) else {}
    latest: Dict[str, Any] = {}
    for stage in sorted(stage_outputs.keys(), key=_stage_sort_key):
        tools = stage_outputs.get(stage)
        if not isinstance(tools, dict):
            continue
        recs = tools.get(tool_name)
        if not isinstance(recs, list) or not recs:
            continue
        rec = recs[-1] if isinstance(recs[-1], dict) else {}
        data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
        latest = dict(data or {})
    return latest


def _coerce_path(value: Any, run_dir: Path) -> Optional[Path]:
    if isinstance(value, list):
        for item in value:
            p = _coerce_path(item, run_dir)
            if p is not None:
                return p
        return None
    if not isinstance(value, str):
        return None
    raw = str(value).strip()
    if not raw:
        return None

    p = Path(raw).expanduser()
    if p.is_absolute():
        return p

    cand1 = run_dir / raw
    cand2 = run_dir / "artifacts" / raw
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return cand1


def _resolve_tool_data_path(*, case_state: Dict[str, Any], run_dir: Path, tool: str, key: str) -> Optional[Path]:
    data = _latest_tool_data(case_state, tool)
    return _coerce_path(data.get(key), run_dir)


def _resolve_path_spec(
    *,
    spec: Dict[str, Any],
    case_state: Dict[str, Any],
    run_dir: Path,
) -> Optional[Path]:
    tool = str(spec.get("tool") or "").strip()
    key = str(spec.get("data_key") or spec.get("key") or "").strip()
    if tool and key:
        return _resolve_tool_data_path(case_state=case_state, run_dir=run_dir, tool=tool, key=key)

    run_path = str(spec.get("run_path") or spec.get("path") or "").strip()
    if run_path:
        return _coerce_path(run_path, run_dir)

    return None


def _check_required_artifact(
    *,
    spec: Dict[str, Any],
    case_state: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    art_id = str(spec.get("id") or "artifact")
    pattern = str(spec.get("pattern") or "").strip()
    if pattern:
        matches = list(run_dir.glob(pattern))
        ok = len(matches) > 0
        return {
            "id": art_id,
            "ok": ok,
            "pattern": pattern,
            "matches": [str(x) for x in matches[:8]],
        }

    path = _resolve_path_spec(spec=spec, case_state=case_state, run_dir=run_dir)
    ok = bool(path and path.exists())
    return {
        "id": art_id,
        "ok": ok,
        "path": str(path) if path else "",
    }


def _nifti_nonempty(path: Path) -> Tuple[bool, Dict[str, Any]]:
    try:
        import numpy as np
        import SimpleITK as sitk

        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)
        nz = int(np.count_nonzero(arr))
        return nz > 0, {"nonzero_voxels": nz}
    except Exception as e:
        try:
            size = int(path.stat().st_size)
            return size > 0, {"fallback": "file_size", "bytes": size, "error": type(e).__name__}
        except Exception:
            return False, {"error": type(e).__name__}


def _nifti_spacing_match(
    path: Path,
    *,
    expected_spacing: List[float],
    tol: float = 0.1,
) -> Tuple[bool, Dict[str, Any]]:
    """Check that the spacing of a NIfTI output matches expected values.

    Used as a post-execution invariant for ``space_mismatch`` fault injection:
    if the tool silently accepted the mutated spacing, the output will have
    grossly wrong spacing and this invariant will catch it.
    """
    try:
        import SimpleITK as sitk

        img = sitk.ReadImage(str(path))
        actual = list(img.GetSpacing())
        ok = len(actual) >= len(expected_spacing) and all(
            abs(float(a) - float(e)) <= tol
            for a, e in zip(actual, expected_spacing)
        )
        return ok, {
            "actual_spacing": actual,
            "expected_spacing": expected_spacing,
            "tolerance": tol,
        }
    except Exception as e:
        return False, {"error": type(e).__name__}


def _nifti_affine_match(lhs: Path, rhs: Path, tol: float = 1e-3) -> Tuple[bool, Dict[str, Any]]:
    try:
        import SimpleITK as sitk

        a = sitk.ReadImage(str(lhs))
        b = sitk.ReadImage(str(rhs))
        same_size = list(a.GetSize()) == list(b.GetSize())
        same_spacing = all(abs(float(x) - float(y)) <= tol for x, y in zip(a.GetSpacing(), b.GetSpacing()))
        same_origin = all(abs(float(x) - float(y)) <= tol for x, y in zip(a.GetOrigin(), b.GetOrigin()))
        same_direction = all(abs(float(x) - float(y)) <= tol for x, y in zip(a.GetDirection(), b.GetDirection()))
        ok = bool(same_size and same_spacing and same_origin and same_direction)
        return ok, {
            "same_size": same_size,
            "same_spacing": same_spacing,
            "same_origin": same_origin,
            "same_direction": same_direction,
        }
    except Exception as e:
        return False, {"error": type(e).__name__}


def _csv_non_empty(path: Path, min_rows: int) -> Tuple[bool, Dict[str, Any]]:
    rows = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for _ in reader:
                rows += 1
                if rows >= min_rows:
                    break
        return rows >= min_rows, {"rows": rows, "min_rows": min_rows}
    except Exception as e:
        return False, {"rows": rows, "min_rows": min_rows, "error": type(e).__name__}


def _json_field(obj: Any, field: str) -> Any:
    cur = obj
    for part in str(field).split("."):
        key = str(part).strip()
        if not key:
            return None
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _evaluate_invariant(
    *,
    spec: Dict[str, Any],
    case_state: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    inv_id = str(spec.get("id") or "invariant")
    inv_type = str(spec.get("type") or "").strip().lower()

    if inv_type == "tool_success":
        tool = str(spec.get("tool") or "")
        ok = _tool_success(case_state, tool)
        return {"id": inv_id, "type": inv_type, "ok": ok, "tool": tool}

    if inv_type == "path_exists":
        path = _resolve_path_spec(spec=spec, case_state=case_state, run_dir=run_dir)
        ok = bool(path and path.exists())
        return {"id": inv_id, "type": inv_type, "ok": ok, "path": str(path) if path else ""}

    if inv_type == "nifti_nonempty":
        path = _resolve_path_spec(spec=spec, case_state=case_state, run_dir=run_dir)
        if not path or not path.exists():
            return {"id": inv_id, "type": inv_type, "ok": False, "path": str(path) if path else "", "detail": "missing_path"}
        ok, info = _nifti_nonempty(path)
        return {"id": inv_id, "type": inv_type, "ok": ok, "path": str(path), "detail": info}

    if inv_type == "nifti_affine_match":
        lhs = spec.get("lhs") if isinstance(spec.get("lhs"), dict) else {}
        rhs = spec.get("rhs") if isinstance(spec.get("rhs"), dict) else {}
        lhs_path = _resolve_path_spec(spec=lhs, case_state=case_state, run_dir=run_dir)
        rhs_path = _resolve_path_spec(spec=rhs, case_state=case_state, run_dir=run_dir)
        if not lhs_path or not rhs_path or (not lhs_path.exists()) or (not rhs_path.exists()):
            return {
                "id": inv_id,
                "type": inv_type,
                "ok": False,
                "lhs": str(lhs_path) if lhs_path else "",
                "rhs": str(rhs_path) if rhs_path else "",
                "detail": "missing_path",
            }
        ok, info = _nifti_affine_match(lhs_path, rhs_path)
        return {
            "id": inv_id,
            "type": inv_type,
            "ok": ok,
            "lhs": str(lhs_path),
            "rhs": str(rhs_path),
            "detail": info,
        }

    if inv_type == "csv_non_empty":
        path = _resolve_path_spec(spec=spec, case_state=case_state, run_dir=run_dir)
        min_rows = int(spec.get("min_rows") or 1)
        if not path or not path.exists():
            return {"id": inv_id, "type": inv_type, "ok": False, "path": str(path) if path else "", "detail": "missing_path"}
        ok, info = _csv_non_empty(path, min_rows)
        return {"id": inv_id, "type": inv_type, "ok": ok, "path": str(path), "detail": info}

    if inv_type == "json_non_empty":
        path = _resolve_path_spec(spec=spec, case_state=case_state, run_dir=run_dir)
        if not path or not path.exists():
            return {"id": inv_id, "type": inv_type, "ok": False, "path": str(path) if path else "", "detail": "missing_path"}
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            ok = bool(obj)
            return {"id": inv_id, "type": inv_type, "ok": ok, "path": str(path)}
        except Exception as e:
            return {"id": inv_id, "type": inv_type, "ok": False, "path": str(path), "detail": type(e).__name__}

    if inv_type == "json_field_non_empty":
        path = _resolve_path_spec(spec=spec, case_state=case_state, run_dir=run_dir)
        field = str(spec.get("json_field") or "")
        if not path or not path.exists() or not field:
            return {
                "id": inv_id,
                "type": inv_type,
                "ok": False,
                "path": str(path) if path else "",
                "field": field,
                "detail": "missing_path_or_field",
            }
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            val = _json_field(obj, field)
            ok = val not in (None, "", [], {})
            return {"id": inv_id, "type": inv_type, "ok": ok, "path": str(path), "field": field, "value": val}
        except Exception as e:
            return {"id": inv_id, "type": inv_type, "ok": False, "path": str(path), "field": field, "detail": type(e).__name__}

    if inv_type == "nifti_spacing_match":
        path = _resolve_path_spec(spec=spec, case_state=case_state, run_dir=run_dir)
        if not path or not path.exists():
            return {"id": inv_id, "type": inv_type, "ok": False, "path": str(path) if path else "", "detail": "missing_path"}
        expected = spec.get("expected_spacing")
        tol = float(spec.get("tolerance") or 0.1)
        if not isinstance(expected, list) or not expected:
            return {"id": inv_id, "type": inv_type, "ok": False, "path": str(path), "detail": "no_expected_spacing"}
        ok, info = _nifti_spacing_match(path, expected_spacing=expected, tol=tol)
        return {"id": inv_id, "type": inv_type, "ok": ok, "path": str(path), "detail": info}

    if inv_type == "candidates_json_valid":
        path = _resolve_path_spec(spec=spec, case_state=case_state, run_dir=run_dir)
        if not path or not path.exists():
            return {"id": inv_id, "type": inv_type, "ok": False, "path": str(path) if path else "", "detail": "missing_path"}
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                return {"id": inv_id, "type": inv_type, "ok": False, "path": str(path), "detail": "not_a_json_object"}
            has_num = "num_candidates" in obj
            num_val = obj.get("num_candidates")
            num_ok = isinstance(num_val, int) and num_val >= 0
            has_candidates = isinstance(obj.get("candidates"), list)
            ok = bool(has_num and num_ok and has_candidates)
            return {
                "id": inv_id,
                "type": inv_type,
                "ok": ok,
                "path": str(path),
                "detail": {
                    "has_num_candidates": has_num,
                    "num_candidates": num_val,
                    "num_ok": num_ok,
                    "has_candidates_list": has_candidates,
                    "n_candidates": len(obj.get("candidates", [])) if has_candidates else 0,
                },
            }
        except Exception as e:
            return {"id": inv_id, "type": inv_type, "ok": False, "path": str(path), "detail": type(e).__name__}

    return {"id": inv_id, "type": inv_type, "ok": False, "detail": "unsupported_invariant_type"}


def _evaluate_success_rule(
    *,
    rule: Any,
    case_state: Dict[str, Any],
    run_dir: Path,
    invariants_by_id: Dict[str, bool],
) -> Tuple[bool, Any]:
    if isinstance(rule, bool):
        return rule, {"literal": rule}

    if isinstance(rule, dict):
        if isinstance(rule.get("all"), list):
            details = []
            ok = True
            for sub in rule["all"]:
                sub_ok, sub_det = _evaluate_success_rule(
                    rule=sub,
                    case_state=case_state,
                    run_dir=run_dir,
                    invariants_by_id=invariants_by_id,
                )
                details.append({"ok": sub_ok, "detail": sub_det})
                if not sub_ok:
                    ok = False
            return ok, {"all": details}

        if isinstance(rule.get("any"), list):
            details = []
            any_ok = False
            for sub in rule["any"]:
                sub_ok, sub_det = _evaluate_success_rule(
                    rule=sub,
                    case_state=case_state,
                    run_dir=run_dir,
                    invariants_by_id=invariants_by_id,
                )
                details.append({"ok": sub_ok, "detail": sub_det})
                any_ok = any_ok or sub_ok
            return any_ok, {"any": details}

        rtype = str(rule.get("type") or "").strip().lower()
        if rtype == "tool_success":
            tool = str(rule.get("tool") or "")
            ok = _tool_success(case_state, tool)
            return ok, {"type": rtype, "tool": tool}

        if rtype == "artifact_exists":
            rec = _check_required_artifact(spec=rule, case_state=case_state, run_dir=run_dir)
            return bool(rec.get("ok")), rec

        if rtype == "invariant_pass":
            inv_id = str(rule.get("id") or "")
            ok = bool(invariants_by_id.get(inv_id, False))
            return ok, {"type": rtype, "id": inv_id}

        return False, {"type": rtype, "detail": "unsupported_success_rule"}

    return False, {"detail": "unsupported_success_criteria"}


def _compute_tcr(
    *,
    contract: Dict[str, Any],
    case_state: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    required_tools = [str(x) for x in (contract.get("required_stage_success") or []) if str(x).strip()]
    required_artifacts = [x for x in (contract.get("required_artifacts") or []) if isinstance(x, dict)]

    tool_checks = [{"tool": tool, "ok": _tool_success(case_state, tool)} for tool in required_tools]
    artifact_checks = [
        _check_required_artifact(spec=spec, case_state=case_state, run_dir=run_dir)
        for spec in required_artifacts
    ]

    completed = sum(1 for r in tool_checks if r.get("ok")) + sum(1 for r in artifact_checks if r.get("ok"))
    total = len(tool_checks) + len(artifact_checks)
    ratio = (float(completed) / float(total)) if total > 0 else 1.0

    return {
        "completed": completed,
        "total": total,
        "ratio": ratio,
        "pass": ratio >= 1.0,
        "required_stage_success": tool_checks,
        "required_artifacts": artifact_checks,
    }


def _fault_profile_status(*, contract: Dict[str, Any], fault: str) -> Dict[str, Any]:
    fault_norm = str(fault or "none").strip().lower()
    if fault_norm == "none":
        return {
            "profile": {},
            "profile_enabled": None,
            "not_applicable": False,
            "not_applicable_reason": "",
        }

    fault_profiles = contract.get("fault_profiles") if isinstance(contract.get("fault_profiles"), dict) else {}
    raw_profile = fault_profiles.get(fault_norm)
    profile = dict(raw_profile) if isinstance(raw_profile, dict) else {}
    has_enabled = isinstance(raw_profile, dict) and ("enabled" in raw_profile)
    profile_enabled = bool(raw_profile.get("enabled")) if has_enabled else None

    not_applicable = False
    reason = ""
    if profile_enabled is False:
        not_applicable = True
        reason = "fault_profile_disabled"
    elif profile_enabled is None:
        not_applicable = True
        reason = "fault_profile_missing_enabled"

    return {
        "profile": profile,
        "profile_enabled": profile_enabled,
        "not_applicable": not_applicable,
        "not_applicable_reason": reason,
    }


def _compute_err(*, fault: str, injector: FaultInjectorV2, success: bool, not_applicable: bool) -> Dict[str, Any]:
    fault_norm = str(fault or "none").strip().lower()
    # Nonrecoverable faults are scored via safe_halt, not ERR.
    nonrecoverable_faults = {"missing_modality", "scope_violation", "timeout"}
    eligible = (fault_norm != "none") and (not bool(not_applicable)) and (fault_norm not in nonrecoverable_faults)
    recovered = bool(eligible and injector.applied and success)
    return {
        "eligible": eligible,
        "fault": str(fault),
        "not_applicable": bool(not_applicable),
        "fault_applied": bool(injector.applied),
        "recovered": recovered,
        "value": (1.0 if recovered else 0.0) if eligible else None,
    }


def _compute_safe_halt(
    *,
    fault: str,
    injector: FaultInjectorV2,
    success: bool,
    actual_status: str,
    observed_error_types: List[str],
    reflection_decisions: List[Dict[str, Any]],
    not_applicable: bool,
) -> Dict[str, Any]:
    fault_norm = str(fault or "none").strip().lower()
    nonrecoverable_faults = {"missing_modality", "scope_violation", "timeout"}
    eligible = (fault_norm in nonrecoverable_faults) and (not bool(not_applicable)) and bool(injector.applied)

    observed = {str(x or "").strip().lower() for x in observed_error_types if str(x or "").strip()}
    reflection_halt = any(
        isinstance(r, dict) and str(r.get("action") or "").strip().lower() == "halt"
        for r in reflection_decisions
    )
    expected_error = False
    if fault_norm == "scope_violation":
        expected_error = any("scopeviolation" in e for e in observed)
    elif fault_norm == "missing_modality":
        expected_error = any(("filenotfound" in e) or ("missing" in e) for e in observed)
    elif fault_norm == "timeout":
        expected_error = any(("timeout" in e) for e in observed)

    actual_failure = str(actual_status or "").strip().lower() != "success"
    safe_halt_ok = bool(eligible and actual_failure and (not success) and (reflection_halt or expected_error))
    return {
        "eligible": bool(eligible),
        "fault": fault_norm,
        "not_applicable": bool(not_applicable),
        "fault_applied": bool(injector.applied),
        "reflection_halt": bool(reflection_halt),
        "expected_error_observed": bool(expected_error),
        "pass": bool(safe_halt_ok),
        "value": (1.0 if safe_halt_ok else 0.0) if eligible else None,
    }


def _parse_semantic_map(raw: Any, *, a: str, b: str) -> Dict[str, str]:
    if isinstance(raw, dict):
        va = raw.get(a)
        vb = raw.get(b)
        if isinstance(va, str) and isinstance(vb, str):
            return {a: va, b: vb}
    if isinstance(raw, str):
        pfx = f"{a}="
        mid = f", {b}="
        if raw.startswith(pfx) and (mid in raw):
            idx = raw.find(mid)
            va = raw[len(pfx):idx]
            vb = raw[idx + len(mid):]
            return {a: va, b: vb}
    return {}


def _compute_semantic_guard(
    *,
    fault: str,
    injector: FaultInjectorV2,
    execution_rows: List[Dict[str, Any]],
    not_applicable: bool,
) -> Dict[str, Any]:
    fault_norm = str(fault or "none").strip().lower()
    eligible = (fault_norm == "semantic_swap") and (not bool(not_applicable)) and bool(injector.applied)
    if not eligible:
        return {
            "eligible": False,
            "fault": fault_norm,
            "not_applicable": bool(not_applicable),
            "fault_applied": bool(injector.applied),
            "pass": None,
            "value": None,
            "details": [],
        }

    ok_exec_rows = [
        row for row in execution_rows
        if isinstance(row, dict) and row.get("ok") is True and isinstance(row.get("arguments"), dict)
    ]
    events = [
        ev for ev in injector.events
        if isinstance(ev, dict) and str(ev.get("mode") or "").strip().lower() == "semantic_swap"
    ]

    all_pass = True
    details: List[Dict[str, Any]] = []
    for ev in events:
        tool_name = str(ev.get("tool_name") or "").strip()
        pair = ev.get("swap_pair")
        if not (isinstance(pair, list) and len(pair) == 2):
            key = str(ev.get("key") or "")
            if "<->" in key:
                left, right = key.split("<->", 1)
                pair = [left.strip(), right.strip()]
        if not (isinstance(pair, list) and len(pair) == 2):
            all_pass = False
            details.append({"tool_name": tool_name, "status": "unparseable_swap_pair"})
            continue

        a, b = str(pair[0]), str(pair[1])
        before_map = _parse_semantic_map(ev.get("before_map"), a=a, b=b)
        if not before_map:
            before_map = _parse_semantic_map(ev.get("from"), a=a, b=b)
        after_map = _parse_semantic_map(ev.get("after_map"), a=a, b=b)
        if not after_map:
            after_map = _parse_semantic_map(ev.get("to"), a=a, b=b)

        corrected = False
        swapped_seen = False
        for row in ok_exec_rows:
            if str(row.get("tool_name") or "").strip() != tool_name:
                continue
            args = row.get("arguments") if isinstance(row.get("arguments"), dict) else {}
            if (a not in args) or (b not in args):
                continue
            av = str(args.get(a))
            bv = str(args.get(b))
            if after_map and (av == str(after_map.get(a))) and (bv == str(after_map.get(b))):
                swapped_seen = True
            if before_map and (av == str(before_map.get(a))) and (bv == str(before_map.get(b))):
                corrected = True
                break

        event_pass = bool(corrected)
        all_pass = all_pass and event_pass
        details.append(
            {
                "tool_name": tool_name,
                "pair": [a, b],
                "before_map": before_map,
                "after_map": after_map,
                "swapped_seen": swapped_seen,
                "corrected": corrected,
                "pass": event_pass,
            }
        )

    return {
        "eligible": True,
        "fault": fault_norm,
        "not_applicable": bool(not_applicable),
        "fault_applied": bool(injector.applied),
        "pass": bool(all_pass),
        "value": 1.0 if all_pass else 0.0,
        "details": details,
    }


def _semantic_candidate_repairs(*, tool_name: str, a: str, b: str) -> List[Dict[str, Any]]:
    """Return candidate semantic repairs without committing to a correction.

    This is intentionally a *detector* helper, not an auto-fixer. The output is
    a small candidate space that downstream logic (human audit, semantic
    reflector, or future online semantic lint) can reason over.
    """
    candidates: List[Dict[str, Any]] = [
        {
            "op": "swap_pair",
            "pair": [a, b],
            "confidence": "high",
            "reason": "Injected fault profile indicates these two arguments were swapped.",
        }
    ]
    # Domain/tool-specific alternates can be proposed as lower-confidence options.
    t = str(tool_name or "").strip()
    if t == "register_to_reference" and {a, b} == {"fixed", "moving"}:
        candidates.append(
            {
                "op": "preserve_current",
                "pair": [a, b],
                "confidence": "low",
                "reason": "Registration may still be executable even with roles reversed; requires semantic role evidence.",
            }
        )
    elif t == "brats_mri_segmentation" and {a, b} == {"t1c_path", "flair_path"}:
        candidates.append(
            {
                "op": "preserve_current",
                "pair": [a, b],
                "confidence": "low",
                "reason": "Both paths exist and are schema-valid; modality-role mismatch is semantic, not syntactic.",
            }
        )
    return candidates


def _semantic_evidence_type(*, tool_name: str, a: str, b: str) -> str:
    t = str(tool_name or "").strip()
    pair = {a, b}
    if t == "register_to_reference" and pair == {"fixed", "moving"}:
        return "registration_role_swap"
    if t == "brats_mri_segmentation" and pair == {"t1c_path", "flair_path"}:
        return "modality_role_swap"
    return "semantic_pair_swap"


def _compute_semantic_detector(
    *,
    fault: str,
    injector: FaultInjectorV2,
    execution_rows: List[Dict[str, Any]],
    reflection_decisions: List[Dict[str, Any]],
    not_applicable: bool,
) -> Dict[str, Any]:
    """Post-hoc semantic detector that emits structured posterior evidence.

    This does not change runtime behavior. It is intended for experiment
    analysis to separate:
    - fault injected
    - fault detectable (with posterior evidence)
    - fault corrected
    """
    fault_norm = str(fault or "none").strip().lower()
    eligible = (fault_norm == "semantic_swap") and (not bool(not_applicable)) and bool(injector.applied)
    if not eligible:
        return {
            "eligible": False,
            "fault": fault_norm,
            "not_applicable": bool(not_applicable),
            "fault_applied": bool(injector.applied),
            "detected": None,
            "online_triggered": None,
            "details": [],
        }

    ok_exec_rows = [
        row for row in execution_rows
        if isinstance(row, dict) and row.get("ok") is True and isinstance(row.get("arguments"), dict)
    ]
    events = [
        ev for ev in injector.events
        if isinstance(ev, dict) and str(ev.get("mode") or "").strip().lower() == "semantic_swap"
    ]
    # In current architecture, semantic_swap often stays latent until benchmark guard.
    online_triggered = bool(reflection_decisions)  # proxy: any reflection happened in run

    details: List[Dict[str, Any]] = []
    any_detected = False
    for ev in events:
        tool_name = str(ev.get("tool_name") or "").strip()
        pair = ev.get("swap_pair")
        if not (isinstance(pair, list) and len(pair) == 2):
            key = str(ev.get("key") or "")
            if "<->" in key:
                left, right = key.split("<->", 1)
                pair = [left.strip(), right.strip()]
        if not (isinstance(pair, list) and len(pair) == 2):
            details.append(
                {
                    "tool_name": tool_name,
                    "status": "unparseable_swap_pair",
                    "detected": False,
                    "candidate_repairs": [],
                    "evidence_type": "unknown",
                }
            )
            continue
        a, b = str(pair[0]), str(pair[1])
        before_map = _parse_semantic_map(ev.get("before_map"), a=a, b=b) or _parse_semantic_map(ev.get("from"), a=a, b=b)
        after_map = _parse_semantic_map(ev.get("after_map"), a=a, b=b) or _parse_semantic_map(ev.get("to"), a=a, b=b)
        current_map: Dict[str, str] = {}
        corrected = False
        swapped_seen = False
        matched_rows = 0
        for row in ok_exec_rows:
            if str(row.get("tool_name") or "").strip() != tool_name:
                continue
            args = row.get("arguments") if isinstance(row.get("arguments"), dict) else {}
            if (a not in args) or (b not in args):
                continue
            matched_rows += 1
            av = str(args.get(a))
            bv = str(args.get(b))
            current_map = {a: av, b: bv}
            if after_map and (av == str(after_map.get(a))) and (bv == str(after_map.get(b))):
                swapped_seen = True
            if before_map and (av == str(before_map.get(a))) and (bv == str(before_map.get(b))):
                corrected = True
                break

        detected = bool(swapped_seen or corrected)
        any_detected = any_detected or detected
        details.append(
            {
                "tool_name": tool_name,
                "pair": [a, b],
                "evidence_type": _semantic_evidence_type(tool_name=tool_name, a=a, b=b),
                "detected": detected,
                "swapped_seen": swapped_seen,
                "corrected": corrected,
                "matched_exec_rows": matched_rows,
                "before_map": before_map,
                "after_map": after_map,
                "current_map": current_map,
                "candidate_repairs": _semantic_candidate_repairs(tool_name=tool_name, a=a, b=b),
                "upstream_context": {
                    "fault_event_source": ev.get("source"),
                    "impact": ev.get("impact"),
                    "fault_mode": ev.get("mode"),
                },
                "notes": (
                    "Post-hoc detector evidence (benchmark-time). Current runtime reflector is usually not "
                    "triggered for semantic swaps unless a tool emits a runtime error or an online semantic lint exists."
                ),
            }
        )

    return {
        "eligible": True,
        "fault": fault_norm,
        "not_applicable": bool(not_applicable),
        "fault_applied": bool(injector.applied),
        "detected": bool(any_detected),
        "online_triggered": bool(online_triggered),
        "details": details,
    }


def _collect_success_tools(case_state: Dict[str, Any]) -> List[str]:
    stage_outputs = case_state.get("stage_outputs") if isinstance(case_state.get("stage_outputs"), dict) else {}
    seen: set[str] = set()
    ordered: List[str] = []
    for stage in sorted(stage_outputs.keys(), key=_stage_sort_key):
        tools = stage_outputs.get(stage)
        if not isinstance(tools, dict):
            continue
        for tool_name, recs in tools.items():
            if not isinstance(recs, list):
                continue
            if any(isinstance(r, dict) and r.get("ok") is True for r in recs):
                if tool_name not in seen:
                    seen.add(tool_name)
                    ordered.append(str(tool_name))
    return ordered


def _api_key_for_provider(provider: str) -> str:
    p = str(provider or "openai_compatible_server").strip().lower()
    if p in {"openai_compatible_server", "server"}:
        return str(os.environ.get("MRI_AGENT_SHELL_SERVER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY")
    if p in {"openai_official", "openai"}:
        return str(os.environ.get("OPENAI_API_KEY") or "EMPTY")
    if p == "gemini":
        return str(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "EMPTY")
    if p == "anthropic":
        return str(os.environ.get("ANTHROPIC_API_KEY") or "EMPTY")
    return "EMPTY"


def _effective_max_new_tokens(*, provider: str, model: str, requested: int) -> Tuple[int, Optional[str]]:
    req = max(64, int(requested or 2048))
    p = str(provider or "").strip().lower()
    m = str(model or "").strip().lower()
    # GPT-5 reasoning-family models frequently need larger completion budgets than our
    # local-vLLM default (2048), otherwise they may exhaust tokens before final JSON/tool call.
    if p in {"openai_official", "openai"} and m.startswith("gpt-5") and req <= 2048:
        bumped = 8192
        return bumped, f"auto_bump_gpt5_reasoning_budget({req}->{bumped})"
    return req, None


def _build_backend(
    *,
    provider: str,
    server_base_url: str,
    api_base_url: str,
    server_model: str,
    api_key: str,
) -> bench_v1.BackendConfig:
    provider_norm = str(provider or "openai_compatible_server").strip().lower() or "openai_compatible_server"
    base_url = str(server_base_url or "").strip()
    if provider_norm not in {"openai_compatible_server", "server"}:
        base_url = str(api_base_url or "").strip()
    return bench_v1.BackendConfig(
        backend_id=(f"{provider_norm}:{str(server_model or '').strip() or 'unknown'}"),
        provider=provider_norm,
        model=server_model,
        base_url=base_url,
        api_key=(api_key if api_key else "EMPTY"),
    )


def _run_react_mode(
    *,
    case_obj: Dict[str, Any],
    case_id: str,
    backend: bench_v1.BackendConfig,
    runs_root: Path,
    max_new_tokens: int,
    max_steps: int,
    max_retries: int,
    injector: FaultInjectorV2,
    token_mode: bool,
    failure_reflector: bool = False,
    contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    case_ref = str(case_obj.get("case_ref") or "").strip()
    domain = str(case_obj.get("domain") or "").strip().lower()
    goal = str(case_obj.get("prompt") or "")

    # --- Inject required tool list from the task contract into the goal ---
    contract = contract or {}
    request_type = str(contract.get("request_type") or "").strip()
    required_tools: List[str] = [
        str(t) for t in (contract.get("required_stage_success") or [])
        if str(t).strip()
    ]
    if required_tools:
        tools_str = " -> ".join(required_tools)
        goal += f"\nRequired tool sequence: {tools_str}. Call them in order."

    if token_mode:
        goal = (
            goal
            + "\nSymbolic reference mode: use @token values (for example @runtime.case_state_path, @case.input, @seq.T2w/@seq.T1c/@seq.CINE) "
            + "for path fields when available, but you MUST still provide all required arguments for each tool call."
            + "\nNever send empty arguments for tools that require inputs."
        )

    # --- Build a task-aware system prompt (no full-pipeline checklist leak) ---
    reactive_prompt = build_reactive_system_prompt(
        request_type=request_type,
        required_tools=required_tools or None,
    )

    llm_kwargs = bench_v1._llm_invoke_kwargs_from_backend(backend, max_tokens=max_new_tokens)

    with bench_v1.patched_dispatch(injector):
        enable_preconditions_token = token_mode and request_type in {
            "full_pipeline",
            "classify",
            "register",
        }
        run_dir = run_agent_loop(
            goal=goal,
            case_id=case_id,
            dicom_case_dir=case_ref,
            runs_root=Path(runs_root),
            llm_mode=str(llm_kwargs.get("llm_mode") or "server"),
            max_steps=int(max_steps),
            max_retries=int(max_retries),
            plan_mode="step",
            server_cfg=llm_kwargs.get("server_cfg"),
            api_model=llm_kwargs.get("api_model"),
            api_base_url=llm_kwargs.get("api_base_url"),
            finalize_with_llm=False,
            enforce_mvp_pipeline=False,
            autofix_mode="off",
            symbolic_binder_mode=("token" if token_mode else "off"),
            # Keep token-mode preconditions only for tasks that truly require
            # multi-step dependency recovery. For short single-tool tasks
            # (e.g. super_resolution/raw_recon), precondition replacement can
            # hijack the plan into unrelated long pipelines.
            enable_preconditions=enable_preconditions_token,
            enable_tool_reflection=False,
            enable_failure_reflector=bool(failure_reflector),
            # Disable free-form shell exploration in benchmark runs to reduce
            # model-specific workflow drift (e.g., repeated ls/find detours).
            enable_sandbox_exec=False,
            domain=get_domain_config(domain),
            reactive_prompt_override=reactive_prompt,
        )

    run_dir_path = Path(run_dir)
    trace_path = run_dir_path / "agent_trace.jsonl"
    reflection_decisions: List[Dict[str, Any]] = []
    for row in _read_jsonl(trace_path):
        if str(row.get("tag") or "") != "failure_reflection":
            continue
        dec = row.get("decision")
        if isinstance(dec, dict):
            reflection_decisions.append(dict(dec))
    return {
        "run_result": {"ok": None},
        "run_dir": str(run_dir_path),
        "trace_path": str(trace_path),
        "reflection_decisions": reflection_decisions,
        "planner_status": (
            "reactive_token_reflector"
            if (token_mode and failure_reflector)
            else ("reactive_token" if token_mode else "reactive")
        ),
    }


def _ensure_output_run_dir(*, run_dir: str, runs_root: Path, case_id: str) -> Path:
    if str(run_dir or "").strip():
        p = Path(run_dir).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        return p

    fallback = runs_root / case_id / ("benchmark_v2_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _cleanup_run_dir(run_dir: str) -> None:
    """Delete a run directory tree to reclaim disk space.

    Called when ``--cleanup-runs=1``.  All scoring data has already been
    captured in the in-memory metrics dict (and will be written to the
    summary JSON), so the NIfTI artefacts, execution logs, and
    case-state files are no longer needed.
    """
    p = Path(run_dir)
    if not p.exists():
        return
    try:
        shutil.rmtree(p)
        # Also remove the parent case-id directory if it is now empty.
        parent = p.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
        print(f"  [cleanup] removed {p}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [cleanup] WARNING: failed to remove {p}: {exc}")


def _run_single_case(
    *,
    case: Dict[str, Any],
    task_id: str,
    contract: Dict[str, Any],
    arm: str,
    fault: str,
    backend: bench_v1.BackendConfig,
    workspace_root: Path,
    runs_root: Path,
    max_new_tokens: int,
    max_steps: int,
    max_retries: int,
) -> Dict[str, Any]:
    goal = _render_goal(
        str(contract.get("goal_template") or ""),
        task_id=task_id,
        case=case,
        request_type=str(contract.get("request_type") or "full_pipeline"),
    )

    case_obj = {
        "id": case["case_id"],
        "task_id": task_id,
        "domain": case["domain"],
        "case_ref": case["case_root"],
        "prompt": goal,
        "request_type": str(contract.get("request_type") or "full_pipeline"),
    }
    mode_map = {
        "bcer": "bcr_sketch",
        "bcer_sketch": "bcr_sketch",
    }

    run_case_id = bench_v1._sanitize_case_id(f"{case['case_id']}__{task_id}__{arm}__{fault}")
    seed_src = f"{run_case_id}|{fault}|{task_id}|{arm}"
    seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)

    fault_profile = _fault_profile_status(contract=contract, fault=fault)
    profile = fault_profile.get("profile") if isinstance(fault_profile.get("profile"), dict) else {}
    profile_enabled = fault_profile.get("profile_enabled")
    not_applicable = bool(fault_profile.get("not_applicable"))
    not_applicable_reason = str(fault_profile.get("not_applicable_reason") or "")
    fault_requested = str(fault or "none").strip().lower() != "none"

    injector = FaultInjectorV2(
        spec=FaultInjectionSpecV2(
            enabled=bool(fault_requested and not not_applicable),
            fault=str(fault),
            seed=seed,
            profile=profile,
        )
    )

    run_payload: Dict[str, Any] = {}
    raised_error: Optional[Dict[str, Any]] = None

    try:
        if arm in mode_map:
            run_payload = bench_v1._run_cerebellum_mode(
                case_obj=case_obj,
                case_id=run_case_id,
                ablation_mode=mode_map[arm],
                backend=backend,
                workspace_root=workspace_root,
                runs_root=runs_root,
                max_new_tokens=max_new_tokens,
                injector=injector,
            )
        elif arm == "react":
            run_payload = _run_react_mode(
                case_obj=case_obj,
                case_id=run_case_id,
                backend=backend,
                runs_root=runs_root,
                max_new_tokens=max_new_tokens,
                max_steps=max_steps,
                max_retries=max_retries,
                injector=injector,
                token_mode=False,
                contract=contract,
            )
        elif arm == "react_token":
            run_payload = _run_react_mode(
                case_obj=case_obj,
                case_id=run_case_id,
                backend=backend,
                runs_root=runs_root,
                max_new_tokens=max_new_tokens,
                max_steps=max_steps,
                max_retries=max_retries,
                injector=injector,
                token_mode=True,
                failure_reflector=False,
                contract=contract,
            )
        elif arm == "react_token_reflector":
            run_payload = _run_react_mode(
                case_obj=case_obj,
                case_id=run_case_id,
                backend=backend,
                runs_root=runs_root,
                max_new_tokens=max_new_tokens,
                max_steps=max_steps,
                max_retries=max_retries,
                injector=injector,
                token_mode=True,
                failure_reflector=True,
                contract=contract,
            )
        else:
            raise ValueError(f"Unsupported arm: {arm}")
    except Exception as e:
        err_tb = traceback.format_exc(limit=80)
        run_dir_hint = bench_v1._extract_run_dir_hint(message=str(e), traceback_text=err_tb)
        if run_dir_hint:
            run_payload["run_dir"] = run_dir_hint
            if arm in {"react", "react_token", "react_token_reflector"}:
                run_payload["trace_path"] = str(Path(run_dir_hint) / "agent_trace.jsonl")
        raised_error = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": err_tb,
        }

    run_dir_s = str(run_payload.get("run_dir") or "")
    trace_path_s = str(run_payload.get("trace_path") or "")
    artifacts = _collect_run_artifacts_safe(run_dir=run_dir_s, trace_path=trace_path_s)

    observed_errors = bench_v1._collect_error_types(
        execution_rows=artifacts.get("execution_rows") if isinstance(artifacts.get("execution_rows"), list) else [],
        trace_rows=artifacts.get("trace_rows") if isinstance(artifacts.get("trace_rows"), list) else [],
        raised_error=raised_error,
    )

    ablation_mode = mode_map.get(arm, "pure_react")
    actual_status = bench_v1._derive_actual_status(
        ablation_mode=ablation_mode,
        run_payload=run_payload,
        artifacts=artifacts,
        raised_error=raised_error,
    )

    run_dir_path = _ensure_output_run_dir(run_dir=str(artifacts.get("run_dir") or run_dir_s), runs_root=runs_root, case_id=run_case_id)
    case_state = artifacts.get("case_state") if isinstance(artifacts.get("case_state"), dict) else {}

    invariant_specs = [x for x in (contract.get("invariants") or []) if isinstance(x, dict)]
    invariants = [
        _evaluate_invariant(spec=inv, case_state=case_state, run_dir=run_dir_path)
        for inv in invariant_specs
    ]
    invariants_by_id = {str(x.get("id")): bool(x.get("ok")) for x in invariants}
    invariant_total = len(invariants)
    invariant_passed = sum(1 for x in invariants if bool(x.get("ok")))
    invariant_evaluable = invariant_total > 0
    invariant_pass = bool(invariant_evaluable and invariant_passed == invariant_total)

    success_ok, success_detail = _evaluate_success_rule(
        rule=contract.get("success_criteria") if isinstance(contract.get("success_criteria"), dict) else {},
        case_state=case_state,
        run_dir=run_dir_path,
        invariants_by_id=invariants_by_id,
    )
    semantic_guard = _compute_semantic_guard(
        fault=fault,
        injector=injector,
        execution_rows=artifacts.get("execution_rows") if isinstance(artifacts.get("execution_rows"), list) else [],
        not_applicable=not_applicable,
    )
    if semantic_guard.get("eligible") and semantic_guard.get("pass") is False:
        success_ok = False
        success_detail = {
            "base_success_detail": success_detail,
            "semantic_guard": semantic_guard,
            "reason": "semantic_swap_not_corrected",
        }

    tcr = _compute_tcr(contract=contract, case_state=case_state, run_dir=run_dir_path)
    reflection_decisions = (
        run_payload.get("reflection_decisions")
        if isinstance(run_payload.get("reflection_decisions"), list)
        else []
    )
    if not reflection_decisions:
        trace_rows = artifacts.get("trace_rows") if isinstance(artifacts.get("trace_rows"), list) else []
        for row in trace_rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("tag") or "").strip() != "failure_reflection":
                continue
            dec = row.get("decision")
            if isinstance(dec, dict):
                reflection_decisions.append(dict(dec))

    semantic_detector = _compute_semantic_detector(
        fault=fault,
        injector=injector,
        execution_rows=artifacts.get("execution_rows") if isinstance(artifacts.get("execution_rows"), list) else [],
        reflection_decisions=reflection_decisions,
        not_applicable=not_applicable,
    )
    err = _compute_err(
        fault=fault,
        injector=injector,
        success=bool(success_ok),
        not_applicable=not_applicable,
    )
    safe_halt = _compute_safe_halt(
        fault=fault,
        injector=injector,
        success=bool(success_ok),
        actual_status=actual_status,
        observed_error_types=observed_errors,
        reflection_decisions=reflection_decisions,
        not_applicable=not_applicable,
    )
    success_tools = _collect_success_tools(case_state)

    run_result = {
        "timestamp": _utc_now_iso(),
        "task_id": task_id,
        "arm": arm,
        "fault": fault,
        "case": {
            "case_id": case.get("case_id"),
            "domain": case.get("domain"),
            "case_root": case.get("case_root"),
            "input_format": case.get("input_format"),
            "modalities": case.get("modalities"),
        },
        "goal": goal,
        "run_case_id": run_case_id,
        "run_payload": {
            "run_dir": str(run_dir_path),
            "trace_path": str(artifacts.get("trace_path") or trace_path_s),
            "planner_status": run_payload.get("planner_status"),
            "planner_artifacts": (
                run_payload.get("planner_artifacts")
                if isinstance(run_payload.get("planner_artifacts"), dict)
                else {}
            ),
            "planner_metadata": (
                run_payload.get("planner_metadata")
                if isinstance(run_payload.get("planner_metadata"), dict)
                else {}
            ),
            "reflection_decisions": reflection_decisions,
        },
        "run_artifacts": {
            "case_state_path": str(artifacts.get("case_state_path") or (run_dir_path / "case_state.json")),
            "execution_log_path": str(artifacts.get("execution_log_path") or (run_dir_path / "execution_log.jsonl")),
            "trace_path": str(artifacts.get("trace_path") or trace_path_s),
        },
        "status": {
            "actual_status": actual_status,
            "success_tools": success_tools,
            "observed_error_types": observed_errors,
        },
        "raised_error": raised_error,
        "fault_injection": {
            "enabled": bool(injector.spec.enabled),
            "requested": bool(fault_requested),
            "fault": injector.spec.fault,
            "profile_enabled": profile_enabled,
            "not_applicable": bool(not_applicable),
            "not_applicable_reason": not_applicable_reason,
            "applied": bool(injector.applied),
            "events": list(injector.events),
        },
        "semantic_detector": semantic_detector,
    }

    metrics = {
        "timestamp": _utc_now_iso(),
        "task_id": task_id,
        "arm": arm,
        "fault": fault,
        "case_id": str(case.get("case_id") or ""),
        "run_case_id": run_case_id,
        "actual_status": actual_status,
        "success": {
            "pass": bool(success_ok),
            "detail": success_detail,
        },
        "tcr": tcr,
        "err": err,
        "safe_halt": safe_halt,
        "semantic_guard": semantic_guard,
        "semantic_detector": semantic_detector,
        "invariants": invariants,
        "invariant_pass": {
            "pass": invariant_pass,
            "evaluable": invariant_evaluable,
            "passed": invariant_passed,
            "total": invariant_total,
        },
        "fault_injection": run_result["fault_injection"],
        "run_dir": str(run_dir_path),
        # Keep a compact run payload in summary rows so reflection decisions survive
        # final JSON serialization (summary writes metrics rows, not full run_result).
        "run_payload": {
            "run_dir": str(run_dir_path),
            "trace_path": str(artifacts.get("trace_path") or trace_path_s),
            "planner_status": run_payload.get("planner_status"),
            "planner_metadata": (
                run_payload.get("planner_metadata")
                if isinstance(run_payload.get("planner_metadata"), dict)
                else {}
            ),
            "reflection_decisions": reflection_decisions,
        },
        "planner_metadata": (
            run_payload.get("planner_metadata")
            if isinstance(run_payload.get("planner_metadata"), dict)
            else {}
        ),
    }

    art_dir = run_dir_path / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "run_result.json").write_text(json.dumps(run_result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (art_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "run_result": run_result,
        "metrics": metrics,
    }


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "runs": 0,
            "success_pass": 0,
            "success_rate": 0.0,
            "avg_tcr": 0.0,
            "err_eligible_runs": 0,
            "err_recovered": 0,
            "err_rate": None,
            "fault_requested_runs": 0,
            "fault_not_applicable_runs": 0,
            "fault_evaluable_runs": 0,
            "fault_applied_runs": 0,
            "fault_applied_rate": None,
            "safe_halt_eligible_runs": 0,
            "safe_halt_pass_runs": 0,
            "safe_halt_rate": None,
            "semantic_guard_eligible_runs": 0,
            "semantic_guard_pass_runs": 0,
            "semantic_guard_rate": None,
            "invariant_evaluable_runs": 0,
            "invariant_pass_runs": 0,
            "invariant_pass_rate": None,
        }

    success_pass = 0
    tcr_sum = 0.0
    err_eligible = 0
    err_recovered = 0
    fault_requested = 0
    fault_not_applicable = 0
    fault_evaluable = 0
    fault_applied = 0
    safe_halt_eligible = 0
    safe_halt_pass = 0
    semantic_guard_eligible = 0
    semantic_guard_pass = 0
    invariant_evaluable = 0
    invariant_pass = 0

    for rec in records:
        m = rec.get("metrics") if isinstance(rec.get("metrics"), dict) else {}
        if bool((m.get("success") or {}).get("pass")):
            success_pass += 1
        tcr_sum += float((m.get("tcr") or {}).get("ratio") or 0.0)

        err = m.get("err") if isinstance(m.get("err"), dict) else {}
        if bool(err.get("eligible")):
            err_eligible += 1
            if bool(err.get("recovered")):
                err_recovered += 1

        fi = m.get("fault_injection") if isinstance(m.get("fault_injection"), dict) else {}
        if bool(fi.get("requested")):
            fault_requested += 1
            if bool(fi.get("not_applicable")):
                fault_not_applicable += 1
            else:
                fault_evaluable += 1
                if bool(fi.get("applied")):
                    fault_applied += 1

        sh = m.get("safe_halt") if isinstance(m.get("safe_halt"), dict) else {}
        if bool(sh.get("eligible")):
            safe_halt_eligible += 1
            if bool(sh.get("pass")):
                safe_halt_pass += 1

        sg = m.get("semantic_guard") if isinstance(m.get("semantic_guard"), dict) else {}
        if bool(sg.get("eligible")):
            semantic_guard_eligible += 1
            if bool(sg.get("pass")):
                semantic_guard_pass += 1

        inv = m.get("invariant_pass") if isinstance(m.get("invariant_pass"), dict) else {}
        if bool(inv.get("evaluable")):
            invariant_evaluable += 1
            if bool(inv.get("pass")):
                invariant_pass += 1

    runs = len(records)
    return {
        "runs": runs,
        "success_pass": success_pass,
        "success_rate": float(success_pass) / float(runs),
        "avg_tcr": float(tcr_sum) / float(runs),
        "err_eligible_runs": err_eligible,
        "err_recovered": err_recovered,
        "err_rate": (float(err_recovered) / float(err_eligible)) if err_eligible > 0 else None,
        "fault_requested_runs": fault_requested,
        "fault_not_applicable_runs": fault_not_applicable,
        "fault_evaluable_runs": fault_evaluable,
        "fault_applied_runs": fault_applied,
        "fault_applied_rate": (float(fault_applied) / float(fault_evaluable)) if fault_evaluable > 0 else None,
        "safe_halt_eligible_runs": safe_halt_eligible,
        "safe_halt_pass_runs": safe_halt_pass,
        "safe_halt_rate": (float(safe_halt_pass) / float(safe_halt_eligible)) if safe_halt_eligible > 0 else None,
        "semantic_guard_eligible_runs": semantic_guard_eligible,
        "semantic_guard_pass_runs": semantic_guard_pass,
        "semantic_guard_rate": (float(semantic_guard_pass) / float(semantic_guard_eligible)) if semantic_guard_eligible > 0 else None,
        "invariant_evaluable_runs": invariant_evaluable,
        "invariant_pass_runs": invariant_pass,
        "invariant_pass_rate": (float(invariant_pass) / float(invariant_evaluable)) if invariant_evaluable > 0 else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark v2 runner for the public paper modes. "
            "Paper BCER is implemented by the constrained-sketch arm (`bcer`/`bcer_sketch`)."
        )
    )
    ap.add_argument("--manifest", required=True, help="Path to cases_manifest.jsonl")
    ap.add_argument("--task", required=True, help="Task id defined in configs/tasks_registry.json")
    ap.add_argument("--arm", required=True, choices=list(ARM_CHOICES))
    ap.add_argument("--fault", default="none", choices=list(FAULT_CHOICES))
    ap.add_argument("--tasks-registry", default="configs/tasks_registry.json")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--workspace-root", default=str(project_root()))
    ap.add_argument("--runs-root", default=str(project_root() / "runs"))
    ap.add_argument("--server-base-url", default=str(os.environ.get("MRI_AGENT_SHELL_SERVER_BASE_URL") or "http://127.0.0.1:8000/v1"))
    ap.add_argument(
        "--provider",
        default=str(os.environ.get("MRI_AGENT_BENCH_PROVIDER") or "openai_compatible_server"),
        choices=["openai_compatible_server", "openai_official", "gemini", "anthropic"],
        help="LLM backend provider for planner/react/BCER execution.",
    )
    ap.add_argument(
        "--api-base-url",
        default=str(os.environ.get("MRI_AGENT_BENCH_API_BASE_URL") or ""),
        help="Optional API base URL for official APIs/proxies (OpenAI/Gemini/Anthropic).",
    )
    ap.add_argument(
        "--server-model",
        default=str(os.environ.get("MEDGEMMA_SERVER_MODEL") or bench_v1.DEFAULT_SERVER_MODEL),
    )
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--output", default="", help="Optional summary json output path.")
    ap.add_argument(
        "--cleanup-runs",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "If 1, delete each run directory (artifacts, logs, NIfTIs) after "
            "scoring is complete. Saves disk space for large-scale runs "
            "(e.g. case_100). The per-run metrics are preserved in the "
            "summary JSON. Default: 0 (keep all runs)."
        ),
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    arm = _canonical_arm(str(args.arm))

    manifest_path = Path(str(args.manifest)).expanduser().resolve()
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not manifest_path.is_file():
        raise SystemExit(
            f"Invalid --manifest: expected a .jsonl file, got non-file path: {manifest_path}"
        )

    tasks_registry = _tasks_obj(Path(str(args.tasks_registry)).expanduser().resolve())
    task_id = str(args.task).strip()
    if task_id not in tasks_registry or not isinstance(tasks_registry.get(task_id), dict):
        raise SystemExit(f"Unknown task_id: {task_id}")
    contract = dict(tasks_registry[task_id])

    manifest_rows = [_normalize_case_row(r) for r in _read_jsonl(manifest_path)]
    selected = [
        r
        for r in manifest_rows
        if task_id in set(r.get("supports_tasks") or [])
        and _task_domain_ok(contract, str(r.get("domain") or "").strip().lower())
        and str(r.get("case_root") or "").strip()
    ]

    if args.max_cases and int(args.max_cases) > 0:
        selected = selected[: int(args.max_cases)]

    if not selected:
        raise SystemExit(f"No cases selected from manifest for task={task_id}")

    workspace_root = Path(str(args.workspace_root)).expanduser().resolve()
    runs_root = Path(str(args.runs_root)).expanduser().resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    provider = str(args.provider or "openai_compatible_server").strip()
    api_key = _api_key_for_provider(provider)
    backend = _build_backend(
        provider=provider,
        server_base_url=str(args.server_base_url),
        api_base_url=str(args.api_base_url),
        server_model=str(args.server_model),
        api_key=api_key,
    )
    effective_max_new_tokens, budget_note = _effective_max_new_tokens(
        provider=provider,
        model=str(args.server_model),
        requested=int(args.max_new_tokens),
    )

    backend_loc = f"@{backend.base_url}" if str(backend.base_url or "").strip() else ""
    print(
        f"[benchmark_v2] task={task_id} arm={arm} fault={args.fault} "
        f"cases={len(selected)} backend={backend.provider}:{backend.model}{backend_loc}"
    )
    if str(args.arm) != arm:
        print(f"[benchmark_v2] arm alias: {args.arm} -> {arm}")
    if budget_note:
        print(
            f"[benchmark_v2] max_new_tokens requested={int(args.max_new_tokens)} "
            f"effective={effective_max_new_tokens} ({budget_note})"
        )

    cleanup_runs = bool(getattr(args, 'cleanup_runs', 0))
    if cleanup_runs:
        print("[benchmark_v2] --cleanup-runs=1: run directories will be deleted after scoring.")

    records: List[Dict[str, Any]] = []
    for idx, case in enumerate(selected, start=1):
        print(f"\n[{idx}/{len(selected)}] case={case['case_id']} domain={case['domain']}")
        out = _run_single_case(
            case=case,
            task_id=task_id,
            contract=contract,
            arm=arm,
            fault=str(args.fault),
            backend=backend,
            workspace_root=workspace_root,
            runs_root=runs_root,
            max_new_tokens=int(effective_max_new_tokens),
            max_steps=int(args.max_steps),
            max_retries=int(args.max_retries),
        )
        records.append(out)

        m = out.get("metrics") if isinstance(out.get("metrics"), dict) else {}
        success = bool((m.get("success") or {}).get("pass"))
        tcr = float((m.get("tcr") or {}).get("ratio") or 0.0)
        err = m.get("err") if isinstance(m.get("err"), dict) else {}
        err_val = err.get("value")
        err_str = "NA" if err_val is None else f"{float(err_val):.3f}"
        fi = m.get("fault_injection") if isinstance(m.get("fault_injection"), dict) else {}
        run_dir_str = str(m.get('run_dir') or '')
        print(
            "  "
            + f"success={success} tcr={tcr:.3f} err={err_str} "
            + f"fault_applied={bool(fi.get('applied'))} "
            + f"not_applicable={bool(fi.get('not_applicable'))} "
            + f"run_dir={run_dir_str}"
        )

        # Cleanup: delete the run directory to save disk space.
        # All scoring data is already captured in the metrics dict.
        if cleanup_runs and run_dir_str:
            _cleanup_run_dir(run_dir_str)

    summary = {
        "generated_at": _utc_now_iso(),
        "task_id": task_id,
        "arm": arm,
        "fault": str(args.fault),
        "manifest": str(manifest_path),
        "tasks_registry": str(Path(str(args.tasks_registry)).expanduser().resolve()),
        "workspace_root": str(workspace_root),
        "runs_root": str(runs_root),
        "backend": {
            "id": backend.backend_id,
            "provider": backend.provider,
            "model": backend.model,
            "base_url": backend.base_url,
        },
        "llm_runtime": {
            "max_new_tokens_requested": int(args.max_new_tokens),
            "max_new_tokens_effective": int(effective_max_new_tokens),
            "budget_note": (budget_note or ""),
        },
        "aggregate": _aggregate(records),
        "results": [r.get("metrics") for r in records],
    }

    if str(args.output or "").strip():
        out_path = Path(str(args.output)).expanduser().resolve()
    else:
        safe_task = re.sub(r"[^a-zA-Z0-9_.-]+", "_", task_id)
        out_path = Path(__file__).resolve().parent / f"benchmark_results_v2_{safe_task}_{args.arm}_{args.fault}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    agg = summary["aggregate"]
    print("\n[benchmark_v2] done")
    print(
        f"  runs={agg.get('runs')} success_rate={float(agg.get('success_rate') or 0.0):.3f} "
        f"avg_tcr={float(agg.get('avg_tcr') or 0.0):.3f} err_rate={agg.get('err_rate')} "
        f"fault_applied_rate={agg.get('fault_applied_rate')} safe_halt_rate={agg.get('safe_halt_rate')} "
        f"invariant_pass_rate={agg.get('invariant_pass_rate')}"
    )
    print(f"  summary={out_path}")


if __name__ == "__main__":
    main()

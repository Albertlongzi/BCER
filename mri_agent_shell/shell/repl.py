from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..agent.brain import Brain
from ..runtime.cerebellum import Cerebellum
from ..runtime.session import (
    SessionState,
    is_known_provider,
    provider_choices,
    provider_default_model,
    provider_help,
    validate_workspace_path,
)
from ..tool_registry import (
    build_shell_registry,
    discover_domain_catalog,
    list_tool_metadata,
    list_tool_names,
)
from .commands import parse_command
from .tui import TEXTUAL_AVAILABLE


class AgentShell:
    _INLINE_KV_RE = re.compile(
        r"([A-Za-z_][A-Za-z0-9_.-]*)\s*:\s*(\"[^\"]*\"|'[^']*'|[^,]+?)(?=(?:\s*,\s*|\s+[A-Za-z_][A-Za-z0-9_.-]*\s*:|$))"
    )
    _ANSI_RESET = "\033[0m"
    _ANSI_USER = "\033[1;36m"
    _ANSI_AGENT = "\033[1;32m"
    _ANSI_THOUGHT = "\033[2;90m"

    def __init__(self, session: SessionState, *, include_core_tools: Optional[bool] = None) -> None:
        self.session = session
        self.registry = build_shell_registry(dry_run=session.dry_run, include_core=include_core_tools)
        self.domain_catalog = discover_domain_catalog(self.registry)
        self.tool_metadata = list_tool_metadata(self.registry)
        self.demo_cases = self._discover_demo_cases()
        self._case_source = "unset"  # unset | auto | manual | template
        self.brain = Brain(self.registry.list_specs(), domain_catalog=self.domain_catalog)
        self.cerebellum = Cerebellum(session=self.session, registry=self.registry)
        self._doctor_ran = False

    def banner_lines(self) -> list[str]:
        lines = [
            "MRI Agent Shell",
            "Type :help for commands.",
            f"Loaded tools: {len(list_tool_names(self.registry))}",
        ]
        lines.extend([f"  {line}" for line in self.session.summary_lines()])
        return lines

    def help_text(self) -> str:
        return "\n".join(
            [
                "Commands:",
                ":help",
                ":doctor",
                ":domains",
                ":tools",
                ":paste",
                ":model",
                ":model set llm=<name> vlm=<name> [provider=<openai_official|openai_compatible_server|stub|anthropic|gemini>] [base_url=<url>] [api_base_url=<url>] [max_tokens=<int>] [temperature=<float>]",
                ":workspace set <path>",
                ":case load <path>",
                ":run",
                ":exit",
                "",
                "Tips:",
                "- `help` (without colon) also prints this help.",
                "- natural-language tasks trigger planning/template flow.",
                "- use `:paste` to enter multiline YAML/JSON; finish with `:end`.",
                "Natural language request: describe your task.",
                "If underspecified, shell returns a YAML template. Paste filled YAML/JSON back, then run :run.",
            ]
        )

    def run_interactive(self) -> int:
        for line in self.banner_lines():
            print(line)
        if not self._doctor_ran:
            self.run_doctor(emit=print, auto=True)
            self._doctor_ran = True

        while True:
            try:
                line = input("mri-agent> ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                return 0

            inline_patch = self._parse_inline_template_patch(line)
            if self._looks_like_template_start(line) and inline_patch is None:
                block = self._collect_multiline(line, stop_on_blank=True, stop_tokens={":end"})
                keep = self.handle_template_block(block, emit=print)
            else:
                keep = self.handle_line(line, emit=print)

            if not keep:
                return 0

    def run_script(self, script_path: Path, *, emit: Callable[[str], None] = print) -> int:
        p = Path(script_path).expanduser().resolve()
        if not p.exists():
            emit(f"Script not found: {p}")
            return 2

        if not self._doctor_ran:
            self.run_doctor(emit=emit, auto=True)
            self._doctor_ran = True

        lines = p.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            raw = lines[i]
            i += 1
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line == ":paste":
                emit(">>> :paste")
                block_lines: List[str] = []
                while i < len(lines):
                    nxt_raw = lines[i]
                    i += 1
                    if nxt_raw.strip() == ":end":
                        break
                    block_lines.append(nxt_raw)
                block = "\n".join(block_lines).strip()
                if not block:
                    emit("Paste block is empty.")
                    continue
                keep = self.handle_template_block(block, emit=emit)
                if not keep:
                    break
                continue

            # Commands and template directives are dispatched immediately.
            if line.startswith(":") or line.startswith("@") or self._looks_like_template_start(line):
                emit(f">>> {line}")
                keep = self.handle_line(line, emit=emit)
                if not keep:
                    break
                continue

            # Natural-language: accumulate consecutive non-command, non-blank lines
            # into a single prompt so multi-line descriptions aren't split.
            nl_parts: List[str] = [line]
            while i < len(lines):
                nxt = lines[i].strip()
                if not nxt or nxt.startswith("#"):
                    # Blank / comment line ends the paragraph.
                    i += 1
                    break
                if nxt.startswith(":") or nxt.startswith("@") or nxt == ":paste" or self._looks_like_template_start(nxt):
                    break
                nl_parts.append(nxt)
                i += 1
            combined = " ".join(nl_parts)
            emit(f">>> {combined}")
            keep = self.handle_line(combined, emit=emit)
            if not keep:
                break
        return 0

    def handle_line(self, line: str, *, emit: Callable[[str], None]) -> bool:
        text = str(line or "").strip()
        if not text:
            return True

        if text.lower() in {"help", "?"}:
            emit(self.help_text())
            return True

        cmd = parse_command(text)
        if cmd is not None:
            return self._handle_command(cmd.kind, cmd.args, cmd.kv, tail=cmd.tail, emit=emit)

        if self._looks_like_template_start(text):
            return self.handle_template_block(text, emit=emit)

        inline_patch = self._parse_inline_template_patch(text)
        if inline_patch is not None:
            return self._apply_template_patch(inline_patch, emit=emit, show_template_on_incomplete=False)

        # Natural-language mode defaults to autonomous planning (Observe -> Think -> Plan).
        return self._handle_natural_language_request(text=text, emit=emit)

    def handle_template_block(self, text: str, *, emit: Callable[[str], None]) -> bool:
        inline_patch = self._parse_inline_template_patch(text)
        if inline_patch is not None and "\n" not in str(text or ""):
            return self._apply_template_patch(inline_patch, emit=emit, show_template_on_incomplete=False)

        try:
            tpl = self.brain.parse_template_text(text)
        except Exception as e:
            inline_patch = self._parse_inline_template_patch(text)
            if inline_patch is not None:
                return self._apply_template_patch(inline_patch, emit=emit, show_template_on_incomplete=False)
            emit(f"Template parse error: {e}")
            return True

        show_tpl = "\n" in str(text or "") or "REQUIRED:" in str(text or "") or "OPTIONAL:" in str(text or "")
        return self._apply_template_patch(tpl, emit=emit, show_template_on_incomplete=show_tpl)

    def _handle_natural_language_request(self, *, text: str, emit: Callable[[str], None]) -> bool:
        self.session.current_request = text
        emit(self._format_user_prompt(text))
        domain, request_type = self.brain._classify_request(text=text, model_cfg=self.session.model_config)
        request_type = str(request_type or "full_pipeline").strip().lower() or "full_pipeline"
        prev_domain = self._extract_template_domain(self.session.current_template)
        if not domain:
            domain = prev_domain
        if not domain:
            domain = self._infer_domain_from_case_path(str(self.session.case_inputs.get("case_input_path") or ""))
        if not domain and self.session.dry_run and "demo" in self.domain_catalog:
            domain = "demo"

        explicit_case_ref = self._extract_case_ref_from_text(text)
        patch_required: Dict[str, Any] = {"request_type": request_type}
        if domain:
            patch_required["domain"] = domain
        if explicit_case_ref:
            patch_required["case_ref"] = explicit_case_ref

        merged = self._merge_template_payload(self.session.current_template, {"REQUIRED": patch_required})
        notes = self._apply_domain_and_case_defaults(
            merged=merged,
            prev_domain=prev_domain,
            new_domain=self._extract_template_domain(merged),
            explicit_case_ref=bool(explicit_case_ref),
            explicit_case_id=False,
        )
        self.session.current_template = merged
        for line in notes:
            emit(line)

        supported = sorted([d for d in self.domain_catalog.keys() if d != "demo"])
        domain = self._extract_template_domain(self.session.current_template)
        if domain and supported and domain not in supported:
            self.session.current_plan = None
            emit(f"Not supported: inferred domain '{domain}' is not enabled for current tools.")
            emit(f"Supported domains: {supported}")
            return True
        if not domain:
            self.session.current_plan = None
            emit("I couldn't infer the domain from this request.")
            emit(f"Supported domains: {supported or ['prostate', 'brain', 'cardiac']}")
            emit("Provide `domain: prostate|brain|cardiac` or load a case with `:case load <path>`.")
            self._emit_draft_status(emit=emit)
            return True

        case_ref_raw = self._extract_template_case_ref(self.session.current_template)
        case_ref = self._resolve_case_ref(case_ref_raw)
        scan = self._inspect_case(case_ref=case_ref, emit=emit)
        if scan.get("ok"):
            scan_case_ref = str(scan.get("case_ref") or case_ref)
            scan_case_id = str(scan.get("case_id") or "")
            req = dict((self.session.current_template or {}).get("REQUIRED") or {})
            opt = dict((self.session.current_template or {}).get("OPTIONAL") or {})
            req["case_ref"] = scan_case_ref
            if scan_case_id and self._case_source != "manual":
                opt["case_id"] = scan_case_id
                self.session.case_id = scan_case_id
            self.session.current_template = self._merge_template_payload(
                self.session.current_template,
                {"REQUIRED": req, "OPTIONAL": opt},
            )
            self.session.case_inputs["case_input_path"] = scan_case_ref
            self.session.path_keys["case.input"] = scan_case_ref
            if not domain and str(scan.get("domain_hint") or "").strip():
                domain = str(scan.get("domain_hint")).strip().lower()
            emit(f"🧠 think: inferred domain={domain} request_type={request_type}")
        else:
            emit(f"🧠 think: inferred domain={domain} request_type={request_type}")
            if str(scan.get("error") or "") == "not_found":
                self.session.current_plan = None
                emit("Current case_ref does not exist. Use `:case load <path>` or update `case_ref` to a valid folder.")
                self._emit_draft_status(emit=emit)
                return True
            if request_type != "report" and self._is_default_case_ref(case_ref_raw):
                self.session.current_plan = None
                emit("No case is bound yet. Load one with `:case load <path>` or set `case_ref` explicitly.")
                self._emit_draft_status(emit=emit)
                return True

        case_ref_for_plan = str(scan.get("case_ref") or self._resolve_case_ref(self._extract_template_case_ref(self.session.current_template)) or "").strip()
        case_id_for_plan = str(
            self._extract_template_case_id(self.session.current_template)
            or scan.get("case_id")
            or self.session.case_id
            or ""
        ).strip()

        if (not self._wants_legacy_template_mode(text)) and case_ref_for_plan:
            dag_obj, dag_err = self._plan_dag_from_langgraph(
                goal=text,
                domain=domain,
                request_type=request_type,
                case_ref=case_ref_for_plan,
                case_id=case_id_for_plan,
            )
            if dag_obj is not None:
                nl = str(dag_obj.get("natural_language_response") or "").strip()
                nl = self._sanitize_natural_language_response(nl)
                if nl:
                    emit(self._format_agent_response(nl))
                planner_status = str(dag_obj.get("planner_status") or "ready").strip().lower()
                if planner_status == "blocked":
                    self.session.current_plan = None
                    self._emit_draft_status(emit=emit, include_ready_hint=False)
                    return True
                self.session.current_plan = dag_obj
                emit("LangGraph planner generated DAG. Type ':run' to execute.")
                self._emit_draft_status(emit=emit, include_ready_hint=False)
                self._emit_plan_outline(self.session.current_plan, emit=emit)
                return True
            emit(f"LangGraph planner unavailable; falling back to deterministic template planner. ({dag_err})")

        plan_input = {
            "REQUIRED": {
                "domain": domain,
                "request_type": request_type,
                "case_ref": self._extract_template_case_ref(self.session.current_template),
            },
            "OPTIONAL": {
                "case_id": self._extract_template_case_id(self.session.current_template),
            },
        }
        plan_res = self.brain.from_template(plan_input, session=self.session, goal=text)
        if plan_res.kind != "plan" or plan_res.plan is None:
            self.session.current_plan = None
            emit(plan_res.message)
            self._emit_draft_status(emit=emit)
            return True

        self.session.current_plan = plan_res.plan.model_dump()
        if scan.get("ok"):
            self._adapt_plan_with_scan(scan=scan, emit=emit)
        emit("Required fields filled. Type ':run' to execute.")
        self._emit_draft_status(emit=emit, include_ready_hint=False)
        self._emit_plan_outline(self.session.current_plan, emit=emit)
        return True

    def _sanitize_natural_language_response(self, text: str) -> str:
        s = str(text or "")
        if not s:
            return ""
        # Remove chain-of-thought tags and payloads if model leaks them.
        s = re.sub(r"<think\b[^>]*>.*?</think>", "", s, flags=re.IGNORECASE | re.DOTALL)
        s = re.sub(r"</?think\b[^>]*>", "", s, flags=re.IGNORECASE)
        s = re.sub(r"</?thought_process\b[^>]*>", "", s, flags=re.IGNORECASE)
        s = re.sub(r"</?final_answer\b[^>]*>", "", s, flags=re.IGNORECASE)
        s = re.sub(r"[ \t]+\n", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    def _format_user_prompt(self, text: str) -> str:
        return f"{self._ANSI_USER}User Prompt:{self._ANSI_RESET} {str(text or '').strip()}"

    def _format_agent_response(self, text: str) -> str:
        return f"{self._ANSI_AGENT}Agent Response:{self._ANSI_RESET} {str(text or '').strip()}"

    def _extract_xml_tag_text(self, text: str, tag: str) -> str:
        s = str(text or "")
        m = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", s, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return ""
        return str(m.group(1) or "").strip()

    def _parse_structured_response(self, text: str) -> Dict[str, str]:
        raw = str(text or "").strip()
        thought = self._extract_xml_tag_text(raw, "thought_process")
        final = self._extract_xml_tag_text(raw, "final_answer")
        return {
            "thought_process": thought,
            "final_answer": final,
        }

    def _emit_structured_agent_response(self, text: str, *, emit: Callable[[str], None]) -> bool:
        parsed = self._parse_structured_response(text)
        thought = str(parsed.get("thought_process") or "").strip()
        final = str(parsed.get("final_answer") or "").strip()
        if not thought and not final:
            return False
        emit(f"{self._ANSI_AGENT}Agent Response:{self._ANSI_RESET}")
        if thought:
            for line in thought.splitlines():
                emit(f"{self._ANSI_THOUGHT}{line}{self._ANSI_RESET}")
        if final:
            emit(final)
        return True

    def _should_emit_debug_run_json(self) -> bool:
        if bool(self.session.dry_run):
            return True
        return str(os.environ.get("MRI_AGENT_SHELL_DEBUG_RUN_RESULT", "")).strip().lower() in {"1", "true", "yes", "on"}

    def _infer_domain_from_case_path(self, case_path: str) -> str:
        p = str(case_path or "").lower()
        if not p:
            return ""
        if "brats" in p or "cbica" in p:
            return "brain"
        if "acdc" in p or "cine" in p or "patient" in p:
            return "cardiac"
        if "prostate" in p or "sub-" in p:
            return "prostate"
        return ""

    def _missing_modalities_for_request(
        self,
        *,
        domain: str,
        request_type: str,
        modalities: set[str],
    ) -> List[str]:
        d = str(domain or "").strip().lower()
        r = str(request_type or "").strip().lower()
        mods = set([str(x).strip().lower() for x in (modalities or set()) if str(x).strip()])
        if not d or r == "report":
            return []

        missing: List[str] = []
        if d == "prostate":
            if r in {"segment"} and "t2w" not in mods:
                missing.append("t2w")
            elif r in {"full_pipeline", "register", "lesion"}:
                for m in ("t2w", "adc", "dwi"):
                    if m not in mods:
                        missing.append(m)
            return missing

        if d == "brain":
            if r in {"full_pipeline", "segment"}:
                if "flair" not in mods:
                    missing.append("flair")
                if "t2w" not in mods and "t2" not in mods:
                    missing.append("t2/t2w")
                if ("t1c" not in mods) and ("t1" not in mods):
                    missing.append("t1c|t1")
            return missing

        if d == "cardiac":
            if r in {"full_pipeline", "segment", "classify"} and "cine" not in mods:
                missing.append("cine")
            return missing
        return []

    def _inspect_case(self, *, case_ref: str, emit: Callable[[str], None]) -> Dict[str, Any]:
        ref = str(case_ref or "").strip()
        if self._is_default_case_ref(ref):
            return {"ok": False, "error": "case_ref unresolved"}

        emit(f"▶️ inspect_case(case_ref={ref})")
        t0 = time.time()
        p = Path(ref).expanduser()
        if not p.exists():
            emit(f"❌ inspect_case(...): path not found: {ref}")
            return {"ok": False, "error": "not_found", "case_ref": ref}

        files = 0
        dirs = 0
        nifti = 0
        dicom = 0
        names: List[str] = []
        modalities: set[str] = set()
        sequence_candidates: Dict[str, str] = {}
        try:
            for i, x in enumerate(p.rglob("*")):
                if i > 800:
                    break
                if x.is_dir():
                    dirs += 1
                    continue
                files += 1
                n = x.name.lower()
                n_full = f"{x.parent.name.lower()} {n}"
                path_s = str(x.resolve())
                if len(names) < 8:
                    names.append(x.name)
                if n.endswith(".nii") or n.endswith(".nii.gz"):
                    nifti += 1
                if n.endswith(".dcm"):
                    dicom += 1
                elif "." not in n and n not in ("dicomdir", "lockfile", "version"):
                    # Extension-less file – probe for DICOM magic bytes (b"DICM" at offset 128)
                    try:
                        with open(x, "rb") as _fh:
                            _fh.seek(128)
                            if _fh.read(4) == b"DICM":
                                dicom += 1
                    except Exception:
                        pass
                if "t2" in n_full:
                    modalities.add("t2w")
                    if x.is_file():
                        sequence_candidates.setdefault("t2w", path_s)
                if "adc" in n_full:
                    modalities.add("adc")
                    if x.is_file():
                        sequence_candidates.setdefault("adc", path_s)
                if "dwi" in n_full or "trace" in n_full or "high_b" in n_full:
                    modalities.add("dwi")
                    if x.is_file():
                        sequence_candidates.setdefault("dwi", path_s)
                if "flair" in n_full:
                    modalities.add("flair")
                    if x.is_file():
                        sequence_candidates.setdefault("flair", path_s)
                if "t1c" in n_full or "t1ce" in n_full:
                    modalities.add("t1c")
                    if x.is_file():
                        sequence_candidates.setdefault("t1c", path_s)
                if "t1" in n_full and "t1c" not in n_full and "t1ce" not in n_full:
                    modalities.add("t1")
                    if x.is_file():
                        sequence_candidates.setdefault("t1", path_s)
                if "cine" in n_full or "bssfp" in n_full or "ssfp" in n_full:
                    modalities.add("cine")
                    if x.is_file():
                        sequence_candidates.setdefault("cine", path_s)
        except Exception:
            pass

        domain_hint = ""
        if {"flair", "t1c"} & modalities:
            domain_hint = "brain"
        elif "cine" in modalities:
            domain_hint = "cardiac"
        elif {"t2w", "adc", "dwi"} & modalities:
            domain_hint = "prostate"
        if not domain_hint:
            domain_hint = self._infer_domain_from_case_path(str(p))

        out = {
            "ok": True,
            "case_ref": str(p.resolve()),
            "case_id": p.name,
            "files": files,
            "dirs": dirs,
            "nifti": nifti,
            "dicom": dicom,
            "modalities": sorted(modalities),
            "sequence_candidates": sequence_candidates,
            "domain_hint": domain_hint,
            "sample_names": names,
        }
        dur = time.time() - t0
        emit(
            "✅ inspect_case(...): "
            f"{dur:.2f}s | outputs={{'case_id': '{out['case_id']}', 'nifti': {nifti}, 'dicom': {dicom}, "
            f"'modalities': {out['modalities']}, 'domain_hint': '{domain_hint}'}}"
        )
        return out

    def _adapt_plan_with_scan(self, *, scan: Dict[str, Any], emit: Callable[[str], None]) -> None:
        plan = self.session.current_plan
        if not isinstance(plan, dict):
            return
        steps = plan.get("steps")
        if not isinstance(steps, list):
            return
        domain = str(plan.get("domain") or "").strip().lower()
        modalities = set([str(x).lower() for x in (scan.get("modalities") or [])])
        seq_candidates = scan.get("sequence_candidates") if isinstance(scan.get("sequence_candidates"), dict) else {}
        if not modalities and not seq_candidates:
            return

        for mod, path in seq_candidates.items():
            m = str(mod or "").strip().lower()
            p = str(path or "").strip()
            if not p:
                continue
            if m == "t2w":
                self.session.set_path_key("seq.T2w", p)
                self.session.set_path_key("seq.T2", p)
            elif m == "adc":
                self.session.set_path_key("seq.ADC", p)
            elif m == "dwi":
                self.session.set_path_key("seq.DWI", p)
            elif m == "flair":
                self.session.set_path_key("seq.FLAIR", p)
            elif m == "t1c":
                self.session.set_path_key("seq.T1c", p)
                self.session.set_path_key("seq.T1ce", p)
            elif m == "t1":
                self.session.set_path_key("seq.T1", p)
            elif m == "cine":
                self.session.set_path_key("seq.CINE", p)

        removed: List[str] = []
        blocked_required: List[str] = []
        kept: List[Dict[str, Any]] = []
        tuned: List[str] = []
        for step in steps:
            if not isinstance(step, dict):
                kept.append(step)
                continue
            tname = str(step.get("tool_name") or "")
            required = bool(step.get("required", True))
            args = dict(step.get("arguments") or {})

            if tname == "identify_sequences":
                case_ref = str(scan.get("case_ref") or "").strip()
                if case_ref:
                    args["dicom_case_dir"] = case_ref
                    tuned.append("identify_sequences.dicom_case_dir")

            if domain == "prostate" and tname == "register_to_reference":
                moving = str(args.get("moving") or "").strip().lower()
                if moving == "adc" and "adc" not in modalities:
                    if required:
                        blocked_required.append(f"{tname}(moving=ADC)")
                        kept.append(step)
                    else:
                        removed.append(f"{tname}(moving=ADC)")
                    continue
                if moving == "dwi" and "dwi" not in modalities:
                    if required:
                        blocked_required.append(f"{tname}(moving=DWI)")
                        kept.append(step)
                    else:
                        removed.append(f"{tname}(moving=DWI)")
                    continue
                fixed = str(args.get("fixed") or "").strip()
                fixed_path = self._scan_pick(seq_candidates, fixed)
                moving_path = self._scan_pick(seq_candidates, moving)
                if fixed_path:
                    args["fixed"] = fixed_path
                    tuned.append("register_to_reference.fixed")
                if moving_path:
                    args["moving"] = moving_path
                    tuned.append("register_to_reference.moving")
            if domain == "prostate" and tname == "detect_lesion_candidates":
                if not ({"t2w", "adc", "dwi"} <= modalities):
                    if required:
                        blocked_required.append(tname)
                        kept.append(step)
                    else:
                        removed.append(tname)
                    continue
            if domain == "prostate" and tname == "segment_prostate":
                t2 = self._scan_pick(seq_candidates, "t2w")
                if t2:
                    args["t2w_ref"] = t2
                    tuned.append("segment_prostate.t2w_ref")
            if domain == "brain" and tname == "brats_mri_segmentation":
                if not ({"t1c", "flair"} & modalities):
                    if required:
                        blocked_required.append(tname)
                        kept.append(step)
                    else:
                        removed.append(tname)
                    continue
                for field, key in (
                    ("t1c_path", "t1c"),
                    ("t1_path", "t1"),
                    ("t2_path", "t2w"),
                    ("flair_path", "flair"),
                ):
                    pv = self._scan_pick(seq_candidates, key)
                    if pv:
                        args[field] = pv
                        tuned.append(f"brats_mri_segmentation.{field}")
            if domain == "cardiac" and tname == "segment_cardiac_cine":
                if "cine" not in modalities:
                    if required:
                        blocked_required.append(tname)
                        kept.append(step)
                    else:
                        removed.append(tname)
                    continue
                cine = self._scan_pick(seq_candidates, "cine")
                if cine:
                    args["cine_path"] = cine
                    tuned.append("segment_cardiac_cine.cine_path")
            step["arguments"] = args
            kept.append(step)

        if removed or tuned:
            plan["steps"] = kept
            if str(scan.get("case_ref") or "").strip():
                plan["case_ref"] = str(scan.get("case_ref"))
            self.session.current_plan = plan
        if removed:
            emit(f"🛠 think: adjusted plan using inspect_case, removed steps={removed}")
            for i, step in enumerate(kept, start=1):
                emit(f"  {i}. {step.get('tool_name')} stage={step.get('stage')} required={step.get('required')}")
        if blocked_required:
            emit(f"❌ blocked: required step preconditions missing -> {blocked_required}")
        if tuned:
            emit(f"🛠 think: bound tool args from case scan ({len(tuned)} updates).")

    def _scan_pick(self, seq_candidates: Dict[str, Any], token: str) -> str:
        raw = str(token or "").strip().lower()
        if not raw:
            return ""
        alias = {
            "t2": "t2w",
            "t2w": "t2w",
            "adc": "adc",
            "dwi": "dwi",
            "t1c": "t1c",
            "t1ce": "t1c",
            "t1": "t1",
            "flair": "flair",
            "cine": "cine",
        }
        key = alias.get(raw, raw)
        val = seq_candidates.get(key)
        return str(val or "").strip()

    def _handle_command(
        self,
        kind: str,
        args: list[str],
        kv: Dict[str, str],
        tail: str = "",
        *,
        emit: Callable[[str], None],
    ) -> bool:
        if kind == "help":
            emit(self.help_text())
            return True

        if kind == "doctor":
            self.run_doctor(emit=emit, auto=False)
            self._doctor_ran = True
            return True

        if kind == "domains":
            if not self.domain_catalog:
                emit("No domains discovered from current registry.")
                return True
            emit("Discovered domains:")
            for dom in sorted(self.domain_catalog.keys()):
                rec = self.domain_catalog.get(dom) or {}
                caps = rec.get("capabilities") or []
                demo = self.demo_cases.get(dom)
                if demo:
                    emit(f"- {dom}: capabilities={caps} tool_count={rec.get('tool_count', 0)} demo_case={demo}")
                else:
                    emit(f"- {dom}: capabilities={caps} tool_count={rec.get('tool_count', 0)}")
            return True

        if kind == "tools":
            emit(f"Tools ({len(self.tool_metadata)}):")
            for rec in self.tool_metadata:
                nm = rec.get("name")
                domains = rec.get("domains") or []
                caps = rec.get("capabilities") or []
                emit(f"- {nm}: domains={domains} capabilities={caps}")
            return True

        if kind == "paste":
            if not sys.stdin.isatty():
                emit("`:paste` requires interactive stdin. In script mode, put YAML block after :paste and end with :end.")
                return True
            emit("Paste mode: enter YAML/JSON template or key:value lines. End with ':end'.")
            block = self._collect_multiline("", stop_on_blank=False, stop_tokens={":end"})
            if not block.strip():
                emit("Paste block is empty.")
                return True
            return self.handle_template_block(block, emit=emit)

        if kind == "exit":
            return False

        if kind == "invalid":
            emit("Invalid command. Type :help.")
            return True

        if kind == "model_show":
            emit(json.dumps(self.session.model_config.model_dump(), indent=2, ensure_ascii=False))
            return True

        if kind == "model_set":
            provider_raw = str(kv.get("provider") or "").strip()
            if provider_raw and not is_known_provider(provider_raw):
                emit(
                    "Invalid provider. Choices: "
                    + ", ".join(provider_choices())
                    + " (aliases: openai, server)"
                )
                return True
            self.session.model_config.update_from_kwargs(kv)
            emit("Model config updated:")
            emit(json.dumps(self.session.model_config.model_dump(), indent=2, ensure_ascii=False))
            emit(provider_help(self.session.model_config.provider))
            return True

        if kind == "workspace_set":
            if not args:
                emit("Usage: :workspace set <path>")
                return True
            try:
                self.session.set_workspace(args[0])
            except Exception as e:
                emit(f"Invalid workspace: {e}")
                return True
            emit(f"Workspace set to: {self.session.workspace_path}")
            return True

        if kind == "case_load":
            if not args:
                emit("Usage: :case load <path>")
                return True
            try:
                p = self.session.set_case_input(args[0])
            except Exception as e:
                emit(f"Invalid case path: {e}")
                return True
            cid = Path(p).name
            self.session.case_id = cid
            self._case_source = "manual"
            self.session.current_template = self._merge_template_payload(
                self.session.current_template,
                {"REQUIRED": {"case_ref": p}, "OPTIONAL": {"case_id": cid}},
            )
            self.session.current_plan = None
            emit(f"Case loaded: {p}")
            emit(f"case_id: {cid}")
            tail_s = str(tail or "").strip()
            if tail_s:
                inline_patch = self._parse_inline_template_patch(tail_s)
                if inline_patch is not None:
                    req_patch = dict((inline_patch or {}).get("REQUIRED") or {})
                    opt_patch = dict((inline_patch or {}).get("OPTIONAL") or {})
                    req_patch.setdefault("case_ref", p)
                    opt_patch.setdefault("case_id", cid)
                    inline_patch["REQUIRED"] = req_patch
                    inline_patch["OPTIONAL"] = opt_patch
                    return self._apply_template_patch(inline_patch, emit=emit, show_template_on_incomplete=False)
                emit("Trailing text after `:case load` was ignored. Use `domain: ...`/YAML on a new line.")
            return True

        if kind == "run":
            if not self.session.current_plan and self.session.current_template:
                res = self.brain.from_template(self.session.current_template, session=self.session)
                if res.kind == "plan" and res.plan is not None:
                    self.session.current_plan = res.plan.model_dump()
                else:
                    self.session.current_plan = None
                    emit(res.message)
                    self._emit_draft_status(emit=emit)
                    return True

            if not self.session.current_plan and self.session.current_request:
                self._handle_natural_language_request(text=str(self.session.current_request), emit=emit)

            if not self.session.current_plan:
                emit("No runnable plan. Provide a request first.")
                return True

            if self._is_dag_plan(self.session.current_plan):
                dag_scope = dict((self.session.current_plan or {}).get("case_scope") or {})
                dag_case_ref = str(dag_scope.get("case_ref") or "").strip()
                if not dag_case_ref:
                    emit("DAG plan is missing case_scope.case_ref.")
                    return True
            else:
                plan_case_ref = str((self.session.current_plan or {}).get("case_ref") or "").strip()
                if self._is_default_case_ref(plan_case_ref) and "case.input" not in self.session.path_keys:
                    emit("Plan is ready but `case.input` is unresolved.")
                    emit("Use `:case load <path>` or set REQUIRED.case_ref to an absolute path, then run :run again.")
                    return True

                if self._plan_references_key(self.session.current_plan, "case.input") and "case.input" not in self.session.path_keys:
                    emit("Plan is ready but `case.input` is unresolved.")
                    emit("Use `:case load <path>` or set REQUIRED.case_ref to an absolute path, then run :run again.")
                    return True

            # Recreate executor so it picks up latest workspace/runs settings.
            self.cerebellum = Cerebellum(session=self.session, registry=self.registry)
            run_t0 = time.time()
            if self._is_dag_plan(self.session.current_plan):
                result = self.cerebellum.execute_dag(self.session.current_plan, emit=emit)
            else:
                result = self.cerebellum.execute_plan(self.session.current_plan, emit=emit)
            run_elapsed_s = max(0.0, float(time.time() - run_t0))
            result_view = dict(result or {}) if isinstance(result, dict) else result
            run_nl = str(
                (result_view or {}).get("natural_language_response")
                or (result_view or {}).get("final_answer")
                or ""
            ).strip()
            rendered_structured = False
            if run_nl:
                rendered_structured = self._emit_structured_agent_response(run_nl, emit=emit)
                if (not rendered_structured):
                    clean = self._sanitize_natural_language_response(run_nl)
                    if clean:
                        emit(self._format_agent_response(clean))

            ok = bool((result_view or {}).get("ok")) if isinstance(result_view, dict) else False
            trace_path = str((result_view or {}).get("trace_path") or "").strip() if isinstance(result_view, dict) else ""
            if ok:
                if trace_path:
                    emit(f"✅ Run completed successfully in {run_elapsed_s:.2f}s. (Trace saved to: {trace_path})")
                else:
                    emit(f"✅ Run completed successfully in {run_elapsed_s:.2f}s.")
            else:
                if trace_path:
                    emit(f"❌ Run failed in {run_elapsed_s:.2f}s. (Trace saved to: {trace_path})")
                else:
                    emit(f"❌ Run failed in {run_elapsed_s:.2f}s.")

            if self._should_emit_debug_run_json():
                emit(json.dumps(result_view, indent=2, ensure_ascii=False))
            return True

        emit("Unknown command.")
        return True

    def run_doctor(self, *, emit: Callable[[str], None], auto: bool) -> Dict[str, Any]:
        checks: List[tuple[str, bool, str]] = []

        ws_ok, ws_msg, ws_path = validate_workspace_path(self.session.workspace_path)
        checks.append(("workspace writable", ws_ok, ws_msg if not ws_ok else ws_path))

        rr = Path(self.session.runs_root).expanduser().resolve()
        rr_ok = True
        rr_msg = str(rr)
        try:
            rr.mkdir(parents=True, exist_ok=True)
            probe = rr / ".mri_agent_shell_runs_write_test"
            probe.write_text("ok\n", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except Exception as e:
            rr_ok = False
            rr_msg = str(e)
        checks.append(("runs_root creatable", rr_ok, rr_msg))

        provider = self.session.model_config.provider
        provider_ok = True
        provider_msg = ""
        if provider == "openai_official":
            key = self.session.model_config.effective_api_key()
            if not key:
                provider_ok = False
                provider_msg = "missing OPENAI_API_KEY (or model.api_key in config)"
            else:
                provider_msg = "OPENAI_API_KEY detected"
        elif provider == "openai_compatible_server":
            base = str(self.session.model_config.server_base_url or "").strip()
            if not base or not (base.startswith("http://") or base.startswith("https://")):
                provider_ok = False
                provider_msg = "base_url must look like http://host:port/v1"
            elif "/v1" not in base:
                provider_ok = False
                provider_msg = "base_url should include /v1 (example: http://127.0.0.1:8000/v1)"
            else:
                provider_msg = f"base_url={base}"
        elif provider == "stub":
            provider_msg = "stub mode (no external backend required)"
        elif provider == "anthropic":
            key = self.session.model_config.effective_api_key() or str(os.environ.get("ANTHROPIC_API_KEY", "")).strip()
            if not key:
                provider_ok = False
                provider_msg = "missing ANTHROPIC_API_KEY"
            else:
                provider_msg = "ANTHROPIC_API_KEY detected"
        elif provider == "gemini":
            key = self.session.model_config.effective_api_key() or str(
                os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
            ).strip()
            if not key:
                provider_ok = False
                provider_msg = "missing GOOGLE_API_KEY/GEMINI_API_KEY"
            else:
                provider_msg = "GOOGLE_API_KEY detected"
        checks.append((f"provider config ({provider})", provider_ok, provider_msg))

        llm = str(self.session.model_config.llm or "").strip()
        vlm = str(self.session.model_config.vlm or "").strip()
        models_ok = bool(llm and vlm)
        models_msg = (
            f"llm={llm or provider_default_model(provider)} vlm={vlm or provider_default_model(provider)}"
        )
        checks.append(("model fields", models_ok, models_msg))

        tui_msg = "Textual available" if TEXTUAL_AVAILABLE else 'Textual missing; install with: pip install -e ".[tui]"'
        checks.append(("textual", bool(TEXTUAL_AVAILABLE), tui_msg))

        domain_count = len(self.domain_catalog)
        tool_count = len(self.tool_metadata)
        pipeline_summary = {
            dom: (rec.get("capabilities") or [])
            for dom, rec in sorted(self.domain_catalog.items())
        }
        checks.append(
            (
                "registry discovery",
                domain_count > 0 and tool_count > 0,
                f"domains={sorted(self.domain_catalog.keys())} pipelines={pipeline_summary} tools={tool_count}",
            )
        )
        if self.demo_cases:
            checks.append(("demo cases", True, str(self.demo_cases)))

        title = "Doctor (startup self-check)" if auto else "Doctor"
        emit(title)
        for name, ok, msg in checks:
            mark = "✅" if ok else "❌"
            emit(f"{mark} {name}: {msg}")

        if provider == "openai_compatible_server":
            emit("ℹ️ openai_compatible_server: set base_url to your vLLM/proxy OpenAI endpoint.")
        elif provider == "openai_official":
            emit("ℹ️ openai_official: relies on OPENAI_API_KEY env/config; no key prompt is shown in shell.")

        if not TEXTUAL_AVAILABLE:
            emit('Install TUI support with: pip install -e ".[tui]"')

        return {
            "checked_at": datetime.utcnow().isoformat() + "Z",
            "ok": all(x[1] for x in checks),
            "checks": [{"name": x[0], "ok": x[1], "detail": x[2]} for x in checks],
        }

    def _looks_like_template_start(self, line: str) -> bool:
        s = str(line or "").lstrip()
        if s.startswith("---"):
            return True
        if self._parse_inline_template_patch(s) is not None:
            return True
        return bool(
            s.startswith("REQUIRED:")
            or s.startswith("OPTIONAL:")
            or s.startswith("GUIDANCE:")
            or s.startswith("{")
        )

    def _collect_multiline(
        self,
        first_line: str,
        *,
        stop_on_blank: bool,
        stop_tokens: Optional[set[str]] = None,
    ) -> str:
        lines = [first_line] if str(first_line) else []
        stop_tokens = stop_tokens or set()
        while True:
            try:
                nxt = input("... ")
            except (EOFError, KeyboardInterrupt):
                break
            stripped = nxt.strip()
            if stripped in stop_tokens:
                break
            if stop_on_blank and not stripped:
                break
            lines.append(nxt)
        return "\n".join(lines)

    def _apply_template_patch(
        self,
        template_patch: Dict[str, Any],
        *,
        emit: Callable[[str], None],
        show_template_on_incomplete: bool,
    ) -> bool:
        prev_domain = self._extract_template_domain(self.session.current_template)
        prev_request_type = self._extract_template_request_type(self.session.current_template)
        patch_domain_raw = self._extract_patch_required_value(template_patch, "domain")
        patch_req_type_raw = self._extract_patch_required_value(template_patch, "request_type")
        patch_domain = str(self.brain._template_value(patch_domain_raw) or "").strip().lower()
        patch_req_type = str(self.brain._template_value(patch_req_type_raw) or "full_pipeline").strip().lower() or "full_pipeline"
        manual_scope_flush = bool(
            (patch_domain and patch_domain != prev_domain) or (patch_req_type_raw is not None and patch_req_type != prev_request_type)
        )
        merged = self._merge_template_payload(self.session.current_template, template_patch)
        explicit_case_ref = self._patch_has_explicit_case_ref(template_patch)
        explicit_case_id = self._patch_has_explicit_case_id(template_patch)
        notes = self._apply_domain_and_case_defaults(
            merged=merged,
            prev_domain=prev_domain,
            new_domain=self._extract_template_domain(merged),
            explicit_case_ref=explicit_case_ref,
            explicit_case_id=explicit_case_id,
            force_scope_flush=manual_scope_flush,
        )
        self.session.current_template = merged
        for line in notes:
            emit(line)

        res = self.brain.from_template(merged, session=self.session)

        if res.kind == "plan" and res.plan is not None:
            self.session.current_plan = res.plan.model_dump()
            emit("Required fields filled. Type ':run' to execute.")
            self._emit_draft_status(emit=emit, include_ready_hint=False)
            self._emit_plan_outline(self.session.current_plan, emit=emit)
            return True

        self.session.current_plan = None
        emit(res.message)
        self._emit_draft_status(emit=emit)
        if show_template_on_incomplete and res.template is not None:
            emit(res.template.to_yaml().rstrip())
        return True

    def _merge_template_payload(self, base: Optional[Dict[str, Any]], patch: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._template_seed()
        if isinstance(base, dict):
            merged = self._deep_merge_dict(merged, base)
        if isinstance(patch, dict):
            merged = self._deep_merge_dict(merged, patch)
        merged.setdefault("REQUIRED", {})
        merged.setdefault("OPTIONAL", {})
        merged.setdefault("GUIDANCE", [])
        return merged

    def _template_seed(self) -> Dict[str, Any]:
        return {
            "REQUIRED": {
                "domain": "",
                "request_type": "full_pipeline",
                "case_ref": "case.input",
            },
            "OPTIONAL": {
                "case_id": self.session.case_id or "",
                "output_format": "markdown",
                "evidence_required": True,
                "constraints": {
                    "retry_on_tool_error": True,
                    "max_attempts": 2,
                },
            },
            "GUIDANCE": [],
        }

    def _deep_merge_dict(self, left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(left)
        for k, v in right.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = self._deep_merge_dict(out[k], v)
            else:
                out[k] = v
        return out

    def _extract_template_domain(self, template_obj: Optional[Dict[str, Any]]) -> str:
        if not isinstance(template_obj, dict):
            return ""
        req = dict(template_obj.get("REQUIRED") or {})
        raw = self.brain._template_value(req.get("domain")) or self.brain._template_value(req.get("anatomy"))
        return str(raw or "").strip().lower()

    def _extract_template_request_type(self, template_obj: Optional[Dict[str, Any]]) -> str:
        if not isinstance(template_obj, dict):
            return "full_pipeline"
        req = dict(template_obj.get("REQUIRED") or {})
        raw = self.brain._template_value(req.get("request_type"))
        return str(raw or "full_pipeline").strip().lower() or "full_pipeline"

    def _extract_template_case_ref(self, template_obj: Optional[Dict[str, Any]]) -> str:
        if not isinstance(template_obj, dict):
            return ""
        req = dict(template_obj.get("REQUIRED") or {})
        opt = dict(template_obj.get("OPTIONAL") or {})
        raw = self.brain._template_value(req.get("case_ref")) or self.brain._template_value(opt.get("case_ref"))
        return str(raw or "").strip()

    def _resolve_case_ref(self, case_ref: str) -> str:
        raw = str(case_ref or "").strip()
        if not raw:
            return ""
        if self._is_default_case_ref(raw):
            bound = str(self.session.path_keys.get("case.input") or self.session.case_inputs.get("case_input_path") or "").strip()
            return bound
        if raw.startswith("@"):
            key = raw[1:]
            mapped = str(self.session.path_keys.get(key) or "").strip()
            if mapped:
                return mapped
        return raw

    def _extract_template_case_id(self, template_obj: Optional[Dict[str, Any]]) -> str:
        if not isinstance(template_obj, dict):
            return ""
        req = dict(template_obj.get("REQUIRED") or {})
        opt = dict(template_obj.get("OPTIONAL") or {})
        raw = self.brain._template_value(req.get("case_id")) or self.brain._template_value(opt.get("case_id"))
        return str(raw or "").strip()

    def _patch_has_explicit_case_ref(self, patch: Dict[str, Any]) -> bool:
        raw = self._extract_template_case_ref(patch)
        if not raw:
            return False
        return not self._is_default_case_ref(raw)

    def _patch_has_explicit_case_id(self, patch: Dict[str, Any]) -> bool:
        return bool(self._extract_template_case_id(patch))

    def _extract_patch_required_value(self, patch: Dict[str, Any], key: str) -> Any:
        if not isinstance(patch, dict):
            return None
        req = patch.get("REQUIRED")
        if isinstance(req, dict) and key in req:
            return req.get(key)
        if key in patch:
            return patch.get(key)
        return None

    def _is_default_case_ref(self, case_ref: str) -> bool:
        raw = str(case_ref or "").strip()
        return raw in {"", "case.input", "@case.input", "case_input", "-"}

    def _normalize_path_str(self, raw_path: str) -> str:
        raw = str(raw_path or "").strip()
        if not raw:
            return ""
        try:
            return str(Path(os.path.abspath(os.path.expanduser(raw))))
        except Exception:
            return raw

    def _derive_case_id_from_ref(self, case_ref: str) -> str:
        p = self._normalize_path_str(case_ref)
        if not p or self._is_default_case_ref(p):
            return ""
        try:
            return Path(p).name
        except Exception:
            return ""

    def _apply_domain_and_case_defaults(
        self,
        *,
        merged: Dict[str, Any],
        prev_domain: str,
        new_domain: str,
        explicit_case_ref: bool,
        explicit_case_id: bool,
        force_scope_flush: bool = False,
    ) -> List[str]:
        notes: List[str] = []
        req = dict(merged.get("REQUIRED") or {})
        opt = dict(merged.get("OPTIONAL") or {})
        merged["REQUIRED"] = req
        merged["OPTIONAL"] = opt

        domain_changed = bool(new_domain) and (new_domain != str(prev_domain or "").strip().lower())
        case_ref = self._extract_template_case_ref(merged)

        if force_scope_flush:
            self.session.case_id = None
            self.session.case_inputs.pop("case_input_path", None)
            self.session.path_keys.pop("case.input", None)
            self._case_source = "unset"
            self.session.current_plan = None
            if not explicit_case_ref:
                req["case_ref"] = "case.input"
                case_ref = "case.input"
            if not explicit_case_id:
                opt["case_id"] = ""
            notes.append("Manual scope change detected: cleared prior case binding.")

        # User-provided case_ref always wins and is normalized.
        if explicit_case_ref:
            norm = self._normalize_path_str(case_ref)
            if norm:
                req["case_ref"] = norm
                if Path(norm).exists():
                    self.session.case_inputs["case_input_path"] = norm
                    self.session.path_keys["case.input"] = norm
                    self._case_source = "template"
                    case_ref = norm
                else:
                    notes.append(f"Warning: case_ref does not exist yet: {norm}")

        # Auto-switch demo case on domain entry/switch unless user manually pinned a case.
        demo_path = self.demo_cases.get(new_domain) if new_domain else None
        should_auto_case = bool(demo_path) and (self._case_source != "manual") and (not explicit_case_ref) and (
            self._is_default_case_ref(case_ref) or (domain_changed and self._case_source == "auto")
        )
        if should_auto_case and demo_path:
            case_ref = self._normalize_path_str(demo_path)
            req["case_ref"] = case_ref
            self.session.case_inputs["case_input_path"] = case_ref
            self.session.path_keys["case.input"] = case_ref
            cid = Path(case_ref).name
            if not explicit_case_id:
                opt["case_id"] = cid
                self.session.case_id = cid
            self._case_source = "auto"
            notes.append(f"Detected domain '{new_domain}'. Auto-loading default case: {cid}")

        # Domain switch must reset/rederive case_id unless user explicitly set one this turn.
        if explicit_case_id:
            cid = self._extract_template_case_id(merged)
            if cid:
                opt["case_id"] = cid
                self.session.case_id = cid
        elif domain_changed:
            derived = self._derive_case_id_from_ref(case_ref)
            if derived:
                opt["case_id"] = derived
                self.session.case_id = derived
            else:
                opt["case_id"] = ""
                self.session.case_id = None
        elif explicit_case_ref:
            derived = self._derive_case_id_from_ref(case_ref)
            if derived:
                opt["case_id"] = derived
                self.session.case_id = derived

        return notes

    def _parse_inline_template_patch(self, text: str) -> Optional[Dict[str, Any]]:
        s = str(text or "").strip()
        if not s or ":" not in s:
            return None
        if s.startswith("{") or s.startswith("REQUIRED:") or s.startswith("OPTIONAL:") or s.startswith("GUIDANCE:"):
            return None
        if "\n" in s:
            return None

        matches = list(self._INLINE_KV_RE.finditer(s))
        if not matches:
            return None

        required: Dict[str, Any] = {}
        optional: Dict[str, Any] = {}
        recognized = False

        for m in matches:
            key = str(m.group(1) or "").strip()
            raw_val = str(m.group(2) or "").strip().rstrip(",")
            if not key:
                continue

            value = self._coerce_inline_value(raw_val)
            low = key.lower()

            if low.startswith("required."):
                required[low.split(".", 1)[1]] = value
                recognized = True
                continue
            if low.startswith("optional."):
                optional[low.split(".", 1)[1]] = value
                recognized = True
                continue
            if low in {"domain", "request_type", "case_ref", "anatomy"}:
                required[low] = value
                recognized = True
                continue
            if low == "case_id":
                optional["case_id"] = value
                recognized = True
                continue
            if low in {"output_format", "evidence_required"}:
                optional[low] = value
                recognized = True
                continue
            if low.startswith("constraints."):
                constraints = optional.setdefault("constraints", {})
                if isinstance(constraints, dict):
                    constraints[low.split(".", 1)[1]] = value
                    recognized = True
                continue

        if not recognized:
            return None
        return {"REQUIRED": required, "OPTIONAL": optional}

    def _coerce_inline_value(self, raw_val: str) -> Any:
        s = str(raw_val or "").strip()
        if not s:
            return ""
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()

        low = s.lower()
        if low == "true":
            return True
        if low == "false":
            return False
        if low in {"null", "none"}:
            return None
        if re.fullmatch(r"-?\d+", s):
            try:
                return int(s)
            except Exception:
                return s
        if re.fullmatch(r"-?\d+\.\d+", s):
            try:
                return float(s)
            except Exception:
                return s
        return s

    def _extract_case_ref_from_text(self, text: str) -> str:
        s = str(text or "")
        candidates: List[str] = []

        for q in re.findall(r"""["']([^"']+)["']""", s):
            qq = str(q or "").strip()
            if qq:
                candidates.append(qq)

        for tok in s.split():
            t = str(tok or "").strip().strip(",;:.!?)]}(")
            if t.startswith("/") or t.startswith("~/"):
                candidates.append(t)

        for raw in candidates:
            try:
                p = Path(str(raw)).expanduser()
                if not p.is_absolute():
                    continue
                if p.exists() and p.is_dir():
                    return str(p.resolve())
            except Exception:
                continue
        return ""

    def _wants_legacy_template_mode(self, text: str) -> bool:
        if str(os.environ.get("MRI_AGENT_SHELL_FORCE_TEMPLATE", "")).strip().lower() in {"1", "true", "yes", "on"}:
            return True
        s = str(text or "").lower()
        hints = (
            "legacy template",
            "template mode",
            "old template",
            "deterministic v1",
            "use v1",
            "fallback template",
        )
        return any(h in s for h in hints)

    def _planner_kwargs_from_model_config(self) -> tuple[str, Dict[str, Any]]:
        cfg = self.session.model_config
        provider = str(cfg.provider or "").strip().lower()
        common: Dict[str, Any] = {"max_new_tokens": int(cfg.max_tokens or 2048)}

        if provider == "stub":
            return "stub", common
        if provider == "openai_compatible_server":
            from llm.adapter_vllm_server import VLLMServerConfig

            server_cfg = VLLMServerConfig(
                base_url=str(cfg.server_base_url or "").strip() or "http://127.0.0.1:8000/v1",
                model=str(cfg.llm or "").strip(),
                max_tokens=int(cfg.max_tokens or 2048),
                temperature=float(cfg.temperature or 0.0),
            )
            common["server_cfg"] = server_cfg
            return "server", common
        if provider == "openai_official":
            common["api_model"] = str(cfg.llm or "").strip() or None
            common["api_base_url"] = str(cfg.api_base_url or "").strip() or None
            return "openai", common
        if provider == "anthropic":
            common["api_model"] = str(cfg.llm or "").strip() or None
            return "anthropic", common
        if provider == "gemini":
            common["api_model"] = str(cfg.llm or "").strip() or None
            return "gemini", common
        return "stub", common

    def _plan_dag_from_langgraph(
        self,
        *,
        goal: str,
        domain: str,
        request_type: str,
        case_ref: str,
        case_id: str,
    ) -> tuple[Optional[Dict[str, Any]], str]:
        try:
            from agent.langgraph.loop import plan_agent_dag
        except Exception as e:
            return None, f"planner import failed: {type(e).__name__}: {e}"

        try:
            llm_mode, kwargs = self._planner_kwargs_from_model_config()
            goal_text = str(goal or "").strip()
            if str(request_type or "").strip():
                goal_text = f"{goal_text}\nRequested workflow type: {str(request_type).strip().lower()}."
            dag = plan_agent_dag(
                goal=goal_text,
                domain=str(domain),
                case_ref=str(case_ref),
                case_id=(str(case_id or "").strip() or None),
                request_type=(str(request_type or "").strip().lower() or None),
                llm_mode=llm_mode,
                workspace_root=str(self.session.workspace_path),
                runs_root=str(self.session.runs_root),
                **kwargs,
            )
            if hasattr(dag, "model_dump"):
                return dict(dag.model_dump()), ""
            if isinstance(dag, dict):
                return dict(dag), ""
            return None, "planner returned unknown DAG object"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"

    def _is_dag_plan(self, plan_obj: Any) -> bool:
        if not isinstance(plan_obj, dict):
            return False
        return isinstance(plan_obj.get("case_scope"), dict) and isinstance(plan_obj.get("nodes"), list)

    def _emit_plan_outline(self, plan_obj: Any, *, emit: Callable[[str], None]) -> None:
        if not isinstance(plan_obj, dict):
            return
        if self._is_dag_plan(plan_obj):
            nodes = plan_obj.get("nodes") if isinstance(plan_obj.get("nodes"), list) else []
            emit("Draft DAG:")
            for i, node in enumerate(nodes, start=1):
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("node_id") or f"node_{i:03d}")
                label = str(node.get("label") or "").strip()
                tool_name = str(node.get("tool_name") or "")
                node_type = str(node.get("node_type") or "tool")
                stage = str(node.get("stage") or "misc")
                required = bool(node.get("required", True))
                deps = [str(d).strip() for d in (node.get("depends_on") or []) if str(d).strip()]
                dep_s = ",".join(deps) if deps else "-"
                if label:
                    emit(
                        f"  {i}. {node_id} {label} | type={node_type} tool={tool_name} "
                        f"stage={stage} required={required} depends_on={dep_s}"
                    )
                else:
                    emit(
                        f"  {i}. {node_id} type={node_type} tool={tool_name} "
                        f"stage={stage} required={required} depends_on={dep_s}"
                    )
            return

        steps = plan_obj.get("steps") if isinstance(plan_obj.get("steps"), list) else []
        for i, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            emit(f"  {i}. {step.get('tool_name')} stage={step.get('stage')} required={step.get('required')}")

    def _emit_draft_status(self, *, emit: Callable[[str], None], include_ready_hint: bool = True) -> None:
        tpl = self.session.current_template or {}
        req = dict(tpl.get("REQUIRED") or {})
        opt = dict(tpl.get("OPTIONAL") or {})

        domain = self.brain._template_value(req.get("domain")) or self.brain._template_value(req.get("anatomy")) or ""
        request_type = self.brain._template_value(req.get("request_type")) or "full_pipeline"
        case_ref = self.brain._template_value(req.get("case_ref")) or self.brain._template_value(opt.get("case_ref")) or ""
        case_ref_resolved = self._resolve_case_ref(str(case_ref))

        missing: List[str] = []
        if not str(domain).strip():
            missing.append("domain")
        if str(request_type).strip().lower() != "report":
            if not str(case_ref_resolved).strip():
                missing.append("case_ref")
            elif self._is_default_case_ref(str(case_ref)) and not str(self.session.path_keys.get("case.input") or "").strip():
                missing.append("case_ref")

        display_case_ref = str(case_ref_resolved or case_ref or "-").strip() or "-"
        if self._is_default_case_ref(str(case_ref)) and not str(case_ref_resolved).strip():
            display_case_ref = "-"

        case_id = str(opt.get("case_id") or self.session.case_id or "").strip() or "-"
        if case_id == "-" and str(case_ref_resolved).strip():
            case_id = self._derive_case_id_from_ref(case_ref_resolved) or "-"

        emit(
            "Draft context: "
            f"domain={domain or '-'} "
            f"case_id={case_id} "
            f"case_ref={display_case_ref} "
            f"request_type={request_type or '-'}"
        )
        if missing:
            emit(f"Missing required fields: {', '.join(sorted(set(missing)))}")
        elif include_ready_hint:
            emit("Required fields filled. Type ':run' to execute.")

    def _discover_demo_cases(self) -> Dict[str, str]:
        demo_root = Path(__file__).resolve().parents[2] / "demo" / "cases"
        if not demo_root.exists():
            return {}

        pref: Dict[str, List[str]] = {
            "prostate": [
                "sub-019_2",
                "sub-057",
                "sub001",
            ],
            "brain": [
                "Brats18_CBICA_AAM_1",
            ],
            "cardiac": [
                "acdc_multiseq_patient061_ed",
            ],
        }
        out: Dict[str, str] = {}
        for domain, names in pref.items():
            for name in names:
                p = Path(name).expanduser()
                if not p.is_absolute():
                    p = demo_root / name
                if p.exists():
                    out[domain] = str(Path(os.path.abspath(str(p))))
                    break
        return out

    def _plan_references_key(self, plan_obj: Any, key: str) -> bool:
        marker = f"@{key}"
        if isinstance(plan_obj, dict):
            return any(self._plan_references_key(v, key) for v in plan_obj.values())
        if isinstance(plan_obj, list):
            return any(self._plan_references_key(v, key) for v in plan_obj)
        if isinstance(plan_obj, str):
            return marker in plan_obj
        return False

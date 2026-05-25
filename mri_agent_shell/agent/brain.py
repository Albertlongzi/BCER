from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field

from ..bootstrap import ensure_import_paths
from ..runtime.session import ModelConfig, SessionState

ensure_import_paths()

from llm.adapter_anthropic_api import AnthropicChatAdapter, AnthropicConfig  # noqa: E402
from llm.adapter_gemini_api import GeminiChatAdapter, GeminiConfig  # noqa: E402
from llm.adapter_openai_api import OpenAIChatAdapter, OpenAIConfig  # noqa: E402
from llm.adapter_vllm_server import VLLMOpenAIChatAdapter, VLLMServerConfig  # noqa: E402


DEFAULT_SUPPORTED_FAMILIES = {
    "prostate": ["full_pipeline", "segment", "register", "lesion", "report", "denoise", "super_resolution", "custom_analysis"],
    "brain": ["full_pipeline", "segment", "report", "denoise", "super_resolution", "custom_analysis"],
    "cardiac": ["full_pipeline", "segment", "classify", "report", "denoise", "super_resolution", "raw_recon", "custom_analysis"],
}


class PlanStep(BaseModel):
    tool_name: str
    stage: str = "misc"
    arguments: Dict[str, Any] = Field(default_factory=dict)
    required: bool = True
    depends_on: List[str] = Field(default_factory=list)
    description: str = ""


class ExecutionPlan(BaseModel):
    goal: str
    domain: str
    request_type: str
    steps: List[PlanStep]
    case_ref: Optional[str] = None
    case_id: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class TemplateRequest(BaseModel):
    domain: str
    request_type: str
    required: Dict[str, Any] = Field(default_factory=dict)
    optional: Dict[str, Any] = Field(default_factory=dict)
    guidance: List[str] = Field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "REQUIRED": self.required,
            "OPTIONAL": self.optional,
            "GUIDANCE": self.guidance,
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_payload(), sort_keys=False, allow_unicode=True)


@dataclass
class BrainResult:
    kind: str  # template | plan | unsupported | error
    message: str
    template: Optional[TemplateRequest] = None
    plan: Optional[ExecutionPlan] = None
    data: Optional[Dict[str, Any]] = None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = str(text or "")
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                frag = s[start : i + 1]
                try:
                    obj = json.loads(frag)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


class Brain:
    def __init__(
        self,
        tool_specs: List[Dict[str, Any]],
        *,
        domain_catalog: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.tool_specs = tool_specs or []
        self.tool_names = sorted(
            [str(t.get("name")) for t in self.tool_specs if isinstance(t, dict) and t.get("name")]
        )
        if isinstance(domain_catalog, dict) and domain_catalog:
            self.domain_catalog = domain_catalog
            supported: Dict[str, List[str]] = {}
            for dom, rec in domain_catalog.items():
                caps = rec.get("capabilities") if isinstance(rec, dict) else None
                if isinstance(caps, list) and caps:
                    supported[str(dom)] = sorted(set([str(x) for x in caps if str(x).strip()]))
            self.supported_families = supported or dict(DEFAULT_SUPPORTED_FAMILIES)
        else:
            self.domain_catalog = {
                k: {"capabilities": list(v), "tool_count": 0, "tools": []}
                for k, v in DEFAULT_SUPPORTED_FAMILIES.items()
            }
            self.supported_families = dict(DEFAULT_SUPPORTED_FAMILIES)

    def domain_names(self) -> List[str]:
        return sorted([d for d in self.supported_families.keys() if d != "demo"])

    def parse_template_text(self, text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if not raw:
            raise ValueError("Template text is empty.")

        # JSON first if it starts with '{'.
        if raw.startswith("{"):
            obj = json.loads(raw)
        else:
            obj = yaml.safe_load(raw)

        if not isinstance(obj, dict):
            raise ValueError("Template must parse to a dictionary.")

        if "REQUIRED" not in obj and "OPTIONAL" not in obj:
            # Accept flat shape; normalize into REQUIRED/OPTIONAL.
            obj = {
                "REQUIRED": {k: v for k, v in obj.items() if k not in {"guidance", "GUIDANCE", "optional", "OPTIONAL"}},
                "OPTIONAL": obj.get("optional") or obj.get("OPTIONAL") or {},
                "GUIDANCE": obj.get("guidance") or obj.get("GUIDANCE") or [],
            }

        if not isinstance(obj.get("REQUIRED") or {}, dict):
            raise ValueError("Template REQUIRED must be an object.")
        if not isinstance(obj.get("OPTIONAL") or {}, dict):
            raise ValueError("Template OPTIONAL must be an object.")
        return obj

    def from_request(self, request: str, session: SessionState) -> BrainResult:
        text = str(request or "").strip()
        if not text:
            return BrainResult(kind="error", message="Request is empty.")

        session.current_request = text
        domain, request_type = self._classify_request(text=text, model_cfg=session.model_config)

        if domain and domain not in self.supported_families:
            return BrainResult(
                kind="unsupported",
                message=f"Domain '{domain}' is not supported by currently registered tools.",
                data={"supported": self.supported_families},
            )
        if domain and request_type not in (self.supported_families.get(domain) or []):
            return BrainResult(
                kind="unsupported",
                message=f"Request type '{request_type}' is not supported for domain '{domain}'.",
                data={"supported_for_domain": self.supported_families.get(domain) or []},
            )

        required = self._required_fields_for(domain=domain, request_type=request_type)
        if self._is_underspecified(required=required, session=session, request_text=text, domain=domain):
            tpl = self._build_template(domain=domain, request_type=request_type, session=session)
            return BrainResult(
                kind="template",
                message="Request is underspecified. Fill REQUIRED fields and paste back.",
                template=tpl,
                data=tpl.to_payload(),
            )

        payload = {
            "REQUIRED": {
                "domain": domain or "",
                "request_type": request_type,
                "case_ref": self._best_case_ref(session),
            },
            "OPTIONAL": {
                "case_id": session.case_id,
                "output_format": "markdown",
                "evidence_required": True,
            },
        }
        return self.from_template(payload, session=session, goal=text)

    def from_template(self, template_obj: Dict[str, Any], session: SessionState, goal: Optional[str] = None) -> BrainResult:
        req = dict(template_obj.get("REQUIRED") or {})
        opt = dict(template_obj.get("OPTIONAL") or {})

        domain = self._template_value(req.get("domain")) or self._template_value(req.get("anatomy"))
        domain = str(domain or "").strip().lower()
        request_type = self._template_value(req.get("request_type"))
        request_type = str(request_type or "full_pipeline").strip().lower()
        # Normalize common aliases so that e.g. "resample" maps to "super_resolution".
        _req_aliases = {
            "resample": "super_resolution", "resampling": "super_resolution",
            "superres": "super_resolution", "superresolution": "super_resolution",
            "super-resolution": "super_resolution", "upsample": "super_resolution",
            "regrid": "super_resolution",
            "denoising": "denoise", "bm3d": "denoise",
            "registration": "register", "segmentation": "segment",
            "classification": "classify",
        }
        request_type = _req_aliases.get(request_type, request_type)

        if not domain:
            return BrainResult(
                kind="template",
                message="Domain is required. Choose one from domain choices.",
                template=self._build_template(domain="", request_type=request_type, session=session),
            )

        if domain not in self.supported_families:
            return BrainResult(
                kind="unsupported",
                message=f"Unsupported domain '{domain}'.",
                data={"supported_domains": sorted(self.supported_families.keys())},
            )
        if request_type not in (self.supported_families.get(domain) or []):
            return BrainResult(
                kind="unsupported",
                message=f"Unsupported request_type '{request_type}' for domain '{domain}'.",
                data={"supported_request_types": self.supported_families.get(domain) or []},
            )

        case_ref = self._template_value(req.get("case_ref")) or self._template_value(opt.get("case_ref")) or self._best_case_ref(session)
        case_id = str(
            self._template_value(req.get("case_id"))
            or self._template_value(opt.get("case_id"))
            or session.case_id
            or ""
        ).strip() or None

        if session.dry_run:
            if not case_ref:
                return BrainResult(
                    kind="template",
                    message="Dry-run still needs a case_ref or :case load path.",
                    template=self._build_template(domain=domain, request_type=request_type, session=session),
                )
            plan = self._build_dummy_plan(goal=goal or session.current_request or "dry-run", case_ref=str(case_ref), case_id=case_id)
            return BrainResult(kind="plan", message="Execution plan created.", plan=plan, data=plan.model_dump())

        if request_type != "report" and not self._has_resolved_case_ref(case_ref=case_ref, session=session):
            return BrainResult(
                kind="template",
                message="Missing case_ref. Provide it in REQUIRED.case_ref or use :case load <path>.",
                template=self._build_template(domain=domain, request_type=request_type, session=session),
            )

        plan = self._build_plan(
            domain=domain,
            request_type=request_type,
            goal=goal or session.current_request or f"{domain} {request_type}",
            case_ref=str(case_ref) if case_ref else "",
            case_id=case_id,
        )
        return BrainResult(kind="plan", message="Execution plan created.", plan=plan, data=plan.model_dump())

    def _has_resolved_case_ref(self, *, case_ref: Any, session: SessionState) -> bool:
        raw = str(case_ref or "").strip()
        if not raw:
            return False
        if raw in {"case.input", "@case.input", "case_input"}:
            bound = str(session.case_inputs.get("case_input_path") or "").strip()
            return bool(bound)
        return True

    def _template_value(self, raw: Any) -> Any:
        if isinstance(raw, dict):
            for key in ("value", "selected", "default"):
                if key in raw and str(raw.get(key) or "").strip():
                    return raw.get(key)
            return ""
        return raw

    def _is_underspecified(
        self,
        *,
        required: List[str],
        session: SessionState,
        request_text: str,
        domain: str,
    ) -> bool:
        if not str(domain or "").strip():
            return True
        if "case_ref" in required:
            if self._find_explicit_path(request_text):
                return False
            if self._best_case_ref(session):
                return False
            return True
        return False

    def _required_fields_for(self, *, domain: str, request_type: str) -> List[str]:
        if not str(domain or "").strip():
            return ["domain", "case_ref"]
        if request_type in {"full_pipeline", "segment", "register", "lesion", "classify"}:
            return ["case_ref"]
        if request_type == "report":
            # report can run from prior state or current case.
            return []
        return ["case_ref"]

    def _build_template(self, *, domain: str, request_type: str, session: SessionState) -> TemplateRequest:
        domains = self.domain_names()
        selected_domain = str(domain or "").strip().lower()
        domain_choices = domains or sorted(DEFAULT_SUPPORTED_FAMILIES.keys())
        req_type_choices = (
            (self.supported_families.get(selected_domain) or ["full_pipeline"])
            if selected_domain
            else sorted(
                {
                    cap
                    for caps in self.supported_families.values()
                    for cap in (caps or [])
                }
            )
        )
        selected_req_type = str(request_type or "").strip().lower() or "full_pipeline"
        if selected_req_type not in req_type_choices:
            selected_req_type = req_type_choices[0] if req_type_choices else "full_pipeline"

        required: Dict[str, Any] = {
            "domain": {
                "value": selected_domain,
                "choices": domain_choices,
            },
            "request_type": {
                "value": selected_req_type,
                "choices": req_type_choices,
            },
            "case_ref": self._best_case_ref(session) or "case.input",
        }

        optional: Dict[str, Any] = {
            "case_id": session.case_id or "",
            "output_format": "markdown",
            "evidence_required": True,
            "constraints": {
                "retry_on_tool_error": True,
                "max_attempts": 2,
            },
        }
        guidance = [
            "Use path keys when possible (e.g., case.input) instead of raw filesystem paths.",
            "You can load a case first with :case load <path>.",
            "After filling template, paste YAML/JSON and run :run.",
        ]
        return TemplateRequest(domain=domain, request_type=request_type, required=required, optional=optional, guidance=guidance)

    def _best_case_ref(self, session: SessionState) -> str:
        if session.path_keys.get("case.input"):
            return "case.input"
        p = session.case_inputs.get("case_input_path")
        return str(p or "")

    def _build_plan(
        self,
        *,
        domain: str,
        request_type: str,
        goal: str,
        case_ref: str,
        case_id: Optional[str],
    ) -> ExecutionPlan:
        # Represent path keys with '@<key>' so cerebellum can resolve deterministically.
        case_arg = f"@{case_ref}" if case_ref and not str(case_ref).startswith("/") else str(case_ref)

        if domain == "prostate":
            steps = self._plan_prostate(request_type=request_type, case_arg=case_arg)
        elif domain == "brain":
            steps = self._plan_brain(request_type=request_type, case_arg=case_arg)
        elif domain == "cardiac":
            steps = self._plan_cardiac(request_type=request_type, case_arg=case_arg)
        else:
            steps = []

        return ExecutionPlan(
            goal=goal,
            domain=domain,
            request_type=request_type,
            steps=steps,
            case_ref=case_ref,
            case_id=case_id,
            notes=["Plan generated by shell brain (deterministic v1)."],
        )

    def _build_dummy_plan(self, *, goal: str, case_ref: str, case_id: Optional[str]) -> ExecutionPlan:
        steps = [
            PlanStep(
                tool_name="dummy_load_case",
                stage="ingest",
                arguments={"case_path": f"@{case_ref}" if not str(case_ref).startswith("/") else str(case_ref), "output_subdir": "dummy/load"},
                required=True,
                depends_on=[],
                description="Load case metadata (dummy).",
            ),
            PlanStep(
                tool_name="dummy_segment",
                stage="segment",
                arguments={"case_path": f"@{case_ref}" if not str(case_ref).startswith("/") else str(case_ref), "anatomy": "demo", "output_subdir": "dummy/segmentation"},
                required=True,
                depends_on=["dummy_load_case"],
                description="Simulate segmentation (dummy).",
            ),
            PlanStep(
                tool_name="dummy_generate_report",
                stage="report",
                arguments={"case_id": case_id or "dummy_case", "mask_path": "@dummy_segment.mask", "output_subdir": "dummy/report"},
                required=True,
                depends_on=["dummy_segment"],
                description="Write a dummy report.",
            ),
        ]
        return ExecutionPlan(
            goal=goal,
            domain="demo",
            request_type="full_pipeline",
            steps=steps,
            case_ref=case_ref,
            case_id=case_id,
        )

    def _plan_prostate(self, *, request_type: str, case_arg: str) -> List[PlanStep]:
        steps: List[PlanStep] = []

        if request_type in {"full_pipeline", "segment", "register", "lesion"}:
            steps.append(
                PlanStep(
                    tool_name="identify_sequences",
                    stage="identify",
                    arguments={"dicom_case_dir": case_arg, "convert_to_nifti": True, "output_subdir": "ingest"},
                    required=True,
                    depends_on=[],
                    description="Identify T2/ADC/DWI sequences.",
                )
            )

        if request_type in {"full_pipeline", "register", "lesion"}:
            steps.extend(
                [
                    PlanStep(
                        tool_name="register_to_reference",
                        stage="register",
                        arguments={"fixed": "T2w", "moving": "ADC", "output_subdir": "registration/ADC", "split_by_bvalue": False},
                        required=True,
                        depends_on=["identify_sequences"],
                        description="Register ADC to T2w.",
                    ),
                    PlanStep(
                        tool_name="register_to_reference",
                        stage="register",
                        arguments={"fixed": "T2w", "moving": "DWI", "output_subdir": "registration/DWI", "split_by_bvalue": True},
                        required=True,
                        depends_on=["identify_sequences"],
                        description="Register DWI to T2w.",
                    ),
                ]
            )

        if request_type in {"full_pipeline", "segment", "lesion"}:
            steps.append(
                PlanStep(
                    tool_name="segment_prostate",
                    stage="segment",
                    arguments={"t2w_ref": "T2w", "output_subdir": "segmentation"},
                    required=True,
                    depends_on=["identify_sequences"],
                    description="Segment prostate gland/zones.",
                )
            )

        if request_type in {"full_pipeline", "segment", "lesion"}:
            steps.append(
                PlanStep(
                    tool_name="extract_roi_features",
                    stage="extract",
                    arguments={"images": [], "output_subdir": "features"},
                    required=(request_type in {"full_pipeline", "lesion"}),
                    depends_on=["segment_prostate"],
                    description="Extract ROI features.",
                )
            )

        if request_type in {"full_pipeline", "lesion"}:
            steps.append(
                PlanStep(
                    tool_name="detect_lesion_candidates",
                    stage="extract",
                    arguments={"output_subdir": "lesion"},
                    required=(request_type == "lesion"),
                    depends_on=["segment_prostate"],
                    description="Detect lesion candidates.",
                )
            )

        if request_type in {"full_pipeline", "segment", "register", "lesion", "report"}:
            steps.extend(
                [
                    PlanStep(
                        tool_name="package_vlm_evidence",
                        stage="report",
                        arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "vlm"},
                        required=False,
                        depends_on=[],
                        description="Package evidence for report/VLM.",
                    ),
                    PlanStep(
                        tool_name="generate_report",
                        stage="report",
                        arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "report", "domain": "prostate"},
                        required=True,
                        depends_on=[],
                        description="Generate final report.",
                    ),
                ]
            )
        return steps

    def _plan_brain(self, *, request_type: str, case_arg: str) -> List[PlanStep]:
        steps: List[PlanStep] = []
        if request_type in {"full_pipeline", "segment"}:
            steps.append(
                PlanStep(
                    tool_name="identify_sequences",
                    stage="identify",
                    arguments={"dicom_case_dir": case_arg, "convert_to_nifti": True, "output_subdir": "ingest"},
                    required=True,
                    depends_on=[],
                    description="Identify MRI sequences.",
                )
            )
            steps.append(
                PlanStep(
                    tool_name="brats_mri_segmentation",
                    stage="segment",
                    arguments={
                        "t1c_path": "T1c",
                        "t1_path": "T1",
                        "t2_path": "T2",
                        "flair_path": "FLAIR",
                        "output_subdir": "segmentation/brats_mri_segmentation",
                    },
                    required=True,
                    depends_on=["identify_sequences"],
                    description="Segment tumor subregions.",
                )
            )
            steps.append(
                PlanStep(
                    tool_name="extract_roi_features",
                    stage="extract",
                    arguments={"images": [], "output_subdir": "features", "roi_interpretation": "binary_masks"},
                    required=(request_type == "full_pipeline"),
                    depends_on=["brats_mri_segmentation"],
                    description="Extract tumor ROI features.",
                )
            )

        if request_type in {"full_pipeline", "segment", "report"}:
            steps.extend(
                [
                    PlanStep(
                        tool_name="package_vlm_evidence",
                        stage="report",
                        arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "vlm"},
                        required=False,
                        depends_on=[],
                        description="Package evidence.",
                    ),
                    PlanStep(
                        tool_name="generate_report",
                        stage="report",
                        arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "report", "domain": "brain"},
                        required=True,
                        depends_on=[],
                        description="Generate final report.",
                    ),
                ]
            )
        return steps

    def _plan_cardiac(self, *, request_type: str, case_arg: str) -> List[PlanStep]:
        steps: List[PlanStep] = []
        if request_type in {"full_pipeline", "segment", "classify"}:
            steps.append(
                PlanStep(
                    tool_name="identify_sequences",
                    stage="identify",
                    arguments={"dicom_case_dir": case_arg, "convert_to_nifti": True, "output_subdir": "ingest"},
                    required=True,
                    depends_on=[],
                    description="Identify cine sequences.",
                )
            )
            steps.append(
                PlanStep(
                    tool_name="segment_cardiac_cine",
                    stage="segment",
                    arguments={"cine_path": "CINE", "output_subdir": "segmentation/cardiac_cine"},
                    required=True,
                    depends_on=["identify_sequences"],
                    description="Segment cardiac cine.",
                )
            )

        if request_type in {"full_pipeline", "classify"}:
            steps.append(
                PlanStep(
                    tool_name="classify_cardiac_cine_disease",
                    stage="report",
                    arguments={"output_subdir": "classification/cardiac_cine"},
                    required=True,
                    depends_on=["segment_cardiac_cine"],
                    description="Classify cardiac disease group.",
                )
            )
            steps.append(
                PlanStep(
                    tool_name="extract_roi_features",
                    stage="extract",
                    arguments={"images": [], "output_subdir": "features", "roi_interpretation": "cardiac"},
                    required=True,
                    depends_on=["segment_cardiac_cine"],
                    description="Extract cardiac ROI features.",
                )
            )

        if request_type in {"full_pipeline", "segment", "classify", "report"}:
            steps.extend(
                [
                    PlanStep(
                        tool_name="package_vlm_evidence",
                        stage="report",
                        arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "vlm"},
                        required=False,
                        depends_on=[],
                        description="Package evidence.",
                    ),
                    PlanStep(
                        tool_name="generate_report",
                        stage="report",
                        arguments={"case_state_path": "@runtime.case_state_path", "output_subdir": "report", "domain": "cardiac"},
                        required=True,
                        depends_on=[],
                        description="Generate final report.",
                    ),
                ]
            )
        return steps

    def _classify_request(self, *, text: str, model_cfg: ModelConfig) -> Tuple[str, str]:
        domain, req_type = self._classify_heuristic(text)
        if domain and req_type:
            return domain, req_type

        llm_guess = self._classify_with_llm(text=text, model_cfg=model_cfg)
        if llm_guess is not None:
            return llm_guess

        # Keep domain undecided when ambiguous; template will expose choices.
        return "", (req_type or "full_pipeline")

    def _classify_heuristic(self, text: str) -> Tuple[str, str]:
        s = str(text or "").lower()

        # domain
        domain = ""
        if any(k in s for k in ("cardiac", "cine", "acdc", "left ventricle", "right ventricle", "myocard")):
            domain = "cardiac"
        elif any(k in s for k in ("brain", "brats", "glioma", "tumor", "flair", "t1ce", "t1c")):
            domain = "brain"
        elif any(k in s for k in ("prostate", "pirads", "lesion", "adc", "dwi", "t2w")):
            domain = "prostate"

        # request type
        req_type = ""
        if any(k in s for k in ("denoise", "denoising", "de-noise", "bm3d", "noise reduction", "noise2void", "n2v")):
            req_type = "denoise"
        elif any(k in s for k in ("resample", "resampling", "regrid", "super resolution", "super-resolution", "superres", "upsample")):
            req_type = "super_resolution"
        elif any(k in s for k in ("register", "registration", "align")):
            req_type = "register"
        elif any(k in s for k in ("lesion", "candidate")):
            req_type = "lesion"
        elif any(k in s for k in ("classify", "classification", "diagnosis")) and domain == "cardiac":
            req_type = "classify"
        elif any(k in s for k in ("segment", "segmentation")):
            req_type = "segment"
        elif any(k in s for k in ("report", "summarize")):
            req_type = "full_pipeline"
        elif any(k in s for k in ("process", "analyze")):
            req_type = "full_pipeline"

        if domain and not req_type:
            req_type = "full_pipeline"
        return domain, req_type

    def _find_explicit_path(self, text: str) -> str:
        # Very lightweight path detector.
        m = re.search(r"(/[^\s]+)", str(text or ""))
        if not m:
            return ""
        path = m.group(1)
        return str(Path(path).expanduser())

    def _classify_with_llm(self, *, text: str, model_cfg: ModelConfig) -> Optional[Tuple[str, str]]:
        provider = str(model_cfg.provider or "").strip().lower()
        if provider == "stub":
            return None

        llm = self._build_classifier_adapter(model_cfg)
        if llm is None:
            return None

        sys_prompt = (
            "You classify MRI-agent user requests. Return JSON only: "
            '{"domain":"prostate|brain|cardiac","request_type":"full_pipeline|segment|register|lesion|classify|report|denoise|super_resolution|custom_analysis"}. '
            "No extra keys."
        )
        user_prompt = json.dumps({"request": text, "supported": self.supported_families}, ensure_ascii=False)

        try:
            raw = llm.generate([{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}])
        except Exception:
            return None

        obj = _extract_json_object(raw)
        if not isinstance(obj, dict):
            return None

        d = str(obj.get("domain") or "").strip().lower()
        t = str(obj.get("request_type") or "").strip().lower()
        if d in self.supported_families and t in (self.supported_families.get(d) or []):
            return d, t
        return None

    def _build_classifier_adapter(self, cfg: ModelConfig):
        provider = str(cfg.provider or "").strip().lower()
        llm_name = str(cfg.llm or "").strip()
        if not llm_name:
            return None

        if provider == "openai_compatible_server":
            c = VLLMServerConfig(
                base_url=str(cfg.server_base_url or "http://127.0.0.1:8000/v1"),
                model=llm_name,
                api_key=(cfg.effective_api_key() or "EMPTY"),
                timeout_s=30,
                max_tokens=256,
                temperature=0.0,
            )
            return VLLMOpenAIChatAdapter(c)
        if provider == "openai_official":
            c = OpenAIConfig(model=llm_name, base_url=(cfg.api_base_url or None), timeout_s=30, max_tokens=256, temperature=0.0)
            return OpenAIChatAdapter(c)
        if provider == "anthropic":
            c = AnthropicConfig(model=llm_name, timeout_s=30, max_tokens=256, temperature=0.0)
            return AnthropicChatAdapter(c)
        if provider == "gemini":
            c = GeminiConfig(model=llm_name, max_tokens=256, temperature=0.0)
            return GeminiChatAdapter(c)
        return None

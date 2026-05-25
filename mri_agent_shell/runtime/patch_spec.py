"""PatchSpec — structured reflector output with Pydantic validation.

This module provides:
- ``PatchSpec``: A Pydantic model that validates reflector LLM output
  with strict JSON schema, action whitelist, and arg-key filtering.
- ``StructuredRepairContext``: A Pydantic model for the full repair
  context payload sent *to* the reflector LLM.
- Builder helpers for assembling the context from DAG execution state.
- System / user prompt templates for the structured reflector.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Allowed repair actions
# ---------------------------------------------------------------------------
ALLOWED_ACTIONS: Tuple[str, ...] = ("retry", "skip", "halt", "insert_then_retry")


# ---------------------------------------------------------------------------
# PatchSpec — output expected *from* the reflector
# ---------------------------------------------------------------------------

class InsertNodeSpec(BaseModel):
    """Lightweight spec for a node the reflector wants to inject before retry."""
    tool_name: str = Field(..., description="Tool to insert (e.g. resample_image, register_to_reference)")
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class PatchSpec(BaseModel):
    """Strict output contract for the Tier-2 reflector.

    The LLM must return JSON matching this schema.  Invalid payloads are
    caught by Pydantic validation and fall back to ``halt``.
    """
    action: Literal["retry", "skip", "halt", "insert_then_retry"] = Field(
        ..., description="Repair action to take",
    )
    corrected_args: Optional[Dict[str, Any]] = Field(
        default=None, description="Corrected arguments for retry (when action=retry)",
    )
    insert_node_spec: Optional[InsertNodeSpec] = Field(
        default=None, description="Node to insert (only when action=insert_then_retry)",
    )
    reason: str = Field(default="", description="Diagnosis / repair rationale")
    natural_language_response: str = Field(default="", description="Human-readable status")
    justification: str = Field(default="", description="Optional extended justification")

    @model_validator(mode="after")
    def _check_action_consistency(self) -> "PatchSpec":
        if self.action == "insert_then_retry" and self.insert_node_spec is None:
            raise ValueError("insert_then_retry requires insert_node_spec")
        return self

    # -- construction helpers ------------------------------------------------

    @classmethod
    def halt(cls, reason: str = "", nl: str = "") -> "PatchSpec":
        return cls(action="halt", reason=reason, natural_language_response=nl)

    @classmethod
    def skip(cls, reason: str = "", nl: str = "") -> "PatchSpec":
        return cls(action="skip", reason=reason, natural_language_response=nl)

    @classmethod
    def retry(cls, corrected_args: Dict[str, Any], reason: str = "", nl: str = "") -> "PatchSpec":
        return cls(action="retry", corrected_args=corrected_args, reason=reason, natural_language_response=nl)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump(exclude_none=True)
        return d

    def to_legacy_verdict(self) -> Dict[str, Any]:
        """Return a dict compatible with the old reflector contract."""
        return {
            "action": self.action if self.action != "insert_then_retry" else "retry",
            "reason": self.reason,
            "natural_language_response": self.natural_language_response,
            "retry_arguments": dict(self.corrected_args or {}),
        }

    # -- parsing / validation -----------------------------------------------

    @classmethod
    def parse_and_validate(
        cls,
        raw: str,
        *,
        allowed_arg_keys: Optional[Set[str]] = None,
        allowed_actions: Tuple[str, ...] = ALLOWED_ACTIONS,
    ) -> "PatchSpec":
        """Parse a JSON string from the LLM and return a validated PatchSpec.

        Raises ``ValueError`` if the JSON is malformed or violates the schema.
        """
        obj = _extract_json_object(raw)
        if obj is None:
            raise ValueError(f"Cannot extract JSON object from reflector output: {raw[:300]}")

        action = str(obj.get("action") or "").strip().lower()
        if action not in allowed_actions:
            raise ValueError(
                f"Invalid action '{action}'; must be one of {allowed_actions}"
            )

        # Accept both naming conventions for corrected args
        corrected_args = obj.get("corrected_args") or obj.get("retry_arguments") or {}
        if not isinstance(corrected_args, dict):
            corrected_args = {}

        # Whitelist filter: only keep keys that appear in the tool's input schema
        if allowed_arg_keys is not None and corrected_args:
            corrected_args = {
                k: v for k, v in corrected_args.items() if k in allowed_arg_keys
            }

        insert_raw = obj.get("insert_node_spec")
        insert_spec: Optional[InsertNodeSpec] = None
        if isinstance(insert_raw, dict) and insert_raw.get("tool_name"):
            try:
                insert_spec = InsertNodeSpec(**insert_raw)
            except Exception:
                insert_spec = None

        return cls(
            action=action,
            corrected_args=corrected_args if corrected_args else None,
            insert_node_spec=insert_spec,
            reason=str(obj.get("reason") or "").strip(),
            natural_language_response=str(obj.get("natural_language_response") or "").strip(),
            justification=str(obj.get("justification") or "").strip(),
        )


# ---------------------------------------------------------------------------
# StructuredRepairContext — input sent *to* the reflector
# ---------------------------------------------------------------------------

class ToolSchemaInfo(BaseModel):
    """Condensed tool schema for the reflector prompt."""
    name: str
    description: str = ""
    required_keys: List[str] = Field(default_factory=list)
    optional_keys: List[str] = Field(default_factory=list)
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class StructuredRepairContext(BaseModel):
    """Everything the reflector needs to reason about a failure."""

    failing_tool: str
    failing_args: Dict[str, Any] = Field(default_factory=dict)
    error_type: str = "RuntimeError"
    error_message: str = ""
    failure_classification: str = "unknown_runtime"

    # tool schema for the failing tool
    tool_schema: Optional[ToolSchemaInfo] = None
    # all currently-bound tokens the reflector may reference
    available_tokens: Dict[str, str] = Field(default_factory=dict)
    # available modalities (sequence names)
    available_modalities: List[str] = Field(default_factory=list)
    # last successful artifacts keyed by node_id
    last_artifacts: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # explicit list of actions the reflector is allowed to propose
    allowed_actions: List[str] = Field(default_factory=lambda: ["retry", "skip", "halt"])
    # tools the reflector is allowed to insert
    insertable_tools: List[str] = Field(default_factory=list)
    # deterministic Tier-1 suggestion (if any)
    deterministic_retry_suggestion: Optional[Dict[str, Any]] = None
    # additional context
    extra: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def to_prompt_json(self) -> str:
        return self.model_dump_json(indent=2, exclude_none=True)


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

_INSERTABLE_TOOLS = [
    "resample_image",
    "register_to_reference",
    "identify_sequences",
    "denoise_image_bm3d",
]


def build_tool_schema_info(*, registry: Any, tool_name: str) -> Optional[ToolSchemaInfo]:
    """Extract a condensed ToolSchemaInfo from the tool registry."""
    try:
        tool = registry.get(tool_name)
        spec = tool.spec
        inp = dict(spec.input_schema or {})
        required = sorted(inp.get("required", []))
        properties = inp.get("properties", {})
        all_keys = sorted(properties.keys()) if isinstance(properties, dict) else []
        optional = sorted(set(all_keys) - set(required))
        return ToolSchemaInfo(
            name=spec.name or tool_name,
            description=str(spec.description or "")[:300],
            required_keys=required,
            optional_keys=optional,
            input_schema=inp,
        )
    except Exception:
        return None


def collect_last_successful_artifacts(
    step_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Collect data keys from successfully completed upstream nodes.

    Returns: ``{node_id: {tool_name, data_keys: [...]}}``
    """
    out: Dict[str, Dict[str, Any]] = {}
    for rec in (step_results or []):
        if not isinstance(rec, dict):
            continue
        if str(rec.get("status") or "").upper() != "DONE":
            continue
        nid = str(rec.get("node_id") or "")
        tn = str(rec.get("tool_name") or "")
        data = rec.get("data")
        if not nid:
            continue
        data_keys = sorted(data.keys()) if isinstance(data, dict) else []
        out[nid] = {"tool_name": tn, "data_keys": data_keys}
    return out


def build_structured_repair_context(
    *,
    tool_name: str,
    last_arguments: Dict[str, Any],
    error: Dict[str, Any],
    failure_classification: str,
    binding_context: Dict[str, Any],
    deterministic_retry: Optional[Dict[str, Any]],
    registry: Any,
    step_results: List[Dict[str, Any]],
    enable_insert: bool = False,
) -> StructuredRepairContext:
    """Build the full StructuredRepairContext for the reflector."""
    tool_schema = build_tool_schema_info(registry=registry, tool_name=tool_name)

    allowed: List[str] = ["retry", "skip", "halt"]
    insertable: List[str] = []
    if enable_insert:
        allowed.append("insert_then_retry")
        insertable = list(_INSERTABLE_TOOLS)

    # Merge token bindings from binder + session
    tokens: Dict[str, str] = {}
    tokens.update(binding_context.get("token_bindings") or {})
    tokens.update(binding_context.get("session_token_bindings") or {})

    return StructuredRepairContext(
        failing_tool=tool_name,
        failing_args=dict(last_arguments),
        error_type=str(error.get("type") or "RuntimeError"),
        error_message=str(error.get("message") or ""),
        failure_classification=failure_classification,
        tool_schema=tool_schema,
        available_tokens=tokens,
        available_modalities=sorted(binding_context.get("available_modalities") or []),
        last_artifacts=collect_last_successful_artifacts(step_results),
        allowed_actions=allowed,
        insertable_tools=insertable,
        deterministic_retry_suggestion=deterministic_retry if deterministic_retry else None,
    )


# ---------------------------------------------------------------------------
# System / User prompts for structured reflector
# ---------------------------------------------------------------------------

STRUCTURED_REFLECTOR_SYSTEM_PROMPT = """\
You are an MRI DAG execution debugger with bounded authority.

CAPABILITIES:
- Fix runtime argument/token/path mismatches and retry
- Swap semantically incorrect arguments (e.g. fixed<->moving)
- Restore omitted required arguments from available tokens/bindings
- Adjust safe hyperparameters (spacing, interpolation)

BOUNDARIES:
- CANNOT bypass ScopeViolation policy (-> halt)
- CANNOT bypass SchemaValidationError domain rules (-> halt)
- CANNOT fabricate data that does not exist (missing modality -> halt)
- CANNOT edit backend source code or config files

OUTPUT FORMAT:
Return strict JSON matching this schema:
{
  "action": "retry" | "skip" | "halt",
  "corrected_args": { ... },
  "reason": "...",
  "natural_language_response": "..."
}

RULES:
1. If deterministic_retry_suggestion is provided AND looks correct, use action=retry with those args.
2. For argument_omission: check tool_schema.required_keys and restore from available_tokens.
3. For semantic_swap: check tool_schema and swap back the incorrect pair.
4. For missing_modality / scope_violation: MUST halt. These are non-recoverable.
5. For space_mismatch: fix spacing/interpolation values to reasonable defaults.
6. Never invent file paths that don't appear in available_tokens or last_artifacts.
"""


def build_reflector_user_prompt(ctx: StructuredRepairContext) -> str:
    """Build the user-role prompt from a StructuredRepairContext."""
    return (
        "Analyze the following failure context and return a PatchSpec JSON.\n\n"
        f"{ctx.to_prompt_json()}"
    )


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from LLM output."""
    text = str(raw or "").strip()
    if not text:
        return None

    # 1. Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2. Try ```json ... ``` fenced block
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 3. Find first { ... } substring
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
    return None

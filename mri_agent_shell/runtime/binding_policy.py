"""Ablation-aware binding policy for the Cerebellum DAG executor.

Each BCER benchmark arm maps to a ``BindingPolicy`` that controls which
argument-completion mechanisms are active during DAG execution.  This keeps
ablation semantics explicit and auditable rather than scattered across ad-hoc
``if ablation_mode ==`` checks.

Mechanism dimensions
--------------------
* **symbolic_bind** – ``resolve_refs()`` resolving ``@node.*``, ``@seq.*``,
  ``@case.*``, ``@runtime.*``, ``@auto`` tokens.
* **implicit_seq_autocomplete** – ``_normalize_step_args`` looking up sequence
  names (``"T2w"`` etc.) via ``seq_paths`` / ``resolve_sequence_or_case_path``.
* **implicit_node_autowire** – ``_normalize_step_args`` auto-resolving
  ``@auto`` or empty args from upstream ``seq_paths`` / ingest artifacts.
* **tool_arg_type_repair** – ``repair_tool_args()`` Pydantic model repair that
  fills defaults and fixes types (but may also do path lookups).
* **repair semantic suppressors** – explicit flags passed into
  ``repair_tool_args()`` to disable semantic binding inside repair while
  preserving type/schema coercion.
* **error_reflection** – LLM-based reflector on required-step failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class BindingPolicy:
    """Immutable policy controlling which arg-completion layers are active."""

    # ---- symbolic token resolution ----
    symbolic_bind_enabled: bool = True
    """Allow resolve_refs() to resolve @node.*, @seq.*, @case.*, @runtime.*, @auto."""

    # ---- implicit deterministic completion ----
    implicit_seq_autocomplete_enabled: bool = True
    """Allow _normalize_step_args to resolve plain sequence names ("T2w" etc.)
    via seq_paths populated by identify_sequences."""

    implicit_node_autowire_enabled: bool = True
    """Allow _normalize_step_args to auto-resolve @auto / empty args from
    upstream seq_paths or ingest artifacts."""

    # ---- Pydantic schema repair ----
    tool_arg_type_repair_enabled: bool = True
    """Allow repair_tool_args() Pydantic model repair (type coercion,
    defaults, schema fix-up).  When False, only GenericToolArgs no-op
    normalisation is used."""

    suppress_sequence_resolve: bool = False
    """When True, keep repair_tool_args() type/schema repair enabled but
    suppress semantic sequence-name -> path resolution (e.g. "T2w"/"ADC")."""

    suppress_node_output_autowire: bool = False
    """When True, keep repair_tool_args() type/schema repair enabled but
    suppress semantic autowiring from prior node outputs / artifacts scans."""

    # ---- reflection ----
    error_reflection_enabled: bool = True
    """Allow the LLM reflector to propose argument patches on failure.
    When False, the reflector is either disabled or replaced with a
    deterministic-only stub (depending on arm configuration)."""

    # ---- scope guard / invariants / logging (always on – listed for matrix) ----
    scope_guard_enabled: bool = True
    provenance_logging_enabled: bool = True

    @property
    def name(self) -> str:  # noqa: D401
        """Short human-readable label derived from flags."""
        if (
            self.symbolic_bind_enabled
            and self.implicit_seq_autocomplete_enabled
            and self.implicit_node_autowire_enabled
            and self.tool_arg_type_repair_enabled
            and self.error_reflection_enabled
        ):
            return "full"
        parts = []
        if not self.symbolic_bind_enabled:
            parts.append("no_sym")
        if not self.implicit_seq_autocomplete_enabled:
            parts.append("no_seqac")
        if not self.implicit_node_autowire_enabled:
            parts.append("no_autowire")
        if not self.tool_arg_type_repair_enabled:
            parts.append("no_repair")
        if not self.error_reflection_enabled:
            parts.append("no_reflect")
        return "_".join(parts) or "custom"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbolic_bind_enabled": self.symbolic_bind_enabled,
            "implicit_seq_autocomplete_enabled": self.implicit_seq_autocomplete_enabled,
            "implicit_node_autowire_enabled": self.implicit_node_autowire_enabled,
            "tool_arg_type_repair_enabled": self.tool_arg_type_repair_enabled,
            "suppress_sequence_resolve": self.suppress_sequence_resolve,
            "suppress_node_output_autowire": self.suppress_node_output_autowire,
            "error_reflection_enabled": self.error_reflection_enabled,
            "scope_guard_enabled": self.scope_guard_enabled,
            "provenance_logging_enabled": self.provenance_logging_enabled,
        }

    # ------------------------------------------------------------------
    # Pre-defined policies for each benchmark arm
    # ------------------------------------------------------------------

    @staticmethod
    def for_arm(arm: str) -> "BindingPolicy":
        """Return the canonical policy for a benchmark arm name."""
        return _ARM_POLICY_MAP.get(arm, POLICY_FULL)


# ── Canonical policies ────────────────────────────────────────────────

POLICY_FULL = BindingPolicy()

POLICY_NO_TOKEN = BindingPolicy(
    symbolic_bind_enabled=False,
    implicit_seq_autocomplete_enabled=False,
    implicit_node_autowire_enabled=False,
    tool_arg_type_repair_enabled=True,  # type/schema repair only
    suppress_sequence_resolve=True,
    suppress_node_output_autowire=True,
    error_reflection_enabled=True,
)

POLICY_NO_REFLECTOR = BindingPolicy(
    error_reflection_enabled=False,
)

POLICY_DETERMINISTIC_ONLY = BindingPolicy(
    error_reflection_enabled=False,  # LLM reflection disabled; deterministic tier-1 only
)

# ReAct arms share full policy (irrelevant — they don't go through Cerebellum)
POLICY_REACT = BindingPolicy()

_ARM_POLICY_MAP: Dict[str, BindingPolicy] = {
    "bcr_full": POLICY_FULL,
    "bcr_sketch": POLICY_FULL,
    "bcr_no_token": POLICY_NO_TOKEN,
    "bcr_no_reflector": POLICY_NO_REFLECTOR,
    "bcr_deterministic_only": POLICY_DETERMINISTIC_ONLY,
    "static_pipeline": POLICY_FULL,
    "react": POLICY_REACT,
    "react_token": POLICY_REACT,
    "pure_react": POLICY_REACT,
}


# ── DAG token degradation ────────────────────────────────────────────

def degrade_dag_tokens(dag: Any, *, policy: BindingPolicy) -> Any:
    """Apply token → literal degradation to a DAG based on *policy*.

    When ``symbolic_bind_enabled`` is ``False`` the executor's ``resolve_refs``
    will be a no-op.  To give the noToken arm a *fair* chance we convert the
    template tokens at compile-time:

    * ``@case.input`` → kept as literal string ``"@case.input"`` (the
      ``_normalize_step_args`` for ``identify_sequences`` will replace it
      with ``case_root`` via a **type-normalisation**, which is allowed).
    * ``@auto`` → removed (set to empty string ``""``) so that the
      type-normalisation in ``_normalize_step_args`` can kick in.
    * ``@node.<id>.<key>`` → removed (``""``) – the noToken arm must rely on
      Pydantic type repair or fail gracefully.
    * ``@runtime.*`` → kept (scope / provenance paths, always resolved by
      seed_runtime which sets them as plain refs — this is infrastructure,
      not "symbolic binding").
    * ``@seq.<name>`` → stripped to plain ``<name>`` (e.g. ``"T2w"``).
    * Plain strings like ``"T2w"`` → kept unchanged.

    Returns the DAG mutated in-place for convenience.
    """
    if policy.symbolic_bind_enabled:
        return dag  # nothing to degrade

    for node in getattr(dag, "nodes", []):
        if not hasattr(node, "arguments") or not isinstance(node.arguments, dict):
            continue
        for key in list(node.arguments.keys()):
            val = node.arguments.get(key)
            if not isinstance(val, str):
                continue
            if val == "@auto":
                # Remove so _normalize_step_args type-normalisation can infer
                node.arguments[key] = ""
            elif val.startswith("@node."):
                # Cross-node symbolic ref → cleared; Pydantic type repair may
                # still recover via ArtifactLocator / state_path lookup.
                node.arguments[key] = ""
            elif val.startswith("@seq."):
                # Strip @seq. prefix → treat as plain sequence name
                node.arguments[key] = val[len("@seq."):]
            elif val == "@case.input":
                # Keep as-is — identify_sequences _normalize handles this as
                # a type normalisation ("replace non-absolute with case_root")
                pass
            elif val.startswith("@runtime."):
                # Infrastructure references (case_state_path, artifacts_dir)
                # are always resolved by seed_runtime (not symbolic binding).
                # Keep them so that package_vlm_evidence / generate_report
                # can still find the state file.
                pass
            elif val.startswith("@"):
                # Any other @-token: clear
                node.arguments[key] = ""
    return dag

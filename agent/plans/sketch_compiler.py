from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.plan_dag import PlanNode
from .sketch_schema import ConstrainedPlanSketch


@dataclass
class SketchCompileResult:
    ok: bool
    nodes: List[PlanNode] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


_NODE_REF_RE = re.compile(r"^@node\.([^.]+)\.(.+)$")
_DROP_OVERRIDE = object()

# Conservative alias/canonicalization for common LLM output-field guesses on the
# benchmark task set. This is compiler-side interface normalization, not planner hints.
_NODE_OUTPUT_FIELD_ALIASES: Dict[str, Dict[str, str]] = {
    "reconstruct_grappa": {
        "output_nifti": "reconstructed_nifti",
    },
    "denoise_image_bm3d": {
        "output_nifti": "denoised_nifti",
    },
    "brats_mri_segmentation": {
        "tumor_mask_path": "wt_mask_path",
        "segmentation_path": "seg_path",
    },
    "segment_cardiac_cine": {
        "segmentation_path": "seg_path",
        "segmentation_dir": "pred_dir",
        "mask_dir": "pred_dir",
    },
    "segment_prostate": {
        "whole_gland_mask_path": "prostate_mask_path",
        "zonal_mask_path": "zone_mask_path",
    },
}

_KNOWN_NODE_OUTPUT_FIELDS: Dict[str, Set[str]] = {
    "identify_sequences": {
        "mapping",
        "confidence",
        "series_inventory_path",
        "dicom_meta_path",
        "dicom_headers_index_path",
        "series",
        "nifti_by_series",
        "note",
    },
    "reconstruct_grappa": {
        "reconstructed_nifti",
        "log_path",
        "output_dir",
        "input_h5",
        "kspace_shape",
        "coil_count",
    },
    "denoise_image_bm3d": {
        "denoised_nifti",
        "input_nifti",
        "output_dir",
        "log_path",
    },
    "brats_mri_segmentation": {
        "seg_path",
        "tumor_subregions_path",
        "wt_mask_path",
        "tc_mask_path",
        "et_mask_path",
        "output_dir",
        "log_path",
    },
    "segment_cardiac_cine": {
        "seg_path",
        "pred_dir",
        "rv_mask_path",
        "myo_mask_path",
        "lv_mask_path",
        "case_results",
        "log_path",
        "input_dir",
        "results_folder",
        "task_name",
        "trainer_class_name",
        "model",
        "folds",
        "cmr_reverse_root",
        "note",
    },
    "segment_prostate": {
        "prostate_mask_path",
        "zone_mask_path",
        "t2w_input_path",
        "note",
    },
}

_IDENTIFY_FIELD_TO_SEQ_TOKEN: Dict[str, str] = {
    "t1_path": "@seq.T1",
    "t1c_path": "@seq.T1c",
    "t2_path": "@seq.T2",
    "flair_path": "@seq.FLAIR",
    "adc_path": "@seq.ADC",
    "dwi_path": "@seq.DWI",
    "highb_path": "@seq.DWI",
    "cine_path": "@seq.CINE",
}


def _parse_numeric_list_literal(raw: str) -> Optional[List[float]]:
    s = str(raw or "").strip()
    if not s:
        return None
    candidates = [s]
    if s.startswith("(") and s.endswith(")"):
        candidates.append("[" + s[1:-1] + "]")
    # Common model variant for spacing literals: "1.0,1.0,1.0" (no brackets).
    if "," in s and not (s.startswith("[") and s.endswith("]")):
        parts = [p.strip() for p in s.split(",")]
        if len(parts) >= 1 and all(parts):
            try:
                return [float(p) for p in parts]
            except Exception:
                pass
    # Also accept simple whitespace-separated tuples, e.g. "1 1 1".
    if " " in s and not any(ch in s for ch in "[](),"):
        parts = [p for p in s.split() if p]
        if len(parts) >= 1:
            try:
                return [float(p) for p in parts]
            except Exception:
                pass
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except Exception:
            parsed = None
        if not isinstance(parsed, list) or not parsed:
            continue
        out: List[float] = []
        ok = True
        for x in parsed:
            try:
                out.append(float(x))
            except Exception:
                ok = False
                break
        if ok:
            return out
    return None


def _dedup_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        s = str(item or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _detect_cycle_edges(edges: Dict[str, List[str]]) -> Optional[List[str]]:
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: List[str] = []

    def dfs(nid: str) -> Optional[List[str]]:
        if nid in visited:
            return None
        if nid in visiting:
            try:
                idx = stack.index(nid)
                return stack[idx:] + [nid]
            except ValueError:
                return [nid, nid]
        visiting.add(nid)
        stack.append(nid)
        for dep in edges.get(nid, []):
            cyc = dfs(dep)
            if cyc:
                return cyc
        stack.pop()
        visiting.remove(nid)
        visited.add(nid)
        return None

    for node_id in list(edges.keys()):
        cyc = dfs(node_id)
        if cyc:
            return cyc
    return None


def _toposort_nodes(nodes: List[PlanNode]) -> List[PlanNode]:
    by_id: Dict[str, PlanNode] = {str(n.node_id): n for n in nodes}
    indeg: Dict[str, int] = {nid: 0 for nid in by_id}
    forward: Dict[str, List[str]] = {nid: [] for nid in by_id}
    for nid, node in by_id.items():
        for dep in (node.depends_on or []):
            dep_s = str(dep or "").strip()
            if dep_s not in by_id:
                continue
            indeg[nid] += 1
            forward.setdefault(dep_s, []).append(nid)

    # Preserve original order as stable tie-breaker.
    original_order = [str(n.node_id) for n in nodes]
    queue: List[str] = [nid for nid in original_order if indeg.get(nid, 0) == 0]
    out_ids: List[str] = []
    seen_q: set[str] = set(queue)
    while queue:
        nid = queue.pop(0)
        out_ids.append(nid)
        for nxt in forward.get(nid, []):
            indeg[nxt] = max(0, int(indeg.get(nxt, 0)) - 1)
            if indeg[nxt] == 0 and nxt not in seen_q:
                seen_q.add(nxt)
                queue.append(nxt)
    if len(out_ids) != len(by_id):
        return nodes
    return [by_id[nid] for nid in out_ids if nid in by_id]


def _build_blueprint_maps(blueprints: List[PlanNode]) -> Tuple[Dict[str, PlanNode], Dict[str, List[PlanNode]], List[str]]:
    by_id: Dict[str, PlanNode] = {}
    by_tool: Dict[str, List[PlanNode]] = {}
    order: List[str] = []
    for bp in blueprints:
        nid = str(bp.node_id or "").strip()
        if not nid or nid in by_id:
            continue
        by_id[nid] = bp
        by_tool.setdefault(str(bp.tool_name or "").strip(), []).append(bp)
        order.append(nid)
    return by_id, by_tool, order


def _collect_dependency_closure(start_ids: Set[str], by_id: Dict[str, PlanNode]) -> Set[str]:
    out: Set[str] = set()
    stack: List[str] = list(start_ids)
    while stack:
        nid = str(stack.pop() or "").strip()
        if not nid or nid in out:
            continue
        out.add(nid)
        node = by_id.get(nid)
        if node is None:
            continue
        for dep in (node.depends_on or []):
            dep_s = str(dep or "").strip()
            if dep_s and dep_s not in out:
                stack.append(dep_s)
    return out


def _rewrite_symbolic_value(
    value: Any,
    *,
    sketch_step_to_compiled_id: Dict[str, str],
    compiled_tool_by_node_id: Dict[str, str],
    case_ref_is_file: bool,
    tool_name: str,
    arg_key: str,
    rewrite_stats: Dict[str, int],
) -> Any:
    """Rewrite sketch-only symbolic aliases into runtime-executable refs."""

    if isinstance(value, dict):
        return {
            str(k): _rewrite_symbolic_value(
                v,
                sketch_step_to_compiled_id=sketch_step_to_compiled_id,
                compiled_tool_by_node_id=compiled_tool_by_node_id,
                case_ref_is_file=case_ref_is_file,
                tool_name=tool_name,
                arg_key=str(k),
                rewrite_stats=rewrite_stats,
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_symbolic_value(
                v,
                sketch_step_to_compiled_id=sketch_step_to_compiled_id,
                compiled_tool_by_node_id=compiled_tool_by_node_id,
                case_ref_is_file=case_ref_is_file,
                tool_name=tool_name,
                arg_key=arg_key,
                rewrite_stats=rewrite_stats,
            )
            for v in value
        ]
    if not isinstance(value, str):
        return value

    s = str(value)
    if s in {"@case_ref", "case_ref"}:
        rewrite_stats["case_ref_token_rewrites"] = int(rewrite_stats.get("case_ref_token_rewrites", 0)) + 1
        return "@case.file" if case_ref_is_file else "@case.input"

    if tool_name == "reconstruct_grappa" and str(arg_key) == "h5_path" and s in {"@case.input", "case.input"} and case_ref_is_file:
        rewrite_stats["h5_case_input_to_case_file"] = int(rewrite_stats.get("h5_case_input_to_case_file", 0)) + 1
        return "@case.file"

    m = _NODE_REF_RE.match(s)
    if m:
        ref_id = str(m.group(1) or "").strip()
        tail = str(m.group(2) or "")
        mapped = sketch_step_to_compiled_id.get(ref_id)
        ref_node_id = mapped or ref_id
        source_tool = str(compiled_tool_by_node_id.get(ref_node_id) or "").strip()
        if source_tool == "identify_sequences":
            seq_tok = _IDENTIFY_FIELD_TO_SEQ_TOKEN.get(tail)
            if seq_tok:
                rewrite_stats["identify_field_to_seq_token"] = int(rewrite_stats.get("identify_field_to_seq_token", 0)) + 1
                return seq_tok
        alias_map = _NODE_OUTPUT_FIELD_ALIASES.get(source_tool) or {}
        aliased_tail = alias_map.get(tail)
        if aliased_tail and aliased_tail != tail:
            tail = aliased_tail
            rewrite_stats["node_output_field_alias_rewrites"] = int(
                rewrite_stats.get("node_output_field_alias_rewrites", 0)
            ) + 1
        if mapped and mapped != ref_id:
            rewrite_stats["node_ref_rewrites"] = int(rewrite_stats.get("node_ref_rewrites", 0)) + 1
            return f"@node.{mapped}.{tail}"
        if tail != str(m.group(2) or ""):
            return f"@node.{ref_node_id}.{tail}"
    return s


def _rewrite_step_input_value(
    raw_value: Any,
    *,
    target_tool_name: str,
    target_arg_key: str,
    blueprint_default_present: bool,
    sketch_step_to_compiled_id: Dict[str, str],
    compiled_tool_by_node_id: Dict[str, str],
    case_ref_is_file: bool,
    rewrite_stats: Dict[str, int],
    diagnostics: Dict[str, Any],
    current_compiled_node_id: Optional[str] = None,
) -> Any:
    """Rewrite/canonicalize a single top-level step input value.

    Returns `_DROP_OVERRIDE` when the explicit sketch override should be ignored
    and the blueprint default (or absence) should be preserved.
    """

    rewritten = _rewrite_symbolic_value(
        raw_value,
        sketch_step_to_compiled_id=sketch_step_to_compiled_id,
        compiled_tool_by_node_id=compiled_tool_by_node_id,
        case_ref_is_file=bool(case_ref_is_file),
        tool_name=str(target_tool_name or ""),
        arg_key=str(target_arg_key or ""),
        rewrite_stats=rewrite_stats,
    )
    if str(target_tool_name or "") == "extract_roi_features" and str(target_arg_key or "") == "radiomics":
        if isinstance(rewritten, bool):
            diagnostics.setdefault("warnings", []).append(
                "Sketch input canonicalization: normalized extract_roi_features.radiomics bool "
                f"to object form for schema compatibility ({rewritten!r})."
            )
            rewrite_stats["extract_radiomics_bool_to_object"] = int(
                rewrite_stats.get("extract_radiomics_bool_to_object", 0)
            ) + 1
            return {"enabled": bool(rewritten)}
    if str(target_tool_name or "") == "resample_image":
        if str(target_arg_key or "") == "target_spacing" and isinstance(rewritten, str):
            parsed_spacing = _parse_numeric_list_literal(rewritten)
            if parsed_spacing:
                rewrite_stats["resample_target_spacing_string_to_array"] = int(
                    rewrite_stats.get("resample_target_spacing_string_to_array", 0)
                ) + 1
                return parsed_spacing
        if (
            str(target_arg_key or "") == "output_nifti"
            and isinstance(rewritten, str)
            and "@runtime.case_state_path/" in rewritten
        ):
            diagnostics.setdefault("warnings", []).append(
                "Sketch input canonicalization: dropped resample_image.output_nifti override "
                "containing composite @runtime.case_state_path/... path; preserving tool/default "
                "output path contract."
            )
            rewrite_stats["resample_output_nifti_runtime_case_state_drops"] = int(
                rewrite_stats.get("resample_output_nifti_runtime_case_state_drops", 0)
            ) + 1
            return _DROP_OVERRIDE
        if (
            str(target_arg_key or "") == "reference_nifti"
            and (not blueprint_default_present)
            and isinstance(rewritten, str)
            and str(rewritten).strip()
        ):
            diagnostics.setdefault("warnings", []).append(
                "Sketch input canonicalization: dropped resample_image.reference_nifti override "
                "to preserve template target_spacing-based super-resolution contract."
            )
            rewrite_stats["resample_reference_override_drops"] = int(
                rewrite_stats.get("resample_reference_override_drops", 0)
            ) + 1
            return _DROP_OVERRIDE
    if (
        str(target_tool_name or "") == "extract_roi_features"
        and str(target_arg_key or "") in {"images", "roi_masks"}
        and isinstance(rewritten, (list, dict))
    ):
        def _sanitize_nested(v: Any) -> Any:
            if isinstance(v, list):
                out_list: List[Any] = []
                changed = False
                for item in v:
                    sv = _sanitize_nested(item)
                    if sv is _DROP_OVERRIDE:
                        changed = True
                        continue
                    if sv is not item:
                        changed = True
                    out_list.append(sv)
                return out_list if changed else v
            if isinstance(v, dict):
                out_dict: Dict[str, Any] = {}
                changed = False
                for kk, vv in v.items():
                    sv = _sanitize_nested(vv)
                    if sv is _DROP_OVERRIDE:
                        changed = True
                        continue
                    if sv is not vv:
                        changed = True
                    out_dict[str(kk)] = sv
                return out_dict if changed else v
            if not isinstance(v, str):
                return v
            m2 = _NODE_REF_RE.match(v)
            if not m2:
                return v
            ref_node_id2 = str(m2.group(1) or "").strip()
            ref_field2 = str(m2.group(2) or "").strip()
            source_tool2 = str(compiled_tool_by_node_id.get(ref_node_id2) or "").strip()
            if not source_tool2:
                return v
            known2 = _KNOWN_NODE_OUTPUT_FIELDS.get(source_tool2)
            if known2 is not None and ref_field2 not in known2:
                diagnostics.setdefault("warnings", []).append(
                    "Sketch input canonicalization: unknown nested output field "
                    f"{source_tool2}.{ref_field2} referenced by {target_tool_name}.{target_arg_key}; "
                    "dropping invalid nested reference."
                )
                rewrite_stats["unknown_nested_node_output_field_drops"] = int(
                    rewrite_stats.get("unknown_nested_node_output_field_drops", 0)
                ) + 1
                return _DROP_OVERRIDE
            return v

        sanitized = _sanitize_nested(rewritten)
        if sanitized is _DROP_OVERRIDE:
            if blueprint_default_present:
                rewrite_stats["preserved_blueprint_defaults"] = int(rewrite_stats.get("preserved_blueprint_defaults", 0)) + 1
                return _DROP_OVERRIDE
            return [] if str(target_arg_key or "") in {"images", "roi_masks"} else sanitized
        if sanitized != rewritten:
            # If every nested item was dropped and a blueprint default exists, preserve default.
            if blueprint_default_present and sanitized in ([], {}):
                rewrite_stats["preserved_blueprint_defaults"] = int(rewrite_stats.get("preserved_blueprint_defaults", 0)) + 1
                return _DROP_OVERRIDE
            rewritten = sanitized

    if not isinstance(rewritten, str):
        return rewritten
    cur_cid = str(current_compiled_node_id or "").strip()
    if cur_cid and str(target_arg_key or "").startswith("output_"):
        self_prefix = f"@node.{cur_cid}."
        if rewritten.startswith(self_prefix):
            diagnostics.setdefault("warnings", []).append(
                f"Sketch input canonicalization: dropped self-referential {target_tool_name}.{target_arg_key} "
                f"override ({rewritten}); preserving blueprint/default output path behavior."
            )
            rewrite_stats["self_output_token_override_drops"] = int(
                rewrite_stats.get("self_output_token_override_drops", 0)
            ) + 1
            return _DROP_OVERRIDE
    m = _NODE_REF_RE.match(rewritten)
    if not m:
        return rewritten

    ref_node_id = str(m.group(1) or "").strip()
    ref_field = str(m.group(2) or "").strip()
    source_tool = str(compiled_tool_by_node_id.get(ref_node_id) or "").strip()
    if not source_tool:
        return rewritten

    known_fields = _KNOWN_NODE_OUTPUT_FIELDS.get(source_tool)
    if known_fields is not None and ref_field not in known_fields:
        warnings = diagnostics.setdefault("warnings", [])
        warnings.append(
            "Sketch input canonicalization: unknown output field "
            f"{source_tool}.{ref_field} referenced by {target_tool_name}.{target_arg_key}; "
            "preserving blueprint default/auto-repair instead."
        )
        rewrite_stats["unknown_node_output_field_drops"] = int(rewrite_stats.get("unknown_node_output_field_drops", 0)) + 1
        if str(target_tool_name or "") == "classify_cardiac_cine_disease" and str(target_arg_key or "") in {
            "ed_seg_path",
            "es_seg_path",
        }:
            # Let tool arg repair infer ED/ES paths from seg_path / prior segment case_results.
            return _DROP_OVERRIDE
        if blueprint_default_present:
            rewrite_stats["preserved_blueprint_defaults"] = int(rewrite_stats.get("preserved_blueprint_defaults", 0)) + 1
            return _DROP_OVERRIDE
        if str(target_tool_name or "") == "extract_roi_features" and str(target_arg_key or "") in {
            "images",
            "roi_masks",
            "roi_mask_path",
        }:
            return _DROP_OVERRIDE
    return rewritten


def _collect_unresolved_sketch_node_refs(
    value: Any,
    *,
    sketch_step_ids: Set[str],
    out: List[str],
) -> None:
    if isinstance(value, dict):
        for v in value.values():
            _collect_unresolved_sketch_node_refs(v, sketch_step_ids=sketch_step_ids, out=out)
        return
    if isinstance(value, list):
        for v in value:
            _collect_unresolved_sketch_node_refs(v, sketch_step_ids=sketch_step_ids, out=out)
        return
    if not isinstance(value, str):
        return
    m = _NODE_REF_RE.match(str(value))
    if not m:
        return
    ref_id = str(m.group(1) or "").strip()
    if ref_id in sketch_step_ids:
        out.append(str(value))


def compile_constrained_plan_sketch(
    *,
    sketch: ConstrainedPlanSketch,
    blueprints: List[PlanNode],
    allowed_tools: Optional[List[str]] = None,
    require_blueprint_match: bool = True,
    ensure_required_blueprint_nodes: bool = True,
    case_ref_is_file: bool = False,
) -> SketchCompileResult:
    """
    Compile an LLM-generated constrained sketch into executable PlanNodes.

    Strategy (Phase 1):
    - Sketch is tool-level.
    - Compiler materializes nodes from template-derived blueprints, preserving
      blueprint defaults/stages/labels while overlaying sketch-provided inputs.
    - Missing required blueprint nodes are autofilled with dependency closure.
    """

    diagnostics: Dict[str, Any] = {
        "compile_ok": False,
        "schema_valid": True,
        "tool_legality_valid": False,
        "dependency_valid": False,
        "cycle_valid": False,
        "errors": [],
        "warnings": [],
        "strategy": "tool_level_sketch_overlay_on_template_blueprints",
        "allowed_tools": sorted(set(str(x).strip() for x in (allowed_tools or []) if str(x).strip())),
        "tool_matches": [],
        "autofilled_nodes": [],
        "added_blueprint_dependencies": [],
        "dropped_optional_steps": [],
        "sketch_summary": {
            "task": str(sketch.task or ""),
            "domain": str(sketch.domain or ""),
            "step_count": len(sketch.steps or []),
            "final_targets": [str(x) for x in (sketch.final_targets or [])],
        },
        "rewrite_stats": {},
    }

    def _err(msg: str) -> None:
        diagnostics.setdefault("errors", []).append(str(msg))

    def _warn(msg: str) -> None:
        diagnostics.setdefault("warnings", []).append(str(msg))

    by_id, by_tool, blueprint_order = _build_blueprint_maps(list(blueprints or []))
    if not by_id:
        _err("No template blueprints available for sketch compiler.")
        return SketchCompileResult(ok=False, nodes=[], diagnostics=diagnostics)

    allowed_set = set(diagnostics["allowed_tools"]) if diagnostics["allowed_tools"] else set(by_tool.keys())
    if not allowed_set:
        _err("Allowed tool set is empty.")
        return SketchCompileResult(ok=False, nodes=[], diagnostics=diagnostics)

    # --- Sketch dependency sanity ---
    step_ids = [str(s.step_id) for s in (sketch.steps or [])]
    step_id_set = set(step_ids)
    dep_ref_errors: List[str] = []
    sketch_edges: Dict[str, List[str]] = {}
    for st in (sketch.steps or []):
        deps = [str(x) for x in (st.depends_on or [])]
        sketch_edges[str(st.step_id)] = deps
        for dep in deps:
            if dep not in step_id_set:
                dep_ref_errors.append(f"{st.step_id} depends_on unknown step_id={dep}")
    cyc = _detect_cycle_edges(sketch_edges)
    if dep_ref_errors:
        for msg in dep_ref_errors:
            _err("Sketch dependency error: " + msg)
    if cyc:
        _err("Sketch dependency cycle detected: " + " -> ".join(cyc))
    if dep_ref_errors or cyc:
        return SketchCompileResult(ok=False, nodes=[], diagnostics=diagnostics)
    diagnostics["dependency_valid"] = True
    diagnostics["cycle_valid"] = True

    # --- Tool legality ---
    illegal_tools = [st.tool for st in sketch.steps if str(st.tool) not in allowed_set]
    if illegal_tools:
        for tool in illegal_tools:
            _err(f"Illegal tool in sketch for this task/domain: {tool}")
        return SketchCompileResult(ok=False, nodes=[], diagnostics=diagnostics)
    diagnostics["tool_legality_valid"] = True

    # --- Materialize explicit sketch steps from blueprints ---
    used_blueprint_ids: set[str] = set()
    compiled_by_node_id: Dict[str, PlanNode] = {}
    chosen_blueprint_by_step: Dict[str, PlanNode] = {}
    sketch_step_to_compiled_id: Dict[str, str] = {}
    sketch_step_to_blueprint_id: Dict[str, str] = {}
    compiled_explicit_ids: List[str] = []
    compiled_tool_by_node_id: Dict[str, str] = {str(nid): str(bp.tool_name or "") for nid, bp in by_id.items()}

    for step in sketch.steps:
        tool = str(step.tool)
        candidates = by_tool.get(tool) or []
        chosen: Optional[PlanNode] = None

        # Prefer exact fixed/moving pair match for repeated registration steps.
        if tool == "register_to_reference" and isinstance(step.inputs, dict):
            sfixed = str(step.inputs.get("fixed") or "").strip()
            smoving = str(step.inputs.get("moving") or "").strip()
            if sfixed or smoving:
                for cand in candidates:
                    if str(cand.node_id) in used_blueprint_ids:
                        continue
                    cargs = cand.arguments if isinstance(cand.arguments, dict) else {}
                    if sfixed and str(cargs.get("fixed") or "").strip() != sfixed:
                        continue
                    if smoving and str(cargs.get("moving") or "").strip() != smoving:
                        continue
                    chosen = cand
                    break

        if chosen is None:
            for cand in candidates:
                if str(cand.node_id) in used_blueprint_ids:
                    continue
                chosen = cand
                break
        if chosen is None and candidates:
            if bool(step.optional):
                _warn(
                    f"Sketch optional step {step.step_id} tool={tool!r} exceeds blueprint multiplicity "
                    f"(blueprint_count={len(candidates)}); dropping optional step."
                )
                diagnostics.setdefault("dropped_optional_steps", []).append(
                    {
                        "sketch_step_id": str(step.step_id),
                        "tool": tool,
                        "reason": "exceeds_blueprint_multiplicity",
                        "blueprint_count": int(len(candidates)),
                    }
                )
                continue
            _err(
                f"Sketch tool {tool!r} appears more times than blueprint supports "
                f"(step_id={step.step_id}, blueprint_count={len(candidates)})."
            )
            continue
        if chosen is None:
            if require_blueprint_match:
                _err(f"Sketch step {step.step_id} tool={tool!r} has no blueprint match.")
                continue
            # Fallback generic node (not preferred; disabled by default)
            chosen = PlanNode(
                node_id=str(step.step_id),
                tool_name=tool,
                stage="misc",
                arguments={},
                required=not bool(step.optional),
                depends_on=[],
                label=(step.goal or None),
            )
            _warn(f"Sketch step {step.step_id} compiled without blueprint defaults (generic fallback).")

        bp_id = str(chosen.node_id)
        used_blueprint_ids.add(bp_id)
        chosen_blueprint_by_step[str(step.step_id)] = chosen
        compiled_explicit_ids.append(bp_id)
        sketch_step_to_compiled_id[str(step.step_id)] = bp_id
        sketch_step_to_blueprint_id[str(step.step_id)] = bp_id
        diagnostics["tool_matches"].append(
            {
                "sketch_step_id": str(step.step_id),
                "tool": tool,
                "blueprint_node_id": bp_id,
                "compiled_node_id": bp_id,
                "origin": "sketch",
            }
        )

    if diagnostics["errors"]:
        return SketchCompileResult(ok=False, nodes=[], diagnostics=diagnostics)

    # Materialize nodes only after all sketch_step_id -> compiled_node_id mappings are known,
    # so cross-step symbolic refs in step.inputs can be rewritten safely.
    rewrite_stats: Dict[str, int] = {}
    for step in sketch.steps:
        sid = str(step.step_id)
        cid = sketch_step_to_compiled_id.get(sid)
        chosen = chosen_blueprint_by_step.get(sid)
        if not cid or chosen is None:
            continue
        merged_args = dict(chosen.arguments or {})
        for k, v in dict(step.inputs or {}).items():
            key_s = str(k)
            rewritten_v = _rewrite_step_input_value(
                v,
                target_tool_name=str(step.tool or ""),
                target_arg_key=key_s,
                blueprint_default_present=(key_s in merged_args),
                sketch_step_to_compiled_id=sketch_step_to_compiled_id,
                compiled_tool_by_node_id=compiled_tool_by_node_id,
                case_ref_is_file=bool(case_ref_is_file),
                rewrite_stats=rewrite_stats,
                diagnostics=diagnostics,
                current_compiled_node_id=cid,
            )
            if rewritten_v is _DROP_OVERRIDE:
                continue
            merged_args[key_s] = rewritten_v
        compiled_required = bool(chosen.required)
        if bool(step.optional) and bool(chosen.required):
            _warn(
                f"Sketch step {sid} marked optional but blueprint node {cid} is required; "
                "preserving blueprint required=True."
            )
        elif (not bool(step.optional)) and (not bool(chosen.required)):
            _warn(
                f"Sketch step {sid} marked required but blueprint node {cid} is optional; "
                "preserving blueprint required=False."
            )
        compiled = chosen.model_copy(
            deep=True,
            update={
                "arguments": merged_args,
                # Preserve template/blueprint required semantics so optional QA/compare
                # nodes do not become required just because the sketch omitted optional=true.
                "required": compiled_required,
                "depends_on": list(step.depends_on or []),  # temporary: sketch step_ids; remap below
                "label": (str(step.goal or "").strip() or chosen.label),
            },
        )
        compiled_by_node_id[cid] = compiled
    diagnostics["rewrite_stats"] = dict(rewrite_stats)

    # --- Remap explicit deps to compiled node ids and union with blueprint deps ---
    for step in sketch.steps:
        cid = sketch_step_to_compiled_id.get(str(step.step_id))
        if not cid or cid not in compiled_by_node_id:
            continue
        compiled = compiled_by_node_id[cid]
        bp = by_id.get(cid)
        sketch_dep_ids: List[str] = []
        for dep_sid in (step.depends_on or []):
            dep_sid_s = str(dep_sid or "").strip()
            dep_cid = sketch_step_to_compiled_id.get(dep_sid_s)
            if not dep_cid:
                continue
            dep_node = compiled_by_node_id.get(dep_cid)
            if dep_node is None:
                continue
            if bool(compiled.required) and (not bool(dep_node.required)):
                _warn(
                    f"Dropped sketch dependency {cid}->{dep_cid} because downstream node is required "
                    "but dependency node is optional (avoids blocking when optional nodes are skipped)."
                )
                continue
            sketch_dep_ids.append(dep_cid)
        bp_dep_ids = [str(x) for x in ((bp.depends_on if bp else []) or []) if str(x).strip()]
        merged_dep_ids = _dedup_keep_order(sketch_dep_ids + bp_dep_ids)
        added = [d for d in bp_dep_ids if d not in sketch_dep_ids]
        if added:
            diagnostics["added_blueprint_dependencies"].append(
                {
                    "sketch_step_id": str(step.step_id),
                    "compiled_node_id": cid,
                    "deps_added": added,
                    "reason": "preserve blueprint executable dependency closure",
                }
            )
        compiled.depends_on = merged_dep_ids

    # --- Autofill required blueprint closure (task/domain contract proxy) ---
    target_blueprint_ids: Set[str] = set(compiled_explicit_ids)
    if ensure_required_blueprint_nodes:
        target_blueprint_ids.update({nid for nid, bp in by_id.items() if bool(bp.required)})

    # Validate final_targets (if present) map to sketch step ids.
    if sketch.final_targets:
        for target in sketch.final_targets:
            if target not in sketch_step_to_compiled_id:
                _warn(f"final_targets references unknown sketch step_id={target}")
            else:
                target_blueprint_ids.add(sketch_step_to_compiled_id[target])

    closure_ids = _collect_dependency_closure(target_blueprint_ids, by_id)
    for nid in blueprint_order:
        if nid not in closure_ids:
            continue
        if nid in compiled_by_node_id:
            continue
        bp = by_id[nid]
        compiled_by_node_id[nid] = bp.model_copy(deep=True)
        diagnostics["autofilled_nodes"].append(
            {
                "compiled_node_id": nid,
                "tool": str(bp.tool_name or ""),
                "origin": "compiler_autofill",
                "required": bool(bp.required),
                "reason": (
                    "required_blueprint_closure"
                    if bool(bp.required)
                    else "dependency_closure_for_selected_or_required_nodes"
                ),
            }
        )

    # --- Final dependency validation on compiled nodes ---
    final_nodes = [compiled_by_node_id[nid] for nid in blueprint_order if nid in compiled_by_node_id]
    final_by_id = {str(n.node_id): n for n in final_nodes}
    dep_errors: List[str] = []
    edges: Dict[str, List[str]] = {}
    for node in final_nodes:
        nid = str(node.node_id)
        deps = [str(d) for d in (node.depends_on or []) if str(d).strip()]
        edges[nid] = deps
        for dep in deps:
            if dep not in final_by_id:
                dep_errors.append(f"Compiled node {nid} depends_on missing node_id={dep}")
        unresolved_sketch_refs: List[str] = []
        _collect_unresolved_sketch_node_refs(node.arguments, sketch_step_ids=step_id_set, out=unresolved_sketch_refs)
        if unresolved_sketch_refs:
            uniq = _dedup_keep_order(unresolved_sketch_refs)
            dep_errors.append(
                f"Compiled node {nid} contains unresolved sketch @node refs: {', '.join(uniq[:6])}"
            )
    cyc2 = _detect_cycle_edges(edges)
    if dep_errors:
        for msg in dep_errors:
            _err(msg)
    if cyc2:
        _err("Compiled DAG cycle detected: " + " -> ".join(cyc2))
    if diagnostics["errors"]:
        return SketchCompileResult(ok=False, nodes=[], diagnostics=diagnostics)

    diagnostics["cycle_valid"] = True
    diagnostics["compile_ok"] = True
    diagnostics["compiled_summary"] = {
        "node_count": len(final_nodes),
        "explicit_sketch_nodes": len(compiled_explicit_ids),
        "autofilled_nodes": len(diagnostics["autofilled_nodes"]),
    }
    return SketchCompileResult(ok=True, nodes=_toposort_nodes(final_nodes), diagnostics=diagnostics)

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_canonical(obj: Any) -> str:
    """
    Canonical JSON string for hashing/logging.
    - stable ordering (sort_keys)
    - compact separators
    - UTF-8 friendly (ensure_ascii=False)
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json_canonical(obj))


def environment_snapshot() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "pid": os.getpid(),
    }


def _to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


def normalize_artifacts(
    items: Optional[Sequence[Any]],
    *,
    default_kind: str = "unknown",
    default_description: str = "",
) -> List["ArtifactRef"]:
    """
    Normalize a list of artifact-like objects into a list[ArtifactRef].

    Accepted item types:
    - ArtifactRef
    - dict (must have at least 'path'; kind/description/media_type optional)
    - str / Path (treated as path)
    - None (ignored)
    """
    if not items:
        return []

    out: List[ArtifactRef] = []
    for a in items:
        if a is None:
            continue
        if isinstance(a, ArtifactRef):
            out.append(a)
            continue
        if isinstance(a, Path):
            out.append(ArtifactRef(path=str(a), kind=default_kind, description=default_description))
            continue
        if isinstance(a, str):
            out.append(ArtifactRef(path=a, kind=default_kind, description=default_description))
            continue
        if isinstance(a, dict):
            if "path" not in a:
                raise ValueError(f"Artifact dict missing 'path': {a}")
            out.append(
                ArtifactRef(
                    path=str(a.get("path")),
                    kind=str(a.get("kind") or default_kind),
                    description=str(a.get("description") or default_description),
                    media_type=(str(a["media_type"]) if a.get("media_type") is not None else None),
                )
            )
            continue
        # Fallback: stringify
        out.append(ArtifactRef(path=str(a), kind=default_kind, description=default_description))
    return out


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    version: str = "0.0.0"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str
    case_id: str
    stage: str = "misc"
    requested_by: str = "agent"

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ToolError:
    type: str
    message: str
    stack: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    kind: str  # e.g., "json", "csv", "text", "nifti", "transform", "figure"
    description: str = ""
    media_type: Optional[str] = None  # e.g., "application/json"

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[ArtifactRef] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[ToolError] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)


@dataclass(frozen=True)
class ExecutionLogEntry:
    timestamp: str
    tool_name: str
    tool_version: str
    call_id: str
    case_id: str
    stage: str
    requested_by: str
    arguments: Dict[str, Any]
    ok: bool
    warnings: List[str]
    error: Optional[Dict[str, Any]]
    runtime_ms: int
    input_hash: str
    output_hash: str
    environment: Dict[str, Any]
    source_artifacts: List[ArtifactRef]
    generated_artifacts: List[ArtifactRef]

    def to_jsonl(self) -> str:
        return json_canonical(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)


@dataclass
class CaseState:
    case_id: str
    run_id: str
    created_at: str = field(default_factory=now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Stage-organized outputs, e.g.:
    # {
    #   "ingest": {"ingest_dicom_to_nifti": [{"call_id": "...", "ok": True, "data": {...}}]},
    #   "registration": {"register_to_reference": [{"call_id": "...", "ok": True, "data": {...}}, ...]},
    # }
    stage_outputs: Dict[str, Dict[str, List[Dict[str, Any]]]] = field(default_factory=dict)
    # Stable stage metadata for display/sorting (semantic stage_name -> stage_order).
    # This keeps naming consistent even if legacy runs used numeric stage keys.
    stage_meta: Dict[str, int] = field(default_factory=dict)
    artifacts_index: List[ArtifactRef] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _to_jsonable(self)

    def add_artifacts(self, artifacts: List[ArtifactRef]) -> None:
        self.artifacts_index.extend(artifacts)

    def add_stage_record(
        self,
        stage: str,
        tool_name: str,
        call_id: str,
        ok: bool,
        data: Dict[str, Any],
        *,
        stage_order: Optional[int] = None,
    ) -> None:
        if stage not in self.stage_outputs:
            self.stage_outputs[stage] = {}
        if tool_name not in self.stage_outputs[stage]:
            self.stage_outputs[stage][tool_name] = []
        rec: Dict[str, Any] = {"call_id": call_id, "ok": ok, "data": data}
        if stage_order is not None:
            rec["stage_order"] = int(stage_order)
            self.stage_meta.setdefault(stage, int(stage_order))
        self.stage_outputs[stage][tool_name].append(rec)

    def set_summary(self, summary: Dict[str, Any]) -> None:
        self.summary = summary

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ToolContext:
    case_id: str
    run_id: str
    run_dir: Path
    artifacts_dir: Path
    case_state_path: Path


ToolFn = Callable[[Dict[str, Any], ToolContext], Dict[str, Any]]

# Future direction:
# - Keep dataclasses for the MVP.
# - If/when we want auto JSON Schema generation + stronger validation,
#   we can migrate these to pydantic models (v2).
ToolReturn = Union[ToolResult, Dict[str, Any]]
from __future__ import annotations

import glob
import json
import shlex
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from commands.registry import Tool
from commands.schemas import ArtifactRef, ToolContext, ToolSpec
from runtime.sandbox_exec import run_command


SANDBOX_SPEC = ToolSpec(
    name="sandbox_exec",
    description=(
        "Execute a shell command in a lightweight working directory under the run artifacts. "
        "This is a prototype utility for simple scripting; it is NOT a security sandbox."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "cmd": {"type": "string", "description": "Command string to execute"},
            "timeout_s": {"type": "integer", "default": 30},
            "output_subdir": {"type": "string", "default": "sandbox"},
            "workdir": {"type": "string", "description": "Optional workdir (absolute or relative to artifacts_dir)"},
            "capture_globs": {"type": "array", "description": "File globs to capture after execution"},
            "max_output_chars": {"type": "integer", "default": 4000},
        },
        "required": ["cmd"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "cmd": {"type": "string"},
            "returncode": {"type": "integer"},
            "workdir": {"type": "string"},
            "stdout_path": {"type": "string"},
            "stderr_path": {"type": "string"},
            "stdout_preview": {"type": "string"},
            "stderr_preview": {"type": "string"},
            "captured_files": {"type": "array"},
        },
        "required": ["cmd", "returncode", "workdir", "stdout_path", "stderr_path"],
    },
    version="0.1.0",
    tags=["sandbox", "exec", "experimental"],
)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[TRUNCATED]"


def _resolve_workdir(ctx: ToolContext, args: Dict[str, Any]) -> Path:
    output_subdir = str(args.get("output_subdir") or "sandbox")
    workdir_arg = args.get("workdir")
    if workdir_arg:
        p = Path(str(workdir_arg)).expanduser()
        if not p.is_absolute():
            p = ctx.artifacts_dir / p
    else:
        p = ctx.artifacts_dir / output_subdir / f"run_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _collect_globbed_files(workdir: Path, patterns: List[str]) -> List[str]:
    captured: List[str] = []
    for pat in patterns:
        if not isinstance(pat, str) or not pat.strip():
            continue
        if Path(pat).is_absolute():
            matches = glob.glob(pat)
        else:
            matches = glob.glob(str(workdir / pat))
        for m in matches:
            try:
                p = Path(m)
                if p.is_file():
                    captured.append(str(p.resolve()))
            except Exception:
                continue
    return sorted(set(captured))


def sandbox_exec_tool(args: Dict[str, Any], ctx: ToolContext) -> Dict[str, Any]:
    cmd_arg = args.get("cmd")
    if cmd_arg is None:
        raise ValueError("sandbox_exec requires cmd")
    cmd_str = str(cmd_arg).strip()
    if not cmd_str:
        raise ValueError("sandbox_exec requires non-empty cmd")

    timeout_s = int(args.get("timeout_s") or 30)
    max_output_chars = int(args.get("max_output_chars") or 4000)
    workdir = _resolve_workdir(ctx, args)

    if isinstance(cmd_arg, list):
        cmd = [str(x) for x in cmd_arg]
    else:
        cmd = shlex.split(cmd_str)

    result = run_command(cmd, workdir=workdir, timeout_s=timeout_s)
    stdout = result.get("stdout") if isinstance(result, dict) else ""
    stderr = result.get("stderr") if isinstance(result, dict) else ""
    returncode = result.get("returncode") if isinstance(result, dict) else 1

    stdout_path = workdir / "stdout.txt"
    stderr_path = workdir / "stderr.txt"
    stdout_path.write_text(stdout or "", encoding="utf-8")
    stderr_path.write_text(stderr or "", encoding="utf-8")

    captured_files: List[str] = []
    if isinstance(args.get("capture_globs"), list):
        captured_files = _collect_globbed_files(workdir, list(args.get("capture_globs") or []))

    data = {
        "cmd": cmd_str,
        "returncode": int(returncode) if returncode is not None else 1,
        "workdir": str(workdir),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_preview": _truncate(stdout or "", max_output_chars),
        "stderr_preview": _truncate(stderr or "", max_output_chars),
        "captured_files": captured_files,
    }

    artifacts: List[ArtifactRef] = [
        ArtifactRef(path=str(stdout_path), kind="text", description="Sandbox stdout"),
        ArtifactRef(path=str(stderr_path), kind="text", description="Sandbox stderr"),
    ]
    for fp in captured_files:
        artifacts.append(ArtifactRef(path=str(fp), kind="file", description="Sandbox output"))

    result_path = workdir / "result.json"
    result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    artifacts.append(ArtifactRef(path=str(result_path), kind="json", description="Sandbox result"))

    warnings: List[str] = ["experimental tool"]
    try:
        art_root = ctx.artifacts_dir.resolve()
        if art_root not in workdir.resolve().parents and art_root != workdir.resolve():
            warnings.append("workdir is outside artifacts_dir; sandbox_exec is not a security boundary")
    except Exception:
        pass

    return {"data": data, "artifacts": artifacts, "generated_artifacts": artifacts, "warnings": warnings}


def build_tool() -> Tool:
    return Tool(spec=SANDBOX_SPEC, func=sandbox_exec_tool)

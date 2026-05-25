"""
Smoke benchmark for BCER.

Exercises the tool registry and dispatcher end-to-end with three dummy tools
(load → segment → report). Does not require medical data, model weights, GPUs,
or LLM API keys. Use this to verify that an install is working.

Usage:

    python -m benchmark.smoke
    python -m benchmark.smoke --runs-root /tmp/bcer_smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Tuple

from commands.dispatcher import ToolDispatcher
from commands.registry import ToolRegistry
from commands.schemas import ToolCall, ToolResult
from mri_agent_shell.dummy_tools import build_dummy_tools


_STEPS: List[Tuple[str, dict]] = [
    (
        "dummy_load_case",
        {"case_path": "/synthetic/case/path", "output_subdir": "01_load"},
    ),
    (
        "dummy_segment",
        {"case_path": "/synthetic/case/path", "anatomy": "demo", "output_subdir": "02_segment"},
    ),
    (
        "dummy_generate_report",
        {"case_id": "smoke_case", "output_subdir": "03_report"},
    ),
]


def _build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for tool in build_dummy_tools():
        reg.register(tool)
    return reg


def _run_one_step(
    *,
    dispatcher: ToolDispatcher,
    state,
    ctx,
    tool_name: str,
    arguments: dict,
    prior_outputs: dict,
) -> Tuple[bool, ToolResult]:
    if tool_name == "dummy_generate_report":
        arguments = {**arguments, "mask_path": prior_outputs.get("mask_path", "")}
    call = ToolCall(
        tool_name=tool_name,
        arguments=arguments,
        call_id=f"smoke-{uuid.uuid4().hex[:8]}",
        case_id=state.case_id,
        stage="smoke",
    )
    result = dispatcher.dispatch(call, state, ctx)
    return result.ok, result


def run_smoke(runs_root: Path) -> int:
    runs_root.mkdir(parents=True, exist_ok=True)
    registry = _build_registry()
    dispatcher = ToolDispatcher(registry=registry, runs_root=runs_root)
    state, ctx = dispatcher.create_run(case_id="smoke_case", run_id="smoke_run")

    print(f"[smoke] runs_root: {runs_root}")
    print(f"[smoke] run_dir:   {ctx.run_dir}")
    print(f"[smoke] {len(_STEPS)} tools registered: {[s['name'] for s in registry.list_specs()]}")
    print()

    prior_outputs: dict = {}
    failures = 0
    for i, (tool_name, args) in enumerate(_STEPS, start=1):
        ok, result = _run_one_step(
            dispatcher=dispatcher,
            state=state,
            ctx=ctx,
            tool_name=tool_name,
            arguments=args,
            prior_outputs=prior_outputs,
        )
        if ok:
            prior_outputs.update(result.data)
            artifact_paths = [a.path for a in result.artifacts]
            print(f"[smoke] step {i}/{len(_STEPS)} {tool_name:<22} OK   -> {artifact_paths}")
        else:
            failures += 1
            err = result.error.message if result.error else "unknown"
            print(f"[smoke] step {i}/{len(_STEPS)} {tool_name:<22} FAIL -> {err}", file=sys.stderr)

    print()
    if failures == 0:
        print(f"[smoke] PASS  3/3 tool dispatches succeeded. Artifacts under {ctx.artifacts_dir}")
        return 0
    print(f"[smoke] FAIL  {failures}/{len(_STEPS)} steps failed", file=sys.stderr)
    return 1


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Directory for run artifacts (default: a fresh tempdir, cleaned on exit).",
    )
    ap.add_argument(
        "--keep",
        action="store_true",
        help="With --runs-root not set, keep the tempdir instead of deleting it.",
    )
    args = ap.parse_args(argv)

    if args.runs_root is not None:
        return run_smoke(args.runs_root.resolve())

    tmp = Path(tempfile.mkdtemp(prefix="bcer_smoke_"))
    try:
        rc = run_smoke(tmp)
        if args.keep:
            print(f"[smoke] kept tempdir: {tmp}")
        return rc
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

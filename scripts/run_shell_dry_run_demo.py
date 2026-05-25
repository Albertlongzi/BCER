#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_import_path() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> int:
    _ensure_import_path()

    from mri_agent_shell.runtime.session import ModelConfig, SessionState  # noqa: WPS433
    from mri_agent_shell.shell.repl import AgentShell  # noqa: WPS433

    ap = argparse.ArgumentParser(description="Run MRI Agent Shell dry-run demo end-to-end.")
    ap.add_argument("--workspace", required=True, help="Workspace path for demo outputs.")
    args = ap.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    case_dir = workspace / "demo_case"
    case_dir.mkdir(parents=True, exist_ok=True)

    session = SessionState(
        workspace_path=str(workspace),
        runs_root=str(workspace / "runs"),
        model_config=ModelConfig(provider="stub", llm="stub", vlm="stub"),
        dry_run=True,
    )
    shell = AgentShell(session=session, include_core_tools=False)

    shell.handle_line(f":workspace set {workspace}", emit=print)
    shell.handle_line(f":case load {case_dir}", emit=print)

    # Request -> plan
    shell.handle_line("process this medical case and generate a report", emit=print)

    # If template is returned, fill and submit automatically.
    if not session.current_plan and session.current_template:
        tpl = dict(session.current_template)
        req = dict(tpl.get("REQUIRED") or {})
        req["case_ref"] = "case.input"
        tpl["REQUIRED"] = req
        shell.handle_line(json.dumps(tpl, ensure_ascii=False), emit=print)

    shell.handle_line(":run", emit=print)

    if not session.last_run:
        print("[FAIL] no last_run state found")
        return 2

    print("[OK] dry-run finished")
    print(json.dumps(session.last_run, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

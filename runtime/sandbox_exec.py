from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional


def run_command(cmd: List[str], *, workdir: Optional[Path] = None, timeout_s: int = 30) -> Dict[str, str]:
    """
    Lightweight sandbox execution stub.

    This runs a command in a temporary directory with a minimal environment.
    It is NOT a security sandbox; it is a placeholder interface for future hardening.
    """
    env = {"PATH": os.environ.get("PATH", "")}
    temp_dir = None
    if workdir is None:
        temp_dir = tempfile.TemporaryDirectory()
        workdir = Path(temp_dir.name)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workdir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(timeout_s),
        )
        return {
            "returncode": int(proc.returncode),
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "workdir": str(workdir),
        }
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def main() -> None:
    ap = argparse.ArgumentParser(description="Lightweight sandbox execution stub.")
    ap.add_argument("--cmd", required=True, help="Command string to execute (e.g., \"python -c 'print(1)'\")")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--workdir", default="")
    args = ap.parse_args()

    cmd = shlex.split(args.cmd)
    workdir = Path(args.workdir).expanduser().resolve() if args.workdir else None
    result = run_command(cmd, workdir=workdir, timeout_s=int(args.timeout))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

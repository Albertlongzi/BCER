from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from mri_agent_shell.runtime.session import ModelConfig, SessionState
from mri_agent_shell.shell.repl import AgentShell


class ShellDoctorTests(unittest.TestCase):
    def test_doctor_reports_missing_openai_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td)
            old = os.environ.pop("OPENAI_API_KEY", None)
            try:
                session = SessionState(
                    workspace_path=str(ws),
                    runs_root=str(ws / "runs"),
                    model_config=ModelConfig(provider="openai_official", llm="gpt-4o-mini"),
                    dry_run=True,
                )
                shell = AgentShell(session=session, include_core_tools=False)
                lines: list[str] = []
                shell.run_doctor(emit=lines.append, auto=False)
                txt = "\n".join(lines)
                self.assertIn("missing OPENAI_API_KEY", txt)
            finally:
                if old is not None:
                    os.environ["OPENAI_API_KEY"] = old

    def test_domains_command_lists_discovered_domains(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            session = SessionState(
                workspace_path=td,
                runs_root=str(Path(td) / "runs"),
                model_config=ModelConfig(provider="stub", llm="stub"),
                dry_run=False,
            )
            shell = AgentShell(session=session, include_core_tools=True)
            lines: list[str] = []
            shell.handle_line(":domains", emit=lines.append)
            txt = "\n".join(lines)
            self.assertIn("prostate", txt)
            self.assertIn("brain", txt)
            self.assertIn("cardiac", txt)


if __name__ == "__main__":
    unittest.main()

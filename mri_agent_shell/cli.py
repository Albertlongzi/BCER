from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .bootstrap import ensure_import_paths
from .runtime.session import (
    ModelConfig,
    SessionState,
    is_known_provider,
    normalize_provider,
    provider_default_model,
    provider_choices,
    provider_help,
    validate_workspace_path,
)
from .shell.repl import AgentShell
from .shell.tui import run_tui

ensure_import_paths()


def _load_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        default = Path.home() / ".config" / "mri_agent_shell" / "config.yaml"
        p = default
    else:
        p = Path(path).expanduser().resolve()

    if not p.exists() or not p.is_file():
        return {}

    raw = p.read_text(encoding="utf-8")
    obj = yaml.safe_load(raw)
    if not isinstance(obj, dict):
        return {}
    return obj


def _env_config() -> Dict[str, Any]:
    model: Dict[str, Any] = {}

    def _env(*names: str) -> str:
        for name in names:
            val = os.environ.get(name)
            if val is not None and str(val).strip() != "":
                return str(val)
        return ""

    provider = _env("MRI_AGENT_SHELL_PROVIDER")
    if provider:
        model["provider"] = provider
    llm = _env("MRI_AGENT_SHELL_LLM")
    if llm:
        model["llm"] = llm
    vlm = _env("MRI_AGENT_SHELL_VLM")
    if vlm:
        model["vlm"] = vlm

    base_url = _env("MRI_AGENT_SHELL_SERVER_BASE_URL", "MEDGEMMA_SERVER_BASE_URL", "MRI_AGENT_SERVER_BASE_URL")
    if base_url:
        model["server_base_url"] = base_url

    api_base_url = _env("MRI_AGENT_SHELL_API_BASE_URL", "OPENAI_BASE_URL")
    if api_base_url:
        model["api_base_url"] = api_base_url

    api_key = _env("MRI_AGENT_SHELL_API_KEY")
    if api_key:
        model["api_key"] = api_key

    max_tokens = _env("MRI_AGENT_SHELL_MAX_TOKENS")
    if max_tokens:
        model["max_tokens"] = max_tokens

    temp = _env("MRI_AGENT_SHELL_TEMPERATURE")
    if temp:
        model["temperature"] = temp

    out: Dict[str, Any] = {}
    if model:
        out["model"] = model

    ws = _env("MRI_AGENT_SHELL_WORKSPACE")
    if ws:
        out["workspace"] = ws
    rr = _env("MRI_AGENT_SHELL_RUNS_ROOT")
    if rr:
        out["runs_root"] = rr

    return out


def _merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (incoming or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            nested = dict(out.get(k) or {})
            nested.update(v)
            out[k] = nested
        else:
            out[k] = v
    return out


def _cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    model: Dict[str, Any] = {}

    for key in ("provider", "llm", "vlm", "base_url", "api_base_url", "max_tokens", "temperature"):
        val = getattr(args, key, None)
        if val is None:
            continue
        if key == "base_url":
            model["server_base_url"] = str(val)
        else:
            model[key] = val

    if model:
        out["model"] = model

    if args.workspace is not None:
        out["workspace"] = args.workspace
    if args.runs_root is not None:
        out["runs_root"] = args.runs_root

    return out


def _prompt_field(*, label: str, default: str, help_text: str, validator=None) -> str:
    while True:
        try:
            raw = input(f"{label} [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return default
        if raw == "":
            candidate = default
        elif raw == "?":
            print(help_text)
            continue
        else:
            candidate = raw

        if validator is None:
            return candidate

        ok, msg = validator(candidate)
        if ok:
            return candidate
        print(f"Invalid value: {msg}")


def _workspace_validator(raw: str) -> tuple[bool, str]:
    ok, msg, _ = validate_workspace_path(raw)
    return ok, msg


def _provider_validator(raw: str) -> tuple[bool, str]:
    if not is_known_provider(raw):
        return False, f"unknown provider '{raw}'. Choices: {', '.join(provider_choices())}"
    p = normalize_provider(raw)
    return True, "ok"


def _interactive_prompt(model: ModelConfig, workspace: str) -> tuple[ModelConfig, str]:
    if not sys.stdin.isatty():
        return model, workspace

    print("Config Prompt (ENTER to accept default, '?' for field help)")
    provider_raw = _prompt_field(
        label="provider",
        default=model.provider,
        help_text=(
            "Providers:\n"
            "- openai_official: OpenAI API (requires OPENAI_API_KEY env/config)\n"
            "- openai_compatible_server: OpenAI-compatible base_url backend (vLLM/proxy)\n"
            "- stub: offline deterministic mode\n"
            "- anthropic, gemini: optional cloud backends"
        ),
        validator=_provider_validator,
    )
    provider = normalize_provider(provider_raw)
    model.provider = provider

    llm_default = model.llm or provider_default_model(provider)
    model.llm = _prompt_field(
        label="llm model",
        default=llm_default,
        help_text=f"Primary LLM model name for provider '{provider}'.",
    )

    model.vlm = _prompt_field(
        label="vlm model",
        default=str(model.vlm or model.llm),
        help_text="Vision-capable model name. ENTER reuses llm.",
    )

    if provider == "openai_compatible_server":
        model.server_base_url = _prompt_field(
            label="server base_url",
            default=str(model.server_base_url or "http://127.0.0.1:8000/v1"),
            help_text="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1",
        )
    else:
        # Do not prompt for API keys in clear text.
        print(provider_help(provider))

    ws = _prompt_field(
        label="workspace path",
        default=workspace,
        help_text="Workspace directory used for reports/logs/runs.",
        validator=_workspace_validator,
    )

    model._defaults()
    return model, ws


def _resolve_config(args: argparse.Namespace) -> tuple[ModelConfig, str, str]:
    defaults: Dict[str, Any] = {
        "workspace": str(Path.cwd()),
        "runs_root": "",
        "model": {
            "provider": "openai_compatible_server",
            "llm": "",
            "vlm": "",
            "server_base_url": "http://127.0.0.1:8000/v1",
            "api_base_url": "",
            "api_key": "",
            "max_tokens": 2048,
            "temperature": 0.0,
        },
    }

    env_cfg = _env_config()
    file_cfg = _load_config_file(args.config)
    cli_cfg = _cli_overrides(args)

    # Priority: CLI > config file > env > defaults
    merged = _merge(defaults, env_cfg)
    merged = _merge(merged, file_cfg)
    merged = _merge(merged, cli_cfg)

    model_cfg = ModelConfig(**dict(merged.get("model") or {}))

    workspace_raw = str(merged.get("workspace") or Path.cwd())
    ok, msg, workspace = validate_workspace_path(workspace_raw)
    if not ok:
        raise ValueError(f"workspace invalid: {msg}")

    runs_root_raw = str(merged.get("runs_root") or "").strip()
    if runs_root_raw:
        runs_root = str(Path(runs_root_raw).expanduser().resolve())
    else:
        runs_root = str(Path(workspace) / "runs")

    Path(runs_root).mkdir(parents=True, exist_ok=True)
    return model_cfg, workspace, runs_root


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Interactive MRI Agent shell")
    ap.add_argument("--plain", action="store_true", help="Force plain REPL (disable Textual UI).")
    ap.add_argument("--dry-run", action="store_true", help="Use dummy tools for end-to-end demo flow.")

    ap.add_argument("--config", default=None, help="Config file path (yaml/json). Default: ~/.config/mri_agent_shell/config.yaml")
    ap.add_argument("--workspace", default=None, help="Workspace path for reports/logs/runs.")
    ap.add_argument("--runs-root", default=None, help="Runs root path (default: <workspace>/runs).")

    ap.add_argument(
        "--provider",
        default=None,
        choices=[
            "openai_official",
            "openai_compatible_server",
            "stub",
            "anthropic",
            "gemini",
            # legacy aliases kept for compatibility:
            "openai",
            "server",
        ],
    )
    ap.add_argument("--llm", default=None)
    ap.add_argument("--vlm", default=None)
    ap.add_argument("--base-url", default=None, help="Alias for server base_url.")
    ap.add_argument("--api-base-url", default=None, help="Optional API base url for compatible gateways.")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)

    ap.add_argument("--no-prompt", action="store_true", help="Skip optional interactive config prompt.")
    ap.add_argument("--no-wizard", action="store_true", help=argparse.SUPPRESS)  # backward-compatible alias
    ap.add_argument("--script", default="", help="Execute a script file of shell commands non-interactively.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        model, workspace, runs_root = _resolve_config(args)
    except Exception as e:
        print(f"Configuration error: {e}")
        return 2

    if not args.script and not args.no_prompt and not args.no_wizard:
        try:
            model, workspace = _interactive_prompt(model, workspace)
            ok, msg, workspace = validate_workspace_path(workspace)
            if not ok:
                print(f"Configuration error: workspace invalid: {msg}")
                return 2
            runs_root = str(Path(workspace) / "runs")
            Path(runs_root).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Configuration error: {e}")
            return 2

    try:
        session = SessionState(
            workspace_path=workspace,
            runs_root=runs_root,
            model_config=model,
            dry_run=bool(args.dry_run),
        )
    except Exception as e:
        print(f"Session init error: {e}")
        return 2

    shell = AgentShell(session=session)

    if not args.script:
        shell.run_doctor(emit=print, auto=True)
        shell._doctor_ran = True  # keep startup check exactly once

    if args.script:
        return shell.run_script(Path(str(args.script)))

    if not args.plain:
        launched = run_tui(shell)
        if launched:
            return 0
        print('Textual not available. Install with: pip install -e ".[tui]". Falling back to plain REPL.')

    return shell.run_interactive()

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


_PROVIDER_ALIASES = {
    "server": "openai_compatible_server",
    "openai_compatible_server": "openai_compatible_server",
    "openai-compatible-server": "openai_compatible_server",
    "openai_server": "openai_compatible_server",
    "openai_compat": "openai_compatible_server",
    "openai": "openai_official",
    "openai_official": "openai_official",
    "official_openai": "openai_official",
    "stub": "stub",
    "anthropic": "anthropic",
    "gemini": "gemini",
}

_PROVIDER_DEFAULT_MODEL = {
    "openai_compatible_server": "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "openai_official": "gpt-4o-mini",
    "anthropic": "claude-3-7-sonnet-20250219",
    "gemini": "gemini-1.5-pro",
    "stub": "stub",
}

_PROVIDER_HELP = {
    "openai_compatible_server": (
        "OpenAI-compatible server backend (vLLM/proxy). "
        "Requires base_url like http://host:port/v1."
    ),
    "openai_official": "OpenAI official API backend. Requires OPENAI_API_KEY env or config key.",
    "stub": "No external calls. Deterministic local behavior for testing.",
    "anthropic": "Anthropic API backend. Requires ANTHROPIC_API_KEY env.",
    "gemini": "Gemini API backend. Requires GOOGLE_API_KEY env.",
}


def is_known_provider(provider: str) -> bool:
    raw = str(provider or "").strip().lower()
    return raw in _PROVIDER_ALIASES


def provider_choices() -> Tuple[str, ...]:
    return tuple(sorted(_PROVIDER_DEFAULT_MODEL.keys()))


def normalize_provider(provider: str) -> str:
    raw = str(provider or "openai_compatible_server").strip().lower()
    return _PROVIDER_ALIASES.get(raw, "openai_compatible_server")


def provider_default_model(provider: str) -> str:
    p = normalize_provider(provider)
    return _PROVIDER_DEFAULT_MODEL.get(p, _PROVIDER_DEFAULT_MODEL["openai_compatible_server"])


def provider_help(provider: str) -> str:
    p = normalize_provider(provider)
    return _PROVIDER_HELP.get(p, "")


def mask_secret(value: str) -> str:
    s = str(value or "")
    if not s:
        return ""
    if len(s) <= 8:
        return "*" * len(s)
    return "*" * (len(s) - 4) + s[-4:]


def validate_workspace_path(path: str) -> Tuple[bool, str, str]:
    raw = str(path or "").strip()
    if not raw:
        return False, "workspace path is empty", raw
    if raw == "?":
        return False, "'?' is help marker, not a path", raw

    try:
        p = Path(raw).expanduser().resolve()
    except Exception as e:
        return False, f"failed to resolve workspace path: {e}", raw

    if not p.exists():
        return False, "workspace directory does not exist", str(p)

    if not p.exists() or not p.is_dir():
        return False, "workspace is not a directory", str(p)

    # Writability check via probe file.
    probe = p / ".mri_agent_shell_write_test"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception as e:
        return False, f"workspace is not writable: {e}", str(p)

    return True, "ok", str(p)


@dataclass
class ModelConfig:
    provider: str = "openai_compatible_server"
    llm: str = _PROVIDER_DEFAULT_MODEL["openai_compatible_server"]
    vlm: Optional[str] = None
    server_base_url: str = "http://127.0.0.1:8000/v1"
    api_base_url: str = ""
    api_key: str = ""
    max_tokens: int = 2048
    temperature: float = 0.0

    def __post_init__(self) -> None:
        self._defaults()

    def _defaults(self) -> None:
        self.provider = normalize_provider(self.provider)

        if not str(self.llm or "").strip():
            self.llm = provider_default_model(self.provider)
        if not str(self.vlm or "").strip():
            self.vlm = str(self.llm)

        if self.provider == "openai_compatible_server" and not str(self.server_base_url or "").strip():
            self.server_base_url = "http://127.0.0.1:8000/v1"

        self.max_tokens = max(64, int(self.max_tokens or 2048))
        self.temperature = float(self.temperature or 0.0)

    def update_from_kwargs(self, values: Dict[str, Any]) -> None:
        if not isinstance(values, dict):
            return
        for key, value in values.items():
            if value is None:
                continue
            if key in {"base_url", "server_base_url"}:
                self.server_base_url = str(value)
                continue
            if key == "api_base_url":
                self.api_base_url = str(value)
                continue
            if key in {"key", "api_key"}:
                self.api_key = str(value)
                continue
            if key == "provider":
                self.provider = str(value)
                continue
            if key == "llm":
                self.llm = str(value)
                continue
            if key == "vlm":
                self.vlm = str(value)
                continue
            if key == "max_tokens":
                try:
                    self.max_tokens = int(value)
                except Exception:
                    pass
                continue
            if key == "temperature":
                try:
                    self.temperature = float(value)
                except Exception:
                    pass
                continue
        self._defaults()

    def effective_api_key(self) -> str:
        if str(self.api_key or "").strip():
            return str(self.api_key).strip()
        if self.provider == "openai_official":
            return str(os.environ.get("OPENAI_API_KEY", "")).strip()
        if self.provider == "anthropic":
            return str(os.environ.get("ANTHROPIC_API_KEY", "")).strip()
        if self.provider == "gemini":
            return str(os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", ""))).strip()
        if self.provider == "openai_compatible_server":
            return str(os.environ.get("MRI_AGENT_SHELL_SERVER_API_KEY", os.environ.get("OPENAI_API_KEY", ""))).strip()
        return ""

    def model_dump(self) -> Dict[str, Any]:
        data = asdict(self)
        data["api_key"] = mask_secret(self.effective_api_key()) if self.effective_api_key() else ""
        data["api_key_set"] = bool(self.effective_api_key())
        data["provider_help"] = provider_help(self.provider)
        return data


@dataclass
class SessionState:
    workspace_path: str
    runs_root: str
    case_id: Optional[str] = None
    case_inputs: Dict[str, str] = field(default_factory=dict)
    path_keys: Dict[str, str] = field(default_factory=dict)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    current_request: Optional[str] = None
    current_template: Optional[Dict[str, Any]] = None
    current_plan: Optional[Dict[str, Any]] = None
    last_run: Optional[Dict[str, Any]] = None
    dry_run: bool = False

    def __post_init__(self) -> None:
        ok, msg, resolved = validate_workspace_path(self.workspace_path)
        if not ok:
            raise ValueError(f"Invalid workspace path '{self.workspace_path}': {msg}")

        self.workspace_path = resolved

        rr_raw = str(self.runs_root or "").strip()
        rr = Path(rr_raw).expanduser().resolve() if rr_raw else (Path(self.workspace_path) / "runs")
        rr.mkdir(parents=True, exist_ok=True)
        self.runs_root = str(rr)

    @property
    def workspace_dir(self) -> Path:
        return Path(self.workspace_path)

    @property
    def runs_dir(self) -> Path:
        return Path(self.runs_root)

    def set_workspace(self, path: str) -> None:
        ok, msg, resolved = validate_workspace_path(path)
        if not ok:
            raise ValueError(msg)

        self.workspace_path = resolved

        rr = Path(resolved) / "runs"
        rr.mkdir(parents=True, exist_ok=True)
        self.runs_root = str(rr)

    def set_case_input(self, case_path: str) -> str:
        raw = str(case_path or "").strip()
        if not raw:
            raise ValueError("case path is empty")
        if raw == "?":
            raise ValueError("'?' is help marker, not a case path")
        p = Path(case_path).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"path does not exist: {p}")
        if not p.is_dir():
            raise ValueError(f"path is not a directory: {p}")
        self.case_inputs["case_input_path"] = str(p)
        self.path_keys["case.input"] = str(p)

        if not self.case_id:
            self.case_id = p.name
        return str(p)

    def set_case_id(self, case_id: str) -> None:
        cid = str(case_id or "").strip()
        if cid:
            self.case_id = cid

    def set_path_key(self, key: str, value: str) -> None:
        k = str(key or "").strip()
        v = str(value or "").strip()
        if not k or not v:
            return

        def _looks_like_path(s: str) -> bool:
            low = s.lower()
            if s.startswith(("/", ".", "~")):
                return True
            if "\\" in s or "/" in s:
                return True
            if len(s) >= 2 and s[1] == ":":
                return True
            return low.endswith(
                (
                    ".nii",
                    ".nii.gz",
                    ".dcm",
                    ".json",
                    ".txt",
                    ".md",
                    ".png",
                    ".jpg",
                    ".jpeg",
                )
            )

        vp = v
        try:
            if _looks_like_path(v):
                vp = str(Path(v).expanduser().resolve())
            else:
                case_root = str(self.case_inputs.get("case_input_path") or "").strip()
                if case_root:
                    cand = Path(case_root) / v
                    if cand.exists():
                        vp = str(cand.resolve())
        except Exception:
            vp = v
        self.path_keys[k] = vp

    def summary_lines(self) -> list[str]:
        lines = [
            f"workspace={self.workspace_path}",
            f"runs_root={self.runs_root}",
            f"case_id={self.case_id or '-'}",
            f"case_input={self.case_inputs.get('case_input_path', '-')}",
            (
                "model="
                f"provider:{self.model_config.provider} "
                f"llm:{self.model_config.llm} "
                f"vlm:{self.model_config.vlm}"
            ),
            f"dry_run={self.dry_run}",
        ]
        return lines

    def model_dump(self) -> Dict[str, Any]:
        out = asdict(self)
        out["model_config"] = self.model_config.model_dump()
        return out

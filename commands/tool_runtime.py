from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

from .schemas import ToolContext


PROJECT_ROOT_ENV = "BCER_PROJECT_ROOT"
ASSETS_ROOT_ENV = "BCER_ASSETS_ROOT"
RUNTIME_CONFIG_ENV = "BCER_TOOL_RUNTIME_CONFIG"
DISPATCH_MODE_ENV = "BCER_TOOL_DISPATCH"


@dataclass(frozen=True)
class RuntimeTier:
    name: str
    conda_env: str
    python: Optional[str] = None
    use_conda_run: bool = True
    timeout_seconds: int = 1800
    env: Dict[str, str] | None = None


@dataclass(frozen=True)
class ToolRuntimeConfig:
    default_dispatch: str
    tiers: Dict[str, RuntimeTier]
    tool_tiers: Dict[str, str]
    project_root: Path
    assets_root: Path

    def tier_for_tool(self, tool_name: str) -> Optional[RuntimeTier]:
        tier_name = self.tool_tiers.get(tool_name)
        if not tier_name:
            return None
        return self.tiers.get(tier_name)


DEFAULT_TIER_TOOLS: Dict[str, List[str]] = {
    "base": [
        "identify_sequences",
        "ingest_dicom_to_nifti",
        "register_to_reference",
        "alignment_qc",
        "materialize_registration",
        "denoise_image_bm3d",
        "resample_image",
        "compare_nifti_slices",
        "generate_qa_snapshot",
        "package_vlm_evidence",
        "generate_report",
        "rag_search",
        "sandbox_exec",
        "classify_brain_glioma_grade",
        "classify_cardiac_cine_disease",
    ],
    "inference": [
        "segment_prostate",
        "brats_mri_segmentation",
        "segment_cardiac_cine",
        "detect_lesion_candidates",
        "correct_prostate_distortion",
    ],
    "recon": [
        "reconstruct_grappa",
    ],
    "radiomics": [
        "extract_roi_features",
    ],
}


def project_root() -> Path:
    raw = os.getenv(PROJECT_ROOT_ENV)
    if raw:
        return Path(raw).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "skills.json").exists() and (parent / "commands").is_dir():
            return parent
    return here.parents[1]


def _flatten_tool_tiers(raw: Mapping[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tier_name, tools in raw.items():
        if not isinstance(tools, Iterable) or isinstance(tools, (str, bytes)):
            continue
        for tool_name in tools:
            if str(tool_name).strip():
                out[str(tool_name).strip()] = str(tier_name)
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}


def load_tool_runtime_config(config_path: str | Path | None = None) -> ToolRuntimeConfig:
    root = project_root()
    cfg_path = Path(config_path or os.getenv(RUNTIME_CONFIG_ENV) or root / "configs" / "tool_runtime.yml")
    cfg = _load_yaml(cfg_path.expanduser().resolve())

    default_dispatch = str(
        os.getenv(DISPATCH_MODE_ENV) or cfg.get("default_dispatch") or "inprocess"
    ).strip().lower()

    assets_root = Path(
        os.getenv(ASSETS_ROOT_ENV) or cfg.get("assets_root") or root / "assets"
    ).expanduser().resolve()

    tier_defs = cfg.get("tiers") if isinstance(cfg.get("tiers"), dict) else {}
    tiers: Dict[str, RuntimeTier] = {}
    for name in ("base", "inference", "recon", "radiomics"):
        raw = tier_defs.get(name, {}) if isinstance(tier_defs, dict) else {}
        if not isinstance(raw, dict):
            raw = {}
        env_name = str(raw.get("conda_env") or f"bcer-{name}").strip()
        tiers[name] = RuntimeTier(
            name=name,
            conda_env=env_name,
            python=(str(raw["python"]).strip() if raw.get("python") else None),
            use_conda_run=bool(raw.get("use_conda_run", True)),
            timeout_seconds=int(raw.get("timeout_seconds") or 1800),
            env={str(k): str(v) for k, v in (raw.get("env") or {}).items()} if isinstance(raw.get("env"), dict) else {},
        )

    tool_tiers = _flatten_tool_tiers(DEFAULT_TIER_TOOLS)
    cfg_tool_tiers = cfg.get("tool_tiers")
    if isinstance(cfg_tool_tiers, dict):
        tool_tiers.update(_flatten_tool_tiers(cfg_tool_tiers))

    return ToolRuntimeConfig(
        default_dispatch=default_dispatch,
        tiers=tiers,
        tool_tiers=tool_tiers,
        project_root=root,
        assets_root=assets_root,
    )


def should_dispatch_subprocess(tool_name: str, cfg: ToolRuntimeConfig) -> bool:
    mode = cfg.default_dispatch
    if mode in {"0", "false", "off", "no", "inprocess"}:
        return False
    if mode in {"1", "true", "on", "yes", "subprocess"}:
        return cfg.tier_for_tool(tool_name) is not None
    if mode == "auto":
        tier = cfg.tool_tiers.get(tool_name)
        return tier in {"inference", "recon", "radiomics"}
    return False


def _context_to_payload(ctx: ToolContext) -> Dict[str, str]:
    return {
        "case_id": ctx.case_id,
        "run_id": ctx.run_id,
        "run_dir": str(ctx.run_dir),
        "artifacts_dir": str(ctx.artifacts_dir),
        "case_state_path": str(ctx.case_state_path),
    }


def _command_for_tier(tier: RuntimeTier) -> List[str]:
    if tier.python:
        return [tier.python]
    if tier.use_conda_run:
        return ["conda", "run", "--no-capture-output", "-n", tier.conda_env, "python"]
    return [sys.executable]


def _child_environment(cfg: ToolRuntimeConfig, tier: RuntimeTier) -> Dict[str, str]:
    env = dict(os.environ)
    env[PROJECT_ROOT_ENV] = str(cfg.project_root)
    env[ASSETS_ROOT_ENV] = str(cfg.assets_root)
    env.setdefault("PYTHONPATH", str(cfg.project_root))
    if str(cfg.project_root) not in env["PYTHONPATH"].split(os.pathsep):
        env["PYTHONPATH"] = str(cfg.project_root) + os.pathsep + env["PYTHONPATH"]

    # Project-local defaults for model/checkpoint assets. Users may override, but
    # public release docs should point them back under assets/.
    env.setdefault("MRI_AGENT_MODEL_REGISTRY", str(cfg.assets_root / "models"))
    env.setdefault("MRI_AGENT_PROSTATE_BUNDLE_DIR", str(cfg.assets_root / "models" / "prostate_mri_anatomy"))
    env.setdefault("MRI_AGENT_BRATS_BUNDLE_DIR", str(cfg.assets_root / "models" / "brats_mri_segmentation"))
    env.setdefault("MRI_AGENT_LESION_WEIGHTS_DIR", str(cfg.assets_root / "models" / "prostate_mri_lesion_seg" / "weight"))
    env.setdefault("MRI_AGENT_PROSTATE_DISTORTION_ROOT", str(cfg.assets_root / "external" / "Prostate_distortion_recover"))
    env.setdefault(
        "MRI_AGENT_PROSTATE_DISTORTION_DIFF_CKPT",
        str(cfg.assets_root / "checkpoints" / "prostate_distortion" / "diff_t2cnn_clean_epoch_092.pt"),
    )
    env.setdefault(
        "MRI_AGENT_PROSTATE_DISTORTION_CNN_CKPT",
        str(cfg.assets_root / "checkpoints" / "prostate_distortion" / "mageultra_epoch_025.pt"),
    )
    env.setdefault("MRI_AGENT_CARDIAC_CMR_REVERSE_ROOT", str(cfg.assets_root / "external" / "cmr_reverse"))
    env.setdefault("MRI_AGENT_CARDIAC_NNUNET_PYTHON", str(cfg.assets_root / "external" / "cmr_reverse" / "revimg" / "bin" / "python"))
    env.setdefault("MRI_AGENT_CARDIAC_RESULTS_FOLDER", str(cfg.assets_root / "models" / "cardiac_nnunet" / "results"))

    for key, value in (tier.env or {}).items():
        env[str(key)] = str(value)
    return env


def run_tool_subprocess(
    *,
    tool_name: str,
    args: Dict[str, Any],
    ctx: ToolContext,
    cfg: Optional[ToolRuntimeConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or load_tool_runtime_config()
    tier = cfg.tier_for_tool(tool_name)
    if tier is None:
        raise RuntimeError(f"No runtime tier configured for tool '{tool_name}'.")

    dispatch_dir = ctx.run_dir / "_tool_dispatch" / tool_name / f"{int(time.time() * 1000)}_{os.getpid()}"
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    args_file = dispatch_dir / "args.json"
    ctx_file = dispatch_dir / "context.json"
    result_file = dispatch_dir / "result.json"
    args_file.write_text(json.dumps(args, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ctx_file.write_text(json.dumps(_context_to_payload(ctx), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cmd = _command_for_tier(tier) + [
        "-m",
        "tools.run_tool",
        "--tool-name",
        tool_name,
        "--args-file",
        str(args_file),
        "--context-file",
        str(ctx_file),
        "--result-file",
        str(result_file),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(cfg.project_root),
        env=_child_environment(cfg, tier),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=tier.timeout_seconds,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Subprocess tool dispatch failed "
            f"(tool={tool_name}, tier={tier.name}, env={tier.conda_env}, rc={proc.returncode}). "
            f"stdout={proc.stdout[-2000:]!r} stderr={proc.stderr[-4000:]!r}"
        )
    if not result_file.exists():
        raise RuntimeError(f"Subprocess tool '{tool_name}' did not write result file: {result_file}")
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        provenance = payload.setdefault("provenance", {})
        provenance["dispatch"] = {
            "mode": "subprocess",
            "tier": tier.name,
            "conda_env": tier.conda_env,
            "command": cmd,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    return payload

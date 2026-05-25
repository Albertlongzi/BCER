from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_manifest_builder_module():
    repo_root = Path(__file__).resolve().parents[2]
    mod_path = repo_root / "scripts" / "manifest_builder.py"
    spec = importlib.util.spec_from_file_location("manifest_builder_v2_test", str(mod_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_builder_splits_h5_cases(tmp_path: Path) -> None:
    mb = _load_manifest_builder_module()

    case_root = tmp_path / "cardiac_h5"
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "patient001.h5").write_bytes(b"x")
    (case_root / "patient002.h5").write_bytes(b"y")

    tasks_registry = tmp_path / "tasks_registry.json"
    tasks_registry.write_text(
        json.dumps(
            {
                "tasks": {
                    "short_recon_grappa": {
                        "domain": ["cardiac"],
                        "required_modalities": {"cardiac": {"any_of": ["h5", "raw_kspace"]}},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    entries = mb.build_manifest(
        root_specs=[
            mb.RootSpec(domain="cardiac", root=case_root, bucket="cardiac", split_h5_files=True, source_tag="cardiac")
        ],
        tasks_registry_path=tasks_registry,
        max_depth=3,
        max_files_per_case=128,
        max_cases=0,
        bucket_limits={},
    )
    assert len(entries) == 2
    assert all(str(e["case_root"]).endswith(".h5") for e in entries)
    assert all(e["input_format"] == "raw_kspace" for e in entries)
    assert all(bool((e.get("modalities") or {}).get("h5")) for e in entries)
    assert all("short_recon_grappa" in (e.get("supports_tasks") or []) for e in entries)


def test_manifest_builder_bucket_limit(tmp_path: Path) -> None:
    mb = _load_manifest_builder_module()

    case_root = tmp_path / "cardiac_h5"
    case_root.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        (case_root / f"patient{i:03d}.h5").write_bytes(b"x")

    tasks_registry = tmp_path / "tasks_registry.json"
    tasks_registry.write_text(json.dumps({"tasks": {}}), encoding="utf-8")

    entries = mb.build_manifest(
        root_specs=[
            mb.RootSpec(domain="cardiac", root=case_root, bucket="cardiac", split_h5_files=True, source_tag="cardiac")
        ],
        tasks_registry_path=tasks_registry,
        max_depth=3,
        max_files_per_case=128,
        max_cases=0,
        bucket_limits={"cardiac": 1},
    )
    assert len(entries) == 1

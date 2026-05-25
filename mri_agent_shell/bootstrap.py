from __future__ import annotations

import sys
from pathlib import Path


def ensure_import_paths() -> None:
    """
    Make both of these imports work from different launch locations:
    - `python -m mri_agent_shell` (from repo root)
    - entrypoint script after `pip install -e .`

    Framework modules use absolute package imports relative to the repository
    root, so we add the repo root and its parent to sys.path.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    repo_parent = repo_root.parent

    for p in (repo_parent, repo_root):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)

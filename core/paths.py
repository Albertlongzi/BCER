from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """
    Return the BCER project root.

    Walks up from this file until we find ``config/skills.json``. Falls back to
    the package parent if nothing is found.
    """
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "config" / "skills.json").exists():
            return parent
    return here.parents[1]

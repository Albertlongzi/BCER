from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_EXTS = {".py", ".md", ".json", ".txt", ".yaml", ".yml"}


def _iter_files(root: Path, exts: Optional[Iterable[str]] = None) -> Iterable[Path]:
    allow = {e.lower() for e in (exts or DEFAULT_EXTS)}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in allow:
            yield p


def search_repo(query: str, root: Optional[Path] = None, *, max_results: int = 20, exts: Optional[Iterable[str]] = None) -> List[Dict[str, str]]:
    """
    Lightweight local-search helper (RAG prototype).

    Returns a list of {path, line, line_no} matches containing the query string.
    This is not indexed and is intended as a minimal baseline for future RAG integration.
    """
    if not query:
        return []
    root = root or Path(__file__).resolve().parents[2]
    q = query.lower()
    hits: List[Dict[str, str]] = []
    for fp in _iter_files(root, exts=exts):
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if q in line.lower():
                hits.append({"path": str(fp), "line_no": str(i), "line": line.strip()})
                if len(hits) >= max_results:
                    return hits
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description="Lightweight local repo search (RAG prototype).")
    ap.add_argument("--query", required=True)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--max-results", type=int, default=20)
    args = ap.parse_args()

    hits = search_repo(args.query, root=Path(args.root), max_results=int(args.max_results))
    for h in hits:
        print(f"{h['path']}:{h['line_no']}: {h['line']}")


if __name__ == "__main__":
    main()

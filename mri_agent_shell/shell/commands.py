from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ParsedCommand:
    kind: str
    args: List[str] = field(default_factory=list)
    kv: Dict[str, str] = field(default_factory=dict)
    tail: str = ""
    raw: str = ""


def parse_command(line: str) -> Optional[ParsedCommand]:
    s = str(line or "").strip()
    if not s.startswith(":"):
        return None

    body = s[1:].strip()
    if not body:
        return ParsedCommand(kind="help", raw=s)

    try:
        toks = shlex.split(body)
    except Exception:
        toks = body.split()

    if not toks:
        return ParsedCommand(kind="help", raw=s)

    head = toks[0].lower()

    if head in {"help", "h", "?"}:
        return ParsedCommand(kind="help", raw=s)

    if head in {"exit", "quit", "q"}:
        return ParsedCommand(kind="exit", raw=s)

    if head == "run":
        return ParsedCommand(kind="run", raw=s)

    if head == "doctor":
        return ParsedCommand(kind="doctor", raw=s)

    if head == "domains":
        return ParsedCommand(kind="domains", raw=s)

    if head == "tools":
        return ParsedCommand(kind="tools", raw=s)

    if head == "paste":
        return ParsedCommand(kind="paste", raw=s)

    if head == "model":
        if len(toks) == 1:
            return ParsedCommand(kind="model_show", raw=s)
        if len(toks) >= 2 and toks[1].lower() == "set":
            kv: Dict[str, str] = {}
            for tok in toks[2:]:
                if "=" not in tok:
                    continue
                k, v = tok.split("=", 1)
                kv[k.strip()] = v.strip()
            return ParsedCommand(kind="model_set", kv=kv, raw=s)
        return ParsedCommand(kind="invalid", args=toks, raw=s)

    if head == "workspace" and len(toks) >= 3 and toks[1].lower() == "set":
        return ParsedCommand(kind="workspace_set", args=[" ".join(toks[2:])], raw=s)

    if head == "case" and len(toks) >= 2 and toks[1].lower() == "load":
        if len(toks) < 3:
            return ParsedCommand(kind="case_load", args=[], raw=s)
        path = str(toks[2]).strip()
        tail = " ".join(toks[3:]).strip() if len(toks) > 3 else ""
        return ParsedCommand(kind="case_load", args=[path], tail=tail, raw=s)

    return ParsedCommand(kind="invalid", args=toks, raw=s)

#!/usr/bin/env python3
"""Aegis plugin entry point.

  aegis.py hook <tool> <HookEvent>   called by the assistant; reads the event JSON on stdin
  aegis.py connect <code> [--tool T] connect from a terminal (default tool: claude-code)
  aegis.py status [--tool T]
  aegis.py disconnect [--tool T]

Inside the assistant, people send the message "aegis connect <code>" (or status /
disconnect); the prompt hook answers it without involving the model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aegis_plugin.adapters import ADAPTERS  # noqa: E402
from aegis_plugin.engine import Engine  # noqa: E402


def run_hook(tool: str, event_name: str) -> int:
    adapter = ADAPTERS.get(tool)
    if adapter is None:
        print(f"[Aegis] unknown assistant {tool!r}", file=sys.stderr)
        return 1
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        data = {}
    event = adapter.parse(event_name, data if isinstance(data, dict) else {})
    if event is None:
        return 0
    output = Engine(adapter).handle(event)
    if output.payload:
        sys.stdout.write(json.dumps(output.payload))
    return output.exit_code


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[0] == "hook":
        try:
            return run_hook(argv[1], argv[2])
        except Exception as exc:  # noqa: BLE001 - a broken hook must explain itself, not crash
            print(f"[Aegis] hook error: {exc}", file=sys.stderr)
            return 1
    tool = "claude-code"
    if "--tool" in argv:
        i = argv.index("--tool")
        tool = argv[i + 1] if i + 1 < len(argv) else tool
        argv = argv[:i] + argv[i + 2 :]
    if not argv or argv[0] not in ("connect", "status", "disconnect") or tool not in ADAPTERS:
        print(__doc__)
        return 1
    engine = Engine(ADAPTERS[tool])
    if argv[0] == "connect":
        message = engine.connect(argv[1] if len(argv) > 1 else "")
    else:
        message = engine.status() if argv[0] == "status" else engine.disconnect()
    print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

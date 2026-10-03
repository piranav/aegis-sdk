#!/usr/bin/env python3
"""Aegis plugin for Claude Code: connect, liveness, and tool-call governance.

Standard library only, so it runs wherever Claude Code's hooks run python3.

Commands:
  connect <code>     exchange an IT-issued code for this machine's subscription key
  status             show the connection
  disconnect         tell Aegis and forget the key
  hook <event>       Claude Code hook entry point (reads the event JSON on stdin)
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
import urllib.error
import urllib.request
from base64 import urlsafe_b64decode
from pathlib import Path

PLUGIN_VERSION = "0.1.0"
TOOL = "claude-code"
STATE_DIR = Path(os.environ.get("AEGIS_STATE_DIR", Path.home() / ".aegis" / TOOL))
CREDENTIALS = STATE_DIR / "credentials.json"
PENDING = STATE_DIR / "pending"
TIMEOUT = float(os.environ.get("AEGIS_TIMEOUT_SECONDS", "8"))


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


# ---------------------------------------------------------------- state


def load_credentials() -> dict | None:
    try:
        return json.loads(CREDENTIALS.read_text())
    except (OSError, ValueError):
        return None


def save_credentials(data: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = CREDENTIALS.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle)
    os.replace(tmp, CREDENTIALS)


def forget_credentials() -> None:
    for path in (CREDENTIALS,):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def remember_audit(session_id: str, tool_use_id: str, audit_id: str) -> None:
    folder = PENDING / _safe(session_id)
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    (folder / _safe(tool_use_id)).write_text(audit_id)


def recall_audit(session_id: str, tool_use_id: str) -> str | None:
    path = PENDING / _safe(session_id) / _safe(tool_use_id)
    try:
        audit_id = path.read_text().strip()
        path.unlink()
        return audit_id or None
    except OSError:
        return None


def clear_session(session_id: str) -> None:
    folder = PENDING / _safe(session_id)
    if folder.is_dir():
        for child in folder.iterdir():
            child.unlink(missing_ok=True)
        folder.rmdir()


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (value or "unknown"))[:120]


# ---------------------------------------------------------------- client info


def client_info() -> dict:
    """Descriptive details IT sees next to the subscription; never used for auth."""
    info = {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "os_user": os.environ.get("USER") or os.environ.get("USERNAME") or "",
        "plugin_version": PLUGIN_VERSION,
    }
    version = os.environ.get("CLAUDE_CODE_VERSION") or os.environ.get("CLAUDE_VERSION")
    if version:
        info["tool_version"] = version
    try:
        account = json.loads((Path.home() / ".claude.json").read_text()).get("oauthAccount") or {}
        info["account_email"] = account.get("emailAddress", "")
        info["account_organization"] = account.get("organizationName", "")
        info["account_organization_id"] = account.get("organizationUuid", "")
    except (OSError, ValueError, AttributeError):
        pass
    return {k: v for k, v in info.items() if v}


# ---------------------------------------------------------------- http


def request(api_url: str, path: str, body: dict, key: str | None = None) -> dict:
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json", "User-Agent": f"aegis-claude/{PLUGIN_VERSION}"}
    if key:
        headers["X-API-Key"] = key
    req = urllib.request.Request(api_url.rstrip("/") + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail", exc.reason)
        except ValueError:
            detail = exc.reason
        raise ApiError(exc.code, str(detail)) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ApiError(0, f"Aegis is unreachable: {exc}") from None


def decode_code(code: str) -> tuple[str, str]:
    if not code.startswith("aegc_"):
        raise ValueError("That isn't an Aegis connection code (it should start with aegc_).")
    raw = code[5:]
    decoded = urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode()
    api_url, _, token = decoded.rpartition("|")
    if not api_url or not token:
        raise ValueError("The connection code is incomplete; copy it again from your invite.")
    return api_url, token


# ---------------------------------------------------------------- commands


def cmd_connect(args: list[str]) -> int:
    if not args:
        print("Usage: /aegis:connect <code from your IT invite>")
        return 1
    code = args[0].strip()
    try:
        api_url, _ = decode_code(code)
        result = request(
            api_url,
            "/v1/subscriptions/connect",
            {"code": code, "tool": TOOL, "client": client_info()},
        )
    except ValueError as exc:
        print(f"Aegis: {exc}")
        return 1
    except ApiError as exc:
        print(f"Aegis could not connect: {exc.detail}")
        return 1
    save_credentials(
        {
            "api_url": api_url,
            "api_key": result["api_key"],
            "subscription_id": result["subscription_id"],
            "organization": result.get("organization"),
            "email": result.get("email"),
            "failure_mode": result.get("failure_mode", "open"),
            "connected_at": int(time.time()),
        }
    )
    print(
        f"Aegis connected: {result.get('email')} at {result.get('organization')}.\n"
        "Your IT team can now see this Claude Code as active, and your organization's "
        "rules apply from the next tool call."
    )
    return 0


def cmd_status(_: list[str]) -> int:
    creds = load_credentials()
    if not creds:
        print("Aegis is not connected. Run /aegis:connect <code> with the code from IT.")
        return 0
    try:
        result = request(
            creds["api_url"],
            "/v1/subscriptions/heartbeat",
            {"event": "heartbeat", "client": client_info()},
            creds["api_key"],
        )
        _cache_failure_mode(creds, result)
        state = result.get("status", "unknown")
    except ApiError as exc:
        state = f"error ({exc.detail})"
    offline = "block tool calls" if creds.get("failure_mode") == "closed" else "allow and warn"
    print(
        f"Aegis: {creds.get('email')} at {creds.get('organization')}\n"
        f"Status: {state}\nServer: {creds['api_url']}\n"
        f"Offline behavior: {offline}"
    )
    return 0


def cmd_disconnect(_: list[str]) -> int:
    creds = load_credentials()
    if not creds:
        print("Aegis is not connected.")
        return 0
    try:
        request(
            creds["api_url"],
            "/v1/subscriptions/heartbeat",
            {"event": "disconnect", "client": client_info()},
            creds["api_key"],
        )
    except ApiError as exc:
        print(f"Aegis couldn't be told about the disconnect ({exc.detail}); forgetting the key.")
    forget_credentials()
    print("Aegis disconnected. Your IT team will see this Claude Code as disconnected.")
    return 0


def _cache_failure_mode(creds: dict, result: dict) -> None:
    mode = result.get("failure_mode")
    if mode in ("open", "closed") and mode != creds.get("failure_mode"):
        creds["failure_mode"] = mode
        save_credentials(creds)


# ---------------------------------------------------------------- hooks


def emit(payload: dict) -> int:
    if payload:
        sys.stdout.write(json.dumps(payload))
    return 0


def pre_tool_output(decision: str, reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }


def _reason(result: dict) -> str:
    violations = "; ".join(result.get("violations") or []) or "Blocked by an organization rule"
    support = result.get("support_message") or ""
    contact = result.get("support_email")
    text = f"[Aegis] {violations}"
    if support:
        text += f". {support}"
    if contact:
        text += f" ({contact})"
    return text


def _unavailable(creds: dict, event: str, exc: ApiError) -> int:
    """Aegis couldn't decide: auth failures always block, outages follow IT's choice."""
    if exc.status in (401, 403):
        message = (
            "[Aegis] This Claude Code's Aegis connection is no longer valid "
            f"({exc.detail}). Ask IT for a new connection code, then run /aegis:connect."
        )
        blocking = True
    else:
        blocking = creds.get("failure_mode") == "closed"
        message = f"[Aegis] {exc.detail}. " + (
            "Your organization blocks tool calls until Aegis is reachable."
            if blocking
            else "Continuing without a governance check."
        )
    if event == "PreToolUse":
        if blocking:
            return emit(pre_tool_output("deny", message))
        return emit({"systemMessage": message})
    if event == "UserPromptSubmit" and blocking:
        return emit({"decision": "block", "reason": message})
    return emit({"systemMessage": message})


def hook(event: str) -> int:
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except ValueError:
        data = {}
    creds = load_credentials()
    session_id = data.get("session_id") or ""

    if event == "SessionStart":
        if not creds:
            return emit(
                {
                    "systemMessage": (
                        "[Aegis] This Claude Code isn't connected to your organization yet. "
                        "Run /aegis:connect <code> with the code from your IT invite."
                    )
                }
            )
        try:
            result = request(
                creds["api_url"],
                "/v1/subscriptions/heartbeat",
                {"event": "session_start", "client": client_info()},
                creds["api_key"],
            )
            _cache_failure_mode(creds, result)
        except ApiError as exc:
            return _unavailable(creds, event, exc)
        return 0

    if not creds:
        return 0

    if event == "SessionEnd":
        clear_session(session_id)
        try:
            request(
                creds["api_url"],
                "/v1/subscriptions/heartbeat",
                {"event": "session_end", "client": {}},
                creds["api_key"],
            )
        except ApiError:
            pass
        return 0

    if event == "UserPromptSubmit":
        body = {
            "agent_name": TOOL,
            "tool_name": "UserPromptSubmit",
            "tool_args": {},
            "session_id": session_id,
            "prompt": (data.get("prompt") or "")[:32000],
        }
        try:
            result = request(creds["api_url"], "/v1/evaluate", body, creds["api_key"])
        except ApiError as exc:
            return _unavailable(creds, event, exc)
        if result.get("decision") == "allow":
            return 0
        return emit({"decision": "block", "reason": _reason(result)})

    if event == "PreToolUse":
        body = {
            "agent_name": TOOL,
            "tool_name": data.get("tool_name") or "unknown",
            "tool_args": data.get("tool_input") or {},
            "session_id": session_id,
        }
        try:
            result = request(creds["api_url"], "/v1/evaluate", body, creds["api_key"])
        except ApiError as exc:
            return _unavailable(creds, event, exc)
        if result.get("audit_id") and data.get("tool_use_id"):
            remember_audit(session_id, data["tool_use_id"], result["audit_id"])
        verdict = result.get("decision")
        if verdict == "allow":
            return 0
        # Escalations need a person: Claude Code asks the user before running the tool.
        return emit(pre_tool_output("deny" if verdict == "deny" else "ask", _reason(result)))

    if event == "PostToolUse":
        audit_id = recall_audit(session_id, data.get("tool_use_id") or "")
        if not audit_id:
            return 0
        output = data.get("tool_response", data.get("tool_output"))
        result_text = output if isinstance(output, str) else json.dumps(output, default=str)
        try:
            result = request(
                creds["api_url"],
                "/v1/evaluate_result",
                {"audit_id": audit_id, "result": result_text[:100000], "result_metadata": {}},
                creds["api_key"],
            )
        except ApiError as exc:
            return _unavailable(creds, event, exc)
        if result.get("decision") == "allow":
            return 0
        return emit({"decision": "block", "reason": _reason(result)})

    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    command, rest = argv[0], argv[1:]
    if command == "hook" and rest:
        try:
            return hook(rest[0])
        except Exception as exc:  # noqa: BLE001 - a broken hook must explain itself, not crash
            print(f"[Aegis] hook error: {exc}", file=sys.stderr)
            return 1
    handlers = {"connect": cmd_connect, "status": cmd_status, "disconnect": cmd_disconnect}
    if command not in handlers:
        print(__doc__)
        return 1
    return handlers[command](rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

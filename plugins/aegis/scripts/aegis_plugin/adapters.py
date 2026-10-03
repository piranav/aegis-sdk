"""Per-assistant adapters: hook payloads in, decision formats out.

Adding an assistant means subclassing ``Adapter``: parse its hook payloads into an
``Event``, describe its tools in canonical terms, and render decisions in its output
format. The engine, connection, and API handling are shared.
"""

from __future__ import annotations

import json
import re
from base64 import urlsafe_b64decode
from dataclasses import dataclass, field
from pathlib import Path

SESSION_START = "session_start"
SESSION_END = "session_end"
PROMPT = "prompt"
PRE_TOOL = "pre_tool"
POST_TOOL = "post_tool"


@dataclass
class Event:
    kind: str
    session_id: str = ""
    prompt: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_use_id: str = ""
    tool_response: object = None


@dataclass
class Output:
    """What the hook process writes to stdout, and its exit code."""

    payload: dict | None = None
    exit_code: int = 0


class Adapter:
    tool = ""
    label = ""
    # Hook event name -> Event.kind
    events: dict[str, str] = {}
    # Vendor tool name -> canonical action, for rules that should hold across assistants.
    actions: dict[str, str] = {}

    def parse(self, event_name: str, data: dict) -> Event | None:
        kind = self.events.get(event_name)
        if kind is None:
            return None
        return Event(
            kind=kind,
            session_id=str(data.get("session_id") or ""),
            prompt=str(data.get("prompt") or ""),
            tool_name=str(data.get("tool_name") or ""),
            tool_input=data.get("tool_input") if isinstance(data.get("tool_input"), dict) else {},
            tool_use_id=str(data.get("tool_use_id") or ""),
            tool_response=data.get("tool_response", data.get("tool_output")),
        )

    def action(self, event: Event) -> str:
        if event.tool_name.startswith("mcp__"):
            return "mcp.call"
        return self.actions.get(event.tool_name, "tool.other")

    def paths(self, event: Event) -> list[str]:
        keys = ("file_path", "path", "notebook_path")
        return [str(event.tool_input[k]) for k in keys if event.tool_input.get(k)]

    def account(self) -> dict:
        """The assistant's signed-in account, as reported by this machine."""
        return {}

    # Rendering. Defaults follow the Claude Code hook contract, which Codex mirrors.

    def allow(self, event: Event) -> Output:
        return Output()

    def deny(self, event: Event, reason: str) -> Output:
        if event.kind == PRE_TOOL:
            return Output(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                }
            )
        if event.kind in (PROMPT, POST_TOOL):
            return Output({"decision": "block", "reason": reason})
        return self.notify(event, reason)

    def ask(self, event: Event, reason: str) -> Output:
        if event.kind == PRE_TOOL:
            return Output(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "ask",
                        "permissionDecisionReason": reason,
                    }
                }
            )
        return self.deny(event, reason)

    def notify(self, event: Event, message: str) -> Output:
        return Output({"systemMessage": message})

    def reply(self, event: Event, message: str) -> Output:
        """Answer an Aegis command typed as a prompt without sending it to the model."""
        return Output({"decision": "block", "reason": message})


class ClaudeCode(Adapter):
    tool = "claude-code"
    label = "Claude Code"
    events = {
        "SessionStart": SESSION_START,
        "SessionEnd": SESSION_END,
        "UserPromptSubmit": PROMPT,
        "PreToolUse": PRE_TOOL,
        "PostToolUse": POST_TOOL,
    }
    actions = {
        "Read": "file.read",
        "Grep": "file.read",
        "Glob": "file.read",
        "Edit": "file.write",
        "MultiEdit": "file.write",
        "Write": "file.write",
        "NotebookEdit": "file.write",
        "Bash": "shell.exec",
        "WebFetch": "web.fetch",
        "WebSearch": "web.fetch",
    }

    def account(self) -> dict:
        try:
            config = json.loads((Path.home() / ".claude.json").read_text())
            account = config.get("oauthAccount") or {}
        except (OSError, ValueError, AttributeError):
            return {}
        return {
            "account_email": account.get("emailAddress", ""),
            "account_organization": account.get("organizationName", ""),
            "account_organization_id": account.get("organizationUuid", ""),
        }


PATCH_FILE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)


class Codex(Adapter):
    tool = "codex"
    label = "Codex"
    events = {
        "SessionStart": SESSION_START,
        "SessionEnd": SESSION_END,
        "UserPromptSubmit": PROMPT,
        "PreToolUse": PRE_TOOL,
        "PostToolUse": POST_TOOL,
    }
    actions = {
        "Bash": "shell.exec",
        "shell": "shell.exec",
        "apply_patch": "file.write",
        "Edit": "file.write",
        "Write": "file.write",
    }

    def paths(self, event: Event) -> list[str]:
        if self.action(event) == "file.write":
            return PATCH_FILE.findall(str(event.tool_input.get("command") or ""))
        return super().paths(event)

    def account(self) -> dict:
        """Email and ChatGPT plan from the ID token; no token ever leaves the machine."""
        try:
            auth = json.loads((Path.home() / ".codex" / "auth.json").read_text())
            token = (auth.get("tokens") or {}).get("id_token") or ""
            payload = token.split(".")[1]
            claims = json.loads(urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        except (OSError, ValueError, IndexError, AttributeError):
            return {}
        plan = (claims.get("https://api.openai.com/auth") or {}).get("chatgpt_plan_type", "")
        return {
            "account_email": claims.get("email", ""),
            "account_organization": f"ChatGPT {plan}".strip() if plan else "",
        }

    def ask(self, event: Event, reason: str) -> Output:
        # Codex doesn't support "ask" yet and would let the call through; refuse instead.
        return self.deny(event, f"{reason} (needs approval; ask IT)")

    def reply(self, event: Event, message: str) -> Output:
        return Output({"decision": "block", "reason": message, "systemMessage": message})


ADAPTERS: dict[str, Adapter] = {a.tool: a for a in (ClaudeCode(), Codex())}

"""Per-assistant adapters: hook payloads in, decision formats out.

Adding an assistant means subclassing ``Adapter``: parse its hook payloads into an
``Event``, describe its tools in canonical terms, render decisions in its output
format, and name the ``Scanner`` that reads its configuration for the inventory. The
engine, connection, and API handling are shared.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from base64 import urlsafe_b64decode
from dataclasses import dataclass, field
from pathlib import Path

from aegis_plugin.inventory import ClaudeCodeScanner, CodexScanner, Scanner
from aegis_plugin.usage import ClaudeTranscript, CodexRollout

SESSION_START = "session_start"
SESSION_END = "session_end"
PROMPT = "prompt"
PRE_TOOL = "pre_tool"
POST_TOOL = "post_tool"
STOP = "stop"  # the assistant finished responding to a prompt


@dataclass
class Event:
    kind: str
    session_id: str = ""
    prompt: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_use_id: str = ""
    tool_response: object = None
    cwd: str = ""
    model: str = ""
    transcript_path: str = ""


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
    # Reads the assistant's configuration into an inventory manifest at session start.
    scanner: Scanner | None = None
    # Reads model calls and token usage from the assistant's session file.
    transcript: ClaudeTranscript | CodexRollout | None = None

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
            cwd=str(data.get("cwd") or ""),
            model=_model_name(data.get("model")),
            transcript_path=str(data.get("transcript_path") or ""),
        )

    def action(self, event: Event) -> str:
        if event.tool_name.startswith("mcp__"):
            return "mcp.call"
        return self.actions.get(event.tool_name, "tool.other")

    def paths(self, event: Event) -> list[str]:
        keys = ("file_path", "path", "notebook_path")
        found = [str(event.tool_input[k]) for k in keys if event.tool_input.get(k)]
        for command in self.commands(event):
            found.extend(command_paths(command))
        return list(dict.fromkeys(found))[:MAX_OBSERVED]

    def commands(self, event: Event) -> list[str]:
        """Shell commands the call runs, so shell and secret-file rules can see them."""
        if self.actions.get(event.tool_name) != "shell.exec":
            return []
        command = event.tool_input.get("command")
        if isinstance(command, list):  # older Codex: ["bash", "-lc", "cat .env"]
            argv = [str(part) for part in command]
            is_shell = len(argv) >= 3 and Path(argv[0]).name in SHELLS and "c" in argv[-2]
            command = argv[-1] if is_shell else shlex.join(argv)
        return [command] if isinstance(command, str) and command.strip() else []

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


MAX_OBSERVED = 16
SHELLS = {"bash", "sh", "zsh", "dash", "fish"}
# Words in a command that name a file: they contain a slash, start with a dot or tilde
# (.env, ~/.ssh/id_rsa), or end in an extension (server.pem). Flags, URLs, and plain
# words are left out.
PATH_WORD = re.compile(r"^(?:[~.]|.*/|[^\s/]+\.[A-Za-z0-9]{1,10}$)")


def command_paths(command: str) -> list[str]:
    try:
        words = shlex.split(command, comments=True)
    except ValueError:
        words = command.split()
    found = []
    for word in words:
        word = word.lstrip("<>&|;(").rstrip(";|&)")
        if word.startswith("-") or "://" in word or "=" in word or not PATH_WORD.match(word):
            continue
        if word not in (".", ".."):
            found.append(word)
    return found[:MAX_OBSERVED]


def _model_name(value: object) -> str:
    """Hook payloads carry the model as a name or as ``{"id": ..., ...}``."""
    if isinstance(value, dict):
        value = value.get("id") or value.get("name")
    return value if isinstance(value, str) else ""


class ClaudeCode(Adapter):
    tool = "claude-code"
    label = "Claude Code"
    scanner = ClaudeCodeScanner()
    transcript = ClaudeTranscript()
    events = {
        "SessionStart": SESSION_START,
        "SessionEnd": SESSION_END,
        "UserPromptSubmit": PROMPT,
        "PreToolUse": PRE_TOOL,
        "PostToolUse": POST_TOOL,
        "Stop": STOP,
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
            home = os.environ.get("CLAUDE_CONFIG_DIR") or Path.home()
            config = json.loads((Path(home) / ".claude.json").read_text())
            account = config.get("oauthAccount") or {}
        except (OSError, ValueError, AttributeError):
            return {}
        return {
            "account_email": account.get("emailAddress", ""),
            "account_organization": account.get("organizationName", ""),
            "account_organization_id": account.get("organizationUuid", ""),
        }


PATCH_FILE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)
# Codex code mode: ``exec`` runs JavaScript that calls the real tools, e.g.
# ``tools.exec_command({cmd: "cat .env"})``. The command strings are what it runs.
EXEC_COMMAND = re.compile(
    r"\b(?:cmd|command)\s*:\s*(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|`[^`]*`)"
)


def _js_string(literal: str) -> str:
    quote, body = literal[0], literal[1:-1]
    if quote == '"':
        try:
            return json.loads(literal)
        except ValueError:
            return body
    return body.replace("\\" + quote, quote) if quote == "'" else body


def _strings(value, depth=0):
    if depth > 4:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item, depth + 1)


class Codex(Adapter):
    tool = "codex"
    label = "Codex"
    scanner = CodexScanner()
    transcript = CodexRollout()
    events = {
        "SessionStart": SESSION_START,
        "SessionEnd": SESSION_END,
        "UserPromptSubmit": PROMPT,
        "PreToolUse": PRE_TOOL,
        "PostToolUse": POST_TOOL,
        "Stop": STOP,
    }
    actions = {
        "Bash": "shell.exec",
        "shell": "shell.exec",
        "apply_patch": "file.write",
        "Edit": "file.write",
        "Write": "file.write",
    }

    def action(self, event: Event) -> str:
        if event.tool_name == "exec" and self.commands(event):
            return "shell.exec"
        return super().action(event)

    def paths(self, event: Event) -> list[str]:
        if self.action(event) == "file.write":
            return PATCH_FILE.findall(str(event.tool_input.get("command") or ""))
        return super().paths(event)

    def commands(self, event: Event) -> list[str]:
        if event.tool_name != "exec":
            return super().commands(event)
        script = "\n".join(_strings(event.tool_input))
        found = [_js_string(m.group(1)) for m in EXEC_COMMAND.finditer(script)]
        return list(dict.fromkeys(c for c in found if c.strip()))[:MAX_OBSERVED]

    def account(self) -> dict:
        """Email and ChatGPT plan from the ID token; no token ever leaves the machine."""
        try:
            home = os.environ.get("CODEX_HOME") or Path.home() / ".codex"
            auth = json.loads((Path(home) / "auth.json").read_text())
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

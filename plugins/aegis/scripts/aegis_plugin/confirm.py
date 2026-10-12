"""Asking the person before a tool call an organization rule holds for review.

Claude Code shows its own permission prompt; the plugin tells Aegis when the person
allowed it. Where the assistant can't prompt (Codex, or Claude Code skipping permission
prompts), the agent is told to ask in the conversation and the person's next message is
relayed to Aegis as their answer. If the agent already asked in its own words, its
question goes with the answer, so the person isn't asked twice.
"""

from __future__ import annotations

import hashlib
import json

DECLINED = (
    "[Aegis] You said not to go ahead, so this was not run. Don't run it again unless "
    "you're asked to."
)


def call_key(tool_name: str, tool_input: dict) -> str:
    canonical = json.dumps([tool_name, tool_input], sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def last_agent_text(transcript_path: str) -> str | None:
    """The agent's last message before the person's latest prompt, read from the session
    file: Claude Code transcripts and Codex rollouts. Tool calls and results are skipped."""
    if not transcript_path:
        return None
    try:
        with open(transcript_path, encoding="utf-8") as handle:
            lines = handle.readlines()[-400:]
    except OSError:
        return None
    last_agent, before_prompt = None, None
    for raw in lines:
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        role, texts = _message(entry)
        if not texts:
            continue
        if role == "assistant":
            last_agent = "\n".join(texts)
        elif role == "user" and not texts[0].lstrip().startswith("<"):
            # Codex adds context blocks as user messages; they aren't the person.
            before_prompt = last_agent
    return before_prompt


def final_agent_text(transcript_path: str) -> str | None:
    """The agent's last message in the session file: its answer once a turn is done."""
    if not transcript_path:
        return None
    try:
        with open(transcript_path, encoding="utf-8") as handle:
            lines = handle.readlines()[-400:]
    except OSError:
        return None
    for raw in reversed(lines):
        try:
            entry = json.loads(raw)
        except ValueError:
            continue
        role, texts = _message(entry)
        if role == "assistant" and texts:
            return "\n".join(texts)
        if role == "user" and texts and not texts[0].lstrip().startswith("<"):
            return None  # The person spoke last: this turn has no answer.
    return None


def _message(entry) -> tuple[str | None, list[str]]:
    if not isinstance(entry, dict):
        return None, []
    if entry.get("type") in ("user", "assistant") and isinstance(entry.get("message"), dict):
        role, content = entry["type"], entry["message"].get("content")  # Claude Code
    elif entry.get("type") == "response_item" and isinstance(entry.get("payload"), dict):
        payload = entry["payload"]  # Codex
        if payload.get("type") != "message":
            return None, []
        role, content = payload.get("role"), payload.get("content")
    else:
        return None, []
    if isinstance(content, str):
        return role, [content] if content.strip() else []
    texts = [
        part.get("text")
        for part in content or []
        if isinstance(part, dict)
        and part.get("type") in ("text", "input_text", "output_text")
        and part.get("text")
    ]
    return role, texts

"""Report a coding assistant's model usage from its own session files.

Hooks never carry token counts, but every hook receives ``transcript_path``: the file
the assistant writes as it works. Claude Code records each model response with its
message id, model, and usage; Codex writes one usage record per model response. At the
end of each turn the plugin reads what was appended since the last report and sends one
``llm.call`` telemetry event per response.

Each event also says which tool calls the response made and which tool results its
input read (by id, name, and estimated size), so Aegis can attribute token spend to the
tools, MCP servers, skills, and subagents that drove it.

Only model names, token counts, tool names, sizes, ids, and timestamps leave the
machine: never prompt, response, tool argument, or tool result text. Reads are
incremental (a byte offset per session, plus the few tool links still open), and the
gateway deduplicates by response id, so a re-sent response is never counted twice.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

MAX_READ_BYTES = 4 * 1024 * 1024
MAX_EVENTS_PER_REQUEST = 500
MAX_REMEMBERED_TOOLS = 500
# Only these arguments are read, and only to name the skill or subagent a call ran.
TARGET_KEYS = {
    "Skill": ("skill", "command"),
    "Task": ("subagent_type",),
    "Agent": ("subagent_type",),
}


def estimate_tokens(value) -> int:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return (len(text) + 3) // 4


@dataclass
class ModelCall:
    call_id: str
    model: str
    occurred_at: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    subagent: bool = False
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)

    def event(self, session_id: str, provider: str) -> dict:
        event = {
            "type": "llm.call",
            "session_id": session_id,
            "call_id": self.call_id,
            "model": self.model,
            "provider": provider,
            "occurred_at": self.occurred_at,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "cache_write_tokens": self.cache_write_tokens,
                "reasoning_tokens": self.reasoning_tokens,
            },
        }
        if self.subagent:
            # Claude Code transcripts don't say which Task call ran a subagent, so its
            # calls are delegated work of an unnamed subagent.
            event["agent_name"] = "subagent"
            event["parent_tool_call_id"] = "subagent"
        if self.tool_calls:
            event["tool_calls"] = self.tool_calls[:256]
        if self.tool_results:
            event["tool_results"] = self.tool_results[:256]
        return event


@dataclass
class Links:
    """Tool links still open between reads: results awaiting the call that reads them,
    the names of recent tool calls, and (Codex) calls awaiting their usage record."""

    pending: list = field(default_factory=list)
    names: dict = field(default_factory=dict)
    requested: list = field(default_factory=list)
    # Codex: a usage record was read since the last ``token_count`` (which then repeats
    # it), and the running total the last ``token_count`` reported.
    recorded: bool = False
    total: dict | None = None

    @classmethod
    def load(cls, cursor: dict) -> Links:
        return cls(
            pending=list(cursor.get("pending") or []),
            names=dict(cursor.get("names") or {}),
            requested=list(cursor.get("requested") or []),
            recorded=bool(cursor.get("recorded")),
            total=cursor.get("total") if isinstance(cursor.get("total"), dict) else None,
        )

    def save(self) -> dict:
        names = dict(list(self.names.items())[-MAX_REMEMBERED_TOOLS:])
        saved = {"pending": self.pending, "names": names, "requested": self.requested}
        if self.recorded:
            saved["recorded"] = True
        if self.total:
            saved["total"] = self.total
        return saved

    def remember(self, tool_id: str, name: str, target) -> dict:
        self.names[tool_id] = [name, target]
        ref = {"id": tool_id, "name": name}
        if target:
            ref["target"] = target
        return ref

    def result(self, tool_id: str, content) -> None:
        name, target = (self.names.get(tool_id) or [None, None])[:2]
        ref = {"id": tool_id, "tokens": estimate_tokens(content)}
        if name:
            ref["name"] = name
        if target:
            ref["target"] = target
        self.pending.append(ref)

    def take(self) -> list:
        taken, self.pending = self.pending, []
        return taken


@dataclass
class Parsed:
    calls: list = field(default_factory=list)
    # Byte offset to resume from next time, relative to where reading started.
    resume_at: int = 0
    model: str = ""
    links: Links = field(default_factory=Links)


def _lines(data: bytes):
    """Complete lines with their starting offsets; a partial last line is left unread."""
    start = 0
    while True:
        end = data.find(b"\n", start)
        if end < 0:
            return
        yield start, end + 1, data[start:end]
        start = end + 1


def _json(raw: bytes) -> dict:
    try:
        value = json.loads(raw)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _int(value) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _target(name: str, tool_input) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    for key in TARGET_KEYS.get(name, ()):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value[:200]
    return None


class ClaudeTranscript:
    """Claude Code transcript: one line per content block of each assistant message.

    Blocks of one message share its id and usage, and the usage can still grow until
    the message completes, so the most recent message is held back unless the session
    is ending; the next report starts at its first line and sends it complete, with the
    tool results it read carried over in the cursor.
    """

    provider = "anthropic"

    def parse(self, data: bytes, model: str, final: bool, links: Links | None = None) -> Parsed:
        links = links or Links()
        calls: dict = {}
        first_line: dict = {}
        resume_at = 0
        for start, end, raw in _lines(data):
            resume_at = end
            line = _json(raw)
            message = line.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content") if isinstance(message.get("content"), list) else []
            if line.get("type") == "user":
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        if item.get("tool_use_id"):
                            links.result(str(item["tool_use_id"]), item.get("content") or "")
                continue
            if line.get("type") != "assistant":
                continue
            usage, message_id = message.get("usage"), message.get("id")
            name = message.get("model") or model
            if not isinstance(usage, dict) or not message_id or name == "<synthetic>":
                continue
            cache_read = _int(usage.get("cache_read_input_tokens"))
            cache_write = _int(usage.get("cache_creation_input_tokens"))
            previous = calls.get(message_id)
            if previous is None:
                first_line[message_id] = start
            call = ModelCall(
                call_id=message_id,
                model=name,
                occurred_at=line.get("timestamp") or "",
                # Aegis counts cached input as input, like OpenTelemetry.
                input_tokens=_int(usage.get("input_tokens")) + cache_read + cache_write,
                output_tokens=_int(usage.get("output_tokens")),
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                subagent=bool(line.get("isSidechain")),
                tool_calls=previous.tool_calls if previous else [],
                # A new response reads every tool result produced since the last one;
                # subagent lines interleaved in older transcripts read their own.
                tool_results=(
                    previous.tool_results
                    if previous
                    else ([] if line.get("isSidechain") else links.take())
                ),
            )
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_use" and item.get("id"):
                    tool_name = str(item.get("name") or "")
                    ref = links.remember(
                        str(item["id"]), tool_name, _target(tool_name, item.get("input"))
                    )
                    if all(r["id"] != ref["id"] for r in call.tool_calls):
                        call.tool_calls.append(ref)
            calls[message_id] = call
            model = name
        if calls and not final:
            last = next(reversed(calls))
            # Holding back the only message in this read would never make progress;
            # report it as it stands (the gateway keeps the first report of an id).
            if first_line[last] > 0:
                held = calls.pop(last)
                # The held message is re-read next time: return its results to pending.
                links.pending = held.tool_results + links.pending
                resume_at = first_line[last]
        return Parsed(list(calls.values()), resume_at, model, links)


class CodexRollout:
    """Codex session file: ``turn_context`` lines set the model, and each model response
    is a ``token_usage_record`` keyed by its response id, written after the tool calls
    it made. Tool outputs are read by the next response. Older Codex versions only write
    ``token_count`` events; those are keyed by their position in the file.

    A thread started on an older Codex and resumed on a newer one has both: the format
    is decided line by line, so a ``token_count`` counts only when no usage record came
    since the previous one (newer versions write both for each response), and never when
    its running total didn't move (a repeat of the last response, not a new one).
    """

    provider = "openai"
    CALL_TYPES = ("function_call", "custom_tool_call", "local_shell_call")
    OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")

    def parse(
        self,
        data: bytes,
        model: str,
        final: bool,
        base_offset: int = 0,
        links: Links | None = None,
    ) -> Parsed:
        links = links or Links()
        calls = []
        resume_at = 0
        for start, end, raw in _lines(data):
            resume_at = end
            line = _json(raw)
            payload = line.get("payload") if isinstance(line.get("payload"), dict) else {}
            kind = payload.get("type")
            if line.get("type") == "turn_context" and payload.get("model"):
                model = str(payload["model"])
            elif line.get("type") == "response_item" and kind in self.CALL_TYPES:
                call_id = payload.get("call_id") or payload.get("id")
                if call_id:
                    name = str(payload.get("name") or kind.removesuffix("_call"))
                    links.requested.append(links.remember(str(call_id), name, None))
            elif line.get("type") == "response_item" and kind in self.OUTPUT_TYPES:
                if payload.get("call_id"):
                    links.result(str(payload["call_id"]), payload.get("output") or "")
            elif line.get("type") == "token_usage_record":
                links.recorded = True
                self._add(
                    calls, links, payload.get("response_id"), payload.get("usage"), model, line
                )
            elif kind == "token_count":
                info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
                total = info.get("total_token_usage")
                repeated = isinstance(total, dict) and total == links.total
                if isinstance(total, dict):
                    links.total = total
                if links.recorded or repeated:
                    links.recorded = False
                    continue
                call_id = f"codex-offset-{base_offset + start}"
                self._add(calls, links, call_id, info.get("last_token_usage"), model, line)
        return Parsed(calls, resume_at, model, links)

    @staticmethod
    def _add(calls: list, links: Links, call_id, usage, model: str, line: dict) -> None:
        if not call_id or not isinstance(usage, dict) or not model:
            return
        # Order in the file: a response's tool calls, its usage record, then the outputs,
        # which the next response reads. So at each record, calls since the last record
        # are this response's, and outputs since then are what its input read.
        requested, links.requested = links.requested, []
        calls.append(
            ModelCall(
                call_id=str(call_id),
                model=model,
                occurred_at=line.get("timestamp") or "",
                input_tokens=_int(usage.get("input_tokens")),
                output_tokens=_int(usage.get("output_tokens")),
                cache_read_tokens=_int(usage.get("cached_input_tokens")),
                cache_write_tokens=_int(usage.get("cache_write_input_tokens")),
                reasoning_tokens=_int(usage.get("reasoning_output_tokens")),
                tool_calls=requested,
                tool_results=links.take(),
            )
        )


@dataclass
class UsageBatch:
    events: list
    offset: int
    model: str
    links: dict = field(default_factory=dict)
    at_end: bool = False


def read_usage(
    fmt,
    path: str,
    session_id: str,
    cursor: dict,
    *,
    hook_model: str = "",
    final: bool = False,
    since: str = "",
):
    """New model calls in ``path`` since ``cursor``; None if the file can't be read.

    ``cursor`` is the saved ``{"offset", "model", "pending", "names", ...}`` for this
    session. The returned batch carries the cursor to save once its events are delivered.
    Calls made before ``since`` (an ISO UTC time) are read past but not reported: a
    session resumed after connecting may hold weeks of history from before Aegis.
    """
    try:
        size = os.path.getsize(path)
        offset = int(cursor.get("offset") or 0)
        if offset > size:  # the file was replaced or truncated; start over
            offset, cursor = 0, {}
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read(MAX_READ_BYTES)
    except (OSError, ValueError):
        return None
    model = cursor.get("model") or hook_model
    links = Links.load(cursor)
    if isinstance(fmt, CodexRollout):
        parsed = fmt.parse(data, model, final, base_offset=offset, links=links)
    else:
        # A read cut off mid-file is not the end of the session, whatever the hook says.
        parsed = fmt.parse(data, model, final and offset + len(data) >= size, links=links)
    events = [
        call.event(session_id, fmt.provider)
        for call in parsed.calls
        if not (since and call.occurred_at and _utc_second(call.occurred_at) < since)
    ]
    return UsageBatch(
        events,
        offset + parsed.resume_at,
        parsed.model,
        parsed.links.save(),
        at_end=offset + len(data) >= size,
    )


def _utc_second(timestamp: str) -> str:
    """``2026-10-09T18:32:45.762Z`` -> ``2026-10-09T18:32:45``: both assistants write UTC."""
    return timestamp[:19]


def utc_iso(epoch_seconds) -> str:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(int(epoch_seconds)))
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def chunks(events: list):
    for index in range(0, len(events), MAX_EVENTS_PER_REQUEST):
        yield events[index : index + MAX_EVENTS_PER_REQUEST]


def safe_path(path: str) -> bool:
    """Only read transcripts the assistant itself would write: regular files."""
    try:
        return bool(path) and Path(path).is_file()
    except OSError:
        return False

"""Turn the Claude Agent SDK message stream into Aegis session telemetry.

Claude hooks never see token usage; it arrives on the message stream. The observer
watches that stream: the init message opens the session, each assistant message is a
model call, and each result message closes the turn with Claude's running cost.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from aegis_sdk.telemetry import (
    AegisTelemetry,
    TokenUsage,
    ToolCallRef,
    ToolResultRef,
    estimate_tokens,
)
from aegis_sdk.telemetry.events import SessionStatus, utc_now

FRAMEWORK = "claude-agent"
_MAX_TRACKED_SESSIONS = 1024


@dataclass
class _PendingCall:
    session_id: str
    model: str
    usage: TokenUsage
    parent_tool_use_id: str | None
    # When the message first arrived, so the timeline orders it before the tool calls
    # it triggered even though it is reported later, once its usage is final.
    occurred_at: datetime
    # Token attribution: the tools this response called, and the results its input read.
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)


class ClaudeTelemetryObserver:
    """Observe Claude Agent SDK messages and report them through ``AegisTelemetry``.

    Claude emits one ``AssistantMessage`` per content block, all sharing the API
    message id, and usage may grow across them. Calls are therefore held until the
    next message id (or the turn's result) arrives, then reported once with the
    final usage.
    """

    def __init__(self, telemetry: AegisTelemetry, *, agent_name: str | None = None) -> None:
        self.telemetry = telemetry
        self.agent_name = agent_name
        self._session_id: str | None = None
        self._pending: OrderedDict[str, _PendingCall] = OrderedDict()
        self._started: OrderedDict[str, None] = OrderedDict()
        # Tool results the next model call (in the same thread) will read.
        self._results: dict[str | None, list[ToolResultRef]] = {}
        self._tools: OrderedDict[str, tuple[str, str | None]] = OrderedDict()

    async def instrument(self, messages: AsyncIterable[Any]) -> AsyncIterator[Any]:
        """Yield ``messages`` unchanged while reporting telemetry for each."""

        async for message in messages:
            self.observe(message)
            yield message

    def observe(self, message: Any) -> None:
        kind = type(message).__name__
        if kind == "SystemMessage":
            self._on_system(message)
        elif kind == "AssistantMessage":
            self._on_assistant(message)
        elif kind == "ResultMessage":
            self._on_result(message)
        elif kind == "UserMessage":
            self._on_user(message)

    def start_session(self, session_id: str, *, prompt: str | None = None) -> None:
        """Open (or reopen, for a new turn) the session; safe to call repeatedly."""

        self.telemetry.start_session(
            session_id,
            input=prompt,
            framework=FRAMEWORK,
            attributes={"agent": self.agent_name} if self.agent_name else None,
        )
        self._remember_started(session_id)

    def _on_system(self, message: Any) -> None:
        data = getattr(message, "data", None) or {}
        session_id = data.get("session_id")
        if getattr(message, "subtype", None) != "init" or not session_id:
            return
        self._session_id = session_id
        if session_id not in self._started:
            self.start_session(session_id)

    def _on_assistant(self, message: Any) -> None:
        session_id = getattr(message, "session_id", None) or self._session_id
        message_id = getattr(message, "message_id", None)
        usage = getattr(message, "usage", None)
        if not session_id or not message_id or not isinstance(usage, Mapping):
            return
        for pending_id in [key for key in self._pending if key != message_id]:
            self._report(pending_id)
        previous = self._pending.get(message_id)
        parent = getattr(message, "parent_tool_use_id", None)
        call = _PendingCall(
            session_id=session_id,
            model=getattr(message, "model", None) or "unknown",
            usage=_token_usage(usage),
            parent_tool_use_id=parent,
            occurred_at=previous.occurred_at if previous else utc_now(),
            tool_calls=previous.tool_calls if previous else [],
            # A new response reads the results its thread produced since the last one.
            tool_results=previous.tool_results if previous else self._results.pop(parent, []),
        )
        for block in getattr(message, "content", None) or ():
            if type(block).__name__ == "ToolUseBlock" and getattr(block, "id", None):
                target = _target(getattr(block, "name", ""), getattr(block, "input", None))
                self._remember_tool(block.id, block.name, target)
                if all(ref.id != block.id for ref in call.tool_calls):
                    call.tool_calls.append(
                        ToolCallRef(id=block.id, name=block.name, target=target)
                    )
        self._pending[message_id] = call

    def _on_user(self, message: Any) -> None:
        """Tool results come back as user messages; size them for the next model call."""

        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return
        thread = getattr(message, "parent_tool_use_id", None)
        for block in content:
            if type(block).__name__ != "ToolResultBlock":
                continue
            tool_id = getattr(block, "tool_use_id", None)
            if not tool_id:
                continue
            name, target = self._tools.get(tool_id, (None, None))
            body = getattr(block, "content", None)
            text = body if isinstance(body, str) else repr(body)
            self._results.setdefault(thread, []).append(
                ToolResultRef(id=tool_id, name=name, target=target, tokens=estimate_tokens(text))
            )

    def _remember_tool(self, tool_id: str, name: str, target: str | None) -> None:
        self._tools[tool_id] = (name, target)
        while len(self._tools) > _MAX_TRACKED_SESSIONS:
            self._tools.popitem(last=False)

    def _on_result(self, message: Any) -> None:
        for pending_id in list(self._pending):
            self._report(pending_id)
        session_id = getattr(message, "session_id", None) or self._session_id
        if not session_id:
            return
        self.telemetry.end_session(
            session_id,
            status=_status(message),
            output=getattr(message, "result", None),
            cost_usd=getattr(message, "total_cost_usd", None),
        )

    def _report(self, message_id: str) -> None:
        call = self._pending.pop(message_id)
        is_subagent = call.parent_tool_use_id is not None
        self.telemetry.record_llm_call(
            call.session_id,
            call_id=message_id,
            model=call.model,
            provider="anthropic",
            agent_name="subagent" if is_subagent else self.agent_name,
            usage=call.usage,
            attributes={"parent_tool_use_id": call.parent_tool_use_id} if is_subagent else None,
            occurred_at=call.occurred_at,
            tool_calls=call.tool_calls,
            tool_results=call.tool_results,
            parent_tool_call_id=call.parent_tool_use_id,
        )

    def _remember_started(self, session_id: str) -> None:
        self._started[session_id] = None
        self._started.move_to_end(session_id)
        while len(self._started) > _MAX_TRACKED_SESSIONS:
            self._started.popitem(last=False)


def _token_usage(usage: Mapping[str, Any]) -> TokenUsage:
    """Anthropic reports cached input separately; Aegis input totals include it."""

    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    return TokenUsage(
        input_tokens=int(usage.get("input_tokens") or 0) + cache_read + cache_write,
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def _status(message: Any) -> SessionStatus:
    if (getattr(message, "terminal_reason", None) or "").startswith("aborted"):
        return "interrupted"
    return "failed" if getattr(message, "is_error", False) else "completed"


def _target(name: str, tool_input: Any) -> str | None:
    """The skill or subagent a ``Skill``/``Task`` call names; no other argument is kept."""

    if not isinstance(tool_input, Mapping):
        return None
    keys = {"Skill": ("skill", "command"), "Task": ("subagent_type",), "Agent": ("subagent_type",)}
    for key in keys.get(name, ()):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None

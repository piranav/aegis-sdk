"""Telemetry events sent to ``POST /v1/telemetry/events``.

``input_tokens`` follows the OpenTelemetry GenAI convention: it *includes* cached
tokens. Framework adapters normalize provider usage to this shape before sending.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, NonNegativeInt

AttributeValue = str | int | float | bool | None
SessionStatus = Literal["completed", "failed", "interrupted"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class TokenUsage(BaseModel):
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cache_read_tokens: NonNegativeInt = 0
    cache_write_tokens: NonNegativeInt = 0
    reasoning_tokens: NonNegativeInt = 0


class _TelemetryEvent(BaseModel):
    session_id: str = Field(min_length=1, max_length=256)
    occurred_at: datetime = Field(default_factory=utc_now)

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class SessionStart(_TelemetryEvent):
    type: Literal["session.start"] = "session.start"
    customer_id: str | None = None
    end_user_id: str | None = None
    framework: str | None = None
    input: str | None = None
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)


class ToolCallRef(BaseModel):
    """A tool call a model response requested."""

    id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=256)
    # The skill or subagent the call names (Claude's ``Skill`` / ``Task``), never its args.
    target: str | None = None
    # Explicit component path, e.g. [{"kind": "mcp_server", "name": "github"}, ...].
    component: list[dict[str, str]] | None = None


class ToolResultRef(BaseModel):
    """A tool result that entered a model call's input. Only its estimated size is sent."""

    id: str = Field(min_length=1, max_length=256)
    tokens: NonNegativeInt = 0
    name: str | None = None
    target: str | None = None
    component: list[dict[str, str]] | None = None


def estimate_tokens(text: str) -> int:
    """About four characters per token: close enough to rank what drives spend."""

    return (len(text) + 3) // 4


class LlmCall(_TelemetryEvent):
    type: Literal["llm.call"] = "llm.call"
    call_id: str = Field(min_length=1, max_length=256)
    model: str = Field(min_length=1, max_length=256)
    provider: str | None = None
    response_model: str | None = None
    agent_name: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    started_at: datetime | None = None
    latency_ms: NonNegativeInt | None = None
    finish_reason: str | None = None
    error_type: str | None = None
    cost_usd: Decimal | None = None
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)
    tool_calls: list[ToolCallRef] = Field(default_factory=list)
    tool_results: list[ToolResultRef] = Field(default_factory=list)
    parent_tool_call_id: str | None = None


class SessionEnd(_TelemetryEvent):
    type: Literal["session.end"] = "session.end"
    status: SessionStatus = "completed"
    output: str | None = None
    cost_usd: Decimal | None = Field(
        default=None, description="Runtime-reported running total for the session"
    )


TelemetryEvent = SessionStart | LlmCall | SessionEnd


class RejectedEvent(BaseModel):
    index: int
    reason: str


class TelemetryIngestResult(BaseModel):
    accepted: int
    duplicates: int
    rejected: list[RejectedEvent] = Field(default_factory=list)

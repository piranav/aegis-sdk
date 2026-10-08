"""Per-run state shared by Aegis hooks and guardrails in an OpenAI Agents run.

Lifecycle hooks, LLM hooks, and tool guardrails each receive a different context
wrapper, but the SDK hands all of them the same ``Usage`` object for the whole run.
Anchoring Aegis state on it gives every callback one session identity, so model
calls and governed tool calls land in the same Aegis session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from aegis_sdk.telemetry.binding import current_binding

_SCOPE_ATTR = "_aegis_run_scope"


@dataclass
class PendingLlmCall:
    started_at: datetime
    started_monotonic: float


@dataclass
class RunScope:
    session_id: str
    session_started: bool = False
    pending_llm_calls: dict[str, PendingLlmCall] = field(default_factory=dict)
    # Inventory: whether this run described its agent, and the skills that run can load.
    inventory_described: bool = False
    skills: list[Any] = field(default_factory=list)
    # Tool results waiting for the model call whose input will read them.
    pending_results: list[Any] = field(default_factory=list)


def run_scope(context: Any) -> RunScope:
    """Return the run's scope, creating it (and its session id) on first use."""

    anchor = getattr(context, "usage", None) or context
    scope = getattr(anchor, _SCOPE_ATTR, None)
    if scope is None:
        scope = RunScope(session_id=current_binding().session_id or uuid4().hex)
        setattr(anchor, _SCOPE_ATTR, scope)
    return scope

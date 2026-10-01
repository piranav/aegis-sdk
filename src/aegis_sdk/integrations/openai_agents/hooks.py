"""Observational RunHooks for the OpenAI Agents SDK.

These hooks do *not* block or alter execution -- they record lifecycle events
(agent start, handoff, tool start/end) into the run context so the audit trail
captures the full agent execution trace.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from agents import Agent
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext, RunContextWrapper
from agents.tool import Tool

logger = logging.getLogger(__name__)

TRACE_KEY = "aegis_trace"
# When ``Runner.run`` is called without a user context, ``context.context`` is
# ``None``.  In that case we attach the trace list to the wrapper itself.
_WRAPPER_TRACE_ATTR = "_aegis_trace_list"
_TOOL_START_MONOTONIC_ATTR = "_aegis_tool_start_monotonic"
_RAW_RESULT_ATTR = "_aegis_raw_tool_result"
_OUTPUT_DECISION_ATTR = "_aegis_output_decision"
_STATE_ATTR = "_aegis_guardrail_state"


class AegisRunHooks(RunHooksBase[Any, Agent]):
    """RunHooks that record agent lifecycle events for governance auditing."""

    async def on_agent_start(
        self,
        context: AgentHookContext[Any],
        agent: Agent,
    ) -> None:
        trace = _get_trace(context)
        trace.append(
            {
                "event": "agent_start",
                "agent": agent.name,
                "ts": _now(),
            }
        )

    async def on_agent_end(
        self,
        context: AgentHookContext[Any],
        agent: Agent,
        output: Any,
    ) -> None:
        trace = _get_trace(context)
        trace.append(
            {
                "event": "agent_end",
                "agent": agent.name,
                "ts": _now(),
            }
        )

    async def on_handoff(
        self,
        context: RunContextWrapper[Any],
        from_agent: Agent,
        to_agent: Agent,
    ) -> None:
        trace = _get_trace(context)
        trace.append(
            {
                "event": "handoff",
                "from": from_agent.name,
                "to": to_agent.name,
                "ts": _now(),
            }
        )

    async def on_tool_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent,
        tool: Tool,
    ) -> None:
        setattr(context, _TOOL_START_MONOTONIC_ATTR, perf_counter())
        trace = _get_trace(context)
        input_state = (
            getattr(context, _STATE_ATTR, None)
            or getattr(context, "_aegis_gateway_guardrail_state", None)
            or {}
        )
        input_decision = input_state.get("input_decision")
        entry = {
            "event": "tool_start",
            "agent": agent.name,
            "tool": tool.name,
            "tool_call_id": getattr(context, "tool_call_id", None),
            "tool_args": _parse_tool_args(getattr(context, "tool_arguments", None)),
            "input_action_id": input_state.get("input_action_id"),
            "ts": _now(),
        }
        if input_decision is not None:
            entry["input_decision"] = input_decision.decision.value
            entry["input_allowed"] = input_decision.allowed
            entry["input_violations"] = input_decision.violations
            entry["classify_ms"] = input_decision.classify_ms
            entry["validate_ms"] = input_decision.validate_ms
            entry["total_ms"] = input_decision.total_ms
        trace.append(entry)

    async def on_tool_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent,
        tool: Tool,
        result: object,
    ) -> None:
        trace = _get_trace(context)
        raw_result = getattr(context, _RAW_RESULT_ATTR, None)
        output_decision = getattr(context, _OUTPUT_DECISION_ATTR, None)
        input_state = (
            getattr(context, _STATE_ATTR, None)
            or getattr(context, "_aegis_gateway_guardrail_state", None)
            or {}
        )
        started_at = getattr(context, _TOOL_START_MONOTONIC_ATTR, None)
        elapsed_ms = None
        if started_at is not None:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 3)

        entry = {
            "event": "tool_end",
            "agent": agent.name,
            "tool": tool.name,
            "tool_call_id": getattr(context, "tool_call_id", None),
            "result_text": _safe_text(result),
            "raw_result_text": raw_result,
            "input_action_id": input_state.get("input_action_id"),
            "elapsed_ms": elapsed_ms,
            "ts": _now(),
        }
        if output_decision is not None:
            entry["output_decision"] = output_decision.decision.value
            entry["output_allowed"] = output_decision.allowed
            entry["output_violations"] = output_decision.violations
            entry["classify_ms"] = output_decision.classify_ms
            entry["validate_ms"] = output_decision.validate_ms
            entry["total_ms"] = output_decision.total_ms
            if output_decision.result_classification is not None:
                entry["output_classification"] = {
                    "contains_pii": output_decision.result_classification.contains_pii,
                    "contains_phi": output_decision.result_classification.contains_phi,
                    "contains_credentials": (
                        output_decision.result_classification.contains_credentials
                    ),
                    "data_categories": output_decision.result_classification.data_categories,
                    "risk_assessment": output_decision.result_classification.risk_assessment,
                }
        trace.append(entry)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_trace(wrapper: RunContextWrapper | AgentHookContext) -> list[dict]:
    """Retrieve (or initialise) the aegis trace list.

    Prefer the user-supplied context object (dict or arbitrary object with a
    ``aegis_trace`` attribute).  If there is no user context (``None``), store
    on the ``RunContextWrapper`` instance so hooks still work.
    """
    ctx = wrapper.context
    if isinstance(ctx, dict):
        return ctx.setdefault(TRACE_KEY, [])
    if ctx is not None:
        if not hasattr(ctx, TRACE_KEY):
            setattr(ctx, TRACE_KEY, [])
        return getattr(ctx, TRACE_KEY)
    if hasattr(wrapper, _WRAPPER_TRACE_ATTR):
        return getattr(wrapper, _WRAPPER_TRACE_ATTR)
    trace: list[dict] = []
    setattr(wrapper, _WRAPPER_TRACE_ATTR, trace)
    return trace


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_tool_args(raw_args: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw_args}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)

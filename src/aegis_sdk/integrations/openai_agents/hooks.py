"""Observational RunHooks for the OpenAI Agents SDK.

These hooks do *not* block or alter execution -- they record lifecycle events
(agent start, handoff, tool start/end) into the run context so the audit trail
captures the full agent execution trace. Given an ``AegisTelemetry``, they also
report the run as an Aegis session with every model call and its token usage. Given
an ``AegisInventory``, they report what the agent is built from and what it uses.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from agents import Agent
from agents.items import ModelResponse
from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext, RunContextWrapper
from agents.tool import Tool

from aegis_sdk.integrations.openai_agents.attribution import (
    response_tool_links,
    tool_result_ref,
)
from aegis_sdk.integrations.openai_agents.inventory import (
    FRAMEWORK,
    describe_agent,
    response_usage_paths,
    tool_usage_path,
)
from aegis_sdk.integrations.openai_agents.run_scope import PendingLlmCall, run_scope
from aegis_sdk.inventory import AegisInventory, Component
from aegis_sdk.telemetry import AegisTelemetry, TokenUsage

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
    """RunHooks that record agent lifecycle events for governance auditing.

    Pass ``telemetry`` to report each run as an Aegis session, and ``inventory`` to
    report the agent's components (models, tools, MCP servers, connectors, skills,
    subagents) and which of them each run uses::

        hooks = AegisRunHooks(
            telemetry=AegisTelemetry.from_client(gateway),
            inventory=AegisInventory.from_client(gateway),
        )
        with bind_session(customer_id="acme"):
            await Runner.run(agent, prompt, hooks=hooks)

    The agent graph is described at most once per ``describe_interval`` seconds per
    root agent, and sent only when it changed.
    """

    def __init__(
        self,
        telemetry: AegisTelemetry | None = None,
        inventory: AegisInventory | None = None,
        *,
        describe_interval: float = 600.0,
    ) -> None:
        self.telemetry = telemetry
        self.inventory = inventory
        self.describe_interval = describe_interval
        # id(root agent) -> (when described, its skills); agents are long-lived objects.
        self._described: dict[int, tuple[float, list[Component]]] = {}

    async def on_agent_start(
        self,
        context: AgentHookContext[Any],
        agent: Agent,
    ) -> None:
        scope = run_scope(context)
        if self.inventory is not None and not scope.inventory_described:
            scope.inventory_described = True
            scope.skills = await self._describe(context, agent)
        if self.telemetry is not None and not scope.session_started:
            # Fires again on every handoff; only the run's first agent opens the session.
            scope.session_started = True
            self.telemetry.start_session(
                scope.session_id,
                input=_user_input_text(getattr(context, "turn_input", None)),
                framework="openai-agents",
                attributes={"agent": agent.name},
            )
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
        if self.telemetry is not None:
            self.telemetry.end_session(run_scope(context).session_id, output=_safe_text(output))
        trace = _get_trace(context)
        trace.append(
            {
                "event": "agent_end",
                "agent": agent.name,
                "ts": _now(),
            }
        )

    async def on_llm_start(
        self,
        context: RunContextWrapper[Any],
        agent: Agent,
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        if self.telemetry is not None:
            run_scope(context).pending_llm_calls[agent.name] = PendingLlmCall(
                started_at=datetime.now(UTC), started_monotonic=perf_counter()
            )

    async def on_llm_end(
        self,
        context: RunContextWrapper[Any],
        agent: Agent,
        response: ModelResponse,
    ) -> None:
        if self.inventory is not None:
            for path in response_usage_paths(response, skills=run_scope(context).skills):
                self.inventory.record_usage(path)
        if self.telemetry is None:
            return
        scope = run_scope(context)
        pending = scope.pending_llm_calls.pop(agent.name, None)
        tool_calls, hosted_results = response_tool_links(response)
        consumed, scope.pending_results = [*scope.pending_results, *hosted_results], []
        self.telemetry.record_llm_call(
            scope.session_id,
            call_id=response.response_id,
            model=_model_name(agent),
            agent_name=agent.name,
            usage=_token_usage(response.usage),
            started_at=pending.started_at if pending else None,
            latency_ms=(
                round((perf_counter() - pending.started_monotonic) * 1000) if pending else None
            ),
            tool_calls=tool_calls,
            tool_results=consumed,
        )

    async def on_handoff(
        self,
        context: RunContextWrapper[Any],
        from_agent: Agent,
        to_agent: Agent,
    ) -> None:
        if self.inventory is not None:
            self.inventory.record_usage([("subagent", to_agent.name)])
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
        if self.inventory is not None and not input_state:
            # Calls that passed an Aegis guardrail are already recorded by the gateway.
            path = tool_usage_path(tool)
            if path:
                self.inventory.record_usage(path)
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
        if self.telemetry is not None:
            # The next model call reads this result; attach its size to that call.
            ref = tool_result_ref(context, tool, result)
            if ref is not None:
                run_scope(context).pending_results.append(ref)
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

    async def _describe(self, context: Any, agent: Agent) -> list[Component]:
        """Report the agent graph's manifest if due; return the skills it can load."""

        assert self.inventory is not None
        cached = self._described.get(id(agent))
        if cached is not None and perf_counter() - cached[0] < self.describe_interval:
            return cached[1]
        try:
            components = await describe_agent(agent, run_context=context)
        except Exception:  # noqa: BLE001 - inventory must never break the agent run
            logger.warning("Aegis could not describe agent %s", agent.name, exc_info=True)
            return cached[1] if cached else []
        self.inventory.report_manifest(components, framework=FRAMEWORK)
        skills = [component for component in components if component.kind == "skill"]
        self._described[id(agent)] = (perf_counter(), skills)
        return skills


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


def _model_name(agent: Agent) -> str:
    """The model an agent runs on: a name, a ``Model`` instance, or the SDK default."""

    model = agent.model
    if isinstance(model, str) and model:
        return model
    name = getattr(model, "model", None)
    if isinstance(name, str) and name:
        return name
    try:
        from agents.models import get_default_model
    except ImportError:  # pragma: no cover - older SDKs
        return "unknown"
    return get_default_model()


def _token_usage(usage: Any) -> TokenUsage:
    """Normalize SDK usage; OpenAI input tokens already include cached tokens."""

    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(input_details, "cached_tokens", 0) or 0,
        cache_write_tokens=getattr(input_details, "cache_write_tokens", 0) or 0,
        reasoning_tokens=getattr(output_details, "reasoning_tokens", 0) or 0,
    )


def _user_input_text(turn_input: list[Any] | None) -> str | None:
    """The user's text from the run input, which is either a string or input items."""

    if not turn_input:
        return None
    texts = []
    for item in turn_input:
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "input_text"
            )
    return "\n".join(text for text in texts if text) or None


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

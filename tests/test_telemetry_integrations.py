"""Framework adapters: OpenAI Agents hooks/guardrails and the Claude message observer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from aegis_sdk import AegisGatewayClient, AegisTelemetry, bind_session
from aegis_sdk.integrations.claude_agent import AegisClaudeGatewayHooks, ClaudeTelemetryObserver
from aegis_sdk.telemetry import InMemoryTelemetryExporter, LlmCall, SessionEnd, SessionStart
from aegis_sdk.telemetry.events import utc_now


def recorder() -> tuple[AegisTelemetry, InMemoryTelemetryExporter]:
    exporter = InMemoryTelemetryExporter()
    return AegisTelemetry(exporter), exporter


# -------------------------------------------------------------------------- OpenAI Agents


def _allow_decision() -> dict[str, Any]:
    return {
        "action": {
            "context": {"agent_name": "support", "tool_name": "lookup"},
            "action_types": ["aegis:ReadAction"],
            "risk_level": "aegis:LowRisk",
        },
        "allowed": True,
        "decision": "allow",
        "audit_id": "audit-1",
    }


async def test_openai_run_reports_one_session_shared_with_tool_guardrails() -> None:
    pytest.importorskip("agents")
    from agents import Agent, ToolInputGuardrailData
    from agents.items import ModelResponse
    from agents.run_context import AgentHookContext, RunContextWrapper
    from agents.tool import ToolContext
    from agents.usage import InputTokensDetails, OutputTokensDetails, Usage

    from aegis_sdk.integrations.openai_agents import AegisRunHooks, make_aegis_gateway_guardrail

    evaluated: list[dict] = []
    gateway = AegisGatewayClient(
        "http://aegis.test",
        "aegk_key",
        transport=httpx.MockTransport(
            lambda request: (
                evaluated.append(json.loads(request.content))
                or httpx.Response(200, json=_allow_decision())
            )
        ),
    )
    telemetry, exporter = recorder()
    hooks = AegisRunHooks(telemetry=telemetry)
    agent = Agent(name="support", model="gpt-4.1-mini")
    run = RunContextWrapper(context=None)
    start_context = AgentHookContext(
        context=None,
        usage=run.usage,
        turn_input=[{"role": "user", "content": "Where is order 7?"}],
    )
    response = ModelResponse(
        output=[],
        usage=Usage(
            requests=1,
            input_tokens=900,
            output_tokens=40,
            input_tokens_details=InputTokensDetails(cached_tokens=600, cache_write_tokens=50),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=12),
        ),
        response_id="resp_1",
    )

    with bind_session(customer_id="acme"):
        await hooks.on_agent_start(start_context, agent)
        await hooks.on_llm_start(run, agent, None, [])
        await hooks.on_llm_end(run, agent, response)
        tool_context = ToolContext(
            context=None,
            usage=run.usage,
            tool_name="lookup",
            tool_call_id="call-1",
            tool_arguments='{"order": 7}',
        )
        await make_aegis_gateway_guardrail(gateway).run(
            ToolInputGuardrailData(context=tool_context, agent=agent)
        )
        # A handoff restarts the agent hook; it must not open a second session.
        await hooks.on_agent_start(AgentHookContext(context=None, usage=run.usage), agent)
        await hooks.on_agent_end(start_context, agent, "Order 7 ships tomorrow.")

    start, call, end = exporter.events
    assert isinstance(start, SessionStart)
    assert (start.customer_id, start.framework, start.input) == (
        "acme",
        "openai-agents",
        "Where is order 7?",
    )
    assert isinstance(call, LlmCall)
    assert (call.call_id, call.model, call.agent_name) == ("resp_1", "gpt-4.1-mini", "support")
    assert call.usage.model_dump() == {
        "input_tokens": 900,
        "output_tokens": 40,
        "cache_read_tokens": 600,
        "cache_write_tokens": 50,
        "reasoning_tokens": 12,
    }
    assert call.latency_ms is not None and call.started_at is not None
    assert isinstance(end, SessionEnd) and end.output == "Order 7 ships tomorrow."
    assert call.session_id == end.session_id == start.session_id
    assert evaluated[0]["session_id"] == start.session_id


async def test_openai_hooks_without_telemetry_only_trace() -> None:
    pytest.importorskip("agents")
    from agents import Agent
    from agents.run_context import AgentHookContext

    from aegis_sdk.integrations.openai_agents import AegisRunHooks

    context = AgentHookContext(context={})
    await AegisRunHooks().on_agent_start(context, Agent(name="support"))

    assert context.context["aegis_trace"][0]["event"] == "agent_start"


# ----------------------------------------------------------------------------- Claude SDK
# The observer dispatches on message class names, so plain stand-ins keep these tests
# independent of the optional claude-agent-sdk dependency.


@dataclass
class SystemMessage:
    subtype: str
    data: dict[str, Any]


@dataclass
class AssistantMessage:
    model: str
    message_id: str
    usage: dict[str, Any]
    session_id: str | None = None
    parent_tool_use_id: str | None = None
    content: list[Any] = field(default_factory=list)


@dataclass
class ResultMessage:
    session_id: str
    is_error: bool = False
    result: str | None = None
    total_cost_usd: float | None = None
    terminal_reason: str | None = None


def _usage(input_tokens: int, output_tokens: int, cache_read: int = 0) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": 0,
    }


async def _stream(*messages: Any):
    for message in messages:
        yield message


async def test_claude_stream_reports_calls_once_with_final_usage_and_cost() -> None:
    telemetry, exporter = recorder()
    observer = ClaudeTelemetryObserver(telemetry, agent_name="billing")
    messages = [
        SystemMessage("init", {"session_id": "sess-1", "model": "claude-sonnet-4-5"}),
        # One API message split across content blocks; usage grows between them.
        AssistantMessage("claude-sonnet-4-5", "msg-1", _usage(10, 5, cache_read=1000)),
        AssistantMessage("claude-sonnet-4-5", "msg-1", _usage(10, 60, cache_read=1000)),
        AssistantMessage(
            "claude-haiku-4-5", "msg-2", _usage(300, 20), parent_tool_use_id="toolu_1"
        ),
        ResultMessage("sess-1", result="Done", total_cost_usd=0.0213),
    ]

    seen = [message async for message in observer.instrument(_stream(*messages))]

    assert seen == messages
    start, first, second, end = exporter.events
    assert (start.session_id, start.framework) == ("sess-1", "claude-agent")
    assert (first.call_id, first.agent_name, first.provider) == ("msg-1", "billing", "anthropic")
    # Anthropic input excludes cache reads; Aegis input totals include them.
    assert (first.usage.input_tokens, first.usage.output_tokens) == (1010, 60)
    assert first.usage.cache_read_tokens == 1000
    assert (second.agent_name, second.attributes) == (
        "subagent",
        {"parent_tool_use_id": "toolu_1"},
    )
    assert (end.status, end.output, str(end.cost_usd)) == ("completed", "Done", "0.0213")
    # Timestamps reflect arrival, not the later report once usage was final.
    assert start.occurred_at <= first.occurred_at <= second.occurred_at <= end.occurred_at


@pytest.mark.parametrize(
    ("result", "status"),
    [
        (ResultMessage("s", is_error=True), "failed"),
        (ResultMessage("s", terminal_reason="aborted_streaming"), "interrupted"),
    ],
)
def test_claude_result_maps_to_session_status(result: ResultMessage, status: str) -> None:
    telemetry, exporter = recorder()

    ClaudeTelemetryObserver(telemetry).observe(result)

    assert exporter.events[-1].status == status


def test_claude_call_is_timed_when_it_first_arrives() -> None:
    telemetry, exporter = recorder()
    observer = ClaudeTelemetryObserver(telemetry)
    observer.observe(SystemMessage("init", {"session_id": "s"}))
    observer.observe(AssistantMessage("claude-sonnet-4-5", "msg-1", _usage(1, 1)))
    arrived = utc_now()
    observer.observe(AssistantMessage("claude-sonnet-4-5", "msg-2", _usage(1, 1)))

    first_call = exporter.events[1]
    assert first_call.call_id == "msg-1" and first_call.occurred_at <= arrived


async def test_claude_prompt_hook_opens_the_turn_with_the_prompt() -> None:
    telemetry, exporter = recorder()
    gateway = AegisGatewayClient("http://aegis.test", "aegk_key")
    hooks = AegisClaudeGatewayHooks(gateway, telemetry=telemetry)

    with bind_session(customer_id="acme"):
        output = await hooks.user_prompt_submit(
            {"hook_event_name": "UserPromptSubmit", "session_id": "sess-9", "prompt": "Hi"},
            None,
            None,
        )

    assert output == {}
    (start,) = exporter.events
    assert (start.session_id, start.input, start.customer_id) == ("sess-9", "Hi", "acme")

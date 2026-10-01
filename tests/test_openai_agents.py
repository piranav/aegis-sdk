"""OpenAI gateway guardrail contract tests."""

import httpx
import pytest
from agents import Agent, ToolInputGuardrailData, ToolOutputGuardrailData
from agents.tool import ToolContext

from aegis_sdk import AegisGatewayClient
from aegis_sdk.integrations.openai_agents import (
    make_aegis_gateway_guardrail,
    make_aegis_gateway_output_guardrail,
)


def _tool_context(
    *,
    tool_name: str = "lookup_customer",
    tool_arguments: str = '{"customer_id": "C-1001"}',
    context: dict | None = None,
) -> ToolContext:
    return ToolContext(
        context={} if context is None else context,
        tool_name=tool_name,
        tool_call_id="call-1",
        tool_arguments=tool_arguments,
    )


@pytest.mark.asyncio
async def test_openai_gateway_guardrails_call_dashboard_api() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/evaluate":
            return httpx.Response(200, json=_gateway_decision_payload(audit_id="audit-input-1"))
        if request.url.path == "/v1/evaluate_result":
            return httpx.Response(
                200,
                json=_gateway_decision_payload(
                    audit_id="audit-input-1",
                    evaluation_phase="output",
                ),
            )
        return httpx.Response(404, json={"detail": "not found"})

    client = AegisGatewayClient(
        "http://aegis.test",
        "aegk_dashboard_key",
        transport=httpx.MockTransport(handler),
    )
    input_guardrail = make_aegis_gateway_guardrail(client)
    output_guardrail = make_aegis_gateway_output_guardrail(client)
    agent = Agent(name="dashboard-agent", instructions="Test agent")
    context = _tool_context()

    input_output = await input_guardrail.run(ToolInputGuardrailData(context=context, agent=agent))
    output_output = await output_guardrail.run(
        ToolOutputGuardrailData(context=context, agent=agent, output="safe result")
    )

    assert input_output.behavior["type"] == "allow"
    assert output_output.behavior["type"] == "allow"
    assert [request.url.path for request in requests] == ["/v1/evaluate", "/v1/evaluate_result"]
    assert all(
        request.headers["authorization"] == "Bearer aegk_dashboard_key" for request in requests
    )
    assert requests[1].read()  # request body is available for inspection if this fails


@pytest.mark.asyncio
async def test_openai_gateway_guardrail_rejects_gateway_errors() -> None:
    client = AegisGatewayClient(
        "http://aegis.test",
        "aegk_disabled",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, json={"detail": "Invalid API key"})
        ),
    )
    guardrail = make_aegis_gateway_guardrail(client)
    agent = Agent(name="dashboard-agent", instructions="Test agent")
    context = _tool_context()

    output = await guardrail.run(ToolInputGuardrailData(context=context, agent=agent))

    assert output.behavior["type"] == "reject_content"
    assert "invalid api key" in output.behavior["message"].lower()


def _gateway_decision_payload(
    *,
    audit_id: str,
    evaluation_phase: str = "input",
) -> dict:
    return {
        "action": {
            "context": {
                "agent_name": "dashboard-agent",
                "tool_name": "lookup_customer",
                "tool_args": {"customer_id": "C-1001"},
                "user_id": None,
                "session_id": None,
                "has_human_approval": False,
                "timestamp": "2026-05-01T12:00:00Z",
            },
            "action_types": ["aegis:Action"],
            "risk_level": "aegis:LowRisk",
            "involves_amount": None,
            "target_entity": None,
            "reasoning": "gateway test",
        },
        "allowed": True,
        "violations": [],
        "violated_shapes": [],
        "decision": "allow",
        "evaluation_phase": evaluation_phase,
        "action_id": audit_id,
        "result_classification": None,
        "classify_ms": 1.0,
        "validate_ms": 1.0,
        "total_ms": 2.0,
        "audit_id": audit_id,
    }

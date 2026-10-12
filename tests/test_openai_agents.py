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


@pytest.mark.asyncio
async def test_escalated_call_asks_the_user_and_runs_after_their_yes() -> None:
    requests: list[tuple[str, dict]] = []
    approved = {"yes": False}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.read() or b"{}")
        requests.append((request.url.path, body))
        if request.url.path == "/v1/confirmations/conf-1/reply":
            approved["yes"] = "yes" in body["message"].lower()
            return httpx.Response(
                200,
                json={"id": "conf-1", "status": "approved" if approved["yes"] else "pending"},
            )
        if body.get("confirmation_id") == "conf-1" and approved["yes"]:
            return httpx.Response(200, json=_gateway_decision_payload(audit_id="audit-2"))
        held = _gateway_decision_payload(audit_id="audit-1")
        held.update(
            allowed=False,
            decision="escalate",
            violations=["Deleting customers needs the user's OK"],
            confirmation={
                "id": "conf-1",
                "question": "Should I delete customer C-1001?",
                "agent_instruction": "[AEGIS] CONFIRMATION NEEDED: ask the user first.",
            },
        )
        return httpx.Response(200, json=held)

    client = AegisGatewayClient(
        "http://aegis.test", "aegk_key", transport=httpx.MockTransport(handler)
    )
    guardrail = make_aegis_gateway_guardrail(client)
    agent = Agent(name="support-agent", instructions="Test agent")

    def attempt(*messages: str):
        context = _tool_context()
        context.turn_input = [{"role": "user", "content": text} for text in messages]
        return guardrail.run(ToolInputGuardrailData(context=context, agent=agent))

    first = await attempt("Delete customer C-1001")
    assert first.behavior["type"] == "reject_content"
    assert "CONFIRMATION NEEDED" in first.behavior["message"]

    # Retrying before the user answers relays nothing and is held again.
    again = await attempt("Delete customer C-1001")
    assert again.behavior["type"] == "reject_content"
    assert not any(path.endswith("/reply") for path, _ in requests)

    # The user's next message is relayed as their answer; the retry then runs.
    done = await attempt("Delete customer C-1001", "Should I?", "Yes, go ahead")
    assert done.behavior["type"] == "allow"
    reply = next(body for path, body in requests if path.endswith("/reply"))
    assert reply["message"] == "Yes, go ahead"
    assert requests[-1][1]["confirmation_id"] == "conf-1"


@pytest.mark.asyncio
async def test_when_the_agent_already_asked_the_user_is_not_asked_twice() -> None:
    import json

    replies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read() or b"{}")
        if request.url.path.endswith("/reply"):
            replies.append(body)
            return httpx.Response(200, json={"id": "conf-1", "status": "approved"})
        if body.get("confirmation_id") == "conf-1":
            return httpx.Response(200, json=_gateway_decision_payload(audit_id="audit-2"))
        held = _gateway_decision_payload(audit_id="audit-1")
        held.update(
            allowed=False,
            decision="escalate",
            confirmation={"id": "conf-1", "question": "q", "agent_instruction": "ask"},
        )
        return httpx.Response(200, json=held)

    client = AegisGatewayClient("http://aegis.test", "k", transport=httpx.MockTransport(handler))
    guardrail = make_aegis_gateway_guardrail(client)
    context = _tool_context()
    context.turn_input = [
        {"role": "user", "content": "Delete customer C-1001"},
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Delete C-1001 permanently. Sure?"}],
        },
        {"role": "user", "content": "yes"},
    ]
    output = await guardrail.run(
        ToolInputGuardrailData(context=context, agent=Agent(name="a", instructions="t"))
    )

    assert output.behavior["type"] == "allow"
    assert replies == [
        {
            "message": "yes",
            "user_id": None,
            "source": "conversation",
            "agent_question": "Delete C-1001 permanently. Sure?",
        }
    ]

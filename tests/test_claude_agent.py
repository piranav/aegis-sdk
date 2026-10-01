"""Claude gateway hooks and provider contract tests."""

import sys
import types
from typing import Any

import httpx
import pytest

from aegis_sdk import AegisGatewayClient
from aegis_sdk.integrations.claude_agent import AegisClaudeGatewayHooks, AegisClaudeGatewayProvider


def _pre_input(
    *,
    tool_name: str = "get_weather",
    tool_input: dict[str, Any] | None = None,
    session_id: str = "session-1",
    tool_use_id: str = "tool-use-1",
    agent_id: str | None = "agent-1",
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "hook_event_name": "PreToolUse",
        "session_id": session_id,
        "transcript_path": "/tmp/transcript.jsonl",
        "cwd": "/tmp",
        "tool_name": tool_name,
        "tool_input": tool_input or {"city": "London"},
        "tool_use_id": tool_use_id,
    }
    if agent_id is not None:
        data["agent_id"] = agent_id
    return data


def _post_input(
    *,
    tool_name: str = "get_weather",
    tool_input: dict[str, Any] | None = None,
    tool_response: Any = "clean result",
    session_id: str = "session-1",
    tool_use_id: str = "tool-use-1",
) -> dict[str, Any]:
    return {
        "hook_event_name": "PostToolUse",
        "session_id": session_id,
        "transcript_path": "/tmp/transcript.jsonl",
        "cwd": "/tmp",
        "tool_name": tool_name,
        "tool_input": tool_input or {"city": "London"},
        "tool_response": tool_response,
        "tool_use_id": tool_use_id,
    }


@pytest.mark.asyncio
async def test_claude_gateway_hooks_call_dashboard_api() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/evaluate":
            return httpx.Response(200, json=_gateway_decision_payload(audit_id="audit-input-1"))
        if request.url.path == "/v1/evaluate_result":
            return httpx.Response(
                200,
                json=_gateway_decision_payload(
                    audit_id="audit-output-1",
                    evaluation_phase="output",
                ),
            )
        return httpx.Response(404, json={"detail": "not found"})

    client = AegisGatewayClient(
        "http://aegis.test",
        "aegk_dashboard_key",
        transport=httpx.MockTransport(handler),
    )
    hooks = AegisClaudeGatewayHooks(client)

    input_output = await hooks.pre_tool_use(_pre_input(), "tool-use-1", None)
    output_output = await hooks.post_tool_use(
        _post_input(tool_response={"city": "London", "temp": "22C"}),
        "tool-use-1",
        None,
    )

    assert input_output == {}
    assert output_output == {}
    assert [request.url.path for request in requests] == ["/v1/evaluate", "/v1/evaluate_result"]
    assert all(
        request.headers["authorization"] == "Bearer aegk_dashboard_key" for request in requests
    )
    assert requests[0].read()
    assert requests[1].read()
    trace = hooks.get_trace()
    assert trace[-1]["event"] == "post_tool_use"
    assert trace[-1]["input_audit_id"] == "audit-input-1"
    assert trace[-1]["output_audit_id"] == "audit-output-1"
    assert hooks.get_stored_tool_use(session_id="session-1", tool_use_id="tool-use-1") is None


@pytest.mark.asyncio
async def test_claude_gateway_pre_tool_denial_blocks_tool() -> None:
    client = AegisGatewayClient(
        "http://aegis.test",
        "aegk_dashboard_key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=_gateway_decision_payload(
                    audit_id="audit-input-1",
                    allowed=False,
                    decision="deny",
                    violations=["human approval required"],
                ),
            )
        ),
    )
    hooks = AegisClaudeGatewayHooks(client)

    output = await hooks.pre_tool_use(_pre_input(tool_name="delete_records"), "tool-use-1", None)

    assert output["suppressOutput"] is True
    assert output["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "human approval required" in output["hookSpecificOutput"]["permissionDecisionReason"]


@pytest.mark.asyncio
async def test_claude_gateway_output_denial_replaces_tool_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/evaluate":
            return httpx.Response(200, json=_gateway_decision_payload(audit_id="audit-input-1"))
        return httpx.Response(
            200,
            json=_gateway_decision_payload(
                audit_id="audit-output-1",
                evaluation_phase="output",
                allowed=False,
                decision="deny",
                violations=["credentials exposed"],
                result_classification={
                    "contains_pii": False,
                    "contains_phi": False,
                    "contains_credentials": True,
                    "data_categories": ["api_key"],
                    "risk_assessment": "API key exposed",
                    "reasoning": "secret-like token",
                },
            ),
        )

    client = AegisGatewayClient(
        "http://aegis.test",
        "aegk_dashboard_key",
        transport=httpx.MockTransport(handler),
    )
    hooks = AegisClaudeGatewayHooks(client)
    await hooks.pre_tool_use(_pre_input(), "tool-use-1", None)

    output = await hooks.post_tool_use(
        _post_input(tool_response={"api_key": "sk-secret"}),
        "tool-use-1",
        None,
    )

    assert output["suppressOutput"] is True
    assert output["decision"] == "block"
    assert output["hookSpecificOutput"]["updatedToolOutput"]["isError"] is True
    assert (
        "credentials exposed"
        in output["hookSpecificOutput"]["updatedToolOutput"]["content"][0]["text"]
    )
    assert "Do not reveal" in output["hookSpecificOutput"]["additionalContext"]


@pytest.mark.asyncio
async def test_claude_gateway_missing_pre_state_fails_closed() -> None:
    client = AegisGatewayClient(
        "http://aegis.test",
        "aegk_dashboard_key",
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
    )
    hooks = AegisClaudeGatewayHooks(client)

    output = await hooks.post_tool_use(_post_input(), "tool-use-1", None)

    assert output["suppressOutput"] is True
    assert "missing input evaluation state" in output["reason"]
    assert output["hookSpecificOutput"]["updatedToolOutput"]["isError"] is True


@pytest.mark.asyncio
async def test_claude_gateway_errors_fail_closed() -> None:
    client = AegisGatewayClient(
        "http://aegis.test",
        "aegk_bad_key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, json={"detail": "Invalid API key"})
        ),
    )
    hooks = AegisClaudeGatewayHooks(client)

    output = await hooks.pre_tool_use(_pre_input(), "tool-use-1", None)

    assert output["suppressOutput"] is True
    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "Invalid API key" in output["reason"]


def test_gateway_provider_lazily_imports_claude_and_merges_existing_hooks(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeHookMatcher:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.hooks = kwargs["hooks"]
            self.matcher = kwargs.get("matcher")
            self.timeout = kwargs.get("timeout")

    class FakeClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.hooks = kwargs["hooks"]
            self.max_turns = kwargs.get("max_turns")

    fake_module = types.SimpleNamespace(
        ClaudeAgentOptions=FakeClaudeAgentOptions,
        HookMatcher=FakeHookMatcher,
    )
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_module)
    client = AegisGatewayClient(
        "http://aegis.test",
        "aegk_dashboard_key",
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
    )
    provider = AegisClaudeGatewayProvider(client, matcher="^mcp__", timeout=90)
    existing_pre = object()
    existing_stop = object()

    options = provider.get_options(
        hooks={"PreToolUse": [existing_pre], "Stop": [existing_stop]},
        max_turns=3,
    )

    assert options.max_turns == 3
    assert options.hooks["PreToolUse"][0].hooks == [provider.hooks.pre_tool_use]
    assert options.hooks["PreToolUse"][0].matcher == "^mcp__"
    assert options.hooks["PreToolUse"][0].timeout == 90
    assert options.hooks["PreToolUse"][1] is existing_pre
    assert options.hooks["PostToolUse"][0].hooks == [provider.hooks.post_tool_use]
    assert options.hooks["Stop"] == [existing_stop]


def _gateway_decision_payload(
    *,
    audit_id: str,
    evaluation_phase: str = "input",
    allowed: bool = True,
    decision: str = "allow",
    violations: list[str] | None = None,
    result_classification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "action": {
            "context": {
                "agent_name": "agent-1",
                "tool_name": "get_weather",
                "tool_args": {"city": "London"},
                "user_id": None,
                "session_id": "session-1",
                "has_human_approval": False,
                "timestamp": "2026-05-01T12:00:00Z",
            },
            "action_types": ["aegis:Action"],
            "risk_level": "aegis:LowRisk",
            "involves_amount": None,
            "target_entity": None,
            "reasoning": "gateway test",
        },
        "allowed": allowed,
        "violations": violations or [],
        "violated_shapes": violations or [],
        "decision": decision,
        "evaluation_phase": evaluation_phase,
        "action_id": audit_id,
        "result_classification": result_classification,
        "classify_ms": 1.0,
        "validate_ms": 1.0,
        "total_ms": 2.0,
        "audit_id": audit_id,
    }

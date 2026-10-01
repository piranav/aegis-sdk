"""Tests for the API-keyed gateway client."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from aegis_sdk import ActionContext, AegisGatewayClient, AegisGatewayError


def test_evaluate_sends_bearer_key_and_parses_decision() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers["authorization"]
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_decision_payload(audit_id="audit-input-1"))

    client = AegisGatewayClient(
        "http://aegis.test",
        "aegk_customer_key",
        transport=httpx.MockTransport(handler),
    )

    decision = client.evaluate(
        ActionContext(
            agent_name="customer-agent",
            tool_name="lookup_balance",
            tool_args={"account_id": "acct_123"},
            timestamp=datetime(2026, 5, 1, 12, 0, tzinfo=UTC),
        )
    )

    assert seen["authorization"] == "Bearer aegk_customer_key"
    assert seen["path"] == "/v1/evaluate"
    assert seen["body"] == {
        "agent_name": "customer-agent",
        "tool_name": "lookup_balance",
        "tool_args": {"account_id": "acct_123"},
        "user_id": None,
        "session_id": None,
        "has_human_approval": False,
        "timestamp": "2026-05-01T12:00:00Z",
    }
    assert decision.audit_id == "audit-input-1"
    assert decision.decision == "allow"


def test_evaluate_result_sends_audit_correlation_payload() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_decision_payload(audit_id="audit-input-1"))

    client = AegisGatewayClient(
        "http://aegis.test/",
        "aegk_customer_key",
        transport=httpx.MockTransport(handler),
    )

    decision = client.evaluate_result(
        "audit-input-1",
        "customer balance is $10",
        {"rows": 1},
    )

    assert seen["path"] == "/v1/evaluate_result"
    assert seen["body"] == {
        "audit_id": "audit-input-1",
        "result": "customer balance is $10",
        "result_metadata": {"rows": 1},
    }
    assert decision.audit_id == "audit-input-1"


@pytest.mark.parametrize("status_code", [401, 403, 429])
def test_gateway_errors_include_status_and_detail(status_code: int) -> None:
    client = AegisGatewayClient(
        "http://aegis.test",
        "aegk_bad_key",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status_code, json={"detail": "Rejected"})
        ),
    )

    with pytest.raises(AegisGatewayError) as exc_info:
        client.evaluate({"agent_name": "a", "tool_name": "t", "tool_args": {}})

    assert exc_info.value.status_code == status_code
    assert str(exc_info.value) == "Rejected"


def test_transport_timeout_raises_gateway_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = AegisGatewayClient(
        "http://aegis.test",
        "aegk_timeout",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(AegisGatewayError) as exc_info:
        client.evaluate({"agent_name": "a", "tool_name": "t", "tool_args": {}})

    assert "timed out" in str(exc_info.value).lower()


def test_from_env_builds_client(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers["authorization"]
        return httpx.Response(200, json=_decision_payload(audit_id="audit-input-1"))

    monkeypatch.setenv("AEGIS_API_URL", "http://aegis.test/")
    monkeypatch.setenv("AEGIS_API_KEY", "aegk_env_key")
    monkeypatch.setenv("AEGIS_GATEWAY_TIMEOUT_SECONDS", "45")

    client = AegisGatewayClient.from_env(transport=httpx.MockTransport(handler))
    decision = client.evaluate({"agent_name": "a", "tool_name": "t", "tool_args": {}})

    assert decision.audit_id == "audit-input-1"
    assert seen["url"] == "http://aegis.test/v1/evaluate"
    assert seen["authorization"] == "Bearer aegk_env_key"


def test_from_env_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AEGIS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="AEGIS_API_KEY is required"):
        AegisGatewayClient.from_env()


def _decision_payload(*, audit_id: str) -> dict:
    return {
        "action": {
            "context": {
                "agent_name": "customer-agent",
                "tool_name": "lookup_balance",
                "tool_args": {"account_id": "acct_123"},
                "user_id": None,
                "session_id": None,
                "has_human_approval": False,
                "timestamp": "2026-05-01T12:00:00Z",
            },
            "action_types": ["aegis:Action"],
            "risk_level": "aegis:LowRisk",
            "involves_amount": None,
            "target_entity": None,
            "reasoning": "test",
        },
        "allowed": True,
        "violations": [],
        "violated_shapes": [],
        "decision": "allow",
        "evaluation_phase": "input",
        "action_id": audit_id,
        "result_classification": None,
        "classify_ms": 1.0,
        "validate_ms": 1.0,
        "total_ms": 2.0,
        "audit_id": audit_id,
    }

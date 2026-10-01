"""HTTP client for the hosted Aegis API gateway."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import Field

from aegis_sdk.types import ActionContext, GovernanceDecision


class GatewayEvaluationResponse(GovernanceDecision):
    """Decision response returned by the API gateway."""

    audit_id: str = Field(description="Audit id used to correlate result evaluation")


class AegisGatewayError(RuntimeError):
    """Raised when the Aegis gateway rejects or cannot process a request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response: httpx.Response | None = None,
    ) -> None:
        self.status_code = status_code
        self.response = response
        super().__init__(message)


class AegisGatewayClient:
    """Small synchronous client for API-keyed Aegis gateway evaluation.

    The dashboard-generated agent API key is sent as a bearer token. Full keys
    are never persisted by the gateway and should be supplied from the agent
    runtime environment.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float | httpx.Timeout = 30.0,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not api_key:
            raise ValueError("api_key is required")
        if client is not None and transport is not None:
            raise ValueError("Pass either client or transport, not both")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout, transport=transport)

    @classmethod
    def from_env(
        cls,
        *,
        base_url_env: str = "AEGIS_API_URL",
        api_key_env: str = "AEGIS_API_KEY",
        timeout_env: str = "AEGIS_GATEWAY_TIMEOUT_SECONDS",
        default_base_url: str = "http://127.0.0.1:8080",
        timeout: float | httpx.Timeout | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> AegisGatewayClient:
        """Build a gateway client from the standard agent-runtime environment."""

        base_url = os.getenv(base_url_env, default_base_url)
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} is required")
        resolved_timeout = timeout
        if resolved_timeout is None:
            resolved_timeout = float(os.getenv(timeout_env, "30"))
        return cls(
            base_url,
            api_key,
            timeout=resolved_timeout,
            client=client,
            transport=transport,
        )

    def __enter__(self) -> AegisGatewayClient:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def evaluate(self, action: ActionContext | Mapping[str, Any]) -> GatewayEvaluationResponse:
        """Evaluate an agent tool call through ``POST /v1/evaluate``."""

        payload = _action_payload(action)
        response = self._request("POST", "/v1/evaluate", json=payload)
        return GatewayEvaluationResponse.model_validate(response.json())

    def evaluate_result(
        self,
        audit_id: str,
        result: str,
        result_metadata: Mapping[str, Any] | None = None,
    ) -> GatewayEvaluationResponse:
        """Evaluate a tool result through ``POST /v1/evaluate_result``."""

        response = self._request(
            "POST",
            "/v1/evaluate_result",
            json={
                "audit_id": audit_id,
                "result": result,
                "result_metadata": dict(result_metadata or {}),
            },
        )
        return GatewayEvaluationResponse.model_validate(response.json())

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers = kwargs.pop("headers", {})
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **headers,
        }
        try:
            response = self._client.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                **kwargs,
            )
        except httpx.TimeoutException as exc:
            raise AegisGatewayError(
                f"Aegis gateway timed out calling {path}",
            ) from exc
        except httpx.RequestError as exc:
            raise AegisGatewayError(
                f"Aegis gateway request failed calling {path}: {exc}",
            ) from exc
        if response.is_error:
            raise AegisGatewayError(
                _error_message(response),
                status_code=response.status_code,
                response=response,
            )
        return response


def _action_payload(action: ActionContext | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(action, ActionContext):
        return action.model_dump(mode="json")
    return dict(action)


def _error_message(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    if isinstance(detail, str) and detail:
        return detail
    return f"Aegis gateway request failed with HTTP {response.status_code}"

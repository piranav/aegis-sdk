"""Client-side SDK for the hosted Aegis governance gateway."""

from aegis_sdk.client import AegisGatewayClient, GatewayEvaluationResponse
from aegis_sdk.errors import AegisGatewayError
from aegis_sdk.inventory import AegisInventory, Component
from aegis_sdk.telemetry import AegisTelemetry, TokenUsage, bind_session
from aegis_sdk.types import (
    ActionContext,
    ClassifiedAction,
    DecisionVerdict,
    EvaluationPhase,
    GovernanceDecision,
    ResultClassification,
)

__all__ = [
    "AegisGatewayClient",
    "AegisInventory",
    "AegisTelemetry",
    "Component",
    "TokenUsage",
    "bind_session",
    "AegisGatewayError",
    "GatewayEvaluationResponse",
    "ActionContext",
    "ClassifiedAction",
    "DecisionVerdict",
    "EvaluationPhase",
    "GovernanceDecision",
    "ResultClassification",
]

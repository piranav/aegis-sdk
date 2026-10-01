"""Client-side SDK for the hosted Aegis governance gateway."""

from aegis_sdk.client import AegisGatewayClient, AegisGatewayError, GatewayEvaluationResponse
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
    "AegisGatewayError",
    "GatewayEvaluationResponse",
    "ActionContext",
    "ClassifiedAction",
    "DecisionVerdict",
    "EvaluationPhase",
    "GovernanceDecision",
    "ResultClassification",
]

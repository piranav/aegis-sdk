"""OpenAI Agents SDK guardrails backed by the Aegis API gateway."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from agents import (
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailData,
    ToolOutputGuardrailData,
    tool_input_guardrail,
    tool_output_guardrail,
)

from aegis_sdk.client import AegisGatewayClient, AegisGatewayError, GatewayEvaluationResponse
from aegis_sdk.integrations.openai_agents.run_scope import run_scope
from aegis_sdk.types import ActionContext

UserIdResolver = Callable[[ToolInputGuardrailData | ToolOutputGuardrailData], str | None]
_STATE_ATTR = "_aegis_gateway_guardrail_state"
_RAW_RESULT_ATTR = "_aegis_raw_tool_result"
_OUTPUT_DECISION_ATTR = "_aegis_output_decision"


def make_aegis_gateway_guardrail(
    client: AegisGatewayClient,
    *,
    user_id_resolver: UserIdResolver | None = None,
):
    """Create an OpenAI tool input guardrail that calls ``/v1/evaluate``."""

    @tool_input_guardrail(name="aegis-gateway-governance")
    async def aegis_gateway_check(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        ctx = _build_action_context(data, user_id_resolver)
        try:
            decision = await asyncio.to_thread(client.evaluate, ctx)
        except AegisGatewayError as exc:
            return ToolGuardrailFunctionOutput.reject_content(
                message=f"[AEGIS] DENY: {exc}",
                output_info={"decision": "deny", "phase": "input", "error": str(exc)},
            )

        _store_input_state(data.context, ctx, decision)
        info = _decision_info(decision)
        if decision.allowed:
            return ToolGuardrailFunctionOutput.allow(output_info=info)

        violation_summary = "; ".join(decision.violations[:3]) or decision.decision.value
        return ToolGuardrailFunctionOutput.reject_content(
            message=f"[AEGIS] {decision.decision.value.upper()}: {violation_summary}",
            output_info=info,
        )

    return aegis_gateway_check


def make_aegis_gateway_output_guardrail(
    client: AegisGatewayClient,
    *,
    user_id_resolver: UserIdResolver | None = None,
):
    """Create an OpenAI tool output guardrail that calls ``/v1/evaluate_result``."""

    @tool_output_guardrail(name="aegis-gateway-result-governance")
    async def aegis_gateway_output_check(
        data: ToolOutputGuardrailData,
    ) -> ToolGuardrailFunctionOutput:
        state = getattr(data.context, _STATE_ATTR, None)
        result_text = _safe_text(data.output)
        setattr(data.context, _RAW_RESULT_ATTR, result_text)

        if state is None:
            info = {
                "decision": "deny",
                "phase": "output",
                "violations": ["Missing Aegis gateway input evaluation state -- fail-closed"],
            }
            return ToolGuardrailFunctionOutput.reject_content(
                message=(
                    "[AEGIS] DENY: Missing Aegis gateway input evaluation state -- "
                    "tool result suppressed."
                ),
                output_info=info,
            )

        ctx = state["action_context"]
        if user_id_resolver and ctx.user_id is None:
            resolved_user_id = user_id_resolver(data)
            if resolved_user_id is not None:
                ctx.user_id = resolved_user_id

        try:
            decision = await asyncio.to_thread(
                client.evaluate_result,
                state["audit_id"],
                result_text,
                {
                    "tool_call_id": getattr(data.context, "tool_call_id", None),
                    "tool_name": getattr(data.context, "tool_name", None),
                },
            )
        except AegisGatewayError as exc:
            return ToolGuardrailFunctionOutput.reject_content(
                message=f"[AEGIS] DENY: {exc}",
                output_info={"decision": "deny", "phase": "output", "error": str(exc)},
            )

        setattr(data.context, _OUTPUT_DECISION_ATTR, decision)
        info = _decision_info(decision)
        info["input_audit_id"] = state["audit_id"]

        if decision.allowed:
            return ToolGuardrailFunctionOutput.allow(output_info=info)

        violation_summary = "; ".join(decision.violations[:3]) or decision.decision.value
        return ToolGuardrailFunctionOutput.reject_content(
            message=(
                f"[AEGIS] {decision.decision.value.upper()}: {violation_summary} "
                "Tool result suppressed by Aegis."
            ),
            output_info=info,
        )

    return aegis_gateway_output_check


def _build_action_context(
    data: ToolInputGuardrailData,
    user_id_resolver: UserIdResolver | None,
) -> ActionContext:
    user_id = user_id_resolver(data) if user_id_resolver else None
    return ActionContext(
        agent_name=data.agent.name,
        tool_name=data.context.tool_name,
        tool_args=_parse_tool_args(data.context.tool_arguments),
        user_id=user_id,
        # Ties the governed tool call to the run's Aegis session.
        session_id=run_scope(data.context).session_id,
    )


def _parse_tool_args(raw_args: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw_args}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def _store_input_state(
    tool_context: Any,
    ctx: ActionContext,
    decision: GatewayEvaluationResponse,
) -> None:
    setattr(
        tool_context,
        _STATE_ATTR,
        {
            "action_context": ctx,
            "input_decision": decision,
            "audit_id": decision.audit_id,
        },
    )


def _decision_info(decision: GatewayEvaluationResponse) -> dict[str, Any]:
    info: dict[str, Any] = {
        "classification": decision.action.action_types,
        "risk": decision.action.risk_level,
        "decision": decision.decision.value,
        "phase": decision.evaluation_phase.value,
        "action_id": decision.action_id,
        "audit_id": decision.audit_id,
        "classify_ms": decision.classify_ms,
        "validate_ms": decision.validate_ms,
        "total_ms": decision.total_ms,
    }
    if decision.violations:
        info["violations"] = decision.violations
    if decision.violated_shapes:
        info["shapes"] = decision.violated_shapes
    if decision.result_classification is not None:
        info["result_classification"] = decision.result_classification.model_dump(mode="json")
    return info


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)

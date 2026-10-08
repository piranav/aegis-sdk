"""Claude Agent SDK hooks backed exclusively by the hosted Aegis gateway."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from aegis_sdk.client import AegisGatewayClient, AegisGatewayError, GatewayEvaluationResponse
from aegis_sdk.integrations.claude_agent.telemetry import ClaudeTelemetryObserver
from aegis_sdk.telemetry import AegisTelemetry
from aegis_sdk.types import ActionContext, GovernanceDecision

logger = logging.getLogger(__name__)
HookOutput = dict[str, Any]
ClaudeHookContext = Any
UserIdResolver = Callable[[Mapping[str, Any], str | None, ClaudeHookContext], str | None]
_DEFAULT_AGENT_NAME = "claude-agent"
_UNKNOWN_SESSION = "__unknown_session__"
_UNKNOWN_TOOL_USE = "__unknown_tool_use__"
_SAFE_REPLACEMENT = "[AEGIS] Tool result suppressed by Aegis governance."


@dataclass(frozen=True)
class StoredGatewayToolUse:
    """Gateway state needed to correlate input and output evaluations."""

    action_context: ActionContext
    audit_id: str
    input_decision: GatewayEvaluationResponse


class AegisClaudeGatewayHooks:
    """Claude Agent SDK hooks backed by the hosted Aegis API gateway."""

    def __init__(
        self,
        client: AegisGatewayClient,
        *,
        user_id_resolver: UserIdResolver | None = None,
        telemetry: AegisTelemetry | None = None,
    ) -> None:
        self.client = client
        self.user_id_resolver = user_id_resolver
        self.telemetry_observer = ClaudeTelemetryObserver(telemetry) if telemetry else None
        self._tool_uses: dict[tuple[str, str], StoredGatewayToolUse] = {}
        self.trace: list[dict[str, Any]] = []

    async def user_prompt_submit(
        self,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: ClaudeHookContext,
    ) -> HookOutput:
        """Open the session for this turn with the user's prompt as its input."""

        session_id = input_data.get("session_id")
        if self.telemetry_observer is not None and session_id:
            self.telemetry_observer.start_session(session_id, prompt=input_data.get("prompt"))
        return {}

    async def pre_tool_use(
        self,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: ClaudeHookContext,
    ) -> HookOutput:
        """Evaluate a Claude tool invocation through ``/v1/evaluate``."""

        ctx = _build_action_context(input_data, tool_use_id, context, self.user_id_resolver)
        key = _state_key(input_data, tool_use_id)
        try:
            decision = await asyncio.to_thread(self.client.evaluate, ctx)
        except AegisGatewayError as exc:
            reason = f"[AEGIS] DENY: {exc}"
            self._append_trace(
                {
                    "event": "pre_tool_use",
                    "session_id": key[0],
                    "tool_use_id": key[1],
                    "agent": ctx.agent_name,
                    "tool": ctx.tool_name,
                    "tool_args": ctx.tool_args,
                    "input_decision": "deny",
                    "input_allowed": False,
                    "error": str(exc),
                }
            )
            return _pre_tool_violation_output(reason)

        self._tool_uses[key] = StoredGatewayToolUse(
            action_context=ctx,
            audit_id=decision.audit_id,
            input_decision=decision,
        )
        self._append_trace(
            {
                "event": "pre_tool_use",
                "session_id": key[0],
                "tool_use_id": key[1],
                "agent": ctx.agent_name,
                "tool": ctx.tool_name,
                "tool_args": ctx.tool_args,
                "input_action_id": decision.action_id,
                "input_audit_id": decision.audit_id,
                "input_decision": decision.decision.value,
                "input_allowed": decision.allowed,
                "classify_ms": decision.classify_ms,
                "validate_ms": decision.validate_ms,
                "total_ms": decision.total_ms,
            }
        )

        if decision.allowed:
            return {}

        return _pre_tool_violation_output(_format_reason(decision))

    async def post_tool_use(
        self,
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: ClaudeHookContext,
    ) -> HookOutput:
        """Evaluate a Claude tool result through ``/v1/evaluate_result``."""

        key = _state_key(input_data, tool_use_id)
        stored = self._tool_uses.pop(key, None)
        if stored is None:
            reason = "[AEGIS] Unable to evaluate tool result: missing input evaluation state"
            logger.warning(
                "Claude gateway PostToolUse missing input state for session=%s tool_use_id=%s",
                key[0],
                key[1],
            )
            return _post_tool_violation_output(reason)

        result_text = _safe_text(input_data.get("tool_response"))
        metadata = {
            "session_id": input_data.get("session_id"),
            "tool_use_id": input_data.get("tool_use_id") or tool_use_id,
            "tool_name": input_data.get("tool_name"),
            "hook_event_name": input_data.get("hook_event_name"),
        }
        try:
            output_decision = await asyncio.to_thread(
                self.client.evaluate_result,
                stored.audit_id,
                result_text,
                metadata,
            )
        except AegisGatewayError as exc:
            reason = f"[AEGIS] DENY: {exc}"
            self._append_trace(
                {
                    "event": "post_tool_use",
                    "session_id": key[0],
                    "tool_use_id": key[1],
                    "agent": stored.action_context.agent_name,
                    "tool": stored.action_context.tool_name,
                    "result_text": result_text,
                    "input_audit_id": stored.audit_id,
                    "output_decision": "deny",
                    "output_allowed": False,
                    "error": str(exc),
                }
            )
            return _post_tool_violation_output(reason)

        self._append_trace(
            {
                "event": "post_tool_use",
                "session_id": key[0],
                "tool_use_id": key[1],
                "agent": stored.action_context.agent_name,
                "tool": stored.action_context.tool_name,
                "result_text": result_text,
                "input_action_id": stored.input_decision.action_id,
                "input_audit_id": stored.audit_id,
                "output_action_id": output_decision.action_id,
                "output_audit_id": output_decision.audit_id,
                "output_decision": output_decision.decision.value,
                "output_allowed": output_decision.allowed,
                "output_violations": output_decision.violations,
                "classify_ms": output_decision.classify_ms,
                "validate_ms": output_decision.validate_ms,
                "total_ms": output_decision.total_ms,
                "output_classification": (
                    output_decision.result_classification.model_dump()
                    if output_decision.result_classification is not None
                    else None
                ),
            }
        )

        if output_decision.allowed:
            return {}

        return _post_tool_violation_output(_format_reason(output_decision))

    def get_stored_tool_use(
        self,
        *,
        session_id: str | None,
        tool_use_id: str | None,
    ) -> StoredGatewayToolUse | None:
        """Return pending gateway state for tests and observability."""

        return self._tool_uses.get(
            (session_id or _UNKNOWN_SESSION, tool_use_id or _UNKNOWN_TOOL_USE)
        )

    def get_trace(self) -> list[dict[str, Any]]:
        """Return a shallow copy of Claude gateway hook trace entries."""

        return list(self.trace)

    def _append_trace(self, entry: dict[str, Any]) -> None:
        entry["ts"] = datetime.now(UTC).isoformat()
        self.trace.append(entry)


def make_aegis_claude_gateway_hooks(
    client: AegisGatewayClient,
    *,
    user_id_resolver: UserIdResolver | None = None,
    telemetry: AegisTelemetry | None = None,
) -> AegisClaudeGatewayHooks:
    """Create stateful Claude hook callbacks for the Aegis API gateway."""

    return AegisClaudeGatewayHooks(client, user_id_resolver=user_id_resolver, telemetry=telemetry)


def _build_action_context(
    input_data: Mapping[str, Any],
    tool_use_id: str | None,
    context: ClaudeHookContext,
    user_id_resolver: UserIdResolver | None,
) -> ActionContext:
    user_id = user_id_resolver(input_data, tool_use_id, context) if user_id_resolver else None
    agent_name = input_data.get("agent_id") or input_data.get("agent_type") or _DEFAULT_AGENT_NAME
    tool_input = input_data.get("tool_input")

    return ActionContext(
        agent_name=str(agent_name),
        tool_name=str(input_data.get("tool_name") or ""),
        tool_args=tool_input if isinstance(tool_input, dict) else {},
        user_id=user_id,
        session_id=input_data.get("session_id"),
    )


def _state_key(
    input_data: Mapping[str, Any],
    tool_use_id: str | None,
) -> tuple[str, str]:
    return (
        str(input_data.get("session_id") or _UNKNOWN_SESSION),
        str(input_data.get("tool_use_id") or tool_use_id or _UNKNOWN_TOOL_USE),
    )


def _pre_tool_violation_output(reason: str) -> HookOutput:
    return {
        "suppressOutput": True,
        "systemMessage": reason,
        "reason": reason,
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
    }


def _format_reason(decision: GovernanceDecision) -> str:
    violations = "; ".join(decision.violations[:3]) or decision.decision.value
    return f"[AEGIS] {decision.decision.value.upper()}: {violations}"


def _post_tool_violation_output(reason: str) -> HookOutput:
    return {
        "suppressOutput": True,
        "systemMessage": f"{reason}. Tool result suppressed by Aegis governance.",
        "decision": "block",
        "reason": reason,
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "updatedToolOutput": {
                "content": [
                    {
                        "type": "text",
                        "text": f"{reason}. {_SAFE_REPLACEMENT}",
                    }
                ],
                "isError": True,
            },
            "additionalContext": (
                "Aegis governance suppressed this tool result because it may violate "
                "policy. Do not reveal, quote, summarize, or rely on the suppressed "
                "content."
            ),
        },
    }


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)

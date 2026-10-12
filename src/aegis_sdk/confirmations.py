"""Confirming escalated tool calls with the person using the agent, in the conversation.

When an organization rule holds a tool call for review, the gateway answers with a
question. The integration blocks the call and tells the agent to ask its user that
question. On the user's next message the integration, not the model, sends that message
to the gateway as their reply; a clear yes lets the agent's retry of the same call run once.
No code is needed from the agent's developer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from aegis_sdk.client import AegisGatewayClient, GatewayConfirmation
from aegis_sdk.errors import AegisGatewayError
from aegis_sdk.types import ActionContext

Resolution = Literal["none", "waiting", "approved", "declined"]

DECLINED_MESSAGE = (
    "[AEGIS] The user declined this action, so it was not run. Do not run it again; tell "
    "the user it was not done."
)


@dataclass
class PendingConfirmation:
    confirmation: GatewayConfirmation
    # The user's latest message when the question was asked; their reply comes after it.
    asked_after: str | None


class ConfirmationTracker:
    """Calls waiting on their user's answer, keyed by agent, user, tool, and arguments."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingConfirmation] = {}

    def remember(
        self, ctx: ActionContext, confirmation: GatewayConfirmation, latest_message: str | None
    ) -> None:
        self._pending[_call_key(ctx)] = PendingConfirmation(confirmation, latest_message)

    def resolve(
        self, client: AegisGatewayClient, ctx: ActionContext, latest_message: str | None
    ) -> tuple[Resolution, str | None]:
        """Relay the user's newest message for a pending call. Returns the outcome and,
        when approved, the confirmation id to evaluate the retry with."""
        key = _call_key(ctx)
        pending = self._pending.get(key)
        if pending is None:
            return "none", None
        if not latest_message or latest_message == pending.asked_after:
            return "waiting", None  # The user hasn't answered yet.
        try:
            reply = client.reply_to_confirmation(
                pending.confirmation.id, latest_message, user_id=ctx.user_id
            )
        except AegisGatewayError:
            self._pending.pop(key, None)  # Expired or unknown: ask again.
            return "none", None
        if reply.get("status") == "approved":
            self._pending.pop(key, None)
            return "approved", pending.confirmation.id
        if reply.get("status") == "declined":
            self._pending.pop(key, None)
            return "declined", None
        pending.asked_after = latest_message  # Not a clear answer; wait for the next one.
        return "waiting", None

    def answered_already(
        self,
        client: AegisGatewayClient,
        ctx: ActionContext,
        confirmation: GatewayConfirmation,
        latest_message: str | None,
        agent_question: str | None,
    ) -> str | None:
        """When the agent asked in its own words before calling the tool and the user
        answered, relay both so the user isn't asked twice. Aegis approves only if the
        question named this action and the answer agrees; returns the confirmation id then."""
        if not latest_message or not agent_question or "?" not in agent_question:
            return None
        try:
            reply = client.reply_to_confirmation(
                confirmation.id, latest_message, user_id=ctx.user_id, agent_question=agent_question
            )
        except AegisGatewayError:
            return None
        return confirmation.id if reply.get("status") == "approved" else None


def _text(item: Any) -> str | None:
    content = item.get("content") if isinstance(item, dict) else getattr(item, "content", "")
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts) or None


def _role(item: Any) -> str | None:
    return item.get("role") if isinstance(item, dict) else getattr(item, "role", None)


def previous_agent_message(items: Any) -> str | None:
    """The agent's message just before the user's latest one: a question it may have asked."""
    if isinstance(items, str):
        return None
    seen_user = False
    for item in reversed(list(items or [])):
        role = _role(item)
        if role == "user":
            if seen_user:
                return None
            seen_user = True
        elif role == "assistant" and seen_user:
            return _text(item)
    return None


def latest_user_message(items: Any) -> str | None:
    """The text of the last user message in an OpenAI-style list of input items."""
    if isinstance(items, str):
        return items
    for item in reversed(list(items or [])):
        if _role(item) == "user":
            return _text(item)
    return None


def _call_key(ctx: ActionContext) -> str:
    canonical = json.dumps(
        [ctx.agent_name, ctx.user_id, ctx.tool_name, ctx.tool_args], sort_keys=True, default=str
    )
    return hashlib.sha256(canonical.encode()).hexdigest()

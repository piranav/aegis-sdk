"""Attach Aegis gateway governance to Claude Agent SDK options."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from aegis_sdk.client import AegisGatewayClient
from aegis_sdk.integrations.claude_agent.hooks import AegisClaudeGatewayHooks, UserIdResolver
from aegis_sdk.integrations.claude_agent.inventory import ClaudeInventoryObserver
from aegis_sdk.inventory import AegisInventory
from aegis_sdk.telemetry import AegisTelemetry


class AegisClaudeGatewayProvider:
    """Build Claude Agent SDK options with API-keyed Aegis gateway hooks.

    With ``telemetry`` or ``inventory``, wrap the message stream in ``instrument`` to
    report sessions, model calls, token usage, and Claude's reported cost, and each
    session's components (model, tools, MCP servers, skills, subagents, plugins)::

        async for message in provider.instrument(client.receive_response()):
            ...
    """

    def __init__(
        self,
        client: AegisGatewayClient,
        *,
        matcher: str | None = None,
        timeout: int = 60,
        user_id_resolver: UserIdResolver | None = None,
        telemetry: AegisTelemetry | None = None,
        inventory: AegisInventory | None = None,
    ) -> None:
        self.client = client
        self.matcher = matcher
        self.timeout = timeout
        self.hooks = AegisClaudeGatewayHooks(
            client, user_id_resolver=user_id_resolver, telemetry=telemetry
        )
        self.inventory_observer = ClaudeInventoryObserver(inventory) if inventory else None

    @classmethod
    def from_env(
        cls,
        *,
        matcher: str | None = None,
        timeout: int = 60,
        user_id_resolver: UserIdResolver | None = None,
        telemetry: bool = False,
        inventory: bool = False,
    ) -> AegisClaudeGatewayProvider:
        """Build a gateway provider from ``AEGIS_API_URL`` and ``AEGIS_API_KEY``."""

        client = AegisGatewayClient.from_env()
        return cls(
            client,
            matcher=matcher,
            timeout=timeout,
            user_id_resolver=user_id_resolver,
            telemetry=AegisTelemetry.from_client(client) if telemetry else None,
            inventory=AegisInventory.from_client(client) if inventory else None,
        )

    def get_options(self, **claude_options: Any) -> Any:
        """Return ``ClaudeAgentOptions`` with gateway hooks prepended."""

        try:
            from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
        except ImportError as exc:  # pragma: no cover - message tested via unit fallback
            raise ImportError(
                "claude-agent-sdk is required for AegisClaudeGatewayProvider. "
                "Install it with `pip install -e '.[claude-agent]'`."
            ) from exc

        existing_hooks = claude_options.pop("hooks", None) or {}
        merged_hooks = {event: list(matchers) for event, matchers in existing_hooks.items()}

        pre_matcher = self._make_matcher(HookMatcher, self.hooks.pre_tool_use)
        post_matcher = self._make_matcher(HookMatcher, self.hooks.post_tool_use)

        merged_hooks["PreToolUse"] = [pre_matcher, *merged_hooks.get("PreToolUse", [])]
        merged_hooks["PostToolUse"] = [post_matcher, *merged_hooks.get("PostToolUse", [])]
        if self.hooks.telemetry_observer is not None:
            # Prompt hooks ignore the tool-name matcher; they fire for every turn.
            prompt_matcher = HookMatcher(
                hooks=[self.hooks.user_prompt_submit], timeout=self.timeout
            )
            merged_hooks["UserPromptSubmit"] = [
                prompt_matcher,
                *merged_hooks.get("UserPromptSubmit", []),
            ]

        return ClaudeAgentOptions(hooks=merged_hooks, **claude_options)

    async def instrument(self, messages: AsyncIterable[Any]) -> AsyncIterator[Any]:
        """Yield ``messages`` unchanged, reporting telemetry when it is enabled."""

        observer = self.hooks.telemetry_observer
        stream = observer.instrument(messages) if observer is not None else messages
        async for message in stream:
            if self.inventory_observer is not None:
                self.inventory_observer.observe(message)
            yield message

    def _make_matcher(self, hook_matcher_type: Any, callback: Any) -> Any:
        kwargs = {"hooks": [callback], "timeout": self.timeout}
        if self.matcher is not None:
            kwargs["matcher"] = self.matcher
        return hook_matcher_type(**kwargs)

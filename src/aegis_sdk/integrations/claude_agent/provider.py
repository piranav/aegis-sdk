"""Attach Aegis gateway governance to Claude Agent SDK options."""

from __future__ import annotations

from typing import Any

from aegis_sdk.client import AegisGatewayClient
from aegis_sdk.integrations.claude_agent.hooks import AegisClaudeGatewayHooks, UserIdResolver


class AegisClaudeGatewayProvider:
    """Build Claude Agent SDK options with API-keyed Aegis gateway hooks."""

    def __init__(
        self,
        client: AegisGatewayClient,
        *,
        matcher: str | None = None,
        timeout: int = 60,
        user_id_resolver: UserIdResolver | None = None,
    ) -> None:
        self.client = client
        self.matcher = matcher
        self.timeout = timeout
        self.hooks = AegisClaudeGatewayHooks(client, user_id_resolver=user_id_resolver)

    @classmethod
    def from_env(
        cls,
        *,
        matcher: str | None = None,
        timeout: int = 60,
        user_id_resolver: UserIdResolver | None = None,
    ) -> AegisClaudeGatewayProvider:
        """Build a gateway provider from ``AEGIS_API_URL`` and ``AEGIS_API_KEY``."""

        return cls(
            AegisGatewayClient.from_env(),
            matcher=matcher,
            timeout=timeout,
            user_id_resolver=user_id_resolver,
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

        return ClaudeAgentOptions(hooks=merged_hooks, **claude_options)

    def _make_matcher(self, hook_matcher_type: Any, callback: Any) -> Any:
        kwargs = {"hooks": [callback], "timeout": self.timeout}
        if self.matcher is not None:
            kwargs["matcher"] = self.matcher
        return hook_matcher_type(**kwargs)

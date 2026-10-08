"""Claude Agent SDK gateway hooks and provider."""

from aegis_sdk.integrations.claude_agent.hooks import (
    AegisClaudeGatewayHooks,
    StoredGatewayToolUse,
    make_aegis_claude_gateway_hooks,
)
from aegis_sdk.integrations.claude_agent.inventory import ClaudeInventoryObserver
from aegis_sdk.integrations.claude_agent.provider import AegisClaudeGatewayProvider
from aegis_sdk.integrations.claude_agent.telemetry import ClaudeTelemetryObserver

__all__ = [
    "AegisClaudeGatewayHooks",
    "AegisClaudeGatewayProvider",
    "ClaudeInventoryObserver",
    "ClaudeTelemetryObserver",
    "StoredGatewayToolUse",
    "make_aegis_claude_gateway_hooks",
]

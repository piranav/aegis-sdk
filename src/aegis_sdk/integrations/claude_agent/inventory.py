"""Describe a Claude Agent SDK session's components from its ``init`` message.

Claude reports what a session loaded when it starts: tool names, MCP servers and their
connection status, the model, subagent types, skills, and plugins. That is the
session's bill of materials as the runtime actually resolved it (settings files,
plugins, and SDK options combined), so it is more faithful than reading options.

Usage needs no work here: every governed tool call reaches ``/v1/evaluate``, where
Aegis resolves ``mcp__<server>__<tool>``, ``Skill``, and ``Task`` calls itself.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from aegis_sdk.inventory import AegisInventory, Component, merge_components

FRAMEWORK = "claude-agent"
_MCP_PREFIX = "mcp__"


def mcp_server_key(name: str) -> str:
    """Claude names MCP tools ``mcp__<server>__<tool>`` with the server name normalized
    this way, so server components use the same form to line up with observed calls."""

    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def manifest_from_init(data: Mapping[str, Any]) -> list[Component]:
    components: list[Component] = []
    model = data.get("model")
    if isinstance(model, str) and model:
        components.append(Component.model(model, provider="anthropic"))

    servers: dict[str, Component] = {}
    for server in data.get("mcp_servers") or ():
        if not isinstance(server, Mapping) or not server.get("name"):
            continue
        key = mcp_server_key(str(server["name"]))
        attributes = {
            k: str(server[k]) for k in ("status", "source") if isinstance(server.get(k), str)
        }
        servers[key] = Component(kind="mcp_server", name=key, attributes=attributes)

    for tool in data.get("tools") or ():
        if not isinstance(tool, str) or not tool:
            continue
        if tool.startswith(_MCP_PREFIX):
            server, separator, name = tool[len(_MCP_PREFIX) :].partition("__")
            if separator and server and name:
                parent = servers.setdefault(server, Component(kind="mcp_server", name=server))
                parent.children.append(Component.tool(name))
                continue
        components.append(Component.tool(tool, attributes={"builtin": True}))
    components.extend(servers.values())

    plugins: dict[str, Component] = {}
    for plugin in data.get("plugins") or ():
        name = plugin.get("name") if isinstance(plugin, Mapping) else plugin
        if isinstance(name, str) and name:
            plugins[name] = Component(kind="plugin", name=name)
    for kind, key in (("skill", "skills"), ("subagent", "agents")):
        for entry in data.get(key) or ():
            if not isinstance(entry, str) or not entry:
                continue
            plugin, separator, name = entry.partition(":")
            if separator and plugin and name:
                holder = plugins.setdefault(plugin, Component(kind="plugin", name=plugin))
                holder.children.append(Component(kind=kind, name=name))
            else:
                components.append(Component(kind=kind, name=entry))
    components.extend(plugins.values())
    return merge_components(components)


class ClaudeInventoryObserver:
    """Report each session's manifest when its ``init`` message arrives.

    ``AegisInventory`` only sends a manifest when it differs from the last one, so
    observing every session (and every turn's init) costs one comparison.
    """

    def __init__(self, inventory: AegisInventory) -> None:
        self.inventory = inventory

    def observe(self, message: Any) -> None:
        if (
            type(message).__name__ != "SystemMessage"
            or getattr(message, "subtype", None) != "init"
        ):
            return
        data = getattr(message, "data", None)
        if isinstance(data, Mapping):
            self.inventory.report_manifest(manifest_from_init(data), framework=FRAMEWORK)

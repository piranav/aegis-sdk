"""Describe an OpenAI Agents SDK agent graph as Aegis inventory components.

The Aegis agent is the whole application behind one API key, so the bill of
materials is the union over every agent reachable through handoffs and agents-as-
tools: their models, function tools, MCP servers (with the tools each exposes),
hosted connectors, shell skills, and the subagents themselves. Which SDK agent holds a
component is kept as its ``agents`` attribute rather than as nesting, so components
match the usage Aegis observes through ``/v1/evaluate`` regardless of which agent ran.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Iterator
from pathlib import PurePath
from typing import Any

from agents import Agent
from agents.tool import (
    CodeInterpreterTool,
    FileSearchTool,
    FunctionTool,
    HostedMCPTool,
    ImageGenerationTool,
    ShellTool,
    ToolOriginType,
    WebSearchTool,
    get_function_tool_origin,
)

from aegis_sdk.inventory import Component, fingerprint, merge_components
from aegis_sdk.inventory.components import PathSegment

logger = logging.getLogger(__name__)

FRAMEWORK = "openai-agents"
_HOSTED_TOOL_TYPES = (
    WebSearchTool,
    FileSearchTool,
    CodeInterpreterTool,
    ImageGenerationTool,
)
# Response items for work the model provider performs itself. These never reach the
# agent runtime's tool hooks, so they are the only way to observe that usage.
_HOSTED_CALL_TOOLS = {
    "web_search_call": "web_search",
    "file_search_call": "file_search",
    "code_interpreter_call": "code_interpreter",
    "image_generation_call": "image_generation",
}


async def describe_agent(
    agent: Agent[Any],
    *,
    run_context: Any = None,
    list_mcp_tools: bool = True,
    mcp_timeout: float = 5.0,
) -> list[Component]:
    """Return the components of ``agent`` and every agent reachable from it.

    With ``list_mcp_tools``, connected MCP servers are asked for their tools so that a
    server changing its tool definitions shows up in Aegis. Servers that are not
    connected, or slow, are still reported, without tools.
    """

    components: list[Component] = []
    for current in _reachable_agents(agent):
        holder = {"agents": current.name}
        components.append(
            Component.model(_model_name(current), provider=_provider(current)).model_copy(
                update={"attributes": holder}
            )
        )
        if current is not agent:
            components.append(
                Component(
                    kind="subagent",
                    name=current.name,
                    description=_text(getattr(current, "handoff_description", None)),
                    fingerprint=fingerprint(_text(current.instructions), _model_name(current)),
                )
            )
        for tool in current.tools:
            component = _describe_tool(tool)
            if component is not None:
                components.append(_held_by(component, current.name))
        for server in current.mcp_servers:
            tools = (
                await _list_mcp_tools(server, run_context, current, mcp_timeout)
                if list_mcp_tools
                else []
            )
            components.append(
                _held_by(
                    Component.mcp_server(
                        server.name,
                        tools=tools,
                        transport=_mcp_transport(server),
                        locator=_mcp_locator(server),
                    ),
                    current.name,
                )
            )
    return _join_holders([*components, *shell_skills(agent)])


def tool_usage_path(tool: Any) -> list[PathSegment] | None:
    """The component a runtime tool call used, for tools Aegis does not gate itself."""

    if isinstance(tool, FunctionTool):
        origin = get_function_tool_origin(tool)
        if origin is not None and origin.type == ToolOriginType.MCP and origin.mcp_server_name:
            return [("mcp_server", origin.mcp_server_name), ("tool", tool.name)]
        if origin is not None and origin.type == ToolOriginType.AGENT_AS_TOOL:
            return [("subagent", origin.agent_name or tool.name)]
    name = getattr(tool, "name", None)
    return [("tool", name)] if isinstance(name, str) and name else None


def response_usage_paths(
    response: Any, *, skills: Iterable[Component] = ()
) -> Iterator[list[PathSegment]]:
    """Usage visible only in a model response: hosted MCP calls, hosted tools, skills.

    Skills are loaded by the model reading their files through the shell tool, so a
    shell command that touches a declared skill's directory counts as using it.
    """

    skill_dirs = {
        skill.name: skill.attributes.get("dir") or skill.name
        for skill in skills
        if skill.kind == "skill"
    }
    for item in getattr(response, "output", None) or ():
        item_type = _field(item, "type")
        if item_type == "mcp_call":
            label, name = _field(item, "server_label"), _field(item, "name")
            if label and name:
                yield [("connector", label), ("tool", name)]
        elif item_type in _HOSTED_CALL_TOOLS:
            yield [("tool", _HOSTED_CALL_TOOLS[item_type])]
        elif item_type == "shell_call" and skill_dirs:
            commands = " ".join(str(c) for c in (_field(_field(item, "action"), "commands") or ()))
            for skill, directory in skill_dirs.items():
                if directory and str(directory) in commands:
                    yield [("skill", skill)]


def iter_components(components: Iterable[Component]) -> Iterator[Component]:
    for component in components:
        yield component
        yield from iter_components(component.children)


# ------------------------------------------------------------------------------ helpers


def _reachable_agents(root: Agent[Any]) -> Iterator[Agent[Any]]:
    """Breadth-first over handoffs and agents-as-tools, each agent once."""

    seen: set[int] = set()
    queue: list[Agent[Any]] = [root]
    while queue:
        current = queue.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for handoff in getattr(current, "handoffs", ()):
            target = handoff if isinstance(handoff, Agent) else getattr(handoff, "agent", None)
            if isinstance(target, Agent):
                queue.append(target)
        for tool in current.tools:
            nested = getattr(tool, "_agent_instance", None)
            if isinstance(nested, Agent):
                queue.append(nested)


def _describe_tool(tool: Any) -> Component | None:
    if isinstance(tool, FunctionTool):
        if getattr(tool, "_agent_instance", None) is not None:
            return None  # described as a subagent
        return Component.tool(
            tool.name, description=tool.description, schema=tool.params_json_schema
        )
    if isinstance(tool, HostedMCPTool):
        return _describe_connector(tool)
    if isinstance(tool, ShellTool):
        return _describe_shell(tool)
    if isinstance(tool, _HOSTED_TOOL_TYPES):
        return Component.tool(tool.name, provider="openai", attributes={"hosted": True})
    name = getattr(tool, "name", None)
    return Component.tool(name) if isinstance(name, str) and name else None


def _describe_connector(tool: HostedMCPTool) -> Component:
    config = dict(tool.tool_config)
    allowed = config.get("allowed_tools")
    names = allowed if isinstance(allowed, list) else (allowed or {}).get("tool_names") or []
    attributes = {"connector_id": config["connector_id"]} if config.get("connector_id") else {}
    return Component(
        kind="connector",
        name=str(config.get("server_label") or config.get("connector_id") or "hosted-mcp"),
        description=_text(config.get("server_description")),
        provider="openai",
        transport="hosted",
        locator=config.get("server_url"),
        attributes=attributes,
        children=[Component.tool(str(name)) for name in names],
    )


def _describe_shell(tool: ShellTool) -> Component:
    environment = dict(tool.environment or {})
    hosted = environment.get("type") != "local"
    # Skills are what the model can load through the shell, so they are components of
    # the agent in their own right, reported next to the shell tool.
    return Component.tool(
        tool.name,
        provider="openai" if hosted else None,
        attributes={"hosted": hosted, "environment": str(environment.get("type") or "local")},
    )


def shell_skills(agent: Agent[Any]) -> list[Component]:
    """Skills mounted into any shell tool across the agent graph."""

    skills: list[Component] = []
    for current in _reachable_agents(agent):
        for tool in current.tools:
            if not isinstance(tool, ShellTool):
                continue
            for skill in dict(tool.environment or {}).get("skills") or ():
                skill = dict(skill)
                if skill.get("type") == "skill_reference":
                    skills.append(
                        Component(
                            kind="skill",
                            name=str(skill["skill_id"]),
                            version=skill.get("version"),
                            provider="openai",
                            attributes={"agents": current.name},
                        )
                    )
                elif skill.get("name"):
                    path = str(skill.get("path") or "")
                    skills.append(
                        Component(
                            kind="skill",
                            name=str(skill["name"]),
                            description=_text(skill.get("description")),
                            fingerprint=fingerprint(skill.get("description"), path),
                            attributes={
                                "agents": current.name,
                                "dir": path.rstrip("/").rsplit("/", 1)[-1] or None,
                            },
                        )
                    )
    return skills


async def _list_mcp_tools(
    server: Any, run_context: Any, agent: Agent[Any], timeout: float
) -> list[Component]:
    try:
        try:
            tools = await asyncio.wait_for(server.list_tools(run_context, agent), timeout)
        except TypeError:  # MCP server implementations without the context parameters
            tools = await asyncio.wait_for(server.list_tools(), timeout)
    except Exception:  # noqa: BLE001 - an unreachable server is still a dependency
        logger.debug("Could not list tools for MCP server %s", server.name, exc_info=True)
        return []
    return [
        Component.tool(
            tool.name,
            description=getattr(tool, "description", None),
            schema=getattr(tool, "inputSchema", None),
        )
        for tool in tools
    ]


def _mcp_transport(server: Any) -> str | None:
    name = type(server).__name__
    if "Stdio" in name:
        return "stdio"
    if "Sse" in name:
        return "sse"
    if "StreamableHttp" in name:
        return "http"
    return None


def _mcp_locator(server: Any) -> str | None:
    """The server URL, or the name of the executable a stdio server runs.

    The command is reduced to its file name here, where it is known to be a path
    without arguments: install paths can contain spaces, which a generic sanitizer
    cannot tell apart from the start of an argument list.
    """

    params = getattr(server, "params", None)
    if params is None:
        return None
    if isinstance(params, dict):
        return params.get("url") or _executable(params.get("command"))
    return getattr(params, "url", None) or _executable(getattr(params, "command", None))


def _executable(command: Any) -> str | None:
    return PurePath(command).name or None if isinstance(command, str) and command else None


def _model_name(agent: Agent[Any]) -> str:
    model = agent.model
    if isinstance(model, str) and model:
        return model
    name = getattr(model, "model", None)
    if isinstance(name, str) and name:
        return name
    try:
        from agents.models import get_default_model
    except ImportError:  # pragma: no cover - older SDKs
        return "unknown"
    return get_default_model()


def _provider(agent: Agent[Any]) -> str | None:
    name = _model_name(agent)
    if "/" in name:
        return name.split("/", 1)[0]
    return "openai" if isinstance(agent.model, str | None) else None


def _held_by(component: Component, agent_name: str) -> Component:
    return component.model_copy(
        update={"attributes": {**component.attributes, "agents": agent_name}}
    )


def _join_holders(components: list[Component]) -> list[Component]:
    """Merge duplicates, recording every SDK agent that holds each component."""

    holders: dict[tuple[str, str], list[str]] = {}
    for component in components:
        holder = component.attributes.get("agents")
        if isinstance(holder, str):
            names = holders.setdefault((component.kind, component.name), [])
            if holder not in names:
                names.append(holder)
    merged = merge_components(components)
    return [
        c.model_copy(
            update={"attributes": {**c.attributes, "agents": ", ".join(holders[(c.kind, c.name)])}}
        )
        if (c.kind, c.name) in holders
        else c
        for c in merged
    ]


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None

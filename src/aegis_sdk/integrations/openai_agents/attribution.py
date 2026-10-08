"""Link each OpenAI Agents model call to the tools it used, for token attribution.

A response lists the tool calls it asked for. A local tool's result reaches the model
in the *next* call's input, so results are held on the run until that call reports.
Hosted MCP calls (connectors) run inside the response itself: their result is part of
that response's input. Only sizes are reported, never result text.
"""

from __future__ import annotations

from typing import Any

from aegis_sdk.integrations.openai_agents.inventory import tool_usage_path
from aegis_sdk.telemetry import ToolCallRef, ToolResultRef, estimate_tokens


def response_tool_links(response: Any) -> tuple[list[ToolCallRef], list[ToolResultRef]]:
    """Tool calls a response requested, and hosted results it consumed in the same call."""

    calls: list[ToolCallRef] = []
    hosted: list[ToolResultRef] = []
    for item in getattr(response, "output", None) or ():
        kind = _field(item, "type")
        if kind == "function_call":
            call_id, name = _field(item, "call_id"), _field(item, "name")
            if call_id and name:
                calls.append(ToolCallRef(id=str(call_id), name=str(name)))
        elif kind in ("shell_call", "local_shell_call", "apply_patch_call"):
            call_id = _field(item, "call_id") or _field(item, "id")
            if call_id:
                calls.append(ToolCallRef(id=str(call_id), name=kind.removesuffix("_call")))
        elif kind == "mcp_call":
            call_id, name, label = (
                _field(item, "id"),
                _field(item, "name"),
                _field(item, "server_label"),
            )
            if not (call_id and name and label):
                continue
            component = [
                {"kind": "connector", "name": str(label)},
                {"kind": "tool", "name": str(name)},
            ]
            calls.append(ToolCallRef(id=str(call_id), name=str(name), component=component))
            output = _field(item, "output") or ""
            hosted.append(
                ToolResultRef(
                    id=str(call_id),
                    tokens=estimate_tokens(output if isinstance(output, str) else str(output)),
                    component=component,
                )
            )
    return calls, hosted


def tool_result_ref(context: Any, tool: Any, result: object) -> ToolResultRef | None:
    """The size of a local tool's result, keyed by the call id the model used."""

    call_id = getattr(context, "tool_call_id", None)
    if not call_id:
        return None
    path = tool_usage_path(tool) or []
    return ToolResultRef(
        id=str(call_id),
        name=getattr(tool, "name", None),
        tokens=estimate_tokens(result if isinstance(result, str) else str(result)),
        component=[{"kind": kind, "name": name} for kind, name in path] or None,
    )


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)

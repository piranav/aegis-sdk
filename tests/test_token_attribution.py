"""Tool links on model calls, for token attribution, from both agent integrations."""

from __future__ import annotations

import json
from types import SimpleNamespace

from agents import FunctionTool
from agents.tool import ToolOrigin, ToolOriginType

from aegis_sdk.integrations.claude_agent.telemetry import ClaudeTelemetryObserver
from aegis_sdk.integrations.openai_agents import AegisRunHooks
from aegis_sdk.integrations.openai_agents.attribution import response_tool_links
from aegis_sdk.telemetry import AegisTelemetry, InMemoryTelemetryExporter


def recorder():
    exporter = InMemoryTelemetryExporter()
    return AegisTelemetry(exporter), exporter


def llm_calls(exporter):
    return [e for e in exporter.events if e.type == "llm.call"]


async def _noop(ctx, args):
    return "ok"


def mcp_tool(name="check_stock"):
    return FunctionTool(
        name=name,
        description="",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=_noop,
        _tool_origin=ToolOrigin(ToolOriginType.MCP, mcp_server_name="warehouse"),
    )


# --------------------------------------------------------------------- OpenAI Agents


def test_responses_list_their_tool_calls_and_hosted_mcp_results():
    response = SimpleNamespace(
        output=[
            {"type": "function_call", "call_id": "call_1", "name": "lookup_order"},
            {
                "type": "mcp_call",
                "id": "mcp_1",
                "name": "ask_question",
                "server_label": "deepwiki",
                "output": "x" * 400,
            },
            {"type": "message"},
        ]
    )
    calls, hosted = response_tool_links(response)
    assert [(c.id, c.name) for c in calls] == [
        ("call_1", "lookup_order"),
        ("mcp_1", "ask_question"),
    ]
    assert calls[1].component[0] == {"kind": "connector", "name": "deepwiki"}
    assert (hosted[0].id, hosted[0].tokens) == ("mcp_1", 100)


async def test_a_tool_result_is_attached_to_the_next_model_call():
    telemetry, exporter = recorder()
    hooks = AegisRunHooks(telemetry=telemetry)
    agent = SimpleNamespace(name="support", model="gpt-5.4-mini")
    context = SimpleNamespace(usage=SimpleNamespace(), turn_input=None, context=None)

    first = SimpleNamespace(
        response_id="resp_1",
        usage=None,
        output=[{"type": "function_call", "call_id": "call_1", "name": "check_stock"}],
    )
    await hooks.on_llm_end(context, agent, first)
    tool_context = SimpleNamespace(**vars(context), tool_call_id="call_1")
    await hooks.on_tool_end(tool_context, agent, mcp_tool(), "SECRET STOCK LEVELS " * 20)
    second = SimpleNamespace(response_id="resp_2", usage=None, output=[])
    await hooks.on_llm_end(context, agent, second)

    resp_1, resp_2 = llm_calls(exporter)
    assert [c.id for c in resp_1.tool_calls] == ["call_1"] and resp_1.tool_results == []
    (result,) = resp_2.tool_results
    assert result.id == "call_1" and result.tokens == 100
    assert result.component == [
        {"kind": "mcp_server", "name": "warehouse"},
        {"kind": "tool", "name": "check_stock"},
    ]
    assert "SECRET" not in json.dumps([e.to_payload() for e in exporter.events], default=str)


# ------------------------------------------------------------------------ Claude Agent


def message(kind, **fields):
    return type(kind, (), {})() if not fields else type(kind, (), fields)()


def block(kind, **fields):
    return type(kind, (), fields)()


def assistant(message_id, content, *, parent=None, model="claude-opus-5-5"):
    return message(
        "AssistantMessage",
        session_id="s1",
        message_id=message_id,
        model=model,
        usage={"input_tokens": 100, "output_tokens": 20},
        parent_tool_use_id=parent,
        content=content,
    )


def user(results, *, parent=None):
    return message("UserMessage", content=results, parent_tool_use_id=parent)


def test_claude_stream_links_tools_results_and_subagents():
    telemetry, exporter = recorder()
    observer = ClaudeTelemetryObserver(telemetry)
    observer.observe(
        assistant(
            "m1",
            [
                block(
                    "ToolUseBlock",
                    id="tu_1",
                    name="mcp__github__search_code",
                    input={"q": "secret"},
                ),
                block(
                    "ToolUseBlock",
                    id="tu_2",
                    name="Task",
                    input={"subagent_type": "Explore", "prompt": "find"},
                ),
            ],
        )
    )
    observer.observe(assistant("s1", [], parent="tu_2", model="claude-haiku-4-5"))
    observer.observe(user([block("ToolResultBlock", tool_use_id="tu_1", content="a" * 800)]))
    observer.observe(user([block("ToolResultBlock", tool_use_id="tu_2", content="done")]))
    observer.observe(assistant("m2", [block("TextBlock", text="answer")]))
    observer.observe(message("ResultMessage", session_id="s1", result="ok", is_error=False))

    calls = {c.call_id: c for c in llm_calls(exporter)}
    assert [(t.id, t.target) for t in calls["m1"].tool_calls] == [
        ("tu_1", None),
        ("tu_2", "Explore"),
    ]
    assert calls["s1"].parent_tool_call_id == "tu_2"
    results = {r.id: r for r in calls["m2"].tool_results}
    assert results["tu_1"].tokens == 200 and results["tu_1"].name == "mcp__github__search_code"
    assert results["tu_2"].target == "Explore"
    payload = json.dumps([e.to_payload() for e in exporter.events], default=str)
    assert '"q"' not in payload and "secret" not in payload

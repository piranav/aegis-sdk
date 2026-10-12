"""Inventory: component model, reporter, and the OpenAI and Claude integrations."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest
from agents import Agent, FunctionTool, HostedMCPTool, ShellTool
from agents.tool import ToolOrigin, ToolOriginType

from aegis_sdk.integrations.claude_agent.inventory import (
    ClaudeInventoryObserver,
    manifest_from_init,
    mcp_server_key,
)
from aegis_sdk.integrations.openai_agents import AegisRunHooks, describe_agent
from aegis_sdk.integrations.openai_agents.inventory import (
    response_usage_paths,
    tool_usage_path,
)
from aegis_sdk.inventory import AegisInventory, Component, merge_components, sanitize_locator
from aegis_sdk.telemetry import InMemoryTelemetryExporter


class RecordingSender:
    def __init__(self, fail: int = 0) -> None:
        self.manifests: list[dict[str, Any]] = []
        self.fail = fail
        self.sent = threading.Event()

    def __call__(self, manifest: dict[str, Any]) -> dict[str, Any]:
        if self.fail:
            self.fail -= 1
            raise RuntimeError("gateway down")
        self.manifests.append(manifest)
        self.sent.set()
        return {}


def inventory(sender: RecordingSender | None = None) -> AegisInventory:
    return AegisInventory(sender or RecordingSender(), InMemoryTelemetryExporter())


def paths(exporter: InMemoryTelemetryExporter) -> list[list[tuple[str, str]]]:
    return [[(s["kind"], s["name"]) for s in e.component] for e in exporter.events]


def by_key(components: list[Component]) -> dict[tuple[str, str], Component]:
    return {(c.kind, c.name): c for c in components}


# ------------------------------------------------------------------------ component model


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://u:p@host.example:8443/mcp?token=x#f", "https://host.example:8443/mcp"),
        ("/opt/bin/node server.js --key=secret", "node"),
        ("", None),
    ],
)
def test_locators_are_sanitized_before_leaving_the_process(raw, expected):
    assert sanitize_locator(raw) == expected
    assert Component(kind="mcp_server", name="s", locator=raw).locator == expected


def test_merging_unions_children_and_keeps_first_metadata():
    first = Component.mcp_server("github", tools=["create_issue"], transport="http")
    second = Component.mcp_server("github", tools=["search_code", "create_issue"])
    (merged,) = merge_components([first, second])
    assert merged.transport == "http"
    assert sorted(c.name for c in merged.children) == ["create_issue", "search_code"]


def test_tool_fingerprint_tracks_its_definition():
    a = Component.tool("lookup", description="Find an order", schema={"type": "object"})
    b = Component.tool("lookup", description="Find an order", schema={"type": "object"})
    c = Component.tool("lookup", description="Find an order (v2)", schema={"type": "object"})
    assert a.fingerprint == b.fingerprint != c.fingerprint


# ------------------------------------------------------------------------------- reporter


def test_manifests_are_sent_once_until_they_change():
    sender = RecordingSender()
    reporter = inventory(sender)
    components = [Component.model("gpt-4.1-mini")]

    assert reporter.report_manifest(components, framework="custom") is True
    assert reporter.report_manifest(components, framework="custom") is False
    assert reporter.report_manifest([*components, Component.tool("x")]) is True
    assert reporter.flush(timeout=5)

    assert [len(m["components"]) for m in sender.manifests] == [1, 2]
    assert sender.manifests[0]["scope"] == "runtime"
    assert sender.manifests[0]["framework"] == "custom"


def test_a_failed_send_is_retried_on_the_next_report():
    sender = RecordingSender(fail=1)
    reporter = inventory(sender)
    components = [Component.model("gpt-4.1-mini")]
    reporter.report_manifest(components)
    assert reporter.flush(timeout=5)
    assert sender.manifests == []

    assert reporter.report_manifest(components) is True
    assert reporter.flush(timeout=5)
    assert len(sender.manifests) == 1


def test_usage_is_exported_as_component_paths():
    reporter = inventory()
    reporter.record_usage([("connector", "deepwiki"), ("tool", "ask_question")], count=2)
    reporter.record_usage([])
    (event,) = reporter.usage_exporter.events
    assert event.to_payload()["component"] == [
        {"kind": "connector", "name": "deepwiki"},
        {"kind": "tool", "name": "ask_question"},
    ]
    assert event.count == 2


# ------------------------------------------------------------------------- OpenAI Agents


class FakeMcpServer:
    def __init__(self, name: str, tools: list[str], *, fail: bool = False) -> None:
        self.name = name
        self.params = {"url": f"https://{name}.example/mcp?api_key=secret"}
        self._tools = tools
        self._fail = fail

    async def list_tools(self, run_context=None, agent=None):
        if self._fail:
            raise ConnectionError("not connected")
        return [
            SimpleNamespace(name=t, description=f"{t} tool", inputSchema={"type": "object"})
            for t in self._tools
        ]


async def _noop(ctx, args):
    return "ok"


def function_tool(name: str, origin: ToolOrigin | None = None) -> FunctionTool:
    return FunctionTool(
        name=name,
        description=f"{name} description",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=_noop,
        _tool_origin=origin,
    )


def support_agent() -> Agent:
    researcher = Agent(name="researcher", model="gpt-4.1-mini", tools=[function_tool("cite")])
    refunds = Agent(
        name="refunds",
        model="gpt-4.1",
        handoff_description="Handles refunds",
        mcp_servers=[FakeMcpServer("payments", ["refund"], fail=True)],
    )
    return Agent(
        name="support",
        model="gpt-4.1-mini",
        tools=[
            function_tool("lookup_order"),
            HostedMCPTool(
                tool_config={
                    "type": "mcp",
                    "server_label": "deepwiki",
                    "server_url": "https://mcp.deepwiki.com/mcp?token=x",
                    "allowed_tools": ["ask_question"],
                    "require_approval": "never",
                }
            ),
            ShellTool(
                executor=lambda request: "ok",
                environment={
                    "type": "local",
                    "skills": [
                        {
                            "name": "refund-policy",
                            "description": "Refund rules",
                            "path": "/srv/skills/refund-policy",
                        }
                    ],
                },
            ),
            researcher.as_tool(tool_name="research", tool_description="Research a topic"),
        ],
        mcp_servers=[FakeMcpServer("warehouse", ["check_stock", "reserve_stock"])],
        handoffs=[refunds],
    )


async def test_describe_agent_covers_the_whole_agent_graph():
    components = by_key(await describe_agent(support_agent()))

    assert set(components) == {
        ("model", "gpt-4.1-mini"),
        ("model", "gpt-4.1"),
        ("tool", "lookup_order"),
        ("tool", "cite"),
        ("tool", "shell"),
        ("connector", "deepwiki"),
        ("mcp_server", "warehouse"),
        ("mcp_server", "payments"),
        ("skill", "refund-policy"),
        ("subagent", "researcher"),
        ("subagent", "refunds"),
    }
    warehouse = components[("mcp_server", "warehouse")]
    assert [t.name for t in warehouse.children] == ["check_stock", "reserve_stock"]
    assert warehouse.locator == "https://warehouse.example/mcp"
    assert components[("mcp_server", "payments")].children == []  # unreachable, still listed
    deepwiki = components[("connector", "deepwiki")]
    assert (deepwiki.transport, deepwiki.locator) == ("hosted", "https://mcp.deepwiki.com/mcp")
    assert [t.name for t in deepwiki.children] == ["ask_question"]
    assert components[("model", "gpt-4.1-mini")].attributes["agents"] == "support, researcher"
    assert components[("skill", "refund-policy")].attributes["dir"] == "refund-policy"
    assert components[("subagent", "refunds")].description == "Handles refunds"


def test_tool_usage_paths_follow_tool_origin():
    mcp = function_tool("check_stock", ToolOrigin(ToolOriginType.MCP, mcp_server_name="warehouse"))
    nested = function_tool(
        "research", ToolOrigin(ToolOriginType.AGENT_AS_TOOL, agent_name="researcher")
    )
    assert tool_usage_path(mcp) == [("mcp_server", "warehouse"), ("tool", "check_stock")]
    assert tool_usage_path(nested) == [("subagent", "researcher")]
    assert tool_usage_path(function_tool("lookup_order")) == [("tool", "lookup_order")]


def test_response_items_reveal_hosted_calls_and_skill_reads():
    skill = Component(kind="skill", name="refund-policy", attributes={"dir": "refund-policy"})
    response = SimpleNamespace(
        output=[
            {"type": "mcp_call", "server_label": "deepwiki", "name": "ask_question"},
            SimpleNamespace(type="web_search_call"),
            {"type": "shell_call", "action": {"commands": ["cat skills/refund-policy/SKILL.md"]}},
            {"type": "shell_call", "action": {"commands": ["ls /tmp"]}},
            {"type": "message"},
        ]
    )
    assert list(response_usage_paths(response, skills=[skill])) == [
        [("connector", "deepwiki"), ("tool", "ask_question")],
        [("tool", "web_search")],
        [("skill", "refund-policy")],
    ]


async def test_run_hooks_report_the_manifest_once_and_record_ungated_usage():
    sender = RecordingSender()
    reporter = inventory(sender)
    hooks = AegisRunHooks(inventory=reporter)
    agent = support_agent()

    for _ in range(2):  # two runs of the same agent: one manifest
        context = SimpleNamespace(usage=SimpleNamespace(), turn_input=None, context=None)
        await hooks.on_agent_start(context, agent)
    assert reporter.flush(timeout=5)
    assert len(sender.manifests) == 1
    assert sender.manifests[0]["framework"] == "openai-agents"

    mcp_tool = function_tool(
        "check_stock", ToolOrigin(ToolOriginType.MCP, mcp_server_name="warehouse")
    )
    await hooks.on_tool_start(context, agent, mcp_tool)
    gated = SimpleNamespace(**vars(context), _aegis_gateway_guardrail_state={"audit_id": "a"})
    await hooks.on_tool_start(gated, agent, function_tool("lookup_order"))
    await hooks.on_handoff(context, agent, agent.handoffs[0])
    response = SimpleNamespace(
        output=[{"type": "shell_call", "action": {"commands": ["cat refund-policy/SKILL.md"]}}],
        usage=None,
        response_id="r1",
    )
    await hooks.on_llm_end(context, agent, response)

    assert paths(reporter.usage_exporter) == [
        [("mcp_server", "warehouse"), ("tool", "check_stock")],
        [("subagent", "refunds")],
        [("skill", "refund-policy")],
    ]


# ------------------------------------------------------------------------- Claude Agent


INIT = {
    "model": "claude-sonnet-4-5",
    "tools": [
        "Bash",
        "Read",
        "Skill",
        "mcp__linear__create_issue",
        "mcp__linear__list_issues",
        "mcp__plugin_docs_search__query",
    ],
    "mcp_servers": [
        {"name": "linear", "status": "connected", "source": "user"},
        {"name": "plugin:docs:search", "status": "connected"},
        {"name": "broken", "status": "failed"},
    ],
    "agents": ["general-purpose", "docs:writer"],
    "skills": ["pdf", "docs:style-guide"],
    "plugins": [{"name": "docs", "path": "/home/me/.claude/plugins/docs"}],
}


def test_claude_init_message_becomes_a_manifest():
    components = by_key(manifest_from_init(INIT))

    assert components[("model", "claude-sonnet-4-5")].provider == "anthropic"
    linear = components[("mcp_server", "linear")]
    assert [t.name for t in linear.children] == ["create_issue", "list_issues"]
    assert linear.attributes == {"status": "connected", "source": "user"}
    # Server names are normalized the way Claude names their tools, so both line up.
    assert mcp_server_key("plugin:docs:search") == "plugin_docs_search"
    assert [t.name for t in components[("mcp_server", "plugin_docs_search")].children] == ["query"]
    assert components[("mcp_server", "broken")].attributes["status"] == "failed"
    docs = components[("plugin", "docs")]
    assert {(c.kind, c.name) for c in docs.children} == {
        ("subagent", "writer"),
        ("skill", "style-guide"),
    }
    assert ("skill", "pdf") in components and ("subagent", "general-purpose") in components
    assert components[("tool", "Bash")].attributes == {"builtin": True}


def test_claude_observer_reports_on_init_only():
    sender = RecordingSender()
    observer = ClaudeInventoryObserver(inventory(sender))
    system_message = type("SystemMessage", (), {})
    other = system_message()
    other.subtype, other.data = "compact_boundary", {}
    init = system_message()
    init.subtype, init.data = "init", INIT
    observer.observe(other)
    observer.observe(init)
    observer.observe(init)
    assert observer.inventory.flush(timeout=5)
    assert len(sender.manifests) == 1
    assert sender.manifests[0]["framework"] == "claude-agent"


def test_stdio_server_locator_survives_install_paths_with_spaces():
    from agents.mcp import MCPServerStdio

    server = MCPServerStdio(
        name="warehouse",
        params={"command": "/Volumes/Project Drive/.venv/bin/python", "args": ["--token", "x"]},
    )
    from aegis_sdk.integrations.openai_agents.inventory import _mcp_locator

    assert _mcp_locator(server) == "python"

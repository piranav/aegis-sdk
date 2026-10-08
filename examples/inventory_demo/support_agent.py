"""Demo: an OpenAI Agents SDK support agent whose dependencies show up in Aegis.

The agent is built from four kinds of dependency:

- a **function tool** (``lookup_order``) gated by Aegis guardrails,
- a local **MCP server** (``warehouse``, over stdio) with stock tools,
- a hosted **connector** (DeepWiki's public MCP server, called by OpenAI directly),
- a **skill** (``refund-policy``) the model loads through the shell tool.

Run with AEGIS_API_URL, AEGIS_API_KEY (an agent key from the dashboard), and
OPENAI_API_KEY set, then open the agent in the dashboard: its Dependencies card lists
each component, whether the agent declared it, and how often each run used it.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
from pathlib import Path

from agents import Agent, HostedMCPTool, Runner, ShellTool, function_tool
from agents.mcp import MCPServerStdio
from agents.tool import ShellCommandRequest

from aegis_sdk import AegisGatewayClient, AegisInventory, AegisTelemetry, bind_session
from aegis_sdk.integrations.openai_agents import (
    AegisRunHooks,
    make_aegis_gateway_guardrail,
    make_aegis_gateway_output_guardrail,
)

HERE = Path(__file__).resolve().parent
SKILLS = HERE / "skills"
READ_ONLY_COMMANDS = {"cat", "ls", "head"}
PROMPT = (
    "Customer acme-retail asks: 1) What's the status of order A-1001? "
    "2) Is SKU KB-42 in stock? If so, reserve 2 units. "
    "3) They opened the keyboard from order A-1001 - can they still return it, and on what terms? "
    "4) Their developer asks how the openai/openai-agents-python repo supports MCP servers; "
    "answer in two sentences using the deepwiki connector."
)


def read_only_shell(request: ShellCommandRequest) -> str:
    """Run only read-only commands inside the skills folder; refuse everything else.

    The model needs the shell to load skills, nothing more, so the executor is a
    least-privilege sandbox rather than a general shell.
    """

    outputs = []
    for command in request.data.action.commands:
        try:
            argv = shlex.split(command)
        except ValueError:
            outputs.append(f"$ {command}\nrefused: unparseable command")
            continue
        paths = [
            (SKILLS / a).resolve() if not a.startswith("/") else Path(a).resolve()
            for a in argv[1:]
            if not a.startswith("-")
        ]
        if (
            not argv
            or argv[0] not in READ_ONLY_COMMANDS
            or any(SKILLS.resolve() not in (p, *p.parents) for p in paths)
        ):
            outputs.append(
                f"$ {command}\nrefused: only read-only commands inside the skills folder"
            )
            continue
        result = subprocess.run(argv, cwd=SKILLS, capture_output=True, text=True, timeout=5)
        outputs.append(f"$ {command}\n{result.stdout}{result.stderr}")
    return "\n".join(outputs)


async def main() -> None:
    with AegisGatewayClient.from_env() as client:

        @function_tool(
            tool_input_guardrails=[make_aegis_gateway_guardrail(client)],
            tool_output_guardrails=[make_aegis_gateway_output_guardrail(client)],
        )
        def lookup_order(order_id: str) -> str:
            """Look up an order's shipping status and contents."""
            return (
                f"Order {order_id}: delivered 6 days ago; contains 1 x KB-42 mechanical keyboard."
            )

        warehouse = MCPServerStdio(
            name="warehouse",
            params={"command": sys.executable, "args": [str(HERE / "warehouse_mcp.py")]},
            cache_tools_list=True,
        )
        deepwiki = HostedMCPTool(
            tool_config={
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "server_description": "Answers questions about public GitHub repositories",
                "allowed_tools": ["ask_question", "read_wiki_structure"],
                "require_approval": "never",
            }
        )
        shell = ShellTool(
            executor=read_only_shell,
            environment={
                "type": "local",
                "skills": [
                    {
                        "name": "refund-policy",
                        "description": "The store's refund and return rules.",
                        "path": str(SKILLS / "refund-policy"),
                    }
                ],
            },
        )

        telemetry = AegisTelemetry.from_client(client)
        inventory = AegisInventory.from_client(client)
        hooks = AegisRunHooks(telemetry=telemetry, inventory=inventory)

        async with warehouse:
            agent = Agent(
                name="acme-support",
                instructions=(
                    "You are a support agent for an online store. Use lookup_order for orders, "
                    "the warehouse tools for stock, the refund-policy skill for returns (read its "
                    "SKILL.md with the shell before answering), and the deepwiki connector for "
                    "questions about GitHub repositories. Be brief."
                ),
                model=os.getenv("OPENAI_AGENT_MODEL", "gpt-5.4-mini"),
                tools=[lookup_order, deepwiki, shell],
                mcp_servers=[warehouse],
            )
            with bind_session(customer_id="acme-retail", end_user_id="user-7"):
                result = await Runner.run(agent, PROMPT, hooks=hooks, max_turns=20)

        inventory.shutdown(timeout=10)
        telemetry.shutdown(timeout=10)
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

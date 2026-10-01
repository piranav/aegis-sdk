"""Run with AEGIS_API_URL, AEGIS_API_KEY, and OPENAI_API_KEY configured."""

import asyncio
import os

from agents import Agent, Runner, function_tool

from aegis_sdk import AegisGatewayClient
from aegis_sdk.integrations.openai_agents import (
    make_aegis_gateway_guardrail,
    make_aegis_gateway_output_guardrail,
)


async def main() -> None:
    with AegisGatewayClient.from_env() as client:

        @function_tool(
            tool_input_guardrails=[make_aegis_gateway_guardrail(client)],
            tool_output_guardrails=[make_aegis_gateway_output_guardrail(client)],
        )
        def lookup_order(order_id: str) -> str:
            """Look up shipping status for an example order."""
            return f"Order {order_id} is in transit."

        agent = Agent(
            name="example-support-agent",
            tools=[lookup_order],
            model=os.getenv("OPENAI_AGENT_MODEL", "gpt-4.1-mini"),
        )
        result = await Runner.run(agent, "Look up example-order.")
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

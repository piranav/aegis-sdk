"""Run with Aegis gateway credentials and Claude authentication configured."""

import asyncio

from claude_agent_sdk import ClaudeSDKClient

from aegis_sdk import AegisGatewayClient
from aegis_sdk.integrations.claude_agent import AegisClaudeGatewayProvider


async def main() -> None:
    with AegisGatewayClient.from_env() as gateway:
        options = AegisClaudeGatewayProvider(gateway).get_options(max_turns=3)
        async with ClaudeSDKClient(options=options) as client:
            await client.query("Run the governed workflow.")
            async for message in client.receive_response():
                print(message)


if __name__ == "__main__":
    asyncio.run(main())

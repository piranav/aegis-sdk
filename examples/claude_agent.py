"""Run with Aegis gateway credentials and Claude authentication configured."""

import asyncio

from claude_agent_sdk import ClaudeSDKClient

from aegis_sdk import AegisGatewayClient, AegisTelemetry, bind_session
from aegis_sdk.integrations.claude_agent import AegisClaudeGatewayProvider


async def main() -> None:
    with AegisGatewayClient.from_env() as gateway:
        telemetry = AegisTelemetry.from_client(gateway)
        provider = AegisClaudeGatewayProvider(gateway, telemetry=telemetry)
        async with ClaudeSDKClient(options=provider.get_options(max_turns=3)) as client:
            with bind_session(customer_id="example-customer"):
                await client.query("Run the governed workflow.")
                async for message in provider.instrument(client.receive_response()):
                    print(message)
        telemetry.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

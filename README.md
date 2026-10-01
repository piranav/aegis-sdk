# Aegis SDK

Python client for the hosted Aegis governance gateway, with integrations for
OpenAI Agents and the Claude Agent SDK. This repository contains only client-side
code. The governance engine, classifiers, policies, storage, API server, and
dashboard are maintained separately in the private Aegis platform repository.

## Install

Install the tagged source directly from GitHub:

```bash
pip install 'aegis-sdk @ git+https://github.com/piranav/aegis-sdk.git@v0.1.0'
# Select an optional framework integration:
pip install 'aegis-sdk[openai-agents] @ git+https://github.com/piranav/aegis-sdk.git@v0.1.0'
pip install 'aegis-sdk[claude-agent] @ git+https://github.com/piranav/aegis-sdk.git@v0.1.0'
```

The package has not been published to PyPI. Python 3.11 or newer is required.

## Configure

Register your agent with the Aegis platform and supply its API key at runtime:

```bash
export AEGIS_API_URL=https://your-aegis-gateway.example
export AEGIS_API_KEY=your-agent-api-key
```

Use HTTPS for hosted gateways. `http://127.0.0.1:8080` is the default for local
platform development. The SDK sends the agent key as a bearer token. Keep keys
out of source control.

## Custom agent

```python
from aegis_sdk import ActionContext, AegisGatewayClient

with AegisGatewayClient.from_env() as client:
    decision = client.evaluate(
        ActionContext(
            agent_name="support-agent",
            tool_name="lookup_order",
            tool_args={"order_id": "example-order"},
        )
    )
    if not decision.allowed:
        raise RuntimeError("Aegis denied this tool call")

    # Execute the approved tool, then evaluate its result before exposing it.
    result = "Your order is in transit."
    output_decision = client.evaluate_result(decision.audit_id, result)
    if not output_decision.allowed:
        raise RuntimeError("Aegis denied this tool result")
    print(result)
```

## OpenAI Agents

```python
from agents import function_tool
from aegis_sdk import AegisGatewayClient
from aegis_sdk.integrations.openai_agents import (
    make_aegis_gateway_guardrail,
    make_aegis_gateway_output_guardrail,
)

client = AegisGatewayClient.from_env()


@function_tool(
    tool_input_guardrails=[make_aegis_gateway_guardrail(client)],
    tool_output_guardrails=[make_aegis_gateway_output_guardrail(client)],
)
def lookup_order(order_id: str) -> str:
    """Look up a shipping status."""
    return "In transit"


# Attach lookup_order to your Agent and close client after the run.
```

Guardrails must be attached to every function tool requiring governance. They
are not enforcement for hosted tools or an entire sandbox. Optional
`AegisRunHooks` records lifecycle events; its trace may contain sensitive tool
arguments/results and should be protected accordingly.

## Claude Agent SDK

```python
from claude_agent_sdk import ClaudeSDKClient
from aegis_sdk import AegisGatewayClient
from aegis_sdk.integrations.claude_agent import AegisClaudeGatewayProvider


async def main():
    with AegisGatewayClient.from_env() as gateway:
        options = AegisClaudeGatewayProvider(gateway).get_options(max_turns=3)
        async with ClaudeSDKClient(options=options) as client:
            await client.query("Run the governed workflow.")
            async for message in client.receive_response():
                print(message)
```

The provider attaches PreToolUse and PostToolUse callbacks. Input denials block
tool execution; output denials replace the result in the current SDK's
`hookSpecificOutput.updatedToolOutput`. Output checks run after the tool has
executed and cannot undo side effects. Use one provider per runtime/session and
retain default SDK permission controls. Do not override governance hook outputs
with permissive custom hooks.

## Development

```bash
pip install -e '.[dev,openai-agents,claude-agent]'
ruff check .
pytest -q
python -m build
```

Tests use mocked gateway responses and do not require a private platform checkout
or paid model requests. See [SECURITY.md](SECURITY.md) for security boundaries.

## Licensing

This repository is public for inspection. It retains the project's existing
proprietary licensing; no open-source license is granted by publication alone.

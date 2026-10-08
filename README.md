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

## Session telemetry

Aegis can show each agent run as a session: the request, every model call with its
token usage, every governed tool call, and the answer. Usage is filterable by agent,
model, and customer in the dashboard's Sessions page.

```python
from aegis_sdk import AegisGatewayClient, AegisTelemetry, TokenUsage, bind_session

gateway = AegisGatewayClient.from_env()
telemetry = AegisTelemetry.from_client(gateway)
```

**OpenAI Agents**: pass the hooks to each run. Tool guardrails attached to the same run
are linked to its session automatically.

```python
from aegis_sdk.integrations.openai_agents import AegisRunHooks

hooks = AegisRunHooks(telemetry=telemetry)
with bind_session(customer_id="acme", end_user_id=user.id):
    result = await Runner.run(agent, prompt, hooks=hooks)
```

**Claude Agent SDK**: give the provider telemetry and wrap the message stream, which is
where Claude reports token usage and cost.

```python
provider = AegisClaudeGatewayProvider(gateway, telemetry=telemetry)
async with ClaudeSDKClient(options=provider.get_options()) as client:
    with bind_session(customer_id="acme"):
        await client.query(prompt)
        async for message in provider.instrument(client.receive_response()):
            ...
```

**Custom agents** report sessions directly:

```python
with telemetry.session(input=prompt, customer_id="acme") as session:
    response = llm.responses.create(model="gpt-4.1-mini", input=prompt)
    session.record_llm_call(
        model=response.model,
        call_id=response.id,
        usage=TokenUsage(input_tokens=response.usage.input_tokens,
                         output_tokens=response.usage.output_tokens),
    )
```

- `bind_session` attributes every session started inside it. Pass `session_id` to
  group several runs (for example, a chat's turns) into one session.
- `input_tokens` includes cached tokens; integrations normalize provider usage.
- Events are batched on a background thread and never block the agent. Network
  errors, 429s, and 5xx responses are retried with backoff. If the gateway stays
  unreachable until the bounded queue fills, new events are dropped rather than
  growing memory (`telemetry.exporter.dropped_events` counts them). Call
  `telemetry.shutdown()` before exit; an `atexit` hook also drains the queue.
- Prompts and answers are sent as previews (truncated by the gateway). Use
  `AegisTelemetry.from_client(gateway, capture_content=False)` to send metadata only.

### Attributing usage to customers

Aegis does not detect who a run is for; your code declares it. Wrap each run in
`bind_session` (or pass the same fields to `telemetry.session(...)`):

```python
with bind_session(
    customer_id=account.id,          # who the work is for: drives "usage by customer"
    end_user_id=user.id,             # the person inside that customer, if any
    attributes={"plan": "enterprise", "region": "eu"},  # extra dimensions to filter on
):
    result = await Runner.run(agent, prompt, hooks=hooks)
```

How it flows: the values are sent once, on the session's start event, and stored on
the Aegis session. Every model call, token count, and governed tool call in that run
is attributed through the session, so you tag the run, not each call.

| Field | Use it for | Avoid |
| --- | --- | --- |
| `customer_id` | Your stable account/tenant id (`acct_8812`) | Names, emails, or ids you might reissue |
| `end_user_id` | A stable, pseudonymous user id | Emails, phone numbers, names |
| `attributes` | Up to 32 low-sensitivity scalars (plan, region, feature) | Secrets, tokens, personal data |

Rules worth knowing:

- **One customer per session.** The first value Aegis receives wins; later events
  cannot reassign a session. If one process serves many customers, start a separate
  session (a separate `bind_session` block) per customer's work.
- **Untagged runs** are grouped as "No customer set" in the dashboard.
- **Ids are matched exactly.** `acme` and `ACME` are different customers, so use the
  same canonical id everywhere.
- **Attribution is self-reported.** Aegis records what your runtime sends and uses it
  for reporting only, never for access control. Set it from your application's
  authenticated context, not from model output or tool arguments.

## Agent inventory

Aegis keeps each agent's bill of materials: the models, tools, MCP servers, connectors
(hosted MCP), skills, plugins, and subagents it is configured with, and which of them
each run actually uses. The dashboard flags declared-but-unused components (excess
capability), used-but-undeclared ones (shadow dependencies), and MCP servers whose tool
definitions changed.

```python
from aegis_sdk import AegisInventory

inventory = AegisInventory.from_client(gateway)
```

**OpenAI Agents**: the hooks describe the whole agent graph (handoffs and agents-as-tools
included, MCP servers with their tools) and report usage guardrails can't see: MCP tool
calls, hosted connector calls, handoffs, and skills loaded through the shell tool.

```python
hooks = AegisRunHooks(telemetry=telemetry, inventory=inventory)
```

**Claude Agent SDK**: each session's `init` message becomes its manifest.

```python
provider = AegisClaudeGatewayProvider(gateway, telemetry=telemetry, inventory=inventory)
async for message in provider.instrument(client.receive_response()):
    ...
```

**Custom agents** describe themselves with `Component`:

```python
from aegis_sdk import Component

inventory.report_manifest([
    Component.model("gpt-4.1-mini", provider="openai"),
    Component.mcp_server("github", transport="http", locator=url, tools=["create_issue"]),
])
inventory.record_usage([("mcp_server", "github"), ("tool", "create_issue")])
```

- Manifests are sent on a background thread and only when they change; a failed send
  is retried on the next report. Usage is batched like telemetry.
- Never pass credentials: components have no field for env, headers, or arguments, and
  `locator` is reduced to scheme, host, and path (or an executable name).
- A manifest replaces earlier ones with the same `scope` (default `runtime`).

See `examples/inventory_demo/` for a runnable agent with a local MCP server, a hosted
connector, a skill, and a governed function tool.

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

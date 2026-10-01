# SDK security boundaries

The SDK transports actions/results to an authenticated Aegis gateway. It does not
contain the policy engine and does not enforce policies without a reachable
server. Gateway errors deny tool calls/results in the framework adapters.

Use HTTPS for remote gateways and protect agent API keys as credentials. Use
trusted application identity when resolving user IDs or human approval; tool
arguments are not proof of authentication or approval. Governance after execution
cannot reverse tool side effects.

Framework hooks and optional traces can retain arguments and results. Protect
logs and memory containing them. Attach guardrails to each governed function tool;
SDK function-tool guardrails do not cover provider-hosted tools.

Report suspected issues privately to the repository owner. Do not include active
credentials or sensitive action/result content in public issues.

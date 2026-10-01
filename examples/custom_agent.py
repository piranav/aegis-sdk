"""Evaluate a custom tool call/result against the configured Aegis gateway."""

from aegis_sdk import ActionContext, AegisGatewayClient


def main() -> None:
    with AegisGatewayClient.from_env() as client:
        decision = client.evaluate(
            ActionContext(
                agent_name="example-agent",
                tool_name="lookup_order",
                tool_args={"order_id": "example-order"},
            )
        )
        if not decision.allowed:
            print("Tool call denied by Aegis.")
            return
        result = "Your order is in transit."
        output = client.evaluate_result(decision.audit_id, result)
        print(result if output.allowed else "Tool result suppressed by Aegis.")


if __name__ == "__main__":
    main()

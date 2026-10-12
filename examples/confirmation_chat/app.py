"""A support chat whose agent asks its user before running actions an org rule holds.

Nothing here handles confirmations: the agent only uses the standard Aegis guardrails.
When the organization has a rule like::

    constraint "Confirm customer deletion"
      when Action
      where tool_name = "delete_customer_record"
      then escalate

the agent asks the user in the chat, and the user's "yes" lets the deletion run.

Run with AEGIS_API_URL, AEGIS_API_KEY, and OPENAI_API_KEY set::

    uvicorn examples.confirmation_chat.app:app --port 8765
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from agents import Agent, Runner, function_tool
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from aegis_sdk import AegisGatewayClient
from aegis_sdk.integrations.openai_agents import (
    make_aegis_gateway_guardrail,
    make_aegis_gateway_output_guardrail,
)

client = AegisGatewayClient.from_env()
# The person signed in to this app; Aegis records them as the one who confirmed.
SIGNED_IN_USER = os.getenv("DEMO_USER", "support-lead@acme.test")


def signed_in_user(_data) -> str:
    return SIGNED_IN_USER


guardrails = {
    "tool_input_guardrails": [
        make_aegis_gateway_guardrail(client, user_id_resolver=signed_in_user)
    ],
    "tool_output_guardrails": [
        make_aegis_gateway_output_guardrail(client, user_id_resolver=signed_in_user)
    ],
}
CUSTOMERS = {"C-1001": "Ada Lovelace", "C-1002": "Grace Hopper"}


@function_tool(**guardrails)
def lookup_customer(customer_id: str) -> str:
    """Look up a customer record by id."""
    name = CUSTOMERS.get(customer_id)
    return f"{customer_id}: {name}" if name else f"No customer {customer_id}"


@function_tool(**guardrails)
def delete_customer_record(customer_id: str) -> str:
    """Permanently delete a customer record by id."""
    name = CUSTOMERS.pop(customer_id, None)
    return f"Deleted {customer_id} ({name})." if name else f"No customer {customer_id}"


agent = Agent(
    name="support-agent",
    instructions="You help support staff manage customer records. Be brief.",
    tools=[lookup_customer, delete_customer_record],
    model=os.getenv("OPENAI_AGENT_MODEL", "gpt-6-luna"),
)

app = FastAPI()
history: list = []
turn = asyncio.Lock()  # One conversation: messages are answered in order.


class Message(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return (Path(__file__).parent / "index.html").read_text()


@app.post("/chat")
async def chat(message: Message) -> dict:
    global history
    async with turn:
        result = await Runner.run(agent, [*history, {"role": "user", "content": message.text}])
        history = result.to_input_list()
    return {"reply": result.final_output, "customers": CUSTOMERS}


@app.post("/reset")
def reset() -> dict:
    history.clear()
    CUSTOMERS.update({"C-1001": "Ada Lovelace", "C-1002": "Grace Hopper"})
    return {"ok": True}

"""Assistant-independent governance flow shared by every adapter."""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import time

from aegis_plugin import VERSION
from aegis_plugin.adapters import (
    POST_TOOL,
    PRE_TOOL,
    PROMPT,
    SESSION_END,
    SESSION_START,
    Adapter,
    Event,
    Output,
)
from aegis_plugin.client import ApiError, State, decode_code, request

# "aegis connect <code>", "aegis status", "aegis disconnect", typed as a message. The
# plugin answers these itself, so connection codes never reach the model provider.
COMMAND = re.compile(r"^\s*/?aegis[\s:]+(connect|status|disconnect)\b\s*(\S*)\s*$", re.I)


class Engine:
    def __init__(self, adapter: Adapter, state: State | None = None):
        self.adapter = adapter
        self.state = state or State(adapter.tool)

    # ------------------------------------------------------------ commands

    def client_info(self) -> dict:
        info = {
            "hostname": socket.gethostname(),
            "os": f"{platform.system()} {platform.release()}",
            "os_user": os.environ.get("USER") or os.environ.get("USERNAME") or "",
            "plugin_version": VERSION,
            **self.adapter.account(),
        }
        return {k: v for k, v in info.items() if v}

    def connect(self, code: str) -> str:
        if not code:
            return "Aegis: add the code from your IT invite, e.g. aegis connect aegc_..."
        try:
            api_url = decode_code(code)
            result = request(
                api_url,
                "/v1/subscriptions/connect",
                {"code": code, "tool": self.adapter.tool, "client": self.client_info()},
            )
        except ValueError as exc:
            return f"Aegis: {exc}"
        except ApiError as exc:
            return f"Aegis could not connect: {exc.detail}"
        self.state.save(
            {
                "api_url": api_url,
                "api_key": result["api_key"],
                "subscription_id": result["subscription_id"],
                "organization": result.get("organization"),
                "email": result.get("email"),
                "failure_mode": result.get("failure_mode", "open"),
                "connected_at": int(time.time()),
            }
        )
        return (
            f"Aegis connected: {result.get('email')} at {result.get('organization')}. "
            f"Your IT team now sees this {self.adapter.label} as active, and your "
            "organization's rules apply from the next tool call."
        )

    def status(self) -> str:
        creds = self.state.credentials()
        if not creds:
            return "Aegis is not connected. Send: aegis connect <code from your IT invite>"
        try:
            result = self.heartbeat(creds, "heartbeat", self.client_info())
            state = result.get("status", "unknown")
        except ApiError as exc:
            state = f"error ({exc.detail})"
        offline = "block tool calls" if creds.get("failure_mode") == "closed" else "allow and warn"
        return (
            f"Aegis: {creds.get('email')} at {creds.get('organization')} · status {state} · "
            f"server {creds['api_url']} · when unreachable: {offline}"
        )

    def disconnect(self) -> str:
        creds = self.state.credentials()
        if not creds:
            return "Aegis is not connected."
        note = ""
        try:
            self.heartbeat(creds, "disconnect", self.client_info())
        except ApiError as exc:
            note = f" (Aegis couldn't be told: {exc.detail})"
        self.state.forget()
        label = self.adapter.label
        return f"Aegis disconnected{note}. Your IT team will see this {label} as disconnected."

    def heartbeat(self, creds: dict, event: str, client: dict) -> dict:
        result = request(
            creds["api_url"],
            "/v1/subscriptions/heartbeat",
            {"event": event, "client": client},
            creds["api_key"],
        )
        mode = result.get("failure_mode")
        if mode in ("open", "closed") and mode != creds.get("failure_mode"):
            creds["failure_mode"] = mode
            self.state.save(creds)
        return result

    # ------------------------------------------------------------ hooks

    def handle(self, event: Event) -> Output:
        a = self.adapter
        if event.kind == PROMPT:
            command = COMMAND.match(event.prompt)
            if command:
                verb, arg = command.group(1).lower(), command.group(2)
                text = {"connect": lambda: self.connect(arg), "status": self.status}.get(
                    verb, self.disconnect
                )()
                return a.reply(event, text)

        creds = self.state.credentials()
        if event.kind == SESSION_START:
            if not creds:
                return a.notify(
                    event,
                    f"[Aegis] This {a.label} isn't connected to your organization yet. "
                    "Send the message: aegis connect <code from your IT invite>",
                )
            try:
                self.heartbeat(creds, "session_start", self.client_info())
            except ApiError as exc:
                return self.unavailable(event, creds, exc)
            return a.allow(event)

        if not creds:
            return a.allow(event)

        if event.kind == SESSION_END:
            self.state.clear_session(event.session_id)
            try:
                self.heartbeat(creds, "session_end", {})
            except ApiError:
                pass
            return a.allow(event)

        if event.kind == PROMPT:
            body = {
                "agent_name": a.tool,
                "tool_name": "UserPromptSubmit",
                "tool_args": {"_aegis": {"tool": a.tool, "action": "prompt.submit"}},
                "session_id": event.session_id,
                "prompt": event.prompt[:32000],
            }
            return self.decide(event, creds, "/v1/evaluate", body)

        if event.kind == PRE_TOOL:
            body = {
                "agent_name": a.tool,
                "tool_name": event.tool_name or "unknown",
                "tool_args": {
                    **event.tool_input,
                    "_aegis": {"tool": a.tool, "action": a.action(event), "paths": a.paths(event)},
                },
                "session_id": event.session_id,
            }
            return self.decide(event, creds, "/v1/evaluate", body)

        if event.kind == POST_TOOL:
            audit_id = self.state.recall_audit(event.session_id, event.tool_use_id)
            if not audit_id:
                return a.allow(event)
            output = event.tool_response
            text = output if isinstance(output, str) else json.dumps(output, default=str)
            body = {"audit_id": audit_id, "result": text[:100000], "result_metadata": {}}
            return self.decide(event, creds, "/v1/evaluate_result", body)

        return a.allow(event)

    def decide(self, event: Event, creds: dict, path: str, body: dict) -> Output:
        try:
            result = request(creds["api_url"], path, body, creds["api_key"])
        except ApiError as exc:
            return self.unavailable(event, creds, exc)
        if event.kind == PRE_TOOL and result.get("audit_id") and event.tool_use_id:
            self.state.remember_audit(event.session_id, event.tool_use_id, result["audit_id"])
        verdict = result.get("decision")
        if verdict == "allow":
            return self.adapter.allow(event)
        violations = result.get("violations") or []
        if (
            event.kind == POST_TOOL
            and violations
            and all("classification unavailable" in v.lower() for v in violations)
            and creds.get("failure_mode") != "closed"
        ):
            # The result couldn't be classified: an outage, not a rule match. The tool
            # already ran, so follow IT's offline choice instead of flagging every call.
            return self.adapter.allow(event)
        reason = explain(result)
        if verdict == "escalate":
            return self.adapter.ask(event, reason)
        return self.adapter.deny(event, reason)

    def unavailable(self, event: Event, creds: dict, exc: ApiError) -> Output:
        """Aegis couldn't decide: auth failures always block, outages follow IT's choice."""
        a = self.adapter
        if exc.is_auth_failure:
            message = (
                f"[Aegis] This {a.label}'s Aegis connection is no longer valid ({exc.detail}). "
                "Ask IT for a new code, then send: aegis connect <code>"
            )
            blocking = True
        else:
            blocking = creds.get("failure_mode") == "closed"
            message = f"[Aegis] {exc.detail}. " + (
                "Your organization blocks tool calls until Aegis is reachable."
                if blocking
                else "Continuing without a governance check."
            )
        if blocking and event.kind in (PRE_TOOL, PROMPT):
            return a.deny(event, message)
        return a.notify(event, message)


def explain(result: dict) -> str:
    violations = "; ".join(result.get("violations") or []) or "Blocked by an organization rule"
    text = f"[Aegis] {violations}"
    if result.get("support_message"):
        text += f". {result['support_message']}"
    if result.get("support_email"):
        text += f" ({result['support_email']})"
    return text

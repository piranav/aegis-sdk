"""Framework-neutral facade for reporting sessions and LLM calls to Aegis."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

from aegis_sdk.telemetry.binding import current_binding
from aegis_sdk.telemetry.events import (
    AttributeValue,
    LlmCall,
    SessionEnd,
    SessionStart,
    SessionStatus,
    TokenUsage,
    ToolCallRef,
    ToolResultRef,
    utc_now,
)
from aegis_sdk.telemetry.exporter import BatchTelemetryExporter, TelemetryExporter

if TYPE_CHECKING:
    from aegis_sdk.client import AegisGatewayClient


class AegisTelemetry:
    """Report what each agent run does: its sessions, model calls, and token usage.

    Framework integrations call this for you; custom agents use ``session()``::

        telemetry = AegisTelemetry.from_client(gateway)
        with telemetry.session(input=prompt, customer_id="acme") as session:
            response = llm.create(...)
            session.record_llm_call(model=response.model, usage=TokenUsage(...))

    Set ``capture_content=False`` to send no prompt or answer text, only metadata.
    """

    def __init__(
        self,
        exporter: TelemetryExporter,
        *,
        framework: str | None = None,
        capture_content: bool = True,
    ) -> None:
        self.exporter = exporter
        self.framework = framework
        self.capture_content = capture_content

    @classmethod
    def from_client(
        cls,
        client: AegisGatewayClient,
        *,
        framework: str | None = None,
        capture_content: bool = True,
        **exporter_options: object,
    ) -> AegisTelemetry:
        """Send telemetry through ``client`` with a background batching exporter."""

        exporter = BatchTelemetryExporter(client.send_telemetry, **exporter_options)
        return cls(exporter, framework=framework, capture_content=capture_content)

    def start_session(
        self,
        session_id: str | None = None,
        *,
        input: str | None = None,  # noqa: A002 - mirrors the wire field
        customer_id: str | None = None,
        end_user_id: str | None = None,
        framework: str | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        occurred_at: datetime | None = None,
    ) -> str:
        """Report a session start; returns its id. Fields default from ``bind_session``."""

        binding = current_binding()
        resolved_id = session_id or binding.session_id or uuid4().hex
        self.exporter.export(
            SessionStart(
                session_id=resolved_id,
                occurred_at=occurred_at or utc_now(),
                customer_id=customer_id or binding.customer_id,
                end_user_id=end_user_id or binding.end_user_id,
                framework=framework or self.framework,
                input=self._content(input),
                attributes={**binding.attributes, **(attributes or {})},
            )
        )
        return resolved_id

    def record_llm_call(
        self,
        session_id: str,
        *,
        model: str,
        usage: TokenUsage | None = None,
        call_id: str | None = None,
        provider: str | None = None,
        response_model: str | None = None,
        agent_name: str | None = None,
        started_at: datetime | None = None,
        latency_ms: int | None = None,
        finish_reason: str | None = None,
        error_type: str | None = None,
        cost_usd: Decimal | float | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        occurred_at: datetime | None = None,
        tool_calls: Sequence[ToolCallRef] = (),
        tool_results: Sequence[ToolResultRef] = (),
        parent_tool_call_id: str | None = None,
    ) -> None:
        """Report one model invocation. ``call_id`` makes retries idempotent.

        ``tool_calls`` are the tools this response asked for; ``tool_results`` the tool
        results its input read for the first time (sizes only). Together they let Aegis
        attribute token spend to the tools, MCP servers, and skills that drove it.
        ``parent_tool_call_id`` marks a subagent's calls with the tool call that ran it.
        """

        self.exporter.export(
            LlmCall(
                session_id=session_id,
                occurred_at=occurred_at or utc_now(),
                call_id=call_id or uuid4().hex,
                model=model,
                provider=provider,
                response_model=response_model,
                agent_name=agent_name,
                usage=usage or TokenUsage(),
                started_at=started_at,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                error_type=error_type,
                cost_usd=Decimal(str(cost_usd)) if cost_usd is not None else None,
                attributes=dict(attributes or {}),
                tool_calls=list(tool_calls),
                tool_results=list(tool_results),
                parent_tool_call_id=parent_tool_call_id,
            )
        )

    def end_session(
        self,
        session_id: str,
        *,
        status: SessionStatus = "completed",
        output: str | None = None,
        cost_usd: Decimal | float | None = None,
        occurred_at: datetime | None = None,
    ) -> None:
        self.exporter.export(
            SessionEnd(
                session_id=session_id,
                occurred_at=occurred_at or utc_now(),
                status=status,
                output=self._content(output),
                cost_usd=Decimal(str(cost_usd)) if cost_usd is not None else None,
            )
        )

    @contextmanager
    def session(
        self,
        session_id: str | None = None,
        *,
        input: str | None = None,  # noqa: A002 - mirrors the wire field
        customer_id: str | None = None,
        end_user_id: str | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> Iterator[SessionHandle]:
        """Scope a session to a block; it ends as failed if the block raises."""

        handle = SessionHandle(
            self,
            self.start_session(
                session_id,
                input=input,
                customer_id=customer_id,
                end_user_id=end_user_id,
                attributes=attributes,
            ),
        )
        try:
            yield handle
        except BaseException:
            if not handle.ended:
                handle.end(status="failed")
            raise
        if not handle.ended:
            handle.end()

    def flush(self, timeout: float | None = None) -> bool:
        return self.exporter.flush(timeout)

    def shutdown(self, timeout: float | None = None) -> None:
        self.exporter.shutdown(timeout)

    def _content(self, text: str | None) -> str | None:
        return text if self.capture_content else None


class SessionHandle:
    """A started session; record calls against it and end it once."""

    def __init__(self, telemetry: AegisTelemetry, session_id: str) -> None:
        self._telemetry = telemetry
        self.id = session_id
        self.ended = False

    def record_llm_call(self, *, model: str, **details: object) -> None:
        self._telemetry.record_llm_call(self.id, model=model, **details)  # type: ignore[arg-type]

    def end(
        self,
        *,
        status: SessionStatus = "completed",
        output: str | None = None,
        cost_usd: Decimal | float | None = None,
    ) -> None:
        if self.ended:
            return
        self.ended = True
        self._telemetry.end_session(self.id, status=status, output=output, cost_usd=cost_usd)

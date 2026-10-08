"""Telemetry recorder, session binding, exporter, and client contract tests."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime

import httpx
import pytest

from aegis_sdk import AegisGatewayClient, AegisGatewayError, AegisTelemetry, bind_session
from aegis_sdk.telemetry import (
    BatchTelemetryExporter,
    InMemoryTelemetryExporter,
    LlmCall,
    SessionEnd,
    SessionStart,
    TokenUsage,
)

T0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def telemetry(**options) -> tuple[AegisTelemetry, InMemoryTelemetryExporter]:
    exporter = InMemoryTelemetryExporter()
    return AegisTelemetry(exporter, framework="custom", **options), exporter


# ------------------------------------------------------------------------------ recorder


def test_session_block_reports_start_calls_and_end() -> None:
    recorder, exporter = telemetry()

    with recorder.session(input="Summarise invoices", customer_id="acme") as session:
        session.record_llm_call(
            model="gpt-4.1-mini", call_id="resp_1", usage=TokenUsage(input_tokens=12)
        )

    start, call, end = exporter.events
    assert isinstance(start, SessionStart) and start.customer_id == "acme"
    assert (start.framework, start.input) == ("custom", "Summarise invoices")
    assert isinstance(call, LlmCall) and call.session_id == start.session_id == session.id
    assert isinstance(end, SessionEnd) and end.status == "completed"


def test_session_block_ends_failed_when_the_agent_raises() -> None:
    recorder, exporter = telemetry()

    with pytest.raises(RuntimeError), recorder.session():
        raise RuntimeError("tool exploded")

    assert exporter.events[-1].status == "failed"


def test_session_handle_ends_once() -> None:
    recorder, exporter = telemetry()

    with recorder.session() as session:
        session.end(status="interrupted")

    assert [type(e).__name__ for e in exporter.events] == ["SessionStart", "SessionEnd"]
    assert exporter.events[-1].status == "interrupted"


def test_bindings_nest_and_explicit_arguments_win() -> None:
    recorder, exporter = telemetry()

    with bind_session(customer_id="acme", attributes={"plan": "gold", "region": "eu"}):
        with bind_session(end_user_id="u-7", attributes={"plan": "trial"}):
            recorder.start_session("conv-1", attributes={"channel": "web"})
        recorder.start_session(customer_id="globex")

    inner, outer = exporter.events
    assert (inner.session_id, inner.customer_id, inner.end_user_id) == ("conv-1", "acme", "u-7")
    assert inner.attributes == {"plan": "trial", "region": "eu", "channel": "web"}
    assert outer.customer_id == "globex" and outer.end_user_id is None


def test_bound_session_id_is_used_when_none_is_given() -> None:
    recorder, exporter = telemetry()

    with bind_session(session_id="conversation-42"):
        assert recorder.start_session() == "conversation-42"


def test_capture_content_false_sends_metadata_only() -> None:
    recorder, exporter = telemetry(capture_content=False)

    with recorder.session(input="secret prompt") as session:
        session.end(output="secret answer")

    assert exporter.events[0].input is None and exporter.events[1].output is None


def test_payload_is_json_ready_and_omits_unset_fields() -> None:
    recorder, exporter = telemetry()

    recorder.record_llm_call("s-1", model="claude-sonnet-4-5", call_id="m-1", cost_usd=0.0125)
    payload = exporter.events[0].to_payload()

    assert payload["type"] == "llm.call" and payload["cost_usd"] == "0.0125"
    assert "provider" not in payload and "latency_ms" not in payload
    json.dumps(payload)


# ------------------------------------------------------------------------------ exporter


class RecordingSender:
    def __init__(self, failures: list[Exception] | None = None) -> None:
        self.batches: list[list[dict]] = []
        self.failures = list(failures or [])
        self.attempts = 0
        self.lock = threading.Lock()

    def __call__(self, events: list[dict]) -> None:
        with self.lock:
            self.attempts += 1
            if self.failures:
                raise self.failures.pop(0)
            self.batches.append(events)


def start_event(index: int) -> SessionStart:
    return SessionStart(session_id=f"s-{index}", occurred_at=T0)


def test_exporter_batches_by_size_and_flushes_the_remainder() -> None:
    sender = RecordingSender()
    exporter = BatchTelemetryExporter(sender, max_batch_size=2, schedule_delay=30)

    for index in range(5):
        exporter.export(start_event(index))
    assert exporter.flush(timeout=5)

    assert [len(batch) for batch in sender.batches] == [2, 2, 1]
    exporter.shutdown(timeout=5)


def test_exporter_sends_after_the_schedule_delay_without_a_flush() -> None:
    delivered = threading.Event()
    exporter = BatchTelemetryExporter(lambda events: delivered.set(), schedule_delay=0.05)

    exporter.export(start_event(1))

    assert delivered.wait(timeout=5)
    exporter.shutdown(timeout=5)


def test_exporter_retries_transient_failures() -> None:
    sender = RecordingSender(
        failures=[AegisGatewayError("down"), AegisGatewayError("busy", status_code=503)]
    )
    exporter = BatchTelemetryExporter(sender, retry_backoff=0.001)

    exporter.export(start_event(1))
    exporter.flush(timeout=5)

    assert (sender.attempts, len(sender.batches), exporter.dropped_events) == (3, 1, 0)
    exporter.shutdown(timeout=5)


def test_exporter_drops_batches_the_gateway_rejects() -> None:
    sender = RecordingSender(failures=[AegisGatewayError("bad key", status_code=401)])
    exporter = BatchTelemetryExporter(sender, retry_backoff=0.001)

    exporter.export(start_event(1))
    exporter.flush(timeout=5)

    assert (sender.attempts, exporter.dropped_events) == (1, 1)
    exporter.shutdown(timeout=5)


def test_exporter_never_blocks_when_the_queue_is_full() -> None:
    release = threading.Event()
    exporter = BatchTelemetryExporter(
        lambda events: release.wait(5), max_batch_size=1, max_queue_size=2, schedule_delay=30
    )

    for index in range(10):
        exporter.export(start_event(index))

    assert exporter.dropped_events >= 7
    release.set()
    exporter.shutdown(timeout=5)


def test_shutdown_drains_and_later_events_are_dropped() -> None:
    sender = RecordingSender()
    exporter = BatchTelemetryExporter(sender, schedule_delay=30)

    exporter.export(start_event(1))
    exporter.shutdown(timeout=5)
    exporter.export(start_event(2))

    assert [e["session_id"] for batch in sender.batches for e in batch] == ["s-1"]
    assert exporter.dropped_events == 1


# -------------------------------------------------------------------------------- client


def test_client_sends_telemetry_batches_with_the_agent_key() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["authorization"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(202, json={"accepted": 1, "duplicates": 0, "rejected": []})

    client = AegisGatewayClient(
        "http://aegis.test", "aegk_key", transport=httpx.MockTransport(handler)
    )
    result = client.send_telemetry([start_event(1).to_payload()])

    assert seen["path"] == "/v1/telemetry/events"
    assert seen["authorization"] == "Bearer aegk_key"
    assert seen["body"]["events"][0]["type"] == "session.start"
    assert result.accepted == 1


def test_from_client_delivers_through_the_gateway() -> None:
    received = []
    client = AegisGatewayClient(
        "http://aegis.test",
        "aegk_key",
        transport=httpx.MockTransport(
            lambda request: (
                received.append(json.loads(request.content))
                or httpx.Response(202, json={"accepted": 2, "duplicates": 0})
            )
        ),
    )
    recorder = AegisTelemetry.from_client(client, schedule_delay=30)

    with recorder.session() as session:
        assert session.id
    assert recorder.flush(timeout=5)

    assert [e["type"] for e in received[0]["events"]] == ["session.start", "session.end"]
    recorder.shutdown(timeout=5)

"""Exporters move telemetry events off the agent's hot path to the Aegis gateway."""

from __future__ import annotations

import atexit
import logging
import queue
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Protocol

from aegis_sdk.errors import AegisGatewayError

logger = logging.getLogger(__name__)

TelemetrySender = Callable[[list[dict[str, Any]]], object]


class Exportable(Protocol):
    """Anything the batch exporter can ship: telemetry events, inventory usage."""

    def to_payload(self) -> dict[str, Any]: ...


class TelemetryExporter(Protocol):
    def export(self, event: Exportable) -> None:
        """Accept an event without blocking the caller."""

    def flush(self, timeout: float | None = None) -> bool:
        """Deliver everything accepted so far; return False if the timeout expired."""

    def shutdown(self, timeout: float | None = None) -> None:
        """Flush and release resources. Later events are dropped."""


class InMemoryTelemetryExporter:
    """Keeps events in memory. Useful in tests and for inspecting what would be sent."""

    def __init__(self) -> None:
        self.events: list[Exportable] = []

    def export(self, event: Exportable) -> None:
        self.events.append(event)

    def flush(self, timeout: float | None = None) -> bool:
        return True

    def shutdown(self, timeout: float | None = None) -> None:
        return None


@dataclass
class _FlushRequest:
    done: threading.Event = field(default_factory=threading.Event)


_STOP = object()


class BatchTelemetryExporter:
    """Batches events on a background thread and sends them with bounded retries.

    Agent code never waits on the network: ``export`` only enqueues. The queue is
    bounded, so when the gateway is unreachable for long enough, new events are
    dropped (and counted in ``dropped_events``) instead of growing memory without
    limit. Transient failures (network errors, 429, 5xx) are retried with exponential
    backoff; other rejections drop the batch, since resending cannot succeed.
    """

    def __init__(
        self,
        send: TelemetrySender,
        *,
        max_batch_size: int = 100,
        schedule_delay: float = 2.0,
        max_queue_size: int = 10_000,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
    ) -> None:
        if max_batch_size < 1 or max_queue_size < 1:
            raise ValueError("max_batch_size and max_queue_size must be positive")
        self._send = send
        self._max_batch_size = max_batch_size
        self._schedule_delay = schedule_delay
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self.dropped_events = 0
        self._worker = threading.Thread(target=self._run, name="aegis-telemetry", daemon=True)
        self._worker.start()
        # Short-lived scripts exit before the next scheduled flush; drain on exit.
        self_ref = weakref.ref(self)
        atexit.register(lambda: (exporter := self_ref()) and exporter.shutdown(timeout=5.0))

    def export(self, event: Exportable) -> None:
        if self._closed.is_set():
            self._drop(1, "exporter is shut down")
            return
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._drop(1, "telemetry queue is full")

    def flush(self, timeout: float | None = None) -> bool:
        if self._closed.is_set():
            return not self._worker.is_alive()
        request = _FlushRequest()
        try:
            self._queue.put(request, timeout=timeout)
        except queue.Full:
            return False
        return request.done.wait(timeout)

    def shutdown(self, timeout: float | None = None) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._queue.put(_STOP, timeout=timeout)
        except queue.Full:
            logger.warning("Aegis telemetry queue stayed full during shutdown")
            return
        self._worker.join(timeout)

    def _run(self) -> None:
        batch: list[Exportable] = []
        deadline = monotonic()
        while True:
            wait = max(0.0, deadline - monotonic()) if batch else self._schedule_delay
            try:
                item = self._queue.get(timeout=wait)
            except queue.Empty:
                item = None

            if item is _STOP:
                self._deliver(batch)
                return
            if isinstance(item, _FlushRequest):
                self._deliver(batch)
                batch = []
                item.done.set()
                continue
            if item is not None:
                if not batch:
                    deadline = monotonic() + self._schedule_delay
                batch.append(item)
            if batch and (len(batch) >= self._max_batch_size or monotonic() >= deadline):
                self._deliver(batch)
                batch = []

    def _deliver(self, batch: list[Exportable]) -> None:
        if not batch:
            return
        payload = [event.to_payload() for event in batch]
        for attempt in range(self._max_retries + 1):
            try:
                self._send(payload)
                return
            except AegisGatewayError as exc:
                if not _is_transient(exc) or attempt == self._max_retries:
                    self._drop(len(batch), f"gateway rejected telemetry: {exc}")
                    return
            except Exception:  # noqa: BLE001 - telemetry must never crash the worker
                logger.exception("Unexpected error sending Aegis telemetry")
                self._drop(len(batch), "unexpected send error")
                return
            # Wakes early on shutdown so exit is never held up by a long backoff.
            self._closed.wait(self._retry_backoff * 2**attempt)

    def _drop(self, count: int, reason: str) -> None:
        with self._lock:
            self.dropped_events += count
        logger.warning("Dropped %d Aegis telemetry event(s): %s", count, reason)


def _is_transient(exc: AegisGatewayError) -> bool:
    return exc.status_code is None or exc.status_code == 429 or exc.status_code >= 500

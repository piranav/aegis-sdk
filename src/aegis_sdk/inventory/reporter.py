"""Report an agent's components and their usage to Aegis without blocking the agent."""

from __future__ import annotations

import atexit
import logging
import threading
import weakref
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any

from aegis_sdk.inventory.components import (
    Component,
    Manifest,
    PathSegment,
    UsageEvent,
    merge_components,
)
from aegis_sdk.telemetry.exporter import BatchTelemetryExporter, TelemetryExporter

if TYPE_CHECKING:
    from aegis_sdk.client import AegisGatewayClient

logger = logging.getLogger(__name__)


class AegisInventory:
    """Tell Aegis what an agent is built from (its manifest) and what it uses.

    Framework integrations call this for you. Manifests are sent on a background thread
    and only when they change, so describing the agent on every run is cheap. Usage is
    batched like telemetry. Neither ever raises into agent code::

        inventory = AegisInventory.from_client(gateway)
        hooks = AegisRunHooks(telemetry=telemetry, inventory=inventory)

    ``scope`` names the configuration this process reports. A manifest replaces only
    earlier manifests with the same scope, so give each separately configured
    deployment of one agent its own scope.
    """

    def __init__(
        self,
        send_manifest: Any,
        usage_exporter: TelemetryExporter,
        *,
        scope: str = "runtime",
        framework: str | None = None,
    ) -> None:
        self._send_manifest = send_manifest
        self.usage_exporter = usage_exporter
        self.scope = scope
        self.framework = framework
        self._lock = threading.Lock()
        self._sent_digests: dict[str, str] = {}
        self._pending: set[Future[Any]] = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aegis-inventory")
        self_ref = weakref.ref(self)
        atexit.register(lambda: (inventory := self_ref()) and inventory.shutdown(timeout=5.0))

    @classmethod
    def from_client(
        cls,
        client: AegisGatewayClient,
        *,
        scope: str = "runtime",
        framework: str | None = None,
        **exporter_options: Any,
    ) -> AegisInventory:
        exporter = BatchTelemetryExporter(client.send_inventory_usage, **exporter_options)
        return cls(client.put_inventory_manifest, exporter, scope=scope, framework=framework)

    def report_manifest(
        self,
        components: Sequence[Component],
        *,
        framework: str | None = None,
        scope: str | None = None,
    ) -> bool:
        """Send the agent's manifest if it changed since the last send; True if queued."""

        manifest = Manifest(
            scope=scope or self.scope,
            framework=framework or self.framework,
            components=merge_components(components),
        )
        digest = manifest.digest()
        with self._lock:
            if self._sent_digests.get(manifest.scope) == digest:
                return False
            # Claimed before sending so concurrent runs don't queue duplicates.
            self._sent_digests[manifest.scope] = digest
            future = self._executor.submit(self._deliver, manifest, digest)
            self._pending.add(future)
        future.add_done_callback(self._pending.discard)
        return True

    def record_usage(self, path: Sequence[PathSegment], *, count: int = 1) -> None:
        """Report ``count`` uses of the component at ``path``, e.g. ``[("tool", "x")]``."""

        if path:
            self.usage_exporter.export(UsageEvent.of(path, count=count))

    def flush(self, timeout: float | None = None) -> bool:
        with self._lock:
            pending = list(self._pending)
        done, not_done = wait(pending, timeout=timeout)
        return not not_done and self.usage_exporter.flush(timeout)

    def shutdown(self, timeout: float | None = None) -> None:
        self.flush(timeout)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.usage_exporter.shutdown(timeout)

    def _deliver(self, manifest: Manifest, digest: str) -> None:
        try:
            self._send_manifest(manifest.to_payload())
        except Exception:  # noqa: BLE001 - inventory must never break the agent
            logger.warning("Aegis could not record the agent manifest", exc_info=True)
            with self._lock:
                # Forget the digest so the next report retries.
                if self._sent_digests.get(manifest.scope) == digest:
                    del self._sent_digests[manifest.scope]

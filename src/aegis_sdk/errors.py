"""Errors raised by the Aegis SDK."""

from __future__ import annotations

import httpx


class AegisGatewayError(RuntimeError):
    """Raised when the Aegis gateway rejects or cannot process a request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response: httpx.Response | None = None,
    ) -> None:
        self.status_code = status_code
        self.response = response
        super().__init__(message)

"""Aegis API access and per-assistant connection state."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from base64 import urlsafe_b64decode
from pathlib import Path

from aegis_plugin import VERSION

TIMEOUT = float(os.environ.get("AEGIS_TIMEOUT_SECONDS", "8"))


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail

    @property
    def is_auth_failure(self) -> bool:
        return self.status in (401, 403)


def request(
    api_url: str, path: str, body: dict, key: str | None = None, method: str = "POST"
) -> dict:
    headers = {"Content-Type": "application/json", "User-Agent": f"aegis-plugin/{VERSION}"}
    if key:
        headers["X-API-Key"] = key
    req = urllib.request.Request(
        api_url.rstrip("/") + path, data=json.dumps(body).encode(), headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("detail", exc.reason)
        except ValueError:
            detail = exc.reason
        raise ApiError(exc.code, str(detail)) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ApiError(0, f"Aegis is unreachable: {exc}") from None


def decode_code(code: str) -> str:
    """Return the API address embedded in a connection code."""
    if not code.startswith("aegc_"):
        raise ValueError("That isn't an Aegis connection code (it should start with aegc_).")
    raw = code[5:]
    decoded = urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode()
    api_url, _, token = decoded.rpartition("|")
    if not api_url or not token:
        raise ValueError("The connection code is incomplete; copy it again from your invite.")
    return api_url


class State:
    """Credentials and in-flight evaluations for one assistant on this machine."""

    def __init__(self, tool: str):
        base = os.environ.get("AEGIS_STATE_DIR")
        self.dir = Path(base) if base else Path.home() / ".aegis" / tool
        self.credentials_path = self.dir / "credentials.json"
        self.pending = self.dir / "pending"

    def credentials(self) -> dict | None:
        try:
            return json.loads(self.credentials_path.read_text())
        except (OSError, ValueError):
            return None

    def save(self, data: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = self.credentials_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle)
        os.replace(tmp, self.credentials_path)

    def forget(self) -> None:
        self.credentials_path.unlink(missing_ok=True)

    def remember_audit(self, session_id: str, tool_use_id: str, audit_id: str) -> None:
        folder = self.pending / _safe(session_id)
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        (folder / _safe(tool_use_id)).write_text(audit_id)

    def recall_audit(self, session_id: str, tool_use_id: str) -> str | None:
        path = self.pending / _safe(session_id) / _safe(tool_use_id)
        try:
            audit_id = path.read_text().strip()
            path.unlink()
            return audit_id or None
        except OSError:
            return None

    def inventory_digest(self, scope: str) -> str | None:
        return read_json_file(self.dir / "inventory.json").get(scope)

    def remember_inventory(self, scope: str, digest: str) -> None:
        path = self.dir / "inventory.json"
        digests = read_json_file(path)
        digests[scope] = digest
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(digests))
        os.replace(tmp, path)

    def usage_cursor(self, session_id: str) -> dict:
        return read_json_file(self.dir / "usage" / f"{_safe(session_id)}.json")

    def save_usage_cursor(self, session_id: str, cursor: dict) -> None:
        folder = self.dir / "usage"
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = folder / f"{_safe(session_id)}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cursor))
        os.replace(tmp, path)

    def forget_usage_cursor(self, session_id: str) -> None:
        (self.dir / "usage" / f"{_safe(session_id)}.json").unlink(missing_ok=True)

    def clear_session(self, session_id: str) -> None:
        folder = self.pending / _safe(session_id)
        if folder.is_dir():
            for child in folder.iterdir():
                child.unlink(missing_ok=True)
            folder.rmdir()


def read_json_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (value or "unknown"))[:120]

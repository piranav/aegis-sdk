"""Components an agent is built from, as reported to ``/v1/inventory``.

Framework integrations build these for you. Build them yourself for a custom agent::

    Component.mcp_server("github", transport="http", locator=url, tools=["create_issue"])

Only describe *what* a component is, never how to authenticate to it: there is no
field for environment variables, headers, or arguments, and ``locator`` is reduced to
scheme, host, and path (or an executable name) before it leaves the process.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field

from aegis_sdk.telemetry.events import AttributeValue, utc_now

ComponentKind = Literal[
    "model",
    "tool",
    "mcp_server",
    "connector",
    "skill",
    "plugin",
    "subagent",
    "remote_agent",
]
PathSegment = tuple[ComponentKind, str]


def sanitize_locator(value: str | None) -> str | None:
    """URLs keep scheme, host, port, and path; commands keep only the executable name."""

    if value is None or not str(value).strip():
        return None
    value = str(value).strip()
    parts = urlsplit(value)
    if parts.scheme and parts.netloc:
        host = parts.hostname or ""
        netloc = f"{host}:{parts.port}" if parts.port else host
        return urlunsplit((parts.scheme.lower(), netloc, parts.path, "", ""))
    executable = value.split()[0]
    return executable.rstrip("/").rsplit("/", 1)[-1] or executable


def fingerprint(*parts: Any) -> str:
    """Short stable hash of a definition, so Aegis can tell when it changes."""

    canonical = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


class Component(BaseModel):
    kind: ComponentKind
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    version: str | None = None
    provider: str | None = None
    transport: str | None = None
    locator: str | None = None
    fingerprint: str | None = None
    attributes: dict[str, AttributeValue] = Field(default_factory=dict)
    children: list[Component] = Field(default_factory=list)

    def model_post_init(self, _context: object) -> None:
        self.locator = sanitize_locator(self.locator)
        if self.description and len(self.description) > 1000:
            self.description = self.description[:999] + "…"

    # Constructors for the common kinds keep integration code declarative.

    @classmethod
    def model(cls, name: str, *, provider: str | None = None) -> Component:
        return cls(kind="model", name=name, provider=provider)

    @classmethod
    def tool(
        cls,
        name: str,
        *,
        description: str | None = None,
        schema: Any = None,
        **fields: Any,
    ) -> Component:
        return cls(
            kind="tool",
            name=name,
            description=description,
            fingerprint=fields.pop("fingerprint", None) or fingerprint(description, schema),
            **fields,
        )

    @classmethod
    def mcp_server(
        cls,
        name: str,
        *,
        tools: Iterable[Component | str] = (),
        **fields: Any,
    ) -> Component:
        children = [t if isinstance(t, Component) else cls.tool(t) for t in tools]
        return cls(kind="mcp_server", name=name, children=children, **fields)

    def merge(self, other: Component) -> Component:
        """Combine two declarations of the same component; children are unioned."""

        children = {(c.kind, c.name): c for c in self.children}
        for child in other.children:
            key = (child.kind, child.name)
            children[key] = children[key].merge(child) if key in children else child
        attributes = {**self.attributes, **other.attributes}
        return self.model_copy(
            update={
                "description": self.description or other.description,
                "version": self.version or other.version,
                "provider": self.provider or other.provider,
                "transport": self.transport or other.transport,
                "locator": self.locator or other.locator,
                "fingerprint": self.fingerprint or other.fingerprint,
                "attributes": attributes,
                "children": list(children.values()),
            }
        )


def merge_components(components: Iterable[Component]) -> list[Component]:
    """Deduplicate top-level components by (kind, name), merging repeated declarations."""

    merged: dict[tuple[str, str], Component] = {}
    for component in components:
        key = (component.kind, component.name)
        merged[key] = merged[key].merge(component) if key in merged else component
    return sorted(merged.values(), key=lambda c: (c.kind, c.name))


class Manifest(BaseModel):
    scope: str = "runtime"
    framework: str | None = None
    components: list[Component] = Field(default_factory=list)

    def digest(self) -> str:
        return fingerprint(self.model_dump(mode="json"))

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True, exclude_defaults=False)


class UsageEvent(BaseModel):
    """``count`` uses of the component at ``path`` (root first)."""

    component: list[dict[str, str]]
    count: int = Field(default=1, ge=1)
    occurred_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def of(cls, path: Sequence[PathSegment], *, count: int = 1) -> UsageEvent:
        return cls(component=[{"kind": kind, "name": name} for kind, name in path], count=count)

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

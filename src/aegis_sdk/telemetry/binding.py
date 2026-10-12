"""Attribute the sessions an agent run creates to a customer, end user, or session id.

Bindings live in a ``ContextVar``, so they follow the code that runs the agent:
framework hooks invoked inside ``with bind_session(...)`` pick them up without any
framework-specific plumbing, and concurrent runs in other tasks stay separate.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from aegis_sdk.telemetry.events import AttributeValue


@dataclass(frozen=True)
class SessionBinding:
    session_id: str | None = None
    customer_id: str | None = None
    end_user_id: str | None = None
    attributes: Mapping[str, AttributeValue] = field(default_factory=dict)

    def merged_with(self, inner: SessionBinding) -> SessionBinding:
        """Inner bindings refine outer ones: set fields win, attributes merge."""

        return SessionBinding(
            session_id=inner.session_id or self.session_id,
            customer_id=inner.customer_id or self.customer_id,
            end_user_id=inner.end_user_id or self.end_user_id,
            attributes={**self.attributes, **inner.attributes},
        )


_current: ContextVar[SessionBinding] = ContextVar(
    "aegis_session_binding", default=SessionBinding()
)


def current_binding() -> SessionBinding:
    return _current.get()


@contextmanager
def bind_session(
    *,
    session_id: str | None = None,
    customer_id: str | None = None,
    end_user_id: str | None = None,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> Iterator[SessionBinding]:
    """Attribute sessions started in this context to a customer and end user.

    Aegis does not infer attribution; this is how you declare it. Values are sent on
    the session's start event, and every model call and governed tool call in the run
    is attributed through the session.

    - ``customer_id``: your stable account id. The first value a session receives
      wins, so start a separate session per customer.
    - ``end_user_id``: a stable, pseudonymous id for the person, never an email.
    - ``attributes``: up to 32 low-sensitivity scalars to filter by (plan, region).
    - ``session_id``: pins the session identity, e.g. to your conversation id so a
      multi-turn chat is one session. Without it, each run gets a generated id.

    Attribution is self-reported and used for reporting only, never for access
    control; set it from authenticated application context, not model output.

    ::

        with bind_session(customer_id="acme", end_user_id=user.id):
            result = await Runner.run(agent, prompt, hooks=hooks)
    """

    binding = current_binding().merged_with(
        SessionBinding(
            session_id=session_id,
            customer_id=customer_id,
            end_user_id=end_user_id,
            attributes=dict(attributes or {}),
        )
    )
    token = _current.set(binding)
    try:
        yield binding
    finally:
        _current.reset(token)

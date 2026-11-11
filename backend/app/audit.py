"""Append-only audit log (ADR-0014).

Events hold identifiers and actions only, never clinical content. The store exposes only
``append`` and ``read``. ``record`` is the single helper that writes events and it is called
from the access-control layer (``access.py``), so endpoints cannot forget to audit.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol

ALLOWED = "allowed"
DENIED = "denied"
FAILURE = "failure"
NOT_FOUND = "not_found"


@dataclass(frozen=True)
class AuditEvent:
    id: int
    timestamp: str
    actor_id: str | None
    actor_role: str | None
    action: str
    resource_type: str
    resource_id: str | None
    patient_id: str | None
    outcome: str
    request_id: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditStore(Protocol):
    def append(self, event: AuditEvent) -> AuditEvent: ...

    def read(
        self,
        actor: str | None = None,
        patient: str | None = None,
        action: str | None = None,
        outcome: str | None = None,
    ) -> tuple[AuditEvent, ...]: ...


class InMemoryAuditStore:
    """Append and read only. (H34 swaps this for an append-only Postgres table.)"""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._lock = threading.Lock()

    def append(self, event: AuditEvent) -> AuditEvent:
        """Store the event, assigning its sequence number; returns the stored copy."""
        with self._lock:
            stored = replace(event, id=len(self._events) + 1)
            self._events.append(stored)
            return stored

    def read(
        self,
        actor: str | None = None,
        patient: str | None = None,
        action: str | None = None,
        outcome: str | None = None,
    ) -> tuple[AuditEvent, ...]:
        with self._lock:
            snapshot = tuple(self._events)
        wanted = (
            ("actor_id", actor),
            ("patient_id", patient),
            ("action", action),
            ("outcome", outcome),
        )
        return tuple(e for e in snapshot if all(v is None or getattr(e, f) == v for f, v in wanted))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def record(
    store: AuditStore,
    *,
    request_id: str,
    actor_id: str | None,
    actor_role: str | None,
    action: str,
    resource_type: str,
    outcome: str,
    resource_id: str | None = None,
    patient_id: str | None = None,
    now: Callable[[], str] = _utc_now,
) -> AuditEvent:
    """The one place audit events are created."""
    event = AuditEvent(
        id=0,
        timestamp=now(),
        actor_id=actor_id,
        actor_role=actor_role,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        patient_id=patient_id,
        outcome=outcome,
        request_id=request_id,
    )
    return store.append(event)

"""Postgres implementations of the three interfaces the app depends on (H17).

``PostgresRepository`` satisfies the ``PatientRepository`` Protocol, ``PostgresAuditStore`` the
``AuditStore`` Protocol, and ``PostgresUserStore`` subclasses ``UserStore`` so that every
caller keeps the type it already had. Nothing outside this package knows SQL exists.

Reads are short, read-only transactions; the only writes are a user upsert, a user's active
flag, and an audit insert.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..audit import AuditEvent
from ..data.models import Patient
from ..storage import POSTGRES, StorageBackend
from ..users import User, UserStore, normalize_email
from . import queries, rows
from .engine import create_engine


def _text(statement: str) -> Any:
    from sqlalchemy import text

    return text(statement)


@dataclass(frozen=True)
class _Connected:
    """Shared engine access: every store holds one of these, nothing else."""

    engine: Any

    def rows(self, statement: str, **params: Any) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            result = connection.execute(_text(statement), params)
            return [dict(row) for row in result.mappings()]

    def one(self, statement: str, **params: Any) -> dict[str, Any] | None:
        found = self.rows(statement, **params)
        return found[0] if found else None

    def scalar(self, statement: str, **params: Any) -> Any:
        with self.engine.connect() as connection:
            return connection.execute(_text(statement), params).scalar()

    def write_scalar(self, statement: str, **params: Any) -> Any:
        with self.engine.begin() as connection:
            return connection.execute(_text(statement), params).scalar()

    def execute(self, statement: str, **params: Any) -> None:
        with self.engine.begin() as connection:
            connection.execute(_text(statement), params)


class PostgresRepository:
    """Patients, their conditions, medications, observations and doctor assignment."""

    source = "postgres"

    def __init__(self, engine: Any) -> None:
        self._db = _Connected(engine)

    def _hydrate(self, row: dict[str, Any]) -> Patient:
        patient_id = row["id"]
        doctors = self._db.rows(queries.SELECT_ASSIGNED_DOCTOR, patient_id=patient_id)
        conditions = self._db.rows(queries.SELECT_CONDITIONS, patient_id=patient_id)
        medications = self._db.rows(queries.SELECT_MEDICATIONS, patient_id=patient_id)
        observations = self._db.rows(queries.SELECT_OBSERVATIONS, patient_id=patient_id)
        return rows.to_patient(
            row,
            conditions=[c["condition_id"] for c in conditions],
            assigned_doctor=doctors[0]["doctor_id"] if doctors else "",
            medications=[rows.to_medication(m) for m in medications],
            observations=[rows.to_observation(o) for o in observations],
        )

    def get(self, patient_id: str) -> Patient | None:
        row = self._db.one(queries.SELECT_PATIENT, id=patient_id)
        return self._hydrate(row) if row else None

    def list_all(self) -> tuple[Patient, ...]:
        return tuple(self._hydrate(r) for r in self._db.rows(queries.SELECT_PATIENTS))

    def list_for_doctor(self, doctor_id: str, query: str | None = None) -> tuple[Patient, ...]:
        found = self._db.rows(queries.SELECT_PATIENTS_FOR_DOCTOR, doctor_id=doctor_id)
        patients = (self._hydrate(r) for r in found)
        return tuple(p for p in patients if rows.matches_doctor(p, query))


class PostgresUserStore(UserStore):
    """The same interface as the in-memory store, backed by the ``users`` table."""

    def __init__(self, engine: Any) -> None:
        super().__init__(())
        self._db = _Connected(engine)

    def get(self, user_id: str) -> User | None:
        row = self._db.one(queries.SELECT_USER, id=user_id)
        return rows.to_user(row) if row else None

    def by_email(self, email: str) -> User | None:
        row = self._db.one(queries.SELECT_USER_BY_EMAIL, email=normalize_email(email))
        return rows.to_user(row) if row else None

    def all(self) -> tuple[User, ...]:
        return tuple(rows.to_user(r) for r in self._db.rows(queries.SELECT_USERS))

    def add(self, user: User) -> User:
        self._db.execute(queries.INSERT_USER, **rows.user_params(user))
        return user

    def set_active(self, user_id: str, active: bool) -> User | None:
        current = self.get(user_id)
        if current is None:
            return None
        self._db.execute(queries.SET_USER_ACTIVE, id=user_id, active=active)
        return replace(current, active=active)

    def active_admin_count(self) -> int:
        return int(self._db.scalar(queries.COUNT_ACTIVE_ADMINS) or 0)


class PostgresAuditStore:
    """Append-only in behaviour and in the schema: this class only inserts and selects."""

    def __init__(self, engine: Any) -> None:
        self._db = _Connected(engine)

    def append(self, event: AuditEvent) -> AuditEvent:
        new_id = self._db.write_scalar(queries.INSERT_AUDIT_EVENT, **rows.audit_params(event))
        return replace(event, id=int(new_id))

    def read(
        self,
        actor: str | None = None,
        patient: str | None = None,
        action: str | None = None,
        outcome: str | None = None,
    ) -> tuple[AuditEvent, ...]:
        found = self._db.rows(
            queries.SELECT_AUDIT_EVENTS,
            actor=actor,
            patient=patient,
            action=action,
            outcome=outcome,
        )
        return tuple(rows.to_audit_event(r) for r in found)


def open_postgres(database_url: str) -> StorageBackend:
    """Build the three Postgres-backed stores on one shared engine."""
    engine = create_engine(database_url)
    return StorageBackend(
        name=POSTGRES,
        repo=PostgresRepository(engine),
        audit=PostgresAuditStore(engine),
        users=PostgresUserStore(engine),
    )

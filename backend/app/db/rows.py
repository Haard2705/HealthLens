"""Database rows to and from the frozen dataclasses. Pure functions, no connection involved.

Keeping the mapping here means it can be tested without Postgres, and the stores stay a thin
layer of queries over it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

from ..audit import AuditEvent
from ..data.models import Medication, Observation, Patient
from ..users import Role, User

Row = Mapping[str, Any]


def _iso_date(value: Any) -> str:
    return value.isoformat() if isinstance(value, date) else str(value)


def to_medication(row: Row) -> Medication:
    return Medication(
        name=row["name"],
        dose=row["dose"],
        frequency=row["frequency"],
        started=_iso_date(row["started"]),
    )


def to_observation(row: Row) -> Observation:
    return Observation(
        date=_iso_date(row["observed_on"]),
        bmi=float(row["bmi"]),
        systolic_bp=int(row["systolic_bp"]),
        hba1c=float(row["hba1c"]),
    )


def to_patient(
    row: Row,
    conditions: Sequence[str] = (),
    assigned_doctor: str = "",
    medications: Sequence[Medication] = (),
    observations: Sequence[Observation] = (),
) -> Patient:
    return Patient(
        id=row["id"],
        name=row["name"],
        age=int(row["age"]),
        sex=row["sex"],
        bmi=float(row["bmi"]),
        smoking_status=row["smoking_status"],
        physically_active=bool(row["physically_active"]),
        general_health=int(row["general_health"]),
        conditions=tuple(conditions),
        assigned_doctor=assigned_doctor,
        medications=tuple(medications),
        observations=tuple(observations),
        is_synthetic=bool(row["is_synthetic"]),
    )


def patient_params(patient: Patient) -> dict[str, Any]:
    return {
        "id": patient.id,
        "name": patient.name,
        "age": patient.age,
        "sex": patient.sex,
        "bmi": patient.bmi,
        "smoking_status": patient.smoking_status,
        "physically_active": patient.physically_active,
        "general_health": patient.general_health,
        "is_synthetic": patient.is_synthetic,
    }


def to_user(row: Row) -> User:
    return User(
        id=row["id"],
        email=row["email"],
        display_name=row["display_name"],
        role=Role(row["role"]),
        password_hash=row["password_hash"],
        active=bool(row["active"]),
        patient_id=row["patient_id"],
        doctor_id=row["doctor_id"],
    )


def user_params(user: User) -> dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "role": user.role.value,
        "password_hash": user.password_hash,
        "active": user.active,
        "patient_id": user.patient_id,
        "doctor_id": user.doctor_id,
    }


def to_audit_event(row: Row) -> AuditEvent:
    occurred = row["occurred_at"]
    timestamp = (
        occurred.isoformat(timespec="seconds") if isinstance(occurred, datetime) else str(occurred)
    )
    return AuditEvent(
        id=int(row["id"]),
        timestamp=timestamp,
        actor_id=row["actor_id"],
        actor_role=row["actor_role"],
        action=row["action"],
        resource_type=row["resource_type"],
        resource_id=row["resource_id"],
        patient_id=row["patient_id"],
        outcome=row["outcome"],
        request_id=row["request_id"],
    )


def audit_params(event: AuditEvent) -> dict[str, Any]:
    """The event's own ISO timestamp is kept; the database assigns the sequence number."""
    return {
        "occurred_at": event.timestamp,
        "actor_id": event.actor_id,
        "actor_role": event.actor_role,
        "action": event.action,
        "resource_type": event.resource_type,
        "resource_id": event.resource_id,
        "patient_id": event.patient_id,
        "outcome": event.outcome,
        "request_id": event.request_id,
    }


def matches_doctor(patient: Patient, query: str | None) -> bool:
    """The same name/id search ``SyntheticRepository.list_for_doctor`` applies."""
    needle = (query or "").strip().lower()
    return not needle or needle in patient.name.lower() or needle in patient.id.lower()

"""Pydantic request/response models."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from .data.models import Medication, Observation, Patient
from .users import EMAIL_PATTERN, Role, User, normalize_email

Email = Annotated[str, StringConstraints(max_length=254, strip_whitespace=True)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LoginRequest(_Strict):
    email: Email = Field(min_length=3)
    password: str = Field(min_length=1, max_length=256)


class UserOut(_Strict):
    id: str
    email: str
    display_name: str
    role: Role
    active: bool
    patient_id: str | None = None

    @classmethod
    def of(cls, user: User) -> UserOut:
        return cls(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            role=user.role,
            active=user.active,
            patient_id=user.patient_id,
        )


class InviteRequest(_Strict):
    email: Email = Field(pattern=EMAIL_PATTERN)
    role: Role
    display_name: str = Field(min_length=1, max_length=120)
    patient_id: str | None = Field(default=None, max_length=32)

    @field_validator("email")
    @classmethod
    def _lower(cls, value: str) -> str:
        return normalize_email(value)


class UserPatch(_Strict):
    active: bool


class PatientSummary(_Strict):
    id: str
    name: str
    age: int
    sex: str
    conditions: tuple[str, ...]
    assigned_doctor: str
    is_synthetic: bool

    @classmethod
    def of(cls, p: Patient) -> PatientSummary:
        return cls(
            id=p.id,
            name=p.name,
            age=p.age,
            sex=p.sex,
            conditions=p.conditions,
            assigned_doctor=p.assigned_doctor,
            is_synthetic=p.is_synthetic,
        )


class PatientRecord(PatientSummary):
    bmi: float
    smoking_status: str
    physically_active: bool
    general_health: int

    @classmethod
    def of(cls, p: Patient) -> PatientRecord:
        return cls(
            **PatientSummary.of(p).model_dump(),
            bmi=p.bmi,
            smoking_status=p.smoking_status,
            physically_active=p.physically_active,
            general_health=p.general_health,
        )


class MedicationOut(_Strict):
    name: str
    dose: str
    frequency: str
    started: str

    @classmethod
    def of(cls, m: Medication) -> MedicationOut:
        return cls(name=m.name, dose=m.dose, frequency=m.frequency, started=m.started)


class ObservationOut(_Strict):
    date: str
    bmi: float
    systolic_bp: int
    hba1c: float

    @classmethod
    def of(cls, o: Observation) -> ObservationOut:
        return cls(date=o.date, bmi=o.bmi, systolic_bp=o.systolic_bp, hba1c=o.hba1c)


class TimelineEvent(_Strict):
    date: str
    type: str
    summary: str

"""Domain records. Field names loosely follow the FHIR resources they map to (ADR-0012)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Medication:  # ~ FHIR MedicationStatement
    name: str
    dose: str
    frequency: str
    started: str  # ISO date


@dataclass(frozen=True)
class Observation:  # ~ FHIR Observation (one quarterly check-up)
    date: str  # ISO date
    bmi: float
    systolic_bp: int
    hba1c: float


@dataclass(frozen=True)
class Patient:  # ~ FHIR Patient (+ Condition codes as registry condition ids)
    id: str
    name: str
    age: int
    sex: str
    bmi: float
    smoking_status: str
    physically_active: bool
    general_health: int  # 1 (excellent) .. 5 (poor), as in BRFSS
    conditions: tuple[str, ...]
    assigned_doctor: str
    medications: tuple[Medication, ...]
    observations: tuple[Observation, ...]
    is_synthetic: bool = True


@dataclass(frozen=True)
class Doctor:
    id: str
    name: str

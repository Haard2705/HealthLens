"""The patient data interface (ADR-0012). The API only ever talks to this Protocol."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from .models import Patient
from .synthetic import build_patients


class PatientRepository(Protocol):
    source: str

    def get(self, patient_id: str) -> Patient | None: ...

    def list_all(self) -> tuple[Patient, ...]: ...

    def list_for_doctor(self, doctor_id: str, query: str | None = None) -> tuple[Patient, ...]: ...


class SyntheticRepository:
    """In-memory, deterministic, fake data."""

    source = "synthetic"

    def __init__(self, patients: Iterable[Patient] | None = None) -> None:
        items = tuple(patients) if patients is not None else build_patients()
        self._by_id = {p.id: p for p in items}

    def get(self, patient_id: str) -> Patient | None:
        return self._by_id.get(patient_id)

    def list_all(self) -> tuple[Patient, ...]:
        return tuple(self._by_id.values())

    def list_for_doctor(self, doctor_id: str, query: str | None = None) -> tuple[Patient, ...]:
        needle = (query or "").strip().lower()
        return tuple(
            p
            for p in self._by_id.values()
            if p.assigned_doctor == doctor_id
            and (not needle or needle in p.name.lower() or needle in p.id.lower())
        )

"""Deterministic synthetic patients (ALL data here is fake). Independent of the API layer.

To use real data later, implement ``PatientRepository`` (see repository.py) over the real
store and pass it to ``create_app``; nothing in this module is imported by the API except
through the repository and the demo-account seeding.
"""

from __future__ import annotations

import random
from datetime import date

from .models import Doctor, Medication, Observation, Patient

DOCTORS = (Doctor("dr-chen", "Dr. Wei Chen"), Doctor("dr-okafor", "Dr. Ngozi Okafor"))

# id, name, age, sex, bmi, smoking, active, general health, conditions, doctor
_ROWS = (
    ("P-0101", "Jordan Avery", 58, "M", 31.4, "former", False, 3, ("hypertension",), "dr-chen"),
    ("P-0102", "Priya Nair", 46, "F", 27.2, "never", True, 2, (), "dr-chen"),
    (
        "P-0103", "Samuel Boateng", 67, "M", 33.8, "daily", False, 4,
        ("diabetes", "hypertension", "copd"), "dr-chen",
    ),
    ("P-0104", "Elena Petrova", 39, "F", 24.1, "never", True, 1, (), "dr-chen"),
    ("P-0105", "Marcus Lee", 52, "M", 29.9, "former", True, 2, ("hypertension",), "dr-chen"),
    (
        "P-0106", "Fatima Hassan", 61, "F", 35.2, "never", False, 3,
        ("diabetes", "arthritis"), "dr-chen",
    ),
    ("P-0107", "Tomas Rivera", 44, "M", 26.0, "daily", True, 2, (), "dr-okafor"),
    (
        "P-0108", "Grace Mwangi", 71, "F", 28.6, "former", False, 3,
        ("arthritis", "heart_disease"), "dr-okafor",
    ),
)  # fmt: skip

_MEDS_BY_CONDITION = {
    "hypertension": (("Lisinopril", "10 mg", "once daily"),),
    "diabetes": (("Metformin", "500 mg", "twice daily"),),
    "copd": (("Albuterol inhaler", "90 mcg", "as needed"),),
    "arthritis": (("Naproxen", "250 mg", "as needed"),),
    "heart_disease": (("Atorvastatin", "20 mg", "once daily"), ("Aspirin", "81 mg", "once daily")),
}
_BASELINE_MEDS = (("Vitamin D", "1000 IU", "once daily"),)
_QUARTERS = 12
_FIRST_VISIT = date(2024, 1, 15)


def _quarter_date(index: int) -> str:
    months = index * 3
    year = _FIRST_VISIT.year + (_FIRST_VISIT.month - 1 + months) // 12
    month = (_FIRST_VISIT.month - 1 + months) % 12 + 1
    return date(year, month, _FIRST_VISIT.day).isoformat()


def _medications(conditions: tuple[str, ...]) -> tuple[Medication, ...]:
    triples = list(_BASELINE_MEDS)
    for condition in conditions:
        triples.extend(_MEDS_BY_CONDITION.get(condition, ()))
    return tuple(
        Medication(name, dose, freq, _quarter_date(i % 4))
        for i, (name, dose, freq) in enumerate(triples)
    )


def _observations(
    patient_id: str, age: int, bmi: float, conditions: tuple[str, ...]
) -> tuple[Observation, ...]:
    rng = random.Random(patient_id)  # string seed: stable across runs and platforms
    drift = rng.uniform(-0.15, 0.25)
    systolic_base = 116 + max(age - 40, 0) * 0.35 + (16 if "hypertension" in conditions else 0)
    hba1c_base = 7.3 if "diabetes" in conditions else 5.2 + max(bmi - 22, 0) * 0.02
    rows = []
    for i in range(_QUARTERS):
        behind = _QUARTERS - 1 - i  # the last quarter equals the patient's current values
        rows.append(
            Observation(
                date=_quarter_date(i),
                bmi=round(bmi - drift * behind + rng.uniform(-0.2, 0.2) * (behind > 0), 1),
                systolic_bp=round(systolic_base + rng.uniform(-6, 6)),
                hba1c=round(hba1c_base + rng.uniform(-0.3, 0.3), 1),
            )
        )
    return tuple(rows)


def build_patients() -> tuple[Patient, ...]:
    return tuple(
        Patient(
            id=pid,
            name=name,
            age=age,
            sex=sex,
            bmi=bmi,
            smoking_status=smoking,
            physically_active=active,
            general_health=health,
            conditions=conditions,
            assigned_doctor=doctor,
            medications=_medications(conditions),
            observations=_observations(pid, age, bmi, conditions),
            is_synthetic=True,
        )
        for pid, name, age, sex, bmi, smoking, active, health, conditions, doctor in _ROWS
    )

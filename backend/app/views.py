"""Pure functions that shape a Patient into response payloads."""

from __future__ import annotations

from .data.models import Patient
from .schemas import TimelineEvent


def build_timeline(patient: Patient) -> tuple[TimelineEvent, ...]:
    events = [
        TimelineEvent(
            date=o.date,
            type="observation",
            summary=f"Check-up: BMI {o.bmi}, systolic BP {o.systolic_bp}, HbA1c {o.hba1c}",
        )
        for o in patient.observations
    ] + [
        TimelineEvent(
            date=m.started,
            type="medication",
            summary=f"Started {m.name} {m.dose} ({m.frequency})",
        )
        for m in patient.medications
    ]
    return tuple(sorted(events, key=lambda e: (e.date, e.type)))

"""Doctor area: only patients assigned to the calling doctor are ever visible."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from ..access import Access, requires
from ..schemas import PatientRecord, PatientSummary, TimelineEvent
from ..users import Role
from ..views import build_timeline

router = APIRouter(prefix="/doctor", tags=["doctor"])

DoctorAccess = Annotated[Access, Depends(requires(Role.DOCTOR))]
DoctorListAccess = Annotated[
    Access, Depends(requires(Role.DOCTOR, action="patient.list", resource_type="patient"))
]


@router.get("/patients")
def list_patients(
    access: DoctorListAccess, q: Annotated[str | None, Query(max_length=80)] = None
) -> dict:
    assert access.user is not None and access.user.doctor_id is not None
    found = access.state.repo.list_for_doctor(access.user.doctor_id, q)
    return {"items": [PatientSummary.of(p) for p in found]}


@router.get("/patients/{patient_id}", response_model=PatientRecord)
def get_patient(patient_id: str, access: DoctorAccess) -> PatientRecord:
    return PatientRecord.of(access.patient(patient_id, "patient.read"))


@router.get("/patients/{patient_id}/timeline")
def get_timeline(patient_id: str, access: DoctorAccess) -> dict[str, tuple[TimelineEvent, ...]]:
    patient = access.patient(patient_id, "patient.timeline")
    return {"items": build_timeline(patient)}

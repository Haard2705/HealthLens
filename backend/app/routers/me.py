"""Patient self-service: a patient reads only their own record."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from ..access import Access, requires
from ..schemas import MedicationOut, ObservationOut, PatientRecord
from ..users import Role

router = APIRouter(prefix="/me", tags=["me"])

PatientAccess = Annotated[Access, Depends(requires(Role.PATIENT))]


@router.get("/record", response_model=PatientRecord)
def my_record(access: PatientAccess) -> PatientRecord:
    return PatientRecord.of(access.own_patient("patient.read"))


@router.get("/medications")
def my_medications(access: PatientAccess) -> dict[str, list[MedicationOut]]:
    patient = access.own_patient("patient.medications")
    return {"items": [MedicationOut.of(m) for m in patient.medications]}


@router.get("/observations")
def my_observations(access: PatientAccess) -> dict[str, list[ObservationOut]]:
    patient = access.own_patient("patient.observations")
    return {"items": [ObservationOut.of(o) for o in patient.observations]}

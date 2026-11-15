"""Individual risk estimates (H19): ``POST /predict`` and a patient's ``/risk-context``.

Both answers come from the same code path in ``predict.py``, so the release gate, the
validation rules and the shape of the response cannot drift apart between them.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict, Field

from .. import catalog, predict, risk_context
from ..access import Access, requires

router = APIRouter(tags=["predict"])


class PredictRequest(BaseModel):
    """``{"condition": "diabetes", ...answers}``. The answers are validated per condition
    against that model's own ``feature_spec``, so they cannot be declared statically here."""

    model_config = ConfigDict(extra="allow", frozen=True)

    condition: str = Field(min_length=1, max_length=40)

    @property
    def answers(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


AnyUser = Annotated[Access, Depends(requires())]
PredictAccess = Annotated[
    Access, Depends(requires(action="predict.create", resource_type="prediction"))
]


@router.post("/predict")
def create_prediction(body: PredictRequest, access: PredictAccess) -> dict:
    """An estimate for one set of answers. 409 when the release gate blocks the condition."""
    state = access.state
    return predict.predict(
        state.models,
        state.registry,
        state.settings.models_path,
        body.condition,
        body.answers,
    )


@router.get("/patients/{patient_id}/risk-context")
def patient_risk_context(
    patient_id: str,
    access: AnyUser,
    condition: Annotated[str, Query(max_length=40)],
) -> dict:
    """The same estimate, built from a patient's record, plus how people like them compare.

    Scoping is the usual one (``Access.patient``): a patient sees only their own record, a
    doctor only assigned patients, an admin any record but always audit-logged. A record the
    caller may not see answers with the same 404 as one that does not exist.
    """
    state = access.state
    patient = access.patient(patient_id, "patient.risk_context")
    cond = predict.known_condition(state.registry, condition)
    if cond.id in patient.conditions:
        return {
            "condition": cond.id,
            "display_name": cond.display_name,
            "patient_id": patient.id,
            "already_diagnosed": True,
        }

    gate = catalog.gate_entry(state.settings.models_path, cond.id)
    model = predict.usable_model(state.models, gate, cond)
    answers = risk_context.supported_answers(patient, state.registry, model.feature_spec)
    payload = predict.estimate(model, cond, gate, answers, state.registry)

    like_this = risk_context.patients_like_this(
        state.settings.processed_path, cond.outcome_column, patient
    )
    extra: dict[str, Any] = {
        "patient_id": patient.id,
        "already_diagnosed": False,
        "patients_like_this_available": like_this is not None,
    }
    if like_this is not None:
        extra["patients_like_this"] = like_this
    return {**payload, **extra}

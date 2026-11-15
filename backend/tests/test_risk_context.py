"""GET /patients/{id}/risk-context: scoping, auditing, already-diagnosed and the comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app import risk_context
from backend.app.config import Settings
from backend.app.data.synthetic import build_patients
from backend.tests.conftest import (
    ADMIN,
    CHEN,
    FATIMA,
    OKAFOR,
    TOMAS,
    audit_events,
    client_for,
)

NOT_FOUND = {"error": {"code": "not_found", "message": "Patient not found."}}
# P-0103 (Dr Chen) already has diabetes, hypertension and COPD, but not heart disease.
SAMUEL = "P-0103"
# P-0106 (Dr Chen) is Fatima's own record: diabetes and arthritis.
FATIMA_ID = "P-0106"


def _url(patient_id: str = SAMUEL, condition: str = "heart_disease") -> str:
    return f"/api/v1/patients/{patient_id}/risk-context?condition={condition}"


# ------------------------------- a working estimate ---------------------------------------


def test_doctor_gets_an_estimate_for_an_assigned_patient(trained_app: FastAPI):
    response = client_for(trained_app, CHEN).get(_url())
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["patient_id"] == SAMUEL
    assert body["condition"] == "heart_disease"
    assert body["already_diagnosed"] is False
    assert 0.0 <= body["risk"] <= 1.0
    assert body["contributions"]
    assert body["explanation"]["patient"]["text"]
    assert body["disclaimer"]


def test_the_record_fills_in_the_answers_it_knows(trained_app: FastAPI):
    """Age becomes a band, M becomes "male", and what a record cannot know stays unanswered."""
    body = client_for(trained_app, CHEN).get(_url()).json()
    values = {c["feature"]: c["value"] for c in body["contributions"]}

    assert values["bmi"] == pytest.approx(33.8)  # Samuel's recorded BMI
    assert values["sex"] == "male"
    assert values["age_band"] == pytest.approx(10)  # 67 years old -> band 10 (65-69)
    assert values["smoker"] == "daily"
    assert values["phys_active"] is False
    assert values["income_band"] is None  # a clinic record has no income


def test_a_condition_the_patient_already_has_is_not_estimated(trained_app: FastAPI):
    body = client_for(trained_app, CHEN).get(_url(condition="diabetes")).json()
    assert body == {
        "condition": "diabetes",
        "display_name": "Diabetes",
        "patient_id": SAMUEL,
        "already_diagnosed": True,
    }
    assert "risk" not in body


def test_a_patient_sees_their_own_risk_context(trained_app: FastAPI):
    response = client_for(trained_app, FATIMA).get(_url(FATIMA_ID))
    assert response.status_code == 200
    assert response.json()["patient_id"] == FATIMA_ID


# ------------------------------- scoping and auditing -------------------------------------


def test_another_doctors_patient_is_the_uniform_404(trained_app: FastAPI):
    response = client_for(trained_app, OKAFOR).get(_url())
    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_a_missing_patient_looks_exactly_the_same(trained_app: FastAPI):
    response = client_for(trained_app, CHEN).get(_url("P-9999"))
    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_a_patient_cannot_read_someone_elses_risk_context(trained_app: FastAPI):
    response = client_for(trained_app, TOMAS).get(_url(FATIMA_ID))
    assert response.status_code == 404
    assert response.json() == NOT_FOUND


def test_risk_context_needs_a_session(trained_app: FastAPI):
    response = TestClient(trained_app).get(_url())
    assert response.status_code == 401


def test_every_call_is_audit_logged(trained_app: FastAPI):
    client_for(trained_app, CHEN).get(_url())
    client_for(trained_app, OKAFOR).get(_url())
    client_for(trained_app, ADMIN).get(_url())

    events = audit_events(trained_app, action="patient.risk_context")
    by_actor = {e["actor_id"]: e for e in events}
    assert by_actor["dr-chen"]["outcome"] == "allowed"
    assert by_actor["dr-okafor"]["outcome"] == "denied"  # the refusal is recorded too
    assert by_actor["u-admin"]["outcome"] == "allowed"  # admin reads are always audited
    assert all(e["patient_id"] == SAMUEL for e in events)


def test_an_admin_may_read_any_patients_risk_context(trained_app: FastAPI):
    assert client_for(trained_app, ADMIN).get(_url()).status_code == 200


# ------------------------------- gate and validation --------------------------------------


def test_the_gate_blocks_risk_context_too(trained_app: FastAPI):
    response = client_for(trained_app, CHEN).get(_url(condition="depression"))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "analytics_only"


def test_an_unknown_condition_is_rejected(trained_app: FastAPI):
    response = client_for(trained_app, CHEN).get(_url(condition="not_a_condition"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_parameter"


def test_missing_artifacts_answer_503(app: FastAPI):
    response = client_for(app, CHEN).get(_url())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "models_unavailable"


# ------------------------------- people like this -----------------------------------------


def test_the_comparison_is_omitted_when_there_is_no_survey_data(trained_app: FastAPI):
    body = client_for(trained_app, CHEN).get(_url()).json()
    assert body["patients_like_this_available"] is False
    assert "patients_like_this" not in body


def _survey_for_samuel(rows: int = 80) -> pd.DataFrame:
    """Respondents in Samuel's own cell: age band 10, BMI 30+ (he is 67 with a BMI of 33.8)."""
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "survey_weight": np.ones(rows),
            "age_band": np.full(rows, 10),
            "bmi": rng.uniform(30.5, 39.0, rows),
            "heart_disease": np.arange(rows) % 4 == 0,  # a quarter report it
        }
    )


def test_the_comparison_is_included_when_the_parquet_is_there(
    trained_app: FastAPI, settings: Settings
):
    _survey_for_samuel().to_parquet(settings.processed_path)
    body = client_for(trained_app, CHEN).get(_url()).json()

    assert body["patients_like_this_available"] is True
    like_this = body["patients_like_this"]
    assert like_this["rate"] == pytest.approx(0.25)
    assert like_this["n"] == 80
    assert like_this["age_band"] == "65-69"
    assert like_this["bmi_band"] == "30+"


def test_a_cell_with_too_few_respondents_is_suppressed(trained_app: FastAPI, settings: Settings):
    _survey_for_samuel(rows=12).to_parquet(settings.processed_path)
    body = client_for(trained_app, CHEN).get(_url()).json()
    assert body["patients_like_this_available"] is False


def test_an_unexpected_parquet_shape_only_drops_the_comparison(
    trained_app: FastAPI, settings: Settings
):
    pd.DataFrame({"foo": [1]}).to_parquet(settings.processed_path)
    response = client_for(trained_app, CHEN).get(_url())
    assert response.status_code == 200
    assert response.json()["patients_like_this_available"] is False


# ------------------------------- the record -> answers mapping ----------------------------


@pytest.mark.parametrize(
    ("age", "band"),
    [(18, 1), (24, 1), (25, 2), (29, 2), (30, 3), (67, 10), (79, 12), (80, 13), (95, 13)],
)
def test_age_becomes_the_right_brfss_band(age: int, band: int):
    assert risk_context.age_band(age) == band


@pytest.mark.parametrize(
    ("bmi", "label"),
    [(17.0, "under 18.5"), (22.0, "18.5-24.9"), (27.5, "25-29.9"), (33.8, "30+")],
)
def test_bmi_becomes_the_right_band_label(bmi: float, label: str):
    assert risk_context.bmi_band_label(bmi) == label


def test_an_impossible_bmi_has_no_band():
    assert risk_context.bmi_band_label(-1.0) is None


def test_diagnosed_conditions_are_answered_from_the_record():
    from ml.src.registry import load_registry

    samuel = next(p for p in build_patients() if p.id == SAMUEL)
    answers = risk_context.answers_from_record(samuel, load_registry())
    assert answers["diabetes"] is True
    assert answers["high_bp"] is True  # hypertension's outcome column
    assert answers["heart_disease"] is False
    assert answers["sex"] == "male"

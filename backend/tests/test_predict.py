"""POST /predict: the release gate, the validation rules and the shape of an estimate (H19).

Every test trains its own tiny model (``models_fixture``); nothing here depends on the real
``ml/models/`` artifacts, which are gitignored and absent in CI.
"""

from __future__ import annotations

import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app import models_store, predict
from backend.app.config import Settings
from backend.app.main import create_app
from backend.tests import models_fixture
from backend.tests.conftest import (
    CHEN,
    FATIMA,
    SENSITIVE_REASON,
    WEAK_REASON,
    audit_events,
    client_for,
)
from ml.src.registry import load_registry

PREDICT = "/api/v1/predict"
ANSWERS = {
    "age_band": 9,
    "sex": "male",
    "bmi": 33.8,
    "smoker": "daily",
    "gen_health": 4,
    "phys_active": False,
}


def _body(condition: str = "diabetes", **overrides) -> dict:
    return {"condition": condition, **ANSWERS, **overrides}


# ------------------------------- a working estimate ---------------------------------------


def test_predict_returns_a_calibrated_shaped_estimate(trained_app: FastAPI):
    client = client_for(trained_app, CHEN)
    response = client.post(PREDICT, json=_body())
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["condition"] == "diabetes"
    assert body["display_name"] == "Diabetes"
    assert 0.0 <= body["risk"] <= 1.0
    assert body["risk_band"] in {"low", "moderate", "high", "very high"}
    assert body["signal_strength"] == "moderate"
    assert body["model_name"] == "logreg"
    assert body["disclaimer"].startswith("This is an estimate from survey data")
    assert body["limitations"]


def test_contributions_are_labelled_and_ordered_by_size(trained_app: FastAPI):
    client = client_for(trained_app, CHEN)
    contributions = client.post(PREDICT, json=_body()).json()["contributions"]

    sizes = [abs(c["contribution"]) for c in contributions]
    assert sizes == sorted(sizes, reverse=True)
    by_feature = {c["feature"]: c for c in contributions}
    assert by_feature["bmi"]["label"] == "BMI"
    assert by_feature["bmi"]["value"] == pytest.approx(33.8)
    assert by_feature["age_band"]["label"] == "Age"
    assert by_feature["phys_active"]["label"] == "Physical activity"
    # Canonical names only: no preprocessor column ever reaches the response.
    assert all("__" not in c["feature"] for c in contributions)


def test_explanation_has_both_levels_in_plain_language(trained_app: FastAPI):
    client = client_for(trained_app, CHEN)
    explanation = client.post(PREDICT, json=_body()).json()["explanation"]

    assert set(explanation) == {"patient", "clinician"}
    patient = explanation["patient"]
    assert patient["condition"] == "diabetes"
    assert "estimate" in patient["text"]
    assert patient["factors"]
    assert "SHAP contributions in log-odds" in explanation["clinician"]["text"]


def test_contributions_plus_base_equal_the_model_margin(settings: Settings):
    """The arithmetic promise of a SHAP explanation, checked on the explainer itself."""
    models_fixture.write_condition(settings.models_path, "diabetes")
    store = models_store.load(settings.models_path, load_registry())
    model = store.get("diabetes")
    assert model is not None

    explanation = predict.explain_answers(model, predict.validate_answers(model, ANSWERS))
    total = explanation["base_value"] + sum(c["contribution"] for c in explanation["contributions"])
    assert total == pytest.approx(explanation["margin"], abs=1e-9)
    expected = 1 / (1 + math.exp(-explanation["margin"]))
    assert explanation["probability"] == pytest.approx(expected, abs=1e-9)


def test_a_prediction_is_audit_logged(trained_app: FastAPI):
    client = client_for(trained_app, CHEN)
    assert client.post(PREDICT, json=_body()).status_code == 200
    events = audit_events(trained_app, action="predict.create")
    assert [e["outcome"] for e in events] == ["allowed"]
    assert events[0]["actor_role"] == "doctor"


# ------------------------------- the release gate -----------------------------------------


@pytest.mark.parametrize(
    ("condition", "reason"), [("asthma", WEAK_REASON), ("depression", SENSITIVE_REASON)]
)
def test_the_gate_blocks_a_condition_with_409_and_no_number(
    trained_app: FastAPI, condition: str, reason: str
):
    """Called directly, with artifacts present: still no estimate, and the gate's own reason."""
    client = client_for(trained_app, CHEN)
    response = client.post(PREDICT, json=_body(condition))

    assert response.status_code == 409
    assert response.json() == {"error": {"code": "analytics_only", "message": reason}}
    assert "risk" not in response.text


def test_a_blocked_condition_is_still_listed_for_analytics(trained_app: FastAPI):
    items = {
        i["id"]: i for i in client_for(trained_app, CHEN).get("/api/v1/conditions").json()["items"]
    }
    assert items["asthma"]["prediction_enabled"] is False
    assert items["depression"]["prediction_enabled"] is False
    assert items["diabetes"]["prediction_enabled"] is True


# ------------------------------- validation -----------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"bmi": 500.0},  # outside the trained range
        {"bmi": 1.0},
        {"age_band": 99},  # outside 1-13
        {"gen_health": 0},
        {"sex": "other"},  # not a trained category
        {"smoker": "occasionally"},
        {"phys_active": "maybe"},
        {"made_up_feature": 1},  # unknown field
    ],
)
def test_bad_answers_are_rejected_with_422(trained_app: FastAPI, overrides: dict):
    client = client_for(trained_app, CHEN)
    response = client.post(PREDICT, json=_body(**overrides))
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "validation_error"


def test_a_required_answer_cannot_be_left_out(trained_app: FastAPI):
    body = _body()
    del body["bmi"]
    response = client_for(trained_app, CHEN).post(PREDICT, json=body)
    assert response.status_code == 422
    assert "bmi" in response.json()["error"]["message"]


def test_optional_answers_may_be_left_out(trained_app: FastAPI):
    """A short answer set still predicts: unanswered questions stay unanswered."""
    client = client_for(trained_app, CHEN)
    response = client.post(
        PREDICT, json={"condition": "diabetes", "age_band": 5, "sex": "female", "bmi": 22.0}
    )
    assert response.status_code == 200, response.text
    values = {c["feature"]: c["value"] for c in response.json()["contributions"]}
    assert values["high_chol"] is None


def test_an_unknown_condition_is_rejected(trained_app: FastAPI):
    response = client_for(trained_app, CHEN).post(PREDICT, json=_body("not_a_condition"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_parameter"


def test_predict_requires_a_session(trained_app: FastAPI):
    response = TestClient(trained_app).post(PREDICT, json=_body())
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"


def test_a_patient_may_predict_for_themselves(trained_app: FastAPI):
    assert client_for(trained_app, FATIMA).post(PREDICT, json=_body()).status_code == 200


# ------------------------------- no artifacts ---------------------------------------------


def test_missing_artifacts_answer_503_with_the_training_command(app: FastAPI):
    response = client_for(app, CHEN).post(PREDICT, json=_body())
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "models_unavailable"
    assert "python -m ml.src.train" in error["message"]


def test_the_app_still_works_with_no_artifacts_at_all(app: FastAPI, anon):
    assert anon.get("/health").json() == {"status": "ok"}
    assert len(anon.get("/api/v1/conditions").json()["items"]) == 9
    assert client_for(app, CHEN).get("/api/v1/doctor/patients").status_code == 200
    assert app.state.hl.models.available is False


def test_half_written_artifacts_are_ignored_rather_than_crashing(settings: Settings):
    folder = settings.models_path / "diabetes"
    folder.mkdir(parents=True)
    (folder / "model_meta.json").write_text('{"features": [], "feature_spec": []}')
    (folder / "explainer.joblib").write_bytes(b"not a joblib file")
    assert models_store.load(settings.models_path, load_registry()).available is False


# ------------------------------- more than two models -------------------------------------


def test_whichever_model_the_pipeline_chose_is_the_one_reported(settings: Settings):
    """A third model in the pipeline needs no change here: best_model is passed through."""
    models_fixture.write_condition(
        settings.models_path,
        "diabetes",
        best_model="random_forest",
        extra_models=("logreg", "xgboost"),
    )
    models_fixture.write_index(
        settings.models_path, {"diabetes": models_fixture.gate_decision(enabled=True)}
    )
    application = create_app(settings)
    with TestClient(application):
        client = client_for(application, CHEN)
        assert client.post(PREDICT, json=_body()).json()["model_name"] == "random_forest"
        metrics = client.get("/api/v1/models", params={"condition": "diabetes"}).json()["metrics"]
        assert set(metrics["models"]) == {"logreg", "random_forest", "xgboost"}
        assert metrics["best_model"] == "random_forest"


def test_risk_bands_label_the_whole_range():
    assert predict.risk_band(0.0) == "low"
    assert predict.risk_band(0.2) == "moderate"
    assert predict.risk_band(0.3) == "high"
    assert predict.risk_band(0.99) == "very high"


def test_a_gate_decision_with_no_reason_still_refuses_clearly(settings: Settings):
    """Defensive: a malformed index entry must not become a number, or a blank message."""
    models_fixture.write_condition(settings.models_path, "diabetes")
    models_fixture.write_index(settings.models_path, {"diabetes": {"prediction_enabled": False}})
    application = create_app(settings)
    with TestClient(application):
        response = client_for(application, CHEN).post(PREDICT, json=_body())

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "analytics_only"
    assert "Diabetes" in error["message"]

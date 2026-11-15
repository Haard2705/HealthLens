"""Conditions, model metrics, prevalence stats and the synthetic data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backend.app import stats
from backend.app.data.synthetic import build_patients
from backend.tests.conftest import CHEN, FATIMA


def test_conditions_without_index_have_null_gate_fields(anon):
    items = anon.get("/api/v1/conditions").json()["items"]
    assert len(items) == 9
    assert {i["id"] for i in items} >= {"diabetes", "copd", "depression"}
    for item in items:
        assert item["prediction_enabled"] is None
        assert item["signal_strength"] is None
        assert item["reason"] is None


def test_conditions_merge_index(anon, settings):
    settings.models_path.mkdir(parents=True)
    index = {
        "conditions": {
            "diabetes": {
                "prediction_enabled": True,
                "signal_strength": "strong",
                "reason": "Passes the gate.",
            }
        }
    }
    (settings.models_path / "index.json").write_text(json.dumps(index))
    items = {i["id"]: i for i in anon.get("/api/v1/conditions").json()["items"]}
    assert items["diabetes"]["prediction_enabled"] is True
    assert items["diabetes"]["signal_strength"] == "strong"
    assert items["copd"]["prediction_enabled"] is None


def test_conditions_survive_a_corrupt_index(anon, settings):
    settings.models_path.mkdir(parents=True)
    (settings.models_path / "index.json").write_text("{not json")
    assert anon.get("/api/v1/conditions").status_code == 200


def test_models_not_trained_is_a_clear_404(anon):
    response = anon.get("/api/v1/models", params={"condition": "diabetes"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_trained"
    assert "not trained yet" in response.json()["error"]["message"].lower()


def test_models_serves_metrics(anon, settings):
    folder = settings.models_path / "diabetes"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text(json.dumps({"test_roc_auc": 0.83}))
    body = anon.get("/api/v1/models", params={"condition": "diabetes"}).json()
    assert body == {"condition": "diabetes", "metrics": {"test_roc_auc": 0.83}}


@pytest.mark.parametrize("bad", ["../../etc/passwd", "nope", "diabetes; drop table"])
def test_models_whitelists_condition(anon, bad):
    assert anon.get("/api/v1/models", params={"condition": bad}).status_code == 422


def test_models_requires_condition(anon):
    assert anon.get("/api/v1/models").status_code == 422


@pytest.mark.parametrize("by", stats.BY_OPTIONS)
def test_prevalence_sample_fallback_is_labelled(anon, by):
    body = anon.get("/api/v1/stats/prevalence", params={"condition": "copd", "by": by}).json()
    assert body["source"] == "sample"
    assert body["groups"]
    assert "NOT survey results" in body["note"]


def test_prevalence_rejects_unknown_by_and_condition(anon):
    assert (
        anon.get("/api/v1/stats/prevalence", params={"condition": "x", "by": "sex"}).status_code
        == 422
    )
    r = anon.get("/api/v1/stats/prevalence", params={"condition": "copd", "by": "income"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_parameter"


def _survey_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    sex = np.array(["male", "female"] * (n // 2))
    return pd.DataFrame(
        {
            "survey_weight": np.where(sex == "male", 2.0, 1.0),
            "sex": sex,
            "age_band": rng.integers(1, 14, n),
            "bmi": rng.uniform(17, 40, n),
            "phys_active": rng.random(n) > 0.5,
            "diabetes": (sex == "male") & (np.arange(n) % 4 == 0),
        }
    )


def test_weighted_prevalence_math():
    df = pd.DataFrame(
        {
            "survey_weight": [1.0] * 30 + [3.0] * 30,
            "sex": ["male"] * 30 + ["female"] * 30,
            "diabetes": [True] * 15 + [False] * 15 + [True] * 30,
        }
    )
    result = stats.weighted_prevalence(df, "diabetes", "diabetes", "sex")
    rates = {g["group"]: g["prevalence"] for g in result["groups"]}
    assert rates == {"male": 0.5, "female": 1.0}
    assert result["source"] == "brfss"


def test_small_groups_are_suppressed():
    df = _survey_frame()
    df.loc[df.index[:10], "sex"] = "other"  # only 10 respondents in this group
    groups = stats.weighted_prevalence(df, "diabetes", "diabetes", "sex")["groups"]
    assert {g["group"] for g in groups} == {"male", "female"}


@pytest.mark.parametrize("by", stats.BY_OPTIONS)
def test_endpoint_uses_parquet_when_present(anon, settings, by):
    _survey_frame().to_parquet(settings.processed_path)
    body = anon.get("/api/v1/stats/prevalence", params={"condition": "diabetes", "by": by}).json()
    assert body["source"] == "brfss"
    assert all(g["n"] >= stats.MIN_GROUP_SIZE for g in body["groups"])


def test_endpoint_reports_unexpected_parquet_shape(anon, settings):
    pd.DataFrame({"foo": [1]}).to_parquet(settings.processed_path)
    r = anon.get("/api/v1/stats/prevalence", params={"condition": "diabetes", "by": "sex"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "data_unavailable"


# ------------------------------- synthetic data & views ----------------------------------


def test_synthetic_patients_are_exact_deterministic_and_flagged():
    first, second = build_patients(), build_patients()
    assert first == second
    assert [p.id for p in first] == [f"P-010{i}" for i in range(1, 9)]
    assert all(p.is_synthetic for p in first)
    assert all(len(p.observations) == 12 and p.medications for p in first)
    samuel = first[2]
    assert (samuel.name, samuel.age, samuel.bmi) == ("Samuel Boateng", 67, 33.8)
    assert samuel.conditions == ("diabetes", "hypertension", "copd")
    assert samuel.observations[-1].bmi == 33.8  # latest observation matches the current record
    assert [p.assigned_doctor for p in first].count("dr-okafor") == 2


def test_timeline_is_chronological(login_as):
    items = login_as(CHEN).get("/api/v1/doctor/patients/P-0106/timeline").json()["items"]
    dates = [i["date"] for i in items]
    assert dates == sorted(dates)
    assert {i["type"] for i in items} == {"observation", "medication"}


def test_patient_observations_and_medications(login_as):
    client = login_as(FATIMA)
    obs = client.get("/api/v1/me/observations").json()["items"]
    meds = client.get("/api/v1/me/medications").json()["items"]
    assert len(obs) == 12
    assert {m["name"] for m in meds} >= {"Metformin", "Naproxen"}
    record = client.get("/api/v1/me/record").json()
    assert record["is_synthetic"] is True
    assert record["conditions"] == ["diabetes", "arthritis"]

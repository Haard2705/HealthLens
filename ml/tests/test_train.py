"""H13: train.py trains every model per condition and writes the files the backend loads."""

import json
import time
from datetime import datetime

import joblib
import pandas as pd
import pytest
from src import evaluate, features, train
from src.registry import load_registry

from .test_logreg import signal_frame

REGISTRY = load_registry()
FIRST, SECOND = REGISTRY.conditions[0], REGISTRY.conditions[1]
FAST = train.Settings.fast()


def strict_json(path):
    """Parse JSON and refuse NaN/Infinity, like a strict client (the backend) would."""

    def reject(value):
        raise ValueError(f"invalid JSON constant {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return features.add_features(signal_frame(n=1500))


@pytest.fixture(scope="module")
def trained(tmp_path_factory, frame):
    """logreg only, so this part runs even where xgboost isn't installed."""
    out = tmp_path_factory.mktemp("models")
    entry = train.train_condition(frame, FIRST, out, ("logreg",), FAST, data_file="x.parquet")
    return out, entry


# ---- choosing the best model -----------------------------------------------------------


def test_select_best_takes_the_highest_validation_auc() -> None:
    assert train.select_best({"logreg": 0.70, "xgboost": 0.75}) == "xgboost"
    assert train.select_best({"logreg": 0.80, "xgboost": 0.75}) == "logreg"


def test_select_best_prefers_the_simpler_model_on_a_near_tie() -> None:
    assert train.select_best({"logreg": 0.750, "xgboost": 0.754}) == "logreg"
    assert train.select_best({"xgboost": 0.754, "logreg": 0.750}) == "logreg"


def test_select_best_needs_models() -> None:
    with pytest.raises(ValueError):
        train.select_best({})


# ---- files for one condition -----------------------------------------------------------


def test_writes_model_metrics_and_meta(trained) -> None:
    out, _ = trained
    folder = out / FIRST.id
    assert {p.name for p in folder.iterdir()} == {
        "logreg.joblib",
        "metrics.json",
        "model_meta.json",
        "explainer.joblib",
    }


def test_saved_model_loads_and_predicts(trained, frame) -> None:
    out, _ = trained
    model = joblib.load(out / FIRST.id / "logreg.joblib")
    split = evaluate.split_condition(frame, FIRST.id)
    p = model.predict_proba(split.x_test)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_metrics_json_has_every_model_and_the_choice(trained) -> None:
    out, _ = trained
    metrics = strict_json(out / FIRST.id / "metrics.json")
    assert metrics["condition"] == FIRST.id
    assert metrics["best_model"] == "logreg"
    assert "validation ROC-AUC" in metrics["selection"]["criterion"]
    assert set(metrics["selection"]["validation_roc_auc"]) == {"logreg"}
    m = metrics["models"]["logreg"]["metrics"]
    assert {"default", "recall_oriented"} <= set(m)


def test_model_meta_tells_the_backend_what_to_send(trained, frame) -> None:
    out, _ = trained
    meta = strict_json(out / FIRST.id / "model_meta.json")
    split = evaluate.split_condition(frame, FIRST.id)
    assert meta["features"] == list(features.predictors_for(FIRST.id))
    assert [f["name"] for f in meta["feature_spec"]] == meta["features"]
    assert FIRST.outcome_column not in meta["features"]
    assert meta["rows"]["train"] + meta["rows"]["test"] == len(split.y_train) + len(split.y_test)
    assert meta["seed"] == 42
    assert meta["data_file"] == "x.parquet"
    assert meta["versions"]["scikit-learn"]
    assert isinstance(meta["git_sha"], str) and meta["git_sha"]
    datetime.fromisoformat(meta["trained_at"])


def test_feature_spec_kinds_and_allowed_values(trained) -> None:
    out, _ = trained
    spec = {f["name"]: f for f in strict_json(out / FIRST.id / "model_meta.json")["feature_spec"]}
    assert spec["sex"] == {
        "name": "sex",
        "kind": "category",
        "allowed": ["female", "male"],
        "derived": False,
    }
    assert spec["bmi"]["kind"] == "number" and spec["bmi"]["range"] == [10.0, 100.0]
    assert spec["phys_active"]["kind"] == "yes_no"
    assert spec["bmi_band"]["derived"] is True


def test_index_entry_summarises_the_condition(trained) -> None:
    _, entry = trained
    assert entry["status"] == "trained"
    assert entry["best_model"] == "logreg"
    assert 0 < entry["test_roc_auc"] <= 1
    assert entry["tier"] == FIRST.tier


def test_unknown_model_name_is_rejected(tmp_path, frame) -> None:
    with pytest.raises(ValueError, match="unknown model"):
        train.train_condition(frame, FIRST, tmp_path, ("forest",), FAST)


# ---- index.json -----------------------------------------------------------------------


def test_index_is_merged_not_overwritten(tmp_path) -> None:
    train.update_index(tmp_path, {"a": {"status": "trained"}})
    train.update_index(tmp_path, {"b": {"status": "trained"}})
    train.update_index(tmp_path, {"a": {"status": "failed"}})
    index = strict_json(tmp_path / "index.json")
    assert index["conditions"] == {"a": {"status": "failed"}, "b": {"status": "trained"}}
    assert "generated_at" in index and "git_sha" in index


# ---- command line ---------------------------------------------------------------------


def test_main_one_condition_logreg_only(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(evaluate, "load_processed", lambda path: signal_frame(n=800))
    args = ["--condition", SECOND.id, "--models", "logreg", "--fast", "--out", str(tmp_path)]
    assert train.main(args) == 0
    assert (tmp_path / SECOND.id / "metrics.json").exists()
    assert set(strict_json(tmp_path / "index.json")["conditions"]) == {SECOND.id}
    assert SECOND.id in capsys.readouterr().out


def test_main_records_a_failed_condition_and_keeps_going(monkeypatch, tmp_path, capsys) -> None:
    df = signal_frame(n=800)
    df[FIRST.outcome_column] = pd.array([False] * len(df), dtype="boolean")  # one class only
    monkeypatch.setattr(evaluate, "load_processed", lambda path: df)
    assert train.main(["--models", "logreg", "--fast", "--out", str(tmp_path)]) == 1
    index = strict_json(tmp_path / "index.json")["conditions"]
    assert index[FIRST.id]["status"] == "failed"
    assert "both" in index[FIRST.id]["reason"]
    assert index[SECOND.id]["status"] == "trained"
    assert "FAILED" in capsys.readouterr().err


def test_main_unknown_condition(monkeypatch, capsys) -> None:
    monkeypatch.setattr(evaluate, "load_processed", lambda path: signal_frame(n=300))
    assert train.main(["--condition", "nope"]) == 1
    assert "unknown condition" in capsys.readouterr().err


# ---- the roadmap's smoke test: every condition, both models, 300 rows ---------------------


@pytest.mark.slow
def test_smoke_every_condition_both_models_300_rows(monkeypatch, tmp_path) -> None:
    pytest.importorskip("xgboost")
    monkeypatch.setattr(evaluate, "load_processed", lambda path: signal_frame(n=300))
    start = time.perf_counter()
    assert train.main(["--fast", "--out", str(tmp_path)]) == 0
    elapsed = time.perf_counter() - start

    index = strict_json(tmp_path / "index.json")["conditions"]
    assert set(index) == set(REGISTRY.ids)
    for condition_id in REGISTRY.ids:
        folder = tmp_path / condition_id
        assert {
            "logreg.joblib",
            "xgboost.joblib",
            "metrics.json",
            "model_meta.json",
            "explainer.joblib",
        } <= {p.name for p in folder.iterdir()}
        metrics = strict_json(folder / "metrics.json")
        assert set(metrics["models"]) == {"logreg", "xgboost"}
        assert metrics["best_model"] in {"logreg", "xgboost"}
    print(f"smoke test trained {len(index)} conditions x 2 models in {elapsed:.1f}s")
    assert elapsed < 120  # roadmap target is < 30 s in CI; this bound only catches runaways

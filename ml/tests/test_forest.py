"""H12: the random forest, trained and scored through the same H10 harness as the baseline."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src import evaluate, forest, logreg
from src.registry import load_registry

from .test_logreg import signal_frame

FIRST = load_registry().ids[0]
FAST = {"n_iter": 3, "cv": 2, "n_trees": 40}
METRIC_KEYS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
    "brier_no_skill",
}


def interaction_frame(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Risk is high when exactly one of (high BMI, older age band) holds, low for neither or both.

    No straight-line combination of BMI and age can rank that correctly; trees can.
    """
    df = signal_frame(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    high_bmi = df["bmi"].astype(float).fillna(28) > 28
    older = df["age_band"].astype(float) > 7
    p = np.where(high_bmi ^ older, 0.5, 0.05)
    column = load_registry().get(FIRST).outcome_column
    df[column] = pd.array(rng.random(n) < p, dtype="boolean")
    return df


@pytest.fixture(scope="module")
def split() -> evaluate.Split:
    return evaluate.split_condition(signal_frame(), FIRST)


@pytest.fixture(scope="module")
def result(split) -> forest.ForestResult:
    return forest.train_forest(split, **FAST)


# ---- building blocks --------------------------------------------------------------------


def test_build_forest_is_preprocessor_plus_random_forest(split) -> None:
    model = forest.build_forest(split.predictors, max_depth=6)
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["prep", "clf"]
    clf = model.named_steps["clf"]
    assert isinstance(clf, RandomForestClassifier)
    assert clf.max_depth == 6
    assert clf.n_jobs == -1


def test_no_class_weighting(split) -> None:
    # Same decision as H11: re-weighting the rare class inflates every probability.
    assert forest.CLASS_WEIGHT is None
    assert forest.build_forest(split.predictors).named_steps["clf"].class_weight is None


def test_param_space_is_modest() -> None:
    assert forest.N_ITER <= 10
    assert set(forest.PARAM_SPACE) == {"max_depth", "min_samples_leaf", "max_features"}
    # Depth is capped so the forest stays small enough to train on the full file.
    assert all(isinstance(d, int) for d in forest.PARAM_SPACE["max_depth"])


def test_tune_returns_settings_from_the_search_space(split) -> None:
    tuned = forest.tune_forest(split.x_train, split.y_train, split.predictors, n_iter=3, cv=2)
    for name, value in tuned.best_params.items():
        assert value in forest.PARAM_SPACE[name]
    assert 0.5 <= tuned.best_cv_score <= 1


def test_tune_samples_large_training_sets(split) -> None:
    tuned = forest.tune_forest(
        split.x_train, split.y_train, split.predictors, n_iter=2, cv=2, max_rows=1000
    )
    assert tuned.n_rows == 1000


def test_fit_forest_uses_the_requested_number_of_trees(split) -> None:
    fit, _ = evaluate.validation_split(split)
    model = forest.fit_forest(split.predictors, {"max_depth": 6}, fit, n_trees=25)
    assert len(model.named_steps["clf"].estimators_) == 25
    assert model.predict_proba(split.x_test).shape == (len(split.x_test), 2)


# ---- training through the harness -------------------------------------------------------


def test_train_forest_reports_both_operating_points(result) -> None:
    assert result.name == "random_forest"
    assert set(result.metrics) >= {"default", "recall_oriented"}
    assert result.metrics["threshold_chosen_on"] == "validation"


def test_train_forest_uses_the_same_rows_as_the_baseline(split, result) -> None:
    fit, val = evaluate.validation_split(split)
    assert (result.n_fit, result.n_validation, result.n_test) == (
        len(fit.y),
        len(val.y),
        len(split.y_test),
    )


def test_train_forest_learns_the_signal_and_stays_calibrated(result) -> None:
    m = result.metrics["default"]
    assert m["roc_auc"] > 0.65
    assert m["brier"] < m["brier_no_skill"]
    assert result.metrics["recall_oriented"]["recall"] >= 0.7


def test_forest_beats_the_linear_baseline_on_an_interaction() -> None:
    split = evaluate.split_condition(interaction_frame(), FIRST)
    trees = forest.train_forest(split, **FAST).metrics["default"]["roc_auc"]
    linear = logreg.train_logreg(split, cv=3).metrics["default"]["roc_auc"]
    assert trees > linear + 0.1


def test_summary_is_json_safe_and_matches_baseline_shape(result, split) -> None:
    summary = result.summary()
    json.dumps(summary)
    assert summary["model"] == "random_forest"
    assert summary["params"]["n_estimators"] == result.n_trees
    assert summary["params"]["class_weight"] is None
    baseline = logreg.train_logreg(split, cv=2).summary()
    assert set(baseline) <= set(summary)


def test_summary_carries_the_same_metric_keys_as_the_baseline(result, split) -> None:
    baseline = logreg.train_logreg(split, cv=2).summary()["metrics"]
    mine = result.summary()["metrics"]
    assert set(mine) == set(baseline)
    for point in ("default", "recall_oriented"):
        assert set(mine[point]) >= METRIC_KEYS
        assert set(mine[point]) == set(baseline[point])


def test_training_is_reproducible(split) -> None:
    a = forest.train_forest(split, **FAST)
    b = forest.train_forest(split, **FAST)
    assert a.params == b.params
    np.testing.assert_allclose(
        a.model.predict_proba(split.x_test), b.model.predict_proba(split.x_test)
    )


def test_saved_pipeline_reloads_and_predicts(tmp_path, result, split) -> None:
    path = tmp_path / "random_forest.joblib"
    joblib.dump(result.model, path)
    reloaded = joblib.load(path)
    np.testing.assert_allclose(
        reloaded.predict_proba(split.x_test), result.model.predict_proba(split.x_test)
    )


# ---- command line ----------------------------------------------------------------------


def test_main_compare(monkeypatch, capsys) -> None:
    monkeypatch.setattr(evaluate, "load_processed", lambda path: signal_frame(n=1500))
    assert forest.main(["--condition", FIRST, "--n-iter", "2", "--compare"]) == 0
    out = capsys.readouterr().out
    assert "logreg" in out and "random_forest" in out and "roc_auc" in out


def test_main_unknown_condition(monkeypatch, capsys) -> None:
    monkeypatch.setattr(evaluate, "load_processed", lambda path: signal_frame(n=300))
    assert forest.main(["--condition", "nope"]) == 1
    assert "unknown condition" in capsys.readouterr().err

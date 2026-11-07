"""H12: XGBoost, trained and scored through the same H10 harness as the baseline."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from sklearn.pipeline import Pipeline  # noqa: E402
from src import boosting, evaluate, logreg  # noqa: E402
from src.registry import load_registry  # noqa: E402

from .test_logreg import signal_frame  # noqa: E402

FIRST = load_registry().ids[0]
FAST = {"n_iter": 3, "cv": 2}


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
def result(split) -> boosting.BoostResult:
    return boosting.train_xgb(split, **FAST)


# ---- building blocks --------------------------------------------------------------------


def test_build_xgb_is_preprocessor_plus_xgboost(split) -> None:
    model = boosting.build_xgb(split.predictors, max_depth=3)
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["prep", "clf"]
    clf = model.named_steps["clf"]
    assert type(clf).__name__ == "XGBClassifier"
    assert clf.get_params()["max_depth"] == 3
    assert clf.get_params()["scale_pos_weight"] == 1.0


def test_param_space_is_modest() -> None:
    assert boosting.N_ITER <= 20
    assert set(boosting.PARAM_SPACE) <= {
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    }


def test_tune_returns_settings_from_the_search_space(split) -> None:
    tuned = boosting.tune_xgb(split.x_train, split.y_train, split.predictors, n_iter=3, cv=2)
    for name, value in tuned.best_params.items():
        assert value in boosting.PARAM_SPACE[name]
    assert 0.5 <= tuned.best_cv_score <= 1


def test_tune_samples_large_training_sets(split) -> None:
    tuned = boosting.tune_xgb(
        split.x_train, split.y_train, split.predictors, n_iter=2, cv=2, max_rows=1000
    )
    assert tuned.n_rows == 1000


def test_early_stopping_keeps_fewer_trees_than_the_limit(split) -> None:
    fit, val = evaluate.validation_split(split)
    params = {"max_depth": 3, "learning_rate": 0.1}
    model, n_trees = boosting.fit_with_early_stopping(split.predictors, params, fit, val)
    assert 1 <= n_trees < boosting.MAX_TREES
    assert model.predict_proba(split.x_test).shape == (len(split.x_test), 2)


# ---- training through the harness -------------------------------------------------------


def test_train_xgb_reports_both_operating_points(result) -> None:
    assert result.name == "xgboost"
    assert set(result.metrics) >= {"default", "recall_oriented"}
    assert result.metrics["threshold_chosen_on"] == "validation"


def test_train_xgb_uses_the_same_rows_as_the_baseline(split, result) -> None:
    fit, val = evaluate.validation_split(split)
    assert (result.n_fit, result.n_validation, result.n_test) == (
        len(fit.y),
        len(val.y),
        len(split.y_test),
    )


def test_train_xgb_learns_the_signal_and_stays_calibrated(result) -> None:
    m = result.metrics["default"]
    assert m["roc_auc"] > 0.65
    assert m["brier"] < m["brier_no_skill"]
    assert result.metrics["recall_oriented"]["recall"] >= 0.7


def test_xgboost_beats_the_linear_baseline_on_an_interaction() -> None:
    split = evaluate.split_condition(interaction_frame(), FIRST)
    xgb = boosting.train_xgb(split, **FAST).metrics["default"]["roc_auc"]
    lin = logreg.train_logreg(split, cv=3).metrics["default"]["roc_auc"]
    assert xgb > lin + 0.1


def test_summary_is_json_safe_and_matches_baseline_shape(result, split) -> None:
    summary = result.summary()
    json.dumps(summary)
    assert summary["model"] == "xgboost"
    assert summary["params"]["n_estimators"] == result.n_trees
    baseline_keys = set(logreg.train_logreg(split, cv=2).summary())
    assert {"model", "condition", "params", "cv_roc_auc", "rows", "metrics"} <= baseline_keys
    assert {"model", "condition", "params", "cv_roc_auc", "rows", "metrics"} <= set(summary)


def test_training_is_reproducible(split) -> None:
    a = boosting.train_xgb(split, **FAST)
    b = boosting.train_xgb(split, **FAST)
    assert a.params == b.params
    assert a.n_trees == b.n_trees
    np.testing.assert_allclose(
        a.model.predict_proba(split.x_test), b.model.predict_proba(split.x_test)
    )


# ---- comparison + command line ----------------------------------------------------------


def test_compare_puts_models_side_by_side(result, split) -> None:
    table = boosting.compare([logreg.train_logreg(split, cv=2).summary(), result.summary()])
    assert list(table.index) == ["logreg", "xgboost"]
    assert list(table.columns) == list(boosting.COMPARE_METRICS)


def test_main_compare(monkeypatch, capsys) -> None:
    monkeypatch.setattr(evaluate, "load_processed", lambda path: signal_frame(n=1500))
    assert boosting.main(["--condition", FIRST, "--n-iter", "2", "--compare"]) == 0
    out = capsys.readouterr().out
    assert "logreg" in out and "xgboost" in out and "roc_auc" in out


def test_main_unknown_condition(monkeypatch, capsys) -> None:
    monkeypatch.setattr(evaluate, "load_processed", lambda path: signal_frame(n=300))
    assert boosting.main(["--condition", "nope"]) == 1
    assert "unknown condition" in capsys.readouterr().err

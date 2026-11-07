"""H11: the logistic regression baseline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src import evaluate, logreg, schema
from src.registry import load_registry

from .test_features import make_frame

FIRST = load_registry().ids[0]


def signal_frame(n: int = 4000, seed: int = 3) -> pd.DataFrame:
    """Synthetic frame where every outcome really rises with BMI and age (and nothing else)."""
    df = make_frame(n=n, seed=seed)
    rng = np.random.default_rng(seed + 100)
    logit = -6 + 0.12 * df["bmi"].astype(float).fillna(28) + 0.15 * df["age_band"].astype(float)
    for column in schema.OUTCOMES:
        yes = rng.random(n) < 1 / (1 + np.exp(-logit))
        df[column] = pd.array(yes, dtype="boolean")
    return df


@pytest.fixture(scope="module")
def split() -> evaluate.Split:
    return evaluate.split_condition(signal_frame(), FIRST)


@pytest.fixture(scope="module")
def result(split) -> logreg.LogregResult:
    return logreg.train_logreg(split, cv=3)


# ---- the pipeline -----------------------------------------------------------------------


def test_build_logreg_is_preprocessor_plus_logistic_regression(split) -> None:
    model = logreg.build_logreg(split.predictors, C=0.5)
    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["prep", "clf"]
    clf = model.named_steps["clf"]
    assert isinstance(clf, LogisticRegression)
    assert clf.class_weight is None
    assert clf.C == 0.5
    balanced = logreg.build_logreg(split.predictors, class_weight="balanced")
    assert balanced.named_steps["clf"].class_weight == "balanced"


def test_probabilities_stay_calibrated(result) -> None:
    # Unweighted training keeps predicted risks honest: better than always guessing prevalence.
    m = result.metrics["default"]
    assert m["brier"] < m["brier_no_skill"]


def test_balanced_weighting_would_inflate_probabilities(split) -> None:
    # Why we don't re-weight: same ranking, but probabilities pushed up past honest values.
    fit, _ = evaluate.validation_split(split)
    plain = logreg.build_logreg(split.predictors).fit(fit.x, fit.y)
    balanced = logreg.build_logreg(split.predictors, class_weight="balanced").fit(fit.x, fit.y)
    p_plain = plain.predict_proba(split.x_test)[:, 1].mean()
    p_balanced = balanced.predict_proba(split.x_test)[:, 1].mean()
    assert p_plain == pytest.approx(split.y_test.mean(), abs=0.03)
    assert p_balanced > p_plain + 0.15


def test_c_grid_is_small_and_spans_strong_to_weak_regularisation() -> None:
    assert len(logreg.C_GRID) <= 6
    assert min(logreg.C_GRID) < 0.1 < 1 < max(logreg.C_GRID)


# ---- tuning -----------------------------------------------------------------------------


def test_tune_picks_a_c_from_the_grid_and_reports_every_candidate(split) -> None:
    tuned = logreg.tune_logreg(split.x_train, split.y_train, split.predictors, cv=3)
    assert tuned.best_c in logreg.C_GRID
    assert set(tuned.cv_scores) == set(logreg.C_GRID)
    assert tuned.cv_scores[tuned.best_c] == max(tuned.cv_scores.values())
    assert all(0.5 <= s <= 1 for s in tuned.cv_scores.values())


def test_tune_returns_a_model_refit_on_all_rows_given(split) -> None:
    tuned = logreg.tune_logreg(split.x_train, split.y_train, split.predictors, cv=3)
    assert tuned.best_c == tuned.model.named_steps["clf"].C
    assert tuned.model.predict_proba(split.x_test).shape == (len(split.x_test), 2)


def test_tuning_is_reproducible(split) -> None:
    a = logreg.tune_logreg(split.x_train, split.y_train, split.predictors, cv=3)
    b = logreg.tune_logreg(split.x_train, split.y_train, split.predictors, cv=3)
    assert a.best_c == b.best_c
    np.testing.assert_allclose(a.model.named_steps["clf"].coef_, b.model.named_steps["clf"].coef_)


# ---- training on a split (uses the H10 harness) -----------------------------------------


def test_train_logreg_reports_both_operating_points(result) -> None:
    assert result.name == "logreg"
    assert set(result.metrics) >= {"default", "recall_oriented", "target_recall"}
    assert result.metrics["threshold_chosen_on"] == "validation"
    assert result.best_c in logreg.C_GRID


def test_train_logreg_never_fits_on_test_or_validation_rows(split, result) -> None:
    fit, val = evaluate.validation_split(split)
    assert result.n_fit == len(fit.y)
    assert result.n_validation == len(val.y)
    assert result.n_test == len(split.y_test)
    assert result.n_fit + result.n_validation + result.n_test == len(split.y_train) + len(
        split.y_test
    )


def test_train_logreg_learns_the_signal(result) -> None:
    # Risk truly rises with BMI and age, so the model must rank far better than a coin.
    assert result.metrics["default"]["roc_auc"] > 0.65
    assert result.metrics["recall_oriented"]["recall"] >= 0.7


def test_summary_is_json_safe_and_has_no_model_object(result) -> None:
    import json

    summary = result.summary()
    json.dumps(summary)
    assert summary["model"] == "logreg"
    assert summary["params"] == {"C": result.best_c, "class_weight": None}
    assert "cv_roc_auc" in summary


# ---- odds ratios ------------------------------------------------------------------------


def test_odds_ratios_use_readable_feature_names(result) -> None:
    table = logreg.odds_ratios(result.model)
    assert list(table.columns) == ["feature", "coefficient", "odds_ratio"]
    assert not table["feature"].str.contains("__").any()
    assert "bmi" in set(table["feature"])
    assert "sex = male" in set(table["feature"])
    assert "income_band (was missing)" in set(table["feature"])


def test_odds_ratio_is_exp_of_coefficient_sorted_by_strength(result) -> None:
    table = logreg.odds_ratios(result.model)
    np.testing.assert_allclose(table["odds_ratio"], np.exp(table["coefficient"]))
    strength = table["coefficient"].abs().to_numpy()
    assert (strength[:-1] >= strength[1:]).all()


def test_odds_ratios_find_the_true_risk_factors(result) -> None:
    table = logreg.odds_ratios(result.model).set_index("feature")
    assert table.loc["bmi", "odds_ratio"] > 1.2
    assert table.loc["age_band", "odds_ratio"] > 1.2
    top_two = set(logreg.odds_ratios(result.model)["feature"].head(2))
    assert top_two == {"bmi", "age_band"}


def test_odds_ratios_require_a_fitted_model(split) -> None:
    with pytest.raises(ValueError, match="fitted"):
        logreg.odds_ratios(logreg.build_logreg(split.predictors))


def test_clean_name_rules() -> None:
    assert logreg.clean_name("num__bmi") == "bmi"
    assert logreg.clean_name("num__missingindicator_income_band") == "income_band (was missing)"
    assert logreg.clean_name("bin__phys_active") == "phys_active"
    assert logreg.clean_name("cat__smoker_some_days") == "smoker = some_days"
    assert logreg.clean_name("cat__bmi_band_obese") == "bmi_band = obese"


# ---- command line -----------------------------------------------------------------------


def test_main_prints_metrics_and_saves_odds_ratios(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(evaluate, "load_processed", lambda path: signal_frame(n=1500))
    assert logreg.main(["--condition", FIRST, "--cv", "3", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "ROC-AUC" in out and "odds ratio" in out.lower()
    saved = pd.read_csv(tmp_path / FIRST / "logreg_odds_ratios.csv")
    assert {"feature", "odds_ratio"} <= set(saved.columns)


def test_main_unknown_condition(monkeypatch, capsys) -> None:
    monkeypatch.setattr(evaluate, "load_processed", lambda path: signal_frame(n=300))
    assert logreg.main(["--condition", "nope"]) == 1
    assert "unknown condition" in capsys.readouterr().err

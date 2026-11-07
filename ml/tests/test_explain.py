"""H14: SHAP explanations that add up exactly and read in canonical feature names."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import logit
from sklearn.dummy import DummyClassifier
from src import evaluate, explain, features, logreg, train
from src.registry import load_registry

from .test_logreg import signal_frame

FIRST = load_registry().ids[0]


@pytest.fixture(scope="module")
def split() -> evaluate.Split:
    return evaluate.split_condition(signal_frame(n=3000), FIRST)


@pytest.fixture(scope="module")
def linear(split) -> explain.Explainer:
    fit, _ = evaluate.validation_split(split)
    model = logreg.build_logreg(split.predictors).fit(fit.x, fit.y)
    return explain.build_explainer(model, explain.background_sample(split.x_train))


def one(split, i: int = 0) -> pd.DataFrame:
    return split.x_test.iloc[[i]]


# ---- names ------------------------------------------------------------------------------


def test_source_feature_maps_encoded_columns_back() -> None:
    assert explain.source_feature("num__bmi") == "bmi"
    assert explain.source_feature("num__missingindicator_income_band") == "income_band"
    assert explain.source_feature("bin__phys_active") == "phys_active"
    assert explain.source_feature("bin__missingindicator_phys_active") == "phys_active"
    assert explain.source_feature("cat__smoker_some_days") == "smoker"
    assert explain.source_feature("cat__bmi_band_obese") == "bmi_band"
    assert explain.source_feature("cat__sex_female") == "sex"


def test_every_encoded_column_belongs_to_a_predictor(linear, split) -> None:
    assert set(linear.sources) <= set(split.predictors)


# ---- the sanity check from the roadmap: contributions add up ------------------------------


def test_linear_base_plus_contributions_equals_model_margin(linear, split) -> None:
    x = split.x_test.iloc[:50]
    base, grouped = linear.contributions(x)
    clf = linear.model.named_steps["clf"]
    margin = clf.decision_function(linear.model.named_steps["prep"].transform(x))
    np.testing.assert_allclose(base + grouped.sum(axis=1).to_numpy(), margin, atol=1e-8)


def test_explain_probability_matches_predict_proba(linear, split) -> None:
    for i in range(5):
        row = one(split, i)
        result = linear.explain(row)
        assert result["probability"] == pytest.approx(linear.model.predict_proba(row)[0, 1])
        total = result["base_value"] + sum(c["contribution"] for c in result["contributions"])
        assert total == pytest.approx(result["margin"])


def test_grouping_keeps_the_total(linear, split) -> None:
    x = split.x_test.iloc[:20]
    _, encoded = linear._contributions(x)
    _, grouped = linear.contributions(x)
    np.testing.assert_allclose(encoded.sum(axis=1), grouped.sum(axis=1).to_numpy())


# ---- explain_one: the shape /predict returns -------------------------------------------


def test_explain_one_lists_every_canonical_feature_biggest_first(linear, split) -> None:
    items = linear.explain_one(one(split))
    assert [set(i) for i in items] == [{"feature", "value", "contribution"}] * len(items)
    assert sorted(i["feature"] for i in items) == sorted(split.predictors)
    assert not any("__" in i["feature"] for i in items)
    sizes = [abs(i["contribution"]) for i in items]
    assert sizes == sorted(sizes, reverse=True)
    json.dumps(items)


def test_explain_one_shows_readable_values(linear, split) -> None:
    row = one(split).copy()
    row["income_band"] = np.nan
    row["phys_active"] = 1.0
    row["sex"] = "female"
    values = {i["feature"]: i["value"] for i in linear.explain_one(row)}
    assert values["income_band"] is None
    assert values["phys_active"] is True
    assert values["sex"] == "female"
    assert isinstance(values["bmi"], float)


def test_the_planted_risk_factors_dominate(linear, split) -> None:
    table = explain.global_importance(linear, split.x_test)
    assert set(table["feature"].head(3)) >= {"age_band"}
    assert {"bmi", "bmi_band"} & set(table["feature"].head(3))


def test_explain_one_row_at_a_time(linear, split) -> None:
    with pytest.raises(ValueError, match="one row"):
        linear.explain(split.x_test.iloc[:2])


def test_unknown_model_type_is_rejected(split) -> None:
    fit, _ = evaluate.validation_split(split)
    from sklearn.pipeline import Pipeline

    dummy = Pipeline(
        [("prep", features.build_preprocessor(split.predictors)), ("clf", DummyClassifier())]
    ).fit(fit.x, fit.y)
    with pytest.raises(TypeError, match="no explainer"):
        explain.build_explainer(dummy, split.x_train)


def test_explainer_survives_save_and_load(linear, split, tmp_path) -> None:
    joblib.dump(linear, tmp_path / "e.joblib")
    again = joblib.load(tmp_path / "e.joblib")
    assert again.explain_one(one(split)) == linear.explain_one(one(split))


# ---- global importance + figures -------------------------------------------------------


def test_global_importance_table(linear, split) -> None:
    table = explain.global_importance(linear, split.x_test)
    assert list(table.columns) == ["feature", "mean_abs_contribution", "share_raising"]
    assert sorted(table["feature"]) == sorted(split.predictors)
    values = table["mean_abs_contribution"].to_numpy()
    assert (values[:-1] >= values[1:]).all()
    assert table["share_raising"].between(0, 1).all()


def test_save_figures_writes_the_bar_chart(linear, split, tmp_path) -> None:
    paths = explain.save_figures(linear, split.x_test.iloc[:300], tmp_path, "Test")
    assert (tmp_path / "shap_bar.png").stat().st_size > 1000
    assert all(p.exists() for p in paths)


# ---- train.py saves a ready-to-use explainer ---------------------------------------------


def test_train_saves_an_explainer_for_the_best_model(monkeypatch, tmp_path) -> None:
    frame = signal_frame(n=1500)
    monkeypatch.setattr(evaluate, "load_processed", lambda path: frame)
    assert (
        train.main(["--condition", FIRST, "--models", "logreg", "--fast", "--out", str(tmp_path)])
        == 0
    )
    explainer = joblib.load(tmp_path / FIRST / explain.EXPLAINER_FILE)
    x = evaluate.split_condition(features.add_features(frame), FIRST).x_test.iloc[[0]]
    result = explainer.explain(x)
    assert result["probability"] == pytest.approx(explainer.model.predict_proba(x)[0, 1])


def test_cli_writes_figures_and_importance(monkeypatch, tmp_path, capsys) -> None:
    frame = signal_frame(n=1500)
    monkeypatch.setattr(evaluate, "load_processed", lambda path: frame)
    models, figures = tmp_path / "models", tmp_path / "figures"
    train.main(["--condition", FIRST, "--models", "logreg", "--fast", "--out", str(models)])
    args = ["--condition", FIRST, "--models-dir", str(models), "--out", str(figures)]
    assert explain.main([*args, "--rows", "300"]) == 0
    assert (figures / FIRST / "shap_bar.png").exists()
    importance = json.loads((figures / FIRST / "logreg_global_importance.json").read_text())
    assert {"feature", "mean_abs_contribution"} <= set(importance[0])
    assert "top factors" in capsys.readouterr().out


def test_cli_untrained_condition(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(evaluate, "load_processed", lambda path: signal_frame(n=300))
    assert explain.main(["--condition", FIRST, "--models-dir", str(tmp_path)]) == 1
    assert "not trained" in capsys.readouterr().err


# ---- XGBoost (runs in CI, where xgboost and shap are installed) ---------------------------


@pytest.fixture(scope="module")
def tree(split) -> explain.Explainer:
    pytest.importorskip("xgboost")
    from src import boosting

    result = boosting.train_xgb(split, n_iter=2, cv=2)  # early-stopped, like real training
    return explain.build_explainer(result.model, explain.background_sample(split.x_train))


def test_tree_base_plus_contributions_equals_model_margin(tree, split) -> None:
    x = split.x_test.iloc[:50]
    base, grouped = tree.contributions(x)
    margin = logit(tree.model.predict_proba(x)[:, 1])
    np.testing.assert_allclose(base + grouped.sum(axis=1).to_numpy(), margin, atol=1e-4)


def test_tree_explain_one_shape(tree, split) -> None:
    items = tree.explain_one(one(split))
    assert sorted(i["feature"] for i in items) == sorted(split.predictors)
    json.dumps(items)


def test_tree_contributions_match_the_shap_library(split) -> None:
    pytest.importorskip("xgboost")
    shap = pytest.importorskip("shap")
    from src import boosting

    fit, _ = evaluate.validation_split(split)
    model = boosting.build_xgb(split.predictors, n_estimators=40, max_depth=3).fit(fit.x, fit.y)
    explainer = explain.build_explainer(model, split.x_train)
    x = split.x_test.iloc[:30]
    _, ours = explainer._contributions(x)
    encoded = model.named_steps["prep"].transform(x)
    theirs = shap.TreeExplainer(model.named_steps["clf"]).shap_values(encoded)
    np.testing.assert_allclose(ours, np.asarray(theirs), atol=1e-4)

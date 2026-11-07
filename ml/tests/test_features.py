"""H09: derived features and the shared preprocessor."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src import features, schema


def make_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """A schema-valid synthetic frame with realistic gaps, for fitting the preprocessor."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "survey_weight": rng.uniform(10, 500, n),
            "age_band": rng.integers(1, 14, n),
            "sex": rng.choice(["male", "female"], n),
            "bmi": rng.normal(28, 5, n).clip(14, 60),
            "smoker": rng.choice(["daily", "some_days", "former", "never"], n),
            "gen_health": rng.integers(1, 6, n),
            "income_band": rng.integers(1, 8, n).astype(float),
            "education_band": rng.integers(1, 5, n),
        }
    )
    for column in schema.BOOL_PREDICTORS:
        frame[column] = rng.random(n) < 0.4
    for column in schema.OUTCOMES:
        frame[column] = rng.random(n) < 0.15
    frame.loc[rng.random(n) < 0.2, "income_band"] = np.nan
    frame.loc[rng.random(n) < 0.1, "bmi"] = np.nan
    for column in (*schema.BOOL_PREDICTORS, *schema.OUTCOMES):
        frame[column] = frame[column].astype("boolean")
    frame["income_band"] = frame["income_band"].astype("Int64")
    frame["age_band"] = frame["age_band"].astype("Int64")
    frame["gen_health"] = frame["gen_health"].astype("Int64")
    frame["education_band"] = frame["education_band"].astype("Int64")
    return frame


def one_row(**overrides) -> pd.DataFrame:
    base = {
        "bmi": 22.0,
        "smoker": "never",
        "phys_active": True,
        "heavy_alcohol": False,
        "high_chol": False,
        "fruit_daily": True,
        "veg_daily": True,
    }
    base.update(overrides)
    frame = pd.DataFrame([base])
    for column in schema.BOOL_PREDICTORS:
        frame[column] = frame[column].astype("boolean")
    return frame


def values(series: pd.Series) -> list:
    return [None if pd.isna(v) else v for v in series.tolist()]


# ---- bmi_band -------------------------------------------------------------


def test_bmi_band_boundaries():
    frame = pd.DataFrame({"bmi": [15.0, 18.49, 18.5, 24.99, 25.0, 29.99, 30.0, 45.0, np.nan]})

    bands = features.add_features(frame)["bmi_band"]

    assert values(bands) == [
        "underweight",
        "underweight",
        "normal",
        "normal",
        "overweight",
        "overweight",
        "obese",
        "obese",
        None,
    ]


def test_bmi_band_is_an_ordered_category():
    bands = features.add_features(pd.DataFrame({"bmi": [20.0]}))["bmi_band"]

    assert list(bands.cat.categories) == ["underweight", "normal", "overweight", "obese"]
    assert bands.cat.ordered


# ---- risk_factor_count ----------------------------------------------------


def test_risk_factor_count_counts_all_five_risk_factors():
    row = one_row(bmi=32.0, smoker="daily", phys_active=False, heavy_alcohol=True, high_chol=True)

    assert features.add_features(row)["risk_factor_count"].tolist() == [5]


def test_risk_factor_count_is_zero_for_a_healthy_profile():
    assert features.add_features(one_row())["risk_factor_count"].tolist() == [0]


def test_some_days_smoker_counts_but_former_smoker_does_not():
    some_days = features.add_features(one_row(smoker="some_days"))["risk_factor_count"]
    former = features.add_features(one_row(smoker="former"))["risk_factor_count"]

    assert some_days.tolist() == [1]
    assert former.tolist() == [0]


def test_risk_factor_count_uses_known_answers_only():
    row = one_row(bmi=np.nan, smoker=None, phys_active=pd.NA, heavy_alcohol=pd.NA, high_chol=True)

    assert features.add_features(row)["risk_factor_count"].tolist() == [1]


def test_risk_factor_count_is_missing_when_nothing_is_known():
    row = one_row(bmi=np.nan, smoker=None, phys_active=pd.NA, heavy_alcohol=pd.NA, high_chol=pd.NA)

    assert pd.isna(features.add_features(row)["risk_factor_count"].iloc[0])


# ---- lifestyle_score ------------------------------------------------------


def test_lifestyle_score_counts_five_healthy_habits():
    assert features.add_features(one_row())["lifestyle_score"].tolist() == [5]


def test_lifestyle_score_is_zero_for_no_healthy_habits():
    row = one_row(
        smoker="daily", phys_active=False, heavy_alcohol=True, fruit_daily=False, veg_daily=False
    )

    assert features.add_features(row)["lifestyle_score"].tolist() == [0]


def test_lifestyle_score_is_missing_when_nothing_is_known():
    row = one_row(
        smoker=None,
        phys_active=pd.NA,
        heavy_alcohol=pd.NA,
        fruit_daily=pd.NA,
        veg_daily=pd.NA,
    )

    assert pd.isna(features.add_features(row)["lifestyle_score"].iloc[0])


# ---- add_features properties ----------------------------------------------


def test_add_features_does_not_modify_its_input():
    frame = make_frame(20)
    before = frame.copy()

    features.add_features(frame)

    pd.testing.assert_frame_equal(frame, before)


def test_derived_features_never_depend_on_outcome_columns():
    frame = make_frame(50)
    with_outcomes = features.add_features(frame)
    without = features.add_features(frame.drop(columns=list(schema.OUTCOMES)))

    for column in features.DERIVED_COLUMNS:
        pd.testing.assert_series_equal(with_outcomes[column], without[column])


def test_add_features_is_deterministic_and_keeps_rows():
    frame = make_frame(50)

    first, second = features.add_features(frame), features.add_features(frame)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 50


# ---- model_frame ----------------------------------------------------------


def test_model_frame_has_plain_numpy_friendly_dtypes():
    frame = features.add_features(make_frame(30))

    out = features.model_frame(frame, ["bmi", "phys_active", "income_band", "sex", "bmi_band"])

    assert list(out.columns) == ["bmi", "phys_active", "income_band", "sex", "bmi_band"]
    assert out["phys_active"].dtype == "float64"
    assert out["income_band"].dtype == "float64"
    assert out["sex"].dtype == object
    assert out["bmi_band"].dtype == object


def test_model_frame_keeps_missing_as_nan():
    frame = pd.DataFrame({"phys_active": pd.array([True, False, pd.NA], dtype="boolean")})

    out = features.model_frame(frame, ["phys_active"])

    assert out["phys_active"].tolist()[:2] == [1.0, 0.0]
    assert np.isnan(out["phys_active"].iloc[2])


def test_model_frame_names_unknown_columns():
    with pytest.raises(KeyError, match="nope"):
        features.model_frame(make_frame(5), ["bmi", "nope"])


# ---- predictors_for (leakage guard) ---------------------------------------


def test_predictors_exclude_every_outcome():
    for condition_id in ("diabetes", "heart_disease"):
        predictors = features.predictors_for(condition_id)
        assert not set(predictors) & set(schema.OUTCOMES), condition_id


def test_predictors_include_base_and_derived_features_but_not_the_weight():
    predictors = features.predictors_for("diabetes")

    assert {"bmi", "age_band", "risk_factor_count", "lifestyle_score", "bmi_band"} <= set(
        predictors
    )
    assert "survey_weight" not in predictors


# ---- build_preprocessor ---------------------------------------------------


def fit_data(n: int = 300):
    frame = features.add_features(make_frame(n))
    predictors = features.predictors_for("diabetes")
    return features.model_frame(frame, predictors), predictors


def test_preprocessor_outputs_numbers_with_no_gaps():
    x, predictors = fit_data()
    pre = features.build_preprocessor(predictors)

    out = pre.fit_transform(x)

    assert out.shape[0] == len(x)
    assert not np.isnan(out).any()


def test_preprocessor_flags_missing_values():
    x, predictors = fit_data()
    pre = features.build_preprocessor(predictors).fit(x)

    names = list(pre.get_feature_names_out())

    assert any("income_band" in n and "missing" in n for n in names)


def test_preprocessor_ignores_categories_it_never_saw():
    x, predictors = fit_data()
    pre = features.build_preprocessor(predictors).fit(x)
    new = x.head(3).copy()
    new["smoker"] = "vapes"

    assert not np.isnan(pre.transform(new)).any()


def test_preprocessor_learns_fill_values_from_training_data_only():
    x, predictors = fit_data()
    train = x.iloc[:200]
    other = x.iloc[200:].copy()
    pre = features.build_preprocessor(predictors).fit(train)
    train_median = train["bmi"].median()

    gap, filled = other.head(1).copy(), other.head(1).copy()
    gap["bmi"] = np.nan
    filled["bmi"] = train_median

    names = list(pre.get_feature_names_out())
    column = names.index("num__bmi")
    assert pre.transform(gap)[0, column] == pytest.approx(pre.transform(filled)[0, column])


def test_preprocessor_trains_a_model_end_to_end():
    x, predictors = fit_data()
    y = (x["bmi"].fillna(28) + np.random.default_rng(1).normal(0, 3, len(x)) > 30).astype(int)
    model = Pipeline(
        [
            ("prep", features.build_preprocessor(predictors)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    ).fit(x, y)

    proba = model.predict_proba(x)[:, 1]

    assert ((proba >= 0) & (proba <= 1)).all()


def test_preprocessor_rejects_unknown_columns():
    with pytest.raises(ValueError, match="mystery"):
        features.build_preprocessor(["bmi", "mystery"])

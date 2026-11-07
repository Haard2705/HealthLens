"""Derived features and the shared preprocessor.

Two kinds of work live here, kept apart on purpose:

* **Stateless, row-by-row features** (`add_features`): the same answer for a row no matter
  what other rows exist, so they are safe to compute before splitting the data.
* **Stateful transforms** (`build_preprocessor`): fill gaps, scale, one-hot encode. They learn
  from data, so they are built as scikit-learn steps that a `Pipeline` fits on training rows only.

Derived features read only non-outcome predictors, so they cannot leak an outcome.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import schema
from .registry import feature_columns, load_registry

BMI_BANDS = ("underweight", "normal", "overweight", "obese")
BMI_EDGES = (-np.inf, 18.5, 25.0, 30.0, np.inf)
OBESE_BMI = 30.0
CURRENT_SMOKER = ("daily", "some_days")

DERIVED_COLUMNS = ("bmi_band", "risk_factor_count", "lifestyle_score")
NUMERIC = (
    "bmi",
    "age_band",
    "gen_health",
    "income_band",
    "education_band",
    "risk_factor_count",
    "lifestyle_score",
)
BINARY = schema.BOOL_PREDICTORS
CATEGORICAL = ("sex", "smoker", "bmi_band")
# Every predictor a model may start from, before the registry removes outcomes.
BASE_PREDICTORS = (*NUMERIC, *BINARY, *CATEGORICAL)


def _flag(df: pd.DataFrame, column: str) -> pd.Series:
    """A True/False/missing column as floats 1.0 / 0.0 / NaN (all NaN if the column is absent)."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    values = df[column].astype("boolean").to_numpy(dtype="float64", na_value=np.nan)
    return pd.Series(values, index=df.index)


def _current_smoker(df: pd.DataFrame) -> pd.Series:
    if "smoker" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    smoker = df["smoker"]
    return smoker.isin(CURRENT_SMOKER).astype("float64").where(smoker.notna())


def _obese(df: pd.DataFrame) -> pd.Series:
    if "bmi" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    bmi = df["bmi"].astype("float64")
    return (bmi >= OBESE_BMI).astype("float64").where(bmi.notna())


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with `bmi_band`, `risk_factor_count` and `lifestyle_score` added.

    The two counts use the answers that are known; they are missing only when none are known.
    """
    out = df.copy()

    if "bmi" in out.columns:
        banded = pd.cut(out["bmi"].astype("float64"), bins=list(BMI_EDGES), right=False)
        out["bmi_band"] = pd.Categorical.from_codes(
            banded.cat.codes, categories=BMI_BANDS, ordered=True
        )
    else:
        out["bmi_band"] = pd.Categorical([None] * len(out), categories=BMI_BANDS, ordered=True)

    smoker, phys_active = _current_smoker(df), _flag(df, "phys_active")
    heavy_alcohol = _flag(df, "heavy_alcohol")
    risks = pd.DataFrame(
        {
            "smoker": smoker,
            "inactive": 1 - phys_active,
            "heavy_alcohol": heavy_alcohol,
            "high_chol": _flag(df, "high_chol"),
            "obese": _obese(df),
        }
    )
    habits = pd.DataFrame(
        {
            "active": phys_active,
            "fruit": _flag(df, "fruit_daily"),
            "veg": _flag(df, "veg_daily"),
            "no_heavy_alcohol": 1 - heavy_alcohol,
            "non_smoker": 1 - smoker,
        }
    )
    out["risk_factor_count"] = risks.sum(axis=1, min_count=1)
    out["lifestyle_score"] = habits.sum(axis=1, min_count=1)
    return out


def model_frame(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Select `columns` and convert pandas' nullable types to plain float / object columns.

    scikit-learn cannot handle `pd.NA`; here True/False/missing becomes 1.0/0.0/NaN, nullable
    integers become floats, and categories become strings with NaN for missing.
    """
    absent = [c for c in columns if c not in df.columns]
    if absent:
        raise KeyError(f"columns not found: {', '.join(absent)}")

    out = pd.DataFrame(index=df.index)
    for column in columns:
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            out[column] = series.astype("boolean").to_numpy(dtype="float64", na_value=np.nan)
        elif pd.api.types.is_numeric_dtype(series):
            out[column] = series.to_numpy(dtype="float64", na_value=np.nan)
        else:
            out[column] = series.astype(object).where(series.notna(), np.nan)
    return out


def predictors_for(condition_id: str) -> tuple[str, ...]:
    """Columns a condition's model may use, with other outcomes removed (leakage guard)."""
    return feature_columns(load_registry(), condition_id, BASE_PREDICTORS)


def build_preprocessor(predictors: list[str] | tuple[str, ...]) -> ColumnTransformer:
    """The shared, unfitted preprocessor. Fit it on training rows only (inside a Pipeline).

    Numbers: median fill + "was missing" flag + scaling. Yes/no: most-frequent fill + flag.
    Text: most-frequent fill + one-hot (unseen categories are ignored).
    """
    known = {*NUMERIC, *BINARY, *CATEGORICAL, *schema.OUTCOMES}
    unknown = [c for c in predictors if c not in known]
    if unknown:
        raise ValueError(f"unknown predictor column(s): {', '.join(unknown)}")

    numeric = [c for c in predictors if c in NUMERIC]
    categorical = [c for c in predictors if c in CATEGORICAL]
    binary = [c for c in predictors if c not in numeric and c not in categorical]

    groups = []
    if numeric:
        steps = [SimpleImputer(strategy="median", add_indicator=True), StandardScaler()]
        groups.append(("num", Pipeline([("fill", steps[0]), ("scale", steps[1])]), numeric))
    if binary:
        fill = SimpleImputer(strategy="most_frequent", add_indicator=True)
        groups.append(("bin", fill, binary))
    if categorical:
        steps = [
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
        groups.append(("cat", Pipeline(steps), categorical))
    return ColumnTransformer(groups, remainder="drop")

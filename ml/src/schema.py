"""The canonical schema: our own column names, types and allowed values.

Every data source (BRFSS today, CCHS later) is mapped onto this by an adapter in
`ml/src/sources/`. Nothing downstream of the adapters knows raw survey column names.
Missing answers are always missing (NaN / <NA>), never "no".
"""

import pandas as pd

OUTCOMES = (
    "diabetes",
    "hypertension",
    "heart_disease",
    "stroke",
    "asthma",
    "copd",
    "arthritis",
    "kidney_disease",
    "depression",
)
BOOL_PREDICTORS = ("phys_active", "heavy_alcohol", "fruit_daily", "veg_daily", "high_chol")
CATEGORIES = {
    "sex": {"male", "female"},
    "smoker": {"daily", "some_days", "former", "never"},
}
INT_RANGES = {
    "age_band": (1, 13),
    "gen_health": (1, 5),
    "income_band": (1, 7),
    "education_band": (1, 4),
}
BMI_RANGE = (10.0, 100.0)

REQUIRED = ("survey_weight", "age_band", "sex", "bmi")
ALL_COLUMNS = (
    *REQUIRED,
    "smoker",
    "gen_health",
    "income_band",
    "education_band",
    *BOOL_PREDICTORS,
    *OUTCOMES,
)


class SchemaError(ValueError):
    """The frame does not match the canonical schema."""


def validate_schema(df: pd.DataFrame) -> None:
    """Raise SchemaError listing every problem found; return None when the frame is valid."""
    problems: list[str] = []

    problems += [f"missing required column: {c}" for c in REQUIRED if c not in df.columns]
    if not any(c in df.columns for c in OUTCOMES):
        problems.append(f"no outcome column present (expected one of {', '.join(OUTCOMES)})")
    problems += [f"unknown column: {c}" for c in df.columns if c not in ALL_COLUMNS]

    def present(column: str) -> pd.Series:
        return df[column].dropna()

    if "survey_weight" in df.columns and (present("survey_weight") <= 0).any():
        problems.append("survey_weight must be positive")
    if "bmi" in df.columns:
        low, high = BMI_RANGE
        if (~present("bmi").between(low, high)).any():
            problems.append(f"bmi outside {low}-{high}")
    for column, allowed in CATEGORIES.items():
        if column in df.columns and not set(present(column)) <= allowed:
            problems.append(f"{column} has values outside {sorted(allowed)}")
    for column, (low, high) in INT_RANGES.items():
        if column in df.columns and (~present(column).between(low, high)).any():
            problems.append(f"{column} outside {low}-{high}")
    for column in (*BOOL_PREDICTORS, *OUTCOMES):
        if column in df.columns and not set(present(column)) <= {True, False}:
            problems.append(f"{column} must be True/False/missing")

    if problems:
        raise SchemaError("; ".join(problems))

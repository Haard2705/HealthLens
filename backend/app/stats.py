"""Survey-weighted prevalence, computed from the cleaned BRFSS parquet when present.

Input is never interpolated: ``condition`` is looked up in the registry and ``by`` in a fixed
table of groupers. Groups with fewer than MIN_GROUP_SIZE respondents are suppressed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

MIN_GROUP_SIZE = 30
BMI_BANDS = (0, 18.5, 25, 30, 1000)
BMI_LABELS = ("under 18.5", "18.5-24.9", "25-29.9", "30+")
AGE_BAND_LABELS = {  # BRFSS 13-level age category
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54",
    8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+",
}  # fmt: skip


def _bmi_band(df: pd.DataFrame) -> pd.Series:
    return pd.cut(df["bmi"], bins=list(BMI_BANDS), labels=list(BMI_LABELS), right=False)


def _label_column(column: str, mapping: dict[Any, str]) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: df[column].map(mapping)


# `by` value -> (required column, function making the group labels)
GROUPERS: dict[str, tuple[str, Callable[[pd.DataFrame], pd.Series]]] = {
    "age_band": ("age_band", _label_column("age_band", AGE_BAND_LABELS)),
    "bmi_band": ("bmi", _bmi_band),
    "phys_active": (
        "phys_active",
        _label_column("phys_active", {True: "active", False: "inactive"}),
    ),
    "sex": ("sex", lambda df: df["sex"]),
}
BY_OPTIONS = tuple(GROUPERS)

# Clearly-labelled illustrative numbers used until the real parquet is available.
SAMPLE_GROUPS = {
    "age_band": ("18-34", "35-49", "50-64", "65+"),
    "bmi_band": BMI_LABELS,
    "phys_active": ("active", "inactive"),
    "sex": ("female", "male"),
}


def sample_prevalence(condition_id: str, by: str) -> dict[str, Any]:
    labels = SAMPLE_GROUPS[by]
    groups = [
        {"group": label, "prevalence": round(0.04 + 0.03 * i, 3), "n": 1000 + 100 * i}
        for i, label in enumerate(labels)
    ]
    return {
        "condition": condition_id,
        "by": by,
        "source": "sample",
        "note": "Illustrative sample numbers, NOT survey results.",
        "groups": groups,
    }


def weighted_prevalence(
    df: pd.DataFrame, outcome_column: str, condition_id: str, by: str
) -> dict[str, Any]:
    column, make_labels = GROUPERS[by]
    needed = {column, outcome_column, "survey_weight"}
    if not needed <= set(df.columns):
        raise KeyError(f"parquet is missing columns {sorted(needed - set(df.columns))}")
    frame = df[list(needed)].dropna(subset=[outcome_column, "survey_weight", column]).copy()
    frame["group"] = make_labels(frame)
    frame["hit"] = frame[outcome_column].astype(float) * frame["survey_weight"]
    grouped = frame.groupby("group", observed=True).agg(
        hit=("hit", "sum"), weight=("survey_weight", "sum"), n=("group", "size")
    )
    groups = [
        {"group": str(label), "prevalence": round(float(row.hit / row.weight), 4), "n": int(row.n)}
        for label, row in grouped.iterrows()
        if row.n >= MIN_GROUP_SIZE and row.weight > 0
    ]
    return {"condition": condition_id, "by": by, "source": "brfss", "groups": groups}


def prevalence(path: Path, outcome_column: str, condition_id: str, by: str) -> dict[str, Any]:
    if not path.exists():
        return sample_prevalence(condition_id, by)
    return weighted_prevalence(pd.read_parquet(path), outcome_column, condition_id, by)

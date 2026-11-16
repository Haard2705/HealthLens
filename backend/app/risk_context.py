"""H19: a patient record turned into model answers, plus the "people like this" comparison.

The record holds what a clinic knows (age in years, sex as M/F, a smoking status); the model
expects the canonical survey columns (an age band, "male"/"female", a smoker category). This
module is the only place that mapping lives, and it says "not answered" rather than guessing:
income, education, diet and alcohol are simply not in a HealthLens record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.registry import Registry

from . import featurespec, stats
from .data.models import Patient

# BRFSS age categories: 1 = 18-24, then five-year bands, 13 = 80 or older.
FIRST_BAND_MAX_AGE = 24
LAST_BAND_MIN_AGE = 80
LAST_BAND = 13
SEX_LABELS = {"m": "male", "f": "female", "male": "male", "female": "female"}


def age_band(age: int) -> int:
    """The BRFSS 13-level age band for an age in years."""
    if age <= FIRST_BAND_MAX_AGE:
        return 1
    if age >= LAST_BAND_MIN_AGE:
        return LAST_BAND
    return 2 + (age - 25) // 5


def answers_from_record(patient: Patient, registry: Registry) -> dict[str, Any]:
    """Canonical answers for this patient. Anything the record does not know is left out."""
    answers: dict[str, Any] = {
        "age_band": age_band(patient.age),
        "sex": SEX_LABELS.get(patient.sex.strip().lower()),
        "bmi": patient.bmi,
        "smoker": patient.smoking_status,
        "gen_health": patient.general_health,
        "phys_active": patient.physically_active,
    }
    # A condition used as a predictor (registry comorbidity opt-in) is answered from the record.
    for condition in registry.conditions:
        answers[condition.outcome_column] = condition.id in patient.conditions
    return answers


def supported_answers(
    patient: Patient, registry: Registry, feature_spec: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    """``answers_from_record`` narrowed to the answers this model actually asks for."""
    wanted = {entry["name"] for entry in featurespec.answerable(feature_spec)}
    return {k: v for k, v in answers_from_record(patient, registry).items() if k in wanted}


# --------------------------------------------------------------------------
# "People like this": the observed rate in the survey for the same age and BMI band
# --------------------------------------------------------------------------


def bmi_band_label(bmi: float) -> str | None:
    """The band label this BMI falls into, using the same edges as /stats/prevalence."""
    edges, labels = stats.BMI_BANDS, stats.BMI_LABELS
    for index in range(len(labels)):
        if edges[index] <= bmi < edges[index + 1]:
            return labels[index]
    return None


def _band_rows(df: pd.DataFrame, band: int, bmi_label: str) -> pd.DataFrame:
    bands = pd.cut(
        df["bmi"].astype("float64"),
        bins=list(stats.BMI_BANDS),
        labels=list(stats.BMI_LABELS),
        right=False,
    )
    return df[(df["age_band"] == band) & (bands == bmi_label)]


def patients_like_this(path: Path, outcome_column: str, patient: Patient) -> dict[str, Any] | None:
    """Survey-weighted rate for respondents in the same age and BMI band, or None.

    None means "cannot be shown": no parquet, missing columns, or fewer than
    ``stats.MIN_GROUP_SIZE`` respondents in the cell (the same suppression rule as the
    population endpoints).
    """
    if not path.exists():
        return None
    bmi_label = bmi_band_label(patient.bmi)
    if bmi_label is None:
        return None
    df = pd.read_parquet(path)
    needed = {"age_band", "bmi", "survey_weight", outcome_column}
    if not needed <= set(df.columns):
        return None

    band = age_band(patient.age)
    rows = _band_rows(df[list(needed)].dropna(subset=list(needed)), band, bmi_label)
    weight = float(rows["survey_weight"].sum())
    if len(rows) < stats.MIN_GROUP_SIZE or weight <= 0:
        return None
    hits = float((rows[outcome_column].astype(float) * rows["survey_weight"]).sum())
    return {
        "rate": round(hits / weight, 4),
        "n": int(len(rows)),
        "age_band": stats.AGE_BAND_LABELS.get(band, str(band)),
        "bmi_band": bmi_label,
    }

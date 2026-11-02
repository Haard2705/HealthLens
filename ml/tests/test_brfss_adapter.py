"""H07: the BRFSS 2025 adapter maps raw coded columns to the canonical schema.

Raw names and codes appear here on purpose: this is the one place besides
`ml/src/sources/brfss.py` that is allowed to know them.
"""

import pandas as pd
import pytest
from src import schema
from src.sources import brfss

# One benign respondent; each test overrides only the columns it cares about.
BASE_ROW = {
    "DIABETE4": 3, "BPHIGH6": 3, "CVDINFR4": 2, "CVDCRHD4": 2, "CVDSTRK3": 2,
    "ASTHMA3": 2, "CHCCOPD3": 2, "CHCKDNY2": 2, "ADDEPEV3": 2, "HAVARTH4": 2,
    "_AGEG5YR": 5, "SEXVAR": 1, "_BMI5": 2500, "_SMOKER3": 4, "_TOTINDA": 1,
    "_RFDRHV9": 1, "_FRTLT1A": 1, "_VEGLT1A": 1, "GENHLTH": 3, "INCOME3": 6,
    "EDUCA": 4, "TOLDHI3": 2, "_LLCPWT": 100.0,
}  # fmt: skip


def adapt(**columns) -> pd.DataFrame:
    """Build a raw frame from BASE_ROW with the given columns replaced, then adapt it."""
    n = len(next(iter(columns.values())))
    raw = pd.DataFrame({name: [float(v)] * n for name, v in BASE_ROW.items()})
    for name, values in columns.items():
        raw[name] = [float(v) if v is not None else float("nan") for v in values]
    return brfss.to_canonical(raw)


def values(series: pd.Series) -> list:
    return [None if pd.isna(v) else v for v in series.tolist()]


YES_NO = [1, 2, 7, 9, None]
YES_NO_EXPECTED = [True, False, None, None, None]


@pytest.mark.parametrize(
    ("raw_col", "canon_col"),
    [
        ("CVDSTRK3", "stroke"),
        ("ASTHMA3", "asthma"),
        ("CHCCOPD3", "copd"),
        ("CHCKDNY2", "kidney_disease"),
        ("ADDEPEV3", "depression"),
        ("HAVARTH4", "arthritis"),
        ("TOLDHI3", "high_chol"),
        ("_TOTINDA", "phys_active"),
        ("_FRTLT1A", "fruit_daily"),
        ("_VEGLT1A", "veg_daily"),
    ],
)
def test_plain_yes_no_columns(raw_col, canon_col):
    out = adapt(**{raw_col: YES_NO})
    assert values(out[canon_col]) == YES_NO_EXPECTED


def test_diabetes_counts_pregnancy_only_and_prediabetes_as_no():
    out = adapt(DIABETE4=[1, 2, 3, 4, 7, 9, None])
    assert values(out["diabetes"]) == [True, False, False, False, None, None, None]


def test_hypertension_counts_pregnancy_only_and_borderline_as_no():
    out = adapt(BPHIGH6=[1, 2, 3, 4, 7, 9, None])
    assert values(out["hypertension"]) == [True, False, False, False, None, None, None]


def test_heavy_alcohol_code_2_means_yes():
    out = adapt(_RFDRHV9=[1, 2, 9, None])
    assert values(out["heavy_alcohol"]) == [False, True, None, None]


@pytest.mark.parametrize(
    ("heart_attack", "chd", "expected"),
    [
        (1, 2, True),
        (2, 1, True),
        (1, 1, True),
        (2, 2, False),
        (7, 2, None),
        (2, 9, None),
        (None, 2, None),
        (None, 1, True),  # one "yes" is enough even if the other is unknown
    ],
)
def test_heart_disease_is_heart_attack_or_chd(heart_attack, chd, expected):
    out = adapt(CVDINFR4=[heart_attack], CVDCRHD4=[chd])
    assert values(out["heart_disease"]) == [expected]


def test_age_band_keeps_1_to_13_and_drops_14():
    out = adapt(_AGEG5YR=[1, 13, 14, None])
    assert values(out["age_band"]) == [1, 13, None, None]


def test_sex_labels():
    assert values(adapt(SEXVAR=[1, 2, 9])["sex"]) == ["male", "female", None]


def test_bmi_has_two_implied_decimals():
    out = adapt(_BMI5=[2750, 1234, None])
    assert values(out["bmi"]) == [27.5, 12.34, None]


def test_smoker_levels():
    out = adapt(_SMOKER3=[1, 2, 3, 4, 9])
    assert values(out["smoker"]) == ["daily", "some_days", "former", "never", None]


def test_general_health_keeps_1_to_5():
    out = adapt(GENHLTH=[1, 5, 7, 9, None])
    assert values(out["gen_health"]) == [1, 5, None, None, None]


def test_income_codes_are_grouped_into_seven_bands():
    raw = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 77, 99, None]
    out = adapt(INCOME3=raw)
    assert values(out["income_band"]) == [1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7, None, None, None]


def test_education_codes_are_grouped_into_four_levels():
    out = adapt(EDUCA=[1, 2, 3, 4, 5, 6, 9, None])
    assert values(out["education_band"]) == [1, 1, 1, 2, 3, 4, None, None]


def test_survey_weight_passes_through():
    assert values(adapt(_LLCPWT=[12.5, 800.0])["survey_weight"]) == [12.5, 800.0]


def test_output_passes_schema_validation():
    out = adapt(DIABETE4=[1, 3, 7], _BMI5=[2750, None, 3100], INCOME3=[1, 77, 11])
    schema.validate_schema(out)


def test_output_has_every_canonical_column():
    out = adapt(DIABETE4=[1])
    assert set(out.columns) == set(schema.ALL_COLUMNS)


def test_row_count_and_input_are_preserved():
    raw = pd.DataFrame({name: [float(v)] * 4 for name, v in BASE_ROW.items()})
    before = raw.copy()
    out = brfss.to_canonical(raw)
    assert len(out) == 4
    pd.testing.assert_frame_equal(raw, before)


def test_missing_raw_column_names_the_column():
    raw = pd.DataFrame({name: [float(v)] for name, v in BASE_ROW.items()}).drop(columns="_BMI5")
    with pytest.raises(KeyError, match="_BMI5"):
        brfss.to_canonical(raw)

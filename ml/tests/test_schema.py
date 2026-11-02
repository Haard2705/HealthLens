"""H07: the canonical schema and its validator."""

import pandas as pd
import pytest
from src import schema
from src.registry import load_registry
from src.registry import validate_against_schema as reg_validate_against_schema


@pytest.fixture
def good() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "survey_weight": [120.5, 80.0],
            "age_band": [3, 13],
            "sex": ["male", "female"],
            "bmi": [27.5, 31.2],
            "smoker": ["never", "daily"],
            "gen_health": [2, 5],
            "income_band": [1, 7],
            "education_band": [4, 1],
            "phys_active": [True, False],
            "diabetes": [True, False],
            "high_bp": [False, pd.NA],
        }
    ).astype({"phys_active": "boolean", "diabetes": "boolean", "high_bp": "boolean"})


def test_schema_outcomes_come_from_the_condition_registry():
    registry = load_registry()

    assert registry.outcome_columns == schema.OUTCOMES
    reg_validate_against_schema(registry, schema.ALL_COLUMNS)


def test_valid_frame_passes(good):
    schema.validate_schema(good)


def test_missing_values_are_allowed(good):
    good.loc[0, "bmi"] = float("nan")
    good.loc[0, "sex"] = None
    schema.validate_schema(good)


@pytest.mark.parametrize("column", ["survey_weight", "age_band", "sex", "bmi"])
def test_missing_required_column_is_rejected(good, column):
    with pytest.raises(schema.SchemaError, match=column):
        schema.validate_schema(good.drop(columns=column))


def test_frame_without_any_outcome_is_rejected(good):
    with pytest.raises(schema.SchemaError, match="outcome"):
        schema.validate_schema(good.drop(columns=["diabetes", "high_bp"]))


@pytest.mark.parametrize("bmi", [5.0, 150.0])
def test_out_of_range_bmi_is_rejected(good, bmi):
    good.loc[0, "bmi"] = bmi
    with pytest.raises(schema.SchemaError, match="bmi"):
        schema.validate_schema(good)


def test_unknown_category_is_rejected(good):
    good.loc[0, "sex"] = "other"
    with pytest.raises(schema.SchemaError, match="sex"):
        schema.validate_schema(good)


@pytest.mark.parametrize(
    ("column", "value"), [("age_band", 14), ("gen_health", 0), ("income_band", 8)]
)
def test_band_outside_its_range_is_rejected(good, column, value):
    good.loc[0, column] = value
    with pytest.raises(schema.SchemaError, match=column):
        schema.validate_schema(good)


def test_non_positive_weight_is_rejected(good):
    good.loc[0, "survey_weight"] = 0.0
    with pytest.raises(schema.SchemaError, match="survey_weight"):
        schema.validate_schema(good)


def test_unknown_column_is_rejected(good):
    good["mystery"] = 1
    with pytest.raises(schema.SchemaError, match="mystery"):
        schema.validate_schema(good)


def test_all_problems_are_reported_together(good):
    good.loc[0, "bmi"] = 500.0
    good.loc[0, "sex"] = "other"
    with pytest.raises(schema.SchemaError) as err:
        schema.validate_schema(good)
    assert "bmi" in str(err.value)
    assert "sex" in str(err.value)

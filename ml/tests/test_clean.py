"""H08: cleaning the canonical frame and saving it to data/processed/."""

import pandas as pd
import pytest
from src import clean, schema

from tests.test_brfss_adapter import BASE_ROW


@pytest.fixture
def canonical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "survey_weight": [120.5, 80.0, 40.0],
            "age_band": [3, 13, 7],
            "sex": ["male", "female", "female"],
            "bmi": [27.5, 31.2, 22.0],
            "smoker": ["never", "daily", None],
            "gen_health": [2, 5, 3],
            "income_band": [1, 7, 4],
            "education_band": [4, 1, 2],
            "phys_active": [True, False, pd.NA],
            "diabetes": [True, False, pd.NA],
        }
    ).astype({"phys_active": "boolean", "diabetes": "boolean"})


def test_valid_frame_comes_back_unchanged(canonical):
    cleaned, report = clean.clean(canonical)

    pd.testing.assert_frame_equal(cleaned, canonical, check_dtype=False, check_categorical=False)
    assert report["rows_in"] == report["rows_out"] == 3


def test_missing_answers_stay_missing_not_false(canonical):
    cleaned, _ = clean.clean(canonical)

    assert pd.isna(cleaned.loc[2, "diabetes"])
    assert pd.isna(cleaned.loc[2, "phys_active"])


@pytest.mark.parametrize("bad_bmi", [5.0, 250.0])
def test_out_of_range_bmi_becomes_missing(canonical, bad_bmi):
    canonical.loc[0, "bmi"] = bad_bmi

    cleaned, report = clean.clean(canonical)

    assert pd.isna(cleaned.loc[0, "bmi"])
    assert cleaned.loc[1, "bmi"] == 31.2
    assert report["bmi_set_missing"] == 1


@pytest.mark.parametrize("bad_weight", [0.0, -3.0, float("nan")])
def test_rows_without_a_positive_weight_are_dropped(canonical, bad_weight):
    canonical.loc[1, "survey_weight"] = bad_weight

    cleaned, report = clean.clean(canonical)

    assert len(cleaned) == 2
    assert list(cleaned.index) == [0, 1]
    assert report["rows_dropped_invalid_weight"] == 1
    assert report["rows_in"] == 3
    assert report["rows_out"] == 2


def test_text_columns_become_categories_with_fixed_levels(canonical):
    cleaned, _ = clean.clean(canonical)

    assert isinstance(cleaned["sex"].dtype, pd.CategoricalDtype)
    assert list(cleaned["sex"].cat.categories) == ["female", "male"]
    assert list(cleaned["smoker"].cat.categories) == ["daily", "former", "never", "some_days"]


def test_output_passes_schema_validation(canonical):
    canonical.loc[0, "bmi"] = 500.0
    cleaned, _ = clean.clean(canonical)

    schema.validate_schema(cleaned)


def test_input_is_not_modified(canonical):
    canonical.loc[0, "bmi"] = 500.0
    before = canonical.copy()

    clean.clean(canonical)

    pd.testing.assert_frame_equal(canonical, before)


def test_cleaning_twice_gives_identical_results(canonical):
    first, _ = clean.clean(canonical)
    second, _ = clean.clean(canonical)

    pd.testing.assert_frame_equal(first, second)


def test_report_counts_missing_values_per_column(canonical):
    _, report = clean.clean(canonical)

    assert report["missing_by_column"] == {"smoker": 1, "phys_active": 1, "diabetes": 1}


def test_format_report_is_readable(canonical):
    _, report = clean.clean(canonical)

    text = clean.format_report(report)

    assert "rows in: 3" in text
    assert "rows out: 3" in text


def test_save_round_trips_and_is_byte_identical_on_rerun(canonical, tmp_path):
    cleaned, _ = clean.clean(canonical)
    one, two = tmp_path / "one.parquet", tmp_path / "two.parquet"

    clean.save(cleaned, one)
    clean.save(cleaned, two)

    assert one.read_bytes() == two.read_bytes()
    pd.testing.assert_frame_equal(pd.read_parquet(one), cleaned)


def test_main_runs_raw_file_to_parquet(tmp_path):
    raw = pd.DataFrame({name: [float(v)] * 3 for name, v in BASE_ROW.items()})
    raw.loc[2, "_LLCPWT"] = 0.0  # one row with no usable weight
    raw_path, out_path = tmp_path / "raw.csv", tmp_path / "out" / "brfss_clean.parquet"
    raw.to_csv(raw_path, index=False)

    clean.main(["--path", str(raw_path), "--out", str(out_path)])

    result = pd.read_parquet(out_path)
    assert len(result) == 2
    schema.validate_schema(result)

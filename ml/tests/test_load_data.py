"""H06: loading and profiling a raw survey file. Runs without the real dataset."""

import pandas as pd
import pytest
from src import load_data


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 2.0, 7.0, 1.0],
            "B": [1.0, None, 3.0, 9.0, 1.0],
            "C": [10.5, 20.5, 30.5, 40.5, 10.5],
            "Y": [1.0, 2.0, 2.0, 2.0, 1.0],
        }
    )


def test_load_raw_reads_csv(tmp_path, tiny_df):
    path = tmp_path / "tiny.csv"
    tiny_df.to_csv(path, index=False)

    loaded = load_data.load_raw(path)

    assert loaded.shape == (5, 4)


def test_load_raw_uses_xport_reader_for_xpt(tmp_path, tiny_df, monkeypatch):
    path = tmp_path / "tiny.XPT"
    path.write_bytes(b"")
    calls = {}

    def fake_read_sas(p, **kwargs):
        calls["path"], calls["kwargs"] = p, kwargs
        return tiny_df

    monkeypatch.setattr(load_data.pd, "read_sas", fake_read_sas)

    loaded = load_data.load_raw(path)

    assert calls["kwargs"]["format"] == "xport"
    assert loaded.shape == (5, 4)


def test_load_raw_missing_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/raw"):
        load_data.load_raw(tmp_path / "nope.csv")


def test_load_raw_rejects_unknown_extension(tmp_path):
    path = tmp_path / "data.xlsx"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported"):
        load_data.load_raw(path)


def test_profile_reports_shape_and_duplicates(tiny_df):
    report = load_data.profile(tiny_df)

    assert report["rows"] == 5
    assert report["columns"] == 4
    assert report["duplicate_rows"] == 1  # row 4 repeats row 0


def test_profile_counts_missing_values_only_where_present(tiny_df):
    report = load_data.profile(tiny_df)

    assert report["missing"] == {"B": 1}


def test_profile_flags_sentinel_code_candidates(tiny_df):
    report = load_data.profile(tiny_df)

    assert report["sentinel_candidates"]["A"] == [7]
    assert report["sentinel_candidates"]["B"] == [9]
    assert "C" not in report["sentinel_candidates"]


def test_profile_ignores_sentinel_values_in_high_cardinality_columns():
    df = pd.DataFrame({"X": list(range(100))})  # contains 7, 9, 77, 99 as real values

    assert load_data.profile(df)["sentinel_candidates"] == {}


def test_profile_sentinel_threshold_is_30_distinct_values():
    at_limit = pd.DataFrame({"X": list(range(30))})
    over_limit = pd.DataFrame({"X": list(range(31))})

    assert load_data.profile(at_limit)["sentinel_candidates"] == {"X": [7, 9]}
    assert load_data.profile(over_limit)["sentinel_candidates"] == {}


def test_profile_class_balance_is_share_per_value(tiny_df):
    report = load_data.profile(tiny_df, target="Y")

    assert report["class_balance"] == {1.0: 0.4, 2.0: 0.6}


def test_profile_without_target_has_no_class_balance(tiny_df):
    assert "class_balance" not in load_data.profile(tiny_df)


def test_profile_unknown_target_raises(tiny_df):
    with pytest.raises(KeyError, match="Z"):
        load_data.profile(tiny_df, target="Z")


def test_format_profile_is_readable_text(tiny_df):
    text = load_data.format_profile(load_data.profile(tiny_df, target="Y"))

    assert "rows: 5" in text
    assert "duplicate rows: 1" in text
    assert "class balance" in text

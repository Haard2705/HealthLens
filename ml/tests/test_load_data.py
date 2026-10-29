"""Tests for H06: loading and profiling the raw survey file.

These never touch the real survey data. Every test builds a tiny synthetic
file in a temporary folder, so they run in CI where data/raw/ is empty.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from load_data import (
    SENTINEL_CANDIDATES,
    find_raw_file,
    format_profile,
    load_raw,
    main,
    profile,
)


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """10 rows that look like BRFSS: a 0/1 target, a code column with 7/9, one duplicate."""
    return pd.DataFrame(
        {
            "Diabetes_binary": [0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            "BMI": [22.0, 31.5, 27.0, 40.2, None, 25.1, 22.0, 33.3, 29.0, 22.0],
            "GenHlth": [1, 3, 7, 4, 2, 9, 1, 5, 3, 1],
            "Sex": ["F", "M", "M", "F", "F", "M", "F", "M", "F", "F"],
        }
    ).assign(BMI=lambda d: d["BMI"].astype("float64"))


@pytest.fixture
def tiny_csv(tmp_path: Path, tiny_df: pd.DataFrame) -> Path:
    path = tmp_path / "tiny.csv"
    tiny_df.to_csv(path, index=False)
    return path


# ---- load_raw ---------------------------------------------------------------


def test_load_raw_reads_csv_with_correct_row_count(tiny_csv: Path) -> None:
    df = load_raw(tiny_csv)
    assert df.shape == (10, 4)


def test_load_raw_reads_parquet_when_engine_available(tmp_path: Path, tiny_df) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "tiny.parquet"
    tiny_df.to_parquet(path)
    assert load_raw(path).shape == (10, 4)


def test_load_raw_missing_file_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="data/raw"):
        load_raw(tmp_path / "nope.csv")


def test_load_raw_rejects_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "data.xlsx"
    path.write_text("x")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_raw(path)


def test_load_raw_xpt_uses_xport_reader(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "LLCP2024.XPT"
    path.write_bytes(b"fake")
    calls = {}

    def fake_read_sas(p, format):
        calls["format"] = format
        return pd.DataFrame({"a": [1]})

    monkeypatch.setattr(pd, "read_sas", fake_read_sas)
    assert load_raw(path).shape == (1, 1)
    assert calls["format"] == "xport"


def test_load_raw_accepts_cdc_trailing_space_in_xpt_name(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "LLCP2025.XPT "
    path.write_bytes(b"fake")
    monkeypatch.setattr(pd, "read_sas", lambda p, format: pd.DataFrame({"a": [1, 2]}))
    assert load_raw(path).shape == (2, 1)
    assert find_raw_file(tmp_path) == path


# ---- find_raw_file ----------------------------------------------------------


def test_find_raw_file_picks_the_only_data_file(tmp_path: Path) -> None:
    (tmp_path / ".gitkeep").write_text("")
    target = tmp_path / "brfss.csv"
    target.write_text("a\n1\n")
    assert find_raw_file(tmp_path) == target


def test_find_raw_file_errors_when_folder_empty(tmp_path: Path) -> None:
    (tmp_path / ".gitkeep").write_text("")
    with pytest.raises(FileNotFoundError, match="No data file"):
        find_raw_file(tmp_path)


def test_find_raw_file_errors_when_ambiguous(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("a\n1\n")
    (tmp_path / "b.csv").write_text("a\n1\n")
    with pytest.raises(ValueError, match="More than one"):
        find_raw_file(tmp_path)


# ---- profile ----------------------------------------------------------------


def test_profile_reports_shape_and_dtypes(tiny_df) -> None:
    p = profile(tiny_df)
    assert p["n_rows"] == 10
    assert p["n_cols"] == 4
    assert p["dtypes"]["BMI"] == "float64"


def test_profile_counts_missing_values(tiny_df) -> None:
    p = profile(tiny_df)
    assert p["missing"]["BMI"] == {"count": 1, "pct": 10.0}
    assert "Sex" not in p["missing"]  # only columns that actually have gaps


def test_profile_counts_duplicate_rows(tiny_df) -> None:
    # rows 0 and 6 are identical (0, 22.0, 1, "F"); row 9 too -> 2 extra copies
    p = profile(tiny_df)
    assert p["duplicates"] == {"count": 2, "pct": 20.0}


def test_profile_flags_sentinel_candidates(tiny_df) -> None:
    p = profile(tiny_df)
    assert p["sentinel_candidates"]["GenHlth"] == {"7": 1, "9": 1}
    # BMI has no exact code values, Sex isn't numeric
    assert "BMI" not in p["sentinel_candidates"]
    assert "Sex" not in p["sentinel_candidates"]


def test_profile_ignores_sentinels_in_continuous_columns() -> None:
    # A wide-ranging numeric column (e.g. a weight) naturally contains 7 or 99 by chance.
    df = pd.DataFrame({"weight": [7.0, 99.0, 1234.5, 5000.0] + list(range(100, 160))})
    assert "weight" not in profile(df)["sentinel_candidates"]


def test_sentinel_codes_cover_brfss_and_cchs_patterns() -> None:
    for code in (6, 7, 8, 9, 77, 88, 99, 777, 999, 9999):
        assert code in SENTINEL_CANDIDATES


def test_profile_class_balance_for_target(tiny_df) -> None:
    p = profile(tiny_df, target="Diabetes_binary")
    balance = p["class_balance"]
    assert balance["column"] == "Diabetes_binary"
    assert balance["counts"] == {"0": 7, "1": 3}
    assert balance["pct"] == {"0": 70.0, "1": 30.0}


def test_profile_class_balance_counts_missing_target() -> None:
    df = pd.DataFrame({"y": [1, 0, None, 0]})
    assert profile(df, target="y")["class_balance"]["counts"] == {"0.0": 2, "1.0": 1, "NaN": 1}


def test_profile_unknown_target_is_a_clear_error(tiny_df) -> None:
    with pytest.raises(KeyError, match="not_a_column"):
        profile(tiny_df, target="not_a_column")


def test_profile_without_target_has_no_class_balance(tiny_df) -> None:
    assert profile(tiny_df)["class_balance"] is None


def test_profile_is_json_serialisable(tiny_df) -> None:
    json.dumps(profile(tiny_df, target="Diabetes_binary"))


def test_profile_handles_empty_frame() -> None:
    p = profile(pd.DataFrame({"a": pd.Series([], dtype="float64")}))
    assert p["n_rows"] == 0
    assert p["duplicates"] == {"count": 0, "pct": 0.0}


# ---- report + CLI -----------------------------------------------------------


def test_format_profile_mentions_every_section(tiny_df) -> None:
    text = format_profile(profile(tiny_df, target="Diabetes_binary"))
    for heading in ("Shape", "Missing values", "Duplicate rows", "Sentinel", "Class balance"):
        assert heading in text


def test_cli_prints_profile_and_writes_json(tiny_csv: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "profile.json"
    code = main(["--path", str(tiny_csv), "--target", "Diabetes_binary", "--json", str(out)])
    assert code == 0
    assert "10 rows" in capsys.readouterr().out
    saved = json.loads(out.read_text())
    assert saved["n_rows"] == 10
    assert saved["source_file"] == "tiny.csv"


def test_cli_returns_error_code_when_no_data(tmp_path: Path, capsys) -> None:
    code = main(["--raw-dir", str(tmp_path)])
    assert code == 1
    assert "No data file" in capsys.readouterr().err


def test_cli_uses_env_override_for_raw_file(tiny_csv: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HEALTHLENS_RAW_FILE", str(tiny_csv))
    assert main([]) == 0
    assert "Profile of tiny.csv" in capsys.readouterr().out


def test_format_profile_truncates_long_sections() -> None:
    df = pd.DataFrame({f"c{i}": [1.0, None] for i in range(5)})
    text = format_profile(profile(df), max_rows=2)
    assert "... and 3 more" in text

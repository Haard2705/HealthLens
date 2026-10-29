"""H06: load the raw survey file and print a profile of it.

Usage (from the repo root):

    python ml/src/load_data.py                        # profiles the one file in data/raw/
    python ml/src/load_data.py --target DIABETE4      # also shows that column's class balance
    python ml/src/load_data.py --path data/raw/LLCP2024.XPT --json docs/figures/profile.json

The profile is measurement only. Nothing is cleaned or changed here: that is H08.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

import config

# Numeric values that surveys use to mean "missing for a reason":
# 6 valid skip (CCHS), 7 don't know, 8 refused / "none" (BRFSS 88), 9 not stated,
# and the same patterns on wider fields. These are only *candidates*: whether a
# value really is a sentinel is decided per variable from the codebook in H08.
SENTINEL_CANDIDATES = frozenset(
    {6, 7, 8, 9, 77, 88, 96, 97, 98, 99, 777, 888, 996, 997, 998, 999, 7777, 9999}
)

# A column with more distinct values than this is treated as continuous
# (BMI, weights), where 7 or 99 can appear by chance and mean nothing.
MAX_CODED_LEVELS = 50


def _suffix(path: Path) -> str:
    # CDC ships the BRFSS file as "LLCP20XX.XPT " (trailing space), so strip it.
    return path.suffix.lower().strip()


def load_raw(path: Path | str) -> pd.DataFrame:
    """Read a raw survey file (.csv, .xpt or .parquet) into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the survey file into data/raw/ "
            "(see healthlens-todo.md §5.1)."
        )
    suffix = _suffix(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".xpt":
        return pd.read_sas(path, format="xport")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported file type {suffix!r}; expected one of {', '.join(config.RAW_EXTENSIONS)}."
    )


def find_raw_file(raw_dir: Path | str = config.RAW_DIR) -> Path:
    """Return the single data file in raw_dir, or explain why there isn't exactly one."""
    raw_dir = Path(raw_dir)
    files = sorted(
        p for p in raw_dir.iterdir() if p.is_file() and _suffix(p) in config.RAW_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No data file in {raw_dir}. Download the BRFSS annual file (or the Kaggle "
            "quick-start CSV) there first."
        )
    if len(files) > 1:
        names = ", ".join(p.name for p in files)
        raise ValueError(f"More than one data file in {raw_dir} ({names}); pass --path.")
    return files[0]


def _pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 2) if whole else 0.0


def _key(value: Any) -> str:
    return "NaN" if pd.isna(value) else str(value)


def _sentinel_candidates(series: pd.Series) -> dict[str, int]:
    """Code values that sit above every ordinary value in a coded (categorical) column."""
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return {}
    values = series.dropna()
    if values.empty or values.nunique() > MAX_CODED_LEVELS:
        return {}
    if not (values == values.round()).all():
        return {}
    is_code = values.isin(SENTINEL_CANDIDATES)
    ordinary = values[~is_code]
    ceiling = ordinary.max() if not ordinary.empty else float("-inf")
    flagged = values[is_code & (values > ceiling)]
    counts = flagged.astype("int64").value_counts().sort_index()
    return {str(code): int(n) for code, n in counts.items()}


def profile(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    """Measure a raw DataFrame: shape, types, gaps, duplicates, sentinel codes, class balance."""
    n_rows = len(df)

    missing = {
        col: {"count": int(n), "pct": _pct(int(n), n_rows)}
        for col, n in df.isna().sum().items()
        if n > 0
    }

    n_dupes = int(df.duplicated().sum()) if n_rows else 0

    sentinels = {}
    for col in df.columns:
        found = _sentinel_candidates(df[col])
        if found:
            sentinels[col] = found

    class_balance = None
    if target is not None:
        if target not in df.columns:
            raise KeyError(f"Target column {target!r} is not in the file.")
        counts = df[target].value_counts(dropna=False)
        counts_by_key = {_key(v): int(n) for v, n in counts.items()}
        ordered = dict(sorted(counts_by_key.items()))
        class_balance = {
            "column": target,
            "counts": ordered,
            "pct": {k: _pct(n, n_rows) for k, n in ordered.items()},
        }

    return {
        "n_rows": n_rows,
        "n_cols": df.shape[1],
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": missing,
        "duplicates": {"count": n_dupes, "pct": _pct(n_dupes, n_rows)},
        "sentinel_candidates": sentinels,
        "class_balance": class_balance,
    }


def format_profile(p: dict[str, Any], max_rows: int = 40) -> str:
    """Turn a profile dict into a readable text report."""
    lines = [f"Shape: {p['n_rows']:,} rows x {p['n_cols']} columns", ""]

    dtype_counts = pd.Series(p["dtypes"]).value_counts()
    lines.append("Column types: " + ", ".join(f"{t} x{n}" for t, n in dtype_counts.items()))
    lines.append("")

    lines.append(f"Missing values ({len(p['missing'])} columns have gaps, worst first):")
    worst = sorted(p["missing"].items(), key=lambda kv: kv[1]["count"], reverse=True)
    for col, m in worst[:max_rows]:
        lines.append(f"  {col:<24} {m['count']:>10,}  ({m['pct']}%)")
    if len(worst) > max_rows:
        lines.append(f"  ... and {len(worst) - max_rows} more")
    lines.append("")

    d = p["duplicates"]
    lines.append(f"Duplicate rows: {d['count']:,} ({d['pct']}%)")
    lines.append("")

    lines.append("Sentinel-code candidates (check each against the codebook):")
    if not p["sentinel_candidates"]:
        lines.append("  none found")
    for col, codes in list(p["sentinel_candidates"].items())[:max_rows]:
        lines.append(f"  {col:<24} " + ", ".join(f"{c}: {n:,}" for c, n in codes.items()))
    if len(p["sentinel_candidates"]) > max_rows:
        lines.append(f"  ... and {len(p['sentinel_candidates']) - max_rows} more")
    lines.append("")

    cb = p["class_balance"]
    if cb is None:
        lines.append("Class balance: (no --target given)")
    else:
        lines.append(f"Class balance for {cb['column']}:")
        for k, n in cb["counts"].items():
            lines.append(f"  {k:<10} {n:>10,}  ({cb['pct'][k]}%)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile the raw survey file (H06).")
    parser.add_argument(
        "--path", type=Path, help="raw file to profile (default: the one file in data/raw/)"
    )
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR, help=argparse.SUPPRESS)
    parser.add_argument("--target", help="outcome column to show class balance for")
    parser.add_argument("--json", type=Path, help="also save the profile as JSON here")
    args = parser.parse_args(argv)

    try:
        path = args.path or config.raw_file_override() or find_raw_file(args.raw_dir)
        df = load_raw(path)
        result = profile(df, target=args.target)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    result["source_file"] = Path(path).name
    print(f"Profile of {Path(path).name}\n")
    print(format_profile(result))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON profile to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

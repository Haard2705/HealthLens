"""Load a raw survey file and print a profile of it.

Run from the repo root:  python -m ml.src.load_data --target DIABETE4
"""

import argparse
from pathlib import Path

import pandas as pd

from . import config

# Survey "don't know / refused / not stated" codes. In a column with few distinct values these
# are probably sentinels, not real answers; in a continuous column the same numbers are real.
SENTINEL_CODES = (7, 9, 77, 99, 777, 999, 7777, 9999)
MAX_CATEGORICAL_VALUES = 30


def load_raw(path: Path) -> pd.DataFrame:
    """Read a CSV or SAS transport (.xpt) file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put the raw file in data/raw/ (see README).")
    suffix = path.suffix.lower()
    if suffix == ".xpt":
        return pd.read_sas(path, format="xport", encoding="latin-1")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type {suffix!r}; expected .xpt or .csv")


def profile(df: pd.DataFrame, target: str | None = None) -> dict:
    """Summarise size, missing values, duplicates, sentinel candidates and class balance."""
    report: dict = {
        "rows": len(df),
        "columns": df.shape[1],
        "duplicate_rows": int(df.duplicated().sum()),
    }

    missing = df.isna().sum()
    report["missing"] = {col: int(n) for col, n in missing.items() if n > 0}

    sentinels: dict[str, list[int]] = {}
    for col in df.select_dtypes("number").columns:
        values = df[col].dropna().unique()
        if len(values) > MAX_CATEGORICAL_VALUES:
            continue
        found = [code for code in SENTINEL_CODES if code in values]
        if found:
            sentinels[col] = found
    report["sentinel_candidates"] = sentinels

    if target is not None:
        if target not in df.columns:
            raise KeyError(f"target column {target!r} not in data")
        report["class_balance"] = df[target].value_counts(normalize=True, dropna=False).to_dict()

    return report


def format_profile(report: dict) -> str:
    lines = [
        f"rows: {report['rows']}",
        f"columns: {report['columns']}",
        f"duplicate rows: {report['duplicate_rows']}",
        f"columns with missing values: {len(report['missing'])}",
        f"columns with sentinel-code candidates: {len(report['sentinel_candidates'])}",
    ]
    if "class_balance" in report:
        lines.append("class balance:")
        lines += [f"  {value}: {share:.2%}" for value, share in report["class_balance"].items()]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=config.RAW_BRFSS_FILE)
    parser.add_argument("--target", default=None, help="outcome column for class balance")
    args = parser.parse_args()

    df = load_raw(args.path)
    print(format_profile(profile(df, args.target)))


if __name__ == "__main__":
    main()

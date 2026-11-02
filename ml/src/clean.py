"""Clean the canonical frame and save it for modelling.

Run from the repo root:  python -m ml.src.clean

Decisions (H08): no deduplication (the raw file has no duplicate rows, and identical answers
from two respondents are two real people); missing answers stay missing, because filling them
in is done inside the model pipeline on training data only, to avoid leakage.
"""

import argparse
from pathlib import Path

import pandas as pd

from . import config, load_data, schema
from .sources import brfss

DEFAULT_OUT = config.DATA_PROCESSED / "brfss_clean.parquet"


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (cleaned frame, report of what changed). The input is not modified."""
    report: dict = {"rows_in": len(df)}
    out = df.copy()

    low, high = schema.BMI_RANGE
    bad_bmi = out["bmi"].notna() & ~out["bmi"].between(low, high)
    out.loc[bad_bmi, "bmi"] = float("nan")
    report["bmi_set_missing"] = int(bad_bmi.sum())

    usable_weight = out["survey_weight"] > 0  # NaN compares False, so missing weights drop too
    report["rows_dropped_invalid_weight"] = int((~usable_weight).sum())
    out = out[usable_weight].reset_index(drop=True)

    for column, allowed in schema.CATEGORIES.items():
        out[column] = pd.Categorical(out[column], categories=sorted(allowed))

    report["rows_out"] = len(out)
    report["missing_by_column"] = {c: int(n) for c, n in out.isna().sum().items() if n > 0}
    schema.validate_schema(out)
    return out, report


def format_report(report: dict) -> str:
    lines = [
        f"rows in: {report['rows_in']}",
        f"rows out: {report['rows_out']}",
        f"rows dropped (no positive weight): {report['rows_dropped_invalid_weight']}",
        f"BMI values set to missing (outside {schema.BMI_RANGE[0]}-{schema.BMI_RANGE[1]}): "
        f"{report['bmi_set_missing']}",
        "missing values by column:",
    ]
    lines += [f"  {col}: {n}" for col, n in report["missing_by_column"].items()]
    return "\n".join(lines)


def save(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=Path, default=config.RAW_BRFSS_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    canonical = brfss.to_canonical(load_data.load_raw(args.path))
    cleaned, report = clean(canonical)
    save(cleaned, args.out)
    print(format_report(report))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

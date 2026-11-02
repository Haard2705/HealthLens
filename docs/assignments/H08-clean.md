# H08 — Cleaning & Recoding

## What we built (plain English)

`python -m ml.src.clean` reads the raw CDC file, translates it with the H07 adapter, cleans it
and saves `data/processed/brfss_clean.parquet` (4.4 MB, gitignored). It also prints a short
report of what it changed. Running it twice gives byte-identical files.

## Key decisions

- **No deduplication.** The raw file has 0 duplicate rows (H06), and two respondents with the
  same answers are two real people, each with their own survey weight.
- **Missing stays missing.** Don't-know / refused / not-asked answers were already turned into
  blanks by the adapter (never "no"). Filling them in happens later inside the model pipeline,
  on training data only, to avoid leakage. A blank cholesterol answer can mean "never had it
  checked" (a skip), but our boolean column cannot hold a third state, so it stays missing.
- **Sentinel codes are handled per variable in the adapter**, using each question's own code
  table, not a global "replace 9 with missing" (9 is a real value in some columns).
- **Rows are dropped only if the weight is missing or not positive.** Nothing else is dropped
  here; a respondent with a missing outcome is dropped later, per condition, at training time.
- **`clean` returns `(frame, report)`** instead of just a frame, so the log is testable.

## How it works

`ml/src/clean.py`:
- `clean(df)`: BMI outside 10–100 → missing; drop rows without a positive weight; give `sex`
  and `smoker` fixed category levels; validate the result against the schema.
- `format_report`, `save` (Parquet), `main` (raw file → adapter → clean → save).
- Tests: `ml/tests/test_clean.py`.

## How to run / verify it

```bash
python -m pytest
python -m ml.src.clean
```

## Results (real 2025 file, 2026-11-02)

| | |
|---|---|
| Rows in / out | 356,158 / 356,158 |
| Rows dropped for a bad weight | 0 |
| BMI values blanked (outside 10–100) | 0 |
| Rerun gives identical file | yes (same SHA-256) |

The file was already clean: CDC removes impossible values before release, so H08 changes
nothing on this data. It matters for CCHS and for protecting the pipeline from bad input.

Missing values per column (of 356,158): income_band 71,050 · fruit_daily 46,441 ·
veg_daily 53,527 · high_chol 44,832 · bmi 32,562 · heavy_alcohol 28,786 · smoker 20,653 ·
age_band 6,682 · all nine outcomes 782–4,248.

## Gotchas / follow-ups

- `income_band` is missing for about 20% of respondents; H09 and the model pipeline must
  decide how to treat that (imputation or a "not stated" category), inside the pipeline.
- Outcome missingness differs by condition, so each model trains on its own subset (H10).

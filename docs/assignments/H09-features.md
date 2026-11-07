# H09 — Feature Engineering

## What we built (plain English)

`ml/src/features.py` adds three derived columns to the cleaned data, picks the columns each
condition's model is allowed to use, and builds the shared *preprocessor*: the recipe that
fills gaps, scales numbers and encodes text before any model sees the data. Every model
(logistic regression, XGBoost, …) uses this same recipe.

## Key decisions

- **Two kinds of work, kept apart.** Row-by-row features (`add_features`) are safe to compute
  before splitting the data. Anything that *learns* from data (fill values, scaling,
  one-hot categories) is a scikit-learn step inside a `Pipeline`, so it is fit on training rows
  only and the test set cannot leak in.
- **Derived features** (never built from an outcome column; a test proves it):
  - `bmi_band`: underweight < 18.5, normal 18.5–25, overweight 25–30, obese ≥ 30.
  - `risk_factor_count` (0–5): current smoker (daily or some days), physically inactive,
    heavy drinker, high cholesterol, obese.
  - `lifestyle_score` (0–5): physically active, fruit daily, vegetables daily, not a heavy
    drinker, not a current smoker.
  - Both counts use the answers that are known and are missing only when none are known.
- **Gaps are kept, not hidden.** The preprocessor adds a "was missing" flag for numbers and
  yes/no answers, because income is missing for about 20% of respondents.
- **Leakage guard:** `predictors_for(condition)` uses the condition registry (H49), so a
  model never gets any other condition's outcome as a predictor unless it opts in.
- Survey weight is not a predictor; it is kept for weighted statistics and (optionally)
  sample weights.

## How it works

- `add_features(df)` → copy with the three derived columns.
- `model_frame(df, columns)` → selected columns as plain floats/strings (scikit-learn cannot
  take pandas' `pd.NA`).
- `build_preprocessor(predictors)` → unfitted `ColumnTransformer`: numbers (median fill,
  flag, scale), yes/no (most-frequent fill, flag), text (most-frequent fill, one-hot, unseen
  categories ignored).
- Tests: `ml/tests/test_features.py`.

## How to use it (for the model code)

```python
df = features.add_features(pd.read_parquet("data/processed/brfss_clean.parquet"))
predictors = features.predictors_for("diabetes")
x = features.model_frame(df, predictors)
model = Pipeline([("prep", features.build_preprocessor(predictors)), ("clf", ...)])
```

Drop rows whose outcome is missing before splitting, per condition (H10).

## How to run / verify it

```bash
python -m pytest
```

## Results (real 2025 file, 2026-11-02)

- `bmi_band` counts equal the CDC's own `_BMI5CAT` counts: 5,552 / 96,272 / 114,838 /
  106,934, missing 32,562.
- The preprocessor fits on all rows and produces 34 numeric columns for 15 predictors.
- A throwaway logistic regression (smoke test only, not a reported result) gave held-out
  ROC-AUC of about 0.79 (diabetes), 0.83 (heart disease), 0.79 (stroke), in line with what
  lifestyle-survey models normally reach. Real model results belong to H10–H13.

## Gotchas / follow-ups

- `risk_factor_count` and `lifestyle_score` overlap on smoking, activity and alcohol by
  design; a linear model may find them redundant. Rudra can drop either.
- Fill values (median / most frequent) are fit inside the pipeline; do not impute earlier.
- Race and ethnicity are not features (ADR-0003).

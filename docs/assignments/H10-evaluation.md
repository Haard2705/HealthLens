# H10 — Evaluation Harness & Split

## What we built (plain English)

`ml/src/evaluate.py` is the referee for every model. It does three things the same way for
every condition and every model:

1. **Splits the data** into 80% for training and 20% held back for the final test.
2. **Carves a validation set** out of the training part, used for picking a cut-off (and,
   in H12, for XGBoost's early stopping). The test set is never used to make choices.
3. **Scores a trained model** with one function that returns the same set of numbers
   (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, confusion matrix).

## Key decisions

- **One split per condition; missing outcomes are dropped first.** Someone who skipped the
  asthma question is left out of the asthma model, never counted as "no" (H08 rule).
- **Stratified**: train and test have the same share of "yes" answers.
- **The same people are in the test set across conditions.** Every respondent gets one random
  number (seed 42) for the whole file; within each class the lowest 20% go to test. On
  synthetic data over 80% of test rows are shared between two conditions, versus about 20%
  for independent random splits. This makes cross-condition comparisons (and SHAP in H14)
  fairer.
- **No deduplication**: the 2025 file has no duplicate rows (ADR-0003, H06 profile).
- **Two operating points, both reported** (`evaluate_model`):
  - `default`: cut-off 0.5.
  - `recall_oriented`: the highest cut-off that still catches **≥ 80% of positives on the
    validation set**, then applied unchanged to the test set. A screening tool should rather
    over-flag than miss people; 80% is a starting value (`TARGET_RECALL`) the team can revise.
- **Unweighted metrics by default** (how the model does on the people in the test set).
  `sample_weight=` gives survey-weighted metrics that estimate US-population performance;
  H15 can use them.
- **Brier "no skill" baseline** (`prevalence × (1 − prevalence)`) is reported next to the
  Brier score, so H50's release gate can check "calibration better than no-skill" directly.
- Threshold is **inclusive**: a score ≥ the cut-off counts as "yes".

## How it works

| Function | Purpose |
|---|---|
| `split_condition(df, condition_id)` → `Split` | Drops missing outcomes, adds H09 features if needed, selects the registry's allowed predictors (`features.predictors_for`), returns `x_/y_/w_` for train and test |
| `validation_split(split)` → `(fit, val)` | Stratified 80/20 of the training rows |
| `evaluate(model, x_test, y_test, threshold=0.5, sample_weight=None)` → dict | All metrics for any model with `predict_proba` |
| `choose_threshold(y_val, scores, target_recall=0.8)` | Highest cut-off reaching the recall goal |
| `evaluate_model(model, x_val, y_val, x_test, y_test)` | `{default, recall_oriented, target_recall, threshold_chosen_on}`, the shape `metrics.json` (H13) will store |
| `python -m ml.src.evaluate [--condition ID]` | Prints split sizes and prevalence per condition |

How the model code (H11/H12) uses it:

```python
split = evaluate.split_condition(df, "diabetes")
fit, val = evaluate.validation_split(split)
model = Pipeline([("prep", features.build_preprocessor(split.predictors)), ("clf", ...)])
model.fit(fit.x, fit.y)
metrics = evaluate.evaluate_model(model, val.x, val.y, split.x_test, split.y_test)
```

## How to run / verify it

```bash
python -m pytest ml/tests/test_evaluate.py
python -m ml.src.evaluate                 # needs data/processed/brfss_clean.parquet (H08)
```

Tests (31) cover: metrics against a hand-computed 4-row example (ROC-AUC 0.75, PR-AUC 0.833,
Brier 0.158), threshold choice, weighting, valid ranges, JSON-safety, missing-outcome rows
dropped, 80/20 stratification, no train/test overlap, reproducibility, cross-condition
overlap, leakage guard (no outcome or weight in the predictors), and the CLI.

## Results

Synthetic sanity check (20,000 fake rows where risk rises with BMI and age; plain logistic
regression; not a reported result):

| Cut-off | Threshold | Recall | Precision | ROC-AUC | Brier (no-skill) |
|---|---|---|---|---|---|
| default | 0.50 | 0.056 | 0.551 | 0.696 | 0.142 (0.156) |
| recall-oriented | 0.138 | 0.796 | 0.255 | 0.696 | 0.142 (0.156) |

With ~19% prevalence, the 0.5 cut-off catches almost nobody; the validation-chosen cut-off
meets the 80% goal on unseen test data. Expect the same pattern on BRFSS.

Real split sizes: run `python -m ml.src.evaluate` on the processed 2025 file and paste the
output here.

## Gotchas / follow-ups

- ROC-AUC, PR-AUC and Brier don't depend on the cut-off; only the confusion-matrix metrics do.
- `evaluate` refuses a test set with only one class (ROC-AUC is undefined there).
- Building on the H09 branch (PR #11); merge that first.
- H49's "`--condition` for evaluate.py" item is done here; `train.py` and `explain.py` follow in
  H13/H14.

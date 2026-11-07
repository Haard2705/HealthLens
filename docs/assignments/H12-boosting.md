# H12 — XGBoost

## What we built (plain English)

`ml/src/boosting.py` trains XGBoost for each condition. XGBoost builds hundreds of small
decision trees, each fixing the mistakes of the ones before. Unlike logistic regression it
can learn patterns like "risk is high when one factor is high but not both". It goes
through exactly the same H10 harness as the H11 baseline (same split, same validation
rows, same scoring), so the comparison is fair. `--compare` prints both side by side.

Steps per condition:

1. H10 split → train / test (test untouched until the end).
2. Training rows → *fit* part (80%) and *validation* part (20%), the same rows as H11.
3. **Randomized search**: 12 random combinations of six settings (tree depth, learning
   rate, min child weight, row/column sampling, L2 penalty), scored by 3-fold ROC-AUC on the
   fit part. If the fit part has more than 100,000 rows, the search uses a stratified
   100,000-row sample to stay quick.
4. **Early stopping**: refit the winning settings on the whole fit part, adding trees (up to
   2,000) until validation log-loss hasn't improved for 50 rounds.
5. Recall-oriented threshold chosen on validation; final metrics on test.

## Key decisions

- **`scale_pos_weight = 1` (deviation from the roadmap), same reasoning as H11.** Weighting
  the rare class inflates every predicted probability; the recall-oriented threshold handles
  imbalance instead. A CI test asserts the trained model's Brier score beats the no-skill
  baseline.
- **Modest tuning, on purpose** ("don't burn a week on 0.002 AUC"): 12 candidates × 3 folds
  with 300 trees each, then one early-stopped fit.
- **No early stopping inside cross-validation**: each fold would need its own validation set.
  Early stopping happens once, in the final fit.
- **The validation set does two jobs** (early stopping and choosing the threshold). It is
  never the test set, so the test metrics stay honest; the threshold may be very slightly
  optimistic on validation itself, which we accept.
- The preprocessor is fit on the fit part only and then applied to the validation rows, so
  early stopping doesn't leak validation information into fill values or scaling.
- **LightGBM (optional in the roadmap) is not included.** It's a stretch item; the harness
  makes adding it later a one-file change.

## How it works

| Function | Purpose |
|---|---|
| `build_xgb(predictors, **params)` | H09 preprocessor → `XGBClassifier` (hist trees, seed 42) |
| `tune_xgb(x, y, predictors)` | `RandomizedSearchCV` over `PARAM_SPACE`; returns best settings + CV ROC-AUC |
| `fit_with_early_stopping(predictors, params, fit, val)` | final fit; returns pipeline + number of trees kept |
| `train_xgb(split)` → `BoostResult` | steps 2–5; `.summary()` has the same shape as H11's for `metrics.json` (H13) |
| `compare([summaries])` | side-by-side table (ROC-AUC, PR-AUC, Brier, recall, precision, threshold) |
| `python -m ml.src.boosting [--condition ID] [--compare]` | train and print |

## How to run / verify it

```bash
python -m pytest ml/tests/test_boosting.py
python -m ml.src.boosting --condition diabetes --compare   # needs the processed BRFSS file
```

14 tests: pipeline shape, search space, sampling, early stopping, same rows as the baseline,
learns a planted signal and stays calibrated, beats the linear model on an interaction,
JSON-safe summary in the baseline's shape, reproducibility, comparison table, CLI. The tests
skip themselves where `xgboost` isn't installed; CI installs it from `ml/requirements.txt`.

## Results

Synthetic checks (20,000 rows; not reported results). These were run with a scikit-learn
gradient-boosting stand-in because the development sandbox couldn't install `xgboost`; CI
runs the same tests with the real library.

| Planted pattern | Model | ROC-AUC | PR-AUC | Brier | Recall @ chosen cut-off |
|---|---|---|---|---|---|
| Risk rises with BMI + age (additive) | logreg | 0.706 | 0.367 | 0.145 | 0.82 |
| | xgboost | 0.704 | 0.367 | 0.145 | 0.82 |
| Risk high when exactly one of BMI / age is high | logreg | 0.530 | 0.266 | 0.197 | 0.85 |
| | xgboost | **0.785** | **0.497** | **0.148** | 0.80 |

Takeaway for the report: when risk factors simply add up, the linear baseline is as good
as XGBoost (and easier to explain); XGBoost only wins when there are real interactions.
Whether BRFSS has enough of those is what the real comparison will show.

Real BRFSS results: run `python -m ml.src.boosting --compare` on the processed 2025 file
and paste the tables (all nine conditions) here.

## Gotchas / follow-ups

- Training all nine conditions on the full file takes a while (minutes per condition);
  `--n-iter 4` is a quick check.
- Builds on H11 (PR #13) → H10 (PR #12) → H09 (PR #11); merge in that order.
- H13 (`train.py`) will call `train_logreg` and `train_xgb`, save both models, and pick the
  best by validation ROC-AUC.

# H11 — Logistic Regression Baseline

## What we built (plain English)

`ml/src/logreg.py` trains the simplest reasonable model for each condition: logistic
regression, which gives every factor (BMI, age, smoking, …) one weight. It's the "floor"
XGBoost (H12) has to beat, and its weights become **odds ratios** for the report ("each
step up in age band multiplies the odds by 1.7").

It uses the H10 harness exactly, so the comparison with XGBoost is fair:

1. H10 split → train / test (test untouched until the end).
2. Training rows → a *fit* part (80%) and a *validation* part (20%).
3. 5-fold cross-validation on the fit part picks `C` from `{0.01, 0.1, 1, 10}` by ROC-AUC.
4. The recall-oriented threshold (≥ 80% recall) is chosen on the validation part.
5. Final metrics on the test set, at 0.5 and at that threshold.

## Key decisions

- **No class re-weighting (deviation from the roadmap's `class_weight="balanced"`).**
  On 20,000 synthetic rows (prevalence ~19%) we compared both:

  | Setting | ROC-AUC | Recall @ chosen cut-off | Precision | Brier (no-skill 0.158) |
  |---|---|---|---|---|
  | balanced | 0.705 | 0.828 | 0.277 | **0.219, worse than guessing** |
  | none (chosen) | 0.706 | 0.820 | 0.279 | **0.145, better than guessing** |

  Ranking and screening performance are the same, but "balanced" inflates every predicted
  probability (a real 15% risk reads as ~40%). That would mislead users wherever we show
  "your risk is X%", and would fail H50's "calibration better than no-skill" gate. The rare
  "yes" class is handled by H10's recall-oriented threshold instead. `build_logreg(...,
  class_weight="balanced")` still exists for comparison. **Team: please confirm.**
- **Small grid, on purpose.** Four values of `C`, ROC-AUC scoring, stratified folds, seed 42.
- **Survey weights are not used in training** (they describe the population, not the
  model's job). H15 can report weighted metrics through H10's `sample_weight`.
- **The final model is the one fit on the fit part** (not refit on fit + validation), so the
  threshold chosen on validation belongs to exactly the model being tested. H13 keeps this.

## How it works

| Function | Purpose |
|---|---|
| `build_logreg(predictors, C, class_weight=None)` | H09 preprocessor → `LogisticRegression` |
| `tune_logreg(x, y, predictors, cv=5)` | `GridSearchCV` on `C`; returns model, best `C`, every CV score |
| `train_logreg(split)` → `LogregResult` | steps 2–5 above; `.summary()` is the JSON block H13 writes to `metrics.json` |
| `odds_ratios(model)` | table of `feature`, `coefficient`, `odds_ratio`, strongest first, readable names |
| `python -m ml.src.logreg [--condition ID]` | prints metrics + top odds ratios, saves `docs/figures/<condition>/logreg_odds_ratios.csv` |

Reading the odds ratios: numeric factors are scaled first, so their odds ratio is **per one
standard deviation**; yes/no answers are "yes" vs "no"; `x (was missing)` is the effect of
not answering; one-hot categories keep every level, so compare levels with each other.

## How to run / verify it

```bash
python -m pytest ml/tests/test_logreg.py
python -m ml.src.logreg --condition diabetes   # needs data/processed/brfss_clean.parquet
```

18 tests: pipeline shape, grid, reproducibility, no fitting on validation/test rows, learns
a planted signal, calibrated probabilities, the "balanced" inflation, odds-ratio maths and
names, finding the true risk factors, CLI.

## Results

Synthetic check (20,000 rows; only BMI and age truly matter; not a reported result):
the two strongest odds ratios were **age_band 1.72** and **bmi 1.63**; everything else was
within 0.88–1.09. ROC-AUC 0.706; recall 0.82 at the validation-chosen cut-off (0.15).

Real BRFSS results: run `python -m ml.src.logreg` on the processed 2025 file and paste the
output (all nine conditions) here.

## Gotchas / follow-ups

- `bmi`, `bmi_band` and `risk_factor_count` overlap, so their weights share the BMI effect;
  read the BMI story from all three together (or drop `bmi_band` for the linear model).
- Branch builds on H10 (PR #12) and H09 (PR #11); merge those first.

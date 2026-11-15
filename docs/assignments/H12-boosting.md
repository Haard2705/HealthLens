# H12 — Random forest and XGBoost

## What we built (plain English)

Two non-linear models, so the report can put **three** models side by side for every
condition (FR3), all through the same H10 harness as the H11 baseline:

- `ml/src/forest.py` — a **random forest**. It grows a few hundred independent decision
  trees; each tree sees a bootstrap sample of the rows and a random subset of the factors at
  every split, and the forest averages their votes. That averaging is what makes it stable:
  unlike a single deep tree it doesn't memorise the training rows.
- `ml/src/boosting.py` — **XGBoost**. It builds hundreds of small trees in sequence, each one
  correcting the mistakes of the ones before. Usually the strongest of the three, and the
  hardest to explain (H14 adds SHAP for that).

Both can learn patterns a linear model can't, such as "risk is high when one factor is high
but not both". Both use the same split, the same validation rows and the same scoring as the
baseline, so the comparison is fair. `--compare` on either CLI prints it against the baseline;
`python -m ml.src.train` (H13) trains all three and writes the comparison into `metrics.json`.

Steps per condition, identical for both models except where noted:

1. H10 split → train / test (test untouched until the end).
2. Training rows → *fit* part (80%) and *validation* part (20%), the same rows as H11.
3. **Randomized search** on the fit part, scored by k-fold ROC-AUC, on a stratified sample of
   it when it is large:
   - forest: 6 candidates × 3 folds over tree depth, leaf size and features-per-split, with
     120-tree search forests, on at most 40,000 rows.
   - XGBoost: 12 candidates × 3 folds over six settings (depth, learning rate, min child
     weight, row/column sampling, L2 penalty) with 300 trees, on at most 100,000 rows.
4. **Final fit** on the whole fit part: the forest refits the winning settings with 300 trees;
   XGBoost refits with **early stopping**, adding trees (up to 2,000) until validation
   log-loss hasn't improved for 50 rounds.
5. Recall-oriented threshold chosen on validation; final metrics on test.

## Key decisions

- **No class weighting in either model** (deviation from the roadmap), same reasoning as H11.
  Weighting the rare class leaves the ranking (ROC-AUC) alone but inflates every predicted
  probability, so a 15% risk can read as 40% and the Brier score falls below the no-skill
  baseline. The recall-oriented threshold from H10 handles the imbalance instead. A CI test
  per model asserts the trained model's Brier score still beats no-skill.
- **The forest's depth and leaf size are capped** (`max_depth` 6/10/14, `min_samples_leaf`
  5/20/50). Fully grown trees on ~285,000 fit rows are slow, enormous in memory, and no more
  accurate once leaves are this size. `n_jobs=-1` keeps the fit parallel.
- **Modest tuning, on purpose** ("don't burn a week on 0.002 AUC"). The forest searches a
  smaller space than XGBoost because it has fewer knobs that matter: depth, leaf size and
  how many factors each split may look at.
- **No early stopping inside cross-validation** (XGBoost): each fold would need its own
  validation set. Early stopping happens once, in the final fit. The forest needs none —
  more trees never make it worse, they just cost time.
- **The validation set does two jobs** (early stopping and choosing the threshold). It is
  never the test set, so the test metrics stay honest; the threshold may be very slightly
  optimistic on validation itself, which we accept.
- The preprocessor is fit on the fit part only and then applied to the validation rows, so
  neither model leaks validation information into fill values or scaling.
- **Explanations (H14):** the forest averages probabilities rather than log-odds, so its SHAP
  contributions are in probability units and the explanation payload says so. Logistic
  regression and XGBoost stay in log-odds.
- **LightGBM (optional in the roadmap) is not included.** It's a stretch item; with two tree
  models in place the harness makes adding a third a one-file change.

## How it works

| Function | Purpose |
|---|---|
| `forest.build_forest(predictors, **params)` | H09 preprocessor → `RandomForestClassifier` (seed 42, `n_jobs=-1`) |
| `forest.tune_forest(x, y, predictors)` | `RandomizedSearchCV` over `PARAM_SPACE`; returns best settings + CV ROC-AUC |
| `forest.fit_forest(predictors, params, fit)` | final 300-tree fit on the fit part only |
| `forest.train_forest(split)` → `ForestResult` | steps 2–5; `.summary()` has the H11 shape for `metrics.json` (H13) |
| `boosting.build_xgb(predictors, **params)` | H09 preprocessor → `XGBClassifier` (hist trees, seed 42) |
| `boosting.tune_xgb(x, y, predictors)` | `RandomizedSearchCV` over `PARAM_SPACE`; returns best settings + CV ROC-AUC |
| `boosting.fit_with_early_stopping(predictors, params, fit, val)` | final fit; returns pipeline + number of trees kept |
| `boosting.train_xgb(split)` → `BoostResult` | steps 2–5; same `.summary()` shape |
| `boosting.compare([summaries])` | side-by-side table (ROC-AUC, PR-AUC, Brier, recall, precision, threshold), used by both CLIs |
| `python -m ml.src.forest [--condition ID] [--compare]` | train the forest and print |
| `python -m ml.src.boosting [--condition ID] [--compare]` | train XGBoost and print |

H13's `MODEL_ORDER` is `("logreg", "random_forest", "xgboost")` — simplest first, because on a
near-tie (validation ROC-AUC within 0.005) `select_best` keeps the simpler model.

## How to run / verify it

```bash
python -m pytest ml/tests/test_forest.py ml/tests/test_boosting.py
python -m ml.src.forest --condition diabetes --compare     # needs the processed BRFSS file
python -m ml.src.train                                     # all three models, all conditions
```

16 tests for the forest and 14 for XGBoost: pipeline shape, no class weighting, search space,
sampling, tree count / early stopping, same rows as the baseline, learns a planted signal and
stays calibrated, beats the linear model on a planted interaction, JSON-safe summary with the
baseline's metric keys, reproducibility, reload-and-predict, comparison table, CLI. The
XGBoost tests skip themselves where `xgboost` isn't installed; the forest's need only
scikit-learn. H13's smoke test trains nine conditions × three models on 300 synthetic rows in
about 29 seconds.

## Results

Synthetic checks (4,000 rows; not reported results): where risk simply rises with BMI and age,
all three models land within a few thousandths of each other; where risk is high when exactly
one of BMI / age is high, both tree models beat the linear baseline by more than 0.10 ROC-AUC.

### Real BRFSS 2025 results — all three models, all nine conditions

Test-set ROC-AUC (356,000 rows; the chosen model is the one `train.py` saved as best, by
**validation** ROC-AUC with the 0.005 tie margin). "Estimates" is the H50 release-gate
decision for individual estimates.

<!-- RESULTS TABLE -->

## Gotchas / follow-ups

- Training all nine conditions on the full file takes tens of minutes; `--fast` (or
  `--n-iter 4` on the single-model CLIs) is the quick check.
- The forest is the slowest model to *save*: a 300-tree forest is a far bigger joblib file
  than the baseline or XGBoost. `ml/models/` is gitignored, so this only costs local disk.
- Builds on H11 → H10 → H09.
- H13 (`train.py`) calls all three trainers, saves all three models, and picks the best by
  validation ROC-AUC.

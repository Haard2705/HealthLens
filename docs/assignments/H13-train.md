# H13 — Training Entry Point, Artifacts & `metrics.json`

## What we built (plain English)

One command trains everything:

```bash
python -m ml.src.train                    # all nine conditions, both models
python -m ml.src.train --condition diabetes
python -m ml.src.train --fast             # small search, for a quick check
```

For each condition in the registry (H49) it makes the H10 split, trains the logistic
regression baseline (H11) and XGBoost (H12) on exactly the same rows, picks the better one
**on the validation set**, and saves the files the backend (H19 `/predict`) loads.

## Key decisions

- **The best model is chosen on validation ROC-AUC, never on test.** Choosing on the test set
  would make the reported test score optimistic. If the two are within **0.005**, the simpler
  model (logistic regression) wins, because it is easier to explain.
- **Both models are saved**, not only the winner, so the report can compare them and the
  backend could switch later without retraining.
- **`index.json` is merged, not overwritten.** Retraining one condition updates only its
  entry. A condition that fails (for example, only one answer class) is recorded as
  `"status": "failed"` with the reason; the others still train, and the command exits with
  code 1 so CI or a teammate notices.
- **Strict JSON** (`allow_nan=False`): a NaN in the metrics fails at training time instead
  of breaking the backend.
- **Run as a module** (`python -m ml.src.train`), like the rest of the pipeline; the roadmap
  wrote `python ml/src/train.py`.
- `--models logreg` trains only the baseline (useful where `xgboost` isn't installed).

## Files written (the contract with the backend)

```
ml/models/                      (gitignored: regenerate with the command above)
├── index.json                  every condition: status, best_model, test ROC-AUC/Brier, tier, sensitive
└── <condition>/
    ├── logreg.joblib           fitted scikit-learn Pipeline (preprocessor + model)
    ├── xgboost.joblib
    ├── metrics.json            best_model, selection criterion + validation ROC-AUCs,
    │                           and for each model: params, CV score, row counts,
    │                           test metrics at 0.5 ("default") and at the recall-oriented
    │                           threshold ("recall_oriented", chosen on validation)
    └── model_meta.json         condition, outcome column, sensitive flag, data source/file,
                                row counts, features (canonical order), feature_spec,
                                seed, library versions, git SHA, trained_at
```

`feature_spec` tells the backend what each input is, in the order the model expects:

```json
{"name": "sex", "kind": "category", "allowed": ["female", "male"], "derived": false}
{"name": "bmi", "kind": "number", "range": [10.0, 100.0], "derived": false}
{"name": "phys_active", "kind": "yes_no", "derived": false}
{"name": "bmi_band", "kind": "category", "allowed": ["underweight", ...], "derived": true}
```

`derived: true` columns (`bmi_band`, `risk_factor_count`, `lifestyle_score`) are **not** asked
from the user: the backend builds a one-row frame of the canonical inputs, then calls
`features.add_features`, then `features.model_frame(df, meta["features"])`, then
`model.predict_proba(x)[:, 1]`. Missing answers can be sent as missing; the pipeline fills
them the same way it did in training.

## How to run / verify it

```bash
python -m pytest ml/tests/test_train.py
python -m ml.src.train --fast             # needs data/processed/brfss_clean.parquet
```

15 tests: model choice (best wins, near-tie → simpler), every file written, saved model
reloads and predicts, metrics/meta contents, feature spec, strict JSON, index merging,
failure handling, CLI, and the roadmap's **smoke test: every condition × both models on 300
synthetic rows**, checking all files and valid JSON (runs in CI with real XGBoost).

## Results

Synthetic run (3,000 rows, `--fast`, scikit-learn stand-in for XGBoost; not reported
results): all nine conditions trained; logistic regression chosen for six, XGBoost for three;
every model's Brier score beat the no-skill baseline. Smoke test (300 rows): 36 s with the
stand-in; see CI for the time with real XGBoost (target < 30 s).

Real BRFSS: run `python -m ml.src.train` on the processed 2025 file and paste the printed
summary (one line per condition) here.

## Gotchas / follow-ups

- `.joblib` files load only with compatible library versions; `model_meta.json["versions"]`
  records them. Retrain after upgrading scikit-learn or XGBoost.
- H50 (release gate) will add `prediction_enabled` and `signal_strength` to `index.json`;
  H14 (SHAP) will read `<condition>/xgboost.joblib`.
- Builds on H12 (PR #14) → H11 (#13) → H10 (#12) → H09 (#11).

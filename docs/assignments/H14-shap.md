# H14 — SHAP Explanations

## What we built (plain English)

`ml/src/explain.py` answers "why did the model say that?". For one person it splits the
prediction into a starting point plus one push per factor:

    base value + push(age) + push(BMI) + push(sex) + ... = the model's output

A positive push raised this person's estimate, a negative one lowered it. The pushes add
up **exactly** to what the model output, so the explanation is a faithful account of the
model, not a story told afterwards. For the whole model, it saves a chart of which factors
move estimates the most.

Example (synthetic data, logistic regression):

```
probability 0.079   (base value −1.657 log-odds → final −2.456)
age_band        3      −0.634   younger age lowered the estimate most
bmi            26.2    −0.201
sex           male     +0.134
lifestyle_score 4      +0.124
fruit_daily   True     −0.108
```

## Key decisions

- **The explainer explains the model actually used for predictions**, the condition's best
  model from H13 (logistic regression *or* XGBoost), not always XGBoost as the roadmap
  assumed. Otherwise `/predict` could show one model's number with another model's reasons.
- **No `shap` package needed at prediction time.** XGBoost has TreeSHAP built in
  (`pred_contribs=True`); for logistic regression SHAP is exact and simple
  (`coefficient × (value − training average)`). This keeps the backend light. A CI test
  checks our XGBoost numbers equal `shap.TreeExplainer`'s, and `shap` is still used for the
  beeswarm figure.
- **Early stopping respected**: tree explanations use the same trees as `predict_proba`
  (up to XGBoost's best iteration). The additivity test would fail otherwise.
- **Canonical names only.** One answer becomes several model columns (one-hot levels,
  "was missing" flags); their pushes are summed back to the feature, so explanations read
  `income_band`, never `num__missingindicator_income_band`.
- **Units are log-odds.** Pushes add up there; H46 turns them into plain sentences.
- **`train.py` now saves `explainer.joblib`** next to each condition's models: the best model
  plus its explainer, so `/predict` loads one file and gets the probability and the reasons.

## How it works

| Piece | Purpose |
|---|---|
| `build_explainer(model, x_background)` | explainer for a fitted pipeline (linear or tree) |
| `explainer.explain_one(row)` → `[{feature, value, contribution}]` | **what `/predict` calls**; one model-frame row; biggest push first; JSON-safe values (missing → `null`, yes/no → `true/false`) |
| `explainer.explain(row)` | the same plus `base_value`, `margin`, `probability`, `units` |
| `global_importance(explainer, x)` | mean \|push\| per feature + share of people it pushed up |
| `save_figures(...)` | `shap_bar.png` always; `shap_beeswarm.png` when `shap` is installed |
| `python -m ml.src.explain [--condition ID] [--model logreg\|xgboost]` | figures + `<model>_global_importance.json` in `docs/figures/<condition>/` |

For the backend (H19):

```python
explainer = joblib.load("ml/models/<condition>/explainer.joblib")
x = features.model_frame(features.add_features(row_df), meta["features"])
result = explainer.explain(x)   # probability + contributions, ready to return as JSON
```

## How to run / verify it

```bash
python -m pytest ml/tests/test_explain.py
python -m ml.src.train && python -m ml.src.explain   # needs the processed BRFSS file
```

21 tests, including the roadmap's sanity check: **base value + sum of contributions = the
model's margin** (exact for logistic regression; within 1e-4 for XGBoost, compared with
`logit(predict_proba)`), plus agreement with `shap.TreeExplainer`, canonical names, readable
values, ordering, save/load, figures and CLI. XGBoost/`shap` tests run in CI.

## Results

Synthetic check (5,000 rows where only BMI and age matter): the global chart ranks
**age_band** and **bmi** far above every other factor. Real BRFSS figures: run
`python -m ml.src.explain` after training and commit the PNGs in `docs/figures/` for the report.

## Gotchas / follow-ups

- Logistic regression explanations are relative to the **average training person**; tree
  explanations to the trees' own average. Both add up exactly; the base values differ.
- Correlated features (`bmi`, `bmi_band`, `risk_factor_count`) share credit, so read them together.
- Builds on H13 (PR #15) and below; merge in order.

# H19 - Prediction API: /predict and a patient's risk context

## What

Two authenticated endpoints that turn the trained models into an answer a person can read, and
one rule enforced in the backend rather than the UI.

### `POST /api/v1/predict`

Body: a registry condition id plus canonical answers.

```json
{"condition": "diabetes", "age_band": 9, "sex": "male", "bmi": 33.8,
 "smoker": "daily", "gen_health": 4, "phys_active": false}
```

`age_band`, `sex` and `bmi` are required; every other answer may be left out and is then
genuinely unanswered (the pipeline imputes it and the explanation says so). The answers are
validated against that condition's own `model_meta.json["feature_spec"]` - its ranges, its
allowed categories, nothing else accepted - so retraining with different predictors changes
the accepted body without an API change. Derived features (`bmi_band`, `risk_factor_count`,
`lifestyle_score`) are never asked for; `ml.src.features.add_features` computes them.

Response:

```json
{"condition": "diabetes", "display_name": "Diabetes",
 "risk": 0.3392, "risk_band": "high", "signal_strength": "moderate", "model_name": "logreg",
 "contributions": [{"feature": "bmi", "label": "BMI", "value": 33.8, "contribution": 0.7451}],
 "explanation": {"patient": {...}, "clinician": {...}},
 "disclaimer": "...", "limitations": ["..."]}
```

`contributions` are SHAP values in log-odds, biggest absolute push first, under canonical
feature names with readable labels (no `num__missingindicator_bmi` ever reaches a client).
They are exact: `base_value + sum(contributions) = margin`, and `sigmoid(margin) = risk`.
`explanation` holds both H46 levels, each with `text`, `sentences`, `factors`, `disclaimer` and
`limitations`.

### `GET /api/v1/patients/{id}/risk-context?condition=<id>`

The same payload, built from the patient's record instead of a submitted form, plus:

- `patients_like_this`: `{rate, n, age_band, bmi_band}` - the survey-weighted rate among
  respondents in the same age and BMI band, from `data/processed/brfss_clean.parquet`. It is
  omitted, with `"patients_like_this_available": false`, when there is no parquet or fewer
  than 30 respondents in that cell (the same suppression rule as `/stats/prevalence`).
- `already_diagnosed`: a condition the patient already has is never estimated. The response is
  `{"condition", "display_name", "patient_id", "already_diagnosed": true}` and no number.

Scoping is the H16 rule, unchanged and reused (`Access.patient`): a patient sees only their own
record, a doctor only assigned patients, an admin any record but always audit-logged. A record
the caller may not see returns the *same* 404 body as one that does not exist. Every call is
audited as `patient.risk_context`, refusals included.

A record cannot answer every question a survey asks, and the mapping (`risk_context.py`) says
so rather than guessing: age becomes a BRFSS band, `M` becomes `male`, diagnosed conditions
answer their own outcome columns, and income, education, diet and alcohol stay unanswered.

## Why: the release gate is server-side

A condition that is not `prediction_enabled` in `ml/models/index.json` returns **409** with the
gate's own reason and never a number:

```json
{"error": {"code": "analytics_only",
           "message": "Population analytics only: this is a sensitive condition, ..."}}
```

That holds for both endpoints and for a direct API call with no UI involved - asthma (signal
too weak) and depression (sensitive) are tested for exactly this. The backend only *reads*
`prediction_enabled`, `signal_strength` and `reason`; the rule itself lives in `ml/src/gate.py`
(ADR-0016) and is never re-implemented here.

`risk_band` is presentation only (low < 0.10, moderate < 0.25, high < 0.50, else very high);
it never changes the number.

## No artifacts, no guess

`ml/models/` is gitignored, so a fresh clone has nothing to load. The app still starts and
every other endpoint still works; `/predict` and `/risk-context` answer **503**:

```json
{"error": {"code": "models_unavailable",
           "message": "No trained models are available yet. Run `python -m ml.src.train` ..."}}
```

Artifacts are loaded **once**, in the FastAPI lifespan (`main._lifespan` ->
`models_store.load`), and cached on the immutable app state. Nothing is loaded or trained per
request. A truncated, stale or foreign `explainer.joblib` is logged and skipped, not fatal.
`shap` is deliberately *not* a backend dependency: the saved explainer computes contributions
itself (XGBoost's own TreeSHAP, or exact linear SHAP).

Which model is in use comes from `metrics.json["best_model"]`, whatever it names, and
`GET /models` passes `metrics.json` through unchanged - so a third or fourth model in the
pipeline needs no change in the backend.

## Run

```bash
pip install -r backend/requirements.txt -r requirements-dev.txt
python -m ml.src.train          # optional: without it, /predict answers 503
uvicorn backend.app.main:app --reload
```

```bash
curl -c jar -X POST localhost:8000/api/v1/auth/login -H 'content-type: application/json' \
  -d '{"email":"dr.chen@demo.healthlens.test","password":"demo-password-123"}'

curl -b jar -X POST localhost:8000/api/v1/predict -H 'content-type: application/json' \
  -d '{"condition":"diabetes","age_band":9,"sex":"male","bmi":33.8,"smoker":"daily","gen_health":4,"phys_active":false}'

curl -b jar -X POST localhost:8000/api/v1/predict -H 'content-type: application/json' \
  -d '{"condition":"asthma","age_band":9,"sex":"male","bmi":33.8}'          # 409

curl -b jar 'localhost:8000/api/v1/patients/P-0103/risk-context?condition=heart_disease'
```

## Verify

```bash
ruff check backend && ruff format --check backend
pytest backend/tests --cov=backend/app      # no database, no network, no artifacts needed
```

`tests/test_predict.py` and `tests/test_risk_context.py` train their own small model in a temp
folder (`tests/models_fixture.py`), so they never depend on `ml/models/`. They pin the gate
(409 for asthma and depression), out-of-range and unknown answers (422), the contribution
arithmetic, the scoping (another doctor's patient -> 404) and its auditing, and the
missing-artifact path (503).

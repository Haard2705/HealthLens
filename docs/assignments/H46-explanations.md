# H46 — Prediction Explanations in Plain Language

## What we built (plain English)

`ml/src/explanations.py` turns the SHAP numbers from H14 into sentences people can read, from
fixed templates (no AI). The same explanation, at two reading levels:

**Patient:**
> Your BMI (39.0) is the biggest factor raising this estimate, followed by your age group
> (80 or older). Being female and eating fruit every day lower it. This is an estimate from
> survey data, not a diagnosis: it doesn't mean you have or will get diabetes. Please read
> the limitations below.

**Clinician:**
> Factors increasing the estimate: BMI 39.0 (+1.32), age 80 or older (+0.93), self-rated
> health poor (+0.09). Factors decreasing the estimate: sex: female (-0.14), daily fruit
> (-0.11), household income: not answered (-0.08). Values are SHAP contributions in
> log-odds relative to the model baseline. Population-based estimate from self-reported
> survey data (CDC BRFSS); not a diagnostic result. See the model card for performance and
> limitations.

(Synthetic data; the model's estimate for this person was 59%.)

## Key decisions

- **Deterministic templates, no AI.** Same input, same words, testable with exact golden
  sentences. The optional AI rewording (behind H45) isn't built; H45 doesn't exist yet. When
  it does, `describe()` already returns a `factors` list (exactly the factors the text
  mentions) to check any rewording against.
- **Only factors that really moved the estimate** (|contribution| ≥ 0.05 log-odds) are
  mentioned: at most 2 up and 2 down for patients, 3 each for clinicians.
- **Every model input has its own wording at both levels**, with readable values: age band
  9 → "60-64", income band 1 → "under $15k", general health 4 → "fair", `smoker = some_days`
  → "smoking on some days", a skipped question → "not answering the question about …".
  A test checks no input falls back to a generic phrase or shows a code name.
- **Condition wording comes from the registry.** New optional field `patient_name` in
  `conditions.yaml` (e.g. Hypertension → "high blood pressure"); without it the display name
  is used. A new condition needs no new code (tested). Other conditions used as predictors
  (registry `comorbidity_features`) are named the same way.
- **Sensitive conditions get softer wording**: "linked with" instead of "biggest factor",
  plus a sentence about support. (H50 already blocks individual estimates for them; this is
  a second layer.)
- **Never a diagnosis**: every text says it isn't one, never says "you have …" except in
  "doesn't mean you have …" (tested for every condition and level), and always returns the
  H15 limitations for the UI to show alongside.

## How it works

```python
from ml.src import explanations
result = explainer.explain(x_row)          # H14
limits = json.load(open(".../limitations.json"))["limitations"]   # H15
out = explanations.describe(result, condition, level="patient", limitations=limits)
# out = {level, condition, text, sentences, factors, disclaimer, limitations}
```

## How to run / verify it

```bash
python -m pytest ml/tests/test_explanations.py
```

46 tests: golden sentences (patient, clinician, sensitive, nothing-moved, only-lowering),
factors list matches the text, negligible factors never mentioned, registry wording, a new
condition with no new code, never a diagnosis (all conditions × both levels), limitations
attached, every predictor has proper wording at both levels for every kind of value, label
tables cover every coded value, and an end-to-end run from a trained model.

## Results

See the examples above. Real-data examples will come from `/predict` once H19 is built.

## Gotchas / follow-ups

- H46 lists H40 as a dependency, but the template layer only needs H14; H40/H19 will call
  `describe()` when they return predictions.
- Changing wording means editing `explanations.py` and its golden tests together, on purpose.
- Registry change: optional `patient_name` field (backwards compatible; the registry tests pass).
- Builds on H15 (PR #18) and below; merge in order.

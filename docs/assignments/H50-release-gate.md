# H50 — Prediction Release Gate & Signal Strength

## What we built (plain English)

An automatic, written rule for which conditions may show a person an individual estimate.
If a model is too weak, or the condition is sensitive, the app shows population statistics
instead and explains why. The rule lives in one file (`ml/src/gate.py`) and one decision
record ([ADR-0016](../adr/0016-prediction-release-gate.md)); the API and the UI only read
its result.

## Key decisions

The rule (all must hold, otherwise "population analytics only"):

| Check | Why |
|---|---|
| Not `sensitive` in the registry | some conditions shouldn't get personal estimates at all |
| Trained successfully | no model, no estimate |
| Held-out ROC-AUC ≥ 0.70 | a weak ranking with a precise percentage misleads |
| Brier < no-skill Brier | the percentages must beat "always guess the average" (H10 reports both) |

Signal strength: **strong** ≥ 0.80, **moderate** 0.70–0.80, **weak** < 0.70. It's still
reported for blocked conditions so the model card (H15) can show it.

- **Status: proposed.** The thresholds are the roadmap's suggested starting point; the team
  confirms them in ADR-0016 after the real BRFSS run.
- **Runs automatically** at the end of `python -m ml.src.train`, and on its own with
  `python -m ml.src.gate`.
- **Every registry condition appears in `index.json`**, trained or not, so the UI dropdown
  can list all of them with either a prediction form or a reason.
- **Reasons are plain sentences** for users, with no code names (tested).

## How it works

`index.json` entry after the gate (what the backend and UI read):

```json
"<condition>": {
  "status": "trained", "best_model": "logreg", "test_roc_auc": 0.81, ...,
  "sensitive": false,
  "prediction_enabled": true,
  "signal_strength": "strong",
  "reason": "Individual estimates enabled: held-out ROC-AUC 0.81 (strong signal) and risk percentages better than guessing the average (Brier 0.098 vs 0.124)."
}
```

plus a top-level `release_gate` block recording the thresholds and pointing to ADR-0016.

## How to run / verify it

```bash
python -m pytest ml/tests/test_gate.py
python -m ml.src.gate      # prints ENABLED / analytics + reason for every condition
```

17 tests: a good model is enabled; below 0.70 is disabled (0.70 itself passes); a
sensitive condition is disabled even at 0.95; probabilities no better than guessing are
disabled; untrained/failed conditions are disabled with their reason; strength bands;
reasons are readable sentences; the real registry's sensitive conditions are blocked;
`index.json` is updated in place, keeping other fields; CLI.

## Results

Real BRFSS: run `python -m ml.src.train` and paste the gate printout here. The H09 smoke
test (ROC-AUC ≈ 0.79 diabetes, 0.83 heart disease, 0.79 stroke) suggests those pass as
"moderate"/"strong"; the sensitive condition will show analytics only by design.

## Gotchas / follow-ups

- H19/H24/H25/H40 must read `prediction_enabled` and `reason`, never recompute the rule.
- Builds on H14 (PR #16) and below; merge in order.

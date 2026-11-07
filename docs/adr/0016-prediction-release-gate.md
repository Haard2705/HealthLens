# ADR-0016: Prediction release gate

**Status:** Proposed (Rudra, 2026-11-06). Team to confirm the thresholds after seeing the
real BRFSS numbers.

## Context
We train a model for every condition in the registry, but not every model is good enough
to tell one person "your estimated risk is 34%". A weak model presented with a precise
number misleads, and some conditions are too sensitive for an individual estimate in a
course project however accurate the model is.

## Decision
A condition shows **individual estimates** only if **all** of these hold; otherwise it shows
**population analytics only**, with the reason:

1. **Not sensitive** in `ml/src/conditions.yaml` (`sensitive: true` always blocks).
2. **Trained** successfully by `python -m ml.src.train`.
3. **Held-out ROC-AUC ≥ 0.70** for the condition's chosen model (test set, never used for choices).
4. **Brier score below the no-skill Brier** (`prevalence × (1 − prevalence)`): the
   percentages must beat always guessing the average rate.

**Signal strength** label shown next to an estimate: **strong** ROC-AUC ≥ 0.80,
**moderate** 0.70–0.80, **weak** < 0.70 (weak is always blocked by rule 3).

`ml/src/gate.py` applies the rule after every training run and writes
`prediction_enabled`, `signal_strength` and a plain-language `reason` into
`ml/models/index.json`. **The API and UI only read these fields.**

## Consequences
- Some conditions may ship as analytics only; that is a feature, and the UI says why.
- Changing a threshold is a one-line change in `gate.py` plus an update to this ADR.
- Thresholds are a starting point from the roadmap; revisit once real results are in
  (e.g. if most conditions land between 0.68 and 0.72).

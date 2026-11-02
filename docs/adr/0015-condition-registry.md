# ADR-0015: Condition registry, frozen at nine conditions

**Status:** Accepted (team decision, 2026-10)

## Context
HealthLens covers many chronic conditions, not just one. Copying a pipeline per
condition would multiply code and bugs, and every new condition would touch training,
API, and UI.

## Decision
- One file, `ml/src/conditions.yaml`, defines every condition. `ml/src/registry.py` loads
  and validates it. Training, API, and UI read the registry; no other code names a
  condition (enforced by a test).
- The set is **frozen at nine**: diabetes, hypertension, heart disease (Tier 1); stroke,
  asthma, COPD, arthritis, chronic kidney disease, depression (Tier 2). A test pins it.
- Registry entries use **canonical** column names. Raw survey variable names live only in
  the source adapter (H07).
- **Leakage guard:** another condition's outcome is never a predictor unless the entry opts
  in through `comorbidity_features` (empty for all nine today). A condition's own outcome
  is never a predictor. `feature_columns()` implements this.
- `sensitive: true` (depression) means analytics only; never an individual prediction.
- `source` is `brfss` for all nine. The field exists so a condition could one day use
  another dataset without a schema change (see ADR-0017).

## Consequences
- Adding a condition is deliberate: edit the registry, the pinning test, the adapter
  mapping, and write a model card.
- Validation reports every problem at once, so a bad edit is quick to fix.
- Body-system labels are `cardiometabolic`, `respiratory`, `musculoskeletal`,
  `mental_health`, `renal`.

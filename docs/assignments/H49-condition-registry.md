# H49 — Condition Registry & Per-Condition Pipeline

_Registry built by Haard and Krish (PR #7, ADR-0015); per-condition pipeline, generated
model cards and patient wording added in H10–H15, H46 and H50 (Rudra)._

## What we built (plain English)

One file, `ml/src/conditions.yaml`, lists every condition HealthLens models: its name, body
system, tier, which column holds the answer, whether it's sensitive, and how to describe it.
Every part of the ML pipeline reads this list instead of naming diseases in code, so the
nine conditions go through **one** pipeline instead of nine copies.

## Key decisions

- **Frozen at nine conditions** (team decision, ADR-0015): diabetes, hypertension, heart
  disease (Tier 1); stroke, asthma, COPD, arthritis, chronic kidney disease, depression
  (Tier 2). A test pins the set, so changing it is deliberate.
- **Leakage guard is opt-in, not opt-out (deviation from the roadmap's `exclude_features`).**
  Another condition's outcome is *never* a predictor unless the entry lists it in
  `comorbidity_features` (empty for all nine). A condition's own outcome is never a
  predictor. Safer than an exclude list: forgetting to exclude something can't leak.
- **Wording fields:** `display_name` (what users and clinicians see), `description` (plain
  "what this is"), and, from H46, optional `patient_name` for sentences to patients
  (Hypertension → "high blood pressure").
- `sensitive: true` (depression) → analytics only, enforced by the H50 release gate.
- Canonical column names only; raw survey names live in the BRFSS adapter (H07).
- Validation reports **every** problem at once, with the entry it's in.

## How it works

| Piece | What reads the registry |
|---|---|
| `registry.py` | loads + validates; `feature_columns()` applies the leakage guard |
| `schema.py` | outcome columns come from the registry |
| `features.predictors_for(id)` | each condition's allowed inputs |
| `evaluate.py` (H10) | `--condition`; split per condition |
| `train.py` (H13) | trains every registry condition; `--condition` |
| `explain.py` (H14) | `--condition`; figures per condition |
| `gate.py` (H50) | `sensitive` flag; lists every condition in `index.json` |
| `model_card.py` (H15) | a generated card per condition; Tier 1 gets subgroup results |
| `explanations.py` (H46) | `plain_name` in sentences; comorbidity wording |

A test (`test_no_python_module_hard_codes_a_disease_name`) fails if any Python module in
`ml/src` other than the registry names a condition.

## How to run / verify it

```bash
python -m ml.src.registry            # prints every condition
python -m pytest ml/tests/test_registry.py
```

Acceptance check, adding a condition: add the entry to `conditions.yaml`, its mapping in
`sources/brfss.py`, and update the pinning test. Then `python -m ml.src.train` trains it,
writes `metrics.json`, the gate lists it, and `python -m ml.src.model_card` writes its card,
with no other code changes. (The `/conditions` API entry is the backend's H19, which will
read the same registry.)

## Results

All nine conditions are in the registry and mapped by the BRFSS adapter (H07); 47 registry
tests pass, including the hard-coded-name guard.

## Gotchas / follow-ups

- The roadmap suggested starting with diabetes only; the team mapped all nine at once in H07.
- Clinician wording uses `display_name`; add a `clinician_name` field only if a real need appears.
- The backend should load the registry through `registry.py`, not parse the YAML itself.

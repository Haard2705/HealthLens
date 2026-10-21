# ADR-0017: One backbone survey dataset for all conditions

**Status:** Proposed. Confirm in H06 once the codebook is checked.

## Context
We model nine conditions. Options: (A) one dataset for everything, (B) a different
dataset per condition, (C) one dataset per group of correlated conditions.

## Decision (proposed)
Use **one survey file, the CDC BRFSS annual data, for all nine conditions** (option A).

Why:
- **Same predictors and population for every condition.** One canonical schema, one
  database table, one cleaning path, one survey weight, and prevalence that can be
  compared across conditions.
- **One prediction form per patient.** The patient-to-model mapping (H40) needs a single
  feature set. With different datasets, each condition would ask different questions.
- **Fewer places to leak or mislabel.** Each extra dataset means another codebook, coding
  scheme, licence, and leakage review.
- **Coverage.** The BRFSS core questionnaire's chronic-conditions section covers all
  nine conditions (verified on the 2024 questionnaire; 2025 is listed as the latest year,
  so H06 should check it and record the chosen year here).

Why not B or C: other public datasets for single conditions are typically small, old, or
from narrow populations, and use different variables. A better score on one condition
would cost us a consistent product.

**When we would revisit:** a Tier 1 model proves too weak *and* a larger, well-documented
dataset exists. The registry's `source` field already allows this. It is not used now.

CCHS (Canada) is a second source for the *same* conditions, not extra conditions. It
stays optional (H05, H29).

## Consequences
- Use a single survey year so definitions and weights are consistent.
- The Kaggle diabetes CSV is only a quick-start for the first days.
- Data stays out of git (`data/raw/` is ignored; CI enforces it).

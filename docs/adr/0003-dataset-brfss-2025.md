# ADR-0003 — Dataset: CDC BRFSS 2025 (CCHS stays optional)

**Status:** accepted 2026-11-02 (team lane: Krish)

## Context

The roadmap started on the 2015 Kaggle "Diabetes Health Indicators" file (diabetes only,
no survey weights) with CCHS as the target. The product now covers many conditions, and CCHS
access is unconfirmed (H05, decision due 6 Nov).

## Decision

- Use the **CDC BRFSS 2025 annual file** (`LLCP2025.XPT`, 356,158 respondents, 283 columns).
  It has real survey weights and a question for every condition we model.
- Map it onto a canonical schema (`ml/src/schema.py`) through one adapter
  (`ml/src/sources/brfss.py`). Adding CCHS later means adding `cchs.py`.
- Prefer CDC **calculated variables** (names starting with `_`) where they exist.
- For diabetes (`DIABETE4`) and hypertension (`BPHIGH6`), count "only during pregnancy" and
  "pre-diabetes / borderline" answers as **no**.
- Leave **race / ethnicity** out of model features.

## Consequences

- Data is US, not Canadian; the model card and dashboard must say so until a CCHS swap.
- Counting pre-diabetes as "no" means the diabetes model predicts diagnosed diabetes only.
  About 2.9% of respondents (pre-diabetes) and 0.75% (pregnancy only) are labelled "no".
- The file has no duplicate rows, so no deduplication step is needed (H06 profile).
- The 763 MB raw file stays out of git; teammates download it themselves.

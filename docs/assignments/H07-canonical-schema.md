# H07 — Canonical Schema & Source Adapters

## What we built (plain English)

Our own set of column names and allowed values (the *canonical schema*) and one adapter that
translates the CDC BRFSS 2025 file into it. After the adapter, nothing in the project knows
the raw CDC names or codes, so swapping in another survey later means writing one new file.

## Key decisions

See [ADR-0003](../adr/0003-dataset-brfss-2025.md) and the
[data dictionary](../data-dictionary.md). The short version:

- Use CDC calculated variables where they exist.
- Don't-know / refused / not-asked answers become **missing, never "no"**.
- Diabetes and hypertension count "pregnancy only" and "pre-diabetes / borderline" as **no**.
- Race / ethnicity is not a model feature.
- Heart disease = heart attack or coronary heart disease (matches CDC `_MICHD`).

## How it works

- `ml/src/schema.py`: column lists, allowed values, `validate_schema(df)` which raises one
  `SchemaError` listing every problem.
- `ml/src/sources/brfss.py`: `to_canonical(raw) -> DataFrame`; does not modify its input;
  names any missing raw column in its error.
- `ml/tests/test_no_raw_column_names.py` fails if a raw column name appears anywhere in
  `ml/src/` outside `sources/`.

## How to run / verify it

```bash
python -m pytest
```

Real-data check (done 2026-11-02): all 356,158 rows pass `validate_schema`, and the counts
match the CDC codebook exactly, for example diabetes yes 51,827 and missing 782, heart
disease yes 31,687 (CDC `_MICHD`), BMI missing 32,562, income missing 71,050.

## Gotchas / follow-ups

- `sources/cchs.py` stub is **not written yet**; it needs the codebook from H05.
- `schema.OUTCOMES` is read from the condition registry (`ml/src/conditions.yaml`, H49), so
  `schema.py` names no disease; a test checks the two never disagree. The registry pins the
  blood-pressure outcome column as `high_bp`, so the schema uses that name too.
- Raw numbers are floats (`1.0`); adapters rely on that and map by value.
- H08 still has to handle remaining cleaning (for example BMI outliers and dedupe policy).

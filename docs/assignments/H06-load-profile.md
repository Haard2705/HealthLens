# H06 — Load & Profile the Dataset

## What we built (plain English)

`python -m ml.src.load_data --target DIABETE4` loads the raw CDC BRFSS 2025 file and prints
its size, duplicate rows, how many columns have missing values or possible "don't know /
refused" codes, and the class balance of a chosen outcome column. Paths live in
`ml/src/config.py`. The tests run on a tiny fake table, so CI never needs the real file.

## Key decisions

- **Dataset: CDC BRFSS 2025 annual file (`LLCP2025.XPT`)** instead of the 2015 Kaggle
  diabetes file. It has real survey weights and a question for every condition in the
  roadmap's registry (H49). Cost: we map columns from the 2025 codebook ourselves (H07/H08).
- **Run as a module** (`python -m ml.src.load_data`), not `python ml/src/load_data.py`, so
  the relative import of `config` works. Small deviation from the roadmap wording.
- **Sentinel detection is heuristic**: codes 7/9/77/99/… only count as candidates in columns
  with at most 30 distinct values, since in a continuous column they are real numbers.
  H08 will use a per-variable code table, not this heuristic.

## How it works

- `load_raw(path)` reads `.xpt` (SAS transport, via `pandas.read_sas`) or `.csv`, with clear
  errors for a missing file or an unsupported type.
- `profile(df, target)` returns a dict; `format_profile` turns it into text.
- Tests: `ml/tests/test_load_data.py`.

## How to run / verify it

```bash
python -m pytest ml/tests/test_load_data.py
python -m ml.src.load_data --target DIABETE4
```

## Results (2025 file, `DIABETE4` = "ever told you had diabetes")

| Metric | Value |
|---|---|
| Rows × columns | 356,158 × 283 |
| Duplicate rows | 0 (so no dedupe decision needed for this file) |
| Columns with missing values | 224 |
| Columns with sentinel-code candidates | 189 |

Class balance of `DIABETE4` (raw codes; matches the CDC codebook): 3 = no 81.62%,
1 = yes 14.55%, 4 = pre-diabetes/borderline 2.85%, 2 = yes but only in pregnancy 0.75%,
7 = don't know 0.16%, 9 = refused 0.06%.

## Gotchas / follow-ups

- Every column is read as float64, so categorical codes look like `1.0`, `3.0`.
- Most missing values are "not asked" (skip patterns), which is **not** the same as "no".
  H08 must treat them per variable.
- Turning `DIABETE4` into a yes/no outcome (what to do with codes 2 and 4) is a decision for
  H07; the codebook is in `data/codebooks/`.

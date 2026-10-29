# H06 — Load & Profile the Placeholder Dataset

## What we built (plain English)

A one-command tool that opens the raw survey file and measures it before anyone cleans it:
how many rows and columns, which columns have gaps, how many rows are exact duplicates, which
columns contain "special" codes like 7 or 9 that probably mean *don't know* or *refused*, and
how the outcome is split (for example, what share of respondents report diabetes). We also
picked the dataset year (BRFSS 2025) and listed every variable the later steps need.

## Key decisions

- **Dataset: CDC BRFSS 2025** (`LLCP2025XPT.zip`, 356,158 records, 284 variables), the most
  recent year CDC has published. Variable list and codings: [`docs/brfss-2025-variables.md`](../brfss-2025-variables.md).
- **The loader reads `.csv`, `.xpt` and `.parquet`**, so the same command works for the BRFSS
  XPT, the Kaggle quick-start CSV, and our own processed files later.
- **Sentinel detection is a hint, not a rule.** A code value (6, 7, 8, 9, 77, 88, 99, …) is only
  flagged when it sits *above* every ordinary value in a coded column. That's why `_AGEG5YR`
  (bands 1–13, with 7 and 9 as real ages) isn't flagged, but `GENHLTH` (1–5, plus 7 and 9) is.
  Continuous columns (BMI, weights) are skipped. The final decision per variable is made from
  the codebook in H08.
- **All paths live in `ml/src/config.py`**, along with the single random seed (42).
- **Pipeline modules import each other by name** (`import config`); `pyproject.toml` puts
  `ml/src` on the test path, so `pytest` and `python ml/src/<module>.py` both work.

## How it works

| File | What it does |
|---|---|
| `ml/src/config.py` | Paths (`RAW_DIR`, `PROCESSED_DIR`, `MODELS_DIR`, …), `SEED`, optional `HEALTHLENS_RAW_FILE` override |
| `ml/src/load_data.py` | `load_raw(path)`, `find_raw_file(dir)`, `profile(df, target)`, `format_profile(...)`, CLI `main()` |
| `ml/tests/test_load_data.py` | 33 tests on tiny synthetic data; never needs the real file |
| `docs/brfss-2025-variables.md` | BRFSS variable name + coding for all 11 conditions and 14 predictors |

## How to run / verify it

```bash
# 1. put the data in place (once): unzip LLCP2025XPT.zip into data/raw/
# 2. profile it
python ml/src/load_data.py --target DIABETE4
# optional: save the numbers for the report
python ml/src/load_data.py --target DIABETE4 --json docs/figures/profile-brfss2025.json
# 3. tests (no data needed)
pytest ml/tests
```

"Working" looks like: a report with the row count (~356k), a missing-values table, the
duplicate count, a list of sentinel candidates, and the `DIABETE4` split.

## Results

Run on the real BRFSS 2025 file and paste the printed profile here (the file is ~1 GB in
memory, so it has to be run on a teammate's machine; it can't be downloaded in CI).

## Gotchas / follow-ups

- **CDC names the XPT file with a trailing space** (`LLCP2025.XPT `); the loader handles it.
- **`_BMI5` has two implied decimals** (2500 = 25.00); **`88` means "none"** on day-count
  questions; **`_RFHYPE6`-style variables use 1 = no, 2 = yes**. All carried into H07/H08.
- **Codings marked ⚠️ in the variable list** still need checking against the 2025 codebook
  before `sources/brfss.py` is written (H07).
- **Decide in H07/H49:** diabetes outcome treats prediabetes and pregnancy-only as "no"
  (proposed), and asthma uses "ever told" (`ASTHMA3`) rather than "current".
- Duplicate rows: the profile reports them; the dedupe decision is recorded as an ADR in H08.

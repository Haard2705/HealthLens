# Data dictionary (canonical schema)

Source of truth in code: `ml/src/schema.py`. Mapping code: `ml/src/sources/brfss.py`.
Raw codes are from the **CDC BRFSS 2025** codebook (`data/codebooks/`). "Missing" always
means don't know, refused or not asked, **never "no"**. CCHS column is filled in by H05/H29.

## Predictors

| Canonical name | Type and values | BRFSS 2025 source | Mapping | CCHS source |
|---|---|---|---|---|
| `survey_weight` | float > 0 | `_LLCPWT` | as is | TBD (H05) |
| `age_band` | 1–13 (5-year bands, 18–24 … 80+) | `_AGEG5YR` | 14 (unknown) → missing | TBD |
| `sex` | `male`, `female` | `SEXVAR` | 1 / 2 | TBD |
| `bmi` | float 10–100 | `_BMI5` | divide by 100 (two implied decimals); blank → missing | TBD |
| `smoker` | `daily`, `some_days`, `former`, `never` | `_SMOKER3` | 1–4; 9 → missing | TBD |
| `gen_health` | 1 (excellent) – 5 (poor) | `GENHLTH` | 7, 9 → missing | TBD |
| `income_band` | 1–7 | `INCOME3` | 1,2→1; 3,4→2; 5→3; 6→4; 7,8→5; 9,10→6; 11→7; 77, 99 → missing | TBD |
| `education_band` | 1–4 | `EDUCA` | 1–3→1; 4→2; 5→3; 6→4; 9 → missing | TBD |
| `phys_active` | bool | `_TOTINDA` | 1 yes, 2 no, 9 → missing | TBD |
| `heavy_alcohol` | bool | `_RFDRHV9` | 1 no, 2 yes, 9 → missing | TBD |
| `fruit_daily` | bool | `_FRTLT1A` | 1 yes, 2 no, 9 → missing | TBD |
| `veg_daily` | bool | `_VEGLT1A` | 1 yes, 2 no, 9 → missing | TBD |
| `high_chol` | bool | `TOLDHI3` | 1 yes, 2 no, 7/9/blank → missing | may be unavailable |

## Outcomes (one boolean per condition)

| Canonical name | BRFSS 2025 source | Mapping |
|---|---|---|
| `diabetes` | `DIABETE4` | 1 yes; **2 (pregnancy only), 3, 4 (pre-diabetes) no**; 7, 9, blank missing |
| `high_bp` (hypertension) | `BPHIGH6` | 1 yes; **2 (pregnancy only), 3, 4 (borderline) no**; 7, 9, blank missing |
| `heart_disease` | `CVDINFR4`, `CVDCRHD4` | yes if either is 1; no if both are 2; otherwise missing (same as CDC `_MICHD`) |
| `stroke` | `CVDSTRK3` | 1 yes, 2 no |
| `asthma` | `ASTHMA3` | 1 yes, 2 no |
| `copd` | `CHCCOPD3` | 1 yes, 2 no |
| `arthritis` | `HAVARTH4` | 1 yes, 2 no |
| `kidney_disease` | `CHCKDNY2` | 1 yes, 2 no |
| `depression` | `ADDEPEV3` | 1 yes, 2 no |

## Deliberately left out

- **Race / ethnicity** (`_RACE`): not a model feature, to avoid building bias into
  predictions. May be used later for subgroup reporting only.
- A condition is either an outcome or a predictor, never both, for the same model; the
  condition registry (H49) lists each condition's exclusions.

# BRFSS 2025: variables for every condition and predictor

Input to **H07** (canonical schema + `sources/brfss.py`) and **H49** (condition registry).
Produced in **H06**.

**File:** CDC BRFSS 2025 combined landline + cell (`LLCP2025XPT.zip`, SAS transport, 284
variables, 356,158 records, files dated August 2026). Download page:
<https://www.cdc.gov/brfss/annual_data/annual_2025.html>.

**How sure are we?**

- ✅ **Confirmed**: value codes read from CDC's *2025 Calculated Variables* document (v4).
- ⚠️ **Confirm in H07**: the variable name is in the 2025 variable layout, but the value codes
  below are BRFSS's long-standing coding for that question and have **not** yet been checked
  against the 2025 codebook. Check each one before writing the adapter.

Codes `7` = don't know, `9` = refused, `.`/blank = not asked or missing, unless stated.
**Missing outcome is never "no"** (H08): those rows are dropped from that condition's model.

## Outcomes (one per condition, §5.4)

| Condition (tier) | Variable | Coding → canonical `True` / `False` / missing | Status |
|---|---|---|---|
| Diabetes (1) | `DIABETE4` | 1 yes → T · 3 no → F · 2 pregnancy-only → F · 4 prediabetes → F · 7/9 → missing | ⚠️ (team decision: is prediabetes F or excluded? record in ADR) |
| Hypertension (1) | `_RFHYPE6` | 2 yes → T · 1 no → F · 9 → missing | ✅ |
| Heart disease (1) | `_MICHD` | 1 MI or CHD → T · 2 neither → F · `.` → missing | ✅ codes; confirm the column is in the public file |
| Stroke (2) | `CVDSTRK3` | 1 → T · 2 → F · 7/9 → missing | ⚠️ |
| Asthma (2) | `ASTHMA3` (ever) or `_ASTHMS1` (1 current · 2 former · 3 never · 9) | pick one in H49; recommended `ASTHMA3` "ever told" for consistency with other conditions | ⚠️ / ✅ |
| COPD (2) | `CHCCOPD3` | 1 → T · 2 → F · 7/9 → missing | ⚠️ |
| Arthritis (2) | `_DRDXAR2` | 1 → T · 2 → F · `.` → missing | ✅ |
| Kidney disease (2) | `CHCKDNY2` | 1 → T · 2 → F · 7/9 → missing | ⚠️ |
| Depression (2, sensitive) | `ADDEPEV3` | 1 → T · 2 → F · 7/9 → missing | ⚠️ |
| High cholesterol (3) | `_RFCHOL3` | 2 → T · 1 → F · 9 / `.` (never checked) → missing | ✅ |
| Cancer (3) | `CHCSCNC1` (skin), `CHCOCNC1` (other) | 1 → T · 2 → F · 7/9 → missing | ⚠️ |

## Predictors (canonical features, §5.3)

| Canonical | Variable | Coding | Status |
|---|---|---|---|
| `bmi` | `_BMI5` | integer with **2 implied decimals** (2500 → 25.00); `.` missing. Divide by 100. | ✅ |
| `bmi_band` | `_BMI5CAT` | 1 under · 2 normal · 3 over · 4 obese (or derive from `bmi` in H09) | ✅ |
| `high_bp` | `_RFHYPE6` | as above (also the hypertension outcome: excluded from that model) | ✅ |
| `high_chol` | `_RFCHOL3` | as above | ✅ |
| `smoker` | `_SMOKER3` | 1 / 2 current → T · 3 former / 4 never → F · 9 missing (keep 4 levels as a category too) | ✅ |
| `phys_active` | `_TOTINDA` | 1 → T · 2 → F · 9 missing | ✅ |
| `fruits_veg` | `_FRTLT1A`, `_VEGLT1A` | 1 ≥ once/day · 2 less · 9 missing | ✅ |
| `heavy_alcohol` | `_RFDRHV9` | 2 → T · 1 → F · 9 missing | ✅ |
| `gen_health` | `GENHLTH` | 1 excellent … 5 poor · 7/9 missing | ⚠️ |
| `age_band` | `_AGEG5YR` | 1 = 18–24 … 13 = 80+ · **14** missing (not 7/9: 7 and 9 are real age bands) | ✅ |
| `sex` | `_SEX` | 1 male · 2 female | ✅ |
| `income_band` | `_INCOMG1` | 1 < $15k … 7 ≥ $200k · 9 missing | ✅ |
| `education_band` | `_EDUCAG` | 1 < high school … 4 college grad · 9 missing | ✅ |
| `survey_weight` | `_LLCPWT` | final weight, continuous | ⚠️ (not in the calculated-variables doc; confirm in codebook) |

Also available if H09 wants them: `PHYSHLTH` / `MENTHLTH` (days of poor health, with 88 = none),
`_AGE80` (imputed age, capped at 80).

## Gotchas found while mapping

- **`_AGEG5YR`: 7 and 9 are real age bands.** A global "replace 7/9 with NaN" would delete
  everyone aged 50–54 and 60–64. Sentinel handling must be per variable (H08 already says so).
- **`88` means "none"** on day-count questions (`PHYSHLTH`, `MENTHLTH`), i.e. zero, not missing.
- **`_BMI5` has implied decimals**: forgetting to divide by 100 gives BMIs in the thousands.
- **Calculated "risk factor" variables flip the usual coding** (`_RFHYPE6`: 1 = no, 2 = yes).
- **Heart disease and hypertension overlap with predictors**: the registry must exclude
  `high_bp` from the hypertension model, and nothing in the heart-disease model may be derived
  from `CVDINFR4` / `CVDCRHD4`.

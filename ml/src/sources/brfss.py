"""Adapter: CDC BRFSS 2025 annual file -> canonical schema.

Codes come from the 2025 codebook (data/codebooks/). Where CDC publishes a calculated
variable (names starting with an underscore) we use it, because it already handles skip
patterns. Don't-know / refused / not-asked answers become missing, never "no".
"""

import pandas as pd

from ..schema import ALL_COLUMNS

YES_NO = {1: True, 2: False}
# For these two questions the codebook adds "only during pregnancy" (2) and "pre-diabetes" /
# "borderline" (4). The team decided on 2026-11-02 to count them as "no".
YES_OTHERS_NO = {1: True, 2: False, 3: False, 4: False}

# canonical column -> (raw column, code map). Codes not listed (7, 9, blank) become missing.
BOOLEAN_COLUMNS = {
    "diabetes": ("DIABETE4", YES_OTHERS_NO),
    "hypertension": ("BPHIGH6", YES_OTHERS_NO),
    "stroke": ("CVDSTRK3", YES_NO),
    "asthma": ("ASTHMA3", YES_NO),
    "copd": ("CHCCOPD3", YES_NO),
    "kidney_disease": ("CHCKDNY2", YES_NO),
    "depression": ("ADDEPEV3", YES_NO),
    "arthritis": ("HAVARTH4", YES_NO),
    "high_chol": ("TOLDHI3", YES_NO),
    "phys_active": ("_TOTINDA", YES_NO),
    "heavy_alcohol": ("_RFDRHV9", {1: False, 2: True}),
    "fruit_daily": ("_FRTLT1A", YES_NO),
    "veg_daily": ("_VEGLT1A", YES_NO),
}
INCOME_BANDS = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 5, 8: 5, 9: 6, 10: 6, 11: 7}
EDUCATION_BANDS = {1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4}
SMOKER_LEVELS = {1: "daily", 2: "some_days", 3: "former", 4: "never"}
SEX_LABELS = {1: "male", 2: "female"}

RAW_COLUMNS = (
    *(raw for raw, _ in BOOLEAN_COLUMNS.values()),
    "CVDINFR4",
    "CVDCRHD4",
    "_AGEG5YR",
    "SEXVAR",
    "_BMI5",
    "_SMOKER3",
    "GENHLTH",
    "INCOME3",
    "EDUCA",
    "_LLCPWT",
)


def _heart_disease(heart_attack: pd.Series, chd: pd.Series) -> pd.Series:
    """Heart attack or coronary heart disease; one "yes" is enough, otherwise both must be "no"."""
    either_yes = (heart_attack == 1) | (chd == 1)
    both_no = (heart_attack == 2) & (chd == 2)
    result = pd.Series(pd.NA, index=heart_attack.index, dtype="boolean")
    result[both_no] = False
    result[either_yes] = True
    return result


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame in the canonical schema; the input is not modified."""
    absent = [c for c in RAW_COLUMNS if c not in raw.columns]
    if absent:
        raise KeyError(f"raw BRFSS file is missing columns: {', '.join(absent)}")

    out = pd.DataFrame(index=raw.index)
    out["survey_weight"] = raw["_LLCPWT"].astype(float)
    out["age_band"] = raw["_AGEG5YR"].where(raw["_AGEG5YR"].between(1, 13)).astype("Int64")
    out["sex"] = raw["SEXVAR"].map(SEX_LABELS)
    out["bmi"] = raw["_BMI5"].astype(float) / 100
    out["smoker"] = raw["_SMOKER3"].map(SMOKER_LEVELS)
    out["gen_health"] = raw["GENHLTH"].where(raw["GENHLTH"].between(1, 5)).astype("Int64")
    out["income_band"] = raw["INCOME3"].map(INCOME_BANDS).astype("Int64")
    out["education_band"] = raw["EDUCA"].map(EDUCATION_BANDS).astype("Int64")
    for canonical, (source, codes) in BOOLEAN_COLUMNS.items():
        out[canonical] = raw[source].map(codes).astype("boolean")
    out["heart_disease"] = _heart_disease(raw["CVDINFR4"], raw["CVDCRHD4"])
    return out[list(ALL_COLUMNS)]

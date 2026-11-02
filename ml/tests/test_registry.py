"""Tests for the condition registry (H49).

The registry is the single source of truth for which conditions HealthLens models.
These tests pin the agreed set of nine conditions, so changing it is a deliberate
edit to this file as well as to conditions.yaml.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import pytest

from ml.src import registry as reg

SRC_DIR = Path(reg.__file__).resolve().parent

AGREED_IDS = {
    "diabetes",
    "hypertension",
    "heart_disease",
    "stroke",
    "asthma",
    "copd",
    "arthritis",
    "kidney_disease",
    "depression",
}
TIER_1 = {"diabetes", "hypertension", "heart_disease"}


def _entry(**overrides) -> dict:
    base = {
        "id": "diabetes",
        "display_name": "Diabetes",
        "body_system": "cardiometabolic",
        "tier": 1,
        "source": "brfss",
        "outcome_column": "diabetes",
        "description": "Self-reported diagnosis.",
    }
    return {**base, **overrides}


def _raw(*entries: dict) -> dict:
    return {"conditions": list(entries)}


def _two_valid() -> dict:
    return _raw(
        _entry(),
        _entry(
            id="stroke",
            display_name="Stroke",
            outcome_column="stroke",
            tier=2,
        ),
    )


# --------------------------------------------------------------------------
# The shipped registry
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shipped() -> reg.Registry:
    return reg.load_registry()


def test_shipped_registry_is_exactly_the_agreed_nine(shipped: reg.Registry) -> None:
    assert set(shipped.ids) == AGREED_IDS
    assert len(shipped.ids) == 9


def test_shipped_tiers_match_the_agreement(shipped: reg.Registry) -> None:
    assert {c.id for c in shipped.by_tier(1)} == TIER_1
    assert {c.id for c in shipped.by_tier(2)} == AGREED_IDS - TIER_1


def test_only_depression_is_sensitive(shipped: reg.Registry) -> None:
    assert {c.id for c in shipped.conditions if c.sensitive} == {"depression"}


def test_hypertension_uses_the_high_bp_column(shipped: reg.Registry) -> None:
    assert shipped.get("hypertension").outcome_column == "high_bp"


def test_shipped_registry_uses_only_the_brfss_source(shipped: reg.Registry) -> None:
    assert {c.source for c in shipped.conditions} == {"brfss"}


def test_registry_is_immutable(shipped: reg.Registry) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        shipped.get("diabetes").tier = 2  # type: ignore[misc]
    assert isinstance(shipped.conditions, tuple)


# --------------------------------------------------------------------------
# Parsing and validation
# --------------------------------------------------------------------------


def test_valid_registry_parses() -> None:
    registry = reg.parse_registry(_two_valid())
    assert registry.ids == ("diabetes", "stroke")
    assert registry.outcome_columns == ("diabetes", "stroke")


def test_optional_fields_have_safe_defaults() -> None:
    condition = reg.parse_registry(_two_valid()).get("diabetes")
    assert condition.sensitive is False
    assert condition.comorbidity_features == ()


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: r["conditions"].append(_entry()), "duplicate condition id"),
        (
            lambda r: r["conditions"].append(_entry(id="other", outcome_column="diabetes")),
            "duplicate outcome_column",
        ),
        (lambda r: r["conditions"][0].update(id="Diabetes"), "invalid id"),
        (lambda r: r["conditions"][0].update(outcome_column="Not-Valid"), "invalid outcome_column"),
        (lambda r: r["conditions"][0].update(tier=3), "tier"),
        (lambda r: r["conditions"][0].update(tier=True), "tier"),
        (lambda r: r["conditions"][0].update(body_system="astrology"), "body_system"),
        (lambda r: r["conditions"][0].update(source="kaggle"), "source"),
        (lambda r: r["conditions"][0].update(sensitive="yes"), "sensitive"),
        (lambda r: r["conditions"][0].update(display_name="  "), "display_name"),
        (lambda r: r["conditions"][0].update(description=""), "description"),
        (lambda r: r["conditions"][0].update(typo_field=1), "unknown field"),
        (lambda r: r["conditions"][0].pop("outcome_column"), "missing field"),
        (lambda r: r["conditions"][0].update(comorbidity_features=["nope"]), "comorbidity"),
        (lambda r: r["conditions"][0].update(comorbidity_features=["diabetes"]), "comorbidity"),
        (
            lambda r: r["conditions"][0].update(comorbidity_features=["stroke", "stroke"]),
            "comorbidity",
        ),
        (lambda r: r["conditions"][0].update(comorbidity_features="stroke"), "comorbidity"),
    ],
)
def test_invalid_registries_are_rejected(mutate, expected: str) -> None:
    raw = copy.deepcopy(_two_valid())
    mutate(raw)
    with pytest.raises(reg.RegistryError, match=expected):
        reg.parse_registry(raw)


@pytest.mark.parametrize("raw", [None, [], "conditions", {"conditions": []}, {"conditions": {}}])
def test_malformed_top_level_is_rejected(raw) -> None:
    with pytest.raises(reg.RegistryError):
        reg.parse_registry(raw)


def test_all_problems_are_reported_together() -> None:
    raw = copy.deepcopy(_two_valid())
    raw["conditions"][0]["tier"] = 9
    raw["conditions"][1]["source"] = "nowhere"
    with pytest.raises(reg.RegistryError) as excinfo:
        reg.parse_registry(raw)
    message = str(excinfo.value)
    assert "tier" in message
    assert "source" in message


def test_comorbidity_may_reference_another_conditions_outcome() -> None:
    raw = copy.deepcopy(_two_valid())
    raw["conditions"][1]["comorbidity_features"] = ["diabetes"]
    registry = reg.parse_registry(raw)
    assert registry.get("stroke").comorbidity_features == ("diabetes",)


# --------------------------------------------------------------------------
# Loading from disk
# --------------------------------------------------------------------------


def test_load_missing_file_raises_clear_error(tmp_path: Path) -> None:
    with pytest.raises(reg.RegistryError, match="not found"):
        reg.load_registry(tmp_path / "nope.yaml")


def test_load_invalid_yaml_raises_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("conditions: [unclosed", encoding="utf-8")
    with pytest.raises(reg.RegistryError, match="not valid YAML"):
        reg.load_registry(path)


def test_load_round_trips_a_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "ok.yaml"
    path.write_text(
        "conditions:\n"
        "  - id: diabetes\n"
        "    display_name: Diabetes\n"
        "    body_system: cardiometabolic\n"
        "    tier: 1\n"
        "    source: brfss\n"
        "    outcome_column: diabetes\n"
        "    description: Self-reported.\n",
        encoding="utf-8",
    )
    assert reg.load_registry(path).ids == ("diabetes",)


# --------------------------------------------------------------------------
# Lookups
# --------------------------------------------------------------------------


def test_get_unknown_condition_raises() -> None:
    registry = reg.parse_registry(_two_valid())
    with pytest.raises(reg.RegistryError, match="unknown condition"):
        registry.get("nonsense")


# --------------------------------------------------------------------------
# Leakage guard: which columns a condition's model may use
# --------------------------------------------------------------------------

BASE = ("bmi", "smoker", "age_band", "diabetes", "stroke")


def test_other_outcomes_are_not_predictors_by_default() -> None:
    registry = reg.parse_registry(_two_valid())
    assert reg.feature_columns(registry, "diabetes", BASE) == ("bmi", "smoker", "age_band")
    assert reg.feature_columns(registry, "stroke", BASE) == ("bmi", "smoker", "age_band")


def test_comorbidity_features_are_opt_in() -> None:
    raw = copy.deepcopy(_two_valid())
    raw["conditions"][1]["comorbidity_features"] = ["diabetes"]
    registry = reg.parse_registry(raw)
    assert reg.feature_columns(registry, "stroke", BASE) == (
        "bmi",
        "smoker",
        "age_band",
        "diabetes",
    )


def test_comorbidity_column_missing_from_base_is_appended_once() -> None:
    raw = copy.deepcopy(_two_valid())
    raw["conditions"][1]["comorbidity_features"] = ["diabetes"]
    registry = reg.parse_registry(raw)
    cols = reg.feature_columns(registry, "stroke", ("bmi", "smoker"))
    assert cols == ("bmi", "smoker", "diabetes")


def test_own_outcome_is_never_a_feature() -> None:
    registry = reg.parse_registry(_two_valid())
    for condition_id in registry.ids:
        assert registry.get(condition_id).outcome_column not in reg.feature_columns(
            registry, condition_id, BASE
        )


def test_non_outcome_predictors_pass_through_unchanged() -> None:
    registry = reg.parse_registry(_two_valid())
    assert reg.feature_columns(registry, "diabetes", ("a", "b", "c")) == ("a", "b", "c")


def test_feature_columns_unknown_condition_raises() -> None:
    registry = reg.parse_registry(_two_valid())
    with pytest.raises(reg.RegistryError, match="unknown condition"):
        reg.feature_columns(registry, "nonsense", BASE)


# --------------------------------------------------------------------------
# Checking the registry against the canonical schema (used by H07)
# --------------------------------------------------------------------------


def test_validate_against_schema_passes_when_columns_exist() -> None:
    registry = reg.parse_registry(_two_valid())
    reg.validate_against_schema(registry, ["diabetes", "stroke", "bmi"])


def test_validate_against_schema_reports_missing_outcome_columns() -> None:
    registry = reg.parse_registry(_two_valid())
    with pytest.raises(reg.RegistryError, match="stroke"):
        reg.validate_against_schema(registry, ["diabetes", "bmi"])


# --------------------------------------------------------------------------
# Keeping disease names out of the code
# --------------------------------------------------------------------------

DISEASE_WORDS = (
    "diabetes",
    "hypertension",
    "heart_disease",
    "stroke",
    "asthma",
    "copd",
    "arthritis",
    "kidney",
    "depression",
)


def test_no_python_module_hard_codes_a_disease_name() -> None:
    """Conditions live in conditions.yaml. Raw-column mappings (ml/src/sources/) may name them."""
    offenders = []
    for path in SRC_DIR.rglob("*.py"):
        if "sources" in path.relative_to(SRC_DIR).parts:
            continue
        text = path.read_text(encoding="utf-8").lower()
        hits = [word for word in DISEASE_WORDS if word in text]
        if hits:
            offenders.append(f"{path.relative_to(SRC_DIR)}: {hits}")
    assert not offenders, "disease names found in code:\n" + "\n".join(offenders)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def test_main_prints_every_condition(capsys: pytest.CaptureFixture[str]) -> None:
    assert reg.main([]) == 0
    out = capsys.readouterr().out
    for condition_id in AGREED_IDS:
        assert condition_id in out


def test_non_mapping_entry_is_rejected() -> None:
    with pytest.raises(reg.RegistryError, match="must be a mapping"):
        reg.parse_registry({"conditions": ["diabetes"]})

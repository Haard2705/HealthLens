"""H50: the release gate decides, in one place, which conditions may show personal estimates."""

import json

import pytest
from src import gate
from src.registry import load_registry, parse_registry

REGISTRY = load_registry()


def trained(auc: float = 0.82, brier: float = 0.10, no_skill: float = 0.13) -> dict:
    return {
        "status": "trained",
        "best_model": "logreg",
        "test_roc_auc": auc,
        "test_brier": brier,
        "test_brier_no_skill": no_skill,
    }


def two_condition_registry(sensitive_second: bool = True):
    base = {
        "display_name": "X",
        "body_system": "cardiometabolic",
        "tier": 1,
        "source": "brfss",
        "description": "d",
    }
    first, second = REGISTRY.conditions[0], REGISTRY.conditions[1]
    return parse_registry(
        {
            "conditions": [
                {**base, "id": first.id, "outcome_column": first.outcome_column},
                {
                    **base,
                    "id": second.id,
                    "outcome_column": second.outcome_column,
                    "sensitive": sensitive_second,
                },
            ]
        }
    )


# ---- the rule ---------------------------------------------------------------------------


def test_a_good_model_is_enabled() -> None:
    d = gate.decide(trained(auc=0.82), sensitive=False)
    assert d.prediction_enabled is True
    assert d.signal_strength == "strong"


def test_below_minimum_auc_is_disabled() -> None:
    d = gate.decide(trained(auc=0.65), sensitive=False)
    assert d.prediction_enabled is False
    assert d.signal_strength == "weak"
    assert "0.65" in d.reason and "0.70" in d.reason


def test_the_minimum_itself_passes() -> None:
    assert gate.decide(trained(auc=0.70), sensitive=False).prediction_enabled is True


def test_sensitive_condition_is_disabled_whatever_the_score() -> None:
    d = gate.decide(trained(auc=0.95), sensitive=True)
    assert d.prediction_enabled is False
    assert d.signal_strength == "strong"  # still reported, for the model card
    assert "sensitive" in d.reason


def test_probabilities_no_better_than_guessing_are_disabled() -> None:
    d = gate.decide(trained(auc=0.85, brier=0.14, no_skill=0.13), sensitive=False)
    assert d.prediction_enabled is False
    assert "guessing" in d.reason


def test_untrained_or_failed_condition_is_disabled() -> None:
    d = gate.decide({"status": "failed", "reason": "only one answer class"}, sensitive=False)
    assert d.prediction_enabled is False
    assert d.signal_strength is None
    assert "only one answer class" in d.reason
    assert gate.decide({}, sensitive=False).prediction_enabled is False


@pytest.mark.parametrize(
    ("auc", "strength"),
    [(0.95, "strong"), (0.80, "strong"), (0.79, "moderate"), (0.70, "moderate"), (0.6, "weak")],
)
def test_signal_strength_bands(auc, strength) -> None:
    assert gate.signal_strength(auc) == strength


def test_reasons_are_plain_sentences() -> None:
    cases = [
        gate.decide(trained(), False),
        gate.decide(trained(auc=0.6), False),
        gate.decide(trained(), True),
        gate.decide(trained(brier=0.2), False),
        gate.decide({}, False),
    ]
    for d in cases:
        assert d.reason[0].isupper() and d.reason.endswith(".")
        assert "_" not in d.reason  # no code names like test_roc_auc


# ---- applying it to index.json -----------------------------------------------------------


def test_apply_gate_fills_every_registry_condition() -> None:
    first = REGISTRY.conditions[0].id
    gated = gate.apply_gate({"conditions": {first: trained()}}, REGISTRY)
    assert set(gated["conditions"]) == set(REGISTRY.ids)
    for entry in gated["conditions"].values():
        assert {"prediction_enabled", "signal_strength", "reason", "sensitive"} <= set(entry)
    assert gated["conditions"][first]["prediction_enabled"] is True
    assert gated["release_gate"]["min_roc_auc"] == 0.70


def test_sensitive_flag_comes_from_the_registry() -> None:
    registry = two_condition_registry(sensitive_second=True)
    second = registry.conditions[1].id
    gated = gate.apply_gate({"conditions": {second: trained(auc=0.9)}}, registry)
    assert gated["conditions"][second]["sensitive"] is True
    assert gated["conditions"][second]["prediction_enabled"] is False


def test_real_registry_blocks_its_sensitive_conditions() -> None:
    index = {"conditions": {c: trained(auc=0.9) for c in REGISTRY.ids}}
    gated = gate.apply_gate(index, REGISTRY)["conditions"]
    for condition in REGISTRY.conditions:
        assert gated[condition.id]["prediction_enabled"] is (not condition.sensitive)


def test_gate_file_rewrites_index_in_place(tmp_path) -> None:
    first = REGISTRY.conditions[0].id
    (tmp_path / "index.json").write_text(
        json.dumps({"conditions": {first: trained(auc=0.6)}, "git_sha": "abc"})
    )
    gate.gate_file(tmp_path)
    saved = json.loads((tmp_path / "index.json").read_text())
    assert saved["git_sha"] == "abc"  # other fields kept
    assert saved["conditions"][first]["prediction_enabled"] is False


def test_cli_prints_every_condition(tmp_path, capsys) -> None:
    assert gate.main(["--models-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for condition_id in REGISTRY.ids:
        assert condition_id in out

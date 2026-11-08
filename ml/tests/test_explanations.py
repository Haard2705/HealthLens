"""H46: SHAP explanations in plain language, checked against golden examples."""

import numpy as np
import pytest
from src import evaluate, explain, explanations, features, logreg
from src.registry import load_registry, parse_registry

from .test_logreg import signal_frame

REGISTRY = load_registry()
PLAIN = next(c for c in REGISTRY.conditions if not c.sensitive and not c.patient_name)
RENAMED = next(c for c in REGISTRY.conditions if c.patient_name and not c.display_name.isupper())
SENSITIVE = next(c for c in REGISTRY.conditions if c.sensitive)


def c(feature, value, contribution):
    return {"feature": feature, "value": value, "contribution": contribution}


EXAMPLE = {
    "base_value": -1.6,
    "contributions": [
        c("bmi", 31.2, 0.42),
        c("age_band", 9, 0.31),
        c("phys_active", True, -0.20),
        c("sex", "male", 0.03),  # negligible: never mentioned
        c("fruit_daily", False, 0.12),
        c("income_band", None, -0.08),
    ],
}


# ---- golden examples: exact wording -------------------------------------------------------


def test_patient_golden_example() -> None:
    out = explanations.describe(EXAMPLE, PLAIN, "patient")
    assert out["sentences"] == [
        "Your BMI (31.2) is the biggest factor raising this estimate, followed by your age "
        "group (60-64).",
        "Being physically active and not answering the question about household income lower it.",
        "This is an estimate from survey data, not a diagnosis: it doesn't mean you have or "
        f"will get {PLAIN.plain_name}. Please read the limitations below.",
    ]
    assert out["text"] == " ".join(out["sentences"])


def test_clinician_golden_example() -> None:
    out = explanations.describe(EXAMPLE, PLAIN, "clinician")
    assert out["sentences"][:3] == [
        "Factors increasing the estimate: BMI 31.2 (+0.42), age 60-64 (+0.31), "
        "fruit less than daily (+0.12).",
        "Factors decreasing the estimate: physically active in the past 30 days (-0.20), "
        "household income: not answered (-0.08).",
        "Values are SHAP contributions in log-odds relative to the model baseline.",
    ]
    assert "not a diagnostic result" in out["text"]


def test_sensitive_condition_gets_gentle_wording_golden() -> None:
    out = explanations.describe(EXAMPLE, SENSITIVE, "patient")
    assert out["sentences"][:3] == [
        "The answers most linked with a higher estimate were your BMI (31.2) and your age "
        "group (60-64).",
        "Being physically active and not answering the question about household income were "
        "linked with a lower one.",
        explanations.SUPPORT_SENTENCE,
    ]
    assert "biggest factor" not in out["text"]


def test_only_lowering_factors() -> None:
    out = explanations.describe([c("phys_active", True, -0.4)], PLAIN, "patient")
    assert out["sentences"][:2] == [
        "None of your answers raised this estimate much.",
        "Being physically active lowers it.",
    ]


def test_nothing_moved_the_estimate() -> None:
    out = explanations.describe([c("bmi", 24.0, 0.01), c("sex", "female", -0.02)], PLAIN)
    assert out["sentences"][0].startswith("None of your answers moved this estimate much")
    assert out["factors"] == []


# ---- the rules ---------------------------------------------------------------------------


def test_factors_list_exactly_what_the_text_mentions() -> None:
    out = explanations.describe(EXAMPLE, PLAIN, "patient")
    assert [f["feature"] for f in out["factors"]] == [
        "bmi",
        "age_band",
        "phys_active",
        "income_band",
    ]
    for f in out["factors"]:
        assert f["phrase"].lower() in out["text"].lower()
    assert out["factors"][0]["direction"] == "raises"
    assert out["factors"][2]["direction"] == "lowers"


def test_negligible_factors_are_never_mentioned() -> None:
    for level in explanations.LEVELS:
        assert "male" not in explanations.describe(EXAMPLE, PLAIN, level)["text"]


def test_condition_wording_comes_from_the_registry() -> None:
    out = explanations.describe(EXAMPLE, RENAMED, "patient")
    assert f"will get {RENAMED.patient_name}." in out["text"]


def test_a_new_condition_needs_no_new_code() -> None:
    entry = {
        "id": "new_thing",
        "display_name": "New thing",
        "body_system": "renal",
        "tier": 2,
        "source": "brfss",
        "outcome_column": "new_thing",
        "description": "d",
        "patient_name": "a brand-new condition",
    }
    new = parse_registry({"conditions": [entry]}).conditions[0]
    out = explanations.describe(EXAMPLE, new, "patient", registry=REGISTRY)
    assert "will get a brand-new condition." in out["text"]


def test_never_states_a_diagnosis() -> None:
    for condition in REGISTRY.conditions:
        for level in explanations.LEVELS:
            text = explanations.describe(EXAMPLE, condition, level)["text"].lower()
            assert "not a diagnos" in text
            # The only "you have" allowed is inside "doesn't mean you have".
            assert text.count("you have") == text.count("doesn't mean you have")
            assert "diagnosed with" not in text


def test_limitations_always_travel_with_the_text() -> None:
    limits = ["This is not a diagnosis.", "Self-reported data."]
    out = explanations.describe(EXAMPLE, PLAIN, limitations=limits)
    assert out["limitations"] == limits
    assert "limitations below" in out["text"]


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="level"):
        explanations.describe(EXAMPLE, PLAIN, level="child")


def test_accepts_explain_one_output_unsorted() -> None:
    shuffled = list(reversed(EXAMPLE["contributions"]))
    assert (
        explanations.describe(shuffled, PLAIN)["text"]
        == explanations.describe(EXAMPLE, PLAIN)["text"]
    )


# ---- every model input has proper wording ---------------------------------------------------

TYPICAL = {
    "bmi": 27.5,
    "age_band": 13,
    "gen_health": 4,
    "income_band": 7,
    "education_band": 1,
    "risk_factor_count": 2,
    "lifestyle_score": 3,
    "sex": "female",
    "smoker": "some_days",
    "bmi_band": "obese",
}


@pytest.mark.parametrize("feature", features.BASE_PREDICTORS)
@pytest.mark.parametrize("level", explanations.LEVELS)
def test_every_predictor_has_a_real_phrase(feature, level) -> None:
    values = [True, False, None] if feature in features.BINARY else [TYPICAL[feature], None]
    for value in values:
        text = explanations.phrase(feature, value, level)
        assert "your answer about" not in text  # the generic fallback
        assert "_" not in text  # no code names
        assert text == text.strip() and text


def test_comorbidity_predictors_use_registry_names() -> None:
    other = RENAMED
    assert (
        explanations.phrase(other.outcome_column, True, "patient") == f"having {other.patient_name}"
    )
    assert explanations.phrase(other.outcome_column, False, "clinician").startswith(
        "no history of "
    )


def test_labels_cover_every_coded_value() -> None:
    assert set(explanations.AGE_LABELS) == set(range(1, 14))
    assert set(explanations.INCOME_LABELS) == set(range(1, 8))
    assert set(explanations.GEN_HEALTH_LABELS) == set(range(1, 6))
    assert set(explanations.EDUCATION_LABELS) == set(range(1, 5))


# ---- end to end with a real explainer (H14) ------------------------------------------------


def test_end_to_end_from_a_trained_model() -> None:
    split = evaluate.split_condition(signal_frame(n=3000), PLAIN.id)
    fit, _ = evaluate.validation_split(split)
    model = logreg.build_logreg(split.predictors).fit(fit.x, fit.y)
    explainer = explain.build_explainer(model, split.x_train)
    oldest = split.x_test[split.x_test["age_band"] == split.x_test["age_band"].max()].iloc[[0]]
    out = explanations.describe(explainer.explain(oldest), PLAIN, "patient")
    assert out["sentences"][0].startswith("Your age group (80 or older) is the biggest factor")
    assert np.isfinite([f["contribution"] for f in out["factors"]]).all()

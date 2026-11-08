"""H46: turn a SHAP explanation (H14) into plain-language sentences, from fixed templates.

    describe(explanation, condition, level="patient")      # or level="clinician"

Patient level, for example:

    "Your BMI (31.2) is the biggest factor raising this estimate, followed by your age group
    (60-64). Being physically active lowers it. This is an estimate from survey data, not a
    diagnosis: it doesn't mean you have or will get <condition>. Please read the
    limitations below."

Clinician level names the same factors with their SHAP values in log-odds.

Rules every output follows:

* Only factors that really moved the estimate are mentioned (|contribution| ≥ 0.05 log-odds);
  the biggest first, at most two each way for patients and three for clinicians.
* Wording for a condition comes from the registry (`plain_name`), so a new condition needs
  no new code. Sensitive conditions get softer wording and a sentence about support.
* It never states a diagnosis, always says it isn't one, and always carries the
  limitations (H15) for the UI to show.
* Deterministic: same input, same words. No AI involved (an optional AI rewording is H45's
  job and must keep exactly these factors).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .registry import Condition, Registry, load_registry

LEVELS = ("patient", "clinician")
# Contributions smaller than this (in log-odds) barely move the estimate; don't mention them.
NEGLIGIBLE = 0.05
MAX_FACTORS = {"patient": 2, "clinician": 3}

AGE_LABELS = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80 or older",
}
INCOME_LABELS = {
    1: "under $15k",
    2: "$15k-$25k",
    3: "$25k-$35k",
    4: "$35k-$50k",
    5: "$50k-$100k",
    6: "$100k-$200k",
    7: "$200k or more",
}
GEN_HEALTH_LABELS = {1: "excellent", 2: "very good", 3: "good", 4: "fair", 5: "poor"}
EDUCATION_LABELS = {
    1: "did not finish high school",
    2: "high school graduate",
    3: "some college or technical school",
    4: "college graduate",
}
SMOKER_TEXT = {
    # value: (patient, clinician)
    "daily": ("smoking every day", "current daily smoker"),
    "some_days": ("smoking on some days", "current occasional smoker"),
    "former": ("being a former smoker", "former smoker"),
    "never": ("never having smoked", "never smoker"),
}
# yes/no answers: (patient if yes, patient if no, clinician if yes, clinician if no, topic)
YES_NO_TEXT = {
    "phys_active": (
        "being physically active",
        "not being physically active",
        "physically active in the past 30 days",
        "no physical activity in the past 30 days",
        "physical activity",
    ),
    "heavy_alcohol": (
        "heavy drinking",
        "not drinking heavily",
        "heavy alcohol use",
        "no heavy alcohol use",
        "alcohol",
    ),
    "fruit_daily": (
        "eating fruit every day",
        "not eating fruit every day",
        "daily fruit",
        "fruit less than daily",
        "fruit",
    ),
    "veg_daily": (
        "eating vegetables every day",
        "not eating vegetables every day",
        "daily vegetables",
        "vegetables less than daily",
        "vegetables",
    ),
    "high_chol": (
        "having been told you have high cholesterol",
        "not having high cholesterol",
        "high cholesterol",
        "no high cholesterol",
        "cholesterol",
    ),
}
TOPICS = {
    "bmi": "BMI",
    "age_band": "age",
    "sex": "sex",
    "smoker": "smoking",
    "gen_health": "general health",
    "income_band": "household income",
    "education_band": "education",
    "bmi_band": "weight",
    "risk_factor_count": "risk factors",
    "lifestyle_score": "healthy habits",
}
RISK_FACTORS_TOTAL = 5  # smoking, inactivity, heavy drinking, high cholesterol, obesity (H09)
HABITS_TOTAL = 5  # active, fruit, vegetables, no heavy drinking, not smoking (H09)

SUPPORT_SENTENCE = (
    "If you're finding things hard, talking with a health professional or someone you "
    "trust can help."
)


# --------------------------------------------------------------------------
# One factor → a phrase
# --------------------------------------------------------------------------


def _int(value: Any) -> int:
    return int(round(float(value)))


def _topic(feature: str, registry: Registry) -> str:
    if feature in YES_NO_TEXT:
        return YES_NO_TEXT[feature][4]
    other = _condition_for_column(feature, registry)
    if other is not None:
        return other.plain_name
    return TOPICS.get(feature, feature.replace("_", " "))


def _condition_for_column(column: str, registry: Registry) -> Condition | None:
    return next((c for c in registry.conditions if c.outcome_column == column), None)


def phrase(feature: str, value: Any, level: str, registry: Registry | None = None) -> str:
    """A short phrase for one factor and the person's answer, e.g. "your BMI (31.2)"."""
    registry = registry or load_registry()
    patient = level == "patient"

    if value is None:
        topic = _topic(feature, registry)
        return f"not answering the question about {topic}" if patient else f"{topic}: not answered"

    if feature in YES_NO_TEXT:
        yes_p, no_p, yes_c, no_c, _ = YES_NO_TEXT[feature]
        return (yes_p if value else no_p) if patient else (yes_c if value else no_c)

    other = _condition_for_column(feature, registry)
    if other is not None:  # another condition used as a predictor (registry comorbidity)
        if patient:
            return f"having {other.plain_name}" if value else f"not having {other.plain_name}"
        name = other.display_name if other.display_name.isupper() else other.display_name.lower()
        return f"history of {name}" if value else f"no history of {name}"

    if feature == "bmi":
        return f"your BMI ({float(value):.1f})" if patient else f"BMI {float(value):.1f}"
    if feature == "age_band":
        label = AGE_LABELS.get(_int(value), str(value))
        return f"your age group ({label})" if patient else f"age {label}"
    if feature == "sex":
        return f"being {value}" if patient else f"sex: {value}"
    if feature == "smoker":
        texts = SMOKER_TEXT.get(str(value), (f"smoking: {value}", f"smoking: {value}"))
        return texts[0] if patient else texts[1]
    if feature == "gen_health":
        label = GEN_HEALTH_LABELS.get(_int(value), str(value))
        return f"rating your general health as {label}" if patient else f"self-rated health {label}"
    if feature == "income_band":
        label = INCOME_LABELS.get(_int(value), str(value))
        return f"your household income ({label})" if patient else f"household income {label}"
    if feature == "education_band":
        label = EDUCATION_LABELS.get(_int(value), str(value))
        return f"your education ({label})" if patient else f"education: {label}"
    if feature == "bmi_band":
        return f"your weight category ({value})" if patient else f"BMI category {value}"
    if feature == "risk_factor_count":
        n = _int(value)
        if patient:
            return f"the number of risk factors in your answers ({n} of {RISK_FACTORS_TOTAL})"
        return f"{n} of {RISK_FACTORS_TOTAL} risk factors"
    if feature == "lifestyle_score":
        n = _int(value)
        if patient:
            return f"your healthy-habits score ({n} of {HABITS_TOTAL})"
        return f"lifestyle score {n}/{HABITS_TOTAL}"

    name = feature.replace("_", " ")
    return f"your answer about {name}" if patient else f"{name}: {value}"


# --------------------------------------------------------------------------
# Sentences
# --------------------------------------------------------------------------


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _join(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _contributions(explanation: dict[str, Any] | Sequence[dict[str, Any]]) -> list[dict]:
    items = explanation["contributions"] if isinstance(explanation, dict) else explanation
    return sorted(items, key=lambda c: abs(c["contribution"]), reverse=True)


def disclaimer(condition: Condition, level: str) -> str:
    if level == "clinician":
        return (
            "Population-based estimate from self-reported survey data (CDC BRFSS); not a "
            "diagnostic result. See the model card for performance and limitations."
        )
    return (
        "This is an estimate from survey data, not a diagnosis: it doesn't mean you have or "
        f"will get {condition.plain_name}. Please read the limitations below."
    )


def describe(
    explanation: dict[str, Any] | Sequence[dict[str, Any]],
    condition: Condition,
    level: str = "patient",
    limitations: Sequence[str] = (),
    registry: Registry | None = None,
) -> dict[str, Any]:
    """Plain-language version of an explanation from `Explainer.explain` / `explain_one`.

    Returns `{level, condition, text, sentences, factors, disclaimer, limitations}`. `factors`
    lists exactly the factors the text mentions, so any later rewording can be checked
    against it.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    registry = registry or load_registry()
    limit = MAX_FACTORS[level]
    items = _contributions(explanation)
    raising = [c for c in items if c["contribution"] >= NEGLIGIBLE][:limit]
    lowering = [c for c in items if c["contribution"] <= -NEGLIGIBLE][:limit]

    def words(c: dict[str, Any]) -> str:
        return phrase(c["feature"], c["value"], level, registry)

    if level == "clinician":
        sentences = _clinician(raising, lowering, words)
    elif condition.sensitive:
        sentences = _patient_gentle(raising, lowering, words)
    else:
        sentences = _patient(raising, lowering, words)
    closing = disclaimer(condition, level)
    sentences.append(closing)

    factors = [
        {
            "feature": c["feature"],
            "phrase": words(c),
            "direction": "raises" if c["contribution"] > 0 else "lowers",
            "contribution": float(c["contribution"]),
        }
        for c in (*raising, *lowering)
    ]
    return {
        "level": level,
        "condition": condition.id,
        "text": " ".join(sentences),
        "sentences": sentences,
        "factors": factors,
        "disclaimer": closing,
        "limitations": list(limitations),
    }


def _patient(raising: list, lowering: list, words) -> list[str]:
    sentences = []
    if raising:
        first = f"{_cap(words(raising[0]))} is the biggest factor raising this estimate"
        if len(raising) > 1:
            first += f", followed by {_join([words(c) for c in raising[1:]])}"
        sentences.append(first + ".")
    if lowering:
        verb = "lowers it" if len(lowering) == 1 else "lower it"
        sentences.append(f"{_cap(_join([words(c) for c in lowering]))} {verb}.")
    if not raising and not lowering:
        sentences.append(
            "None of your answers moved this estimate much; it's close to the average for "
            "people in the survey."
        )
    elif not raising:
        sentences.insert(0, "None of your answers raised this estimate much.")
    return sentences


def _patient_gentle(raising: list, lowering: list, words) -> list[str]:
    """Softer wording for sensitive conditions: "linked with", no "biggest factor"."""
    sentences = []
    if raising:
        listed = _join([words(c) for c in raising])
        sentences.append(
            f"The answers most linked with a higher estimate were {listed}."
            if len(raising) > 1
            else f"The answer most linked with a higher estimate was {listed}."
        )
    if lowering:
        sentences.append(
            f"{_cap(_join([words(c) for c in lowering]))} "
            f"{'was' if len(lowering) == 1 else 'were'} linked with a lower one."
        )
    if not raising and not lowering:
        sentences.append("None of your answers stood out in this estimate.")
    sentences.append(SUPPORT_SENTENCE)
    return sentences


def _clinician(raising: list, lowering: list, words) -> list[str]:
    def listed(group: list) -> str:
        return ", ".join(f"{words(c)} ({c['contribution']:+.2f})" for c in group)

    sentences = []
    sentences.append(
        f"Factors increasing the estimate: {listed(raising)}."
        if raising
        else "No factor materially increased the estimate."
    )
    sentences.append(
        f"Factors decreasing the estimate: {listed(lowering)}."
        if lowering
        else "No factor materially decreased the estimate."
    )
    sentences.append("Values are SHAP contributions in log-odds relative to the model baseline.")
    return sentences

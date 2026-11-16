"""H19: turn one set of answers into a risk estimate, its factors and its sentences.

The pieces all come from ``ml/``: the explainer (H14) produces the probability and the SHAP
contributions, ``explanations.describe`` (H46) turns those into sentences, and the release gate
(H50) decides whether an individual estimate may be shown at all.

Two rules this module exists to enforce:

1. **The gate is server-side.** A condition that is not ``prediction_enabled`` in
   ``ml/models/index.json`` gets 409 and the gate's own reason - never a number - however the
   API is called. The UI is not trusted to hide it.
2. **No model, no guess.** Missing artifacts answer 503 with the command that creates them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ml.src import explanations
from ml.src import features as ml_features
from ml.src.registry import Condition, Registry

from . import catalog, featurespec
from .errors import AppError
from .models_store import TRAIN_COMMAND, LoadedModel, ModelStore

# Presentation only: the band never changes the number, it only labels it.
RISK_BANDS = ((0.10, "low"), (0.25, "moderate"), (0.50, "high"))
HIGHEST_BAND = "very high"
GATE_CODE = "analytics_only"
UNAVAILABLE_CODE = "models_unavailable"
UNAVAILABLE_MESSAGE = (
    f"No trained models are available yet. Run `{TRAIN_COMMAND}` to create them "
    "(ml/models is not committed to git)."
)


def risk_band(probability: float) -> str:
    """A coarse label for a probability, for the UI to colour."""
    return next((name for edge, name in RISK_BANDS if probability < edge), HIGHEST_BAND)


def known_condition(registry: Registry, condition_id: str) -> Condition:
    """The registry entry, or 422. Input is never interpolated into a path before this."""
    if condition_id not in registry.ids:
        raise AppError(
            422, "invalid_parameter", f"condition must be one of: {', '.join(registry.ids)}"
        )
    return registry.get(condition_id)


def usable_model(store: ModelStore, gate: dict[str, Any], condition: Condition) -> LoadedModel:
    """The loaded model for a condition, or the 409/503 that says why there is no estimate.

    The gate decision is checked first, so a blocked condition answers 409 with the gate's
    reason whether or not its artifacts happen to be on this machine. Only when the index has
    no decision at all (no artifacts, nothing trained) does 503 take over.
    """
    enabled = gate.get("prediction_enabled")
    if enabled is False:
        raise AppError(409, GATE_CODE, str(gate.get("reason") or _default_gate_reason(condition)))
    model = store.get(condition.id)
    if model is None:
        raise AppError(503, UNAVAILABLE_CODE, UNAVAILABLE_MESSAGE)
    return model


def _default_gate_reason(condition: Condition) -> str:
    return (
        f"Population analytics only: individual estimates are not enabled for "
        f"{condition.display_name}."
    )


def validate_answers(model: LoadedModel, raw: dict[str, Any]) -> dict[str, Any]:
    """Check the submitted answers against the model's own feature spec (ranges, categories)."""
    try:
        parsed = model.answers.model_validate(raw)
    except ValidationError as exc:
        raise AppError(422, "validation_error", featurespec.problems(exc)) from exc
    return parsed.model_dump()


def explain_answers(model: LoadedModel, answers: dict[str, Any]) -> dict[str, Any]:
    """The explainer's own output for one row: base value, margin, probability, contributions."""
    frame = featurespec.answer_frame(answers, model.feature_spec)
    row = ml_features.model_frame(ml_features.add_features(frame), model.features)
    return model.explainer.explain(row)


def estimate(
    model: LoadedModel,
    condition: Condition,
    gate: dict[str, Any],
    answers: dict[str, Any],
    registry: Registry,
) -> dict[str, Any]:
    """The full ``/predict`` payload for one validated set of answers."""
    explanation = explain_answers(model, answers)
    probability = float(explanation["probability"])
    limits = list(model.limitations)
    described = {
        level: explanations.describe(
            explanation, condition, level=level, limitations=limits, registry=registry
        )
        for level in explanations.LEVELS
    }
    return {
        "condition": condition.id,
        "display_name": condition.display_name,
        "risk": round(probability, 4),
        "risk_band": risk_band(probability),
        "signal_strength": gate.get("signal_strength"),
        "model_name": model.model_name,
        "contributions": [
            {
                "feature": item["feature"],
                "label": featurespec.feature_label(item["feature"], registry),
                "value": item["value"],
                "contribution": round(float(item["contribution"]), 4),
            }
            for item in explanation["contributions"]
        ],
        "explanation": described,
        "disclaimer": described["patient"]["disclaimer"],
        "limitations": limits,
    }


def predict(
    store: ModelStore,
    registry: Registry,
    models_dir: Path,
    condition_id: str,
    raw_answers: dict[str, Any],
) -> dict[str, Any]:
    """Validate, gate-check and estimate. The single entry point both endpoints use."""
    condition = known_condition(registry, condition_id)
    gate = catalog.gate_entry(models_dir, condition.id)
    model = usable_model(store, gate, condition)
    answers = validate_answers(model, raw_answers)
    return estimate(model, condition, gate, answers, registry)

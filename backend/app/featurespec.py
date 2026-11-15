"""Request validation and readable labels, both driven by a model's own ``feature_spec``.

``ml/models/<condition>/model_meta.json`` lists every input the trained pipeline expects, its
kind and its allowed values. The API builds its Pydantic request model from that spec rather
than hard-coding feature names, so retraining with different predictors needs no API change.

Derived features (``bmi_band``, ``risk_factor_count``, ``lifestyle_score``) are never asked for:
``ml.src.features.add_features`` computes them from the answers.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from ml.src import explanations
from ml.src.registry import Registry

# The few answers a caller must give; everything else may be left out and is treated as
# "not answered" (the pipeline imputes it, and the explanation says it was not answered).
REQUIRED_INPUTS = ("age_band", "sex", "bmi")

_NUMBER = "number"
_CATEGORY = "category"
_YES_NO = "yes_no"


class FeatureSpecError(ValueError):
    """The saved feature spec is not in the shape this API understands."""


def answerable(feature_spec: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    """The spec entries a caller supplies: everything that is not derived from the others."""
    return tuple(entry for entry in feature_spec if not entry.get("derived"))


def _annotation(entry: dict[str, Any]) -> Any:
    kind, name = entry.get("kind"), entry.get("name")
    if kind == _YES_NO:
        return bool
    if kind == _CATEGORY:
        allowed = entry.get("allowed")
        if not allowed:
            return str
        return Literal[tuple(str(value) for value in allowed)]
    if kind == _NUMBER:
        bounds = entry.get("range")
        if not bounds:
            return float
        low, high = bounds
        number = int if isinstance(low, int) and isinstance(high, int) else float
        return Annotated[number, Field(ge=low, le=high)]
    raise FeatureSpecError(f"feature {name!r} has unknown kind {kind!r}")


def request_model(condition_id: str, feature_spec: tuple[dict[str, Any], ...]) -> type[BaseModel]:
    """A strict Pydantic model for one condition's answers. Unknown fields are rejected."""
    fields: dict[str, Any] = {}
    for entry in answerable(feature_spec):
        name = entry.get("name")
        if not isinstance(name, str):
            raise FeatureSpecError(f"feature spec entry without a name: {entry!r}")
        annotation = _annotation(entry)
        required = name in REQUIRED_INPUTS
        fields[name] = (annotation, ...) if required else (annotation | None, None)
    return create_model(
        f"{condition_id}_answers",
        __config__=ConfigDict(extra="forbid", frozen=True),
        **fields,
    )


def problems(error: ValidationError) -> str:
    """A ValidationError as one line, in the same shape as the app's 422 handler produces."""
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors())


def answer_frame(answers: dict[str, Any], feature_spec: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    """One canonical row, typed so that ``add_features`` and the pipeline see what they expect.

    Numbers become float64 (missing = NaN), yes/no becomes pandas' nullable boolean, and
    categories stay object columns. Nothing is guessed: a missing answer stays missing.
    """
    kinds = {entry["name"]: entry.get("kind") for entry in answerable(feature_spec)}
    columns = {}
    for name, kind in kinds.items():
        value = answers.get(name)
        if kind == _YES_NO:
            columns[name] = pd.Series([value], dtype="boolean")
        elif kind == _NUMBER:
            columns[name] = pd.Series([np.nan if value is None else float(value)], dtype="float64")
        else:
            columns[name] = pd.Series([value], dtype=object)
    return pd.DataFrame(columns)


def feature_label(feature: str, registry: Registry) -> str:
    """A readable label for a canonical feature name ("bmi" -> "BMI", "age_band" -> "Age")."""
    yes_no = explanations.YES_NO_TEXT.get(feature)
    if yes_no is not None:
        label = yes_no[4]
    else:
        condition = next((c for c in registry.conditions if c.outcome_column == feature), None)
        label = (
            condition.display_name
            if condition is not None
            else explanations.TOPICS.get(feature, feature.replace("_", " "))
        )
    return label[:1].upper() + label[1:]

"""Trained artifacts, loaded once at startup and then only read (H19).

``ml/models/`` is gitignored, so a clone has no artifacts at all: that is not an error. The
store is simply empty, every other endpoint keeps working, and ``/predict`` answers 503 with
the command that creates them. The backend never trains and never loads a model per request.

Each condition folder holds (written by ``python -m ml.src.train``):

    explainer.joblib     the chosen pipeline plus its SHAP explainer (H14)
    model_meta.json      the features and their allowed values (what we validate against)
    metrics.json         which model was chosen
    limitations.json     the plain-language limitations the UI shows (H15)
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ml.src.registry import Registry

from .catalog import read_json
from .featurespec import FeatureSpecError, request_model

logger = logging.getLogger(__name__)

EXPLAINER_FILE = "explainer.joblib"
UNKNOWN_MODEL = "unknown"
META_FILE = "model_meta.json"
METRICS_FILE = "metrics.json"
LIMITATIONS_FILE = "limitations.json"
TRAIN_COMMAND = "python -m ml.src.train"
# Which model is in use comes from metrics.json["best_model"], whatever it names: the backend
# never has a list of model names of its own, so a new model in the pipeline needs no change
# here. _KIND_NAMES is only a last resort when metrics.json cannot be read, and it describes
# the explainer's own two maths kinds (linear or tree), not the pipeline's model line-up.
_KIND_NAMES = {"linear": "logreg", "tree": "tree-ensemble"}


@dataclass(frozen=True)
class LoadedModel:
    """One condition's artifacts, ready to answer a request."""

    condition_id: str
    model_name: str
    features: tuple[str, ...]
    feature_spec: tuple[dict[str, Any], ...]
    explainer: Any  # ml.src.explain.Explainer (imported lazily, see ``load``)
    answers: type[BaseModel]  # the request model built from feature_spec
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelStore:
    """What was loaded at startup. Empty when no artifacts are present."""

    models: Mapping[str, LoadedModel] = field(default_factory=dict)
    loaded_from: Path | None = None

    def get(self, condition_id: str) -> LoadedModel | None:
        return self.models.get(condition_id)

    @property
    def condition_ids(self) -> tuple[str, ...]:
        return tuple(self.models)

    @property
    def available(self) -> bool:
        return bool(self.models)


def _limitations(folder: Path) -> tuple[str, ...]:
    data = read_json(folder / LIMITATIONS_FILE) or {}
    items = data.get("limitations")
    return tuple(str(i) for i in items) if isinstance(items, list) else ()


def _model_name(folder: Path, explainer: Any) -> str:
    metrics = read_json(folder / METRICS_FILE) or {}
    best = metrics.get("best_model")
    if isinstance(best, str) and best:
        return best
    return _KIND_NAMES.get(getattr(explainer, "kind", ""), UNKNOWN_MODEL)


def load_one(folder: Path, condition_id: str) -> LoadedModel | None:
    """One condition's artifacts, or None when they are absent or unusable."""
    meta = read_json(folder / META_FILE)
    explainer_path = folder / EXPLAINER_FILE
    if meta is None or not explainer_path.exists():
        return None

    features = meta.get("features")
    spec = meta.get("feature_spec")
    if not isinstance(features, list) or not isinstance(spec, list):
        logger.warning("%s: model_meta.json has no features/feature_spec; skipping", condition_id)
        return None

    import joblib  # local: only needed when artifacts exist

    try:
        explainer = joblib.load(explainer_path)
        answers = request_model(condition_id, tuple(spec))
    except FeatureSpecError as exc:
        logger.warning("%s: unusable feature spec (%s)", condition_id, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        # Unpickling raises almost anything for a truncated, stale or foreign file, and an
        # ImportError when the saved pipeline needs a library we do not have. Startup must
        # survive all of it: the condition simply has no model and /predict answers 503.
        logger.warning("%s: cannot load %s (%s)", condition_id, explainer_path, exc)
        return None

    return LoadedModel(
        condition_id=condition_id,
        model_name=_model_name(folder, explainer),
        features=tuple(str(f) for f in features),
        feature_spec=tuple(spec),
        explainer=explainer,
        answers=answers,
        limitations=_limitations(folder),
    )


def load(models_dir: Path, registry: Registry) -> ModelStore:
    """Load every condition that has usable artifacts. Never raises; logs what it found."""
    found = {}
    for condition_id in registry.ids:
        model = load_one(models_dir / condition_id, condition_id)
        if model is not None:
            found[condition_id] = model
    if found:
        logger.info("models: loaded %s from %s", ", ".join(found), models_dir)
    else:
        logger.info("models: none found under %s (run %s)", models_dir, TRAIN_COMMAND)
    return ModelStore(models=found, loaded_from=models_dir)

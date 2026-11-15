"""A tiny but real set of trained artifacts, built in a temp folder for the tests.

The tests never depend on ``ml/models/`` (gitignored, and absent on a fresh clone or in CI).
Instead they train a very small logistic regression on generated rows and save exactly the
files ``ml.src.train`` saves, so the backend loads the same shapes it will see in production.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.src import explain, features, train
from ml.src.registry import load_registry

ROWS = 240
SEED = 7


def _frame(outcome_column: str, predictors: tuple[str, ...], rows: int = ROWS) -> pd.DataFrame:
    """Rows with a real signal in bmi and age_band, so the fitted model is not degenerate."""
    rng = np.random.default_rng(SEED)
    bmi = rng.normal(28, 5, rows).clip(15, 55)
    age_band = rng.integers(1, 14, rows)
    data: dict[str, Any] = {
        "bmi": bmi,
        "age_band": age_band,
        "gen_health": rng.integers(1, 6, rows),
        "income_band": rng.integers(1, 8, rows),
        "education_band": rng.integers(1, 5, rows),
        "sex": rng.choice(["male", "female"], rows),
        "smoker": rng.choice(["daily", "some_days", "former", "never"], rows),
        "phys_active": rng.random(rows) > 0.4,
        "heavy_alcohol": rng.random(rows) > 0.8,
        "fruit_daily": rng.random(rows) > 0.5,
        "veg_daily": rng.random(rows) > 0.5,
        "high_chol": rng.random(rows) > 0.6,
    }
    logit = -6.0 + 0.12 * bmi + 0.18 * age_band
    data[outcome_column] = rng.random(rows) < 1 / (1 + np.exp(-logit))
    frame = features.add_features(pd.DataFrame(data))
    missing = [c for c in predictors if c not in frame.columns]
    for column in missing:  # another condition's outcome, opted in via the registry
        frame[column] = rng.random(rows) > 0.7
    return frame


def write_condition(
    models_dir: Path,
    condition_id: str,
    *,
    best_model: str = "logreg",
    extra_models: tuple[str, ...] = (),
    limitations: tuple[str, ...] = ("Trained on self-reported survey answers.",),
) -> Path:
    """Train and save a small model for one condition, exactly as ``ml.src.train`` would.

    ``best_model`` is written into metrics.json as the chosen model and ``extra_models`` adds
    further entries there, so the backend can be tested against a pipeline that offers more
    than two models.
    """
    registry = load_registry()
    condition = registry.get(condition_id)
    predictors = features.predictors_for(condition_id)
    frame = _frame(condition.outcome_column, predictors)

    x = features.model_frame(frame, predictors)
    y = frame[condition.outcome_column].astype(bool)
    model = Pipeline(
        [
            ("prep", features.build_preprocessor(predictors)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    model.fit(x, y)

    folder = models_dir / condition_id
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / f"{best_model}.joblib")
    joblib.dump(explain.build_explainer(model, x), folder / explain.EXPLAINER_FILE)

    names = (best_model, *extra_models)
    (folder / "metrics.json").write_text(
        json.dumps(
            {
                "condition": condition_id,
                "best_model": best_model,
                "selection": {"criterion": "test fixture", "validation_roc_auc": {}},
                "models": {name: {"metrics": {"default": {"roc_auc": 0.78}}} for name in names},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    meta = {
        "condition": condition_id,
        "display_name": condition.display_name,
        "outcome_column": condition.outcome_column,
        "sensitive": condition.sensitive,
        "features": list(predictors),
        "feature_spec": train.feature_spec(predictors),
    }
    (folder / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (folder / "limitations.json").write_text(
        json.dumps({"condition": condition_id, "limitations": list(limitations)}, indent=2),
        encoding="utf-8",
    )
    return folder


def write_index(models_dir: Path, decisions: dict[str, dict[str, Any]]) -> Path:
    """An ``index.json`` holding the release-gate decision per condition."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "index.json"
    path.write_text(json.dumps({"conditions": decisions}, indent=2), encoding="utf-8")
    return path


def gate_decision(
    *, enabled: bool, strength: str | None = "moderate", reason: str = ""
) -> dict[str, Any]:
    return {
        "prediction_enabled": enabled,
        "signal_strength": strength,
        "reason": reason or ("Individual estimates enabled." if enabled else "Analytics only."),
    }

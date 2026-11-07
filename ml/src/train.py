"""H13: one command trains every model for every condition and saves what the backend loads.

    python -m ml.src.train                       # every condition in the registry
    python -m ml.src.train --condition ID        # just one
    python -m ml.src.train --fast                # quick run (small search) for checking

For each condition (H49 registry) it splits the data (H10), trains the logistic regression
baseline (H11) and XGBoost (H12) on exactly the same rows, picks the best model on the
**validation** set (never the test set), and writes:

    ml/models/<condition>/logreg.joblib       fitted pipeline (preprocessor + model)
    ml/models/<condition>/xgboost.joblib
    ml/models/<condition>/metrics.json        every model's metrics, best_model, how it was chosen
    ml/models/<condition>/model_meta.json     what the backend needs to call the model safely
    ml/models/index.json                      every trained condition (merged, not overwritten)

The model files are gitignored: everyone regenerates them with this command.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import config, evaluate, features, schema
from .registry import Condition, RegistryError, load_registry

MODELS_DIR = config.REPO_ROOT / "ml" / "models"
INDEX_FILE = "index.json"
# Simplest first: on a tie the earlier (easier to explain) model wins.
MODEL_ORDER = ("logreg", "xgboost")
# Validation ROC-AUC differences smaller than this count as a tie.
TIE_MARGIN = 0.005
SELECTION_CRITERION = (
    f"highest validation ROC-AUC; models within {TIE_MARGIN} of the best count as tied "
    "and the simpler one (logreg before xgboost) is chosen"
)


@dataclass(frozen=True)
class Settings:
    """How hard to search. `fast()` is for smoke tests and quick checks."""

    logreg_cv: int = 5
    xgb_n_iter: int = 12
    xgb_cv: int = 3
    xgb_tuning_rows: int = 100_000

    @classmethod
    def fast(cls) -> Settings:
        return cls(logreg_cv=2, xgb_n_iter=2, xgb_cv=2, xgb_tuning_rows=20_000)


def _trainers(settings: Settings) -> dict[str, Callable[[evaluate.Split], Any]]:
    """Model name -> function(split) -> result with .model and .summary(). Imported lazily so
    a missing optional library only matters for the model that needs it."""

    def logreg(split: evaluate.Split) -> Any:
        from . import logreg as module

        return module.train_logreg(split, cv=settings.logreg_cv)

    def xgboost(split: evaluate.Split) -> Any:
        from . import boosting

        return boosting.train_xgb(
            split,
            n_iter=settings.xgb_n_iter,
            cv=settings.xgb_cv,
            max_tuning_rows=settings.xgb_tuning_rows,
        )

    return {"logreg": logreg, "xgboost": xgboost}


# --------------------------------------------------------------------------
# Choosing the best model
# --------------------------------------------------------------------------


def select_best(validation_auc: dict[str, float], order: tuple[str, ...] = MODEL_ORDER) -> str:
    """Best validation ROC-AUC wins; near-ties go to the simpler model (earlier in `order`)."""
    if not validation_auc:
        raise ValueError("no models to choose from")
    top = max(validation_auc.values())
    tied = [name for name in validation_auc if validation_auc[name] >= top - TIE_MARGIN]
    return min(tied, key=lambda name: order.index(name) if name in order else len(order))


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def _version(module_name: str) -> str | None:
    try:
        return __import__(module_name).__version__
    except ImportError:
        return None


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=config.REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def feature_spec(predictors: tuple[str, ...]) -> list[dict[str, Any]]:
    """For each model input: its kind and allowed values, in the order the model expects."""
    spec = []
    for name in predictors:
        entry: dict[str, Any] = {"name": name}
        if name in features.CATEGORICAL:
            entry["kind"] = "category"
            if name in schema.CATEGORIES:
                entry["allowed"] = sorted(schema.CATEGORIES[name])
            elif name == "bmi_band":
                entry["allowed"] = list(features.BMI_BANDS)
        elif name in features.NUMERIC:
            entry["kind"] = "number"
            if name in schema.INT_RANGES:
                entry["range"] = list(schema.INT_RANGES[name])
            elif name == "bmi":
                entry["range"] = list(schema.BMI_RANGE)
        else:
            entry["kind"] = "yes_no"
        entry["derived"] = name in features.DERIVED_COLUMNS
        spec.append(entry)
    return spec


def model_meta(
    condition: Condition, split: evaluate.Split, data_file: str, seed: int
) -> dict[str, Any]:
    return {
        "condition": condition.id,
        "display_name": condition.display_name,
        "outcome_column": condition.outcome_column,
        "sensitive": condition.sensitive,
        "data_source": condition.source,
        "data_file": data_file,
        "rows": {
            "train": len(split.y_train),
            "test": len(split.y_test),
            "positives_train": int(split.y_train.sum()),
            "positives_test": int(split.y_test.sum()),
        },
        "features": list(split.predictors),
        "feature_spec": feature_spec(split.predictors),
        "how_to_predict": (
            "Build a one-row frame of canonical columns, call features.add_features, then "
            "features.model_frame(df, features) and model.predict_proba(...)[:, 1]."
        ),
        "seed": seed,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": _version("xgboost"),
            "joblib": joblib.__version__,
        },
        "git_sha": git_sha(),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------
# Training one condition
# --------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    # allow_nan=False: NaN/Infinity aren't valid JSON, so fail here rather than in the backend.
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train_condition(
    df: pd.DataFrame,
    condition: Condition,
    out_dir: Path,
    models: tuple[str, ...] = MODEL_ORDER,
    settings: Settings | None = None,
    data_file: str = "",
    seed: int = config.RANDOM_SEED,
) -> dict[str, Any]:
    """Train, compare and save every requested model for one condition. Returns its index entry."""
    settings = settings or Settings()
    trainers = _trainers(settings)
    unknown = [m for m in models if m not in trainers]
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(unknown)}")

    split = evaluate.split_condition(df, condition.id, seed=seed)
    _, val = evaluate.validation_split(split, seed=seed)
    folder = out_dir / condition.id
    folder.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}
    validation_auc: dict[str, float] = {}
    for name in models:
        result = trainers[name](split)
        joblib.dump(result.model, folder / f"{name}.joblib")
        summaries[name] = result.summary()
        validation_auc[name] = evaluate.evaluate(result.model, val.x, val.y)["roc_auc"]

    best = select_best(validation_auc)
    _write_json(
        folder / "metrics.json",
        {
            "condition": condition.id,
            "best_model": best,
            "selection": {"criterion": SELECTION_CRITERION, "validation_roc_auc": validation_auc},
            "models": summaries,
        },
    )
    _write_json(folder / "model_meta.json", model_meta(condition, split, data_file, seed))

    best_test = summaries[best]["metrics"]["default"]
    return {
        "status": "trained",
        "display_name": condition.display_name,
        "tier": condition.tier,
        "sensitive": condition.sensitive,
        "best_model": best,
        "models": list(models),
        "test_roc_auc": best_test["roc_auc"],
        "test_brier": best_test["brier"],
        "test_brier_no_skill": best_test["brier_no_skill"],
        "path": condition.id,
    }


def update_index(out_dir: Path, entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Merge new entries into index.json, keeping conditions that weren't retrained."""
    path = out_dir / INDEX_FILE
    index: dict[str, Any] = {"conditions": {}}
    if path.exists():
        index = json.loads(path.read_text(encoding="utf-8"))
    index["conditions"].update(entries)
    index["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    index["git_sha"] = git_sha()
    _write_json(path, index)
    return index


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train every model for every condition.")
    parser.add_argument("--condition", help="one condition id (default: every condition)")
    parser.add_argument("--data", type=Path, default=evaluate.DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=MODELS_DIR)
    parser.add_argument(
        "--models", default=",".join(MODEL_ORDER), help="comma-separated (default: all)"
    )
    parser.add_argument("--fast", action="store_true", help="small search, for quick checks")
    args = parser.parse_args(argv)

    registry = load_registry()
    try:
        conditions = [registry.get(args.condition)] if args.condition else registry.conditions
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    settings = Settings.fast() if args.fast else Settings()

    df = features.add_features(evaluate.load_processed(args.data))
    args.out.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}
    failed = 0
    for condition in conditions:
        try:
            entry = train_condition(
                df, condition, args.out, models, settings, data_file=Path(args.data).name
            )
        except ValueError as exc:
            failed += 1
            entry = {"status": "failed", "reason": str(exc), "display_name": condition.display_name}
            print(f"{condition.id}: FAILED: {exc}", file=sys.stderr)
        else:
            print(
                f"{condition.id}: best {entry['best_model']}, "
                f"test ROC-AUC {entry['test_roc_auc']:.3f}, "
                f"Brier {entry['test_brier']:.3f} (no-skill {entry['test_brier_no_skill']:.3f})"
            )
        entries[condition.id] = entry

    update_index(args.out, entries)
    print(f"saved to {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""H12: XGBoost, the stronger non-linear model, compared fairly with the H11 baseline.

XGBoost builds hundreds of small decision trees, each one correcting the mistakes of the
ones before ("if BMI > 30 and age band > 9 then…"). It can find interactions a linear model
can't, at the cost of being harder to explain (H14 adds SHAP for that).

Same harness as H11, so the comparison is fair:

1. H10 split → train / test (test untouched until the end).
2. Training rows → *fit* part and *validation* part.
3. Modest randomized search (k-fold ROC-AUC) on the fit part, optionally on a stratified
   sample of it to keep it quick. Don't burn a week on 0.002 AUC.
4. Refit the best settings on the whole fit part, with **early stopping** on the validation
   part: add trees until validation log-loss stops improving.
5. Recall-oriented threshold chosen on validation; final metrics on test.

Run ``python -m ml.src.boosting --condition ID --compare`` to train both models and print
them side by side.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from . import config, evaluate, features, logreg
from .registry import RegistryError, load_registry

NAME = "xgboost"
CV_FOLDS = 3
N_ITER = 12
# Trees used while searching (no early stopping inside cross-validation).
SEARCH_TREES = 300
# Upper limit for the final fit; early stopping usually ends well before it.
MAX_TREES = 2000
EARLY_STOPPING_ROUNDS = 50
# Tuning on more rows than this uses a stratified sample, to keep the search fast.
MAX_TUNING_ROWS = 100_000
# 1 = no re-weighting, same reasoning as H11: weighting the rare class inflates every
# probability. The rare "yes" class is handled by H10's recall-oriented threshold.
SCALE_POS_WEIGHT = 1.0

PARAM_SPACE: dict[str, list[Any]] = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 5, 20],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0],
}


def make_classifier(seed: int = config.RANDOM_SEED, **params: Any) -> Any:
    """An XGBClassifier with the project defaults; `params` override them."""
    from xgboost import XGBClassifier  # imported here so the rest of the pipeline works without it

    settings: dict[str, Any] = {
        "n_estimators": SEARCH_TREES,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "random_state": seed,
        "n_jobs": -1,
    }
    settings.update(params)
    return XGBClassifier(**settings)


def build_xgb(
    predictors: tuple[str, ...] | list[str], seed: int = config.RANDOM_SEED, **params: Any
) -> Pipeline:
    """Unfitted pipeline: the shared H09 preprocessor, then XGBoost."""
    return Pipeline(
        [
            ("prep", features.build_preprocessor(predictors)),
            ("clf", make_classifier(seed=seed, **params)),
        ]
    )


@dataclass(frozen=True)
class Tuned:
    best_params: dict[str, Any]
    best_cv_score: float
    n_rows: int


def _sample(
    x: pd.DataFrame, y: pd.Series, max_rows: int, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    if len(y) <= max_rows:
        return x, y
    x_small, _, y_small, _ = train_test_split(
        x, y, train_size=max_rows, stratify=y, random_state=seed
    )
    return x_small, y_small


def tune_xgb(
    x: pd.DataFrame,
    y: pd.Series,
    predictors: tuple[str, ...] | list[str],
    n_iter: int = N_ITER,
    cv: int = CV_FOLDS,
    max_rows: int = MAX_TUNING_ROWS,
    seed: int = config.RANDOM_SEED,
) -> Tuned:
    """Randomized search over PARAM_SPACE by k-fold ROC-AUC. Returns the winning settings."""
    x_tune, y_tune = _sample(x, y, max_rows, seed)
    search = RandomizedSearchCV(
        build_xgb(predictors, seed=seed),
        param_distributions={f"clf__{k}": v for k, v in PARAM_SPACE.items()},
        n_iter=n_iter,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed),
        random_state=seed,
        refit=False,
    )
    search.fit(x_tune, y_tune)
    best = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
    return Tuned(best_params=best, best_cv_score=float(search.best_score_), n_rows=len(y_tune))


def fit_with_early_stopping(
    predictors: tuple[str, ...] | list[str],
    params: dict[str, Any],
    fit: evaluate.Part,
    val: evaluate.Part,
    seed: int = config.RANDOM_SEED,
) -> tuple[Pipeline, int]:
    """Fit on the fit part, adding trees until validation log-loss stops improving.

    The preprocessor is fit on the fit part only and then used to transform the validation
    rows, so nothing about the validation set leaks into the fill values or scaling.
    Returns the fitted pipeline and the number of trees kept.
    """
    prep = features.build_preprocessor(predictors).fit(fit.x)
    clf = make_classifier(
        seed=seed,
        **{**params, "n_estimators": MAX_TREES, "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
    )
    clf.fit(prep.transform(fit.x), fit.y, eval_set=[(prep.transform(val.x), val.y)], verbose=False)
    n_trees = int(clf.best_iteration) + 1
    return Pipeline([("prep", prep), ("clf", clf)]), n_trees


@dataclass(frozen=True)
class BoostResult:
    """A trained XGBoost model for one condition, plus everything metrics.json needs (H13)."""

    name: str
    condition: str
    model: Pipeline
    params: dict[str, Any]
    n_trees: int
    cv_roc_auc: float
    n_tuning_rows: int
    metrics: dict[str, Any]
    n_fit: int
    n_validation: int
    n_test: int

    def summary(self) -> dict[str, Any]:
        """JSON-safe description (no model object); same shape as LogregResult.summary()."""
        return {
            "model": self.name,
            "condition": self.condition,
            "params": {
                **{k: _plain(v) for k, v in self.params.items()},
                "n_estimators": self.n_trees,
                "scale_pos_weight": SCALE_POS_WEIGHT,
            },
            "cv_roc_auc": self.cv_roc_auc,
            "tuning_rows": self.n_tuning_rows,
            "rows": {"fit": self.n_fit, "validation": self.n_validation, "test": self.n_test},
            "metrics": self.metrics,
        }


def _plain(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def train_xgb(
    split: evaluate.Split,
    n_iter: int = N_ITER,
    cv: int = CV_FOLDS,
    max_tuning_rows: int = MAX_TUNING_ROWS,
    seed: int = config.RANDOM_SEED,
) -> BoostResult:
    """Tune, fit (with early stopping) and evaluate XGBoost on one condition's split."""
    fit, val = evaluate.validation_split(split, seed=seed)
    tuned = tune_xgb(
        fit.x, fit.y, split.predictors, n_iter=n_iter, cv=cv, max_rows=max_tuning_rows, seed=seed
    )
    model, n_trees = fit_with_early_stopping(split.predictors, tuned.best_params, fit, val, seed)
    metrics = evaluate.evaluate_model(model, val.x, val.y, split.x_test, split.y_test)
    return BoostResult(
        name=NAME,
        condition=split.condition,
        model=model,
        params=tuned.best_params,
        n_trees=n_trees,
        cv_roc_auc=tuned.best_cv_score,
        n_tuning_rows=tuned.n_rows,
        metrics=metrics,
        n_fit=len(fit.y),
        n_validation=len(val.y),
        n_test=len(split.y_test),
    )


# --------------------------------------------------------------------------
# Fair comparison (same split, same harness)
# --------------------------------------------------------------------------

COMPARE_METRICS = ("roc_auc", "pr_auc", "brier", "recall", "precision", "threshold")


def compare(
    summaries: list[dict[str, Any]], operating_point: str = "recall_oriented"
) -> pd.DataFrame:
    """One row per model: threshold-free scores plus recall/precision at the chosen cut-off."""
    rows = []
    for s in summaries:
        m = s["metrics"][operating_point]
        rows.append({"model": s["model"], **{k: m[k] for k in COMPARE_METRICS}})
    return pd.DataFrame(rows).set_index("model")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train XGBoost (and optionally the baseline).")
    parser.add_argument("--condition", help="one condition id (default: every condition)")
    parser.add_argument("--data", type=Path, default=evaluate.DEFAULT_DATA)
    parser.add_argument("--n-iter", type=int, default=N_ITER, help="random-search candidates")
    parser.add_argument("--compare", action="store_true", help="also train the H11 baseline")
    args = parser.parse_args(argv)

    registry = load_registry()
    try:
        ids = [registry.get(args.condition).id] if args.condition else list(registry.ids)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    df = features.add_features(evaluate.load_processed(args.data))
    for condition_id in ids:
        split = evaluate.split_condition(df, condition_id)
        result = train_xgb(split, n_iter=args.n_iter)
        summaries = [result.summary()]
        if args.compare:
            summaries.insert(0, logreg.train_logreg(split).summary())
        print(f"{condition_id}: {result.n_trees} trees, settings {result.summary()['params']}")
        print(compare(summaries).round(3).to_string())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

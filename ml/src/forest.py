"""H12: random forest, the second non-linear model in the three-way comparison.

A random forest grows a few hundred independent decision trees. Each tree sees a bootstrap
sample of the rows and a random subset of the factors at every split, and the forest averages
their votes. That averaging is what makes it stable: unlike a single deep tree it doesn't
memorise the training rows, and unlike boosting (H12's XGBoost) the trees don't chase each
other's mistakes, so there is no learning rate and no early stopping to get wrong.

It sits between the two other models in the report: it finds interactions the linear baseline
can't, while staying simple enough to describe in one sentence.

Same harness as H11 and the XGBoost half of H12, so the three-way comparison is fair:

1. H10 split → train / test (test untouched until the end).
2. Training rows → *fit* part and *validation* part (`evaluate.validation_split`).
3. Small randomized search (k-fold ROC-AUC) on the fit part, optionally on a stratified
   sample of it, with shallow "search forests" to keep it quick.
4. Refit the winning settings on the whole fit part with the full number of trees.
5. Recall-oriented threshold chosen on validation; final metrics on test.

Run ``python -m ml.src.forest --condition ID --compare`` to train it next to the baseline.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from . import boosting, config, evaluate, features, logreg
from .registry import RegistryError, load_registry

NAME = "random_forest"
CV_FOLDS = 3
N_ITER = 6
# Trees used while searching; the final fit uses more.
SEARCH_TREES = 120
# Trees in the final forest. More is slightly better and much slower; 300 is the plateau.
N_TREES = 300
# Tuning on more rows than this uses a stratified sample, to keep the search fast.
MAX_TUNING_ROWS = 40_000
# None = no re-weighting, same reasoning as H11 and the XGBoost half of H12: weighting the
# rare class inflates every probability past the no-skill Brier. The rare "yes" class is
# handled by H10's recall-oriented threshold instead.
CLASS_WEIGHT: str | None = None

# Depth and leaf size are capped on purpose: fully grown trees on ~285,000 rows are slow,
# enormous in memory and no more accurate once the leaves are this size.
PARAM_SPACE: dict[str, list[Any]] = {
    "max_depth": [6, 10, 14],
    "min_samples_leaf": [5, 20, 50],
    "max_features": ["sqrt", 0.3],
}


def make_classifier(seed: int = config.RANDOM_SEED, **params: Any) -> RandomForestClassifier:
    """A RandomForestClassifier with the project defaults; `params` override them."""
    settings: dict[str, Any] = {
        "n_estimators": SEARCH_TREES,
        "class_weight": CLASS_WEIGHT,
        "random_state": seed,
        "n_jobs": -1,
    }
    settings.update(params)
    return RandomForestClassifier(**settings)


def build_forest(
    predictors: tuple[str, ...] | list[str], seed: int = config.RANDOM_SEED, **params: Any
) -> Pipeline:
    """Unfitted pipeline: the shared H09 preprocessor, then the forest."""
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


def tune_forest(
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
        build_forest(predictors, seed=seed),
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


def fit_forest(
    predictors: tuple[str, ...] | list[str],
    params: dict[str, Any],
    fit: evaluate.Part,
    n_trees: int = N_TREES,
    seed: int = config.RANDOM_SEED,
) -> Pipeline:
    """Fit the winning settings on the fit part only, with the full number of trees.

    Nothing about the validation or test rows touches the preprocessor's fill values or
    scaling: the whole pipeline sees the fit part and nothing else.
    """
    model = build_forest(predictors, seed=seed, **{**params, "n_estimators": n_trees})
    model.fit(fit.x, fit.y)
    return model


@dataclass(frozen=True)
class ForestResult:
    """A trained random forest for one condition, plus everything metrics.json needs (H13)."""

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
                "class_weight": CLASS_WEIGHT,
            },
            "cv_roc_auc": self.cv_roc_auc,
            "tuning_rows": self.n_tuning_rows,
            "rows": {"fit": self.n_fit, "validation": self.n_validation, "test": self.n_test},
            "metrics": self.metrics,
        }


def _plain(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def train_forest(
    split: evaluate.Split,
    n_iter: int = N_ITER,
    cv: int = CV_FOLDS,
    max_tuning_rows: int = MAX_TUNING_ROWS,
    n_trees: int = N_TREES,
    seed: int = config.RANDOM_SEED,
) -> ForestResult:
    """Tune, fit and evaluate the random forest on one condition's split."""
    fit, val = evaluate.validation_split(split, seed=seed)
    tuned = tune_forest(
        fit.x, fit.y, split.predictors, n_iter=n_iter, cv=cv, max_rows=max_tuning_rows, seed=seed
    )
    model = fit_forest(split.predictors, tuned.best_params, fit, n_trees=n_trees, seed=seed)
    metrics = evaluate.evaluate_model(model, val.x, val.y, split.x_test, split.y_test)
    return ForestResult(
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
# Command line
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a random forest (and the baseline).")
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
        result = train_forest(split, n_iter=args.n_iter)
        summaries = [result.summary()]
        if args.compare:
            summaries.insert(0, logreg.train_logreg(split).summary())
        print(f"{condition_id}: {result.n_trees} trees, settings {result.summary()['params']}")
        # The side-by-side table is shared with the XGBoost module, so both CLIs print alike.
        print(boosting.compare(summaries).round(3).to_string())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

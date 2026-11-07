"""H11: the logistic regression baseline, the simple model every other model must beat.

Logistic regression gives each factor one weight. Exponentiated, that weight is an
**odds ratio**: how much the odds of the condition change per step of the factor, with the
other factors held equal. That makes it easy to explain in the report.

Training follows the H10 harness exactly, so the comparison with XGBoost (H12) is fair:

1. `evaluate.split_condition` → train / test (test is untouched until the very end).
2. `evaluate.validation_split` → the training rows become a *fit* part and a *validation* part.
3. A small cross-validated grid on `C` (regularisation strength), on the fit part only.
4. The recall-oriented threshold is chosen on the validation part.
5. Final metrics on the test part, at 0.5 and at that threshold.

Run ``python -m ml.src.logreg --condition ID`` for a quick report and the odds-ratio table.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from . import config, evaluate, features
from .registry import RegistryError, load_registry

NAME = "logreg"
# From strong (0.01) to weak (10) regularisation. Small on purpose: this is the baseline.
C_GRID = (0.01, 0.1, 1.0, 10.0)
CV_FOLDS = 5
MAX_ITER = 2000
# No class re-weighting (deviation from the roadmap's "balanced", see the H11 doc):
# "balanced" leaves the ranking (ROC-AUC) unchanged but inflates every probability, so a
# 15% risk can read as 40% and the Brier score falls below the no-skill baseline. The rare
# "yes" class is handled by the recall-oriented threshold from H10 instead.
CLASS_WEIGHT: str | None = None
DEFAULT_OUT = config.REPO_ROOT / "docs" / "figures"


def build_logreg(
    predictors: tuple[str, ...] | list[str],
    C: float = 1.0,
    class_weight: str | None = CLASS_WEIGHT,
) -> Pipeline:
    """Unfitted pipeline: the shared H09 preprocessor, then a logistic regression.

    Pass `class_weight="balanced"` to compare against the re-weighted variant.
    """
    clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=MAX_ITER)
    return Pipeline([("prep", features.build_preprocessor(predictors)), ("clf", clf)])


@dataclass(frozen=True)
class Tuned:
    model: Pipeline
    best_c: float
    cv_scores: dict[float, float]


def tune_logreg(
    x: pd.DataFrame,
    y: pd.Series,
    predictors: tuple[str, ...] | list[str],
    cv: int = CV_FOLDS,
    seed: int = config.RANDOM_SEED,
) -> Tuned:
    """Pick C by stratified k-fold ROC-AUC on the rows given, then refit on all of them."""
    search = GridSearchCV(
        build_logreg(predictors),
        param_grid={"clf__C": list(C_GRID)},
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed),
        refit=True,
    )
    search.fit(x, y)
    scores = {
        float(c): float(s)
        for c, s in zip(
            search.cv_results_["param_clf__C"], search.cv_results_["mean_test_score"], strict=True
        )
    }
    return Tuned(
        model=search.best_estimator_,
        best_c=float(search.best_params_["clf__C"]),
        cv_scores=scores,
    )


@dataclass(frozen=True)
class LogregResult:
    """A trained baseline for one condition, plus everything metrics.json needs (H13)."""

    name: str
    condition: str
    model: Pipeline
    best_c: float
    cv_scores: dict[float, float]
    metrics: dict[str, Any]
    n_fit: int
    n_validation: int
    n_test: int

    def summary(self) -> dict[str, Any]:
        """JSON-safe description (no model object)."""
        return {
            "model": self.name,
            "condition": self.condition,
            "params": {"C": self.best_c, "class_weight": CLASS_WEIGHT},
            "cv_roc_auc": {str(c): s for c, s in self.cv_scores.items()},
            "rows": {"fit": self.n_fit, "validation": self.n_validation, "test": self.n_test},
            "metrics": self.metrics,
        }


def train_logreg(
    split: evaluate.Split, cv: int = CV_FOLDS, seed: int = config.RANDOM_SEED
) -> LogregResult:
    """Tune, fit and evaluate the baseline on one condition's split (see module docstring)."""
    fit, val = evaluate.validation_split(split, seed=seed)
    tuned = tune_logreg(fit.x, fit.y, split.predictors, cv=cv, seed=seed)
    metrics = evaluate.evaluate_model(tuned.model, val.x, val.y, split.x_test, split.y_test)
    return LogregResult(
        name=NAME,
        condition=split.condition,
        model=tuned.model,
        best_c=tuned.best_c,
        cv_scores=tuned.cv_scores,
        metrics=metrics,
        n_fit=len(fit.y),
        n_validation=len(val.y),
        n_test=len(split.y_test),
    )


# --------------------------------------------------------------------------
# Odds ratios (for the report)
# --------------------------------------------------------------------------


def clean_name(encoded: str) -> str:
    """Turn the preprocessor's column names back into readable ones.

    num__bmi → bmi · num__missingindicator_x → "x (was missing)" · cat__smoker_daily →
    "smoker = daily".
    """
    group, _, rest = encoded.partition("__")
    if not rest:
        return encoded
    if rest.startswith("missingindicator_"):
        return f"{rest.removeprefix('missingindicator_')} (was missing)"
    if group == "cat":
        for column in sorted(features.CATEGORICAL, key=len, reverse=True):
            if rest.startswith(f"{column}_"):
                return f"{column} = {rest.removeprefix(column + '_')}"
    return rest


def odds_ratios(model: Pipeline) -> pd.DataFrame:
    """One row per model input: coefficient and odds ratio, strongest effect first.

    Numbers are scaled by the preprocessor, so a numeric odds ratio is **per one standard
    deviation** of that factor. Yes/no answers are per "yes" vs "no". One-hot categories keep
    every level, so compare a category's levels with each other rather than reading one alone.
    """
    try:
        check_is_fitted(model)
    except NotFittedError as exc:
        raise ValueError("odds_ratios needs a fitted model") from exc
    names = [clean_name(n) for n in model.named_steps["prep"].get_feature_names_out()]
    coef = model.named_steps["clf"].coef_.ravel()
    table = pd.DataFrame({"feature": names, "coefficient": coef, "odds_ratio": np.exp(coef)})
    order = table["coefficient"].abs().sort_values(ascending=False, kind="stable").index
    return table.loc[order].reset_index(drop=True)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _report(result: LogregResult, table: pd.DataFrame, top: int = 10) -> str:
    lines = [f"{result.condition}: logistic regression, C = {result.best_c}"]
    for key in ("default", "recall_oriented"):
        m = result.metrics[key]
        lines.append(
            f"  {key:<16} threshold {m['threshold']:.3f}  recall {m['recall']:.3f}  "
            f"precision {m['precision']:.3f}  ROC-AUC {m['roc_auc']:.3f}  "
            f"PR-AUC {m['pr_auc']:.3f}  Brier {m['brier']:.3f} (no-skill {m['brier_no_skill']:.3f})"
        )
    lines.append(f"  strongest odds ratios (top {top}):")
    for row in table.head(top).itertuples():
        lines.append(f"    {row.feature:<32} {row.odds_ratio:7.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and report the logistic regression baseline."
    )
    parser.add_argument("--condition", help="one condition id (default: every condition)")
    parser.add_argument("--data", type=Path, default=evaluate.DEFAULT_DATA)
    parser.add_argument("--cv", type=int, default=CV_FOLDS, help="cross-validation folds")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="folder for odds-ratio CSVs")
    args = parser.parse_args(argv)

    registry = load_registry()
    try:
        ids = [registry.get(args.condition).id] if args.condition else list(registry.ids)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    df = features.add_features(evaluate.load_processed(args.data))
    for condition_id in ids:
        result = train_logreg(evaluate.split_condition(df, condition_id), cv=args.cv)
        table = odds_ratios(result.model)
        print(_report(result, table))
        folder = args.out / condition_id
        folder.mkdir(parents=True, exist_ok=True)
        table.to_csv(folder / "logreg_odds_ratios.csv", index=False)
        print(f"  saved {folder / 'logreg_odds_ratios.csv'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

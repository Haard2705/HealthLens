"""H10: the per-condition split and the one scoring function every model goes through.

If each model were split and scored its own way, comparing them would mean nothing. So:

* `split_condition` makes **one** train/test split per condition (80/20, stratified, fixed
  seed). Rows whose outcome is missing are dropped first: a skipped question is not a "no".
  Each respondent gets one random number for the whole file, so the same people tend to be
  in the test set for every condition.
* `validation_split` carves a validation set out of the *training* rows. Thresholds (and
  XGBoost's early stopping, H12) are chosen there, never on the test set.
* `evaluate` scores any fitted model with `predict_proba` the same way.
* `evaluate_model` reports two operating points: the plain 0.5 cut-off, and a
  recall-oriented cut-off chosen on validation data (a screening tool should rather
  over-flag than miss people).

Run ``python -m ml.src.evaluate [--condition ID]`` to print split sizes from the
processed file.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from . import config, features
from .registry import RegistryError, load_registry

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
DEFAULT_THRESHOLD = 0.5
# Recall-oriented operating point: catch at least 80% of people who have the condition.
TARGET_RECALL = 0.8
DEFAULT_DATA = config.DATA_PROCESSED / "brfss_clean.parquet"

METRIC_KEYS = (
    "n",
    "prevalence",
    "threshold",
    "weighted",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
    "brier_no_skill",
    "confusion_matrix",
)


@dataclass(frozen=True)
class Split:
    """One condition's train/test split. x is predictors only; w is the survey weight."""

    condition: str
    predictors: tuple[str, ...]
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    w_train: pd.Series
    w_test: pd.Series


class Part(NamedTuple):
    x: pd.DataFrame
    y: pd.Series
    w: pd.Series


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def _check_fraction(value: float, name: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def split_condition(
    df: pd.DataFrame,
    condition_id: str,
    test_size: float = TEST_SIZE,
    seed: int = config.RANDOM_SEED,
) -> Split:
    """Stratified train/test split for one condition. The input frame is not modified.

    Every row of the full frame draws one random number from `seed`. Within each class
    (has the condition / doesn't), the rows with the lowest numbers go to test. That keeps
    the class balance identical in train and test, and because the numbers are shared,
    the same respondents land in test for every condition wherever their answers allow.
    """
    _check_fraction(test_size, "test_size")
    condition = load_registry().get(condition_id)

    if not set(features.DERIVED_COLUMNS) <= set(df.columns):
        df = features.add_features(df)

    draw = pd.Series(np.random.default_rng(seed).random(len(df)), index=df.index)
    known = df[condition.outcome_column].notna()
    y = df.loc[known, condition.outcome_column].astype(bool).astype("int64")
    if y.nunique() < 2:
        raise ValueError(f"{condition_id}: outcome needs both yes and no answers to split")

    test_index: list[Any] = []
    for value in (0, 1):
        members = draw[y.index[y == value]].sort_values(kind="stable")
        n_test = round(len(members) * test_size)
        test_index.extend(members.index[:n_test])
    in_test = y.index.isin(test_index)

    predictors = features.predictors_for(condition_id)
    x = features.model_frame(df.loc[known], predictors)
    w = df.loc[known, "survey_weight"].astype("float64")
    return Split(
        condition=condition_id,
        predictors=predictors,
        x_train=x[~in_test],
        x_test=x[in_test],
        y_train=y[~in_test],
        y_test=y[in_test],
        w_train=w[~in_test],
        w_test=w[in_test],
    )


def validation_split(
    split: Split, size: float = VALIDATION_SIZE, seed: int = config.RANDOM_SEED
) -> tuple[Part, Part]:
    """Carve a stratified validation set out of the training rows: (fit part, validation part)."""
    _check_fraction(size, "size")
    x_fit, x_val, y_fit, y_val, w_fit, w_val = train_test_split(
        split.x_train,
        split.y_train,
        split.w_train,
        test_size=size,
        stratify=split.y_train,
        random_state=seed,
    )
    return Part(x_fit, y_fit, w_fit), Part(x_val, y_val, w_val)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _positive_scores(model: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x))[:, 1].astype("float64")


def _check_labels(y: np.ndarray) -> None:
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("labels must be 0/1")
    if len(np.unique(y)) < 2:
        raise ValueError("the evaluation set must contain both classes")


def score(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """All metrics from labels and predicted probabilities. A score >= threshold means "yes"."""
    y = np.asarray(y_true).astype("int64")
    p = np.asarray(scores, dtype="float64")
    w = None if sample_weight is None else np.asarray(sample_weight, dtype="float64")
    if len(p) != len(y) or (w is not None and len(w) != len(y)):
        raise ValueError("labels, scores and weights must all be the same length")
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    _check_labels(y)

    predicted = (p >= threshold).astype("int64")
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1], sample_weight=w).ravel()
    prevalence = float(np.average(y, weights=w))

    def cell(value: float) -> int | float:
        return int(value) if w is None else float(value)

    return {
        "n": len(y),
        "prevalence": prevalence,
        "threshold": float(threshold),
        "weighted": w is not None,
        "accuracy": float(accuracy_score(y, predicted, sample_weight=w)),
        "precision": float(precision_score(y, predicted, sample_weight=w, zero_division=0)),
        "recall": float(recall_score(y, predicted, sample_weight=w, zero_division=0)),
        "f1": float(f1_score(y, predicted, sample_weight=w, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
        "pr_auc": float(average_precision_score(y, p, sample_weight=w)),
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
        # A model that always predicts the prevalence scores p(1 - p); beat this to add value.
        "brier_no_skill": prevalence * (1 - prevalence),
        "confusion_matrix": {"tn": cell(tn), "fp": cell(fp), "fn": cell(fn), "tp": cell(tp)},
    }


def evaluate(
    model: Any,
    x_test: pd.DataFrame,
    y_test: Sequence[int] | np.ndarray | pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
    sample_weight: Sequence[float] | np.ndarray | pd.Series | None = None,
) -> dict[str, Any]:
    """Score a fitted model (anything with `predict_proba`) on held-out data.

    Unweighted by default: that measures the model on the people in the test set. Pass
    survey weights to estimate how it would do on the US adult population instead.
    """
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    y = np.asarray(y_test)
    if sample_weight is not None and len(sample_weight) != len(y):
        raise ValueError("labels and weights must be the same length")
    return score(y, _positive_scores(model, x_test), threshold, sample_weight)


def choose_threshold(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> float:
    """The highest cut-off that still catches at least `target_recall` of the positives.

    Higher cut-offs flag fewer people, so this is the most precise threshold that meets
    the recall goal. Call it on validation data only.
    """
    if not 0 < target_recall <= 1:
        raise ValueError(f"target_recall must be in (0, 1], got {target_recall}")
    y = np.asarray(y_true).astype("int64")
    _check_labels(y)
    _, recall, thresholds = precision_recall_curve(y, np.asarray(scores, dtype="float64"))
    # recall[i] is the recall at thresholds[i]; recall falls as the threshold rises.
    reaching = thresholds[recall[:-1] >= target_recall - 1e-12]
    return float(reaching.max())


def evaluate_model(
    model: Any,
    x_val: pd.DataFrame,
    y_val: Sequence[int] | np.ndarray | pd.Series,
    x_test: pd.DataFrame,
    y_test: Sequence[int] | np.ndarray | pd.Series,
    target_recall: float = TARGET_RECALL,
) -> dict[str, Any]:
    """Test-set metrics at 0.5 and at a recall-oriented threshold picked on validation data."""
    threshold = choose_threshold(y_val, _positive_scores(model, x_val), target_recall)
    return {
        "default": evaluate(model, x_test, y_test),
        "recall_oriented": evaluate(model, x_test, y_test, threshold=threshold),
        "target_recall": target_recall,
        "threshold_chosen_on": "validation",
    }


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def load_processed(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def describe(split: Split) -> str:
    def side(y: pd.Series) -> str:
        return f"{len(y):>8,} rows, {y.mean():6.2%} yes"

    return f"{split.condition:<16} train {side(split.y_train)} | test {side(split.y_test)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print each condition's train/test split.")
    parser.add_argument("--condition", help="one condition id (default: every condition)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args(argv)

    registry = load_registry()
    try:
        ids = [registry.get(args.condition).id] if args.condition else list(registry.ids)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    df = features.add_features(load_processed(args.data))
    for condition_id in ids:
        print(describe(split_condition(df, condition_id)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

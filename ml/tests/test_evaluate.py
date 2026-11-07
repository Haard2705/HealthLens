"""H10: the per-condition split and the one scoring function every model goes through."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src import config, evaluate, features, schema
from src.registry import load_registry

from .test_features import make_frame

CONDITIONS = load_registry().ids
FIRST, SECOND = CONDITIONS[0], CONDITIONS[1]


class FixedScores:
    """A stand-in model whose predicted probabilities are given up front."""

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def predict_proba(self, x):
        assert len(x) == len(self.scores)
        return np.column_stack([1 - self.scores, self.scores])


# Hand-computed example used throughout:
#   y      = 0    0    1     1
#   scores = 0.1  0.4  0.35  0.8
# At 0.5 only the last is predicted positive: TP=1, FP=0, FN=1, TN=2.
Y = np.array([0, 0, 1, 1])
SCORES = [0.1, 0.4, 0.35, 0.8]
X4 = pd.DataFrame({"a": [0, 1, 2, 3]})


@pytest.fixture
def frame() -> pd.DataFrame:
    df = make_frame(n=1000, seed=1)
    # Real data has gaps in the outcomes: those rows must leave that condition's split.
    rng = np.random.default_rng(5)
    df.loc[rng.random(len(df)) < 0.1, schema.OUTCOMES[0]] = pd.NA
    return df


# ---- evaluate: hand-computed numbers --------------------------------------------------


def test_evaluate_matches_hand_computed_metrics() -> None:
    m = evaluate.evaluate(FixedScores(SCORES), X4, Y)
    assert m["threshold"] == 0.5
    assert m["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 / 3)
    # ROC-AUC: of the 4 (positive, negative) pairs, 3 are ranked correctly.
    assert m["roc_auc"] == pytest.approx(0.75)
    # Average precision: 0.5 * 1.0 (first positive) + 0.5 * 2/3 (second positive).
    assert m["pr_auc"] == pytest.approx(0.5 + 1 / 3)
    # Brier: mean of (0.1-0)^2, (0.4-0)^2, (0.35-1)^2, (0.8-1)^2.
    assert m["brier"] == pytest.approx((0.01 + 0.16 + 0.4225 + 0.04) / 4)


def test_evaluate_reports_prevalence_and_no_skill_brier() -> None:
    m = evaluate.evaluate(FixedScores(SCORES), X4, Y)
    assert m["n"] == 4
    assert m["prevalence"] == pytest.approx(0.5)
    # Always predicting the prevalence scores p * (1 - p).
    assert m["brier_no_skill"] == pytest.approx(0.25)


def test_evaluate_uses_the_given_threshold() -> None:
    m = evaluate.evaluate(FixedScores(SCORES), X4, Y, threshold=0.3)
    assert m["threshold"] == 0.3
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
    assert m["recall"] == pytest.approx(1.0)


def test_threshold_is_inclusive() -> None:
    m = evaluate.evaluate(FixedScores(SCORES), X4, Y, threshold=0.35)
    assert m["confusion_matrix"]["tp"] == 2


def test_evaluate_with_survey_weights() -> None:
    # Weighting the missed positive 3x drops weighted recall from 1/2 to 1/4.
    m = evaluate.evaluate(FixedScores(SCORES), X4, Y, sample_weight=[1, 1, 3, 1])
    assert m["recall"] == pytest.approx(0.25)
    assert m["weighted"] is True
    assert evaluate.evaluate(FixedScores(SCORES), X4, Y)["weighted"] is False


def test_precision_is_zero_not_an_error_when_nothing_is_flagged() -> None:
    m = evaluate.evaluate(FixedScores(SCORES), X4, Y, threshold=0.99)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_metric_keys_and_ranges_on_a_real_model(frame) -> None:
    condition = FIRST
    split = evaluate.split_condition(frame, condition)
    model = Pipeline(
        [
            ("prep", features.build_preprocessor(split.predictors)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    ).fit(split.x_train, split.y_train)
    m = evaluate.evaluate(model, split.x_test, split.y_test)

    assert set(m) == set(evaluate.METRIC_KEYS)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"):
        assert 0.0 <= m[key] <= 1.0, key
    assert sum(m["confusion_matrix"].values()) == m["n"] == len(split.y_test)


def test_evaluate_rejects_a_single_class_test_set() -> None:
    with pytest.raises(ValueError, match="both classes"):
        evaluate.evaluate(FixedScores([0.2, 0.7]), X4.iloc[:2], np.array([1, 1]))


def test_evaluate_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluate.evaluate(FixedScores(SCORES), X4, Y, sample_weight=[1, 1])


def test_evaluate_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        evaluate.evaluate(FixedScores(SCORES), X4, Y, threshold=1.5)


def test_metrics_are_plain_python_for_json() -> None:
    import json

    json.dumps(evaluate.evaluate(FixedScores(SCORES), X4, Y))


# ---- choose_threshold -----------------------------------------------------------------


def test_choose_threshold_is_the_highest_that_reaches_the_target_recall() -> None:
    # Catching both positives needs the threshold down at 0.35; one of two only needs 0.8.
    assert evaluate.choose_threshold(Y, SCORES, target_recall=1.0) == pytest.approx(0.35)
    assert evaluate.choose_threshold(Y, SCORES, target_recall=0.5) == pytest.approx(0.8)


def test_choose_threshold_default_target_is_documented_value() -> None:
    assert evaluate.TARGET_RECALL == 0.8
    y = np.array([0] * 6 + [1] * 5)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.25, 0.45, 0.55, 0.7, 0.9])
    t = evaluate.choose_threshold(y, scores)
    assert ((scores >= t) & (y == 1)).sum() / 5 >= 0.8
    assert t == pytest.approx(0.45)  # 4 of 5 positives at or above it


def test_choose_threshold_rejects_bad_target() -> None:
    with pytest.raises(ValueError, match="target_recall"):
        evaluate.choose_threshold(Y, SCORES, target_recall=0)


# ---- evaluate_model: both thresholds, chosen on validation only -------------------------


def test_evaluate_model_reports_default_and_recall_oriented() -> None:
    val_scores = [0.1, 0.4, 0.35, 0.8]
    test_scores = [0.2, 0.4, 0.3, 0.9]

    class SwitchModel:
        def predict_proba(self, x):
            s = np.asarray(val_scores if x["a"].iloc[0] == 0 else test_scores)
            return np.column_stack([1 - s, s])

    x_val = pd.DataFrame({"a": [0, 0, 0, 0]})
    x_test = pd.DataFrame({"a": [1, 1, 1, 1]})
    result = evaluate.evaluate_model(SwitchModel(), x_val, Y, x_test, Y, target_recall=1.0)

    assert result["default"]["threshold"] == 0.5
    # Chosen from validation scores (0.35), then applied unchanged to the test set.
    assert result["recall_oriented"]["threshold"] == pytest.approx(0.35)
    # Test scores 0.2 0.4 0.3 0.9 at 0.35 -> predicted 0 1 0 1.
    assert result["recall_oriented"]["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert result["target_recall"] == 1.0
    assert result["threshold_chosen_on"] == "validation"


# ---- split_condition ------------------------------------------------------------------


def test_split_drops_rows_with_missing_outcome(frame) -> None:
    column = schema.OUTCOMES[0]
    condition = load_registry().conditions[0]
    assert condition.outcome_column == column
    split = evaluate.split_condition(frame, condition.id)
    known = int(frame[column].notna().sum())
    assert len(split.y_train) + len(split.y_test) == known < len(frame)


def test_split_is_about_80_20_and_stratified(frame) -> None:
    split = evaluate.split_condition(frame, FIRST)
    total = len(split.y_train) + len(split.y_test)
    assert len(split.y_test) / total == pytest.approx(0.2, abs=0.01)
    assert split.y_train.mean() == pytest.approx(split.y_test.mean(), abs=0.02)


def test_split_train_and_test_do_not_overlap(frame) -> None:
    split = evaluate.split_condition(frame, FIRST)
    assert set(split.x_train.index).isdisjoint(split.x_test.index)
    assert split.x_train.index.equals(split.y_train.index)
    assert split.x_test.index.equals(split.w_test.index)


def test_split_is_reproducible(frame) -> None:
    a = evaluate.split_condition(frame, FIRST)
    b = evaluate.split_condition(frame, FIRST)
    assert a.x_test.index.equals(b.x_test.index)
    c = evaluate.split_condition(frame, FIRST, seed=config.RANDOM_SEED + 1)
    assert not a.x_test.index.equals(c.x_test.index)


def test_same_respondents_land_in_test_across_conditions(frame) -> None:
    a = set(evaluate.split_condition(frame, FIRST).x_test.index)
    b = set(evaluate.split_condition(frame, SECOND).x_test.index)
    # Independent random splits would share about 20% of test rows; ours share most of them.
    overlap = len(a & b) / min(len(a), len(b))
    assert overlap > 0.8


def test_split_uses_only_allowed_predictors(frame) -> None:
    for condition_id in CONDITIONS:
        split = evaluate.split_condition(frame, condition_id)
        assert tuple(split.x_train.columns) == features.predictors_for(condition_id)
        outcome = load_registry().get(condition_id).outcome_column
        assert outcome not in split.x_train.columns
        assert "survey_weight" not in split.x_train.columns


def test_split_targets_are_zero_one_ints(frame) -> None:
    split = evaluate.split_condition(frame, FIRST)
    assert set(split.y_train.unique()) <= {0, 1}
    assert split.y_train.dtype == "int64"


def test_split_adds_derived_features_when_missing(frame) -> None:
    assert "bmi_band" not in frame.columns
    split = evaluate.split_condition(frame, FIRST)
    assert "bmi_band" in split.x_train.columns


def test_split_does_not_modify_the_input(frame) -> None:
    before = frame.copy()
    evaluate.split_condition(frame, FIRST)
    pd.testing.assert_frame_equal(frame, before)


def test_split_rejects_bad_test_size(frame) -> None:
    with pytest.raises(ValueError, match="test_size"):
        evaluate.split_condition(frame, FIRST, test_size=1.2)


def test_split_needs_both_classes(frame) -> None:
    frame[schema.OUTCOMES[0]] = pd.array([False] * len(frame), dtype="boolean")
    with pytest.raises(ValueError, match="both"):
        evaluate.split_condition(frame, FIRST)


# ---- validation split (for threshold choice and early stopping) ------------------------


def test_validation_split_carves_from_train_only(frame) -> None:
    split = evaluate.split_condition(frame, FIRST)
    fit, val = evaluate.validation_split(split)
    assert set(fit.x.index) | set(val.x.index) == set(split.x_train.index)
    assert set(fit.x.index).isdisjoint(val.x.index)
    assert set(val.x.index).isdisjoint(split.x_test.index)
    assert len(val.y) / len(split.y_train) == pytest.approx(0.2, abs=0.01)
    assert fit.w.index.equals(fit.x.index)


# ---- end to end with a baseline ---------------------------------------------------------


def test_dummy_baseline_scores_like_a_coin(frame) -> None:
    split = evaluate.split_condition(frame, FIRST)
    dummy = DummyClassifier(strategy="prior").fit(split.x_train, split.y_train)
    m = evaluate.evaluate(dummy, split.x_test, split.y_test)
    assert m["roc_auc"] == pytest.approx(0.5)
    assert m["brier"] == pytest.approx(m["brier_no_skill"], abs=0.01)


# ---- command line -----------------------------------------------------------------------


def test_main_prints_split_summary(frame, monkeypatch, capsys) -> None:
    monkeypatch.setattr(evaluate, "load_processed", lambda path: frame)
    assert evaluate.main(["--condition", FIRST]) == 0
    out = capsys.readouterr().out
    assert FIRST in out
    assert "train" in out and "test" in out


def test_main_all_conditions_by_default(frame, monkeypatch, capsys) -> None:
    monkeypatch.setattr(evaluate, "load_processed", lambda path: frame)
    assert evaluate.main([]) == 0
    out = capsys.readouterr().out
    for condition_id in CONDITIONS:
        assert condition_id in out


def test_main_unknown_condition_is_a_clear_error(frame, monkeypatch, capsys) -> None:
    monkeypatch.setattr(evaluate, "load_processed", lambda path: frame)
    assert evaluate.main(["--condition", "not_a_condition"]) == 1
    assert "unknown condition" in capsys.readouterr().err

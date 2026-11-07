"""H15: subgroup results, calibration, plain-language limitations and generated model cards."""

import json

import numpy as np
import pandas as pd
import pytest
from src import evaluate, model_card, train
from src.registry import load_registry

from .test_logreg import signal_frame

REGISTRY = load_registry()
TIER1 = next(c for c in REGISTRY.conditions if c.tier == 1)
TIER2 = next(c for c in REGISTRY.conditions if c.tier == 2)


# ---- subgroups --------------------------------------------------------------------------


def test_subgroup_labels_are_readable() -> None:
    x = pd.DataFrame(
        {
            "sex": ["male", "female", np.nan],
            "age_band": [1.0, 8.0, 13.0],
            "income_band": [2.0, 5.0, np.nan],
        }
    )
    labels = model_card.subgroup_labels(x)
    assert labels["sex"].tolist() == ["male", "female", "not stated"]
    assert labels["age_group"].tolist() == ["18-34", "50-64", "65+"]
    assert labels["income_group"].tolist() == ["under $35k", "$35k-$100k", "not stated"]


def test_subgroup_metrics_by_hand() -> None:
    # 40 people in group "a": 10 positives, 6 of them above the cut-off -> recall 0.6.
    y = np.array([1] * 10 + [0] * 30 + [1] * 2 + [0] * 3)
    p = np.array([0.9] * 6 + [0.1] * 4 + [0.2] * 30 + [0.5] * 5)
    groups = pd.DataFrame({"sex": ["a"] * 40 + ["b"] * 5})
    rows = {r["group"]: r for r in model_card.subgroup_metrics(y, p, groups, threshold=0.5)}
    a = rows["a"]
    assert a["n"] == 40 and a["positives"] == 10
    assert a["recall"] == pytest.approx(0.6)
    assert a["observed_rate"] == pytest.approx(0.25)
    assert a["mean_predicted"] == pytest.approx((6 * 0.9 + 4 * 0.1 + 30 * 0.2) / 40)
    assert a["calibration_gap"] == pytest.approx(a["mean_predicted"] - 0.25)
    assert 0 <= a["pr_auc"] <= 1


def test_groups_are_listed_in_natural_order() -> None:
    groups = pd.DataFrame(
        {"income_group": ["$100k+", "not stated", "under $35k", "$35k-$100k"] * 10}
    )
    y = np.array([0, 1] * 20)
    rows = model_card.subgroup_metrics(y, np.linspace(0, 1, 40), groups, 0.5)
    assert [r["group"] for r in rows] == ["under $35k", "$35k-$100k", "$100k+", "not stated"]


def test_a_big_group_where_nobody_reports_it_is_not_scored() -> None:
    groups = pd.DataFrame({"sex": ["a"] * 100})
    (row,) = model_card.subgroup_metrics(np.zeros(100), np.full(100, 0.1), groups, 0.5)
    assert row["n"] == 100 and row["recall"] is None
    assert row["note"] == "too few people or cases"


def test_small_groups_are_not_scored() -> None:
    y = np.array([1, 0, 1, 0])
    groups = pd.DataFrame({"sex": ["b"] * 4})
    (row,) = model_card.subgroup_metrics(y, np.array([0.6, 0.4, 0.3, 0.2]), groups, 0.5)
    assert row["recall"] is None and row["note"] == "too few people or cases"


# ---- calibration ------------------------------------------------------------------------


def test_calibration_table_bins_people_equally() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = rng.random(1000) < p  # perfectly calibrated by construction
    table = model_card.calibration_table(y, p, bins=10)
    assert len(table) == 10
    assert table["n"].tolist() == [100] * 10
    assert (table["mean_predicted"].diff().dropna() > 0).all()
    assert np.abs(table["mean_predicted"] - table["observed_rate"]).max() < 0.12


def test_calibration_plot_is_written(tmp_path) -> None:
    table = model_card.calibration_table(np.array([0, 1] * 50), np.linspace(0, 1, 100))
    path = model_card.save_calibration_plot({"a": table}, tmp_path / "c.png", "T")
    assert path.stat().st_size > 1000


# ---- limitations ------------------------------------------------------------------------


def row(group_by="sex", group="female", recall=0.8, gap=0.0, observed=0.2):
    return {
        "group_by": group_by,
        "group": group,
        "recall": recall,
        "calibration_gap": gap,
        "mean_predicted": observed + (gap or 0.0),
        "observed_rate": observed,
    }


def test_general_limitations_always_present() -> None:
    items = model_card.limitations([], overall_recall=0.8)
    text = " ".join(items).lower()
    for phrase in ("not a diagnosis", "self-reported", "not causes", "us adults"):
        assert phrase in text


def test_weaker_subgroup_gets_a_plain_sentence() -> None:
    items = model_card.limitations([row(group_by="age_group", group="18-34", recall=0.55)], 0.8)
    extra = items[len(model_card.GENERAL_LIMITATIONS) :]
    assert extra == [
        'It misses more cases for people in the group "age group 18-34" '
        "(catches 55% vs 80% overall)."
    ]


def test_miscalibrated_subgroup_gets_a_sentence() -> None:
    items = model_card.limitations([row(gap=0.08, observed=0.10)], 0.8)
    assert any("overestimates risk" in item and "18% vs 10%" in item for item in items)


def test_small_or_unstated_groups_dont_create_sentences() -> None:
    rows = [row(recall=None, gap=None), row(group="not stated", recall=0.1)]
    assert model_card.limitations(rows, 0.8) == list(model_card.GENERAL_LIMITATIONS)


# ---- end to end: train, then generate cards ------------------------------------------------


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("cards")
    frame = signal_frame(n=2500)
    models, cards, figures = root / "models", root / "cards", root / "figures"
    mp = pytest.MonkeyPatch()
    mp.setattr(evaluate, "load_processed", lambda path: frame)
    for c in (TIER1, TIER2):
        train.main(["--condition", c.id, "--models", "logreg", "--fast", "--out", str(models)])
    args = ["--models-dir", str(models), "--cards-dir", str(cards), "--figures-dir", str(figures)]
    code = model_card.main(args)
    mp.undo()
    return code, models, cards, figures


def test_cli_writes_cards_for_trained_conditions_only(built) -> None:
    code, _, cards, _ = built
    assert code == 0
    assert {p.stem for p in cards.glob("*.md")} == {TIER1.id, TIER2.id}


def test_tier1_card_has_every_section_and_subgroups(built) -> None:
    _, _, cards, _ = built
    text = (cards / f"{TIER1.id}.md").read_text()
    for heading in (
        "## Intended use",
        "## Not intended for",
        "## Data",
        "## Performance on held-out data",
        "## Results by group",
        "## Release gate (ADR-0016)",
        "## Limitations",
    ):
        assert heading in text
    assert TIER1.display_name in text
    assert "Age group: 65+" in text and "Sex: female" in text
    assert "logreg (chosen)" in text


def test_tier2_card_has_overall_metrics_only(built) -> None:
    _, _, cards, _ = built
    text = (cards / f"{TIER2.id}.md").read_text()
    assert "Tier 2: overall metrics only" in text
    assert "Age group:" not in text


def test_limitations_json_for_the_ui(built) -> None:
    _, models, _, _ = built
    data = json.loads((models / TIER1.id / "limitations.json").read_text())
    assert data["condition"] == TIER1.id
    assert data["limitations"][: len(model_card.GENERAL_LIMITATIONS)] == list(
        model_card.GENERAL_LIMITATIONS
    )


def test_calibration_figure_written(built) -> None:
    _, _, _, figures = built
    assert (figures / TIER1.id / "calibration.png").exists()


def test_cli_with_nothing_trained(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(evaluate, "load_processed", lambda path: signal_frame(n=300))
    args = ["--models-dir", str(tmp_path), "--cards-dir", str(tmp_path / "c")]
    assert model_card.main(args) == 1
    assert "not trained" in capsys.readouterr().err


def test_cli_unknown_condition(monkeypatch, capsys) -> None:
    monkeypatch.setattr(evaluate, "load_processed", lambda path: signal_frame(n=300))
    assert model_card.main(["--condition", "nope"]) == 1
    assert "unknown condition" in capsys.readouterr().err


def test_card_image_link_points_at_the_real_file(built) -> None:
    _, _, cards, _ = built
    text = (cards / f"{TIER1.id}.md").read_text()
    link = text.split("![Calibration](")[1].split(")")[0]
    assert not link.startswith("/")
    assert (cards / link).resolve().exists()


def test_refuses_when_data_changed_since_training(tmp_path, capsys) -> None:
    mp = pytest.MonkeyPatch()
    models = tmp_path / "models"
    mp.setattr(evaluate, "load_processed", lambda path: signal_frame(n=1500))
    train.main(["--condition", TIER1.id, "--models", "logreg", "--fast", "--out", str(models)])
    mp.setattr(evaluate, "load_processed", lambda path: signal_frame(n=1600))  # file changed
    args = ["--condition", TIER1.id, "--models-dir", str(models), "--cards-dir", str(tmp_path)]
    code = model_card.main([*args, "--figures-dir", str(tmp_path / "f")])
    mp.undo()
    assert code == 1
    assert "changed since training" in capsys.readouterr().err
    assert not (tmp_path / f"{TIER1.id}.md").exists()

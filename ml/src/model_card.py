"""H15 (+ H49's generated cards): where each model is weaker, and saying so plainly.

    python -m ml.src.model_card                   # every trained condition
    python -m ml.src.model_card --condition ID

For every trained condition it writes:

    docs/model-cards/<condition>.md          the model card (intended use, data, metrics,
                                             subgroup results, limitations, release-gate result)
    docs/figures/<condition>/calibration.png predicted vs observed risk, logreg vs XGBoost
    ml/models/<condition>/limitations.json   plain-language limitations the UI shows (H25)

Tier 1 conditions get subgroup results by sex, age group and income group (recall, PR-AUC
and calibration per group). Tier 2 conditions get overall metrics only, as the roadmap says.
Cards are regenerated from the files `train.py` writes, so they never drift from the models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import config, evaluate, features
from .registry import Condition, RegistryError, load_registry

MODELS_DIR = config.REPO_ROOT / "ml" / "models"
CARDS_DIR = config.REPO_ROOT / "docs" / "model-cards"
FIGURES_DIR = config.REPO_ROOT / "docs" / "figures"
MIN_GROUP_ROWS = 30
# A subgroup whose recall is this much below the overall recall gets a limitation sentence.
RECALL_GAP = 0.10
# ...or whose average predicted risk is this far from its observed rate.
CALIBRATION_GAP = 0.05
CALIBRATION_BINS = 10

AGE_GROUPS = {"18-34": (1, 3), "35-49": (4, 6), "50-64": (7, 9), "65+": (10, 13)}
INCOME_GROUPS = {"under $35k": (1, 3), "$35k-$100k": (4, 5), "$100k+": (6, 7)}
SUBGROUP_COLUMNS = {"sex": "Sex", "age_group": "Age group", "income_group": "Household income"}

GENERAL_LIMITATIONS = (
    "This is not a diagnosis. It's a statistical estimate from survey answers, and a health "
    "professional should be consulted about any concern.",
    "The model learned from self-reported answers (people's own account of their diagnoses, "
    "weight and habits), which can be inaccurate.",
    "The survey is a snapshot in time, so the model finds patterns linked with the condition, "
    "not causes. Changing one factor won't necessarily change the risk by the amount shown.",
    "It was trained on US adults (CDC BRFSS 2025). It may not describe people in Canada or "
    "other countries, and it says nothing about children.",
    "It only knows the factors the survey asks about. Family history, lab results, medication "
    "and many other things a clinician would consider aren't included.",
    "Training was unweighted; population percentages on the dashboard use the survey weights.",
)


# --------------------------------------------------------------------------
# Subgroups
# --------------------------------------------------------------------------


def _band(values: pd.Series, groups: dict[str, tuple[int, int]], missing: str) -> pd.Series:
    out = pd.Series(missing, index=values.index, dtype=object)
    numeric = pd.to_numeric(values, errors="coerce")
    for label, (low, high) in groups.items():
        out[numeric.between(low, high)] = label
    return out


def subgroup_labels(x: pd.DataFrame) -> pd.DataFrame:
    """Readable groups for each row: sex, age group, income group."""
    sex = (
        x["sex"].where(x["sex"].notna(), "not stated")
        if "sex" in x
        else pd.Series("not stated", index=x.index)
    )
    return pd.DataFrame(
        {
            "sex": sex.astype(object),
            "age_group": _band(
                x.get("age_band", pd.Series(index=x.index)), AGE_GROUPS, "not stated"
            ),
            "income_group": _band(
                x.get("income_band", pd.Series(index=x.index)), INCOME_GROUPS, "not stated"
            ),
        },
        index=x.index,
    )


def _group_order(column: str, group: object) -> tuple[int, str]:
    """Natural order: defined bands first (youngest/lowest first), "not stated" last."""
    order = {"age_group": list(AGE_GROUPS), "income_group": list(INCOME_GROUPS)}.get(column, [])
    if group == "not stated":
        return (len(order) + 1, "")
    return (order.index(group), "") if group in order else (len(order), str(group))


def subgroup_metrics(
    y: pd.Series, scores: np.ndarray, groups: pd.DataFrame, threshold: float
) -> list[dict[str, Any]]:
    """Recall, PR-AUC and calibration per group, at the model's recall-oriented threshold.

    Groups with fewer than MIN_GROUP_ROWS rows, or without both outcomes, get no metrics
    (too few people to say anything reliable) and a note saying so.
    """
    y = pd.Series(np.asarray(y), index=groups.index).astype(int)
    p = pd.Series(np.asarray(scores, dtype=float), index=groups.index)
    rows = []
    for column in groups.columns:
        found = groups.groupby(column).groups
        for group in sorted(found, key=lambda g: _group_order(column, g)):
            idx = found[group]
            yg, pg = y[idx], p[idx]
            row: dict[str, Any] = {
                "group_by": column,
                "group": str(group),
                "n": len(yg),
                "positives": int(yg.sum()),
                "observed_rate": float(yg.mean()),
                "mean_predicted": float(pg.mean()),
            }
            if len(yg) < MIN_GROUP_ROWS or yg.nunique() < 2:
                row.update(recall=None, pr_auc=None, calibration_gap=None, note="too few people")
            else:
                caught = int(((pg >= threshold) & (yg == 1)).sum())
                row.update(
                    recall=caught / int(yg.sum()),
                    pr_auc=float(average_precision_score(yg, pg)),
                    calibration_gap=float(pg.mean() - yg.mean()),
                    note=None,
                )
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def calibration_table(
    y: np.ndarray, scores: np.ndarray, bins: int = CALIBRATION_BINS
) -> pd.DataFrame:
    """Group people into equal-sized bins by predicted risk; compare predicted vs observed."""
    frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(scores, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), q=min(bins, len(frame)), labels=False)
    table = frame.groupby("bin").agg(
        mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")
    )
    return table.reset_index(drop=True)


def save_calibration_plot(curves: dict[str, pd.DataFrame], path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    top = max(float(t[["mean_predicted", "observed_rate"]].max().max()) for t in curves.values())
    ax.plot([0, top], [0, top], linestyle="--", color="grey", label="perfectly calibrated")
    for name, table in curves.items():
        ax.plot(table["mean_predicted"], table["observed_rate"], marker="o", label=name)
    ax.set_xlabel("average predicted risk (per tenth of people)")
    ax.set_ylabel("share who actually reported the condition")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Limitations (what the UI shows) and the card
# --------------------------------------------------------------------------


def limitations(subgroups: list[dict[str, Any]], overall_recall: float) -> list[str]:
    """General limitations plus one sentence per subgroup where the model is clearly weaker."""
    items = list(GENERAL_LIMITATIONS)
    for row in subgroups:
        if row["recall"] is None or row["group"] == "not stated":
            continue
        who = f"{SUBGROUP_COLUMNS[row['group_by']].lower()} {row['group']}"
        if overall_recall - row["recall"] >= RECALL_GAP:
            items.append(
                f'It misses more cases for people in the group "{who}" '
                f"(catches {row['recall']:.0%} vs {overall_recall:.0%} overall)."
            )
        gap = row["calibration_gap"]
        if gap is not None and abs(gap) >= CALIBRATION_GAP:
            direction = "overestimates" if gap > 0 else "underestimates"
            items.append(
                f'It {direction} risk for people in the group "{who}" '
                f"(average estimate {row['mean_predicted']:.0%} vs {row['observed_rate']:.0%} "
                "actually reported)."
            )
    return items


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


def _num(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def render_card(
    condition: Condition,
    metrics: dict[str, Any],
    meta: dict[str, Any],
    gate_entry: dict[str, Any],
    subgroups: list[dict[str, Any]] | None,
    limits: list[str],
    calibration_figure: str | None = None,
) -> str:
    """The model card as Markdown."""
    best = metrics["best_model"]
    name = condition.display_name
    lines = [
        f"# Model card: {name}",
        "",
        "_Generated by `python -m ml.src.model_card` from the trained model files. "
        "Regenerate after retraining; edit the wording in this generator, not here._",
        "",
        f"**What is predicted:** {condition.description}",
        "",
        f"**Chosen model:** {best} ({metrics['selection']['criterion']}).",
        "",
        "## Intended use",
        "",
        f"- Explore which self-reported risk factors are associated with {name.lower()} "
        "among US adults, in a course project with synthetic patients.",
        "- Give an indicative, population-based estimate for one person, with the factors "
        "that pushed it up or down, **only if** the release gate below allows it.",
        "",
        "## Not intended for",
        "",
        "- Diagnosing anyone, or deciding on tests or treatment.",
        "- Screening real patients, or any decision about insurance, employment or access to care.",
        "- People outside the training population (children; populations outside the US) "
        "without re-validation.",
        "",
        "## Data",
        "",
        f"- Source: {meta['data_source'].upper()} 2025 "
        f"(`{meta['data_file'] or 'processed file'}`); self-reported survey answers.",
        f"- Rows with a known answer: {meta['rows']['train']:,} training, "
        f"{meta['rows']['test']:,} held-out test "
        f"({meta['rows']['positives_test']:,} reporting the condition).",
        f"- Inputs ({len(meta['features'])}): {', '.join(meta['features'])}.",
        "- Missing answers stay missing and are handled inside the model; a missing outcome is "
        'never counted as "no".',
        f"- Seed {meta['seed']}; code version `{meta['git_sha'][:10]}`; "
        f"trained {meta['trained_at']}.",
        "",
        "## Performance on held-out data",
        "",
        "| Model | ROC-AUC | PR-AUC | Brier (no-skill) | Recall @ chosen cut-off | "
        "Precision | Cut-off |",
        "|---|---|---|---|---|---|---|",
    ]
    for model_name, summary in metrics["models"].items():
        d, r = summary["metrics"]["default"], summary["metrics"]["recall_oriented"]
        marker = " (chosen)" if model_name == best else ""
        lines.append(
            f"| {model_name}{marker} | {_num(d['roc_auc'])} | {_num(d['pr_auc'])} | "
            f"{_num(d['brier'])} ({_num(d['brier_no_skill'])}) | {_pct(r['recall'])} | "
            f"{_pct(r['precision'])} | {_num(r['threshold'])} |"
        )
    lines += [
        "",
        "The cut-off is chosen on validation data to catch at least "
        f"{_pct(metrics['models'][best]['metrics']['target_recall'])} of people who report the "
        "condition (screening favours not missing people over avoiding false alarms).",
    ]
    if calibration_figure:
        lines += ["", f"![Calibration]({calibration_figure})"]

    lines += ["", "## Results by group", ""]
    if subgroups is None:
        lines.append(
            "Not computed for this condition (Tier 2: overall metrics only). "
            "Run with `--subgroups` to add them."
        )
    else:
        lines += [
            "Chosen model, at the chosen cut-off. Groups with fewer than "
            f"{MIN_GROUP_ROWS} people are not scored.",
            "",
            "| Group | People | Reporting it | Recall | PR-AUC | Avg. estimate vs actual |",
            "|---|---|---|---|---|---|",
        ]
        for row in subgroups:
            label = f"{SUBGROUP_COLUMNS[row['group_by']]}: {row['group']}"
            lines.append(
                f"| {label} | {row['n']:,} | {_pct(row['observed_rate'])} | "
                f"{_pct(row['recall'])} | {_num(row['pr_auc'])} | "
                f"{_pct(row['mean_predicted'])} vs {_pct(row['observed_rate'])} |"
            )

    state = "enabled" if gate_entry.get("prediction_enabled") else "population analytics only"
    lines += [
        "",
        "## Release gate (ADR-0016)",
        "",
        f"Individual estimates: **{state}**. Signal strength: "
        f"**{gate_entry.get('signal_strength') or '–'}**.",
        "",
        f"> {gate_entry.get('reason', 'Gate not applied yet.')}",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in limits],
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Building one card from the trained files
# --------------------------------------------------------------------------


def build_card(
    df: pd.DataFrame,
    condition: Condition,
    models_dir: Path,
    cards_dir: Path,
    figures_dir: Path,
    with_subgroups: bool,
) -> Path:
    folder = models_dir / condition.id
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((folder / "model_meta.json").read_text(encoding="utf-8"))
    index_path = models_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    gate_entry = index.get("conditions", {}).get(condition.id, {})

    split = evaluate.split_condition(df, condition.id)
    curves, scores = {}, {}
    for name in metrics["models"]:
        model = joblib.load(folder / f"{name}.joblib")
        scores[name] = model.predict_proba(split.x_test)[:, 1]
        curves[name] = calibration_table(split.y_test, scores[name])
    figure = save_calibration_plot(
        curves,
        figures_dir / condition.id / "calibration.png",
        f"{condition.display_name}: calibration",
    )

    best = metrics["best_model"]
    recall_point = metrics["models"][best]["metrics"]["recall_oriented"]
    subgroups = None
    if with_subgroups:
        subgroups = subgroup_metrics(
            split.y_test, scores[best], subgroup_labels(split.x_test), recall_point["threshold"]
        )
    limits = limitations(subgroups or [], recall_point["recall"])
    (folder / "limitations.json").write_text(
        json.dumps({"condition": condition.id, "limitations": limits}, indent=2) + "\n",
        encoding="utf-8",
    )

    cards_dir.mkdir(parents=True, exist_ok=True)
    card_path = cards_dir / f"{condition.id}.md"
    relative_figure = Path("..") / "figures" / condition.id / figure.name
    if figures_dir.resolve() != FIGURES_DIR.resolve():
        relative_figure = figure
    card_path.write_text(
        render_card(condition, metrics, meta, gate_entry, subgroups, limits, str(relative_figure)),
        encoding="utf-8",
    )
    return card_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate model cards (H15).")
    parser.add_argument("--condition", help="one condition id (default: every trained condition)")
    parser.add_argument("--data", type=Path, default=evaluate.DEFAULT_DATA)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--cards-dir", type=Path, default=CARDS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--subgroups", action="store_true", help="subgroup results for every tier, not just Tier 1"
    )
    args = parser.parse_args(argv)

    registry = load_registry()
    try:
        conditions = [registry.get(args.condition)] if args.condition else registry.conditions
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    df = features.add_features(evaluate.load_processed(args.data))
    written = 0
    for condition in conditions:
        if not (args.models_dir / condition.id / "metrics.json").exists():
            print(f"{condition.id}: not trained yet (run python -m ml.src.train)", file=sys.stderr)
            continue
        path = build_card(
            df,
            condition,
            args.models_dir,
            args.cards_dir,
            args.figures_dir,
            with_subgroups=args.subgroups or condition.tier == 1,
        )
        print(f"{condition.id}: wrote {path}")
        written += 1
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())

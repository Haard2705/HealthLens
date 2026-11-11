"""H14: SHAP explanations, for one prediction (what `/predict` returns) and for the whole model.

A SHAP explanation splits one prediction into a starting point plus one push per factor:

    base value + contribution(BMI) + contribution(age) + ... = the model's output

in **log-odds** (the scale the models work on internally). A positive contribution raised
this person's estimate, a negative one lowered it. Contributions add up exactly, so they are
an honest account of what the model did, not a story told afterwards.

How they're computed (no `shap` package needed at prediction time, so the backend stays light):

* **XGBoost**: XGBoost's built-in TreeSHAP (`pred_contribs=True`), the same numbers
  `shap.TreeExplainer` gives (a CI test checks that).
* **Logistic regression**: exact linear SHAP, `coefficient × (value − training average)`.

The preprocessor turns one answer into several model columns (e.g. `sex` → `sex = female`,
`sex = male`; `income_band` → the number and a "was missing" flag). Their contributions are
**summed back to the canonical feature**, so explanations read "income_band", never
`num__missingindicator_income_band`.

    python -m ml.src.explain --condition ID     # global importance figures + JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config, evaluate, features
from .registry import RegistryError, load_registry

FIGURES_DIR = config.REPO_ROOT / "docs" / "figures"
MODELS_DIR = config.REPO_ROOT / "ml" / "models"
EXPLAINER_FILE = "explainer.joblib"
BACKGROUND_ROWS = 2000
GLOBAL_SAMPLE_ROWS = 2000


def source_feature(encoded: str) -> str:
    """The canonical feature a preprocessor output column came from.

    num__bmi → bmi · num__missingindicator_bmi → bmi · bin__phys_active → phys_active ·
    cat__smoker_some_days → smoker
    """
    group, _, rest = encoded.partition("__")
    if not rest:
        return encoded
    rest = rest.removeprefix("missingindicator_")
    if group == "cat":
        for column in sorted(features.CATEGORICAL, key=len, reverse=True):
            if rest.startswith(f"{column}_"):
                return column
    return rest


def _sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-np.asarray(margin)))


@dataclass(frozen=True)
class Explainer:
    """Explains one fitted pipeline. Build with `build_explainer`; save with joblib."""

    model: Pipeline
    kind: str  # "linear" or "tree"
    encoded_names: tuple[str, ...]
    sources: tuple[str, ...]  # canonical feature for each encoded column
    predictors: tuple[str, ...]  # canonical features, in the model's input order
    background_mean: np.ndarray | None = None  # linear only: average encoded row in training

    # ---- encoded-column contributions ----------------------------------------------------

    def _contributions(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(base value per row, contribution per row × encoded column), in log-odds."""
        encoded = np.asarray(self.model.named_steps["prep"].transform(x), dtype="float64")
        clf = self.model.named_steps["clf"]
        if self.kind == "linear":
            coef = clf.coef_.ravel()
            contrib = coef * (encoded - self.background_mean)
            base = float(clf.intercept_[0] + coef @ self.background_mean)
            return np.full(len(x), base), contrib

        from xgboost import DMatrix

        raw = clf.get_booster().predict(
            DMatrix(encoded), pred_contribs=True, iteration_range=_iteration_range(clf)
        )
        return raw[:, -1], raw[:, :-1]

    def contributions(self, x: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """(base values, contributions summed per canonical feature), one row per input row."""
        base, contrib = self._contributions(x)
        by_column = pd.DataFrame(contrib, columns=list(self.encoded_names), index=x.index)
        grouped = by_column.T.groupby(list(self.sources), sort=False).sum().T
        return base, grouped.reindex(columns=list(self.predictors), fill_value=0.0)

    # ---- what /predict calls ------------------------------------------------------------

    def explain_one(self, row: pd.DataFrame) -> list[dict[str, Any]]:
        """`[{feature, value, contribution}]` for a single model-frame row, biggest push first.

        `row` is one row of `features.model_frame(df, meta["features"])`. Contributions are in
        log-odds; positive means "raised this estimate".
        """
        return self.explain(row)["contributions"]

    def explain(self, row: pd.DataFrame) -> dict[str, Any]:
        """explain_one plus the base value, total margin and predicted probability."""
        if len(row) != 1:
            raise ValueError(f"explain one row at a time (got {len(row)})")
        base, grouped = self.contributions(row)
        items = [
            {
                "feature": name,
                "value": _readable(name, row.iloc[0][name]),
                "contribution": float(grouped.iloc[0][name]),
            }
            for name in self.predictors
        ]
        items.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        margin = float(base[0] + grouped.iloc[0].sum())
        return {
            "base_value": float(base[0]),
            "margin": margin,
            "probability": float(_sigmoid(margin)),
            "units": "log-odds",
            "contributions": items,
        }


def _iteration_range(clf: Any) -> tuple[int, int]:
    """Use the same trees as predict_proba: up to the early-stopping best, if there was one."""
    try:
        return (0, int(clf.best_iteration) + 1)
    except AttributeError:
        return (0, 0)  # XGBoost: (0, 0) means every tree


def _readable(name: str, value: Any) -> Any:
    """Model-frame value → JSON-friendly: NaN → None, yes/no floats → True/False."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if name in features.BINARY:
        return bool(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_explainer(model: Pipeline, x_background: pd.DataFrame) -> Explainer:
    """Explainer for a fitted pipeline. `x_background` = training rows (model-frame columns).

    For logistic regression the background sets the "average person" that contributions are
    measured against; tree explanations use the trees' own training statistics instead.
    """
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    encoded_names = tuple(prep.get_feature_names_out())
    predictors = tuple(x_background.columns)
    sources = tuple(source_feature(n) for n in encoded_names)
    if isinstance(clf, LogisticRegression):
        encoded = np.asarray(prep.transform(x_background), dtype="float64")
        return Explainer(model, "linear", encoded_names, sources, predictors, encoded.mean(axis=0))
    if type(clf).__name__ == "XGBClassifier":
        return Explainer(model, "tree", encoded_names, sources, predictors)
    raise TypeError(f"no explainer for {type(clf).__name__}")


def background_sample(
    x_train: pd.DataFrame, rows: int = BACKGROUND_ROWS, seed: int = config.RANDOM_SEED
) -> pd.DataFrame:
    return x_train.sample(n=min(rows, len(x_train)), random_state=seed)


# --------------------------------------------------------------------------
# Global importance (for the report and the dashboard)
# --------------------------------------------------------------------------


def global_importance(explainer: Explainer, x: pd.DataFrame) -> pd.DataFrame:
    """Mean |contribution| per feature over `x`, plus how often it raised the estimate."""
    _, grouped = explainer.contributions(x)
    table = pd.DataFrame(
        {
            "feature": grouped.columns,
            "mean_abs_contribution": grouped.abs().mean().to_numpy(),
            "share_raising": (grouped > 0).mean().to_numpy(),
        }
    )
    return table.sort_values("mean_abs_contribution", ascending=False, kind="stable").reset_index(
        drop=True
    )


def save_figures(explainer: Explainer, x: pd.DataFrame, folder: Path, title: str) -> list[Path]:
    """Bar chart of global importance (always) and a SHAP beeswarm (when `shap` is installed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    table = global_importance(explainer, x).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(table) + 1.2))
    ax.barh(table["feature"], table["mean_abs_contribution"], color="#4C72B0")
    ax.set_xlabel("mean |SHAP contribution| (log-odds)")
    ax.set_title(f"{title}: what moves the estimate most")
    fig.tight_layout()
    written = [folder / "shap_bar.png"]
    fig.savefig(written[0], dpi=150)
    plt.close(fig)

    try:
        import shap
    except ImportError:
        return written
    _, grouped = explainer.contributions(x)
    colour = x[list(grouped.columns)].apply(
        lambda s: s if pd.api.types.is_numeric_dtype(s) else s.astype("category").cat.codes
    )
    shap.summary_plot(grouped.to_numpy(), colour, show=False, plot_size=(8, 6))
    plt.title(f"{title}: SHAP values per person")
    plt.tight_layout()
    written.append(folder / "shap_beeswarm.png")
    plt.savefig(written[1], dpi=150)
    plt.close("all")
    return written


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Global SHAP figures for trained models.")
    parser.add_argument("--condition", help="one condition id (default: every trained condition)")
    parser.add_argument("--model", help="logreg or xgboost (default: the condition's best model)")
    parser.add_argument("--data", type=Path, default=evaluate.DEFAULT_DATA)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--out", type=Path, default=FIGURES_DIR)
    parser.add_argument("--rows", type=int, default=GLOBAL_SAMPLE_ROWS)
    args = parser.parse_args(argv)

    registry = load_registry()
    try:
        conditions = [registry.get(args.condition)] if args.condition else registry.conditions
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    df = features.add_features(evaluate.load_processed(args.data))
    done = 0
    for condition in conditions:
        folder = args.models_dir / condition.id
        if not (folder / "metrics.json").exists():
            print(f"{condition.id}: not trained yet (run python -m ml.src.train)", file=sys.stderr)
            continue
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        name = args.model or metrics["best_model"]
        model = joblib.load(folder / f"{name}.joblib")
        split = evaluate.split_condition(df, condition.id)
        explainer = build_explainer(model, background_sample(split.x_train))
        sample = background_sample(split.x_test, rows=args.rows)

        out = args.out / condition.id
        paths = save_figures(explainer, sample, out, f"{condition.display_name} ({name})")
        table = global_importance(explainer, sample)
        (out / f"{name}_global_importance.json").write_text(
            table.to_json(orient="records", indent=2), encoding="utf-8"
        )
        top = ", ".join(table["feature"].head(5))
        print(f"{condition.id} ({name}): top factors {top}")
        print("  saved " + ", ".join(str(p) for p in paths))
        done += 1
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())

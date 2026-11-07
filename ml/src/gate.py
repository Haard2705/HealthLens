"""H50: the prediction release gate, which decides which conditions may show individual estimates.

A model with ROC-AUC 0.58 that still prints "your risk is 34%" is worse than no model. So
every condition passes through one written rule (ADR-0016) before the API or UI may show a
personal estimate. Conditions that don't pass still get population analytics, with the reason.

The rule (all must hold):

1. The condition is **not sensitive** in the registry (sensitive ones are analytics only).
2. It was **trained** successfully.
3. Held-out (test) **ROC-AUC ≥ 0.70** for the chosen model.
4. **Brier score < the no-skill Brier** (its probabilities beat always guessing the average).

Signal strength (for the label next to an estimate): strong ≥ 0.80, moderate 0.70-0.80,
weak < 0.70.

The result is written into ``ml/models/index.json`` as ``prediction_enabled``,
``signal_strength`` and ``reason``. The API and UI only *read* these fields; they never
re-implement the rule.

    python -m ml.src.gate        # re-applies the rule to the current index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import Registry, load_registry

MIN_ROC_AUC = 0.70
STRONG_ROC_AUC = 0.80
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
INDEX_FILE = "index.json"


@dataclass(frozen=True)
class Decision:
    prediction_enabled: bool
    signal_strength: str | None  # "strong" | "moderate" | "weak" | None (no usable model)
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "prediction_enabled": self.prediction_enabled,
            "signal_strength": self.signal_strength,
            "reason": self.reason,
        }


def signal_strength(roc_auc: float) -> str:
    if roc_auc >= STRONG_ROC_AUC:
        return "strong"
    if roc_auc >= MIN_ROC_AUC:
        return "moderate"
    return "weak"


def decide(entry: dict[str, Any], sensitive: bool) -> Decision:
    """Apply the rule to one condition's index entry (written by train.py)."""
    if entry.get("status") != "trained":
        why = entry.get("reason", "it has not been trained")
        return Decision(False, None, f"Population analytics only: no model is available ({why}).")

    auc = float(entry["test_roc_auc"])
    brier, no_skill = float(entry["test_brier"]), float(entry["test_brier_no_skill"])
    strength = signal_strength(auc)

    if sensitive:
        return Decision(
            False,
            strength,
            "Population analytics only: this is a sensitive condition, so we don't show "
            "individual estimates whatever the model's accuracy.",
        )
    if auc < MIN_ROC_AUC:
        return Decision(
            False,
            strength,
            f"Population analytics only: the model's accuracy on held-out data (ROC-AUC "
            f"{auc:.2f}) is below our minimum of {MIN_ROC_AUC:.2f}, so a personal estimate "
            "could mislead.",
        )
    if brier >= no_skill:
        return Decision(
            False,
            strength,
            f"Population analytics only: the model's risk percentages (Brier {brier:.3f}) are "
            f"no better than always guessing the average rate (Brier {no_skill:.3f}).",
        )
    return Decision(
        True,
        strength,
        f"Individual estimates enabled: held-out ROC-AUC {auc:.2f} ({strength} signal) and "
        f"risk percentages better than guessing the average (Brier {brier:.3f} vs "
        f"{no_skill:.3f}).",
    )


def apply_gate(index: dict[str, Any], registry: Registry) -> dict[str, Any]:
    """Return the index with every condition's decision filled in. Conditions in the registry
    but missing from the index are added as not trained."""
    conditions = dict(index.get("conditions", {}))
    for condition in registry.conditions:
        entry = dict(conditions.get(condition.id, {"display_name": condition.display_name}))
        entry["sensitive"] = condition.sensitive
        entry.update(decide(entry, condition.sensitive).as_dict())
        conditions[condition.id] = entry
    return {
        **index,
        "conditions": conditions,
        "release_gate": {
            "min_roc_auc": MIN_ROC_AUC,
            "strong_roc_auc": STRONG_ROC_AUC,
            "requires_brier_below_no_skill": True,
            "sensitive_conditions_blocked": True,
            "decision_record": "docs/adr/0016-prediction-release-gate.md",
        },
    }


def gate_file(models_dir: Path = MODELS_DIR, registry: Registry | None = None) -> dict[str, Any]:
    """Re-apply the gate to models_dir/index.json in place and return the result."""
    path = Path(models_dir) / INDEX_FILE
    index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"conditions": {}}
    gated = apply_gate(index, registry or load_registry())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gated, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return gated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the prediction release gate (H50).")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args(argv)
    gated = gate_file(args.models_dir)
    for condition_id, entry in gated["conditions"].items():
        state = "ENABLED " if entry["prediction_enabled"] else "analytics"
        strength = entry["signal_strength"] or "-"
        print(f"{condition_id:<16} {state}  {strength:<8}  {entry['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

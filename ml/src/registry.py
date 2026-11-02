"""Condition registry: the single source of truth for what HealthLens models.

Every condition is one entry in ``conditions.yaml``. Training, the API and the UI read
this registry, so no other code mentions a specific condition by name. Adding a
condition is a registry entry plus its mapping in the source adapter, nothing else.

The set is deliberately frozen: ``ml/tests/test_registry.py`` pins it, so changing
it is a conscious edit to both files.

Run ``python -m ml.src.registry`` to print the registry.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_REGISTRY_PATH = Path(__file__).with_name("conditions.yaml")

ALLOWED_TIERS = (1, 2)
ALLOWED_SOURCES = ("brfss",)
ALLOWED_BODY_SYSTEMS = (
    "cardiometabolic",
    "respiratory",
    "musculoskeletal",
    "mental_health",
    "renal",
)

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_REQUIRED_FIELDS = frozenset(
    {"id", "display_name", "body_system", "tier", "source", "outcome_column", "description"}
)
_OPTIONAL_FIELDS = frozenset({"sensitive", "comorbidity_features"})
_ALL_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS


class RegistryError(ValueError):
    """The registry file is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class Condition:
    """One modelled condition. Immutable, so it is safe to share across threads and requests."""

    id: str
    display_name: str
    body_system: str
    tier: int
    source: str
    outcome_column: str
    description: str
    sensitive: bool = False
    comorbidity_features: tuple[str, ...] = ()


@dataclass(frozen=True)
class Registry:
    conditions: tuple[Condition, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.conditions)

    @property
    def outcome_columns(self) -> tuple[str, ...]:
        return tuple(c.outcome_column for c in self.conditions)

    def get(self, condition_id: str) -> Condition:
        for condition in self.conditions:
            if condition.id == condition_id:
                return condition
        raise RegistryError(f"unknown condition {condition_id!r}; known: {', '.join(self.ids)}")

    def by_tier(self, tier: int) -> tuple[Condition, ...]:
        return tuple(c for c in self.conditions if c.tier == tier)


# --------------------------------------------------------------------------
# Parsing and validation
# --------------------------------------------------------------------------


def parse_registry(raw: Any) -> Registry:
    """Validate raw parsed data and build a Registry.

    All problems found are reported together in one RegistryError.
    """
    if not isinstance(raw, Mapping) or not isinstance(raw.get("conditions"), list):
        raise RegistryError("registry must be a mapping with a 'conditions' list")
    entries = raw["conditions"]
    if not entries:
        raise RegistryError("registry must contain at least one condition")

    errors: list[str] = []
    conditions: list[Condition] = []
    for index, entry in enumerate(entries):
        condition = _parse_entry(entry, index, errors)
        if condition is not None:
            conditions.append(condition)

    errors.extend(_cross_entry_errors(conditions))
    if errors:
        raise RegistryError("invalid registry:\n  - " + "\n  - ".join(errors))
    return Registry(conditions=tuple(conditions))


def _parse_entry(entry: Any, index: int, errors: list[str]) -> Condition | None:
    label = f"entry {index}"
    if not isinstance(entry, Mapping):
        errors.append(f"{label}: must be a mapping")
        return None

    label = f"entry {index} ({entry.get('id', '?')})"
    before = len(errors)

    unknown = sorted(set(entry) - _ALL_FIELDS)
    if unknown:
        errors.append(f"{label}: unknown field(s) {unknown}")
    missing = sorted(_REQUIRED_FIELDS - set(entry))
    if missing:
        errors.append(f"{label}: missing field(s) {missing}")
        return None

    _check_name(entry["id"], "id", label, errors)
    _check_name(entry["outcome_column"], "outcome_column", label, errors)
    for field in ("display_name", "description"):
        if not isinstance(entry[field], str) or not entry[field].strip():
            errors.append(f"{label}: {field} must be a non-empty string")
    _check_choice(entry["body_system"], "body_system", ALLOWED_BODY_SYSTEMS, label, errors)
    _check_choice(entry["source"], "source", ALLOWED_SOURCES, label, errors)
    if type(entry["tier"]) is not int or entry["tier"] not in ALLOWED_TIERS:
        errors.append(f"{label}: tier must be one of {list(ALLOWED_TIERS)}")

    sensitive = entry.get("sensitive", False)
    if not isinstance(sensitive, bool):
        errors.append(f"{label}: sensitive must be true or false")

    comorbidity = entry.get("comorbidity_features", [])
    if not isinstance(comorbidity, list) or not all(isinstance(c, str) for c in comorbidity):
        errors.append(f"{label}: comorbidity_features must be a list of column names")
        comorbidity = []

    if len(errors) > before:
        return None
    return Condition(
        id=entry["id"],
        display_name=entry["display_name"].strip(),
        body_system=entry["body_system"],
        tier=entry["tier"],
        source=entry["source"],
        outcome_column=entry["outcome_column"],
        description=entry["description"].strip(),
        sensitive=sensitive,
        comorbidity_features=tuple(comorbidity),
    )


def _check_name(value: Any, field: str, label: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not _NAME_PATTERN.match(value):
        errors.append(f"{label}: invalid {field} {value!r} (use lowercase letters, digits, _)")


def _check_choice(
    value: Any, field: str, allowed: Sequence[Any], label: str, errors: list[str]
) -> None:
    if value not in allowed:
        errors.append(f"{label}: {field} {value!r} not in {list(allowed)}")


def _cross_entry_errors(conditions: Sequence[Condition]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_outcomes: set[str] = set()
    for condition in conditions:
        if condition.id in seen_ids:
            errors.append(f"duplicate condition id {condition.id!r}")
        seen_ids.add(condition.id)
        if condition.outcome_column in seen_outcomes:
            errors.append(f"duplicate outcome_column {condition.outcome_column!r}")
        seen_outcomes.add(condition.outcome_column)

    for condition in conditions:
        others = {c.outcome_column for c in conditions if c.id != condition.id}
        if len(set(condition.comorbidity_features)) != len(condition.comorbidity_features):
            errors.append(f"{condition.id}: duplicate comorbidity_features")
        for column in condition.comorbidity_features:
            if column == condition.outcome_column:
                errors.append(f"{condition.id}: comorbidity_features includes its own outcome")
            elif column not in others:
                errors.append(
                    f"{condition.id}: comorbidity feature {column!r} is not another "
                    "condition's outcome_column"
                )
    return errors


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_registry(path: Path | str = DEFAULT_REGISTRY_PATH) -> Registry:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RegistryError(f"registry file not found: {path}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RegistryError(f"registry file is not valid YAML: {path}\n{exc}") from exc
    return parse_registry(raw)


# --------------------------------------------------------------------------
# Helpers used by the rest of the pipeline
# --------------------------------------------------------------------------


def feature_columns(
    registry: Registry, condition_id: str, base_predictors: Iterable[str]
) -> tuple[str, ...]:
    """Columns a condition's model may use as predictors (the leakage guard).

    Outcome columns of conditions are never predictors, unless a condition explicitly
    opts in to another condition's outcome via ``comorbidity_features``. A condition's
    own outcome is never a predictor. Order follows ``base_predictors``, with opt-in
    columns appended.
    """
    condition = registry.get(condition_id)
    outcomes = set(registry.outcome_columns)
    opted_in = set(condition.comorbidity_features)

    columns = [c for c in base_predictors if c not in outcomes or c in opted_in]
    columns.extend(c for c in condition.comorbidity_features if c not in columns)
    return tuple(columns)


def validate_against_schema(registry: Registry, schema_columns: Iterable[str]) -> None:
    """Check that every outcome column exists in the canonical schema."""
    available = set(schema_columns)
    missing = sorted(set(registry.outcome_columns) - available)
    if missing:
        raise RegistryError(f"outcome column(s) missing from the canonical schema: {missing}")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    del argv  # no options yet
    registry = load_registry()
    lines = [f"{'id':<16}{'tier':<6}{'system':<17}{'outcome column':<18}sensitive"]
    lines.extend(
        f"{c.id:<16}{c.tier:<6}{c.body_system:<17}{c.outcome_column:<18}{c.sensitive}"
        for c in registry.conditions
    )
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

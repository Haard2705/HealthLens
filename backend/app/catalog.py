"""Condition catalogue: registry fields merged with the release-gate decision from index.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.src.registry import Registry

INDEX_FILE = "index.json"
GATE_FIELDS = ("prediction_enabled", "signal_strength", "reason")


def read_json(path: Path) -> dict[str, Any] | None:
    """A JSON object from disk, or None when the file is missing or unreadable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_index(models_dir: Path) -> dict[str, Any]:
    index = read_json(models_dir / INDEX_FILE) or {}
    conditions = index.get("conditions")
    return conditions if isinstance(conditions, dict) else {}


def list_conditions(registry: Registry, models_dir: Path) -> list[dict[str, Any]]:
    index = read_index(models_dir)
    items = []
    for c in registry.conditions:
        entry = index.get(c.id) if isinstance(index.get(c.id), dict) else {}
        items.append(
            {
                "id": c.id,
                "display_name": c.display_name,
                "plain_name": c.plain_name,
                "body_system": c.body_system,
                "tier": c.tier,
                "description": c.description,
                "sensitive": c.sensitive,
                **{field: entry.get(field) for field in GATE_FIELDS},
            }
        )
    return items


def gate_entry(models_dir: Path, condition_id: str) -> dict[str, Any]:
    entry = read_index(models_dir).get(condition_id)
    return entry if isinstance(entry, dict) else {}


def prediction_enabled_count(registry: Registry, models_dir: Path) -> int:
    return sum(1 for c in list_conditions(registry, models_dir) if c["prediction_enabled"] is True)

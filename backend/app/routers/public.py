"""Public, non-patient endpoints: conditions, model metrics, aggregate stats."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request

from .. import catalog, stats
from ..errors import AppError
from ..state import AppState

router = APIRouter(tags=["public"])


def _state(request: Request) -> AppState:
    return request.app.state.hl


def _known_condition(state: AppState, condition_id: str):
    ids = state.registry.ids
    if condition_id not in ids:  # whitelist: never used unless it is a registry id
        raise AppError(422, "invalid_parameter", f"condition must be one of: {', '.join(ids)}")
    return state.registry.get(condition_id)


@router.get("/conditions")
def conditions(request: Request) -> dict:
    state = _state(request)
    return {"items": catalog.list_conditions(state.registry, state.settings.models_path)}


@router.get("/models")
def model_metrics(request: Request, condition: Annotated[str, Query(max_length=40)]) -> dict:
    state = _state(request)
    cond = _known_condition(state, condition)
    path = state.settings.models_path / cond.id / "metrics.json"
    metrics = catalog.read_json(path)
    if metrics is None:
        raise AppError(404, "model_not_trained", f"Model not trained yet for {cond.id}.")
    return {"condition": cond.id, "metrics": metrics}


@router.get("/stats/prevalence")
def prevalence(
    request: Request,
    condition: Annotated[str, Query(max_length=40)],
    by: Annotated[str, Query(max_length=20)],
) -> dict:
    state = _state(request)
    cond = _known_condition(state, condition)
    if by not in stats.BY_OPTIONS:
        raise AppError(
            422, "invalid_parameter", f"by must be one of: {', '.join(stats.BY_OPTIONS)}"
        )
    try:
        return stats.prevalence(state.settings.processed_path, cond.outcome_column, cond.id, by)
    except KeyError as exc:
        raise AppError(
            503, "data_unavailable", "Survey data is not in the expected shape."
        ) from exc

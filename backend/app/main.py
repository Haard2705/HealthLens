"""FastAPI application factory.

uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from ml.src.registry import load_registry

from . import models_store
from .config import Settings
from .data.repository import PatientRepository
from .errors import install_error_handlers
from .routers import admin, auth, doctor, me, predict, public
from .security import LoginRateLimiter
from .state import AppState
from .storage import open_storage
from .users import seed_demo_users

API_PREFIX = "/api/v1"
logger = logging.getLogger(__name__)


def build_state(settings: Settings, repo: PatientRepository | None = None) -> AppState:
    """Pick the storage backend (memory unless DATABASE_URL is set) and build the app state."""
    backend = open_storage(settings, repo)
    registry = load_registry()
    users = backend.users or seed_demo_users(backend.repo, settings.effective_demo_password)
    logger.info("storage: %s (patients from %s)", backend.name, backend.repo.source)
    return AppState(
        settings=settings,
        users=users,
        audit=backend.audit,
        repo=backend.repo,
        limiter=LoginRateLimiter(),
        registry=registry,
        storage=backend.name,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the trained models once, here, and never again while the process lives."""
    state: AppState = app.state.hl
    app.state.hl = replace(
        state, models=models_store.load(state.settings.models_path, state.registry)
    )
    yield


async def _request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request.state.request_id = uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


def create_app(settings: Settings | None = None, repo: PatientRepository | None = None) -> FastAPI:
    """Build the app. Raises ConfigError (via Settings) if production config is unsafe."""
    settings = settings or Settings()
    app = FastAPI(title="HealthLens API", version="0.1.0", lifespan=_lifespan)
    app.state.hl = build_state(settings, repo)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Content-Type"],
    )
    app.middleware("http")(_request_id_middleware)
    install_error_handlers(app)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    v1 = APIRouter(prefix=API_PREFIX)
    for module in (auth, public, predict, doctor, me, admin):
        v1.include_router(module.router)
    app.include_router(v1)
    return app


def _default_app() -> FastAPI:
    return create_app()


app = _default_app()

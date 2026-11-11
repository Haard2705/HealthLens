"""FastAPI application factory.

uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from ml.src.registry import load_registry

from .audit import InMemoryAuditStore
from .config import Settings
from .data.repository import PatientRepository, SyntheticRepository
from .errors import install_error_handlers
from .routers import admin, auth, doctor, me, public
from .security import LoginRateLimiter
from .state import AppState
from .users import seed_demo_users

API_PREFIX = "/api/v1"


def build_state(settings: Settings, repo: PatientRepository | None = None) -> AppState:
    repo = repo or SyntheticRepository()
    return AppState(
        settings=settings,
        users=seed_demo_users(repo, settings.effective_demo_password),
        audit=InMemoryAuditStore(),
        repo=repo,
        limiter=LoginRateLimiter(),
        registry=load_registry(),
    )


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
    app = FastAPI(title="HealthLens API", version="0.1.0")
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
    for module in (auth, public, doctor, me, admin):
        v1.include_router(module.router)
    app.include_router(v1)
    return app


def _default_app() -> FastAPI:
    return create_app()


app = _default_app()

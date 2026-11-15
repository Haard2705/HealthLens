"""Everything the app needs at runtime, built once in ``create_app`` and held on app.state."""

from __future__ import annotations

from dataclasses import dataclass

from ml.src.registry import Registry

from .audit import AuditStore
from .config import Settings
from .data.repository import PatientRepository
from .security import LoginRateLimiter
from .users import UserStore


@dataclass(frozen=True)
class AppState:
    settings: Settings
    users: UserStore
    audit: AuditStore
    repo: PatientRepository
    limiter: LoginRateLimiter
    registry: Registry

"""Which stores the app runs on: memory by default, Postgres when ``DATABASE_URL`` is set.

Zero-setup is the rule (H17): a clone with no environment variables runs entirely in memory on
the synthetic patients, so a marker can `pip install` and `uvicorn` with no database. Setting
``DATABASE_URL`` swaps in Postgres-backed implementations of the *same* interfaces -
``PatientRepository``, ``UserStore`` and ``AuditStore`` - and nothing else in the app changes.
``GET /admin/system`` reports which one is live as ``"storage"``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .audit import AuditStore, InMemoryAuditStore
from .config import Settings
from .data.repository import PatientRepository, SyntheticRepository
from .users import UserStore

MEMORY = "memory"
POSTGRES = "postgres"


@dataclass(frozen=True)
class StorageBackend:
    """The stores the app state is built from. ``users=None`` means "seed the demo accounts"."""

    name: str
    repo: PatientRepository
    audit: AuditStore
    users: UserStore | None = None


def open_storage(settings: Settings, repo: PatientRepository | None = None) -> StorageBackend:
    """Pick the backend. An explicit ``repo`` (tests, ``create_app(repo=...)``) always wins."""
    if repo is not None or not settings.database_url:
        return StorageBackend(MEMORY, repo or SyntheticRepository(), InMemoryAuditStore())
    from .db import stores  # local import: SQLAlchemy is only needed when a URL is set

    return stores.open_postgres(settings.database_url)

"""Shared fixtures. No Postgres, no network: everything runs against in-memory stores."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app

DEMO_PASSWORD = "demo-pass-for-tests"
ADMIN = "admin@demo.healthlens.test"
CHEN = "dr.chen@demo.healthlens.test"
OKAFOR = "dr.okafor@demo.healthlens.test"
FATIMA = "fatima.hassan@demo.healthlens.test"  # P-0106, dr-chen
JORDAN = "jordan.avery@demo.healthlens.test"  # P-0101, dr-chen
TOMAS = "tomas.rivera@demo.healthlens.test"  # P-0107, dr-okafor


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        jwt_secret="test-secret-test-secret-test-secret-123",
        demo_password=DEMO_PASSWORD,
        model_dir=tmp_path / "models",
        processed_data_path=tmp_path / "missing.parquet",
        _env_file=None,
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def anon(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def login_as(app: FastAPI) -> Callable[[str], TestClient]:
    """Return a client (own cookie jar) logged in as the given demo email."""

    def _login(email: str, password: str = DEMO_PASSWORD) -> TestClient:
        client = TestClient(app)
        response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
        assert response.status_code == 200, response.text
        return client

    return _login


def audit_events(app: FastAPI, **filters: str) -> list[dict]:
    """Read the audit log straight from the store (not via the API)."""
    return [e.as_dict() for e in app.state.hl.audit.read(**filters)]

"""Shared fixtures. No Postgres, no network: everything runs against in-memory stores."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.tests import models_fixture

DEMO_PASSWORD = "demo-pass-for-tests"
ADMIN = "admin@demo.healthlens.test"
CHEN = "dr.chen@demo.healthlens.test"
OKAFOR = "dr.okafor@demo.healthlens.test"
FATIMA = "fatima.hassan@demo.healthlens.test"  # P-0106, dr-chen
JORDAN = "jordan.avery@demo.healthlens.test"  # P-0101, dr-chen
TOMAS = "tomas.rivera@demo.healthlens.test"  # P-0107, dr-okafor

# Conditions the `trained_app` fixture has artifacts for.
ENABLED_CONDITIONS = ("diabetes", "heart_disease")
BLOCKED_CONDITIONS = ("asthma", "depression")
WEAK_REASON = (
    "Population analytics only: the model's accuracy on held-out data (ROC-AUC 0.58) is "
    "below our minimum of 0.70, so a personal estimate could mislead."
)
SENSITIVE_REASON = (
    "Population analytics only: this is a sensitive condition, so we don't show individual "
    "estimates whatever the model's accuracy."
)


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
def app(settings: Settings) -> Iterator[FastAPI]:
    """The app with its lifespan run once, so the model store is loaded as in production."""
    application = create_app(settings)
    with TestClient(application):
        yield application


@pytest.fixture
def trained_app(settings: Settings) -> Iterator[FastAPI]:
    """An app whose model folder holds small, really-trained artifacts (see models_fixture).

    ``diabetes`` and ``heart_disease`` pass the gate; ``asthma`` is too weak and ``depression``
    is sensitive, so both are blocked - the same situation the real index.json describes.
    """
    for condition_id in ENABLED_CONDITIONS:
        models_fixture.write_condition(settings.models_path, condition_id)
    for condition_id in BLOCKED_CONDITIONS:
        models_fixture.write_condition(settings.models_path, condition_id)
    models_fixture.write_index(
        settings.models_path,
        {
            **{c: models_fixture.gate_decision(enabled=True) for c in ENABLED_CONDITIONS},
            "asthma": models_fixture.gate_decision(
                enabled=False, strength="weak", reason=WEAK_REASON
            ),
            "depression": models_fixture.gate_decision(
                enabled=False, strength="moderate", reason=SENSITIVE_REASON
            ),
        },
    )
    application = create_app(settings)
    with TestClient(application):
        yield application


@pytest.fixture
def anon(app: FastAPI) -> TestClient:
    return TestClient(app)


def client_for(application: FastAPI, email: str, password: str = DEMO_PASSWORD) -> TestClient:
    """A client with its own cookie jar, logged in as the given demo account."""
    client = TestClient(application)
    response = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert response.status_code == 200, response.text
    return client


@pytest.fixture
def login_as(app: FastAPI) -> Callable[[str], TestClient]:
    """Return a client (own cookie jar) logged in as the given demo email."""

    def _login(email: str, password: str = DEMO_PASSWORD) -> TestClient:
        return client_for(app, email, password)

    return _login


def audit_events(app: FastAPI, **filters: str) -> list[dict]:
    """Read the audit log straight from the store (not via the API)."""
    return [e.as_dict() for e in app.state.hl.audit.read(**filters)]

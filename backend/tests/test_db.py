"""H17 storage: URL handling, row mapping, backend selection, and optional Postgres tests.

Everything above the ``integration`` mark runs with no database at all. The integration tests
skip themselves unless ``DATABASE_URL`` is set, so CI and a fresh clone stay green:

    DATABASE_URL="postgresql://..." pytest backend/tests -m integration
"""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI

from backend.app import storage
from backend.app.audit import AuditEvent
from backend.app.config import Settings
from backend.app.data.repository import SyntheticRepository
from backend.app.data.synthetic import build_patients
from backend.app.db import engine, queries, rows
from backend.app.users import Role, User
from backend.tests.conftest import ADMIN, client_for

DATABASE_URL = os.environ.get("DATABASE_URL", "")
needs_postgres = pytest.mark.skipif(
    not DATABASE_URL, reason="set DATABASE_URL to run the Postgres integration tests"
)

SUPABASE = "postgresql://postgres:secret@db.abcdefgh.supabase.co:5432/postgres"


# ------------------------------- the connection URL ---------------------------------------


def test_a_supabase_url_gets_the_driver_and_tls():
    url = engine.normalise_url(SUPABASE)
    assert url.startswith("postgresql+psycopg://")
    assert "sslmode=require" in url
    assert "db.abcdefgh.supabase.co:5432" in url


def test_the_short_postgres_scheme_is_accepted():
    assert engine.normalise_url("postgres://u:p@example.com/db").startswith("postgresql+psycopg://")


def test_an_explicit_driver_is_left_alone():
    url = engine.normalise_url("postgresql+psycopg://u:p@example.com/db")
    assert url.startswith("postgresql+psycopg://")


def test_an_sslmode_the_user_chose_is_not_overridden():
    url = engine.normalise_url(f"{SUPABASE}?sslmode=verify-full")
    assert "sslmode=verify-full" in url
    assert "require" not in url


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_a_local_database_is_not_forced_onto_tls(host: str):
    assert "sslmode" not in engine.normalise_url(f"postgresql://u:p@{host}:5432/healthlens")


@pytest.mark.parametrize(
    "bad", ["", "   ", "mysql://u:p@host/db", "sqlite:///file.db", "postgresql://"]
)
def test_a_url_we_cannot_use_is_refused_clearly(bad: str):
    with pytest.raises(engine.DatabaseUrlError):
        engine.normalise_url(bad)


def test_the_pooled_port_is_recognised():
    assert engine.is_pooled(SUPABASE.replace(":5432", ":6543")) is True
    assert engine.is_pooled(SUPABASE) is False
    assert engine.is_pooled("not a url at all") is False


# ------------------------------- rows <-> dataclasses -------------------------------------


def test_a_patient_survives_the_round_trip():
    original = build_patients()[2]  # P-0103, three conditions, several medications
    row = rows.patient_params(original)
    rebuilt = rows.to_patient(
        row,
        conditions=original.conditions,
        assigned_doctor=original.assigned_doctor,
        medications=original.medications,
        observations=original.observations,
    )
    assert rebuilt == original


def test_medications_and_observations_accept_real_dates():
    from datetime import date

    medication = rows.to_medication(
        {"name": "Metformin", "dose": "500 mg", "frequency": "twice daily",
         "started": date(2024, 4, 15)}
    )  # fmt: skip
    assert medication.started == "2024-04-15"

    observation = rows.to_observation(
        {"observed_on": date(2024, 7, 15), "bmi": 31.0, "systolic_bp": 138, "hba1c": 7.1}
    )
    assert observation.date == "2024-07-15"
    assert observation.systolic_bp == 138


def test_a_user_survives_the_round_trip():
    original = User("u-1", "a@demo.healthlens.test", "A", Role.DOCTOR, "hash", True, None, "dr-a")
    assert rows.to_user(rows.user_params(original)) == original


def test_an_audit_event_survives_the_round_trip():
    original = AuditEvent(
        id=7,
        timestamp="2026-11-15T18:00:00+00:00",
        actor_id="dr-chen",
        actor_role="doctor",
        action="patient.read",
        resource_type="patient",
        resource_id="P-0103",
        patient_id="P-0103",
        outcome="allowed",
        request_id="abc123",
    )
    stored = {**rows.audit_params(original), "id": 7}
    assert rows.to_audit_event(stored) == original


def test_a_datetime_timestamp_becomes_an_iso_string():
    from datetime import UTC, datetime

    row = {
        "id": 1,
        "occurred_at": datetime(2026, 11, 15, 18, 0, tzinfo=UTC),
        "actor_id": None,
        "actor_role": None,
        "action": "access",
        "resource_type": "endpoint",
        "resource_id": "/x",
        "patient_id": None,
        "outcome": "denied",
        "request_id": "r1",
    }
    assert rows.to_audit_event(row).timestamp == "2026-11-15T18:00:00+00:00"


@pytest.mark.parametrize(
    ("query", "expected"),
    [(None, True), ("", True), ("boat", True), ("p-0103", True), ("nobody", False)],
)
def test_the_doctor_search_matches_name_or_id(query: str | None, expected: bool):
    samuel = build_patients()[2]  # Samuel Boateng, P-0103
    assert rows.matches_doctor(samuel, query) is expected


# ------------------------------- choosing a backend ---------------------------------------


def test_no_database_url_means_memory(settings: Settings):
    backend = storage.open_storage(settings)
    assert backend.name == "memory"
    assert backend.repo.source == "synthetic"
    assert backend.users is None  # the demo accounts get seeded


def test_an_explicit_repository_always_wins(settings: Settings):
    """``create_app(settings, repo=...)`` must not be overridden by an environment variable."""
    with_url = settings.model_copy(update={"database_url": SUPABASE})
    backend = storage.open_storage(with_url, repo=SyntheticRepository())
    assert backend.name == "memory"


def test_admin_system_reports_the_live_storage(app: FastAPI):
    body = client_for(app, ADMIN).get("/api/v1/admin/system").json()
    assert body["storage"] == "memory"
    assert body["data_source"] == "synthetic"
    assert body["models_loaded"] == []


# ------------------------------- Postgres (optional) --------------------------------------


@pytest.fixture
def postgres_engine():
    created = engine.create_engine(DATABASE_URL)
    # Apply the schema here rather than in one test, so these tests pass in any order.
    from backend.scripts import seed_db

    seed_db.apply_schema(created)
    yield created
    created.dispose()


@pytest.mark.integration
@needs_postgres
def test_the_schema_applies_and_the_seed_loads(postgres_engine):
    from backend.scripts import seed_db

    seed_db.apply_schema(postgres_engine)
    written = seed_db.seed(postgres_engine, "integration-test-password")
    assert written["patients"] == 8


@pytest.mark.integration
@needs_postgres
def test_the_postgres_repository_matches_the_synthetic_one(postgres_engine):
    from backend.app.db.stores import PostgresRepository

    repo = PostgresRepository(postgres_engine)
    assert len(repo.list_all()) == 8
    samuel = repo.get("P-0103")
    assert samuel is not None
    assert set(samuel.conditions) == {"diabetes", "hypertension", "copd"}
    assert samuel.assigned_doctor == "dr-chen"
    assert len(repo.list_for_doctor("dr-chen")) == 6
    assert repo.get("P-9999") is None


@pytest.mark.integration
@needs_postgres
def test_the_audit_log_can_only_grow(postgres_engine):
    from sqlalchemy import text

    from backend.app.db.stores import PostgresAuditStore

    store = PostgresAuditStore(postgres_engine)
    stored = store.append(
        AuditEvent(
            id=0,
            timestamp="2026-11-15T18:00:00+00:00",
            actor_id="dr-chen",
            actor_role="doctor",
            action="patient.read",
            resource_type="patient",
            resource_id="P-0103",
            patient_id="P-0103",
            outcome="allowed",
            request_id="integration",
        )
    )
    assert stored.id > 0
    assert any(e.request_id == "integration" for e in store.read(actor="dr-chen"))

    with pytest.raises(Exception, match="append-only"), postgres_engine.begin() as connection:
        connection.execute(text("DELETE FROM audit_events WHERE request_id = 'integration'"))


@pytest.mark.integration
@needs_postgres
def test_the_api_runs_on_postgres(settings: Settings, postgres_engine):
    """The app boots against Postgres and every store it holds is the Postgres one."""
    from fastapi.testclient import TestClient

    from backend.app.main import create_app
    from backend.scripts import seed_db

    seed_db.seed(postgres_engine, "integration-test-password")

    application = create_app(settings.model_copy(update={"database_url": DATABASE_URL}))
    with TestClient(application) as client:
        state = application.state.hl
        assert state.storage == "postgres"
        assert state.repo.source == "postgres"
        assert len(state.repo.list_all()) == 8
        assert state.users.by_email(ADMIN) is not None
        assert client.get("/health").status_code == 200


# ------------------------------- the append-only promise, in the source -------------------


def _statements() -> list[tuple[str, str]]:
    return [
        (name, value)
        for name, value in vars(queries).items()
        if name.isupper() and isinstance(value, str)
    ]


def test_no_statement_touches_the_audit_log_except_insert_and_select():
    """A guard, not a formality: ADR-0014 checked against the SQL we actually ship."""
    audit = [v for _, v in _statements() if "audit_events" in v]
    assert audit, "no audit statements found - did queries.py move?"
    for statement in audit:
        upper = " ".join(statement.upper().split())
        assert "UPDATE AUDIT_EVENTS" not in upper
        assert "DELETE FROM AUDIT_EVENTS" not in upper
        assert "TRUNCATE" not in upper
        assert upper.startswith(("INSERT INTO AUDIT_EVENTS", "SELECT"))


def test_every_value_in_every_statement_is_a_bound_parameter():
    """No value is ever formatted into SQL: the only holes are SQLAlchemy ``:name`` binds."""
    statements = _statements()
    assert len(statements) > 10
    for name, statement in statements:
        assert "%s" not in statement, name
        assert "{" not in statement and "}" not in statement, name

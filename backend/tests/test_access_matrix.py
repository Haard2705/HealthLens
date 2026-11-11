"""Role x endpoint x expected-status matrix: the heart of the access rules."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.tests.conftest import ADMIN, CHEN, FATIMA, JORDAN, OKAFOR, TOMAS, audit_events
from backend.tests.conftest import DEMO_PASSWORD as DEMO_PW

NOT_FOUND = {"error": {"code": "not_found", "message": "Patient not found."}}

# (method, path, {identity: expected status}) ; identity None = unauthenticated.
MATRIX = [
    # --- public ---
    ("GET", "/health", {None: 200, ADMIN: 200, CHEN: 200, FATIMA: 200}),
    ("GET", "/api/v1/conditions", {None: 200, ADMIN: 200, CHEN: 200, FATIMA: 200}),
    ("GET", "/api/v1/stats/prevalence?condition=diabetes&by=sex", {None: 200, FATIMA: 200}),
    ("GET", "/api/v1/models?condition=diabetes", {None: 404, FATIMA: 404}),
    # --- any authenticated user ---
    ("GET", "/api/v1/auth/me", {None: 401, ADMIN: 200, CHEN: 200, FATIMA: 200}),
    # --- doctor area ---
    ("GET", "/api/v1/doctor/patients", {None: 401, ADMIN: 403, CHEN: 200, FATIMA: 403}),
    ("GET", "/api/v1/doctor/patients/P-0106", {None: 401, ADMIN: 403, CHEN: 200, FATIMA: 403}),
    ("GET", "/api/v1/doctor/patients/P-0106/timeline", {None: 401, CHEN: 200, FATIMA: 403}),
    # doctor B / unassigned and genuinely missing look identical
    ("GET", "/api/v1/doctor/patients/P-0106", {OKAFOR: 404}),
    ("GET", "/api/v1/doctor/patients/P-0106/timeline", {OKAFOR: 404}),
    ("GET", "/api/v1/doctor/patients/P-9999", {CHEN: 404}),
    # --- patient self-service ---
    ("GET", "/api/v1/me/record", {None: 401, FATIMA: 200, CHEN: 403, ADMIN: 403}),
    ("GET", "/api/v1/me/medications", {None: 401, FATIMA: 200, CHEN: 403, ADMIN: 403}),
    ("GET", "/api/v1/me/observations", {None: 401, FATIMA: 200, CHEN: 403, ADMIN: 403}),
    # --- admin area ---
    ("GET", "/api/v1/admin/users", {None: 401, ADMIN: 200, CHEN: 403, FATIMA: 403}),
    ("GET", "/api/v1/admin/audit", {None: 401, ADMIN: 200, CHEN: 403, FATIMA: 403}),
    ("GET", "/api/v1/admin/system", {None: 401, ADMIN: 200, CHEN: 403, FATIMA: 403}),
    ("GET", "/api/v1/admin/patients/P-0106", {None: 401, ADMIN: 200, CHEN: 403, FATIMA: 403}),
    ("GET", "/api/v1/admin/patients/P-9999", {ADMIN: 404}),
]

CASES = [(m, p, who, status) for m, p, rules in MATRIX for who, status in rules.items()]


@pytest.mark.parametrize(("method", "path", "who", "status"), CASES)
def test_role_endpoint_matrix(anon, login_as, method, path, who, status):
    client = anon if who is None else login_as(who)
    assert client.request(method, path).status_code == status


def test_hidden_record_is_indistinguishable_from_missing(login_as):
    chen = login_as(CHEN)
    hidden = login_as(OKAFOR).get("/api/v1/doctor/patients/P-0106")
    missing = chen.get("/api/v1/doctor/patients/P-9999")
    assert hidden.status_code == missing.status_code == 404
    assert hidden.json() == missing.json() == NOT_FOUND


def test_doctor_lists_only_assigned_patients(login_as):
    chen_ids = {p["id"] for p in login_as(CHEN).get("/api/v1/doctor/patients").json()["items"]}
    okafor_ids = {p["id"] for p in login_as(OKAFOR).get("/api/v1/doctor/patients").json()["items"]}
    assert chen_ids == {f"P-010{i}" for i in range(1, 7)}
    assert okafor_ids == {"P-0107", "P-0108"}


def test_doctor_search_is_scoped_and_case_insensitive(login_as):
    chen = login_as(CHEN)
    found = chen.get("/api/v1/doctor/patients", params={"q": "fatima"}).json()["items"]
    assert [p["id"] for p in found] == ["P-0106"]
    # Tomas belongs to Dr Okafor, so Dr Chen's search must not reveal him.
    assert chen.get("/api/v1/doctor/patients", params={"q": "tomas"}).json()["items"] == []


def test_patient_cannot_read_another_patient_via_any_route(login_as):
    jordan = login_as(JORDAN)
    own = jordan.get("/api/v1/me/record").json()
    assert own["id"] == "P-0101"
    assert jordan.get("/api/v1/doctor/patients/P-0106").status_code == 403
    assert jordan.get("/api/v1/admin/patients/P-0106").status_code == 403


def test_patient_a_cannot_see_patient_b_record_content(login_as):
    assert login_as(FATIMA).get("/api/v1/me/record").json()["id"] == "P-0106"
    assert login_as(TOMAS).get("/api/v1/me/record").json()["id"] == "P-0107"


def test_deactivated_user_loses_access_immediately(app, login_as):
    admin = login_as(ADMIN)
    fatima = login_as(FATIMA)
    assert fatima.get("/api/v1/me/record").status_code == 200
    user_id = next(
        u["id"] for u in admin.get("/api/v1/admin/users").json()["items"] if u["email"] == FATIMA
    )
    assert admin.patch(f"/api/v1/admin/users/{user_id}", json={"active": False}).status_code == 200
    assert fatima.get("/api/v1/me/record").status_code == 401
    fresh = TestClient(app).post("/api/v1/auth/login", json={"email": FATIMA, "password": DEMO_PW})
    assert fresh.status_code == 401


# ------------------------------ audit behaviour ------------------------------------------


def test_admin_patient_read_is_audited(app, login_as):
    login_as(ADMIN).get("/api/v1/admin/patients/P-0106")
    events = audit_events(app, patient="P-0106", action="patient.read")
    assert len(events) == 1
    event = events[0]
    assert event["actor_role"] == "admin"
    assert event["outcome"] == "allowed"
    assert event["resource_type"] == "patient"
    assert event["request_id"]


def test_doctor_reads_are_audited(app, login_as):
    login_as(CHEN).get("/api/v1/doctor/patients/P-0106")
    events = audit_events(app, patient="P-0106", outcome="allowed")
    assert [e["actor_role"] for e in events] == ["doctor"]


def test_denied_attempts_are_audited(app, login_as):
    login_as(OKAFOR).get("/api/v1/doctor/patients/P-0106")
    login_as(FATIMA).get("/api/v1/admin/users")
    denied = audit_events(app, outcome="denied")
    assert {(e["actor_role"], e["patient_id"]) for e in denied} >= {
        ("doctor", "P-0106"),
        ("patient", None),
    }


def test_unauthenticated_access_is_audited_as_denied(app, anon):
    anon.get("/api/v1/admin/users")
    event = audit_events(app, outcome="denied")[-1]
    assert event["actor_id"] is None
    assert event["actor_role"] is None


def test_audit_events_contain_no_clinical_fields(app, login_as):
    login_as(CHEN).get("/api/v1/doctor/patients/P-0103")
    login_as(CHEN).get("/api/v1/doctor/patients/P-0103/timeline")
    login_as(FATIMA).get("/api/v1/me/medications")
    allowed = {
        "id", "timestamp", "actor_id", "actor_role", "action", "resource_type",
        "resource_id", "patient_id", "outcome", "request_id",
    }  # fmt: skip
    clinical = ("diabetes", "hypertension", "copd", "metformin", "bmi", "hba1c", "35.2")
    events = audit_events(app)
    assert events
    for event in events:
        assert set(event) == allowed
        blob = " ".join(str(v).lower() for v in event.values())
        assert not any(word in blob for word in clinical), event


def test_logins_are_audited(app, anon):
    anon.post("/api/v1/auth/login", json={"email": CHEN, "password": DEMO_PW})
    anon.post("/api/v1/auth/login", json={"email": CHEN, "password": "wrong"})
    actions = [(e["action"], e["outcome"]) for e in audit_events(app, action="auth.login")]
    assert actions == [("auth.login", "allowed"), ("auth.login", "denied")]

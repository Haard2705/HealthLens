"""Admin endpoints: users, invite, last-admin guard, audit filters, system."""

from __future__ import annotations

from backend.tests.conftest import ADMIN, CHEN, FATIMA

U = "/api/v1/admin/users"


def _id_of(admin, email):
    return next(u["id"] for u in admin.get(U).json()["items"] if u["email"] == email)


def test_user_list_has_no_password_material(login_as):
    body = login_as(ADMIN).get(U).text
    assert "argon2" not in body
    assert "password" not in body


def test_invite_creates_inactive_user(login_as):
    admin = login_as(ADMIN)
    response = admin.post(
        f"{U}/invite",
        json={"email": "New.Doc@demo.healthlens.test", "role": "doctor", "display_name": "New"},
    )
    assert response.status_code == 201
    assert response.json()["email"] == "new.doc@demo.healthlens.test"
    assert response.json()["active"] is False


def test_invite_patient_must_link_existing_patient(login_as):
    admin = login_as(ADMIN)
    base = {"email": "pat@demo.healthlens.test", "role": "patient", "display_name": "Pat"}
    assert admin.post(f"{U}/invite", json=base).status_code == 422
    assert admin.post(f"{U}/invite", json={**base, "patient_id": "P-9999"}).status_code == 422
    assert admin.post(f"{U}/invite", json={**base, "patient_id": "P-0101"}).status_code == 201


def test_invite_rejects_bad_email_role_and_extra_fields(login_as):
    admin = login_as(ADMIN)
    ok = {"email": "a@demo.healthlens.test", "role": "doctor", "display_name": "A"}
    assert admin.post(f"{U}/invite", json={**ok, "email": "not-an-email"}).status_code == 422
    assert admin.post(f"{U}/invite", json={**ok, "role": "superuser"}).status_code == 422
    assert admin.post(f"{U}/invite", json={**ok, "is_admin": True}).status_code == 422
    assert admin.post(f"{U}/invite", json={**ok, "patient_id": "P-0101"}).status_code == 422


def test_invite_duplicate_email_conflicts(login_as):
    admin = login_as(ADMIN)
    body = {"email": CHEN, "role": "doctor", "display_name": "Dup"}
    response = admin.post(f"{U}/invite", json=body)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "email_taken"


def test_non_admin_cannot_invite_or_patch(login_as):
    assert login_as(CHEN).post(f"{U}/invite", json={}).status_code == 403
    assert login_as(FATIMA).patch(f"{U}/u-admin", json={"active": False}).status_code == 403


def test_cannot_disable_last_active_admin(login_as):
    admin = login_as(ADMIN)
    response = admin.patch(f"{U}/u-admin", json={"active": False})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "last_admin"
    assert admin.get("/api/v1/auth/me").status_code == 200


def test_can_disable_admin_when_another_is_active(app, login_as):
    from backend.app.users import Role, User

    app.state.hl.users.add(User("u-admin2", "a2@demo.healthlens.test", "A2", Role.ADMIN, None))
    admin = login_as(ADMIN)
    assert admin.patch(f"{U}/u-admin2", json={"active": False}).status_code == 200
    assert admin.patch(f"{U}/u-admin", json={"active": False}).status_code == 409


def test_patch_unknown_user_and_bad_body(login_as):
    admin = login_as(ADMIN)
    assert admin.patch(f"{U}/u-nope", json={"active": False}).status_code == 404
    assert admin.patch(f"{U}/u-admin", json={"active": "maybe"}).status_code == 422


def test_reactivating_restores_access(login_as):
    admin = login_as(ADMIN)
    uid = _id_of(admin, CHEN)
    admin.patch(f"{U}/{uid}", json={"active": False})
    admin.patch(f"{U}/{uid}", json={"active": True})
    assert login_as(CHEN).get("/api/v1/auth/me").status_code == 200


def test_audit_filters(login_as):
    admin = login_as(ADMIN)
    login_as(CHEN).get("/api/v1/doctor/patients/P-0103")
    login_as("dr.okafor@demo.healthlens.test").get("/api/v1/doctor/patients/P-0103")
    by_patient = admin.get("/api/v1/admin/audit", params={"patient": "P-0103"}).json()["items"]
    assert {e["outcome"] for e in by_patient} == {"allowed", "denied"}
    denied = admin.get(
        "/api/v1/admin/audit", params={"outcome": "denied", "patient": "P-0103"}
    ).json()["items"]
    assert len(denied) == 1 and denied[0]["actor_id"] == "dr-okafor"
    by_actor = admin.get("/api/v1/admin/audit", params={"actor": "dr-chen"}).json()["items"]
    assert by_actor and all(e["actor_id"] == "dr-chen" for e in by_actor)
    by_action = admin.get("/api/v1/admin/audit", params={"action": "auth.login"}).json()["items"]
    assert by_action and all(e["action"] == "auth.login" for e in by_action)


def test_audit_limit_and_newest_first(login_as):
    admin = login_as(ADMIN)
    items = admin.get("/api/v1/admin/audit", params={"limit": 2}).json()["items"]
    assert len(items) == 2
    assert items[0]["id"] > items[1]["id"]
    assert admin.get("/api/v1/admin/audit", params={"limit": 0}).status_code == 422


def test_audit_store_is_append_only(app):
    store = app.state.hl.audit
    public = {n for n in dir(store) if not n.startswith("_")}
    assert public == {"append", "read"}


def test_system_status(login_as):
    body = login_as(ADMIN).get("/api/v1/admin/system").json()
    assert body["data_source"] == "synthetic"
    assert body["patient_count"] == 8
    assert body["conditions_total"] == 9
    assert body["conditions_prediction_enabled"] == 0


def test_system_counts_enabled_conditions(app, login_as, settings):
    import json

    settings.models_path.mkdir(parents=True)
    index = {"conditions": {"diabetes": {"prediction_enabled": True}}}
    (settings.models_path / "index.json").write_text(json.dumps(index))
    assert login_as(ADMIN).get("/api/v1/admin/system").json()["conditions_prediction_enabled"] == 1

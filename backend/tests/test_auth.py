"""Login failure modes, session cookie, tokens, config safety and the error envelope."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app import security
from backend.app.config import ConfigError, Settings
from backend.app.main import create_app
from backend.tests.conftest import ADMIN, CHEN, DEMO_PASSWORD, audit_events

LOGIN = "/api/v1/auth/login"


def _login(client, email, password):
    return client.post(LOGIN, json={"email": email, "password": password})


def test_login_sets_httponly_lax_cookie(anon):
    response = _login(anon, ADMIN, DEMO_PASSWORD)
    assert response.status_code == 200
    assert response.json()["role"] == "admin"
    cookie = response.headers["set-cookie"].lower()
    assert "hl_session=" in cookie
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert "secure" not in cookie


def test_cookie_is_secure_in_production():
    settings = Settings(
        env="production",
        jwt_secret="x" * 40,
        demo_password="prod-demo-password",
        _env_file=None,
    )
    client = TestClient(create_app(settings), base_url="https://testserver")
    response = _login(client, ADMIN, "prod-demo-password")
    assert "secure" in response.headers["set-cookie"].lower()


def test_unknown_user_and_wrong_password_get_the_same_error(anon):
    unknown = _login(anon, "nobody@demo.healthlens.test", "whatever")
    wrong = _login(anon, ADMIN, "wrong-password")
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json()
    assert unknown.json()["error"]["code"] == "invalid_credentials"


def test_email_is_case_insensitive(anon):
    assert _login(anon, ADMIN.upper(), DEMO_PASSWORD).status_code == 200


def test_rate_limit_after_five_failures(anon):
    for _ in range(5):
        assert _login(anon, CHEN, "bad").status_code == 401
    blocked = _login(anon, CHEN, DEMO_PASSWORD)  # even the right password is refused now
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "rate_limited"
    # a different email from the same IP is unaffected
    assert _login(anon, ADMIN, DEMO_PASSWORD).status_code == 200


def test_rate_limit_window_expires():
    now = [0.0]
    limiter = security.LoginRateLimiter(clock=lambda: now[0])
    for _ in range(security.MAX_FAILURES):
        limiter.record_failure("a@x.test", "1.1.1.1")
    assert limiter.is_blocked("a@x.test", "1.1.1.1")
    now[0] += security.WINDOW_SECONDS + 1
    assert not limiter.is_blocked("a@x.test", "1.1.1.1")
    limiter.record_failure("a@x.test", "1.1.1.1")
    limiter.reset("a@x.test", "1.1.1.1")
    assert not limiter.is_blocked("a@x.test", "1.1.1.1")


def test_success_resets_the_failure_count(anon):
    for _ in range(4):
        _login(anon, CHEN, "bad")
    assert _login(anon, CHEN, DEMO_PASSWORD).status_code == 200
    for _ in range(4):
        assert _login(anon, CHEN, "bad").status_code == 401


def test_failed_login_audit_does_not_store_the_typed_email(app, anon):
    _login(anon, "typo-or-password@x.test", "bad")
    event = audit_events(app, action="auth.login")[0]
    assert "typo" not in str(event.values())
    assert event["actor_id"] is None


def test_tampered_token_is_rejected(app, anon):
    anon.cookies.set("hl_session", "not.a.jwt")
    assert anon.get("/api/v1/auth/me").status_code == 401
    forged = security.issue_token("u-admin", "some-other-secret-some-other-secret", 60)
    anon.cookies.set("hl_session", forged)
    assert anon.get("/api/v1/auth/me").status_code == 401


def test_expired_token_is_rejected(app, anon):
    secret = app.state.hl.settings.jwt_secret
    expired = security.issue_token("u-admin", secret, 60, now=1_000_000)
    anon.cookies.set("hl_session", expired)
    assert anon.get("/api/v1/auth/me").status_code == 401


def test_token_for_unknown_user_is_rejected(app, anon):
    token = security.issue_token("u-ghost", app.state.hl.settings.jwt_secret, 60)
    anon.cookies.set("hl_session", token)
    assert anon.get("/api/v1/auth/me").status_code == 401


def test_logout_clears_session(login_as):
    client = login_as(CHEN)
    assert client.get("/api/v1/auth/me").json()["email"] == CHEN
    assert client.post("/api/v1/auth/logout").status_code == 204
    assert client.get("/api/v1/auth/me").status_code == 401


def test_logout_requires_login(anon):
    assert anon.post("/api/v1/auth/logout").status_code == 401


def test_password_helpers():
    hashed = security.hash_password("s3cret")
    assert hashed.startswith("$argon2")
    assert security.verify_password(hashed, "s3cret")
    assert not security.verify_password(hashed, "other")
    assert not security.verify_password("garbage", "s3cret")


# ---------------------------------- configuration ----------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # default secret
        {"jwt_secret": ""},
        {"jwt_secret": "short"},
        {"jwt_secret": "x" * 40},  # fine secret but no DEMO_PASSWORD
    ],
)
def test_production_refuses_unsafe_config(overrides):
    with pytest.raises(ConfigError):
        Settings(env="production", _env_file=None, **overrides)


def test_production_accepts_safe_config():
    s = Settings(env="production", jwt_secret="y" * 40, demo_password="pw", _env_file=None)
    assert s.is_production


def test_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test")
    settings = Settings(_env_file=None)
    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


def test_default_demo_password_only_outside_production():
    assert Settings(env="development", _env_file=None).effective_demo_password


# ----------------------------------- error envelope --------------------------------------


def test_unknown_route_uses_envelope(anon):
    response = anon.get("/api/v1/nope")
    assert response.status_code == 404
    assert response.json() == {"error": {"code": "not_found", "message": "Not Found"}}


def test_method_not_allowed_uses_envelope(anon):
    response = anon.delete("/health")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_validation_error_uses_envelope_and_does_not_echo_input(anon):
    response = anon.post(LOGIN, json={"email": "a@b.test", "password": ""})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert set(body) == {"error"}


def test_request_id_header_present(anon):
    assert anon.get("/health").headers["x-request-id"]


def test_unexpected_exception_uses_envelope(app):
    @app.get("/boom")
    def boom():
        raise RuntimeError("secret internals")

    response = TestClient(app, raise_server_exceptions=False).get("/boom")
    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "Something went wrong."}
    }

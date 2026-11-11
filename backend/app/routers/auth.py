"""Login, logout and "who am I"."""

from __future__ import annotations

import hashlib
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from .. import audit
from ..access import Access, anonymous_access, requires
from ..errors import AppError
from ..schemas import LoginRequest, UserOut
from ..security import COOKIE_NAME, dummy_hash, issue_token, verify_password
from ..users import normalize_email

router = APIRouter(prefix="/auth", tags=["auth"])

INVALID = AppError(401, "invalid_credentials", "Incorrect email or password.")
TOO_MANY = AppError(429, "rate_limited", "Too many failed attempts. Try again in a few minutes.")


def _email_ref(email: str) -> str:
    """A stable reference for audit rows that does not store what was typed."""
    return hashlib.sha256(email.encode()).hexdigest()[:12]


@router.post("/login", response_model=UserOut)
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    access: Annotated[Access, Depends(anonymous_access)],
) -> UserOut:
    state = access.state
    email = normalize_email(body.email)
    ip = request.client.host if request.client else "unknown"
    ref = _email_ref(email)

    if state.limiter.is_blocked(email, ip):
        access.record("auth.login", audit.DENIED, "session", ref)
        raise TOO_MANY

    user = state.users.by_email(email)
    stored = user.password_hash if user and user.password_hash else dummy_hash()
    password_ok = verify_password(stored, body.password)  # always runs: similar timing
    if not (user and user.password_hash and password_ok and user.active):
        state.limiter.record_failure(email, ip)
        access.as_user(user).record("auth.login", audit.DENIED, "session", ref)
        raise INVALID

    state.limiter.reset(email, ip)
    settings = state.settings
    response.set_cookie(
        COOKIE_NAME,
        issue_token(user.id, settings.jwt_secret, settings.session_minutes),
        max_age=settings.session_minutes * 60,
        httponly=True,
        samesite="lax",
        secure=settings.is_production,
        path="/",
    )
    access.as_user(user).record("auth.login", audit.ALLOWED, "session", ref)
    return UserOut.of(user)


@router.post("/logout", status_code=204)
def logout(
    response: Response,
    _access: Annotated[Access, Depends(requires(action="auth.logout", resource_type="session"))],
) -> None:
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="lax")


@router.get("/me", response_model=UserOut)
def me(access: Annotated[Access, Depends(requires())]) -> UserOut:
    assert access.user is not None  # requires() guarantees an authenticated user
    return UserOut.of(access.user)

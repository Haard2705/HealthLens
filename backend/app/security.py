"""Password hashing (argon2), signed session tokens (PyJWT) and the login rate limiter.

No homemade crypto: this module only wires maintained libraries together.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from functools import lru_cache

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

COOKIE_NAME = "hl_session"
JWT_ALGORITHM = "HS256"
MAX_FAILURES = 5
WINDOW_SECONDS = 300

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


@lru_cache(maxsize=8)
def hash_password_cached(password: str) -> str:
    """Hash once per process; used to seed many demo accounts quickly."""
    return hash_password(password)


def verify_password(stored_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(stored_hash, password)
    except (VerificationError, InvalidHashError):
        return False


@lru_cache(maxsize=1)
def dummy_hash() -> str:
    """Hash checked for unknown users so unknown-user and wrong-password take similar time."""
    return hash_password("not-a-real-password-dummy")


def issue_token(user_id: str, secret: str, minutes: int, now: float | None = None) -> str:
    issued = int(now if now is not None else time.time())
    claims = {"sub": user_id, "iat": issued, "exp": issued + minutes * 60}
    return jwt.encode(claims, secret, algorithm=JWT_ALGORITHM)


def read_token(token: str, secret: str) -> str | None:
    """Return the user id from a valid, unexpired token, else None (tampered/expired/garbage)."""
    try:
        claims = jwt.decode(
            token, secret, algorithms=[JWT_ALGORITHM], options={"require": ["sub", "exp"]}
        )
    except jwt.PyJWTError:
        return None
    sub = claims.get("sub")
    return sub if isinstance(sub, str) else None


class LoginRateLimiter:
    """In-memory: MAX_FAILURES failures per WINDOW_SECONDS per (email, ip) -> blocked."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._failures: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def _recent(self, key: tuple[str, str]) -> deque[float]:
        failures = self._failures[key]
        cutoff = self._clock() - WINDOW_SECONDS
        while failures and failures[0] <= cutoff:
            failures.popleft()
        return failures

    def is_blocked(self, email: str, ip: str) -> bool:
        return len(self._recent((email, ip))) >= MAX_FAILURES

    def record_failure(self, email: str, ip: str) -> None:
        self._recent((email, ip)).append(self._clock())

    def reset(self, email: str, ip: str) -> None:
        self._failures.pop((email, ip), None)

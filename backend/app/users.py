"""User accounts: model, in-memory store and demo seeding. (H17 moves this to Postgres.)"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, replace
from enum import StrEnum

from .data.repository import PatientRepository
from .data.synthetic import DOCTORS
from .security import hash_password_cached

DEMO_DOMAIN = "demo.healthlens.test"
EMAIL_PATTERN = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$"
_EMAIL_RE = re.compile(EMAIL_PATTERN)


class Role(StrEnum):
    ADMIN = "admin"
    DOCTOR = "doctor"
    PATIENT = "patient"


def normalize_email(email: str) -> str:
    return email.strip().lower()


def is_valid_email(email: str) -> bool:
    return len(email) <= 254 and _EMAIL_RE.match(email) is not None


@dataclass(frozen=True)
class User:
    id: str
    email: str
    display_name: str
    role: Role
    password_hash: str | None
    active: bool = True
    patient_id: str | None = None  # set for PATIENT accounts
    doctor_id: str | None = None  # set for DOCTOR accounts (matches Patient.assigned_doctor)


class UserStore:
    def __init__(self, users: tuple[User, ...] = ()) -> None:
        self._users = {u.id: u for u in users}
        self._lock = threading.Lock()

    def get(self, user_id: str) -> User | None:
        return self._users.get(user_id)

    def by_email(self, email: str) -> User | None:
        wanted = normalize_email(email)
        return next((u for u in self._users.values() if u.email == wanted), None)

    def all(self) -> tuple[User, ...]:
        return tuple(self._users.values())

    def add(self, user: User) -> User:
        with self._lock:
            self._users[user.id] = user
        return user

    def set_active(self, user_id: str, active: bool) -> User | None:
        """Replace the stored user with an updated copy (records are never mutated)."""
        with self._lock:
            current = self._users.get(user_id)
            if current is None:
                return None
            updated = replace(current, active=active)
            self._users[user_id] = updated
            return updated

    def active_admin_count(self) -> int:
        return sum(1 for u in self._users.values() if u.role is Role.ADMIN and u.active)


def _patient_email(name: str) -> str:
    return f"{name.lower().replace(' ', '.')}@{DEMO_DOMAIN}"


def seed_demo_users(repo: PatientRepository, password: str) -> UserStore:
    pw = hash_password_cached(password)
    users = [
        User("u-admin", f"admin@{DEMO_DOMAIN}", "Demo Admin", Role.ADMIN, pw),
        *(
            User(
                d.id,
                f"dr.{d.id.removeprefix('dr-')}@{DEMO_DOMAIN}",
                d.name,
                Role.DOCTOR,
                pw,
                doctor_id=d.id,
            )
            for d in DOCTORS
        ),
        *(
            User(f"u-{p.id}", _patient_email(p.name), p.name, Role.PATIENT, pw, patient_id=p.id)
            for p in repo.list_all()
        ),
    ]
    return UserStore(tuple(users))

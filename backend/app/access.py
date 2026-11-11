"""The access-control layer: authentication, role checks, record scoping and auditing.

Every protected endpoint depends on ``requires(...)``, and every read of a patient record goes
through ``Access.patient``. Both write audit events, so an endpoint cannot forget to.

Rules (ADR-0011): a patient reads only their own record; a doctor reads only patients assigned
to them; an admin may read any record but each such read is audited. A record that is hidden
from the caller yields the same 404 as a record that does not exist.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from fastapi import Request

from . import audit
from .data.models import Patient
from .errors import AppError, patient_not_found
from .security import COOKIE_NAME, read_token
from .state import AppState
from .users import Role, User


def _unauthenticated() -> AppError:
    return AppError(401, "unauthenticated", "Authentication required.")


def _forbidden() -> AppError:
    return AppError(403, "forbidden", "You do not have access to this resource.")


@dataclass(frozen=True)
class Access:
    state: AppState
    request_id: str
    user: User | None = None

    def as_user(self, user: User | None) -> Access:
        return replace(self, user=user)

    def record(
        self,
        action: str,
        outcome: str,
        resource_type: str,
        resource_id: str | None = None,
        patient_id: str | None = None,
    ) -> None:
        """Append one audit event for this caller. The only audit entry point for endpoints."""
        audit.record(
            self.state.audit,
            request_id=self.request_id,
            actor_id=self.user.id if self.user else None,
            actor_role=self.user.role.value if self.user else None,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            patient_id=patient_id,
            outcome=outcome,
        )

    # ---- patient records -------------------------------------------------------------

    def patient(self, patient_id: str, action: str) -> Patient:
        """Return the patient if this caller may see them, else raise the uniform 404."""
        found = self.state.repo.get(patient_id)
        if found is None:
            outcome = audit.NOT_FOUND if self._is_admin() else audit.DENIED
            self.record(action, outcome, "patient", patient_id, patient_id)
            raise patient_not_found()
        if not self._may_read(found):
            self.record(action, audit.DENIED, "patient", patient_id, patient_id)
            raise patient_not_found()
        self.record(action, audit.ALLOWED, "patient", patient_id, patient_id)
        return found

    def own_patient(self, action: str) -> Patient:
        patient_id = self.user.patient_id if self.user else None
        if patient_id is None:  # a patient account with no linked record
            self.record(action, audit.DENIED, "patient")
            raise patient_not_found()
        return self.patient(patient_id, action)

    def _is_admin(self) -> bool:
        return self.user is not None and self.user.role is Role.ADMIN

    def _may_read(self, patient: Patient) -> bool:
        user = self.user
        if user is None:
            return False
        if user.role is Role.ADMIN:
            return True
        if user.role is Role.DOCTOR:
            return user.doctor_id is not None and patient.assigned_doctor == user.doctor_id
        return user.patient_id == patient.id


def _state(request: Request) -> AppState:
    return request.app.state.hl


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _authenticate(request: Request) -> User | None:
    """The user behind a valid session cookie, or None. Deactivated users never authenticate."""
    state = _state(request)
    token = request.cookies.get(COOKIE_NAME)
    user_id = read_token(token, state.settings.jwt_secret) if token else None
    user = state.users.get(user_id) if user_id else None
    return user if user is not None and user.active else None


def anonymous_access(request: Request) -> Access:
    """Dependency for endpoints that handle their own authentication (login)."""
    return Access(_state(request), _request_id(request))


def _endpoint_key(request: Request) -> tuple[str, str | None]:
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    first_param = next(iter(request.path_params.values()), None)
    return path, first_param


def requires(
    *roles: Role, action: str | None = None, resource_type: str = "endpoint"
) -> Callable[[Request], Access]:
    """Build a dependency that authenticates, checks the role and audits.

    With ``action`` set, a successful pass writes an ``allowed`` event (for non-patient-scoped
    endpoints such as the admin ones). Failures always write ``denied`` events.
    """

    def dependency(request: Request) -> Access:
        access = anonymous_access(request)
        path, first_param = _endpoint_key(request)
        user = _authenticate(request)
        if user is None:
            access.record("access", audit.DENIED, "endpoint", path)
            raise _unauthenticated()
        access = access.as_user(user)
        if roles and user.role not in roles:
            access.record("access", audit.DENIED, "endpoint", path)
            raise _forbidden()
        if action:
            access.record(action, audit.ALLOWED, resource_type, first_param)
        return access

    return dependency

"""Admin area: users, assignments, audit log and system status. Patient reads are audited."""

from __future__ import annotations

import re
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from .. import catalog
from ..access import Access, requires
from ..errors import AppError
from ..schemas import InviteRequest, PatientRecord, UserOut, UserPatch
from ..users import DEMO_DOMAIN, Role, User

router = APIRouter(prefix="/admin", tags=["admin"])


def _admin(action: str | None = None, resource_type: str = "endpoint"):
    return Annotated[
        Access, Depends(requires(Role.ADMIN, action=action, resource_type=resource_type))
    ]


PlainAdmin = _admin()
UserListAdmin = _admin("user.list", "user")
InviteAdmin = _admin("user.invite", "user")
UpdateAdmin = _admin("user.update", "user")
AuditAdmin = _admin("audit.read", "audit")
SystemAdmin = _admin("system.read", "system")


@router.get("/users")
def list_users(access: UserListAdmin) -> dict:
    return {"items": [UserOut.of(u) for u in access.state.users.all()]}


@router.post("/users/invite", status_code=201, response_model=UserOut)
def invite_user(body: InviteRequest, access: InviteAdmin) -> UserOut:
    state = access.state
    if state.users.by_email(body.email):
        raise AppError(409, "email_taken", "A user with that email already exists.")
    doctor_id = None
    if body.role is Role.PATIENT:
        if not body.patient_id or state.repo.get(body.patient_id) is None:
            raise AppError(
                422, "validation_error", "patient_id must reference an existing patient."
            )
    elif body.patient_id:
        raise AppError(422, "validation_error", "patient_id is only valid for the patient role.")
    if body.role is Role.DOCTOR:
        slug = re.sub(r"[^a-z0-9]+", "-", body.email.split("@")[0]).strip("-")
        doctor_id = f"dr-{slug}"
    # Invite-only: the account exists but cannot sign in until the invite is accepted (H32).
    user = User(
        id=f"u-{uuid.uuid4().hex[:8]}",
        email=body.email,
        display_name=body.display_name,
        role=body.role,
        password_hash=None,
        active=False,
        patient_id=body.patient_id,
        doctor_id=doctor_id,
    )
    return UserOut.of(state.users.add(user))


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(user_id: str, body: UserPatch, access: UpdateAdmin) -> UserOut:
    users = access.state.users
    target = users.get(user_id)
    if target is None:
        raise AppError(404, "not_found", "User not found.")
    last_admin = target.role is Role.ADMIN and target.active and users.active_admin_count() <= 1
    if not body.active and last_admin:
        raise AppError(409, "last_admin", "Cannot deactivate the last active admin.")
    updated = users.set_active(user_id, body.active)
    assert updated is not None
    return UserOut.of(updated)


@router.get("/audit")
def read_audit(
    access: AuditAdmin,
    actor: Annotated[str | None, Query(max_length=64)] = None,
    patient: Annotated[str | None, Query(max_length=64)] = None,
    action: Annotated[str | None, Query(max_length=64)] = None,
    outcome: Annotated[str | None, Query(max_length=32)] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> dict:
    events = access.state.audit.read(actor=actor, patient=patient, action=action, outcome=outcome)
    newest_first = list(reversed(events))[:limit]
    return {"items": [e.as_dict() for e in newest_first]}


@router.get("/patients/{patient_id}", response_model=PatientRecord)
def read_patient(patient_id: str, access: PlainAdmin) -> PatientRecord:
    return PatientRecord.of(access.patient(patient_id, "patient.read"))


@router.get("/system")
def system(access: SystemAdmin) -> dict:
    state = access.state
    models = state.settings.models_path
    return {
        "data_source": state.repo.source,
        "patient_count": len(state.repo.list_all()),
        "user_count": len(state.users.all()),
        "conditions_total": len(state.registry.ids),
        "conditions_prediction_enabled": catalog.prediction_enabled_count(state.registry, models),
        "environment": state.settings.env,
        "demo_domain": DEMO_DOMAIN,
    }

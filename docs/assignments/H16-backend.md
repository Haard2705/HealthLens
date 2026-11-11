# H16 - Backend: API, access control and synthetic data

## What

A FastAPI backend (`backend/app`) serving `/api/v1` plus `GET /health`:

- **Auth**: login/logout/me with argon2 password hashes and a signed JWT in an httpOnly,
  SameSite=Lax `hl_session` cookie (Secure when `ENV=production`).
- **Access control** (`access.py`): patient reads only own record; doctor reads only assigned
  patients; admin manages users and may read a patient (always audit-logged). Hidden records
  return the same 404 body as missing ones.
- **Audit log** (`audit.py`): append-only; one helper, called from the access layer. No clinical
  content in events.
- **Synthetic data** (`data/synthetic.py`): the 8 demo patients behind a `PatientRepository`
  Protocol (`data/repository.py`). Every record has `is_synthetic=true`.
- **Public data**: `/conditions` (registry + release-gate fields from `ml/models/index.json`),
  `/models?condition=`, `/stats/prevalence` (survey-weighted if
  `data/processed/brfss_clean.parquet` exists, else a labelled `"source":"sample"`).
- Errors always look like `{"error": {"code": "...", "message": "..."}}`.

## Why

ADR-0011 (backend-enforced RBAC), ADR-0012 (swappable patient source), ADR-0014 (audit). Doing
it in-memory first lets the dashboard demo run with no database.

## Run

```bash
pip install -r backend/requirements.txt -r requirements-dev.txt
uvicorn backend.app.main:app --reload          # http://localhost:8000/docs
```

Demo accounts (password = `DEMO_PASSWORD`, default `demo-password-123` outside production):
`admin@demo.healthlens.test`, `dr.chen@demo.healthlens.test`,
`dr.okafor@demo.healthlens.test`, and one per patient, e.g.
`fatima.hassan@demo.healthlens.test` (P-0106, Dr Chen).

```bash
curl -c jar -X POST localhost:8000/api/v1/auth/login -H 'content-type: application/json' \
  -d '{"email":"dr.chen@demo.healthlens.test","password":"demo-password-123"}'
curl -b jar localhost:8000/api/v1/doctor/patients
```

## Verify

```bash
ruff check backend && ruff format --check backend
pytest backend/tests --cov=backend/app      # no Postgres or network needed
```

The key suite is `tests/test_access_matrix.py` (role x endpoint x status). Manual check: log in
as Dr Okafor, `GET /doctor/patients/P-0106` -> 404; log in as admin, `GET
/admin/audit?outcome=denied` shows the attempt.

## Swapping synthetic for real data

Implement `PatientRepository` (`get`, `list_all`, `list_for_doctor`, `source`) over the real
store (FHIR/CSV importer or Postgres) and pass it: `create_app(settings, repo=MyRepo())`. Keep
`Patient`/`Medication`/`Observation` (`data/models.py`) as the contract. Demo users are seeded
from `repo.list_all()` in `users.py`; real deployments should load users from the DB instead.

## Gotchas

- State is in memory: users, audit log and rate limits reset on restart. **Next (H17): Postgres**
  for users, assignments and the append-only audit table (H34).
- Invite creates an **inactive** user with no password; the accept-invite flow is H32.
- One argon2 hash is computed per process and reused for all demo accounts (same password).
- `.test` emails fail `email-validator`, so emails are checked with a simple regex.
- Login rate limit is per (email, IP), in-memory, single-process only.
- Unknown/invalid `condition` or `by` returns 422 `invalid_parameter`.
- `/predict` and `/patients/{id}/risk-context` (P2) are NOT built; `ml/models/` has no trained
  artifacts yet. `backend/Dockerfile` and `backend/openapi.json` exist (Dockerfile untested).
- Production start fails unless `JWT_SECRET` (>= 32 chars) and `DEMO_PASSWORD` are set.

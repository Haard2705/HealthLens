# HealthLens

[![CI](https://github.com/Haard2705/HealthLens/actions/workflows/ci.yml/badge.svg)](https://github.com/Haard2705/HealthLens/actions/workflows/ci.yml)

**An Interactive Analytics and Risk Prediction Dashboard for Public Health Data.**
COMP 4310 — Web Health Informatics, Lakehead University.

HealthLens turns the CDC BRFSS 2025 health survey (356,158 respondents) into something a
non-specialist can explore: weighted prevalence charts for nine chronic conditions, a side-by-side
comparison of the models trained for each condition, and individual risk estimates that explain
themselves — but only for conditions where the model is good enough to be trusted.

> **Not a diagnostic tool.** This is a course project. It shows population-level patterns in
> self-reported survey data and does not give medical advice. Every patient record in the
> application is synthetic.

---

## What works today

| Area | Status |
|---|---|
| Data pipeline: download → clean → features | Working, one command each |
| Per-condition models for 9 conditions | Working |
| SHAP explanations, as plain-language sentences | Working |
| Release gate (weak or sensitive models cannot show an estimate) | Working |
| Backend API: login, 3 roles, access control, audit log | Working |
| Frontend: dashboard and role screens | Working |
| PostgreSQL / Supabase persistence | Optional (in-memory by default) |

The nine conditions are diabetes, hypertension, heart disease, stroke, asthma, COPD, arthritis,
chronic kidney disease and depression.

---

## Quick start

Fastest path to a running application. **No database needed** — the backend uses an in-memory
store seeded with synthetic patients unless you give it a database URL.

**Prerequisites:** Python 3.12+ and Node.js 20+ (developed on Python 3.13 and Node 22).

```bash
git clone https://github.com/Haard2705/HealthLens.git
cd HealthLens
```

**Terminal 1 — backend (http://localhost:8000):**

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r backend/requirements.txt -r ml/requirements.txt
python -m uvicorn backend.app.main:app --port 8000
```

**Terminal 2 — frontend (http://localhost:3000):**

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**.

### Demo logins

Sign-in is invite-only in the real design; these are throwaway accounts over synthetic data,
created only so the prototype can be inspected.

| Role | Email | What they can see |
|---|---|---|
| Patient | `fatima.hassan@demo.healthlens.test` | Only her own record |
| Doctor | `dr.chen@demo.healthlens.test` | Only her assigned patients |
| Admin | `admin@demo.healthlens.test` | Users, system status, audit log |

Password for all three: `demo-password-123` (override with the `DEMO_PASSWORD` environment variable).

**Worth trying:** sign in as the doctor and open patient `P-0107`, who belongs to a different
doctor. The API returns the same "not found" response as a patient that does not exist, so the
interface cannot be used to discover who is a patient — and the attempt is written to the audit
log, which you can see by signing in as the admin.

---

## Model artifacts

The application runs without trained models: prevalence charts and the clinical demo work, and
prediction endpoints return a clear "models unavailable" response.

To train them yourself (this is also how every number in the report was produced):

```bash
python scripts/download_brfss.py      # ~760 MB from the CDC, into data/raw/
python -m ml.src.clean                # → data/processed/brfss_clean.parquet
python -m ml.src.train                # → ml/models/<condition>/
python -m ml.src.gate                 # applies the release gate
```

`data/` and `ml/models/` are gitignored: survey data and model binaries are never committed,
and CI blocks them.

---

## Optional: PostgreSQL via Supabase

The backend uses an in-memory store by default. To persist to PostgreSQL, set `DATABASE_URL`
and seed it. Full instructions, including where to find the connection string in Supabase, are
in [`docs/assignments/H17-database.md`](docs/assignments/H17-database.md).

```bash
cp .env.example .env        # then put your connection string in DATABASE_URL
python -m backend.scripts.seed_db
```

`GET /api/v1/admin/system` reports which store is active. Never commit `.env`.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest ml/tests --cov=ml/src            # 338 tests
pytest backend/tests --cov=backend/app  # 136 tests
cd frontend && npm run lint && npm run build
```

The most important suite is `backend/tests/test_access_matrix.py`: a parametrised
role × endpoint × expected-status matrix proving that a doctor cannot reach another doctor's
patient, a patient cannot reach admin routes, and denied attempts are audit-logged.

---

## Repository layout

```
ml/        src/      data pipeline, models, SHAP explanations, release gate
           tests/    338 tests
backend/   app/      FastAPI: auth, RBAC, audit log, statistics, predictions
           tests/    136 tests
frontend/            Next.js + TypeScript application
data/                raw/ and processed/ gitignored; codebooks kept
docs/      adr/      architecture decision records
           assignments/  implementation notes, one per task
scripts/             dataset download
```

Project plan and task tracker: [`healthlens-todo.md`](healthlens-todo.md).

---

## How it is built

- **Pipeline** (Python, pandas, scikit-learn, XGBoost). Only the source adapter knows raw BRFSS
  column names, so switching to the Canadian CCHS later is a mapping change. Survey sentinel
  codes become missing values; a skipped question is never read as "no". Preprocessing is fitted
  inside a scikit-learn pipeline on training rows only, so nothing leaks from the test set.
- **Model choice.** Each condition is split 80/20, with a validation slice carved from the
  training part. Models are compared on validation data; the test set is used once, for reporting.
  Near-ties go to the simpler model.
- **Release gate.** A condition may show an individual estimate only if held-out ROC-AUC ≥ 0.70,
  its probabilities beat always guessing the average (Brier score), and it is not flagged
  sensitive. Enforced in the API, not just hidden in the interface.
- **Access control.** Checked server-side on every request. A record you may not see returns the
  same 404 as a record that does not exist. Passwords are hashed with argon2.

---

## Team

Haard Pathak (backend, access control, deployment) · Krish Patel (data pipeline, synthetic
patients) · Rudra Ukani (models, explanations, evaluation) · Vishwarajsinh Chauhan (frontend).

Every change goes through a pull request with review and green CI before reaching `main`.

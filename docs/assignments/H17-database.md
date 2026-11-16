# H17 - PostgreSQL / Supabase persistence (without breaking zero-setup)

## What

The backend runs on two interchangeable storage backends behind the same interfaces:

| | default | with `DATABASE_URL` set |
|---|---|---|
| patients | `SyntheticRepository` (in memory) | `PostgresRepository` |
| users | `UserStore` (in memory, demo accounts seeded) | `PostgresUserStore` |
| audit log | `InMemoryAuditStore` | `PostgresAuditStore` |
| `GET /admin/system` -> `"storage"` | `"memory"` | `"postgres"` |

**Memory is the default and stays the default.** A marker can clone the repository, install the
requirements and run `uvicorn` with no database, no Docker and no environment variables: the
eight synthetic patients and the demo accounts are there as before. Setting `DATABASE_URL`
swaps in the Postgres implementations of the *same* `PatientRepository` / `UserStore` /
`AuditStore` interfaces (ADR-0012); nothing else in the app changes, and the backend logs
which one it chose at startup.

Everything lives in `backend/app/db/`: `engine.py` (the URL and the engine), `queries.py`
(every SQL statement, all parameterised), `rows.py` (rows to and from the frozen dataclasses)
and `stores.py` (the three stores). The schema is one plain file, `backend/db/schema.sql` -
no Alembic. `audit_events` is append-only: the backend only ever issues `INSERT` and `SELECT`
against it, and a database trigger rejects `UPDATE`, `DELETE` and `TRUNCATE` so that it is a
guarantee rather than a convention.

## Set up Supabase (do this once, by hand)

1. Go to <https://supabase.com>, sign in, and choose **New project**.
2. Pick an organisation, name the project (e.g. `healthlens`), and choose a region near you.
3. Supabase generates a **database password**. Copy it somewhere safe *now* - the page shows
   it once. If you lose it: **Project Settings -> Database -> Reset database password**.
4. Wait for the project to finish provisioning (a minute or two).
5. Open **Project Settings -> Database -> Connection string -> URI** and copy the string. It
   looks like:

   ```
   postgresql://postgres:[YOUR-PASSWORD]@db.abcdefghijklm.supabase.co:5432/postgres
   ```

6. Replace `[YOUR-PASSWORD]` with the password from step 3. If the password contains `@`, `:`,
   `/` or `#`, percent-encode it (`@` -> `%40`, `:` -> `%3A`, `/` -> `%2F`, `#` -> `%23`).

### Direct (5432) or pooled (6543)

The same page offers two ports, and they are not interchangeable:

| port | what it is | use it for |
|---|---|---|
| **5432** | **direct connection** | the seed script, and the API when running locally |
| 6543 | the transaction **pooler** (PgBouncer) | serverless / many short-lived connections |

Use **5432** tonight. The code handles either - a `:6543` URL automatically disables
server-side prepared statements, which transaction pooling cannot keep between statements -
but the direct port is simpler and supports everything. On IPv4-only networks Supabase may
show the direct host as unavailable and offer the pooler instead; in that case use the pooler
URI exactly as Supabase gives it.

TLS is not optional on Supabase. You do not need to add anything: `sslmode=require` is added
automatically to any non-local URL, and an `sslmode` you set yourself is left alone.

## Create the schema and load the demo data

From the repository root, with the repo virtualenv active:

```bash
pip install -r backend/requirements.txt

# Windows PowerShell
$env:DATABASE_URL = "postgresql://postgres:YOUR-PASSWORD@db.abcdefghijklm.supabase.co:5432/postgres"
# macOS / Linux
export DATABASE_URL="postgresql://postgres:YOUR-PASSWORD@db.abcdefghijklm.supabase.co:5432/postgres"

python -m backend.scripts.seed_db
```

It applies `backend/db/schema.sql` (every statement is idempotent) and writes the eight
synthetic patients with their doctor assignments, conditions, medications and observations,
plus the demo accounts. **Running it again is safe** - it upserts and replaces the child rows
rather than duplicating them. Options: `--schema-only` creates the tables and loads nothing,
`--database-url` passes the URL instead of using the environment.

Expected output:

```
schema applied from .../backend/db/schema.sql
seeded 8 patients and 11 accounts across 2 doctors
start the API with the same DATABASE_URL set to use it
```

Then start the API with the same variable set:

```bash
uvicorn backend.app.main:app --reload
```

Check it took effect (log in as the admin first):

```bash
curl -c jar -X POST localhost:8000/api/v1/auth/login -H 'content-type: application/json' \
  -d '{"email":"admin@demo.healthlens.test","password":"demo-password-123"}'
curl -b jar localhost:8000/api/v1/admin/system
# {"data_source":"postgres","storage":"postgres","patient_count":8, ...}
```

The accounts in Postgres carry the password hash that `DEMO_PASSWORD` had **when you ran the
seed script**. If you change `DEMO_PASSWORD` later, run the seed script again.

In the Supabase dashboard, **Table Editor** shows `patients`, `doctor_patients`,
`patient_conditions`, `medications`, `observations`, `users` and `audit_events`; the audit
table fills up as you click around the API.

## Switch back to memory

Unset the variable and restart - that is all:

```bash
# Windows PowerShell
Remove-Item Env:\DATABASE_URL
# macOS / Linux
unset DATABASE_URL

uvicorn backend.app.main:app --reload   # "storage":"memory" again
```

Also check `.env` in the repository root: `Settings` reads it, so a `DATABASE_URL=` line there
keeps Postgres on even with the shell variable gone. Comment the line out to go back to memory.
`create_app(settings, repo=...)` (used by the tests) always wins over `DATABASE_URL`.

## Schema

`backend/db/schema.sql` mirrors the frozen dataclasses in `backend/app/data/models.py` and
`backend/app/users.py`:

- `patients` - one row per patient (`is_synthetic` kept, so fake data is always labelled)
- `doctor_patients` - which doctor may see which patient; the only source of assignments
- `patient_conditions` - registry condition ids a patient is already diagnosed with
- `medications`, `observations` - the record's children, cascading on delete
- `users` - accounts, argon2 hash in `password_hash`, `role` constrained to the three roles
- `audit_events` - append-only; identifiers and actions only, never clinical content

## Verify

```bash
ruff check backend && ruff format --check backend
pytest backend/tests                     # no database needed: the Postgres tests skip

DATABASE_URL="postgresql://..." pytest backend/tests -m integration
```

The integration tests are marked `@pytest.mark.integration` and skip themselves when
`DATABASE_URL` is unset, so CI and anyone cloning the repository stay green without a database.
They apply the schema, seed it, compare `PostgresRepository` against the synthetic data, prove
the audit log refuses a `DELETE`, and boot the app on Postgres.

Not done tonight, on purpose: no migration tool (the schema file is applied as-is), no
connection retry/backoff beyond SQLAlchemy's `pool_pre_ping`, and users are still seeded from
the synthetic patients rather than invited through the API.

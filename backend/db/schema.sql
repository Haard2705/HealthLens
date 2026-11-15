-- HealthLens schema (H17). One plain file, applied by backend/scripts/seed_db.py.
--
-- It is the single source of truth for the Postgres tables: the backend only ever sends
-- parameterised queries against these names (backend/app/db/queries.py), so there is no
-- second copy of the schema in Python and no migration tool to keep in step tonight.
--
-- Every statement is idempotent: running the seed script twice is safe.
-- The tables mirror the frozen dataclasses in backend/app/data/models.py and backend/app/users.py.

CREATE TABLE IF NOT EXISTS patients (
    id                text PRIMARY KEY,
    name              text NOT NULL,
    age               integer NOT NULL CHECK (age >= 0 AND age < 130),
    sex               text NOT NULL,
    bmi               double precision NOT NULL,
    smoking_status    text NOT NULL,
    physically_active boolean NOT NULL,
    general_health    integer NOT NULL CHECK (general_health BETWEEN 1 AND 5),
    is_synthetic      boolean NOT NULL DEFAULT true
);

-- Which doctor may see which patient. The API reads assignments only through this table.
CREATE TABLE IF NOT EXISTS doctor_patients (
    doctor_id  text NOT NULL,
    patient_id text NOT NULL REFERENCES patients (id) ON DELETE CASCADE,
    PRIMARY KEY (doctor_id, patient_id)
);
CREATE INDEX IF NOT EXISTS doctor_patients_patient_idx ON doctor_patients (patient_id);

-- Conditions a patient has already been diagnosed with (registry ids from ml/src/conditions.yaml).
CREATE TABLE IF NOT EXISTS patient_conditions (
    patient_id   text NOT NULL REFERENCES patients (id) ON DELETE CASCADE,
    condition_id text NOT NULL,
    PRIMARY KEY (patient_id, condition_id)
);

CREATE TABLE IF NOT EXISTS medications (
    id         bigserial PRIMARY KEY,
    patient_id text NOT NULL REFERENCES patients (id) ON DELETE CASCADE,
    name       text NOT NULL,
    dose       text NOT NULL,
    frequency  text NOT NULL,
    started    date NOT NULL
);
CREATE INDEX IF NOT EXISTS medications_patient_idx ON medications (patient_id);

CREATE TABLE IF NOT EXISTS observations (
    id          bigserial PRIMARY KEY,
    patient_id  text NOT NULL REFERENCES patients (id) ON DELETE CASCADE,
    observed_on date NOT NULL,
    bmi         double precision NOT NULL,
    systolic_bp integer NOT NULL,
    hba1c       double precision NOT NULL
);
CREATE INDEX IF NOT EXISTS observations_patient_idx ON observations (patient_id, observed_on);

CREATE TABLE IF NOT EXISTS users (
    id            text PRIMARY KEY,
    email         text NOT NULL UNIQUE,
    display_name  text NOT NULL,
    role          text NOT NULL CHECK (role IN ('admin', 'doctor', 'patient')),
    password_hash text,
    active        boolean NOT NULL DEFAULT true,
    patient_id    text REFERENCES patients (id) ON DELETE SET NULL,
    doctor_id     text
);

-- Append-only audit log (ADR-0014): identifiers and actions only, never clinical content.
-- The backend issues INSERT and SELECT against this table and nothing else; the trigger below
-- makes that a database guarantee rather than a convention.
CREATE TABLE IF NOT EXISTS audit_events (
    id            bigserial PRIMARY KEY,
    occurred_at   timestamptz NOT NULL DEFAULT now(),
    actor_id      text,
    actor_role    text,
    action        text NOT NULL,
    resource_type text NOT NULL,
    resource_id   text,
    patient_id    text,
    outcome       text NOT NULL,
    request_id    text NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_events_actor_idx ON audit_events (actor_id);
CREATE INDEX IF NOT EXISTS audit_events_patient_idx ON audit_events (patient_id);

CREATE OR REPLACE FUNCTION healthlens_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_events is append-only: % is not allowed', TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_events_append_only ON audit_events;
CREATE TRIGGER audit_events_append_only
    BEFORE UPDATE OR DELETE OR TRUNCATE ON audit_events
    FOR EACH STATEMENT EXECUTE FUNCTION healthlens_append_only();

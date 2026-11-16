"""Every SQL statement the backend sends, in one place.

All of them are parameterised (``:name``); no value is ever formatted into a statement. The
table and column names match ``backend/db/schema.sql`` exactly. ``audit_events`` appears only
in an INSERT and a SELECT: the log is append-only (ADR-0014).
"""

from __future__ import annotations

PATIENT_COLUMNS = (
    "id, name, age, sex, bmi, smoking_status, physically_active, general_health, is_synthetic"
)

SELECT_PATIENT = f"SELECT {PATIENT_COLUMNS} FROM patients WHERE id = :id"
SELECT_PATIENTS = f"SELECT {PATIENT_COLUMNS} FROM patients ORDER BY id"
SELECT_PATIENTS_FOR_DOCTOR = f"""
    SELECT {PATIENT_COLUMNS}
      FROM patients p
      JOIN doctor_patients dp ON dp.patient_id = p.id
     WHERE dp.doctor_id = :doctor_id
     ORDER BY p.id
"""
SELECT_ASSIGNED_DOCTOR = """
    SELECT doctor_id FROM doctor_patients WHERE patient_id = :patient_id ORDER BY doctor_id
"""
SELECT_CONDITIONS = """
    SELECT condition_id FROM patient_conditions WHERE patient_id = :patient_id
     ORDER BY condition_id
"""
SELECT_MEDICATIONS = """
    SELECT name, dose, frequency, started FROM medications
     WHERE patient_id = :patient_id ORDER BY started, name
"""
SELECT_OBSERVATIONS = """
    SELECT observed_on, bmi, systolic_bp, hba1c FROM observations
     WHERE patient_id = :patient_id ORDER BY observed_on
"""

USER_COLUMNS = "id, email, display_name, role, password_hash, active, patient_id, doctor_id"
SELECT_USER = f"SELECT {USER_COLUMNS} FROM users WHERE id = :id"
SELECT_USER_BY_EMAIL = f"SELECT {USER_COLUMNS} FROM users WHERE email = :email"
SELECT_USERS = f"SELECT {USER_COLUMNS} FROM users ORDER BY role, email"
COUNT_ACTIVE_ADMINS = "SELECT count(*) FROM users WHERE role = 'admin' AND active"
INSERT_USER = f"""
    INSERT INTO users ({USER_COLUMNS})
    VALUES (:id, :email, :display_name, :role, :password_hash, :active, :patient_id, :doctor_id)
    ON CONFLICT (id) DO UPDATE SET
        email = excluded.email,
        display_name = excluded.display_name,
        role = excluded.role,
        password_hash = excluded.password_hash,
        active = excluded.active,
        patient_id = excluded.patient_id,
        doctor_id = excluded.doctor_id
"""
SET_USER_ACTIVE = "UPDATE users SET active = :active WHERE id = :id"

INSERT_AUDIT_EVENT = """
    INSERT INTO audit_events (
        occurred_at, actor_id, actor_role, action, resource_type, resource_id,
        patient_id, outcome, request_id
    ) VALUES (
        :occurred_at, :actor_id, :actor_role, :action, :resource_type, :resource_id,
        :patient_id, :outcome, :request_id
    ) RETURNING id
"""
# The filters are all optional, so each one is "no filter given, or it matches". The casts are
# required: given a bare :actor on both sides of an OR, Postgres cannot infer the parameter's
# type and refuses the statement with "could not determine data type".
SELECT_AUDIT_EVENTS = """
    SELECT id, occurred_at, actor_id, actor_role, action, resource_type, resource_id,
           patient_id, outcome, request_id
      FROM audit_events
     WHERE (CAST(:actor AS TEXT) IS NULL OR actor_id = CAST(:actor AS TEXT))
       AND (CAST(:patient AS TEXT) IS NULL OR patient_id = CAST(:patient AS TEXT))
       AND (CAST(:action AS TEXT) IS NULL OR action = CAST(:action AS TEXT))
       AND (CAST(:outcome AS TEXT) IS NULL OR outcome = CAST(:outcome AS TEXT))
     ORDER BY id
"""

# Seeding (backend/scripts/seed_db.py only).
INSERT_PATIENT = f"""
    INSERT INTO patients ({PATIENT_COLUMNS}) VALUES (
        :id, :name, :age, :sex, :bmi, :smoking_status, :physically_active,
        :general_health, :is_synthetic
    ) ON CONFLICT (id) DO UPDATE SET
        name = excluded.name, age = excluded.age, sex = excluded.sex, bmi = excluded.bmi,
        smoking_status = excluded.smoking_status,
        physically_active = excluded.physically_active,
        general_health = excluded.general_health, is_synthetic = excluded.is_synthetic
"""
INSERT_ASSIGNMENT = """
    INSERT INTO doctor_patients (doctor_id, patient_id) VALUES (:doctor_id, :patient_id)
    ON CONFLICT DO NOTHING
"""
INSERT_CONDITION = """
    INSERT INTO patient_conditions (patient_id, condition_id) VALUES (:patient_id, :condition_id)
    ON CONFLICT DO NOTHING
"""
DELETE_MEDICATIONS = "DELETE FROM medications WHERE patient_id = :patient_id"
DELETE_OBSERVATIONS = "DELETE FROM observations WHERE patient_id = :patient_id"
INSERT_MEDICATION = """
    INSERT INTO medications (patient_id, name, dose, frequency, started)
    VALUES (:patient_id, :name, :dose, :frequency, :started)
"""
INSERT_OBSERVATION = """
    INSERT INTO observations (patient_id, observed_on, bmi, systolic_bp, hba1c)
    VALUES (:patient_id, :observed_on, :bmi, :systolic_bp, :hba1c)
"""

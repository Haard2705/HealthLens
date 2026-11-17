"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { useSession } from "@/components/SessionProvider";
import { ErrorNote, Loading, SignInRequired, WrongRole } from "@/components/States";
import { apiGet } from "@/lib/api";
import { useResource } from "@/lib/useResource";
import { SEX_LABEL, conditionName } from "@/lib/format";
import type { Condition, Items, PatientSummary } from "@/lib/types";

/** A patient assigned to the other demo doctor, so the 404 path can be shown on purpose. */
const UNASSIGNED_DEMO_ID = "P-0107";

export function PatientList() {
  const { user, loading: sessionLoading } = useSession();
  const [query, setQuery] = useState("");
  const [search, setSearch] = useState("");

  // Debounce, so typing does not fire a request per keystroke.
  useEffect(() => {
    const timer = setTimeout(() => setSearch(query.trim()), 250);
    return () => clearTimeout(timer);
  }, [query]);

  const signedInDoctor = user?.role === "doctor";

  const loadPatients = useCallback(
    (signal: AbortSignal) =>
      signedInDoctor
        ? apiGet<Items<PatientSummary>>("/doctor/patients", { q: search || undefined }, signal)
        : Promise.resolve({ items: [] }),
    [signedInDoctor, search],
  );
  const patients = useResource(loadPatients);

  const loadConditions = useCallback(
    (signal: AbortSignal) => apiGet<Items<Condition>>("/conditions", undefined, signal),
    [],
  );
  const conditions = useResource(loadConditions);
  const names = conditions.data?.items ?? [];

  if (sessionLoading) return <Loading what="your account" />;
  if (!user) return <SignInRequired what="your patient list" />;
  if (!signedInDoctor) return <WrongRole expected="doctor" />;

  const items = patients.data?.items ?? [];

  return (
    <div className="stack">
      <div className="row between">
        <div>
          <h1 className="pt">My patients</h1>
          <p className="muted">Patients assigned to you.</p>
        </div>
        <label className="f">
          Search
          <input
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Name or ID"
          />
        </label>
      </div>

      {patients.loading && <Loading what="your patients" />}
      {patients.error && <ErrorNote message={patients.error} onRetry={patients.reload} />}

      {!patients.loading && !patients.error && (
        <div className="card tw">
          <table>
            <thead>
              <tr>
                <th>Patient</th>
                <th>Conditions on file</th>
                <th>Record</th>
                <th>
                  <span className="sr">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {items.length === 0 && (
                <tr>
                  <td colSpan={4} className="muted">
                    No patients match that search.
                  </td>
                </tr>
              )}
              {items.map((patient) => (
                <tr key={patient.id}>
                  <td>
                    <strong>{patient.name}</strong>
                    <div className="small muted mono">
                      {patient.id} · {patient.age} · {SEX_LABEL[patient.sex] ?? patient.sex}
                    </div>
                  </td>
                  <td>
                    {patient.conditions.length > 0 ? (
                      <div className="chips">
                        {patient.conditions.map((id) => (
                          <span className="pill" key={id}>
                            {conditionName(names, id)}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="muted small">none recorded</span>
                    )}
                  </td>
                  <td>
                    <span className="pill warn">synthetic</span>
                  </td>
                  <td className="r">
                    <Link className="btn small" href={`/doctor/${patient.id}`}>
                      Open
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="card flat row between">
        <div className="small">
          <strong>Demo control.</strong> Try to open a patient who is not assigned to you. The API
          answers with the same “not found” it would give for a patient that does not exist, and
          the attempt is written to the audit log.
        </div>
        <Link className="btn small" href={`/doctor/${UNASSIGNED_DEMO_ID}`}>
          Try {UNASSIGNED_DEMO_ID}
        </Link>
      </div>
    </div>
  );
}

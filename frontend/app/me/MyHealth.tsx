"use client";

import { useCallback } from "react";
import { RecordDetail } from "@/components/RecordDetail";
import { useSession } from "@/components/SessionProvider";
import {
  ErrorNote,
  Loading,
  RecordNotFound,
  SignInRequired,
  WrongRole,
} from "@/components/States";
import { apiGet } from "@/lib/api";
import { useResource } from "@/lib/useResource";
import type {
  Condition,
  Items,
  Medication,
  Observation,
  PatientRecord,
} from "@/lib/types";

export function MyHealth() {
  const { user, loading: sessionLoading } = useSession();
  const isPatient = user?.role === "patient";

  const loadRecord = useCallback(
    (signal: AbortSignal) =>
      isPatient
        ? apiGet<PatientRecord>("/me/record", undefined, signal)
        : Promise.reject(new Error("not a patient")),
    [isPatient],
  );
  const record = useResource(loadRecord);

  const loadMedications = useCallback(
    (signal: AbortSignal) =>
      isPatient
        ? apiGet<Items<Medication>>("/me/medications", undefined, signal)
        : Promise.resolve({ items: [] }),
    [isPatient],
  );
  const medications = useResource(loadMedications);

  const loadObservations = useCallback(
    (signal: AbortSignal) =>
      isPatient
        ? apiGet<Items<Observation>>("/me/observations", undefined, signal)
        : Promise.resolve({ items: [] }),
    [isPatient],
  );
  const observations = useResource(loadObservations);

  const loadConditions = useCallback(
    (signal: AbortSignal) => apiGet<Items<Condition>>("/conditions", undefined, signal),
    [],
  );
  const conditions = useResource(loadConditions);

  if (sessionLoading) return <Loading what="your account" />;
  if (!user) return <SignInRequired what="your own health record" />;
  if (!isPatient) return <WrongRole expected="patient" />;

  if (record.cause?.isNotFound) {
    return <RecordNotFound backHref="/" backLabel="Back to the home page" />;
  }
  if (record.loading || conditions.loading) return <Loading what="your record" />;
  if (record.error || !record.data) {
    return <ErrorNote message={record.error ?? "No record came back."} onRetry={record.reload} />;
  }

  const sorted = [...(observations.data?.items ?? [])].sort((a, b) =>
    a.date.localeCompare(b.date),
  );

  return (
    <div className="stack">
      <div className="card flat small">
        <strong>This is your own record.</strong> Nobody else&apos;s record is reachable from this
        account, and your care team can see that you looked at it.
      </div>

      {(medications.error || observations.error) && (
        <ErrorNote
          message={medications.error ?? observations.error ?? "Part of your record did not load."}
          onRetry={medications.error ? medications.reload : observations.reload}
        />
      )}

      <RecordDetail
        record={record.data}
        conditions={conditions.data?.items ?? []}
        observations={sorted}
        medications={medications.data?.items ?? []}
        mode="patient"
      />
    </div>
  );
}

"use client";

import Link from "next/link";
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
import { medicationsFromTimeline, observationsFromTimeline } from "@/lib/format";
import type { Condition, Items, PatientRecord, TimelineEvent } from "@/lib/types";

export function PatientDetail({ patientId }: { patientId: string }) {
  const { user, loading: sessionLoading } = useSession();
  const isDoctor = user?.role === "doctor";

  const loadRecord = useCallback(
    (signal: AbortSignal) =>
      isDoctor
        ? apiGet<PatientRecord>(
            `/doctor/patients/${encodeURIComponent(patientId)}`,
            undefined,
            signal,
          )
        : Promise.reject(new Error("not a doctor")),
    [isDoctor, patientId],
  );
  const record = useResource(loadRecord);

  const loadTimeline = useCallback(
    (signal: AbortSignal) =>
      isDoctor
        ? apiGet<Items<TimelineEvent>>(
            `/doctor/patients/${encodeURIComponent(patientId)}/timeline`,
            undefined,
            signal,
          )
        : Promise.resolve({ items: [] }),
    [isDoctor, patientId],
  );
  const timeline = useResource(loadTimeline);

  const loadConditions = useCallback(
    (signal: AbortSignal) => apiGet<Items<Condition>>("/conditions", undefined, signal),
    [],
  );
  const conditions = useResource(loadConditions);

  if (sessionLoading) return <Loading what="your account" />;
  if (!user) return <SignInRequired what="a patient record" />;
  if (!isDoctor) return <WrongRole expected="doctor" />;

  if (record.cause?.isNotFound) {
    return <RecordNotFound backHref="/doctor" backLabel="Back to my patients" />;
  }
  if (record.loading || conditions.loading) return <Loading what="the record" />;
  if (record.error || !record.data) {
    return <ErrorNote message={record.error ?? "No record came back."} onRetry={record.reload} />;
  }

  const events = timeline.data?.items ?? [];

  return (
    <div className="stack">
      <div className="row">
        <Link className="btn small" href="/doctor">
          ← My patients
        </Link>
      </div>

      {timeline.error && (
        <ErrorNote
          message={`The timeline did not load, so the trends are empty. ${timeline.error}`}
          onRetry={timeline.reload}
        />
      )}

      <RecordDetail
        record={record.data}
        conditions={conditions.data?.items ?? []}
        observations={observationsFromTimeline(events)}
        medications={medicationsFromTimeline(events)}
        mode="clinician"
      />
    </div>
  );
}

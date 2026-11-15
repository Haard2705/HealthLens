"use client";

import { useCallback } from "react";
import { apiGet } from "@/lib/api";
import { useResource } from "@/lib/useResource";
import type { Condition, Items } from "@/lib/types";

interface Fact {
  value: string;
  label: string;
}

/**
 * The only live numbers on the home page. They come from the public `/conditions` endpoint;
 * nothing here needs a sign-in, and a failure degrades to a quiet note rather than an error.
 */
export function AtAGlance() {
  const load = useCallback(
    (signal: AbortSignal) => apiGet<Items<Condition>>("/conditions", undefined, signal),
    [],
  );
  const { data, loading, error } = useResource(load);

  if (loading) {
    return (
      <p className="muted small" role="status">
        Loading the numbers…
      </p>
    );
  }
  if (error || !data) {
    return (
      <p className="muted small">
        The at-a-glance numbers need the API. Start the backend on port 8000 and reload.
      </p>
    );
  }

  const conditions = data.items;
  const enabled = conditions.filter((c) => c.prediction_enabled === true).length;
  const gateHasRun = conditions.some((c) => c.prediction_enabled !== null);
  const sensitive = conditions.filter((c) => c.sensitive).length;

  const facts: Fact[] = [
    { value: String(conditions.length), label: "conditions covered" },
    {
      value: gateHasRun ? `${enabled} of ${conditions.length}` : "not yet",
      label: gateHasRun ? "can show a personal estimate" : "models trained in this build",
    },
    { value: String(sensitive), label: "kept to analytics only, by rule" },
    { value: "3", label: "roles with different access" },
  ];

  return (
    <section className="facts" aria-label="At a glance">
      {facts.map((fact) => (
        <div className="fact" key={fact.label}>
          <span className="v">{fact.value}</span>
          <span className="l">{fact.label}</span>
        </div>
      ))}
    </section>
  );
}

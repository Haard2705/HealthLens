"use client";

import { useCallback, useState } from "react";
import { BarList, type BarRow } from "@/components/BarList";
import { ErrorNote, Loading } from "@/components/States";
import { apiGet } from "@/lib/api";
import { useResource } from "@/lib/useResource";
import {
  MIN_GROUP_SIZE,
  groupOrderIndex,
  humanise,
  pct,
  suppressedGroups,
} from "@/lib/format";
import type { Condition, Items, Prevalence } from "@/lib/types";

const GROUP_BY = [
  { value: "age_band", label: "Age band" },
  { value: "bmi_band", label: "BMI band" },
  { value: "phys_active", label: "Physical activity" },
  { value: "sex", label: "Sex" },
];

export function ExploreView() {
  const [condition, setCondition] = useState("diabetes");
  const [by, setBy] = useState("age_band");

  const loadConditions = useCallback(
    (signal: AbortSignal) => apiGet<Items<Condition>>("/conditions", undefined, signal),
    [],
  );
  const conditions = useResource(loadConditions);

  const loadPrevalence = useCallback(
    (signal: AbortSignal) =>
      apiGet<Prevalence>("/stats/prevalence", { condition, by }, signal),
    [condition, by],
  );
  const prevalence = useResource(loadPrevalence);

  const chosen = conditions.data?.items.find((c) => c.id === condition);
  const groupLabel = (GROUP_BY.find((g) => g.value === by)?.label ?? by).toLowerCase();

  const rows: BarRow[] = (() => {
    const payload = prevalence.data;
    if (!payload) return [];
    const present = payload.groups.map((g) => g.group);
    const shown: BarRow[] = payload.groups.map((g) => ({
      label: g.group,
      value: g.prevalence,
      n: g.n,
    }));
    const hidden: BarRow[] = suppressedGroups(by, payload.source, present).map((label) => ({
      label,
      suppressed: true,
    }));
    return [...shown, ...hidden].sort(
      (a, b) => groupOrderIndex(by, a.label) - groupOrderIndex(by, b.label),
    );
  })();

  return (
    <div className="stack">
      <div className="row" style={{ alignItems: "flex-end" }}>
        <label className="f">
          Condition
          <select
            value={condition}
            onChange={(event) => setCondition(event.target.value)}
            disabled={conditions.loading || !conditions.data}
          >
            {conditions.data ? (
              conditions.data.items.map((item) => (
                <option value={item.id} key={item.id}>
                  {item.display_name}
                </option>
              ))
            ) : (
              <option value={condition}>Loading…</option>
            )}
          </select>
        </label>
        <label className="f">
          Group by
          <select value={by} onChange={(event) => setBy(event.target.value)}>
            {GROUP_BY.map((group) => (
              <option value={group.value} key={group.value}>
                {group.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      {conditions.error && (
        <ErrorNote message={conditions.error} onRetry={conditions.reload} />
      )}

      <div className="card">
        <div className="row between">
          <h2 style={{ fontSize: 16 }}>
            {chosen?.display_name ?? humanise(condition)} by {groupLabel}
          </h2>
          {chosen && (
            <span className="pill">
              {humanise(chosen.body_system)} · Tier {chosen.tier}
            </span>
          )}
        </div>

        <div style={{ marginTop: 10 }}>
          {prevalence.loading && <Loading what="the prevalence numbers" />}
          {prevalence.error && (
            <ErrorNote message={prevalence.error} onRetry={prevalence.reload} />
          )}
          {!prevalence.loading && !prevalence.error && rows.length === 0 && (
            <p className="muted">No groups were returned for this combination.</p>
          )}
          {rows.length > 0 && <BarList rows={rows} format={(value) => pct(value)} />}
        </div>

        <p className="small muted" style={{ marginTop: 12 }}>
          Survey-weighted rates: each respondent is counted by their survey weight, so the share
          reflects the adult population rather than who happened to answer. Groups with fewer than{" "}
          {MIN_GROUP_SIZE} respondents are hidden, not shown as a misleading number.
          {prevalence.data?.note ? ` ${prevalence.data.note}` : ""}
        </p>
        {prevalence.data?.source === "sample" && (
          <p className="small" style={{ marginTop: 6, color: "var(--warn-ink)" }}>
            This build has no cleaned survey file on disk, so the API is answering with clearly
            labelled illustrative numbers. Run the data pipeline to see real rates here.
          </p>
        )}
      </div>

      {chosen && (
        <div className="card flat small">
          <strong>{chosen.display_name}.</strong> {chosen.description}
          {chosen.sensitive && " This condition is treated as sensitive: population analytics only."}
        </div>
      )}
    </div>
  );
}

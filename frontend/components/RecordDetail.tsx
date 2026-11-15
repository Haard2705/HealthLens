"use client";

import { useState } from "react";
import { TrendChart, type TrendPoint } from "./TrendChart";
import { RiskContextPanel } from "./RiskContextPanel";
import {
  GENERAL_HEALTH,
  METRICS,
  SEX_LABEL,
  conditionName,
  quarterLabel,
  type Metric,
} from "@/lib/format";
import type { Condition, Medication, Observation, PatientRecord } from "@/lib/types";

export type RecordMode = "clinician" | "patient";

interface RecordDetailProps {
  record: PatientRecord;
  conditions: Condition[];
  observations: Observation[];
  medications: Medication[];
  mode: RecordMode;
  /** Shown above the record when an admin is reading someone else's chart. */
  accessNote?: string;
}

type Tab = "trends" | "meds" | "risk";

function latestChange(values: number[]): number | null {
  if (values.length < 5) return null;
  return values[values.length - 1] - values[values.length - 5];
}

function Sidebar({ record, conditions, observations, mode }: Omit<RecordDetailProps, "medications">) {
  const kpis = METRICS.map((metric) => {
    const values = observations.map((o) => o[metric.key]);
    const latest = values.at(-1);
    const change = latestChange(values);
    return { metric, latest, change };
  });

  return (
    <aside className="stack rec-side" aria-label="Patient summary">
      <div className="card stack" style={{ gap: 10 }}>
        <div>
          <h1 className="pt">{mode === "patient" ? "My health" : record.name}</h1>
          <div className="small muted mono">
            {record.id} · {record.age} · {SEX_LABEL[record.sex] ?? record.sex} · BMI{" "}
            {record.bmi.toFixed(1)}
          </div>
        </div>
        <div className="chips">
          {record.conditions.length > 0 ? (
            record.conditions.map((id) => (
              <span className="pill" key={id}>
                {conditionName(conditions, id)}
              </span>
            ))
          ) : (
            <span className="pill">no conditions on file</span>
          )}
          {record.is_synthetic && <span className="pill warn">synthetic record</span>}
        </div>
        <div className="small muted">
          {mode === "patient" ? "You describe" : "Self-reported"}: general health{" "}
          {GENERAL_HEALTH[record.general_health] ?? record.general_health}, {record.smoking_status}{" "}
          smoker, {record.physically_active ? "physically active" : "not physically active"}.
        </div>
      </div>

      <div className="kpis">
        {kpis.map(({ metric, latest, change }) => (
          <div className="card kpi" key={metric.key}>
            <span className="small muted">{metric.short}</span>
            <span className="v">
              {latest === undefined ? "—" : latest.toFixed(metric.decimals)}{" "}
              <span className="small muted">{metric.unit}</span>
            </span>
            {change !== null &&
              (Math.abs(change) < 10 ** -metric.decimals / 2 ? (
                <span className="d muted">no change vs a year ago</span>
              ) : (
                <span className={`d ${change > 0 ? "up" : "down"}`}>
                  {change > 0 ? "+" : ""}
                  {change.toFixed(metric.decimals)} vs a year ago
                </span>
              ))}
          </div>
        ))}
      </div>
    </aside>
  );
}

function Trends({
  observations,
  metric,
  onMetric,
  mode,
}: {
  observations: Observation[];
  metric: Metric;
  onMetric: (key: Metric["key"]) => void;
  mode: RecordMode;
}) {
  const points: TrendPoint[] = observations.map((o) => ({
    label: quarterLabel(o.date),
    value: o[metric.key],
  }));

  return (
    <div className="card">
      <div className="row between">
        <h2 style={{ fontSize: 16 }}>
          {metric.label} over {observations.length} check-ups
        </h2>
        <div className="seg" role="group" aria-label="Metric">
          {METRICS.map((item) => (
            <button
              type="button"
              key={item.key}
              aria-pressed={item.key === metric.key}
              onClick={() => onMetric(item.key)}
            >
              {item.short}
            </button>
          ))}
        </div>
      </div>
      <div style={{ marginTop: 10 }}>
        {points.length === 0 ? (
          <p className="muted">No check-ups on file.</p>
        ) : (
          <TrendChart points={points} metric={metric} />
        )}
      </div>
      <p className="small muted" style={{ marginTop: 8 }}>
        The dashed line is a common reference value, shown for context only — it is not a target
        {mode === "patient" ? " and not advice" : ""}. Units: {metric.unit}.
      </p>
    </div>
  );
}

function Medications({ medications, mode }: { medications: Medication[]; mode: RecordMode }) {
  return (
    <>
      <div className="card tw">
        {medications.length === 0 ? (
          <p className="muted">No medications on file.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Medication</th>
                <th>Dose</th>
                <th>How often</th>
                <th>Since</th>
              </tr>
            </thead>
            <tbody>
              {medications.map((medication) => (
                <tr key={`${medication.name}-${medication.started}`}>
                  <td>
                    <strong>{medication.name}</strong>
                  </td>
                  <td>{medication.dose || "—"}</td>
                  <td>{medication.frequency || "—"}</td>
                  <td className="num">{medication.started}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      {mode === "patient" && (
        <p className="small muted">
          Questions about your medication belong with your care team. HealthLens cannot give
          medication advice.
        </p>
      )}
    </>
  );
}

export function RecordDetail({
  record,
  conditions,
  observations,
  medications,
  mode,
  accessNote,
}: RecordDetailProps) {
  const [tab, setTab] = useState<Tab>("trends");
  const [metricKey, setMetricKey] = useState<Metric["key"]>("bmi");
  const metric = METRICS.find((m) => m.key === metricKey) ?? METRICS[0];

  const tabs: { key: Tab; label: string }[] = [
    { key: "trends", label: mode === "patient" ? "My trends" : "Observation trends" },
    { key: "meds", label: "Medications" },
    { key: "risk", label: mode === "patient" ? "Understand my risk" : "Risk context" },
  ];

  return (
    <div className="stack">
      {accessNote && (
        <div className="alert warn small" role="note">
          {accessNote}
        </div>
      )}
      <div className="rec">
        <Sidebar
          record={record}
          conditions={conditions}
          observations={observations}
          mode={mode}
        />
        <div className="stack">
          <div className="tabs">
            {tabs.map((item) => (
              <button
                type="button"
                key={item.key}
                aria-current={tab === item.key}
                onClick={() => setTab(item.key)}
              >
                {item.label}
              </button>
            ))}
          </div>

          {tab === "trends" && (
            <Trends
              observations={observations}
              metric={metric}
              onMetric={setMetricKey}
              mode={mode}
            />
          )}
          {tab === "meds" && <Medications medications={medications} mode={mode} />}
          {tab === "risk" && (
            <RiskContextPanel
              patientId={record.id}
              conditions={conditions}
              existingConditions={record.conditions}
              mode={mode}
            />
          )}
        </div>
      </div>
    </div>
  );
}

/** Small, pure formatting and domain helpers shared by the screens. */

import type { Condition, Medication, Observation, TimelineEvent } from "./types";

/** The release gate's ROC-AUC floor (ADR-0016, `ml/src/gate.py`). */
export const GATE_MIN_AUC = 0.7;
/** Groups smaller than this are suppressed by the API (`backend/app/stats.py`). */
export const MIN_GROUP_SIZE = 30;

export const SAFETY_BANNER =
  "Model results are real (CDC BRFSS 2025); patients are synthetic · not medical advice";

export function pct(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function count(value: number): string {
  return value.toLocaleString("en-US");
}

export function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export function titleCase(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

/** "mental_health" -> "Mental health" */
export function humanise(value: string): string {
  return titleCase(value.replace(/_/g, " "));
}

export const SEX_LABEL: Record<string, string> = { M: "Male", F: "Female" };

export const GENERAL_HEALTH: Record<number, string> = {
  1: "excellent",
  2: "very good",
  3: "good",
  4: "fair",
  5: "poor",
};

/** Whether this condition may show an individual estimate. null means "gate has not run". */
export function predictionEnabled(condition: Condition): boolean | null {
  return condition.prediction_enabled;
}

export function conditionName(conditions: Condition[], id: string): string {
  return conditions.find((c) => c.id === id)?.display_name ?? humanise(id);
}

/** The order the API's group labels should read in, per `by` value. */
const GROUP_ORDER: Record<string, string[]> = {
  age_band: [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
    "55-59", "60-64", "65-69", "70-74", "75-79", "80+",
    "18-34", "35-49", "50-64", "65+",
  ],
  bmi_band: ["under 18.5", "18.5-24.9", "25-29.9", "30+"],
  phys_active: ["active", "inactive"],
  sex: ["female", "male", "F", "M"],
};

export function groupOrderIndex(by: string, label: string): number {
  const order = GROUP_ORDER[by];
  if (!order) return 0;
  const index = order.indexOf(label);
  return index === -1 ? order.length : index;
}

/**
 * Group labels the API suppressed: present in the known vocabulary for this `by` but absent
 * from the response, because fewer than MIN_GROUP_SIZE respondents fell into them.
 * Only meaningful for real survey data; the sample source has its own short label set.
 */
export function suppressedGroups(by: string, source: string, present: string[]): string[] {
  if (source !== "brfss") return [];
  const order = GROUP_ORDER[by];
  if (!order) return [];
  const known = by === "age_band" ? order.slice(0, 13) : order;
  return known.filter((label) => !present.includes(label));
}

export interface Metric {
  key: "bmi" | "systolic_bp" | "hba1c";
  label: string;
  short: string;
  unit: string;
  decimals: number;
  /** A common reference value, drawn as a dashed line for context only. */
  reference: number;
}

export const METRICS: Metric[] = [
  { key: "bmi", label: "BMI", short: "BMI", unit: "kg/m²", decimals: 1, reference: 30 },
  {
    key: "systolic_bp",
    label: "Systolic blood pressure",
    short: "Systolic BP",
    unit: "mmHg",
    decimals: 0,
    reference: 140,
  },
  { key: "hba1c", label: "HbA1c", short: "HbA1c", unit: "%", decimals: 1, reference: 6.5 },
];

const CHECKUP = /BMI\s+([\d.]+),\s*systolic BP\s+(\d+),\s*HbA1c\s+([\d.]+)/i;

/**
 * Observations for a doctor's view. The doctor endpoints expose check-ups through the
 * timeline summary only (`/me/observations` is patient-scoped), so the three numbers are
 * read back out of that sentence. Anything that does not match is skipped rather than guessed.
 */
export function observationsFromTimeline(events: TimelineEvent[]): Observation[] {
  return events
    .filter((e) => e.type === "observation")
    .map((e) => {
      const found = CHECKUP.exec(e.summary);
      if (!found) return null;
      return {
        date: e.date,
        bmi: Number(found[1]),
        systolic_bp: Number(found[2]),
        hba1c: Number(found[3]),
      };
    })
    .filter((o): o is Observation => o !== null)
    .sort((a, b) => a.date.localeCompare(b.date));
}

const STARTED = /^Started\s+(.+?)\s+\(([^)]+)\)\s*$/i;

/**
 * Medications for a doctor's view, read back out of the timeline's "Started X Y (frequency)"
 * sentence. Structured medication rows are only exposed to the patient themselves
 * (`/me/medications`); anything that does not match keeps the sentence as the name.
 */
export function medicationsFromTimeline(events: TimelineEvent[]): Medication[] {
  return events
    .filter((e) => e.type === "medication")
    .map((e) => {
      const found = STARTED.exec(e.summary);
      if (!found) return { name: e.summary, dose: "", frequency: "", started: e.date };
      const parts = found[1].split(" ");
      const dose = parts.length > 1 ? parts.slice(-2).join(" ") : "";
      const name = parts.length > 1 ? parts.slice(0, -2).join(" ") || found[1] : found[1];
      return { name, dose, frequency: found[2], started: e.date };
    })
    .sort((a, b) => a.started.localeCompare(b.started));
}

/** "2024-04-15" -> "Q2 '24" */
export function quarterLabel(date: string): string {
  const [year, month] = date.split("-");
  const quarter = Math.floor((Number(month) - 1) / 3) + 1;
  return `Q${quarter} '${year.slice(2)}`;
}

export function metricValue(observation: Observation, key: Metric["key"]): number {
  return observation[key];
}

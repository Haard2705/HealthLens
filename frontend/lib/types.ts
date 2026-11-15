/** Response shapes from the HealthLens API (`backend/openapi.json`). */

export type Role = "admin" | "doctor" | "patient";

export interface User {
  id: string;
  email: string;
  display_name: string;
  role: Role;
  active: boolean;
  patient_id: string | null;
}

export interface Condition {
  id: string;
  display_name: string;
  plain_name: string;
  body_system: string;
  tier: number;
  description: string;
  sensitive: boolean;
  /** null until the release gate has run over a trained model. */
  prediction_enabled: boolean | null;
  signal_strength: "strong" | "moderate" | "weak" | null;
  reason: string | null;
}

export interface PrevalenceGroup {
  group: string;
  prevalence: number;
  n: number;
}

export interface Prevalence {
  condition: string;
  by: string;
  /** "brfss" for real survey numbers, "sample" for the illustrative placeholder. */
  source: string;
  note?: string;
  groups: PrevalenceGroup[];
}

/** One model's metrics at a cut-off, as written by `ml/src/evaluate.py`. */
export interface MetricSet {
  roc_auc?: number;
  pr_auc?: number;
  brier?: number;
  brier_no_skill?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  accuracy?: number;
  threshold?: number;
}

export interface ModelSummary {
  model?: string;
  metrics?: {
    default?: MetricSet;
    recall_oriented?: MetricSet;
    target_recall?: number;
  };
}

export interface ModelMetrics {
  condition: string;
  best_model?: string;
  selection?: { criterion?: string; validation_roc_auc?: Record<string, number> };
  models?: Record<string, ModelSummary>;
}

export interface ModelsResponse {
  condition: string;
  metrics: ModelMetrics;
}

export interface PatientSummary {
  id: string;
  name: string;
  age: number;
  sex: string;
  conditions: string[];
  assigned_doctor: string;
  is_synthetic: boolean;
}

export interface PatientRecord extends PatientSummary {
  bmi: number;
  smoking_status: string;
  physically_active: boolean;
  general_health: number;
}

export interface Medication {
  name: string;
  dose: string;
  frequency: string;
  started: string;
}

export interface Observation {
  date: string;
  bmi: number;
  systolic_bp: number;
  hba1c: number;
}

export interface TimelineEvent {
  date: string;
  type: string;
  summary: string;
}

export interface AuditEvent {
  id: number;
  timestamp: string;
  actor_id: string | null;
  actor_role: string | null;
  action: string;
  resource_type: string;
  resource_id: string | null;
  patient_id: string | null;
  outcome: string;
  request_id: string;
}

export interface SystemInfo {
  data_source: string;
  patient_count: number;
  user_count: number;
  conditions_total: number;
  conditions_prediction_enabled: number;
  environment: string;
  demo_domain: string;
}

export interface Items<T> {
  items: T[];
}

/** `POST /predict` and `GET /patients/{id}/risk-context` are still being built (H22). */
export interface RiskContribution {
  feature: string;
  value?: string | number | null;
  contribution: number;
}

export interface RiskContext {
  condition?: string;
  patient_id?: string;
  probability?: number;
  baseline?: number;
  signal_strength?: string | null;
  prediction_enabled?: boolean | null;
  reason?: string | null;
  explanation?: string | null;
  contributions?: RiskContribution[];
}

"use client";

import { useCallback, useState } from "react";
import { ApiError, apiGet } from "@/lib/api";
import { useResource } from "@/lib/useResource";
import { ErrorNote, Loading } from "@/components/States";
import { GATE_MIN_AUC, clamp01, humanise } from "@/lib/format";
import type {
  Condition,
  Items,
  MetricSet,
  ModelMetrics,
  ModelsResponse,
} from "@/lib/types";

interface Row {
  condition: Condition;
  /** null when no trained model is on disk for this condition yet. */
  metrics: ModelMetrics | null;
  best: MetricSet | null;
  atThreshold: MetricSet | null;
  bestModel: string | null;
}

const MODEL_LABEL: Record<string, string> = {
  logreg: "Logistic regression",
  lr: "Logistic regression",
  xgboost: "XGBoost",
  xgb: "XGBoost",
};

function modelLabel(name: string | null): string {
  if (!name) return "—";
  return MODEL_LABEL[name] ?? humanise(name);
}

async function loadRows(signal: AbortSignal): Promise<Row[]> {
  const conditions = await apiGet<Items<Condition>>("/conditions", undefined, signal);
  return Promise.all(
    conditions.items.map(async (condition) => {
      try {
        const response = await apiGet<ModelsResponse>(
          "/models",
          { condition: condition.id },
          signal,
        );
        const metrics = response.metrics;
        const bestModel = metrics.best_model ?? null;
        const summary = bestModel ? metrics.models?.[bestModel] : undefined;
        return {
          condition,
          metrics,
          best: summary?.metrics?.default ?? null,
          atThreshold: summary?.metrics?.recall_oriented ?? null,
          bestModel,
        };
      } catch (error: unknown) {
        // "Not trained yet" is an expected state, not a page failure.
        if (error instanceof ApiError && error.isNotFound) {
          return { condition, metrics: null, best: null, atThreshold: null, bestModel: null };
        }
        throw error;
      }
    }),
  );
}

/** The ROC-AUC bar: starts at 0.50 (a coin flip), with a tick at the gate minimum. */
function AucBar({ auc, enabled }: { auc: number | undefined; enabled: boolean }) {
  if (auc === undefined) {
    return <span className="muted small">no model yet</span>;
  }
  const width = clamp01((auc - 0.5) / 0.5) * 100;
  const tick = ((GATE_MIN_AUC - 0.5) / 0.5) * 100;
  return (
    <div className="ab">
      <div className="t">
        <div className={`f${enabled ? "" : " off"}`} style={{ width: `${width.toFixed(1)}%` }} />
        <div className="gt" style={{ left: `${tick.toFixed(1)}%` }} />
      </div>
      <span className="v">{auc.toFixed(2)}</span>
    </div>
  );
}

function shortWhy(row: Row): string {
  const { condition } = row;
  if (condition.sensitive) return "Sensitive condition";
  if (row.metrics === null) return "Not trained in this build";
  const auc = row.best?.roc_auc;
  if (auc !== undefined && auc < GATE_MIN_AUC) return `Score below ${GATE_MIN_AUC.toFixed(2)}`;
  if (condition.prediction_enabled === true) return "Clears all three checks";
  if (condition.prediction_enabled === false) return "Held back by the gate";
  return "Gate has not run";
}

function Practice({ row }: { row: Row }) {
  const { condition, best, atThreshold } = row;
  const enabled = condition.prediction_enabled === true;
  const caught = atThreshold?.recall;
  const right = atThreshold?.precision;
  const brier = best?.brier;
  const noSkill = best?.brier_no_skill;

  const cards: { value: string; label: string }[] = [
    { value: modelLabel(row.bestModel), label: "model chosen on validation data" },
  ];
  if (caught !== undefined) {
    cards.push({
      value: `${Math.round(caught * 100)} in 100`,
      label: "real cases the model catches",
    });
  }
  if (right !== undefined) {
    cards.push({
      value: `${Math.round(right * 100)} in 100`,
      label: "of those it flags really have it",
    });
  }
  if (brier !== undefined && noSkill !== undefined) {
    cards.push({
      value: `${brier.toFixed(3)} vs ${noSkill.toFixed(3)}`,
      label: "Brier score vs guessing the average",
    });
  }

  return (
    <div className="prac">
      <div className="pl">
        {cards.map((card) => (
          <div className="pc" key={card.label}>
            <span className="v">{card.value}</span>
            <span className="l">{card.label}</span>
          </div>
        ))}
      </div>
      <p className="small">
        {row.metrics === null ? (
          <>
            No model has been trained for {condition.display_name.toLowerCase()} in this build, so
            there is nothing to score. Run the training pipeline and this row fills in by itself.
          </>
        ) : enabled && caught !== undefined && right !== undefined ? (
          <>
            Tuned to catch most cases, it finds about{" "}
            <strong>{Math.round(caught * 100)} in every 100</strong> people who really have{" "}
            {condition.plain_name}. But of every 100 people it flags, only about{" "}
            <strong>{Math.round(right * 100)}</strong> actually have it, so the rest are false
            alarms. That is why it is a prompt to look closer, never a diagnosis.
          </>
        ) : (
          <>
            {condition.reason ??
              "This model is not used for personal estimates, so only population patterns are shown."}
          </>
        )}
      </p>
    </div>
  );
}

export function ModelResults() {
  const load = useCallback((signal: AbortSignal) => loadRows(signal), []);
  const { data, loading, error, reload } = useResource(load);
  const [open, setOpen] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "on" | "off">("all");

  if (loading) return <Loading what="the model results" />;
  if (error || !data) {
    return <ErrorNote message={error ?? "No model results came back."} onRetry={reload} />;
  }

  const total = data.length;
  const enabledCount = data.filter((r) => r.condition.prediction_enabled === true).length;
  const gateHasRun = data.some((r) => r.condition.prediction_enabled !== null);
  const rows = data
    .filter((row) =>
      filter === "all"
        ? true
        : filter === "on"
          ? row.condition.prediction_enabled === true
          : row.condition.prediction_enabled !== true,
    )
    .sort((a, b) => (b.best?.roc_auc ?? -1) - (a.best?.roc_auc ?? -1));

  return (
    <div className="stack">
      <div>
        <h1 className="pt">Model results</h1>
        <p className="muted">
          One model per condition, scored on people it never saw. A written rule decides which may
          show a personal estimate.
        </p>
      </div>

      <div className="grid">
        <div className="card kpi">
          <span className="small muted">May show individual estimates</span>
          <span className="v">{gateHasRun ? `${enabledCount} of ${total}` : `0 of ${total}`}</span>
        </div>
        <div className="card kpi">
          <span className="small muted">The rule</span>
          <span className="small" style={{ marginTop: 4 }}>
            ROC-AUC at least {GATE_MIN_AUC.toFixed(2)}, probabilities better than guessing the
            average, and not a sensitive condition.
          </span>
        </div>
      </div>

      {!gateHasRun && (
        <div className="alert warn small">
          No trained models are on disk in this build, so there are no scores to show yet. The gate
          rule, the conditions and the shape of this page are real: train the models
          (<span className="mono">python -m ml.src.train</span>) and every row fills in.
        </div>
      )}

      <div className="card stack" style={{ gap: 12 }}>
        <div className="row between">
          <h2 style={{ fontSize: 16 }}>How well each model ranks people</h2>
          <div className="seg" role="group" aria-label="Filter conditions">
            {(
              [
                ["all", `All ${total}`],
                ["on", "Personal estimate on"],
                ["off", "Analytics only"],
              ] as const
            ).map(([key, label]) => (
              <button
                type="button"
                key={key}
                aria-pressed={filter === key}
                onClick={() => setFilter(key)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        <p className="small muted">
          The bar is the ROC-AUC: the chance the model puts someone who has the condition above
          someone who does not. It starts at 0.50, a coin flip. The black tick is the{" "}
          {GATE_MIN_AUC.toFixed(2)} minimum for showing a personal estimate.{" "}
          <strong>Select any row</strong> to see what the score means in practice.
        </p>

        <div className="legend">
          <span className="k">
            <span className="sw" />
            personal estimate on
          </span>
          <span className="k">
            <span className="sw off" />
            analytics only
          </span>
          <span className="k">
            <span className="sw tick" />
            gate minimum {GATE_MIN_AUC.toFixed(2)}
          </span>
        </div>

        <div className="tw">
          <table className="mt">
            <thead>
              <tr>
                <th aria-hidden="true" />
                <th>Condition</th>
                <th>Ranks people well</th>
                <th>Signal</th>
                <th>Personal estimate</th>
                <th>Why</th>
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 && (
                <tr>
                  <td colSpan={6} className="muted">
                    No conditions match this filter.
                  </td>
                </tr>
              )}
              {rows.map((row) => {
                const isOpen = open === row.condition.id;
                const enabled = row.condition.prediction_enabled === true;
                return [
                  <tr
                    className="r"
                    key={row.condition.id}
                    tabIndex={0}
                    role="button"
                    aria-expanded={isOpen}
                    onClick={() => setOpen(isOpen ? null : row.condition.id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        setOpen(isOpen ? null : row.condition.id);
                      }
                    }}
                  >
                    <td className="exp" aria-hidden="true">
                      {isOpen ? "−" : "+"}
                    </td>
                    <td>
                      <strong>{row.condition.display_name}</strong>
                      <div className="small muted">{humanise(row.condition.body_system)}</div>
                    </td>
                    <td>
                      <AucBar auc={row.best?.roc_auc} enabled={enabled} />
                    </td>
                    <td className="small muted">{row.condition.signal_strength ?? "—"}</td>
                    <td>
                      {enabled ? (
                        <span className="pill ok">on</span>
                      ) : (
                        <span className="pill">analytics only</span>
                      )}
                    </td>
                    <td className="small muted">{shortWhy(row)}</td>
                  </tr>,
                  isOpen ? (
                    <tr className="det" key={`${row.condition.id}-detail`}>
                      <td />
                      <td colSpan={5}>
                        <Practice row={row} />
                      </td>
                    </tr>
                  ) : null,
                ];
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card stack" style={{ gap: 8 }}>
        <h2 style={{ fontSize: 16 }}>What the gate checks, in order</h2>
        <ul className="small muted">
          <li>
            The condition is not marked sensitive in the registry. Sensitive ones stay analytics
            only whatever their score.
          </li>
          <li>A model was trained successfully for it.</li>
          <li>
            Held-out ROC-AUC is at least {GATE_MIN_AUC.toFixed(2)}, so it ranks people better than
            a coin flip by a useful margin.
          </li>
          <li>
            Its Brier score beats the no-skill Brier, so its percentages are better than always
            guessing the overall rate.
          </li>
        </ul>
        <p className="small muted">
          The frontend only reads the gate&apos;s answer; it never re-implements the rule. Signal
          words follow the same thresholds: strong from 0.80, moderate from{" "}
          {GATE_MIN_AUC.toFixed(2)}, weak below it.
        </p>
      </div>

      <p className="small muted">
        Scores are measured once, on a held-out test split the models never saw during training or
        selection. The model is chosen on a separate validation split, so the final number is
        honest.{" "}
        {enabledCount > 0 &&
          `${enabledCount} of ${total} conditions currently pass the gate; the rest show population analytics only.`}
      </p>
    </div>
  );
}

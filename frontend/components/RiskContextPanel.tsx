"use client";

import { useCallback, useState } from "react";
import { ErrorNote, Loading, RiskUnavailable } from "./States";
import { apiGet } from "@/lib/api";
import { useResource } from "@/lib/useResource";
import { clamp01, humanise, pct } from "@/lib/format";
import type { Condition, RiskContext, RiskContribution } from "@/lib/types";
import type { RecordMode } from "./RecordDetail";

interface RiskContextPanelProps {
  patientId: string;
  conditions: Condition[];
  /** Condition ids already on the chart: we estimate risk, we do not re-diagnose. */
  existingConditions: string[];
  mode: RecordMode;
}

function Contributions({ items }: { items: RiskContribution[] }) {
  const max = Math.max(...items.map((item) => Math.abs(item.contribution)), 0.0001);
  return (
    <div>
      {items.map((item) => {
        const raises = item.contribution > 0;
        const width = (Math.abs(item.contribution) / max) * 50;
        return (
          <div className="shap" key={`${item.feature}-${item.contribution}`}>
            <span>
              {humanise(item.feature)}{" "}
              {item.value !== undefined && item.value !== null && (
                <span className="muted small">{String(item.value)}</span>
              )}
            </span>
            <div className="t">
              <div
                className="f"
                style={{
                  background: raises ? "var(--up)" : "var(--down)",
                  [raises ? "left" : "right"]: "50%",
                  width: `${width.toFixed(1)}%`,
                }}
              />
            </div>
            <span className={`num small ${raises ? "up" : "down"}`}>
              {raises ? "+" : ""}
              {item.contribution.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/**
 * The per-patient risk panel. The estimate endpoint (H22) is built on another branch, so this
 * tries it and falls back to a tidy explanation rather than failing the whole page.
 */
export function RiskContextPanel({
  patientId,
  conditions,
  existingConditions,
  mode,
}: RiskContextPanelProps) {
  const [conditionId, setConditionId] = useState(conditions[0]?.id ?? "diabetes");
  const condition = conditions.find((c) => c.id === conditionId);
  const alreadyOnChart = existingConditions.includes(conditionId);
  const gateBlocked = condition?.prediction_enabled === false;
  const skipFetch = alreadyOnChart || gateBlocked;

  const load = useCallback(
    (signal: AbortSignal) =>
      skipFetch
        ? Promise.resolve<RiskContext>({})
        : apiGet<RiskContext>(
            `/patients/${encodeURIComponent(patientId)}/risk-context`,
            { condition: conditionId },
            signal,
          ),
    [patientId, conditionId, skipFetch],
  );
  const risk = useResource(load);

  const limits = (
    <div className="alert small">
      <strong>Limits of any estimate here.</strong> It comes from self-reported survey answers,
      shows patterns rather than causes, and describes groups of people rather than what will
      happen to one person. It is not a diagnosis.
    </div>
  );

  const picker = (
    <div className="row">
      <label className="f">
        Condition
        <select value={conditionId} onChange={(event) => setConditionId(event.target.value)}>
          {conditions.map((item) => (
            <option value={item.id} key={item.id}>
              {item.display_name}
            </option>
          ))}
        </select>
      </label>
    </div>
  );

  return (
    <div className="stack">
      {picker}

      {condition && alreadyOnChart && (
        <div className="card">
          <h3>{condition.display_name} is already on the chart</h3>
          <p className="muted small" style={{ marginTop: 6 }}>
            HealthLens estimates risk for conditions someone does not already have. It does not
            re-diagnose what a clinician has recorded.
          </p>
        </div>
      )}

      {/* The release gate's answer lives on /conditions, so it shows with no estimate endpoint. */}
      {condition && gateBlocked && !alreadyOnChart && (
        <div className="card">
          <div className="row between">
            <h3>{condition.display_name}: population analytics only</h3>
            {condition.signal_strength && (
              <span className="pill">{condition.signal_strength} signal</span>
            )}
          </div>
          <p className="muted small" style={{ marginTop: 6 }}>
            {condition.reason ??
              "The release gate did not enable individual estimates for this condition."}
          </p>
          <p className="small" style={{ marginTop: 6 }}>
            You can still see how common it is on the Explore page.
          </p>
        </div>
      )}

      {!skipFetch && (
        <>
          {risk.loading && <Loading what="the risk context" />}
          {risk.cause?.isUnimplemented && (
            <RiskUnavailable
              detail={
                mode === "patient"
                  ? "This build does not calculate a personal estimate yet. When it does, it will arrive with a plain-language reason for every number."
                  : "This build has no personal-risk endpoint yet. The population rates for every condition are on Explore, and each model's held-out scores are on Model results."
              }
            />
          )}
          {risk.error && !risk.cause?.isUnimplemented && (
            <ErrorNote message={risk.error} onRetry={risk.reload} />
          )}

          {risk.data && (
            <div className="grid">
              <div className="card">
                <div className="row between">
                  <span className="small muted">
                    {mode === "patient" ? "Estimated chance" : "Population-level risk"}
                  </span>
                  {risk.data.signal_strength && (
                    <span className="pill acc">{risk.data.signal_strength} signal</span>
                  )}
                </div>
                <div className="kpi">
                  <span className="v" style={{ fontSize: 40 }}>
                    {risk.data.probability === undefined
                      ? "—"
                      : pct(clamp01(risk.data.probability), 0)}
                  </span>
                </div>
                {risk.data.probability !== undefined && (
                  <>
                    <div className="meter" style={{ margin: "12px 0 6px" }}>
                      <div
                        className="f"
                        style={{ width: `${clamp01(risk.data.probability) * 100}%` }}
                      />
                      {risk.data.baseline !== undefined && (
                        <div
                          className="tick"
                          style={{ left: `${clamp01(risk.data.baseline) * 100}%` }}
                          title="Typical adult"
                        />
                      )}
                    </div>
                    {risk.data.baseline !== undefined && (
                      <p className="small muted">
                        Black tick: a typical adult ({pct(clamp01(risk.data.baseline), 0)}).
                      </p>
                    )}
                  </>
                )}
              </div>

              {risk.data.contributions && risk.data.contributions.length > 0 && (
                <div className="card">
                  <h3>What moved this estimate</h3>
                  <p className="small muted" style={{ marginBottom: 8 }}>
                    Vermilion raises it, blue lowers it. Sign and length both carry the meaning, so
                    the chart does not rely on colour alone.
                  </p>
                  <Contributions items={risk.data.contributions} />
                </div>
              )}
            </div>
          )}

          {risk.data?.explanation && (
            <div className="card">
              <h3>{mode === "patient" ? "In plain words" : "Plain-language summary"}</h3>
              <p style={{ marginTop: 6 }}>{risk.data.explanation}</p>
            </div>
          )}
        </>
      )}

      {limits}
    </div>
  );
}

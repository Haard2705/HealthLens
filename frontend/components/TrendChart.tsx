import type { Metric } from "@/lib/format";

export interface TrendPoint {
  label: string;
  value: number;
}

interface TrendChartProps {
  points: TrendPoint[];
  metric: Metric;
}

const W = 640;
const H = 230;
const PAD = { left: 46, right: 14, top: 14, bottom: 30 };

/** A small inline-SVG line chart. No chart library: the shape is simple and stays readable. */
export function TrendChart({ points, metric }: TrendChartProps) {
  if (points.length < 2) {
    return <p className="muted small">Not enough check-ups yet to draw a trend.</p>;
  }

  const values = points.map((p) => p.value);
  const withRef = [...values, metric.reference];
  let low = Math.min(...withRef);
  let high = Math.max(...withRef);
  const span = high - low || 1;
  low -= span * 0.18;
  high += span * 0.18;

  const x = (index: number) =>
    PAD.left + ((W - PAD.left - PAD.right) * index) / (points.length - 1);
  const y = (value: number) =>
    PAD.top + (H - PAD.top - PAD.bottom) * (1 - (value - low) / (high - low));

  const gridLines = [0, 1, 2, 3].map((i) => low + ((high - low) * i) / 3);
  const path = points.map((p, i) => `${x(i)},${y(p.value)}`).join(" ");
  const last = points.length - 1;
  const title = `${metric.label} over ${points.length} check-ups, latest ${points[last].value.toFixed(metric.decimals)} ${metric.unit}`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={title}>
      {gridLines.map((value) => (
        <g key={value}>
          <line className="gl" x1={PAD.left} x2={W - PAD.right} y1={y(value)} y2={y(value)} />
          <text x={PAD.left - 6} y={y(value) + 4} textAnchor="end">
            {value.toFixed(metric.decimals)}
          </text>
        </g>
      ))}

      <line
        x1={PAD.left}
        x2={W - PAD.right}
        y1={y(metric.reference)}
        y2={y(metric.reference)}
        stroke="var(--up)"
        strokeDasharray="4 4"
        strokeWidth="1"
      />
      <text
        x={W - PAD.right}
        y={y(metric.reference) - 5}
        textAnchor="end"
        style={{ fill: "var(--up)" }}
      >
        reference {metric.reference}
      </text>

      <polygon
        points={`${PAD.left},${H - PAD.bottom} ${path} ${x(last)},${H - PAD.bottom}`}
        fill="var(--accent)"
        opacity="0.1"
      />
      <polyline
        points={path}
        fill="none"
        stroke="var(--accent)"
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />

      {points.map((point, i) =>
        i % 4 === 0 || i === last ? (
          <text
            key={point.label + i}
            x={x(i)}
            y={H - 8}
            textAnchor={i === 0 ? "start" : i === last ? "end" : "middle"}
          >
            {point.label}
          </text>
        ) : null,
      )}

      <circle
        cx={x(last)}
        cy={y(points[last].value)}
        r="5"
        fill="var(--accent)"
        stroke="var(--surface)"
        strokeWidth="2"
      />
      <text
        x={x(last) - 9}
        y={y(points[last].value) - 9}
        textAnchor="end"
        style={{ fill: "var(--ink)" }}
      >
        {points[last].value.toFixed(metric.decimals)}
      </text>
    </svg>
  );
}

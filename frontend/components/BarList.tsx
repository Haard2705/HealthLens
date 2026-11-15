import { MIN_GROUP_SIZE, count } from "@/lib/format";

export interface BarRow {
  label: string;
  /** Omitted when the group was suppressed for being too small. */
  value?: number;
  n?: number;
  suppressed?: boolean;
}

interface BarListProps {
  rows: BarRow[];
  format: (value: number) => string;
}

/**
 * Horizontal bars drawn with divs: all bars share one scale, so lengths are comparable.
 * Suppressed groups keep their row, hatched, rather than disappearing without a word.
 */
export function BarList({ rows, format }: BarListProps) {
  const max = Math.max(...rows.map((r) => (r.suppressed ? 0 : (r.value ?? 0))), 0.0001);

  return (
    <div>
      {rows.map((row) =>
        row.suppressed ? (
          <div className="bar sup" key={row.label}>
            <span>{row.label}</span>
            <div className="t" title={`Hidden: fewer than ${MIN_GROUP_SIZE} respondents`} />
            <span className="small muted num">hidden</span>
          </div>
        ) : (
          <div className="bar" key={row.label}>
            <span>
              {row.label}
              {row.n !== undefined && <span className="n"> n={count(row.n)}</span>}
            </span>
            <div className="t">
              <div
                className="f"
                style={{ width: `${(((row.value ?? 0) / max) * 100).toFixed(1)}%` }}
              />
            </div>
            <span className="num">{format(row.value ?? 0)}</span>
          </div>
        ),
      )}
    </div>
  );
}

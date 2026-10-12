import { ArrowUpRight, History as HistoryIcon, Play } from "lucide-react";
import type { Forecast } from "../types";
import { money, shortDate } from "../api";
import { Empty, Notice, PanelHeading } from "./Common";
export function History({
  forecasts,
  onReplay,
  compact = false,
}: {
  forecasts: Forecast[];
  onReplay: (f: Forecast) => void;
  compact?: boolean;
}) {
  return (
    <section className="panel history-panel">
      <PanelHeading
        title={compact ? "Recent predictions" : "Prediction archive"}
        action={
          <span className="tag">
            <HistoryIcon size={12} />
            {forecasts.length} recorded
          </span>
        }
      />
      {!compact && (
        <Notice>
          Immutable historical replays preserve the model version, input
          snapshot, and recorded computation. The issue time is when this replay
          was generated; target sessions are historical.
        </Notice>
      )}
      {forecasts.length ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Target session</th>
                <th>Model</th>
                <th>Median</th>
                <th>Observed</th>
                {!compact && (
                  <>
                    <th>Input cutoff</th>
                    <th>Issued (UTC)</th>
                  </>
                )}
                <th aria-label="Prediction actions" />
              </tr>
            </thead>
            <tbody>
              {forecasts.slice(0, compact ? 5 : 100).map((f) => (
                <tr key={f.id}>
                  <td>
                    <span className="history-symbol">{f.symbol}</span>
                    <span className="muted">{shortDate(f.target_time)}</span>
                  </td>
                  <td>
                    <span className="tag violet">
                      {f.model_kind.toUpperCase()}
                    </span>
                  </td>
                  <td>{money(f.median)}</td>
                  <td className="amber-text">{money(f.actual)}</td>
                  {!compact && (
                    <>
                      <td>{f.input_cutoff}</td>
                      <td className="timestamp-cell">
                        {new Date(f.issue_time)
                          .toISOString()
                          .replace("T", " ")
                          .slice(0, 19)}
                      </td>
                    </>
                  )}
                  <td>
                    <button
                      className="text-button"
                      aria-label={`Replay prediction for ${f.target_time}`}
                      onClick={() => onReplay(f)}
                    >
                      <Play size={12} />
                      {compact ? <ArrowUpRight size={14} /> : "Replay"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty title="Your research, recorded">
          Generated forecasts are saved here so you can revisit every
          computation.
        </Empty>
      )}
    </section>
  );
}

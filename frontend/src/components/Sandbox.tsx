import { useState } from "react";
import { FlaskConical, ArrowRight, LoaderCircle } from "lucide-react";
import type { Forecast, Sensitivity } from "../types";
import { api, money } from "../api";
import { Empty, Notice, PanelHeading } from "./Common";
export function Sandbox({ forecast }: { forecast: Forecast | null }) {
  const [feature, setFeature] = useState("close"),
    [percent, setPercent] = useState(2),
    [result, setResult] = useState<Sensitivity | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  async function run() {
    if (!forecast) return;
    setBusy(true);
    setError("");
    try {
      setResult(
        await api<Sensitivity>("/what-if", {
          forecast_id: forecast.id,
          feature,
          percent,
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="panel sandbox">
      <PanelHeading
        title="What-if sandbox"
        action={<FlaskConical size={17} className="violet-text" />}
      />
      <p className="section-intro">
        Change one input. Discover how the model responds.
      </p>
      {forecast ? (
        <>
          <div className="sandbox-controls">
            <label>
              Last input observation
              <select
                aria-label="What-if feature"
                value={feature}
                onChange={(e) => {
                  setFeature(e.target.value);
                  setResult(null);
                }}
              >
                <option value="close">Closing price</option>
                <option value="volume">Trading volume</option>
              </select>
            </label>
            <div className="perturbation-value">
              {percent > 0 ? "+" : ""}
              {percent}%
            </div>
          </div>
          <input
            aria-label="Input perturbation percent"
            type="range"
            min="-20"
            max="20"
            step="1"
            value={percent}
            onChange={(e) => {
              setPercent(+e.target.value);
              setResult(null);
            }}
          />
          <div className="range-labels">
            <span>−20%</span>
            <span>No change</span>
            <span>+20%</span>
          </div>
          <button
            className="button secondary full"
            onClick={run}
            disabled={busy}
          >
            {busy ? (
              <LoaderCircle size={15} className="spin" />
            ) : (
              <FlaskConical size={15} />
            )}
            Run sensitivity analysis
            <ArrowRight size={15} />
          </button>
          {error && (
            <p role="alert" className="negative">
              {error}
            </p>
          )}
          {result && (
            <div className="sensitivity-results">
              <div>
                <span>Original median</span>
                <strong>{money(result.original.median)}</strong>
              </div>
              <ArrowRight size={18} />
              <div>
                <span>Perturbed median</span>
                <strong className="violet-text">
                  {money(result.perturbed.median)}
                </strong>
              </div>
              <details>
                <summary>Exact changed inputs</summary>
                <pre>{JSON.stringify(result.changes, null, 2)}</pre>
              </details>
            </div>
          )}
          <Notice>
            Model sensitivity, not a causal market prediction. OHLC
            relationships are preserved and derived features are recomputed.
          </Notice>
        </>
      ) : (
        <Empty title="Start with a forecast">
          A recorded prediction provides the original input snapshot for this
          experiment.
        </Empty>
      )}
    </section>
  );
}

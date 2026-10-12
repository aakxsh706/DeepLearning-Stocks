import { useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ArrowRight,
  ChevronDown,
  Cpu,
  Sparkles,
  LoaderCircle,
} from "lucide-react";
import type { Forecast, Model, Market } from "../types";
import { money, pct, shortDate } from "../api";
import { chartTooltip, Empty, PanelHeading } from "./Common";
export function ForecastPanel({
  forecast,
  models,
  modelId,
  setModelId,
  onGenerate,
  busy,
  market,
  onExplain,
}: {
  forecast: Forecast | null;
  models: Model[];
  modelId: string;
  setModelId: (id: string) => void;
  onGenerate: () => void;
  busy: boolean;
  market: Market | null;
  onExplain: () => void;
}) {
  const model = models.find((m) => m.id === modelId);
  const chartData = useMemo(() => {
    if (!forecast || !market) return [];
    const data: Record<string, unknown>[] = market.candles
      .filter((d) => d.time <= forecast.input_cutoff)
      .slice(-12)
      .map((d) => ({ date: d.time, observed: d.close }));
    if (data.length) data[data.length - 1].forecast = forecast.base_price;
    data.push({
      date: forecast.target_time,
      forecast: forecast.median,
      interval: [forecast.lower, forecast.upper],
      actual: forecast.actual ?? undefined,
    });
    return data;
  }, [forecast, market]);
  return (
    <section className="panel forecast-panel">
      <PanelHeading
        title="Forecast lab"
        action={
          <span className="tag violet">
            <Sparkles size={11} /> Next session
          </span>
        }
      />
      <label className="field-label" htmlFor="model-select">
        FORECAST MODEL
      </label>
      <div className="select-wrap">
        <Cpu size={16} />
        <select
          id="model-select"
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          disabled={!models.length}
        >
          {!models.length && <option value="">No trained models yet</option>}
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.kind.toUpperCase()} · {m.lookback} days · {m.id.slice(-7)}
            </option>
          ))}
        </select>
        <ChevronDown size={14} />
      </div>
      <div className="model-meta">
        <span>
          {model ? `${model.lookback} observations` : "Train a model to begin"}
        </span>
        <span>
          {model ? `${model.hidden_size} hidden units` : "LSTM / GRU"}
        </span>
      </div>
      {forecast ? (
        <>
          <div className="forecast-number">
            <span>Median estimate</span>
            <div>
              {money(forecast.median)}
              <small
                className={
                  forecast.median >= forecast.base_price
                    ? "positive"
                    : "negative"
                }
              >
                {forecast.returns.median >= 0 ? "+" : ""}
                {pct(forecast.returns.median)}
              </small>
            </div>
            <p>Target: {shortDate(forecast.target_time)}</p>
          </div>
          <div className="forecast-mini-chart">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={chartData}
                margin={{ top: 10, right: 4, left: 0, bottom: 0 }}
              >
                <defs>
                  <linearGradient
                    id="forecastShade"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop offset="0%" stopColor="#a897ff" stopOpacity={0.3} />
                    <stop
                      offset="100%"
                      stopColor="#a897ff"
                      stopOpacity={0.02}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid vertical={false} stroke="#20283b" />
                <XAxis dataKey="date" hide />
                <YAxis hide domain={["dataMin - 1", "dataMax + 1"]} />
                <Tooltip {...chartTooltip} />
                <Area
                  dataKey="interval"
                  name="Estimated interval"
                  fill="url(#forecastShade)"
                  stroke="#a897ff"
                />
                <Line
                  dataKey="observed"
                  name="Observed price"
                  stroke="#5ecabd"
                  dot={false}
                  strokeWidth={1.8}
                />
                <Line
                  dataKey="forecast"
                  name="Forecast median"
                  stroke="#ac9afe"
                  strokeDasharray="4 3"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
                <Line
                  dataKey="actual"
                  name="Actual outcome"
                  stroke="#e6b768"
                  dot={{ r: 4 }}
                />
                <ReferenceLine
                  x={forecast.input_cutoff}
                  stroke="#43506a"
                  strokeDasharray="3 3"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="forecast-range">
            <div>
              <span>Lower estimate</span>
              <strong>{money(forecast.lower)}</strong>
            </div>
            <div>
              <span>Upper estimate</span>
              <strong>{money(forecast.upper)}</strong>
            </div>
          </div>
          <div className="interval-note">
            Estimated interval · coverage measured in Evaluation
          </div>
        </>
      ) : (
        <Empty title="A forecast you can inspect">
          Generate a prediction from a trained model to see its estimates and
          recorded computations.
        </Empty>
      )}
      <button
        className="button primary full"
        onClick={onGenerate}
        disabled={busy || !modelId}
      >
        {busy ? (
          <LoaderCircle className="spin" size={15} />
        ) : (
          <Sparkles size={15} />
        )}{" "}
        {busy ? "Running inference…" : "Generate forecast"}
        {!busy && <ArrowRight size={16} />}
      </button>
      {forecast && (
        <button className="text-button explain-link" onClick={onExplain}>
          Explain this prediction <ArrowRight size={14} />
        </button>
      )}
      <details className="technical-details">
        <summary>Model & forecast provenance</summary>
        <dl>
          <dt>Model version</dt>
          <dd>{forecast?.model_version || model?.id || "—"}</dd>
          <dt>Training cutoff</dt>
          <dd>
            {shortDate(forecast?.training_cutoff || model?.training_cutoff)}
          </dd>
          <dt>Input cutoff</dt>
          <dd>{forecast?.input_cutoff || "—"}</dd>
          <dt>Issued at (UTC)</dt>
          <dd>
            {forecast ? new Date(forecast.issue_time).toISOString() : "—"}
          </dd>
          <dt>Data snapshot</dt>
          <dd>{forecast?.data_snapshot_id || "—"}</dd>
        </dl>
      </details>
    </section>
  );
}

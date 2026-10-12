import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  BarChart,
  Bar,
} from "recharts";
import { ArrowUpRight, Trophy } from "lucide-react";
import type { Evaluation, Model } from "../types";
import { api, money, pct } from "../api";
import {
  chartTooltip,
  Empty,
  Metric,
  Notice,
  PanelHeading,
  Spinner,
} from "./Common";
export function EvaluationCenter({
  models,
  modelId,
  onModelChange,
}: {
  models: Model[];
  modelId: string;
  onModelChange: (id: string) => void;
}) {
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null),
    [races, setRaces] = useState<Evaluation[]>([]),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(false);
  useEffect(() => {
    let cancelled = false;
    if (!modelId) {
      setEvaluation(null);
      return;
    }
    setLoading(true);
    setError("");
    api<Evaluation>(`/evaluations?model_id=${encodeURIComponent(modelId)}`)
      .then((data) => {
        if (!cancelled) setEvaluation(data);
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e.message);
          setEvaluation(null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [modelId]);
  useEffect(() => {
    let cancelled = false;
    Promise.allSettled(
      models.map((m) =>
        api<Evaluation>(`/evaluations?model_id=${encodeURIComponent(m.id)}`),
      ),
    ).then((results) => {
      if (!cancelled)
        setRaces(
          results.flatMap((r) => (r.status === "fulfilled" ? [r.value] : [])),
        );
    });
    return () => {
      cancelled = true;
    };
  }, [models]);
  const m = evaluation?.metrics;
  const rows =
    evaluation?.rows.map((r) => ({ ...r, interval: [r.lower, r.upper] })) || [];
  const samePeriod = races.filter(
    (r) =>
      r.rows.length === evaluation?.rows.length &&
      r.rows.every(
        (row, i) =>
          row.date === evaluation?.rows[i]?.date &&
          row.base_price === evaluation?.rows[i]?.base_price,
      ),
  );
  return (
    <div className="evaluation">
      <section className="panel evaluation-heading">
        <PanelHeading
          eyebrow="EVERY FORECAST, ACCOUNTABLE"
          title="Evaluation center"
          action={
            <select
              aria-label="Evaluation model"
              value={modelId}
              onChange={(e) => onModelChange(e.target.value)}
            >
              {!models.length && <option value="">No trained models</option>}
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.kind.toUpperCase()} · {model.id.slice(-7)}
                </option>
              ))}
            </select>
          }
        />
        <Notice>
          Historical test results only. Models are evaluated on observations
          held out from training and validation. No forward-running performance
          is claimed.
        </Notice>
      </section>
      {loading ? (
        <Spinner label="Evaluating held-out predictions…" />
      ) : error ? (
        <div className="error-banner" role="alert">
          {error}
        </div>
      ) : !evaluation ? (
        <section className="panel">
          <Empty title="Evidence before confidence">
            Train a model to evaluate its performance against a simple
            persistence baseline.
          </Empty>
        </section>
      ) : (
        <>
          <div className="evaluation-metrics">
            <Metric
              label="Mean absolute error"
              value={money(m?.mae)}
              detail={`Baseline ${money(evaluation.baseline.mae)}`}
            />
            <Metric
              label="Root mean squared error"
              value={money(m?.rmse)}
              detail={`Baseline ${money(evaluation.baseline.rmse)}`}
              color="violet"
            />
            <Metric
              label="Directional accuracy"
              value={pct(m?.directional_accuracy)}
              detail="Correct sign of next-session return"
              color="amber"
            />
            <Metric
              label="Interval coverage"
              value={pct(m?.coverage)}
              detail={`Mean width ${money(m?.interval_width)}`}
              color="cyan"
            />
          </div>
          <section className="panel">
            <PanelHeading
              title="Forecasts meet reality"
              action={
                <span className="tag">
                  Held-out test · {rows.length} sessions
                </span>
              }
            />
            <div className="legend evaluation-legend">
              <span>
                <i className="amber" />
                Actual outcome
              </span>
              <span>
                <i className="violet" />
                Forecast median
              </span>
              <span>
                <i className="band" />
                Lower–upper interval
              </span>
            </div>
            <div className="evaluation-chart">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart
                  data={rows}
                  margin={{ top: 10, right: 15, left: 0, bottom: 0 }}
                >
                  <CartesianGrid stroke="#202b3c" vertical={false} />
                  <XAxis
                    dataKey="date"
                    stroke="#79879d"
                    tickLine={false}
                    axisLine={false}
                    minTickGap={65}
                    tickFormatter={(s) => s.slice(5)}
                  />
                  <YAxis
                    stroke="#79879d"
                    tickLine={false}
                    axisLine={false}
                    domain={["auto", "auto"]}
                    width={55}
                  />
                  <Tooltip {...chartTooltip} />
                  <Area
                    dataKey="interval"
                    name="Forecast interval"
                    stroke="none"
                    fill="#8271bb"
                    fillOpacity={0.16}
                  />
                  <Line
                    dataKey="median"
                    name="Forecast median"
                    stroke="#a897ff"
                    dot={false}
                    strokeWidth={1.6}
                  />
                  <Line
                    dataKey="actual"
                    name="Actual close"
                    stroke="#e6b768"
                    dot={false}
                    strokeWidth={1.6}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </section>
          <div className="evaluation-bottom">
            <section className="panel">
              <PanelHeading
                title="Model race"
                action={<Trophy size={17} className="amber-text" />}
              />
              <p className="section-intro">
                Same observations. Same horizon. A fair comparison with
                yesterday’s close.
              </p>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>MAE ↓</th>
                      <th>RMSE ↓</th>
                      <th>vs. baseline</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>
                        <span className="model-pill">P</span>Persistence
                      </td>
                      <td>{money(evaluation.baseline.mae)}</td>
                      <td>{money(evaluation.baseline.rmse)}</td>
                      <td className="muted">Reference</td>
                    </tr>
                    {samePeriod.map((r) => {
                      const model = models.find((x) => x.id === r.model_id),
                        better = r.metrics.mae < r.baseline.mae;
                      return (
                        <tr key={r.model_id}>
                          <td>
                            <span className="model-pill violet">
                              {model?.kind?.[0].toUpperCase()}
                            </span>
                            {model?.kind.toUpperCase()}{" "}
                            <small className="muted">
                              {r.model_id.slice(-5)}
                            </small>
                          </td>
                          <td>{money(r.metrics.mae)}</td>
                          <td>{money(r.metrics.rmse)}</td>
                          <td className={better ? "positive" : "negative"}>
                            {better ? "Lower" : "Higher"} error{" "}
                            <ArrowUpRight size={12} />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="footnote">
                Only models with identical test dates and input base prices are
                compared. A more complex model may perform worse.
              </p>
            </section>
            <section className="panel">
              <PanelHeading
                title="Recent forecast errors"
                action={<span className="tag">USD</span>}
              />
              <div className="error-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={rows.slice(-40)}>
                    <CartesianGrid stroke="#202b3c" vertical={false} />
                    <XAxis
                      dataKey="date"
                      stroke="#79879d"
                      tickLine={false}
                      minTickGap={60}
                      tickFormatter={(s) => s.slice(5)}
                    />
                    <YAxis stroke="#79879d" width={40} />
                    <Tooltip {...chartTooltip} />
                    <ReferenceLine y={0} stroke="#60708a" />
                    <Bar
                      dataKey="error"
                      name="Forecast − actual"
                      fill="#a897ff"
                      radius={[2, 2, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>
          <section className="panel">
            <PanelHeading
              title="Walk-forward windows"
              action={<span className="tag">Chronological</span>}
            />
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Window</th>
                    <th>Test dates</th>
                    <th>Sessions</th>
                    <th>Model MAE</th>
                    <th>Baseline MAE</th>
                    <th>Coverage</th>
                  </tr>
                </thead>
                <tbody>
                  {evaluation.walk_forward.map((row, i) => {
                    const metrics = row.metrics as Evaluation["metrics"];
                    const baseline = row.baseline as Evaluation["baseline"];
                    return (
                      <tr key={i}>
                        <td>{String(row.fold ?? i + 1)}</td>
                        <td>
                          {String(row.start)} → {String(row.end)}
                        </td>
                        <td>{String(row.count)}</td>
                        <td>{money(metrics?.mae)}</td>
                        <td>{money(baseline?.mae)}</td>
                        <td>{pct(metrics?.coverage)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="footnote">
              Sequential held-out windows use the same frozen checkpoint; no
              model refit occurs between windows.
            </p>
            <div className="split-grid">
              {Object.entries(
                models.find((model) => model.id === modelId)?.splits || {},
              ).map(([split, dates]) => (
                <div key={split}>
                  <strong>{split}</strong>
                  <span>
                    {dates.start} → {dates.end}
                  </span>
                  <small>{dates.count} windows</small>
                </div>
              ))}
            </div>
            <details className="technical-details">
              <summary>Metric definitions & split methodology</summary>
              <dl>
                {Object.entries(evaluation.definitions).map(([key, value]) => (
                  <div key={key}>
                    <dt>{key.replaceAll("_", " ")}</dt>
                    <dd>
                      {typeof value === "string"
                        ? value
                        : JSON.stringify(value)}
                    </dd>
                  </div>
                ))}
              </dl>
              <p>
                Directional accuracy compares the sign of forecast median minus
                last close with the sign of actual close minus last close. Exact
                zero is a separate direction. Coverage is the proportion of
                actual outcomes inside the lower–upper interval, inclusive.
              </p>
            </details>
          </section>
        </>
      )}
    </div>
  );
}

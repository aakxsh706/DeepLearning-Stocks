import { useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  BarChart,
  Bar,
} from "recharts";
import {
  FlaskConical,
  Play,
  CheckCircle2,
  LoaderCircle,
  Timer,
  Cpu,
} from "lucide-react";
import type { Job } from "../types";
import { api, num } from "../api";
import { chartTooltip, Empty, Metric, Notice, PanelHeading } from "./Common";
export function TrainingStudio({
  symbol,
  jobs,
  onRefresh,
  onError,
}: {
  symbol: string;
  jobs: Job[];
  onRefresh: () => void;
  onError: (s: string) => void;
}) {
  const [kind, setKind] = useState("lstm"),
    [lookback, setLookback] = useState(30),
    [epochs, setEpochs] = useState(8),
    [samples, setSamples] = useState(1200),
    [busy, setBusy] = useState(false),
    [selected, setSelected] = useState("");
  const job = jobs.find((j) => j.id === selected) || jobs[0];
  const active = jobs.some(
    (j) => j.status === "running" || j.status === "queued",
  );
  const p = job?.progress;
  const progress =
    job?.status === "completed"
      ? 100
      : p
        ? Math.round((p.epoch / (p.epochs || job.config.epochs)) * 100)
        : 0;
  async function train() {
    setBusy(true);
    try {
      const result = await api<Job>("/training", {
        symbol,
        kind,
        lookback,
        epochs,
        max_samples: samples,
      });
      setSelected(result.id);
      onRefresh();
    } catch (e) {
      onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const histogram = useMemo(() => {
    const value = p?.weight_histogram as
      | { bins?: number[]; counts?: number[] }
      | undefined;
    return (
      value?.counts?.map((count, i) => ({
        weight: num(value.bins?.[i], 2),
        count,
      })) || []
    );
  }, [p]);
  return (
    <div className="studio-grid">
      <section className="panel training-config">
        <PanelHeading eyebrow="LEARN BY TRAINING" title="Training studio" />
        <p className="section-intro">
          Train a real recurrent model on chronological market observations.
          Watch its learning unfold.
        </p>
        <div className="form-grid">
          <label>
            Instrument
            <input value={symbol} disabled />
          </label>
          <label>
            Architecture
            <select
              aria-label="Training architecture"
              value={kind}
              onChange={(e) => setKind(e.target.value)}
            >
              <option value="lstm">LSTM</option>
              <option value="gru">GRU</option>
            </select>
          </label>
          <label>
            Lookback observations
            <input
              aria-label="Training lookback"
              type="number"
              min="10"
              max="60"
              value={lookback}
              onChange={(e) => setLookback(+e.target.value)}
            />
          </label>
          <label>
            Maximum epochs
            <input
              aria-label="Training epochs"
              type="number"
              min="1"
              max="20"
              value={epochs}
              onChange={(e) => setEpochs(+e.target.value)}
            />
          </label>
          <label className="wide">
            Maximum source observations
            <input
              aria-label="Training samples"
              type="number"
              min="200"
              max="2000"
              step="100"
              value={samples}
              onChange={(e) => setSamples(+e.target.value)}
            />
          </label>
        </div>
        <button
          className="button primary full"
          disabled={
            busy ||
            active ||
            lookback < 10 ||
            lookback > 60 ||
            epochs < 1 ||
            epochs > 20 ||
            samples < 200 ||
            samples > 2000
          }
          onClick={train}
        >
          {busy || active ? (
            <LoaderCircle size={16} className="spin" />
          ) : (
            <Play size={16} />
          )}{" "}
          {active ? "Training job in progress" : "Train a demo model"}
        </button>
        <Notice>
          One job at a time, up to 20 epochs. A separate worker keeps inference
          and the interface responsive.
        </Notice>
        <div className="training-principles">
          <h3>Built for honest evaluation</h3>
          <p>
            <CheckCircle2 size={14} /> Chronological train / validation / test
          </p>
          <p>
            <CheckCircle2 size={14} /> Training-only normalization
          </p>
          <p>
            <CheckCircle2 size={14} /> Early stopping & versioned checkpoints
          </p>
          <p>
            <CheckCircle2 size={14} /> Quantile loss & reproducible seeds
          </p>
        </div>
      </section>
      <section className="panel training-monitor">
        <PanelHeading
          title="Learning in progress"
          action={
            job && (
              <span
                className={`tag ${job.status === "completed" ? "cyan" : job.status === "failed" ? "negative" : "amber"}`}
              >
                {job.status === "running" && (
                  <LoaderCircle size={12} className="spin" />
                )}
                {job.status}
              </span>
            )
          }
        />
        {job ? (
          <>
            <div className="job-selector">
              <label htmlFor="job-select">TRAINING RUN</label>
              <select
                id="job-select"
                value={job.id}
                onChange={(e) => setSelected(e.target.value)}
              >
                {jobs.map((j) => (
                  <option value={j.id} key={j.id}>
                    {j.config?.kind?.toUpperCase()} · {j.config?.symbol} ·{" "}
                    {j.id.slice(0, 8)} · {j.status}
                  </option>
                ))}
              </select>
            </div>
            <div className="training-progress">
              <div>
                <strong>
                  Epoch {p?.epoch || 0}
                  <span> / {job.config.epochs}</span>
                </strong>
                <span>{progress}%</span>
              </div>
              <div className="progress-track">
                <i style={{ width: `${progress}%` }} />
              </div>
            </div>
            <div className="training-metrics">
              <Metric
                label="Training loss"
                value={num(p?.loss, 5)}
                detail="Pinball loss · percentage points"
              />
              <Metric
                label="Validation loss"
                value={num(p?.val_loss, 5)}
                color="violet"
                detail="Held-out validation windows"
              />
              <Metric
                label="Elapsed time"
                value={`${Math.round(p?.elapsed_seconds ?? p?.elapsed ?? 0)}s`}
                color="amber"
                detail={
                  <>
                    <Timer size={12} /> Worker elapsed time
                  </>
                }
              />
            </div>
            <div className="chart-title">
              <h3>Loss over epochs</h3>
              <div className="legend">
                <span>
                  <i className="cyan" />
                  Training
                </span>
                <span>
                  <i className="violet" />
                  Validation
                </span>
              </div>
            </div>
            <div className="training-chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={job.history}
                  margin={{ right: 12, left: 0, bottom: 0 }}
                >
                  <CartesianGrid stroke="#202b3c" vertical={false} />
                  <XAxis
                    dataKey="epoch"
                    stroke="#79879d"
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    stroke="#79879d"
                    tickLine={false}
                    axisLine={false}
                    width={45}
                  />
                  <Tooltip {...chartTooltip} />
                  <Line
                    dataKey="loss"
                    name="Training loss"
                    stroke="#52d5c2"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    dataKey="val_loss"
                    name="Validation loss"
                    stroke="#aa95f7"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="worker-details">
              <span>
                Learning rate <strong>{num(p?.learning_rate, 5)}</strong>
              </span>
              <span>
                Gradient norm <strong>{num(p?.gradient_norm, 4)}</strong>
              </span>
              <span>
                Weight change L2 <strong>{num(p?.weight_change, 4)}</strong>
              </span>
            </div>
            {job.error && (
              <div className="error-banner" role="alert">
                {job.error}
              </div>
            )}
            <div className="checkpoint">
              <Cpu size={16} />
              <div>
                <strong>
                  {job.model_id
                    ? "Checkpoint saved"
                    : job.status === "failed"
                      ? "Training interrupted"
                      : "Checkpoint pending"}
                </strong>
                <span>
                  {job.model_id ||
                    "The best validation checkpoint is saved on successful completion."}
                </span>
              </div>
            </div>
            {histogram.length > 0 && (
              <details className="technical-details">
                <summary>Captured weight distribution</summary>
                <div className="weight-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={histogram}>
                      <XAxis
                        dataKey="weight"
                        stroke="#79879d"
                        tickLine={false}
                      />
                      <YAxis stroke="#79879d" width={45} />
                      <Tooltip {...chartTooltip} />
                      <Bar
                        dataKey="count"
                        fill="#a897ff"
                        radius={[3, 3, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </details>
            )}
          </>
        ) : (
          <Empty
            title="Your first experiment starts here"
            action={<FlaskConical size={30} />}
          >
            Choose a model and start a training run. All curves and statistics
            come from the actual PyTorch worker.
          </Empty>
        )}
      </section>
    </div>
  );
}

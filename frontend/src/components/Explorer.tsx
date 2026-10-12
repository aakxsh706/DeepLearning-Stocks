import { useState } from "react";
import { motion } from "framer-motion";
import {
  Play,
  Pause,
  RotateCcw,
  SkipForward,
  Network,
  ArrowRight,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  ScanLine,
  Info,
} from "lucide-react";
import type { Forecast, Telemetry } from "../types";
import { money, num, shortDate } from "../api";
import { Empty, Notice, PanelHeading } from "./Common";
const lessons = [
  {
    title: "A window into the past",
    text: "The model receives a sequence of past market observations. The highlighted candle and input values belong to the selected timestep.",
  },
  {
    title: "Put inputs on a shared scale",
    text: "Inputs are scaled using settings learned from the training data. No test observations are used to fit these settings.",
  },
  {
    title: "Read the sequence, one day at a time",
    text: "The model updates its hidden state as it reads each timestep. Press play to replay the recorded computations at a slower, understandable pace.",
  },
  {
    title: "Inspect a hidden unit",
    text: "These values are learned internal representations. Individual units do not necessarily have a simple financial meaning. Select a unit to inspect its state.",
  },
  {
    title: "Turn a representation into estimates",
    text: "The output head transforms the final hidden representation into lower, median, and upper return estimates. Intermediate timesteps are not separate forecasts.",
  },
  {
    title: "From a return to a price",
    text: "The estimated return is converted to a price using the last observed close. The calculation below uses the actual recorded output.",
  },
  {
    title: "An interval, not a guarantee",
    text: "The interval is an estimated range whose coverage must be evaluated. Open Evaluation to see how often held-out prices actually land inside it.",
  },
  {
    title: "Hold the forecast accountable",
    text: "Historical replay can compare the estimate with the recorded target observation. These are retrospective results, not predictions issued before that market session.",
  },
];
function activationColor(value: number) {
  const amount = Math.min(1, Math.abs(value));
  return value >= 0
    ? `rgba(80,213,192,${0.13 + amount * 0.83})`
    : `rgba(166,137,253,${0.13 + amount * 0.83})`;
}
function NetworkDiagram({
  telemetry,
  t,
  unit,
  onUnit,
  forecast,
}: {
  telemetry: Telemetry;
  t: number;
  unit: number;
  onUnit: (n: number) => void;
  forecast: Forecast;
}) {
  const inputs = telemetry.input_normalized[t] || [],
    hidden = telemetry.hidden[t] || [];
  const inputCount = Math.min(6, inputs.length),
    units = Array.from({ length: Math.min(6, hidden.length) }, (_, i) => i);
  const inputY = (i: number) => 64 + i * 36;
  const unitY = (i: number) => 64 + i * 36;
  const outputY = (i: number) => 100 + i * 56;
  return (
    <div className="network-scroll">
      <svg
        className="network-diagram"
        viewBox="0 0 830 302"
        role="img"
        aria-label={`Recorded ${telemetry.model_kind.toUpperCase()} computation at timestep ${t + 1}`}
      >
        <defs>
          <linearGradient id="connectionGradient">
            <stop offset="0%" stopColor="#45c7b5" />
            <stop offset="100%" stopColor="#a897ff" />
          </linearGradient>
          <filter id="nodeGlow">
            <feGaussianBlur stdDeviation="4" />
          </filter>
        </defs>
        <text x="65" y="24" className="svg-label">
          INPUT FEATURES
        </text>
        <text x="319" y="24" className="svg-label">
          {telemetry.model_kind.toUpperCase()} HIDDEN STATE
        </text>
        <text x="592" y="24" className="svg-label">
          OUTPUT HEAD
        </text>
        {Array.from({ length: inputCount }, (_, i) =>
          units.map((j) => (
            <path
              key={`${i}-${j}`}
              d={`M 208 ${inputY(i)} C 265 ${inputY(i)}, 288 ${unitY(j)}, 348 ${unitY(j)}`}
              stroke="url(#connectionGradient)"
              strokeWidth="1"
              opacity={0.16}
              fill="none"
            />
          )),
        )}
        {units.map((i) =>
          [0, 1, 2].map((j) => (
            <path
              key={`o${i}-${j}`}
              d={`M 400 ${unitY(i)} C 482 ${unitY(i)}, 510 ${outputY(j)}, 592 ${outputY(j)}`}
              stroke="#a897ff"
              strokeWidth="1"
              opacity={0.16}
              fill="none"
            />
          )),
        )}
        {Array.from({ length: inputCount }, (_, i) => (
          <g key={i}>
            <text x="65" y={inputY(i) + 4} className="svg-feature">
              {telemetry.feature_names[i].replaceAll("_", " ").slice(0, 17)}
            </text>
            <rect
              x="160"
              y={inputY(i) - 13}
              width="48"
              height="26"
              rx="5"
              fill="#122e30"
              stroke="#27514e"
            />
            <text
              x="184"
              y={inputY(i) + 4}
              textAnchor="middle"
              className="svg-value cyan-text"
            >
              {num(inputs[i], 2)}
            </text>
          </g>
        ))}
        {units.map((i) => (
          <g
            key={i}
            role="button"
            tabIndex={0}
            aria-label={`Inspect hidden unit ${i + 1}`}
            onClick={() => onUnit(i)}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onUnit(i)}
            className="svg-node"
          >
            <circle
              cx="374"
              cy={unitY(i)}
              r={unit === i ? 23 : 20}
              fill={activationColor(hidden[i])}
              stroke={unit === i ? "#d9cbff" : "#645889"}
              strokeWidth={unit === i ? 2 : 1}
            />
            <text
              x="374"
              y={unitY(i) + 3}
              className="svg-value"
              textAnchor="middle"
            >
              {num(hidden[i], 2)}
            </text>
            <text x="407" y={unitY(i) + 4} className="svg-unit">
              h{i + 1}
            </text>
          </g>
        ))}
        <path
          d="M 346 57 C 307 10, 425 10, 402 58"
          stroke="#8b77bd"
          fill="none"
          strokeDasharray="3 4"
          opacity=".65"
        />
        <text x="439" y="52" className="svg-subtitle">
          hₜ₋₁ → hₜ
        </text>
        {[
          ["Lower", forecast.lower],
          ["Median", forecast.median],
          ["Upper", forecast.upper],
        ].map(([label, value], i) => (
          <g key={label}>
            <rect
              x="592"
              y={outputY(i) - 19}
              width="170"
              height="38"
              rx="7"
              fill={i === 1 ? "#24203f" : "#171c30"}
              stroke={i === 1 ? "#8771c5" : "#39314f"}
            />
            <text x="609" y={outputY(i) + 4} className="svg-feature">
              {label}
            </text>
            <text
              x="746"
              y={outputY(i) + 4}
              className="svg-output"
              textAnchor="end"
            >
              {money(Number(value))}
            </text>
          </g>
        ))}
        <text x="65" y="290" className="svg-subtitle">
          {inputCount} of {inputs.length} features · normalized
        </text>
        <text x="316" y="290" className="svg-subtitle">
          Showing {units.length} of {hidden.length} units
        </text>
        <text x="592" y="290" className="svg-subtitle">
          Final timestep · price in USD
        </text>
      </svg>
    </div>
  );
}
export function Explorer({
  forecast,
  telemetry,
  t,
  setT,
  playing,
  setPlaying,
  speed,
  setSpeed,
  technical,
  expanded,
  onExpand,
  explain,
  setExplain,
}: {
  forecast: Forecast | null;
  telemetry: Telemetry | null;
  t: number;
  setT: (n: number) => void;
  playing: boolean;
  setPlaying: (b: boolean) => void;
  speed: number;
  setSpeed: (n: number) => void;
  technical: boolean;
  expanded?: boolean;
  onExpand?: () => void;
  explain: boolean;
  setExplain: (b: boolean) => void;
}) {
  const [unit, setUnit] = useState(0),
    [lesson, setLesson] = useState(0),
    [raw, setRaw] = useState(false);
  const length = telemetry?.hidden.length || 0;
  const selectedUnit = Math.min(unit, (telemetry?.hidden_size || 1) - 1);
  return (
    <section className={`panel explorer-panel ${expanded ? "expanded" : ""}`}>
      <PanelHeading
        eyebrow="INTELLIGENCE, MADE VISIBLE"
        title="Inside the neural network"
        action={
          <div className="inline-actions">
            <span className="tag">
              <span className="legend-dot cyan" />
              Recorded computation
            </span>
            {onExpand && (
              <button
                onClick={onExpand}
                className="icon-button"
                aria-label="Expand neural network explorer"
              >
                <ScanLine size={17} />
              </button>
            )}
          </div>
        }
      />
      <div className="explorer-subtitle">
        <p>Follow the data. See the computation. Understand the forecast.</p>
        <button
          className={`text-button ${explain ? "cyan-text" : ""}`}
          onClick={() => setExplain(!explain)}
        >
          <BookOpen size={14} />
          {explain ? "Close walkthrough" : "Guided walkthrough"}
        </button>
      </div>
      {telemetry && forecast ? (
        <>
          <div className="pipeline-strip">
            {[
              ["01", "Market data"],
              ["02", "Normalize"],
              ["03", telemetry.model_kind.toUpperCase() + " network"],
              ["04", "Forecast"],
            ].map(([n, label], i) => (
              <div key={n} className={i === 2 ? "current" : ""}>
                <span>{n}</span>
                {label}
                {i < 3 && <ArrowRight size={12} />}
              </div>
            ))}
          </div>
          {explain && (
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              className="lesson"
            >
              <div className="lesson-index">
                {String(lesson + 1).padStart(2, "0")}
                <small>/ 08</small>
              </div>
              <div>
                <h3>{lessons[lesson].title}</h3>
                <p>{lessons[lesson].text}</p>
                {lesson === 5 && (
                  <code>
                    {money(forecast.base_price)} × (1 +{" "}
                    {num(forecast.returns.median, 6)}) ={" "}
                    {money(forecast.median)}
                  </code>
                )}
                {lesson === 7 && (
                  <code>
                    Forecast {money(forecast.median)} → Observed{" "}
                    {money(forecast.actual)}
                  </code>
                )}
              </div>
              <div className="lesson-nav">
                <button
                  className="icon-button"
                  aria-label="Previous explanation step"
                  disabled={lesson === 0}
                  onClick={() => setLesson(lesson - 1)}
                >
                  <ChevronLeft size={17} />
                </button>
                <button
                  className="icon-button"
                  aria-label="Next explanation step"
                  disabled={lesson === 7}
                  onClick={() => setLesson(lesson + 1)}
                >
                  <ChevronRight size={17} />
                </button>
              </div>
            </motion.div>
          )}
          <NetworkDiagram
            telemetry={telemetry}
            forecast={forecast}
            t={Math.min(t, length - 1)}
            unit={selectedUnit}
            onUnit={setUnit}
          />
          <div className="playback">
            <div className="playback-buttons">
              <button
                className="icon-button"
                aria-label="Restart replay"
                onClick={() => {
                  setPlaying(false);
                  setT(0);
                }}
              >
                <RotateCcw size={16} />
              </button>
              <button
                className="play-button"
                aria-label={playing ? "Pause replay" : "Play replay"}
                onClick={() => {
                  if (t === length - 1) setT(0);
                  setPlaying(!playing);
                }}
              >
                {playing ? (
                  <Pause size={16} />
                ) : (
                  <Play size={16} fill="currentColor" />
                )}
              </button>
              <button
                className="icon-button"
                aria-label="Step forward"
                disabled={t === length - 1}
                onClick={() => {
                  setPlaying(false);
                  setT(Math.min(t + 1, length - 1));
                }}
              >
                <SkipForward size={17} />
              </button>
            </div>
            <div className="timestep">
              <div>
                <span>
                  TIMESTEP{" "}
                  <strong>
                    {String(t + 1).padStart(2, "0")} <em>/ {length}</em>
                  </strong>
                </span>
                <span>{shortDate(telemetry.dates[t])}</span>
              </div>
              <input
                aria-label="Replay timestep"
                type="range"
                min="0"
                max={Math.max(0, length - 1)}
                value={t}
                onChange={(e) => {
                  setPlaying(false);
                  setT(+e.target.value);
                }}
              />
            </div>
            <select
              className="speed-select"
              aria-label="Playback speed"
              value={speed}
              onChange={(e) => setSpeed(+e.target.value)}
            >
              {[0.5, 1, 2, 4].map((s) => (
                <option key={s} value={s}>
                  {s}× speed
                </option>
              ))}
            </select>
          </div>
          <div className="playback-caption">
            <Info size={12} /> Slowed replay of a completed inference.
            Connections show architecture, not weight magnitude.
          </div>
          <div className="explorer-detail-grid">
            <div className="heatmap-block">
              <div className="mini-heading">
                <h3>Hidden-state activity</h3>
                <span>
                  {telemetry.hidden_size} units × {length} timesteps
                </span>
              </div>
              <div className="heatmap-label">
                <span>UNIT</span>
                <span>PAST → MOST RECENT</span>
              </div>
              <div className="heatmap">
                {Array.from(
                  { length: Math.min(telemetry.hidden_size, 16) },
                  (_, i) => (
                    <div
                      className={`heatmap-row ${i === selectedUnit ? "selected" : ""}`}
                      key={i}
                    >
                      <span>{String(i + 1).padStart(2, "0")}</span>
                      {telemetry.hidden.map((state, j) => (
                        <button
                          key={j}
                          title={`Unit ${i + 1}, ${telemetry.dates[j]}: ${num(state[i], 4)}`}
                          aria-label={`Unit ${i + 1}, timestep ${j + 1}, activation ${num(state[i], 3)}`}
                          onClick={() => {
                            setUnit(i);
                            setT(j);
                            setPlaying(false);
                          }}
                          style={{
                            background: activationColor(state[i]),
                            outline: j === t ? "1px solid #dce9fb" : undefined,
                          }}
                        />
                      ))}
                    </div>
                  ),
                )}
              </div>
              <div className="heatmap-key">
                <span>
                  <i style={{ background: "#a689fd" }} /> Negative activation
                </span>
                <span>
                  <i style={{ background: "#50d5c0" }} /> Positive activation
                </span>
              </div>
              {telemetry.hidden_size > 16 && (
                <small className="muted">
                  {telemetry.hidden_size - 16} additional units available in the
                  unit selector.
                </small>
              )}
            </div>
            <div className="unit-inspector">
              <div className="mini-heading">
                <h3>Unit inspector</h3>
                <select
                  aria-label="Hidden unit"
                  value={selectedUnit}
                  onChange={(e) => setUnit(+e.target.value)}
                >
                  {Array.from({ length: telemetry.hidden_size }, (_, i) => (
                    <option key={i} value={i}>
                      Unit {i + 1}
                    </option>
                  ))}
                </select>
              </div>
              <div className="activation-readout">
                <span>Hidden activation hₜ</span>
                <strong>{num(telemetry.hidden[t]?.[selectedUnit], 4)}</strong>
              </div>
              {(telemetry.gates || telemetry.gru_gates) &&
                Object.entries(
                  telemetry.gates || telemetry.gru_gates || {},
                ).map(([gate, values]) => {
                  const value = values[t]?.[selectedUnit] ?? 0;
                  const candidate = gate === "candidate";
                  return (
                    <div
                      className="gate-row"
                      key={gate}
                      title={
                        gate === "forget"
                          ? "Controls retention of the previous cell state"
                          : gate === "input"
                            ? "Controls the addition of candidate information"
                            : gate === "output"
                              ? "Controls exposure of cell state through the hidden state"
                              : "Candidate information before gating"
                      }
                    >
                      <span>
                        {gate.charAt(0).toUpperCase() + gate.slice(1)}{" "}
                        {candidate ? "state" : "gate"}
                      </span>
                      <div className="gate-track">
                        <i
                          style={{
                            width: `${(candidate ? (value + 1) / 2 : value) * 100}%`,
                          }}
                        />
                      </div>
                      <code>{num(value, 3)}</code>
                    </div>
                  );
                })}
              {telemetry.cell && (
                <div className="cell-state">
                  <span>Cell state cₜ</span>
                  <code>{num(telemetry.cell[t]?.[selectedUnit], 4)}</code>
                </div>
              )}
              <p className="inspector-note">
                {telemetry.gates || telemetry.gru_gates
                  ? "Gates: 0 = closed, 1 = open. Candidate state: −1 to 1."
                  : "Values are learned representations, not financial signals."}
              </p>
            </div>
          </div>
          {(technical || expanded) && (
            <details className="technical-details" open={technical}>
              <summary>Input tensors & model equations</summary>
              <div className="tensor-header">
                <span>Selected observation: {telemetry.dates[t]}</span>
                <button className="text-button" onClick={() => setRaw(!raw)}>
                  {raw ? "Show normalized" : "Show original units"}
                </button>
              </div>
              <div className="input-tensors">
                {telemetry.feature_names.map((name, i) => (
                  <div key={name}>
                    <span>{name}</span>
                    <code>
                      {num(
                        (raw
                          ? telemetry.input_raw
                          : telemetry.input_normalized)[t]?.[i],
                        5,
                      )}
                    </code>
                  </div>
                ))}
              </div>
              <code className="equation">
                {telemetry.model_kind === "lstm"
                  ? "iₜ,fₜ,oₜ = σ(Wxₜ + Uhₜ₋₁ + b) · gₜ = tanh(…)\ncₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ · hₜ = oₜ ⊙ tanh(cₜ)"
                  : "rₜ,zₜ = σ(Wxₜ + Uhₜ₋₁ + b)\nhₜ = (1 − zₜ) ⊙ nₜ + zₜ ⊙ hₜ₋₁"}
              </code>
              <p>
                Input tensor [1, {length}, {telemetry.feature_names.length}] ·
                Hidden tensor [1, {telemetry.hidden_size}] · Output [1, 3].{" "}
                {raw
                  ? "Original feature units."
                  : "Normalized inputs use training-set mean and scale."}
              </p>
              <details>
                <summary>Saved preprocessing & output head</summary>
                <pre>
                  {JSON.stringify(
                    {
                      normalization: telemetry.normalization,
                      head: telemetry.head,
                    },
                    null,
                    2,
                  )}
                </pre>
              </details>
            </details>
          )}
        </>
      ) : (
        <Empty
          title="Every prediction has a story"
          action={
            <span className="tag">
              <Network size={14} /> Actual model telemetry
            </span>
          }
        >
          Generate a forecast to explore the input sequence, hidden states,
          gates, and output that produced it.
        </Empty>
      )}
      {!telemetry && (
        <Notice>
          No simulated activations. This explorer displays values captured from
          the trained model.
        </Notice>
      )}
    </section>
  );
}

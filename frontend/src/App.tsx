import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  ArrowRight,
  BarChart3,
  BookOpen,
  ChevronDown,
  CircleHelp,
  Cpu,
  FlaskConical,
  History as HistoryIcon,
  LayoutDashboard,
  Menu,
  Network,
  Radio,
  RefreshCw,
  Search,
  Star,
  X,
  Zap,
} from "lucide-react";
import { api } from "./api";
import type {
  Forecast,
  Instrument,
  Job,
  Market,
  Model,
  Tab,
  Telemetry,
} from "./types";
import { useEvents } from "./useEvents";
import { MarketPanel } from "./components/MarketPanel";
import { DataSourcePanel } from "./components/DataSourcePanel";
import { ForecastPanel } from "./components/ForecastPanel";
import { Explorer } from "./components/Explorer";
import { TrainingStudio } from "./components/TrainingStudio";
import { EvaluationCenter } from "./components/EvaluationCenter";
import { Sandbox } from "./components/Sandbox";
import { History } from "./components/History";
import { Modal, Notice } from "./components/Common";
const tabs: {
  id: Tab;
  label: string;
  icon: typeof Activity;
  description: string;
}[] = [
  {
    id: "overview",
    label: "Overview",
    icon: LayoutDashboard,
    description: "Your research, at a glance",
  },
  {
    id: "explorer",
    label: "Network explorer",
    icon: Network,
    description: "From market observations to model outputs",
  },
  {
    id: "training",
    label: "Training studio",
    icon: FlaskConical,
    description: "Build a model. Watch it learn.",
  },
  {
    id: "evaluation",
    label: "Evaluation center",
    icon: BarChart3,
    description: "Measure the model against reality",
  },
  {
    id: "history",
    label: "Prediction archive",
    icon: HistoryIcon,
    description: "Every prediction. Every computation.",
  },
];
const glossary: Record<string, string> = {
  Neuron:
    "A computational unit that transforms input values using learned weights and an activation function.",
  Weight:
    "A learned parameter that controls how much one value contributes to another.",
  Activation:
    "The value produced by a computational unit for a specific input.",
  Epoch: "One pass through the selected training windows.",
  Loss: "The objective minimized during training. Here, quantile (pinball) loss penalizes over- and underestimation differently.",
  "Hidden state":
    "A learned representation updated as the model reads the input sequence.",
  Overfitting:
    "Learning details of training data that do not generalize to unseen observations.",
  "Prediction interval":
    "An estimated range between lower and upper model quantiles. Its empirical coverage must be measured on unseen data.",
};
function Brand() {
  return (
    <div className="brand">
      <svg viewBox="0 0 36 36" width="31" height="31" aria-hidden="true">
        <path
          d="M8 27V9l20 18V9M8 18l10-9 10 9-10 9Z"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.9"
        />
        <circle cx="8" cy="9" r="2.8" fill="currentColor" />
        <circle cx="28" cy="27" r="2.8" fill="#b19cf7" />
      </svg>
      <span>
        Neuro<span>Stock</span>
        <small>INTELLIGENCE, MADE VISIBLE</small>
      </span>
    </div>
  );
}
export default function App() {
  const [tab, setTab] = useState<Tab>("overview"),
    [technical, setTechnical] = useState(false),
    [sidebar, setSidebar] = useState(false),
    [glossaryOpen, setGlossaryOpen] = useState(false),
    [instruments, setInstruments] = useState<Instrument[]>([]),
    [symbol, setSymbol] = useState("AAPL"),
    [models, setModels] = useState<Model[]>([]),
    [modelId, setModelId] = useState(""),
    [market, setMarket] = useState<Market | null>(null),
    [forecasts, setForecasts] = useState<Forecast[]>([]),
    [forecast, setForecast] = useState<Forecast | null>(null),
    [telemetry, setTelemetry] = useState<Telemetry | null>(null),
    [jobs, setJobs] = useState<Job[]>([]),
    [watchlist, setWatchlist] = useState<string[]>([]),
    [loading, setLoading] = useState(true),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [cutoff, setCutoff] = useState(""),
    [t, setT] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(1),
    [explain, setExplain] = useState(false);
  const selection = useRef(0);
  const currentSymbol = useRef(symbol);
  currentSymbol.current = symbol;
  const available = useMemo(
    () => models.filter((m) => m.symbol === symbol),
    [models, symbol],
  );
  const refreshJobs = useCallback(() => {
    api<Job[]>("/training")
      .then(setJobs)
      .catch(() => {});
    api<Model[]>("/models")
      .then(setModels)
      .catch(() => {});
  }, []);
  const refreshHistory = useCallback(
    () =>
      api<Forecast[]>(`/forecasts?symbol=${encodeURIComponent(symbol)}`)
        .then(setForecasts)
        .catch(() => {}),
    [symbol],
  );
  const connected = useEvents((event) => {
    if (event.type === "training") refreshJobs();
    if (event.type === "inference") void refreshHistory();
    if (
      event.type === "market" &&
      (event.payload as Record<string, unknown>)?.symbol === symbol &&
      (event.payload as Record<string, unknown>)?.kind === "observation"
    )
      api<Market>(`/market/${symbol}?limit=520`)
        .then(setMarket)
        .catch(() => {});
    if (
      event.type === "connected" &&
      ((event.payload as Record<string, unknown>)?.replay_gap ||
        (event.payload as Record<string, unknown>)?.cursor_reset)
    ) {
      refreshJobs();
      void refreshHistory();
    }
  });
  const initialLoad = useCallback(async () => {
    const results = await Promise.allSettled([
      api<Instrument[]>("/instruments"),
      api<Model[]>("/models"),
      api<string[]>("/watchlist"),
      api<Job[]>("/training"),
    ]);
    const [i, m, w, j] = results;
    if (i.status === "fulfilled") setInstruments(i.value);
    else setError(`Cannot connect to the API: ${i.reason.message}`);
    if (m.status === "fulfilled") setModels(m.value);
    if (w.status === "fulfilled") setWatchlist(w.value);
    if (j.status === "fulfilled") setJobs(j.value);
  }, []);
  useEffect(() => {
    void initialLoad();
  }, [initialLoad]);
  useEffect(() => {
    if (available.length && !available.some((m) => m.id === modelId))
      setModelId(available[0].id);
    if (!available.length) setModelId("");
  }, [models, symbol, modelId]);
  const chooseForecast = useCallback(async (f: Forecast, open = false) => {
    const ticket = ++selection.current;
    setPlaying(false);
    setT(0);
    setForecast(f);
    setTelemetry(f.telemetry || null);
    if (open) setTab("explorer");
    try {
      const tel =
        f.telemetry || (await api<Telemetry>(`/forecasts/${f.id}/telemetry`));
      if (ticket === selection.current) setTelemetry(tel);
    } catch (e) {
      if (ticket === selection.current) setError((e as Error).message);
    }
  }, []);
  useEffect(() => {
    let cancelled = false;
    ++selection.current;
    setLoading(true);
    setMarket(null);
    setForecast(null);
    setTelemetry(null);
    setForecasts([]);
    setPlaying(false);
    setT(0);
    setCutoff("");
    Promise.allSettled([
      api<Market>(`/market/${symbol}?limit=520`),
      api<Forecast[]>(`/forecasts?symbol=${symbol}`),
    ]).then(([m, f]) => {
      if (cancelled) return;
      if (m.status === "fulfilled") {
        setMarket(m.value);
        setCutoff(m.value.candles.at(-1)?.time || "");
      } else setError(m.reason.message);
      if (f.status === "fulfilled") {
        setForecasts(f.value);
        if (f.value[0]) void chooseForecast(f.value[0]);
      }
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [symbol, chooseForecast]);
  useEffect(() => {
    if (!playing || !telemetry) return;
    const timer = setInterval(
      () =>
        setT((n) => {
          if (n >= telemetry.hidden.length - 1) {
            setPlaying(false);
            return n;
          }
          return n + 1;
        }),
      1000 / speed,
    );
    return () => clearInterval(timer);
  }, [playing, speed, telemetry]);
  useEffect(() => {
    if (!jobs.some((j) => j.status === "running" || j.status === "queued"))
      return;
    const timer = setInterval(refreshJobs, 4000);
    return () => clearInterval(timer);
  }, [jobs, refreshJobs]);
  async function generate() {
    if (!modelId) return;
    setBusy(true);
    setError("");
    try {
      const f = await api<Forecast>("/forecasts", {
        symbol,
        model_id: modelId,
        ...(cutoff ? { cutoff } : {}),
      });
      if (currentSymbol.current !== symbol) return;
      await chooseForecast(f);
      await refreshHistory();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function toggleWatch() {
    const next = watchlist.includes(symbol)
      ? watchlist.filter((s) => s !== symbol)
      : [...watchlist, symbol];
    try {
      setWatchlist(await api<string[]>("/watchlist", { symbols: next }, "PUT"));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const nav = (id: Tab) => {
    setTab(id);
    setPlaying(false);
    setSidebar(false);
    window.scrollTo({ top: 0, behavior: "instant" });
  };
  const explorerProps = {
    forecast,
    telemetry,
    t,
    setT,
    playing,
    setPlaying,
    speed,
    setSpeed,
    technical,
    explain,
    setExplain,
  };
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to main content
      </a>
      {sidebar && (
        <button
          className="sidebar-overlay"
          aria-label="Close navigation"
          onClick={() => setSidebar(false)}
        />
      )}
      <aside className={`sidebar ${sidebar ? "open" : ""}`}>
        <Brand />
        <div className="workspace-label">
          <span className="workspace-icon">N</span>
          <div>
            Personal workspace<small>Research environment</small>
          </div>
          <ChevronDown size={14} />
        </div>
        <div className="nav-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {tabs.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${tab === item.id ? "active" : ""}`}
              onClick={() => nav(item.id)}
            >
              <item.icon size={18} />
              {item.label}
              {tab === item.id && <span className="nav-active-dot" />}
            </button>
          ))}
        </nav>
        <div className="nav-label watchlist-label">
          YOUR WATCHLIST <Star size={12} />
        </div>
        <div className="watchlist">
          {watchlist.length ? (
            watchlist.map((s) => (
              <button
                key={s}
                className={s === symbol ? "selected" : ""}
                onClick={() => setSymbol(s)}
              >
                <span className="watchlist-icon">{s.slice(0, 1)}</span>
                <span>
                  {s}
                  <small>
                    {instruments.find((i) => i.symbol === s)?.name ||
                      "Historical observations"}
                  </small>
                </span>
                <ArrowRight size={13} />
              </button>
            ))
          ) : (
            <p>Star an instrument to save it here.</p>
          )}
        </div>
        <div className="sidebar-bottom">
          <div className="learning-card">
            <BookOpen size={20} />
            <h3>
              A little curiosity.
              <br />A lot of understanding.
            </h3>
            <p>Get to know the ideas behind the network.</p>
            <button onClick={() => setGlossaryOpen(true)}>
              Explore the glossary <ArrowRight size={13} />
            </button>
          </div>
          <button
            className="sidebar-help"
            onClick={() => setGlossaryOpen(true)}
          >
            <CircleHelp size={17} />
            Learning resources <ArrowRight size={13} />
          </button>
          <div className="local-status">
            <span className={`status-dot ${connected ? "cyan" : "amber"}`} />
            {connected
              ? "Local research environment"
              : "Connecting to local backend"}
            <span>v1.0</span>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button mobile-menu"
              onClick={() => setSidebar(true)}
              aria-label="Open navigation"
            >
              <Menu size={20} />
            </button>
            <span>Workspace</span>
            <span className="breadcrumb-slash">/</span>
            <strong>{tabs.find((x) => x.id === tab)?.label}</strong>
          </div>
          <div className="topbar-right">
            <span className={`connection-status ${connected ? "" : "offline"}`}>
              <Radio size={13} />
              {connected
                ? "Event stream connected"
                : "Event stream reconnecting"}
            </span>
            <div className="mode-toggle" aria-label="Explanation level">
              <button
                className={!technical ? "active" : ""}
                onClick={() => setTechnical(false)}
              >
                <BookOpen size={12} />
                Beginner
              </button>
              <button
                className={technical ? "active" : ""}
                onClick={() => setTechnical(true)}
              >
                <Cpu size={12} />
                Technical
              </button>
            </div>
            <button
              className="avatar"
              onClick={() => setGlossaryOpen(true)}
              aria-label="Open learning resources"
            >
              N
            </button>
          </div>
        </header>
        <main id="main">
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                <span className="short-line" />
                THE NEUROSTOCK WORKSPACE
              </div>
              <h1>
                {tab === "overview" ? (
                  <>
                    The market. <span>Understood.</span>
                  </>
                ) : (
                  tabs.find((x) => x.id === tab)?.label
                )}
              </h1>
              <p>
                {tab === "overview"
                  ? "Explore the market. Follow the data. See how a neural network thinks."
                  : tabs.find((x) => x.id === tab)?.description}
              </p>
            </div>
            <span className="mode-badge">
              <span className="status-dot" />
              HISTORICAL REPLAY
              <span className="badge-divider" />
              <span className="badge-cadence">1D</span>
            </span>
          </div>
          {error && (
            <div className="error-banner" role="alert">
              <span>{error}</span>
              <button
                className="text-button"
                onClick={() => {
                  setError("");
                  void initialLoad();
                }}
              >
                <RefreshCw size={13} />
                Retry
              </button>
              <button
                className="icon-button"
                aria-label="Dismiss error"
                onClick={() => setError("")}
              >
                <X size={15} />
              </button>
            </div>
          )}
          <div className="workspace-toolbar">
            <div className="instrument-select">
              <Search size={15} />
              <select
                aria-label="Select stock"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
              >
                {instruments.length ? (
                  instruments.map((i) => (
                    <option key={i.symbol} value={i.symbol}>
                      {i.symbol} · {i.name}
                    </option>
                  ))
                ) : (
                  <option value="AAPL">AAPL · Apple Inc.</option>
                )}
              </select>
              <ChevronDown size={14} />
            </div>
            <div className="toolbar-context">
              <span>
                <span className="legend-dot cyan" />
                Observed
              </span>
              <span>
                <span className="legend-dot violet" />
                Model output
              </span>
              <span>
                <span className="legend-dot amber" />
                Actual outcome
              </span>
            </div>
            <label className="cutoff-label">
              INPUT CUTOFF
              <input
                aria-label="Forecast input cutoff"
                type="date"
                value={cutoff}
                max={market?.candles.at(-1)?.time}
                min={market?.candles[0]?.time}
                onChange={(e) => setCutoff(e.target.value)}
              />
            </label>
          </div>
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.23 }}
          >
            {tab === "overview" ? (
              <>
                <div className="overview-top">
                  <MarketPanel
                    market={market}
                    instrument={instruments.find((i) => i.symbol === symbol)}
                    selectedDate={telemetry?.dates[t]}
                    loading={loading}
                    watched={watchlist.includes(symbol)}
                    onWatch={toggleWatch}
                  />
                  <ForecastPanel
                    forecast={forecast}
                    models={available}
                    modelId={modelId}
                    setModelId={setModelId}
                    onGenerate={generate}
                    busy={busy}
                    market={market}
                    onExplain={() => {
                      setExplain(true);
                      setTab("explorer");
                    }}
                  />
                </div>
                <DataSourcePanel symbol={symbol} market={market} />
                <Explorer
                  {...explorerProps}
                  onExpand={() => setTab("explorer")}
                />
                <div className="overview-bottom">
                  <History
                    forecasts={forecasts}
                    onReplay={(f) => void chooseForecast(f, true)}
                    compact
                  />
                  <Sandbox key={forecast?.id} forecast={forecast} />
                </div>
              </>
            ) : tab === "explorer" ? (
              <>
                <div className="explorer-page-top">
                  <div className="replay-select">
                    <label htmlFor="replay-select">RECORDED PREDICTION</label>
                    <select
                      id="replay-select"
                      value={forecast?.id || ""}
                      onChange={(e) => {
                        const f = forecasts.find(
                          (x) => x.id === e.target.value,
                        );
                        if (f) void chooseForecast(f);
                      }}
                    >
                      {!forecasts.length && (
                        <option value="">
                          Generate a forecast in Overview
                        </option>
                      )}
                      {forecasts.map((f) => (
                        <option value={f.id} key={f.id}>
                          {f.symbol} · {f.model_kind.toUpperCase()} · target{" "}
                          {f.target_time} · {f.id.slice(0, 7)}
                        </option>
                      ))}
                    </select>
                  </div>
                  <button
                    className="button secondary"
                    onClick={() => setTab("overview")}
                  >
                    Back to market <ArrowRight size={15} />
                  </button>
                </div>
                <Explorer {...explorerProps} expanded />
                <div className="overview-bottom explorer-market">
                  <MarketPanel
                    market={market}
                    instrument={instruments.find((i) => i.symbol === symbol)}
                    selectedDate={telemetry?.dates[t]}
                    loading={loading}
                    watched={watchlist.includes(symbol)}
                    onWatch={toggleWatch}
                  />
                  <Sandbox key={forecast?.id} forecast={forecast} />
                </div>
              </>
            ) : tab === "training" ? (
              <TrainingStudio
                symbol={symbol}
                jobs={jobs}
                onRefresh={refreshJobs}
                onError={setError}
              />
            ) : tab === "evaluation" ? (
              <EvaluationCenter
                models={available}
                modelId={modelId}
                onModelChange={setModelId}
              />
            ) : (
              <History
                forecasts={forecasts}
                onReplay={(f) => void chooseForecast(f, true)}
              />
            )}
          </motion.div>
          {!models.length && !loading && tab === "overview" && (
            <div className="first-run">
              <Zap size={22} />
              <div>
                <h3>Your market data is ready. Now give it a model.</h3>
                <p>
                  Start a bounded training run to unlock forecasts and the
                  network explorer.
                </p>
              </div>
              <button
                className="button primary"
                onClick={() => nav("training")}
              >
                Open training studio <ArrowRight size={16} />
              </button>
            </div>
          )}
          <footer className="page-footer">
            <span>
              <Activity size={13} />
              Built to understand, not to promise.
            </span>
            <span>Research & education · No trading recommendations</span>
            <a
              href="https://www.tradingview.com/"
              target="_blank"
              rel="noreferrer"
            >
              Charts by TradingView
            </a>
          </footer>
        </main>
      </div>
      {glossaryOpen && (
        <Modal
          title="A field guide to neural networks"
          onClose={() => setGlossaryOpen(false)}
        >
          <p className="section-intro">
            A few useful ideas for your next exploration.
          </p>
          <dl className="glossary">
            {Object.entries(glossary).map(([term, meaning]) => (
              <div key={term}>
                <dt>{term}</dt>
                <dd>{meaning}</dd>
              </div>
            ))}
          </dl>
          <Notice>
            Use the guided walkthrough in the explorer to see these concepts
            attached to a real recorded prediction.
          </Notice>
        </Modal>
      )}
    </div>
  );
}

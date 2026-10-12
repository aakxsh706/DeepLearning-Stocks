export type Tab =
  | "overview"
  | "explorer"
  | "training"
  | "evaluation"
  | "history";
export type Instrument = {
  symbol: string;
  name: string;
  exchange: string;
  timezone: string;
};
export type Candle = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};
export type Market = {
  symbol: string;
  candles: Candle[];
  source: string;
  mode: string;
  delay_seconds: number | null;
  timezone: string;
  session_status: string;
  as_of: string;
};
export type Model = {
  id: string;
  kind: string;
  symbol: string;
  lookback: number;
  hidden_size: number;
  training_cutoff: string;
  version?: string;
  feature_names: string[];
  metrics?: Record<string, number>;
  splits?: Record<string, { start: string; end: string; count: number }>;
};
export type Telemetry = {
  feature_names: string[];
  input_raw: number[][];
  input_normalized: number[][];
  dates: string[];
  hidden: number[][];
  cell?: number[][];
  gates?: Record<string, number[][]>;
  gru_gates?: Record<string, number[][]>;
  hidden_size: number;
  model_id: string;
  model_kind: string;
  units: string;
  output: {
    returns: { lower: number; median: number; upper: number };
    prices: { lower: number; median: number; upper: number };
  };
  normalization?: { mean: number[]; scale: number[] };
  head?: Record<string, unknown>;
  weights?: Record<string, unknown>;
};
export type Forecast = {
  id: string;
  symbol: string;
  model_id: string;
  model_kind: string;
  model_version: string;
  issue_time: string;
  target_time: string;
  input_cutoff: string;
  data_snapshot_id: string;
  mode: string;
  base_price: number;
  lower: number;
  median: number;
  upper: number;
  returns: { lower: number; median: number; upper: number };
  lookback: number;
  training_cutoff: string;
  actual?: number | null;
  outcome?: Record<string, unknown>;
  telemetry?: Telemetry;
};
export type TrainingPoint = {
  epoch: number;
  epochs?: number;
  loss: number;
  val_loss: number;
  gradient_norm: number;
  learning_rate: number;
  elapsed_seconds?: number;
  elapsed?: number;
  weight_histogram?: unknown;
  weight_change?: number;
};
export type Job = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  config: {
    symbol: string;
    kind: string;
    lookback: number;
    epochs: number;
    max_samples: number;
  };
  progress?: TrainingPoint;
  history: TrainingPoint[];
  model_id?: string;
  error?: string;
};
export type Evaluation = {
  model_id: string;
  split: string;
  metrics: {
    mae: number;
    rmse: number;
    directional_accuracy: number;
    coverage: number;
    interval_width: number;
  };
  baseline: { mae: number; rmse: number; directional_accuracy: number };
  rows: {
    date: string;
    actual: number;
    median: number;
    lower: number;
    upper: number;
    base_price: number;
    error: number;
  }[];
  walk_forward: Record<string, unknown>[];
  definitions: Record<string, unknown>;
};
export type Sensitivity = {
  original: Forecast;
  perturbed: Forecast;
  changes: unknown;
  sensitivity_only: boolean;
};

# NeuroStock model methodology

NeuroStock is an educational, daily historical forecasting and computation-replay application. The bundled results are retrospectively generated historical tests, not forecasts issued in 2017, live connectivity, trading recommendations, or evidence of profitable prediction.

## Data and price convention

Two real, offline CSV snapshots contain 506 completed daily observations apiece for Apple (AAPL) and Microsoft (MSFT), from **2015-02-17 through 2017-02-16**. Sources, exact source revisions, original/normalized SHA-256 hashes, and date coverage are recorded in `data/provenance.json`. AAPL comes from Plotly's `finance-charts-apple.csv`; MSFT is the same date subset of Plotly's `all_stocks_5yr.csv`. `scripts/acquire_data.py --verify-only` validates the bundled data without network. Running it without that flag re-downloads and verifies source hashes before replacement.

The application uses source OHLC prices and volume. AAPL's separate adjusted-close column and precomputed technical indicators are intentionally excluded. Neither stock splits during the bundled 2015–2017 interval. Prices in this interval are nominal historical traded prices; dividends are not adjusted. Targets therefore predict next-session price changes rather than total investment returns. Imported data spanning splits must be adjusted consistently across **all OHLC and volume**, with point-in-time adjustment policies documented, before extending this demo. Do not combine adjusted close with unadjusted open/high/low. Source snapshots may contain vendor revisions and are not a point-in-time institutional dataset.

The exchange calendar uses `exchange_calendars` XNYS trading sessions as the US equity daily-session schedule (also appropriate for these Nasdaq stocks in the bundled interval), explicitly bounded 2000–2035. Weekends, exchange holidays and exceptional closures are skipped. Imports and inference snapshots must contain consecutive exchange sessions; missing dates and nontrading rows are rejected so that a next-session label cannot silently become a multiday label. The display timezone is America/New_York. Forecasts made at the final bundled close target the next exchange session, whose outcome is unavailable in the bundle.

## Inputs, boundaries, and preprocessing

Eight causal features are calculated after each completed candle: simple close return, open/previous-close gap, high/previous-close relative change, low/previous-close relative change, close/open return, `log1p(volume)`, trailing ten-return population volatility, and close/ten-close-moving-average gap. All rolling statistics look backward and include the completed current session. The first ten rows are warm-up and are excluded from windows.

A sample contains `lookback` feature rows ending at session t and a label `close(t+1)/close(t) − 1`. The default lookback is 30 and supported training range is 10–60. All target dates are partitioned chronologically using raw-observation boundaries: first 65% training, following 15% validation, final 20% historical test. Input windows can overlap earlier partitions because those observations would already be known, but **targets do not cross partitions**. There is no pre-split shuffle and training batches remain chronological.

The standard scaler is fit once to unique feature rows used by training inputs, ending one row before the final training label. Validation/test observations never contribute to means or scales. Scaler means/scales, feature order, fit dates, fit count, date splits, data hash, model seed and architecture are versioned with the model. The validation window affects checkpoint selection; accordingly inference is rejected before the validation cutoff as well as before the training cutoff. A forecast whose input cutoff equals the validation cutoff may target the first held-out test session.

## Models and actual telemetry

Both recurrent architectures have one layer with 16 hidden units and eight inputs. They process the same complete daily sequences and share a three-parameter dense output head. No dropout is used. Initial states are zeros. PyTorch seed 42 and deterministic CPU operations make an identical configuration reproducible on the pinned stack; UUIDs and wall-clock timestamps naturally differ between runs.

The instrumented LSTM is the model used both for optimization and inference, with PyTorch's `i,f,g,o` order:

```
[a_i, a_f, a_g, a_o] = W_ih x_t + b_ih + W_hh h_(t-1) + b_hh
i = sigmoid(a_i); f = sigmoid(a_f); g = tanh(a_g); o = sigmoid(a_o)
c_t = f * c_(t-1) + i * g
h_t = o * tanh(c_t)
```

Recorded input, forget, and output gates lie in [0,1]; the **candidate** uses tanh and lies in [-1,1]. Cells are unbounded internal states. Hidden activations lie in [-1,1]. The GRU uses PyTorch's `r,z,n` convention, applies its reset gate after the hidden affine map, and records reset/update/candidate values under `gru_gates`. It does not pretend to have LSTM cell or forget states. Unit meanings are learned representations; a unit or gate is not assigned a financial interpretation.

Telemetry includes every actual input feature row, the float32 normalized tensor passed into the model, dates, original candles, hidden states, LSTM cells/gates or GRU gates, the exact dense head weights/biases/raw values, selected recurrent weights, return quantiles and corresponding prices. Full telemetry is captured for explicitly requested forecasts and stored for replay. Replaying a stored forecast reads its original snapshot rather than regenerating a new model animation. Playback time is a slowed display of completed computation, not a measure of inference latency.

## Quantiles, optimization, and uncertainty

Let the dense head produce `a,b,c` in return percentage-point units. Quantiles are parameterized as:

```
median_return = a / 100
lower_return  = (a − softplus(b)) / 100
upper_return  = (a + softplus(c)) / 100
price_q       = last_completed_close * (1 + return_q)
```

The positive softplus spacings prevent crossing by construction. Training minimizes mean pinball loss for quantiles 0.10, 0.50 and 0.90, measured in percentage points. Optimization uses Adam at 0.008, chronological batches of 64, maximum gradient norm 2, and validation early stopping after four non-improving epochs. The best validation checkpoint is restored. Training jobs are bounded to 1–20 epochs and 200–2,000 source observations. Demo artifacts are trained for up to 12 epochs; histories show the actual number completed.

The nominal central interval is **80%**, not an asserted probability that the next price will be inside it. Coverage must be measured on unseen dates and can be badly miscalibrated, especially with these small data. No post-hoc calibration on the test set is applied. Training progress reports measured loss, validation loss, pre-clipping gradient norm averaged over batches, actual learning rate, elapsed seconds, histograms of all learned parameters, L2 parameter changes since the previous epoch, and checkpoint selection.

## Evaluation and model race

All metrics use the model's untouched chronological test targets and the same next-session horizon. Persistence predicts exactly the final input close. MAE is mean absolute median-price error; RMSE is root mean square median-price error. Both are in price USD. Directional accuracy is the fraction for which `sign(median − base_close) == sign(actual − base_close)`. Zero is its own direction class, so a persistence forecast counts as directionally correct only when the actual price is unchanged. This is intentionally disclosed and does not imply that the baseline is a directional classifier.

Interval coverage counts inclusive lower/upper containment. Width is mean upper minus lower price in USD. The model race compares predictions against this same persistence rule, dates and lookback endpoint; differing custom training-data limits can create different test periods and should not be ranked without aligning dates.

Walk-forward results advance the completed observation window one session at a time through the historical test period. Model weights and training-fit scaler remain frozen. Results are reported in three nonoverlapping chronological blocks. This is **rolling-origin evaluation of a fixed checkpoint**, not expanding-window retraining or independent cross-validation. It matches deployed daily use of a fixed version. The implementation makes no claim to have refit the model at each origin. Test prices are allowed in subsequent input windows only after that observation has occurred; no test target is used for optimization or normalization.

Training, validation and test dates remain explicit in metadata. There are no forward-running evaluation results in the offline bundle. Forecast issue time is the wall-clock time the record was created; historical input and target dates remain distinct and mode is Replay. Known historical outcomes attach to the correct target session automatically. They are not passed into inference. Forecast records are immutable; outcomes are a separately stored entity.

## What-if and artifact safety

What-if changes the **last input candle only**, by −20% to +20% in close or volume. A close perturbation expands high/low as necessary to preserve `low ≤ open,close ≤ high`; every changed field is shown exactly. All features, including returns, rolling statistics and normalization, are recomputed from the original historical snapshot. The price anchor also changes when close changes, and this is reflected in the displayed base price. The training scaler and weights do not change. Historical outcome comparisons are omitted for perturbed forecasts. These are model-sensitivity experiments, not causal predictions of the market.

Each model version is a unique immutable directory containing schema-version-2 JSON metadata, precomputed evaluation and NumPy numeric weights. The loader only accepts strict registered model IDs, rejects symlink artifacts, verifies both weights and evaluation SHA-256 hashes, checks the evaluation's model identifier, disables NumPy pickle loading, and requires exact parameter names/shapes. There is no arbitrary path or pickled-model upload API. Bundled versions live under `artifacts/demo`; new training versions use `ARTIFACT_DIR` (default `artifacts`). Bundled manifest lineage records the original trained version when the same learned weights are repackaged under a new integrity-verified version.

## Limits and verification

This small dated sample cannot establish predictive superiority, regime robustness, execution viability or calibrated uncertainty. Costs, spreads, liquidity, delistings, survivorship bias, corporate fundamentals and exogenous events are not modeled. One-session point forecasts often underperform persistence, and those errors remain visible. Daily models must never be used for minute-level inference without separate data, preprocessing, training, artifacts and evaluation.

Automated numerical tests match the complete custom LSTM sequence and final cell against PyTorch `nn.LSTM` using identical weights/initial states at float64 tolerance 1e-12, and match the GRU cell to `nn.GRUCell`. Tests poison future observations to verify unchanged training inputs, labels and fitted scaler; check chronological windows, return-to-price/head correspondence, quantile ordering, stored-snapshot replay, outcome matching, what-if constraints, baseline metrics, calendar holidays and exceptional closures, and unsafe version rejection. API and browser tests cover persistence and interaction separately.

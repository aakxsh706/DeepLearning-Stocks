# NeuroStock

A local stock forecasting lab where every neural-network animation comes from recorded PyTorch inference. Explore historical daily candles, generate quantile forecasts, step through LSTM gates and hidden states, train a small model in a separate worker, and compare it honestly with persistence.

The bundled demo contains **real AAPL and MSFT historical observations from February 2015–February 2017** and four small trained models. It works without API keys or runtime Internet access. These observations and predictions are labeled **Historical Replay**, never live. The models are educational; they do not establish profitable trading performance.

## Get the source

```bash
git clone https://github.com/aakxsh706/DeepLearning-Stocks.git
cd DeepLearning-Stocks
```

Run the setup commands below from this repository directory.

## Run with Docker Compose

Requirements: Docker Engine/Desktop with Compose v2, a few GB of free disk space, and network access for the first image/dependency build.

```bash
cp .env.example .env
docker compose up --build
```

Open **http://localhost:3000**. API documentation: **http://localhost:8000/docs**.

Compose starts PostgreSQL, applies migrations, starts FastAPI, starts the independent training worker, and serves the production frontend through nginx. Forecasts, outcomes, training jobs, and watchlists persist in the PostgreSQL volume; new checkpoints persist in the model volume. Bundled model checkpoints are included in the image.

```bash
docker compose logs -f api worker
docker compose stop
```

`docker compose down` removes containers while retaining data volumes. Do not remove volumes if you want to keep your forecasts and training history. HTTP ports bind to loopback; this local single-user app has no authentication.

## Run directly without Docker

Requirements: **Python 3.13**, **Node.js 22**, and npm. Apple Silicon macOS and Linux are the supported native environments; use Docker on Intel Macs because this PyTorch version has no native Intel-mac wheels. The Python lock includes the complete installed dependency set; npm uses `package-lock.json`.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.lock
npm --prefix frontend ci
.venv/bin/python scripts/dev.py
```

Open **http://localhost:5173**. The launcher applies migrations, waits for the API, then starts the separate worker and frontend. Ctrl-C stops the three processes. The default development database is SQLite (`neurostock.db`); use PostgreSQL for the Compose deployment and production database behavior.

If the default ports are occupied:

```bash
API_PORT=8017 WEB_PORT=5177 .venv/bin/python scripts/dev.py
```

For local PostgreSQL, set `DATABASE_URL` in `.env` before launching:

```dotenv
DATABASE_URL=postgresql+psycopg://neurostock:your-password@localhost:5432/neurostock
```

To run processes individually from the project root:

```bash
PYTHONPATH=backend .venv/bin/alembic -c backend/alembic.ini upgrade head
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
# In a second terminal:
PYTHONPATH=backend .venv/bin/python -m app.worker
# In a third terminal:
npm --prefix frontend run dev -- --host 127.0.0.1
```

The launcher loads `.env`; individual commands require environment variables exported by your shell. `API_PROXY_TARGET` can point Vite at a different backend port. On Linux x86_64, installing the CPU wheel first avoids CUDA dependencies:

```bash
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
.venv/bin/python -m pip install -r backend/requirements.lock
```

## Reproduce the demo

1. Select **AAPL** and a bundled **LSTM** model in Overview.
2. Set the input cutoff to **2017-02-14** to inspect a prediction with a known historical outcome, or use the latest available date for a target not present in the snapshot.
3. Click **Generate forecast**. The lower, median, and upper estimates come from the saved trained checkpoint.
4. Open **Explain this prediction**. Play, pause, restart, or step through recorded timesteps. The highlighted candle and heatmap column track the same observation.
5. Switch to **Technical** to inspect original/scaled features, tensor shapes, equations, preprocessing, and the output head. Select hidden units to view exact gate values.
6. Use **What-if sandbox** to perturb the final close or volume. Any necessary high/low adjustment is listed. This is model sensitivity, not a causal market forecast.
7. Open **Prediction archive** and replay a stored prediction. Its telemetry and input snapshot are preserved independently of subsequent model training.
8. In **Training studio**, submit a bounded LSTM or GRU run. Epoch loss, validation loss, gradient norms, weight statistics, elapsed time, and checkpoint status come from the separate worker.
9. In **Evaluation center**, inspect held-out price errors, interval coverage, historical forecast/actual rows, walk-forward reporting blocks, and persistence comparisons. The model race compares only aligned observations.

No neural activation or training metric is randomly animated. Playback is explicitly a slowed replay of completed inference. The diagram's connections represent architecture; they do not represent literal signal traffic.

## Train and evaluate outside the UI

```bash
.venv/bin/python scripts/train.py --symbol AAPL --kind all --lookback 30 --epochs 12 --max-samples 1200
.venv/bin/python scripts/train.py --symbol MSFT --kind all --lookback 30 --epochs 12 --max-samples 1200
.venv/bin/python scripts/evaluate.py MODEL_ID --output runtime/evaluation.json
```

Replace `MODEL_ID` with an ID printed by training or returned by `/api/models`. Training is limited to 1–20 epochs, 10–60 lookback observations, 200–2,000 selected observations, and one queued/running UI job. Early stopping can finish before the requested epoch count. Numeric weights, training-only scaler settings, feature names, split boundaries, configuration, checksums, and test evaluation are versioned together.

For a model trained inside Compose, evaluate it in the container where its model volume is available: `docker compose exec api python scripts/evaluate.py MODEL_ID`.

The standard demo has 506 observations per symbol; a larger `max_samples` is a limit, not a claim of additional data. Training uses reproducible CPU seeds; timings and newly generated version IDs will differ.

## Historical data and external integration

`data/provenance.json` identifies the source URLs, observation ranges, hashes, and price conventions. `scripts/acquire_data.py` verifies the bundled files or reacquires the pinned source data; see its `--help` for exact modes. The data uses nominal OHLC over a window without stock splits. Dividend adjustments are not applied; these are price forecasts, not total-return forecasts. Precomputed adjusted prices and source indicators are excluded from features.

Historical replay requires **no credentials**. The optional **Alpha Vantage** integration requires `ALPHA_VANTAGE_API_KEY` on the server. A quote is fetched explicitly, displayed separately, and labeled **Delayed / end-of-day** with unknown exact delay. API failures and entitlement/rate-limit problems are explicit; the app does not invent prices or claim live connectivity. External quotes never silently become model inputs. The provider interface also supports historical retrieval and completed-candle access for server-side import workflows.

Validated daily CSV additions are synchronized to observation records and available forecast outcomes are reconciled automatically. Existing immutable forecasts retain their original snapshots. The bundled replay has no prospective, forward-running track record. Intraday inference, automatic external-history ingestion, public multi-user hosting, and the optional CNN-LSTM are not implemented.

## Tests and verification

```bash
.venv/bin/pytest -q
npm --prefix frontend run build
# API and worker must already be running:
.venv/bin/python scripts/smoke.py --base-url http://localhost:8000 --train
# Frontend and API must already be running; install browser once if necessary:
cd frontend
npx playwright install chromium
npm run test:e2e
```

The Python suite checks causal windows, chronological splits, training-only scaling, ordered quantiles, return-to-price conversion, instrumented-cell equivalence, exact telemetry, deterministic replay, holiday calendars, outcome matching, what-if validity, immutable records, baseline alignment, training limits/concurrency, worker progress, and WebSocket recovery. The running-service smoke test uses a real trained checkpoint, database records, the event stream, and an actual worker job.

Browser tests exercise selecting a stock, generating a forecast, stepping through computation, inspecting output, replaying a saved prediction, and mobile layout. See [verification notes](docs/VERIFICATION.md) for the actual commands run and environment limits.

## Architecture and methodology

- [Architecture, processes, audit records, and events](docs/ARCHITECTURE.md)
- [Model methodology, splits, equations, evaluation, and limitations](docs/METHODOLOGY.md)
- [Data source and adjustment provenance](data/provenance.json)
- [API implementation](backend/app/main.py) and interactive OpenAPI documentation at `/docs`
- [Instrumented recurrent cells](backend/app/models.py)

Evaluation is intentionally transparent: complex models may perform worse than persistence. Reported intervals are estimated 0.10–0.90 quantiles, not guaranteed confidence or trading recommendations. Historical replay and forward-running evaluation are different kinds of evidence.

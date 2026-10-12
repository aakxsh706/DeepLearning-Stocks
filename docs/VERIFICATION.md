# Verification record

These are observed checks, not claims that untested integrations are working.

| Check | Result |
| --- | --- |
| Python unit and integration suite | **35 passed** (`.venv/bin/pytest -q`) |
| Browser end-to-end suite | **6 passed** in installed Google Chrome, including desktop and 390 × 844 mobile |
| TypeScript and production frontend build | Passed (`npm --prefix frontend run build`) |
| Frontend dependency audit | Zero reported vulnerabilities at the time checked |
| Python dependency consistency | Passed (`.venv/bin/python -m pip check`) |
| Historical source verification | Both 506-row datasets passed offline hashes and validation |
| Actual PostgreSQL migration | `0001_initial` applied successfully to PostgreSQL 17.10 |
| PostgreSQL immutability | Database rejected both UPDATE and DELETE of a stored forecast |
| Running-service smoke test | Real forecast, telemetry, exact replay, sensitivity, baseline evaluation, missed-event recovery, and separate worker training passed |
| Local launcher | Migration, API startup, worker, Vite proxy, alternate-port WebSocket origin, and clean child shutdown verified |
| Compose configuration | YAML parsed; migration/health dependencies and shared volumes checked |

## Browser workflow

```bash
# Run API/worker on 8017 and frontend on 5177, or adjust the base URL.
cd frontend
PLAYWRIGHT_BASE_URL=http://127.0.0.1:5177 PLAYWRIGHT_CHANNEL=chrome npm run test:e2e
```

The tests use real API results, trained models, PostgreSQL records, and the independent Python worker. They do not replace model outputs with mocked values. Assertions cover:

1. Stock selection, forecast creation, exact displayed hidden activation, unit selection, play/pause/step controls, output inspection, and stored replay.
2. What-if inference preserving the immutable original and evaluation displaying persistence.
3. A real two-epoch bounded GRU job, measured gradient statistics, streamed progress, and a saved checkpoint.
4. Mobile document width, reduced motion, keyboard stepping, and archive navigation.
5. Missing external credentials, explicit provider failure, and continued availability of historical replay.
6. Glossary-dialog keyboard focus containment, Escape dismissal, and focus restoration.

An initial mobile archive test detected horizontal overflow from an absolutely positioned screen-reader label in the table. The containing region and table header were corrected; all browser tests then passed. Final screenshots are [desktop](screenshots/desktop.png) and [mobile](screenshots/mobile.png).

## Real model check

For the bundled AAPL LSTM, historical test MAE is approximately **$0.81386**, compared with persistence MAE **$0.80412** on the same 102 target sessions. This model loses to the baseline on MAE in that test window. These numbers describe an old held-out sample; they are not live or prospective performance.

The instrumented LSTM matches an equivalent standard PyTorch LSTM with matched weights and states at float64 tolerance 1e-12. GRU consistency is also tested. Tests verify exact normalized input tensors for held-out windows, forecast/head/price correspondence, chronological split and scaler isolation, future-input poisoning, ordered quantiles, holiday calendars, missing-session rejection, artifact tampering, immutable snapshots, delayed outcome matching, source revision rejection, job concurrency, and WebSocket cursor recovery.

## Verification limits

Docker Engine/Compose is not installed on the build host. The Compose stack and container images were **not built or launched** here. The application processes and database path were instead tested natively against a temporary PostgreSQL 17.10 instance and with SQLite integration tests. The Compose file pins PostgreSQL 17.6, so that exact image remains unverified.

No Alpha Vantage credentials were provided. Missing-credential behavior is tested, but actual external quote connectivity, entitlement, provider delay, and rate limits are **not verified**. There is no live feed or forward-running performance claim. The external integration is opt-in and supplies end-of-day quotes separately from historical model inputs.

Python tests emit one upstream Starlette/AnyIO deprecation warning concerning `BlockingPortal`; it does not fail the suite. Tests used Python 3.13.9 and Node.js 22.19.0 on macOS. Cross-platform/container runtime verification remains to be performed on a machine with Docker.

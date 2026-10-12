"""NeuroStock local research API. All bundled-data forecasts are Historical Replay."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import delete, func, select, update, text
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from . import db
from .data import ensure_demo_data, list_instruments, data_source_info, load_frame
from .provider import external_provider, frame_records, replay_provider, ProviderUnavailable
from .schemas import ForecastRequest, TrainingRequest, WatchlistRequest, WhatIfRequest

logger = logging.getLogger(__name__)
CORS_ORIGINS = [v.strip() for v in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000").split(",") if v.strip()]
EVENT_RETENTION = max(100, min(int(os.getenv("EVENT_RETENTION", "2000")), 10000))
_INFERENCE_SLOTS = threading.BoundedSemaphore(2)
_RATE_LOCK = threading.Lock()
_RATE_BUCKETS: dict[str, deque] = defaultdict(deque)


def fresh_id():
    return uuid.uuid4().hex


def public_metadata(meta):
    # Filesystem paths are an implementation detail, never a load-file capability.
    return {key: value for key, value in meta.items() if not any(part in key.lower() for part in ("path", "location"))}


def register_models(session):
    from .ml import list_model_artifacts
    metadata = list_model_artifacts()
    for meta in metadata:
        model_id = meta.get("id", meta.get("model_id"))
        existing = session.get(db.ModelVersion, model_id)
        if existing is not None and existing.metadata_json != meta:
            raise HTTPException(409, "A registered model artifact was modified; restore its original immutable version")
        if existing is None:
            db.insert_ignore(session, db.ModelVersion, {"id": model_id, "symbol": meta["symbol"], "kind": meta["kind"], "metadata_json": meta, "created_at": db.utcnow()})
            db.insert_ignore(session, db.PreprocessingArtifact, {
                "id": f"{model_id}-preprocessing", "model_id": model_id,
                "metadata_json": {key: meta[key] for key in ("feature_names", "normalization", "preprocessing", "training_cutoff", "scaler", "preprocessing_path", "preprocessing_sha256") if key in meta}})
    session.flush()
    return metadata


def _require_instrument(session, symbol):
    if session.get(db.Instrument, symbol) is None:
        raise HTTPException(404, "Instrument is not available in the bundled historical dataset")


def _require_model(session, model_id, symbol=None):
    register_models(session)
    model = session.get(db.ModelVersion, model_id)
    if model is None:
        raise HTTPException(404, "Model version is not registered. Train a demo model first.")
    if symbol is not None and model.symbol != symbol:
        raise HTTPException(422, "This model was trained for a different instrument")
    return model


def serialize_forecast(session, row):
    result = dict(row.payload)
    outcome = session.scalar(select(db.ForecastOutcome).where(db.ForecastOutcome.forecast_id == row.id))
    if outcome:
        result["actual"] = outcome.payload["actual"]
        result["outcome"] = {**outcome.payload, "observed_at": outcome.observed_at}
    return result


def reconcile_outcomes(session):
    """Attach available target-session observations without ever changing forecasts."""
    pending = session.scalars(select(db.Forecast).outerjoin(
        db.ForecastOutcome, db.ForecastOutcome.forecast_id == db.Forecast.id
    ).where(db.ForecastOutcome.id.is_(None))).all()
    added = 0
    for forecast in pending:
        observation = session.scalar(select(db.Observation).where(
            db.Observation.symbol == forecast.symbol, db.Observation.time == forecast.target_time,
            db.Observation.source_id == "bundled_daily"))
        if observation is None:
            continue
        actual = float(observation.values["close"])
        payload = forecast.payload
        actual_return = actual / payload["base_price"] - 1
        median_return = payload["median"] / payload["base_price"] - 1
        # Exact zero is its own directional category, including the baseline.
        sign = lambda value: 0 if value == 0 else (1 if value > 0 else -1)
        outcome = {"actual": actual, "target_time": forecast.target_time,
                   "error": payload["median"] - actual, "absolute_error": abs(payload["median"] - actual),
                   "covered": payload["lower"] <= actual <= payload["upper"],
                   "direction_correct": sign(actual_return) == sign(median_return),
                   "mode": "Replay", "historical_outcome": True,
                   "note": "Matched historical target session; not a forward-running prediction."}
        db.insert_ignore(session, db.ForecastOutcome, {"id": fresh_id(), "forecast_id": forecast.id,
                         "target_time": forecast.target_time, "payload": outcome, "observed_at": db.utcnow()})
        added += 1
    return added


class SourceRevisionError(ValueError):
    pass


def verified_source_frame(session, symbol):
    frame = load_frame(symbol)
    known = {row.time: row.values for row in session.scalars(select(db.Observation).where(
        db.Observation.symbol == symbol, db.Observation.source_id == "bundled_daily"))}
    new_candles = []
    for candle in frame_records(frame):
        if candle["time"] in known:
            if candle != known[candle["time"]]:
                raise SourceRevisionError(
                    f"Historical source revision detected for {symbol} on {candle['time']}. "
                    "Restore the original data or import the revised dataset as a separate source version; "
                    "stored forecasts and outcomes were preserved.")
        else:
            new_candles.append(candle)
    return frame, new_candles


def import_new_observations(session, emit_events=False):
    """Append validated source rows; historical revisions never overwrite snapshots.

    An explicitly imported, later completed session is discovered on the next
    maintenance pass and can resolve a pending outcome automatically.
    """
    for instrument in list_instruments():
        symbol = instrument["symbol"]
        _frame, new_candles = verified_source_frame(session, symbol)
        appended = 0
        for candle in new_candles:
            inserted = db.insert_ignore(session, db.Observation, {"symbol": symbol, "time": candle["time"],
                                        "source_id": "bundled_daily", "values": candle})
            appended += inserted.rowcount
        if emit_events and appended:
            db.emit(session, "market", {"symbol": symbol, "mode": "Replay", "as_of": new_candles[-1]["time"],
                                       "appended_observations": appended, "source": "bundled_daily"})


def seed_database():
    ensure_demo_data()
    db.init_db()
    with db.session_scope() as session:
        if session.bind.dialect.name == "postgresql":
            session.execute(text("SELECT pg_advisory_xact_lock(81263941)"))
        else:
            session.execute(text("BEGIN IMMEDIATE"))
        if session.get(db.DataSource, "bundled_daily") is None:
            info = data_source_info()
            session.add(db.DataSource(id="bundled_daily", name=info.get("name", "Bundled daily history"), mode="Replay", metadata_json=info))
        instruments = list_instruments()
        for instrument in instruments:
            if session.get(db.Instrument, instrument["symbol"]) is None:
                session.add(db.Instrument(**{key: instrument[key] for key in ("symbol", "name", "exchange", "timezone")}))
        session.flush()
        import_new_observations(session)
        session.flush()
        if session.get(db.Watchlist, "local") is None:
            session.add(db.Watchlist(id="local", symbols=[item["symbol"] for item in instruments]))
        register_models(session)
        reconcile_outcomes(session)


def maintain_audit():
    with db.session_scope() as session:
        import_new_observations(session, emit_events=True)
        reconcile_outcomes(session)
        latest = session.scalar(select(func.max(db.StreamEvent.id))) or 0
        if latest > EVENT_RETENTION:
            session.execute(delete(db.StreamEvent).where(db.StreamEvent.id <= latest - EVENT_RETENTION))


async def maintenance_loop():
    while True:
        await asyncio.sleep(30)
        try:
            await run_in_threadpool(maintain_audit)
        except Exception:
            logger.exception("Audit maintenance failed")


@asynccontextmanager
async def lifespan(_app):
    await run_in_threadpool(seed_database)
    task = asyncio.create_task(maintenance_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="NeuroStock", version="1.0.0", lifespan=lifespan,
              description="Local research dashboard. Historical Replay is never represented as live market data.")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST", "PUT"], allow_headers=["Content-Type"])


@contextmanager
def expensive_operation(request: Request):
    client = request.client.host if request.client else "local"
    now = time.monotonic()
    with _RATE_LOCK:
        # Bounded in-memory rate accounting for the intentionally local API.
        if len(_RATE_BUCKETS) > 1000:
            _RATE_BUCKETS.clear()
        queue = _RATE_BUCKETS[client]
        while queue and now - queue[0] > 60:
            queue.popleft()
        if len(queue) >= 30:
            raise HTTPException(429, "At most 30 inference/evaluation operations per minute", headers={"Retry-After": "60"})
        queue.append(now)
    if not _INFERENCE_SLOTS.acquire(blocking=False):
        raise HTTPException(429, "Two inference/evaluation operations are already running", headers={"Retry-After": "2"})
    try:
        yield
    finally:
        _INFERENCE_SLOTS.release()


@app.get("/api/health")
@app.get("/health", include_in_schema=False)
def health():
    with db.session_scope() as session:
        session.execute(select(1))
    return {"status": "ok", "mode": "Replay", "training_execution": "separate worker process"}


@app.get("/api/instruments")
def instruments():
    with db.session_scope() as session:
        return [{"symbol": row.symbol, "name": row.name, "exchange": row.exchange, "timezone": row.timezone}
                for row in session.scalars(select(db.Instrument).order_by(db.Instrument.symbol))]


@app.get("/api/market/{symbol}")
def market(symbol: str, limit: int = Query(260, ge=1, le=2000)):
    with db.session_scope() as session:
        _require_instrument(session, symbol)
        try:
            frame, _ = verified_source_frame(session, symbol)
        except SourceRevisionError as error:
            raise HTTPException(409, str(error)) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
    candles = frame_records(frame.tail(limit))
    return {"symbol": symbol, "candles": candles, "source": data_source_info(symbol)["name"],
            "source_metadata": data_source_info(symbol), "mode": "Replay", "delay_seconds": None,
            "timezone": "America/New_York", "session_status": "Historical replay",
            "as_of": candles[-1]["time"] if candles else None}


@app.get("/api/models")
def models():
    with db.session_scope() as session:
        return [public_metadata(item) for item in register_models(session)]


@app.post("/api/forecasts", status_code=201)
def create_forecast(body: ForecastRequest, request: Request):
    from .ml import predict
    with expensive_operation(request), db.session_scope() as session:
        _require_instrument(session, body.symbol)
        _require_model(session, body.model_id, body.symbol)
        try:
            frame, _ = verified_source_frame(session, body.symbol)
        except SourceRevisionError as error:
            raise HTTPException(409, str(error)) from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        cutoff = body.cutoff.isoformat() if body.cutoff else str(frame["date"].iloc[-1])[:10]
        frame = frame.loc[frame["date"].astype(str).str[:10] <= cutoff].copy()
        if frame.empty:
            raise HTTPException(422, "No completed observations exist before this cutoff")
        candles = frame_records(frame)
        snapshot_id = fresh_id()
        digest = hashlib.sha256(json.dumps(candles, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        try:
            output = predict(body.model_id, body.symbol, cutoff=cutoff, record_telemetry=True, override_frame=frame)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(422, str(error)) from None
        telemetry = output.pop("telemetry", None)
        if not telemetry:
            raise HTTPException(500, "Inference completed without replay telemetry; no forecast was stored")
        # Actuals belong to a separate immutable entity, matched only by target session.
        output.pop("actual", None)
        output.pop("outcome", None)
        output.pop("changes", None)
        forecast_id = fresh_id()
        output.update({"id": forecast_id, "symbol": body.symbol, "model_id": body.model_id,
                       "model_version": body.model_id, "issue_time": db.utcnow(),
                       "data_snapshot_id": snapshot_id, "data_snapshot_sha256": digest, "mode": "Replay"})
        session.add(db.DataSnapshot(id=snapshot_id, symbol=body.symbol, input_cutoff=output["input_cutoff"],
                                    digest=digest, candles=candles, source_id="bundled_daily"))
        session.flush()
        row = db.Forecast(id=forecast_id, symbol=body.symbol, model_id=body.model_id,
                          issue_time=output["issue_time"], target_time=output["target_time"],
                          input_cutoff=output["input_cutoff"], data_snapshot_id=snapshot_id, payload=output)
        session.add(row)
        session.flush()
        session.add(db.TelemetrySnapshot(id=fresh_id(), forecast_id=forecast_id, payload=telemetry))
        reconcile_outcomes(session)
        session.flush()
        result = serialize_forecast(session, row)
        db.emit(session, "inference", {"forecast_id": forecast_id, "symbol": body.symbol,
                                       "model_id": body.model_id, "mode": "Replay", "target_time": row.target_time})
        return {**result, "telemetry": telemetry}


@app.get("/api/forecasts")
def forecasts(symbol: str | None = None, limit: int = Query(100, ge=1, le=500)):
    with db.session_scope() as session:
        query = select(db.Forecast).order_by(db.Forecast.issue_time.desc()).limit(limit)
        if symbol:
            _require_instrument(session, symbol)
            query = query.where(db.Forecast.symbol == symbol)
        return [serialize_forecast(session, row) for row in session.scalars(query)]


@app.get("/api/forecasts/{forecast_id}")
def get_forecast(forecast_id: str):
    with db.session_scope() as session:
        row = session.get(db.Forecast, forecast_id)
        if row is None:
            raise HTTPException(404, "Forecast was not found")
        return serialize_forecast(session, row)


@app.get("/api/forecasts/{forecast_id}/telemetry")
def get_telemetry(forecast_id: str):
    with db.session_scope() as session:
        row = session.scalar(select(db.TelemetrySnapshot).where(db.TelemetrySnapshot.forecast_id == forecast_id))
        if row is None:
            raise HTTPException(404, "Replay telemetry was not found")
        return row.payload


@app.post("/api/what-if")
def what_if(body: WhatIfRequest, request: Request):
    from .ml import predict
    with expensive_operation(request), db.session_scope() as session:
        forecast = session.get(db.Forecast, body.forecast_id)
        if forecast is None:
            raise HTTPException(404, "Forecast was not found")
        _require_model(session, forecast.model_id, forecast.symbol)
        snapshot = session.get(db.DataSnapshot, forecast.data_snapshot_id)
        frame = pd.DataFrame(snapshot.candles).rename(columns={"time": "date"})
        try:
            output = predict(forecast.model_id, forecast.symbol, cutoff=forecast.input_cutoff,
                             perturbation={"feature": body.feature, "percent": body.percent},
                             record_telemetry=True, override_frame=frame)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(422, str(error)) from None
        output.pop("actual", None)
        changes = output.pop("changes", [])
        output.update({"mode": "Replay", "sensitivity_only": True})
        return {"original": serialize_forecast(session, forecast), "perturbed": output,
                "changes": changes, "sensitivity_only": True,
                "description": "Sensitivity to changed input values; not a causal market prediction. Original forecast remains unchanged."}


@app.get("/api/evaluations")
def evaluations(request: Request, model_id: str = Query(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")):
    from .ml import evaluate_model
    with db.session_scope() as session:
        _require_model(session, model_id)
        saved = session.scalar(select(db.EvaluationRun).where(db.EvaluationRun.model_id == model_id))
        if saved:
            return saved.payload
    with expensive_operation(request):
        try:
            result = evaluate_model(model_id)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(422, str(error)) from None
    try:
        with db.session_scope() as session:
            session.add(db.EvaluationRun(id=fresh_id(), model_id=model_id, payload=result))
    except IntegrityError:
        # Identical concurrent requests converge on the already persisted run.
        with db.session_scope() as session:
            result = session.scalar(select(db.EvaluationRun).where(db.EvaluationRun.model_id == model_id)).payload
    return result


@app.post("/api/training", status_code=202)
def submit_training(body: TrainingRequest):
    with db.session_scope() as session:
        _require_instrument(session, body.symbol)
        job_id = fresh_id()
        reserved = session.execute(update(db.TrainingSlot).where(
            db.TrainingSlot.id == 1, db.TrainingSlot.job_id.is_(None)).values(job_id=job_id))
        if reserved.rowcount != 1:
            raise HTTPException(409, "One training job is already queued or running; wait for it to finish")
        job = db.TrainingJob(id=job_id, status="queued", config=body.model_dump(), history=[],
                             created_at=db.utcnow(), updated_at=db.utcnow())
        session.add(job)
        session.flush()
        result = db.job_dict(job)
        db.emit(session, "training", result)
        return result


@app.get("/api/training")
def training_jobs():
    with db.session_scope() as session:
        return [db.job_dict(row) for row in session.scalars(select(db.TrainingJob).order_by(db.TrainingJob.created_at.desc()).limit(100))]


@app.get("/api/training/{job_id}")
def training_job(job_id: str):
    with db.session_scope() as session:
        job = session.get(db.TrainingJob, job_id)
        if job is None:
            raise HTTPException(404, "Training job was not found")
        return db.job_dict(job)


@app.get("/api/watchlist")
def watchlist():
    with db.session_scope() as session:
        return session.get(db.Watchlist, "local").symbols


@app.put("/api/watchlist")
def update_watchlist(body: WatchlistRequest):
    with db.session_scope() as session:
        for symbol in body.symbols:
            _require_instrument(session, symbol)
        row = session.get(db.Watchlist, "local")
        row.symbols = body.symbols
        return body.symbols


@app.get("/api/provider/status")
def provider_status():
    return {"active": replay_provider.status(), "external": external_provider.status()}


@app.get("/api/provider/quote/{symbol}")
def external_quote(symbol: str):
    with db.session_scope() as session:
        _require_instrument(session, symbol)
    try:
        result = external_provider.quote(symbol)
    except ProviderUnavailable as error:
        raise HTTPException(503, {"mode": "Disconnected", "message": str(error), "replay_available": True}) from None
    with db.session_scope() as session:
        db.emit(session, "market", result)
    return result


def stream_bounds():
    with db.session_scope() as session:
        return session.execute(select(func.min(db.StreamEvent.id), func.max(db.StreamEvent.id))).one()


def stream_batch(cursor):
    with db.session_scope() as session:
        return [db.event_dict(row) for row in session.scalars(select(db.StreamEvent).where(
            db.StreamEvent.id > cursor).order_by(db.StreamEvent.id).limit(100))]


@app.websocket("/ws")
async def event_stream(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if origin and origin not in CORS_ORIGINS:
        await websocket.close(code=1008, reason="Origin is not allowed")
        return
    raw_cursor = websocket.query_params.get("after", "0")
    if not raw_cursor.isdigit() or len(raw_cursor) > 18:
        await websocket.close(code=1008, reason="after must be an event identifier")
        return
    cursor = int(raw_cursor)
    await websocket.accept()
    oldest, latest = await run_in_threadpool(stream_bounds)
    latest = latest or 0
    gap = bool(oldest and cursor < oldest - 1)
    reset = cursor > latest
    if reset:
        cursor = 0
    await websocket.send_json({"id": f"connection-{fresh_id()}", "type": "connected", "timestamp": db.utcnow(),
                               "payload": {"mode": "Replay", "after": cursor, "latest_event_id": str(latest),
                                           "replay_gap": gap, "cursor_reset": reset,
                                           "message": "Refresh current API state if replay_gap or cursor_reset is true."}})
    try:
        while True:
            events = await run_in_threadpool(stream_batch, cursor)
            for event in events:
                await websocket.send_json(event)
                cursor = int(event["id"])
            # Receive wait detects close promptly without inventing market ticks.
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=0.4)
                if message["type"] == "websocket.disconnect":
                    break
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, RuntimeError):
        return

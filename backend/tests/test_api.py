"""Integration tests use real trained CPU models and isolated SQLite audit stores."""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, update
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import sessionmaker

from app import db, main, ml
from app.data import load_frame, next_session


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    original = ml.ARTIFACT_DIR
    ml.ARTIFACT_DIR = tmp_path_factory.mktemp("api-models")
    try:
        yield ml.train_model(symbol="AAPL", kind="lstm", lookback=12, epochs=2, max_samples=400)
    finally:
        ml.ARTIFACT_DIR = original


@pytest.fixture
def client(tmp_path, monkeypatch, trained_model):
    engine = db.build_engine(f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setattr(db, "engine", engine)
    monkeypatch.setattr(db, "SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))
    main._RATE_BUCKETS.clear()
    with TestClient(main.app) as client:
        client.database_url = str(engine.url)
        yield client
    engine.dispose()


def generate(client, trained_model):
    response = client.post("/api/forecasts", json={"symbol": "AAPL", "model_id": trained_model["id"],
                                                  "cutoff": trained_model["validation_cutoff"]})
    assert response.status_code == 201, response.text
    return response.json()


def test_real_forecast_telemetry_snapshot_and_exact_outcome(client, trained_model):
    forecast = generate(client, trained_model)
    assert forecast["mode"] == "Replay"
    assert forecast["target_time"] == next_session(forecast["input_cutoff"])
    assert forecast["lower"] <= forecast["median"] <= forecast["upper"]
    assert forecast["outcome"]["historical_outcome"] is True
    frame = load_frame("AAPL")
    target = frame.loc[frame.date == forecast["target_time"], "close"].iloc[0]
    assert forecast["actual"] == target
    telemetry = client.get(f"/api/forecasts/{forecast['id']}/telemetry").json()
    assert telemetry == forecast["telemetry"]
    assert telemetry["output"]["prices"]["median"] == forecast["median"]
    assert len(telemetry["hidden"]) == trained_model["lookback"]
    assert set(telemetry["gates"]) == {"input", "forget", "candidate", "output"}
    assert np.isfinite(np.array(telemetry["input_normalized"])).all()
    again = client.get(f"/api/forecasts/{forecast['id']}").json()
    assert again["issue_time"] == forecast["issue_time"]
    assert again["median"] == forecast["median"]
    with db.session_scope() as session:
        snapshot = session.get(db.DataSnapshot, forecast["data_snapshot_id"])
        assert snapshot.candles[-1]["time"] == forecast["input_cutoff"]
        assert snapshot.digest == forecast["data_snapshot_sha256"]
        assert len(snapshot.candles) >= trained_model["lookback"]
        assert session.scalar(select(db.PreprocessingArtifact).where(db.PreprocessingArtifact.model_id == trained_model["id"]))


def test_stored_replay_and_zero_perturbation_use_original_snapshot(client, trained_model, monkeypatch):
    original = generate(client, trained_model)
    # Subsequent replay and what-if must not reload mutable historical files.
    def no_current_data(*_args, **_kwargs):
        raise AssertionError("Current market data must not replace a stored input snapshot")
    monkeypatch.setattr(ml, "load_frame", no_current_data)
    replay = client.get(f"/api/forecasts/{original['id']}/telemetry")
    assert replay.json() == original["telemetry"]
    result = client.post("/api/what-if", json={"forecast_id": original["id"], "feature": "close", "percent": 0})
    assert result.status_code == 200, result.text
    payload = result.json()
    assert payload["sensitivity_only"] is True
    assert payload["changes"] == []
    assert payload["perturbed"]["median"] == original["median"]
    assert payload["perturbed"]["telemetry"]["hidden"] == original["telemetry"]["hidden"]


def test_what_if_ohlc_constraints_exact_changes_and_immutability(client, trained_model):
    original = generate(client, trained_model)
    result = client.post("/api/what-if", json={"forecast_id": original["id"], "feature": "close", "percent": 20})
    assert result.status_code == 200, result.text
    payload = result.json()
    changes = {change["feature"]: change for change in payload["changes"]}
    assert changes["close"]["changed"] == pytest.approx(changes["close"]["original"] * 1.2)
    final_candle = payload["perturbed"]["telemetry"]["input_candles"][-1]
    assert final_candle["high"] >= max(final_candle["close"], final_candle["open"])
    assert final_candle["low"] <= min(final_candle["close"], final_candle["open"])
    assert client.get(f"/api/forecasts/{original['id']}").json()["median"] == original["median"]
    assert len(client.get("/api/forecasts").json()) == 1


@pytest.mark.parametrize("payload", [
    {"feature": "open", "percent": 1}, {"feature": "close", "percent": 20.1},
    {"feature": "volume", "percent": -21}, {"feature": "volume", "percent": True},
    {"feature": "close", "percent": "nan"}, {"feature": "close", "percent": 1, "arbitrary_path": "/tmp/model"},
])
def test_what_if_validation(client, payload):
    assert client.post("/api/what-if", json={"forecast_id": "unknown", **payload}).status_code == 422


def test_forecast_rejects_unknown_model_symbol_and_leaky_cutoff(client, trained_model):
    assert client.post("/api/forecasts", json={"symbol": "AAPL", "model_id": "../../weights"}).status_code == 422
    assert client.post("/api/forecasts", json={"symbol": "AAPL", "model_id": "missing"}).status_code == 404
    assert client.post("/api/forecasts", json={"symbol": "MSFT", "model_id": trained_model["id"]}).status_code == 422
    assert client.post("/api/forecasts", json={"symbol": "AAPL", "model_id": trained_model["id"], "cutoff": "2015-02-18"}).status_code == 422


def test_database_rejects_mutation_and_deletion_of_forecast(client, trained_model):
    forecast = generate(client, trained_model)
    for statement in (update(db.Forecast).where(db.Forecast.id == forecast["id"]).values(payload={"tampered": True}),
                      delete(db.Forecast).where(db.Forecast.id == forecast["id"])):
        with pytest.raises(DatabaseError, match="immutable audit record"):
            with db.session_scope() as session:
                session.execute(statement)
    assert client.get(f"/api/forecasts/{forecast['id']}").json()["median"] == forecast["median"]


def test_outcome_arrival_preserves_forecast_and_matches_session(client, trained_model):
    target = next_session(trained_model["validation_cutoff"])
    with db.session_scope() as session:
        observed = session.scalar(select(db.Observation).where(db.Observation.symbol == "AAPL", db.Observation.time == target))
        candle = dict(observed.values)
        session.delete(observed)
    forecast = generate(client, trained_model)
    assert "actual" not in forecast
    # The maintenance pass discovers the completed row in the source file,
    # imports it, and attaches the outcome without rewriting the forecast.
    main.maintain_audit()
    resolved = client.get(f"/api/forecasts/{forecast['id']}").json()
    assert resolved["actual"] == candle["close"]
    assert resolved["median"] == forecast["median"]
    main.maintain_audit()
    with db.session_scope() as session:
        assert len(session.scalars(select(db.ForecastOutcome)).all()) == 1


def test_websocket_reconnect_replays_only_unseen_persisted_events(client):
    with client.websocket_connect("/ws?after=0") as socket:
        assert socket.receive_json()["type"] == "connected"
        with db.session_scope() as session:
            db.emit(session, "market", {"symbol": "AAPL", "mode": "Replay"})
        first = socket.receive_json()
        assert first["type"] == "market"
    with db.session_scope() as session:
        db.emit(session, "market", {"symbol": "MSFT", "mode": "Replay"})
    with client.websocket_connect(f"/ws?after={first['id']}") as socket:
        assert socket.receive_json()["type"] == "connected"
        second = socket.receive_json()
        assert int(second["id"]) > int(first["id"])
        assert second["payload"]["symbol"] == "MSFT"
        assert second["timestamp"]


def test_market_reads_do_not_emit_recursive_update_events(client):
    client.get("/api/market/AAPL")
    client.get("/api/market/AAPL")
    with db.session_scope() as session:
        assert session.scalars(select(db.StreamEvent)).all() == []


def test_revised_source_is_rejected_without_rewriting_audit(client, trained_model, monkeypatch):
    forecast = generate(client, trained_model)
    original_load = main.load_frame
    def revised_frame(symbol):
        frame = original_load(symbol)
        frame.loc[0, "volume"] += 1
        return frame
    monkeypatch.setattr(main, "load_frame", revised_frame)
    assert client.get("/api/market/AAPL").status_code == 409
    rejected = client.post("/api/forecasts", json={"symbol": "AAPL", "model_id": trained_model["id"]})
    assert rejected.status_code == 409
    assert "source revision" in rejected.json()["detail"].lower()
    assert client.get(f"/api/forecasts/{forecast['id']}").json()["median"] == forecast["median"]


def test_websocket_reports_retention_gap(client, monkeypatch):
    with db.session_scope() as session:
        for index in range(5):
            db.emit(session, "market", {"sequence": index})
    monkeypatch.setattr(main, "EVENT_RETENTION", 2)
    main.maintain_audit()
    with client.websocket_connect("/ws?after=1") as socket:
        connection = socket.receive_json()
        assert connection["payload"]["replay_gap"] is True
        assert int(socket.receive_json()["id"]) == 4


def test_watchlist_source_and_no_credential_provider(client, monkeypatch):
    assert len(client.get("/api/instruments").json()) == 2
    market = client.get("/api/market/AAPL?limit=10").json()
    assert len(market["candles"]) == 10
    assert market["mode"] == "Replay"
    assert market["delay_seconds"] is None
    assert market["source_metadata"]["price_convention"]
    assert client.put("/api/watchlist", json={"symbols": ["MSFT"]}).json() == ["MSFT"]
    assert client.get("/api/watchlist").json() == ["MSFT"]
    assert client.put("/api/watchlist", json={"symbols": ["MSFT", "MSFT"]}).status_code == 422
    assert client.put("/api/watchlist", json={"symbols": ["NOPE"]}).status_code == 404
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    response = client.get("/api/provider/quote/AAPL")
    assert response.status_code == 503
    assert response.json()["detail"]["replay_available"] is True
    assert response.json()["detail"]["mode"] == "Disconnected"


def test_same_test_dates_baseline_evaluation_is_persisted(client, trained_model):
    response = client.get("/api/evaluations", params={"model_id": trained_model["id"]})
    assert response.status_code == 200
    result = response.json()
    assert result["split"] == "historical_test"
    assert result["metrics"]["count"] == result["baseline"]["count"] == len(result["rows"])
    assert result["baseline"]["mae"] == pytest.approx(np.mean([abs(row["base_price"] - row["actual"]) for row in result["rows"]]))
    assert len(result["walk_forward"]) == 3
    with db.session_scope() as session:
        assert session.scalar(select(db.EvaluationRun).where(db.EvaluationRun.model_id == trained_model["id"]))


def test_single_training_reservation_and_real_separate_worker(client, trained_model):
    config = {"symbol": "AAPL", "kind": "gru", "lookback": 10, "epochs": 1, "max_samples": 200}
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda _: client.post("/api/training", json=config), range(2)))
    assert sorted(response.status_code for response in responses) == [202, 409]
    job = next(response.json() for response in responses if response.status_code == 202)
    assert job["status"] == "queued"
    env = {**os.environ, "DATABASE_URL": client.database_url, "ARTIFACT_DIR": str(ml.ARTIFACT_DIR),
           "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    process = subprocess.run([sys.executable, "-c", "from app.worker import run_next_job; assert run_next_job()"],
                             env=env, capture_output=True, text=True, timeout=90)
    assert process.returncode == 0, process.stderr
    finished = client.get(f"/api/training/{job['id']}").json()
    assert finished["status"] == "completed", finished
    assert finished["model_id"]
    assert finished["history"]
    assert finished["progress"]["epoch"] == 1
    assert finished["progress"]["gradient_norm"] >= 0
    assert client.post("/api/training", json=config).status_code == 202
    assert client.post("/api/training", json={**config, "epochs": 21}).status_code == 422


def test_training_failure_is_persisted_and_releases_reservation(client, monkeypatch):
    from app import worker
    config = {"symbol": "AAPL", "kind": "lstm", "epochs": 1, "max_samples": 200}
    job = client.post("/api/training", json=config).json()
    def fail_training(**_):
        raise ValueError("Deliberate test failure")
    monkeypatch.setattr(ml, "train_model", fail_training)
    assert worker.run_next_job()
    failed = client.get(f"/api/training/{job['id']}").json()
    assert failed["status"] == "failed"
    assert "Deliberate test failure" in failed["error"]
    assert client.post("/api/training", json=config).status_code == 202

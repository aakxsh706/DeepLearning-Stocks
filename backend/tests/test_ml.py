import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from app import ml
from app.data import FEATURE_NAMES, load_frame, make_features, next_session, snapshot_id, validate_frame
from app.models import InstrumentedGRUCell, InstrumentedLSTMCell, QuantileRNN


def test_instrumented_lstm_matches_pytorch_sequence():
    torch.manual_seed(71)
    batch, steps, features, hidden = 3, 11, 8, 7
    cell = InstrumentedLSTMCell(features, hidden).double()
    standard = torch.nn.LSTM(features, hidden, batch_first=True).double()
    with torch.no_grad():
        for name, parameter in cell.named_parameters():
            getattr(standard, name + "_l0").copy_(parameter)
    x = torch.randn(batch, steps, features, dtype=torch.float64)
    h0 = torch.randn(batch, hidden, dtype=torch.float64)
    c0 = torch.randn(batch, hidden, dtype=torch.float64)
    expected, (_, final_cell) = standard(x, (h0[None], c0[None]))
    h, c, values = h0, c0, []
    for step in x.unbind(1):
        h, c, gates = cell(step, (h, c))
        values.append(h)
        assert torch.all((gates["forget"] >= 0) & (gates["forget"] <= 1))
    torch.testing.assert_close(torch.stack(values, 1), expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(c, final_cell[0], atol=1e-12, rtol=1e-12)


def test_instrumented_gru_matches_pytorch():
    torch.manual_seed(123)
    actual = InstrumentedGRUCell(8, 6).double()
    expected = torch.nn.GRUCell(8, 6).double()
    expected.load_state_dict(actual.state_dict())
    x, h = torch.randn(4, 8, dtype=torch.float64), torch.randn(4, 6, dtype=torch.float64)
    output, _ = actual(x, h)
    torch.testing.assert_close(output, expected(x, h), atol=1e-12, rtol=1e-12)


def test_chronological_windows_and_training_only_scaler():
    frame = load_frame("AAPL")
    data = ml.prepare_data(frame, lookback=30)
    targets, masks = data["targets"], data["masks"]
    assert targets[masks["train"]].max() < targets[masks["validation"]].min()
    assert targets[masks["validation"]].max() < targets[masks["test"]].min()
    np.testing.assert_allclose(data["scaler"].mean_, data["features"].iloc[10:data["train_end"] - 1].mean())
    target = targets[0]
    normalized = data["scaler"].transform(data["features"].iloc[target - 30:target].to_numpy())
    np.testing.assert_allclose(data["x"][0], normalized, atol=1e-6)
    assert data["y"][0] == pytest.approx(frame.close.iloc[target] / frame.close.iloc[target - 1] - 1)
    # Poison future prices and volumes: all training inputs, labels, and fitted stats stay fixed.
    poisoned = frame.copy()
    columns = ["open", "high", "low", "close", "volume"]
    poisoned.loc[data["train_end"]:, columns] *= 3
    other = ml.prepare_data(poisoned, lookback=30)
    np.testing.assert_array_equal(data["scaler"].mean_, other["scaler"].mean_)
    np.testing.assert_array_equal(data["x"][masks["train"]], other["x"][masks["train"]])
    np.testing.assert_array_equal(data["y"][masks["train"]], other["y"][masks["train"]])


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    original = ml.ARTIFACT_DIR
    ml.ARTIFACT_DIR = tmp_path_factory.mktemp("models")
    progress = []
    metadata = ml.train_model("AAPL", "lstm", lookback=10, epochs=2, max_samples=200, progress_callback=progress.append)
    yield metadata, progress
    ml.ARTIFACT_DIR = original


def test_forecast_telemetry_replay_and_output_head(artifact):
    metadata, progress = artifact
    cutoff = metadata["splits"]["test"]["start"]
    a = ml.predict(metadata["id"], "AAPL", cutoff)
    b = ml.predict(metadata["id"], "AAPL", cutoff)
    assert a["telemetry"] == b["telemetry"]
    assert a["data_snapshot_id"] == b["data_snapshot_id"]
    assert a["lower"] <= a["median"] <= a["upper"]
    for q in ("lower", "median", "upper"):
        assert a[q] == a["base_price"] * (1 + a["returns"][q])
        assert a[q] == a["telemetry"]["output"]["prices"][q]
    t = a["telemetry"]
    assert len(t["hidden"]) == metadata["lookback"]
    raw = np.array(t["head"]["weight"]) @ np.array(t["hidden"][-1]) + np.array(t["head"]["bias"])
    np.testing.assert_allclose(raw, t["head"]["raw"], atol=1e-7)
    assert raw[0] / 100 == pytest.approx(a["returns"]["median"], abs=1e-8)
    assert a["target_time"] == next_session(a["input_cutoff"])
    assert a["actual"] == load_frame("AAPL").set_index("date").loc[a["target_time"], "close"]
    assert len(progress) == 2 and all(p["gradient_norm"] > 0 for p in progress)
    assert all(sum(p["weight_histogram"]["counts"]) > 0 for p in progress)


def test_what_if_recomputes_and_preserves_ohlc(artifact):
    metadata, _ = artifact
    original = ml.predict(metadata["id"], "AAPL")
    for percent in (-20, 20):
        changed = ml.predict(metadata["id"], "AAPL", perturbation={"feature": "close", "percent": percent})
        candle = changed["telemetry"]["input_candles"][-1]
        assert candle["low"] <= min(candle["open"], candle["close"]) <= max(candle["open"], candle["close"]) <= candle["high"]
        assert changed["base_price"] == pytest.approx(original["base_price"] * (1 + percent / 100))
        assert changed["telemetry"]["input_raw"][-1] != original["telemetry"]["input_raw"][-1]
        assert changed["sensitivity_only"] and changed["actual"] is None
        assert changed["changes"][0]["original"] != changed["changes"][0]["changed"]
    for bad in (float("nan"), 21, True):
        with pytest.raises(ValueError):
            ml.predict(metadata["id"], "AAPL", perturbation={"feature": "volume", "percent": bad})


def test_snapshot_replay_uses_original_data_and_cannot_leak(artifact):
    metadata, _ = artifact
    frame = load_frame("AAPL")
    cutoff = metadata["splits"]["test"]["start"]
    snapshot = frame[frame.date <= cutoff]
    original = ml.predict(metadata["id"], "AAPL", cutoff)
    replay = ml.predict(metadata["id"], "AAPL", cutoff, override_frame=snapshot)
    assert original["telemetry"] == replay["telemetry"]
    assert original["data_snapshot_id"] == snapshot_id(snapshot)
    with pytest.raises(ValueError, match="model-selection"):
        ml.predict(metadata["id"], "AAPL", metadata["training_cutoff"])
    with pytest.raises(ValueError):
        ml.predict("../../malicious", "AAPL")
    with pytest.raises(ValueError):
        ml.predict(metadata["id"], "MSFT")


def test_baseline_and_walk_forward_use_identical_test_periods(artifact):
    metadata, _ = artifact
    evaluation = ml.evaluate_model(metadata["id"])
    rows = evaluation["rows"]
    baseline_error = np.array([row["base_price"] - row["actual"] for row in rows])
    assert evaluation["baseline"]["mae"] == pytest.approx(np.abs(baseline_error).mean())
    assert evaluation["baseline"]["rmse"] == pytest.approx(np.sqrt(np.square(baseline_error).mean()))
    assert sum(f["count"] for f in evaluation["walk_forward"]) == len(rows)
    assert rows[0]["date"] > metadata["validation_cutoff"]
    assert all(row["input_cutoff"] < row["date"] for row in rows)


def test_serving_matches_training_pipeline_and_saved_evaluation(artifact):
    metadata, _ = artifact
    prepared = ml.prepare_data(load_frame("AAPL"), metadata["lookback"], metadata["max_samples"])
    evaluation = ml.evaluate_model(metadata["id"])
    test_x = prepared["x"][prepared["masks"]["test"]]
    for index in (0, len(test_x) // 2, len(test_x) - 1):
        row = evaluation["rows"][index]
        forecast = ml.predict(metadata["id"], "AAPL", row["input_cutoff"])
        np.testing.assert_array_equal(forecast["telemetry"]["input_normalized"], test_x[index])
        for quantile in ("lower", "median", "upper"):
            # Batched and single-example float32 GEMMs may differ by a few ulps.
            assert forecast[quantile] == pytest.approx(row[quantile], abs=2e-6)
        assert forecast["actual"] == row["actual"]


def test_inference_does_not_read_future_features(artifact):
    metadata, _ = artifact
    frame = load_frame("AAPL")
    cutoff = metadata["validation_cutoff"]
    original = ml.predict(metadata["id"], "AAPL", cutoff, override_frame=frame)
    future = frame.date > cutoff
    frame.loc[future, ["open", "high", "low", "close", "volume"]] *= 5
    changed = ml.predict(metadata["id"], "AAPL", cutoff, override_frame=frame)
    assert original["telemetry"] == changed["telemetry"]
    assert original["data_snapshot_id"] == changed["data_snapshot_id"]
    assert changed["actual"] == pytest.approx(original["actual"] * 5)


def test_evaluation_corruption_is_detected(artifact):
    metadata, _ = artifact
    path = ml.ARTIFACT_DIR / metadata["id"] / "evaluation.json"
    original = path.read_bytes()
    try:
        path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="Evaluation artifact integrity"):
            ml.evaluate_model(metadata["id"])
    finally:
        path.write_bytes(original)


def test_missing_and_nontrading_session_rows_are_rejected():
    frame = load_frame("AAPL")
    with pytest.raises(ValueError, match="consecutive exchange sessions"):
        validate_frame(frame.drop(index=30))
    frame.loc[4, "date"] = "2015-02-21"  # Saturday, replacing Monday.
    with pytest.raises(ValueError, match="consecutive exchange sessions"):
        validate_frame(frame)


@pytest.mark.parametrize("day,expected", [("2017-02-17", "2017-02-21"), ("2017-04-13", "2017-04-17"),
                                          ("2025-06-18", "2025-06-20"), ("2012-10-26", "2012-10-31")])
def test_exchange_calendar_holidays_and_exceptional_closure(day, expected):
    assert next_session(day) == expected

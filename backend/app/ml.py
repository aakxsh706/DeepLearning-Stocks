"""Bounded reproducible training, immutable model versions, and recorded inference."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_NAMES, ROOT, load_frame, make_features, next_session, snapshot_id, validate_frame
from .models import QuantileRNN, quantile_loss

ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", ROOT / "artifacts"))
MODEL_ID = re.compile(r"^(lstm|gru)-[A-Z]{1,8}-[a-f0-9]{12}$")
SCHEMA_VERSION = 2
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def prepare_data(frame, lookback=30, max_samples=1200):
    """Split by target date first; overlapping past context is not a future label."""
    frame = validate_frame(frame).tail(max_samples).reset_index(drop=True)
    if len(frame) < 160 or not 10 <= lookback <= 60:
        raise ValueError("At least 160 observations and lookback 10–60 required")
    features = make_features(frame)
    first = int(np.flatnonzero(np.isfinite(features.to_numpy()).all(axis=1))[0])
    train_end, validation_end = int(len(frame) * 0.65), int(len(frame) * 0.80)
    targets = np.arange(first + lookback, len(frame))
    train_targets = targets[targets < train_end]
    if len(train_targets) < 20:
        raise ValueError("Too few training windows for this lookback")
    # Every scaler row is used by at least one training input, never a future target.
    scaler_rows = features.iloc[first:train_end - 1].to_numpy()
    scaler = StandardScaler().fit(scaler_rows)
    values = (features.to_numpy() - scaler.mean_) / scaler.scale_
    x = np.stack([values[t - lookback:t] for t in targets]).astype(np.float32)
    y = np.array([frame.close.iloc[t] / frame.close.iloc[t - 1] - 1 for t in targets], dtype=np.float32)
    masks = {"train": targets < train_end,
             "validation": (targets >= train_end) & (targets < validation_end),
             "test": targets >= validation_end}
    return {"frame": frame, "features": features, "x": x, "y": y, "targets": targets,
            "masks": masks, "scaler": scaler, "scaler_rows": scaler_rows,
            "train_end": train_end, "validation_end": validation_end, "first_feature": first}


def metric_summary(actual, median, lower, upper, base):
    actual, median, lower, upper, base = [np.asarray(a, dtype=float) for a in (actual, median, lower, upper, base)]
    error = median - actual
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
            "directional_accuracy": float((np.sign(median - base) == np.sign(actual - base)).mean()),
            "coverage": float(((actual >= lower) & (actual <= upper)).mean()),
            "interval_width": float((upper - lower).mean()), "count": len(actual)}


def _evaluate(model, prepared, model_id):
    mask = prepared["masks"]["test"]
    targets, frame = prepared["targets"][mask], prepared["frame"]
    with torch.inference_mode():
        returns, _ = model(torch.from_numpy(prepared["x"][mask]))
    # Match serving's double-precision return-to-price conversion exactly.
    returns = returns.numpy().astype(float)
    base = frame.close.iloc[targets - 1].to_numpy()
    actual = frame.close.iloc[targets].to_numpy()
    prices = base[:, None] * (1 + returns)
    lower, median, upper = prices.T
    metrics = metric_summary(actual, median, lower, upper, base)
    baseline = metric_summary(actual, base, base, base, base)
    baseline = {key: baseline[key] for key in ("mae", "rmse", "directional_accuracy", "count")}
    rows = [{"date": str(frame.date.iloc[t]), "input_cutoff": str(frame.date.iloc[t - 1]),
             "actual": float(a), "median": float(m), "lower": float(l), "upper": float(u),
             "base_price": float(b), "baseline": float(b), "error": float(m - a)}
            for t, a, m, l, u, b in zip(targets, actual, median, lower, upper, base)]
    folds = []
    for i, indices in enumerate(np.array_split(np.arange(len(rows)), 3)):
        if not len(indices):
            continue
        folds.append({"fold": i + 1, "start": rows[int(indices[0])]["date"],
                      "end": rows[int(indices[-1])]["date"], "count": len(indices),
                      "strategy": "Frozen checkpoint, rolling one-session origin; no refit",
                      "metrics": metric_summary(actual[indices], median[indices], lower[indices], upper[indices], base[indices]),
                      "baseline": metric_summary(actual[indices], base[indices], base[indices], base[indices], base[indices])})
    return {"model_id": model_id, "split": "historical_test", "metrics": metrics, "baseline": baseline,
            "rows": rows, "walk_forward": folds,
            "definitions": {
                "mae": "Mean absolute median-price error in USD.",
                "rmse": "Root mean squared median-price error in USD.",
                "directional_accuracy": "Fraction where sign(forecast median − input close) equals sign(actual close − input close). Zero is its own class; persistence predicts zero.",
                "coverage": "Fraction of actual next-session closes within the inclusive 0.10–0.90 estimated quantiles. Nominal target 80%; empirical coverage can differ.",
                "interval_width": "Mean upper minus lower price in USD.",
                "walk_forward": "Sequential one-session test origins with rolling observed context and frozen model/scaler. Three contiguous reporting blocks; no expanding-window retraining.",
                "baseline": "Persistence predicts the last completed close, on exactly the same dates and horizon.",
                "split": "Chronological target-date split 65% train, 15% validation, 20% historical test. No forward-running results are claimed."}}


def train_model(symbol="AAPL", kind="lstm", lookback=30, epochs=8, max_samples=1200, progress_callback=None):
    if kind not in {"lstm", "gru"} or not 10 <= int(lookback) <= 60 or not 1 <= int(epochs) <= 20 or not 200 <= int(max_samples) <= 2000:
        raise ValueError("Training limits: kind lstm/gru, lookback 10–60, epochs 1–20, observations 200–2000")
    start = time.perf_counter()
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    prepared = prepare_data(load_frame(symbol), lookback, max_samples)
    model = QuantileRNN(len(FEATURE_NAMES), 16, kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    masks = prepared["masks"]
    x = {key: torch.from_numpy(prepared["x"][mask]) for key, mask in masks.items()}
    y = {key: torch.from_numpy(prepared["y"][mask]) for key, mask in masks.items()}
    best, best_loss, best_epoch, stale, history = None, float("inf"), 0, 0, []
    previous = torch.cat([p.detach().flatten() for p in model.parameters()])
    for epoch in range(1, epochs + 1):
        model.train()
        losses, gradients = [], []
        for offset in range(0, len(x["train"]), 64):
            xb, yb = x["train"][offset:offset + 64], y["train"][offset:offset + 64]
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(xb)
            loss = quantile_loss(prediction, yb)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append((float(loss.detach()), len(xb)))
            gradients.append(float(gradient))
        model.eval()
        with torch.inference_mode():
            val_prediction, _ = model(x["validation"])
            val_loss = float(quantile_loss(val_prediction, y["validation"]))
        weights = torch.cat([p.detach().flatten() for p in model.parameters()])
        counts, bins = np.histogram(weights.numpy(), bins=12)
        event = {"epoch": epoch, "epochs": epochs,
                 "loss": sum(value * n for value, n in losses) / len(x["train"]),
                 "val_loss": val_loss, "gradient_norm": float(np.mean(gradients)),
                 "learning_rate": optimizer.param_groups[0]["lr"],
                 "elapsed_seconds": time.perf_counter() - start,
                 "weight_histogram": {"bins": bins.tolist(), "counts": counts.tolist()},
                 "weight_change": float(torch.linalg.vector_norm(weights - previous)),
                 "checkpoint": "best" if val_loss < best_loss else "unchanged"}
        previous = weights.clone()
        history.append(event)
        if val_loss < best_loss - 1e-6:
            best, best_loss, best_epoch, stale = copy.deepcopy(model.state_dict()), val_loss, epoch, 0
        else:
            stale += 1
        if progress_callback:
            progress_callback(event)
        if stale >= 4:
            break
    model.load_state_dict(best)
    model.eval()
    model_id = f"{kind}-{symbol}-{uuid.uuid4().hex[:12]}"
    evaluation = _evaluate(model, prepared, model_id)
    frame = prepared["frame"]
    splits = {}
    for split, mask in masks.items():
        indices = prepared["targets"][mask]
        splits[split] = {"start": str(frame.date.iloc[indices[0]]), "end": str(frame.date.iloc[indices[-1]]), "count": len(indices)}
    metadata = {"id": model_id, "version": model_id, "schema_version": SCHEMA_VERSION,
                "kind": kind, "symbol": symbol, "lookback": lookback, "hidden_size": 16,
                "feature_names": FEATURE_NAMES, "quantiles": [0.1, 0.5, 0.9], "seed": seed,
                "created_at": datetime.now(timezone.utc).isoformat(), "horizon": "next trading session",
                "training_cutoff": splits["train"]["end"], "validation_cutoff": splits["validation"]["end"],
                "metrics": evaluation["metrics"], "baseline": evaluation["baseline"], "splits": splits,
                "history": history, "best_epoch": best_epoch, "epochs_completed": len(history),
                "early_stopped": len(history) < epochs, "max_samples": max_samples,
                "data_snapshot_id": snapshot_id(frame), "training_rows": len(frame),
                "normalization": {"mean": prepared["scaler"].mean_.tolist(), "scale": prepared["scaler"].scale_.tolist(),
                                  "fit_start": str(frame.date.iloc[prepared["first_feature"]]),
                                  "fit_end": str(frame.date.iloc[prepared["train_end"] - 2]),
                                  "fit_rows": len(prepared["scaler_rows"])},
                "price_convention": "Source historical OHLCV snapshot; see provenance for adjustment convention",
                "loss_units": "Pinball loss in return percentage points", "status": "ready"}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = ARTIFACT_DIR / (".training-" + model_id)
    temporary.mkdir()
    np.savez_compressed(temporary / "weights.npz", **{key: tensor.detach().numpy() for key, tensor in best.items()})
    metadata["weights_sha256"] = hashlib.sha256((temporary / "weights.npz").read_bytes()).hexdigest()
    _json(temporary / "evaluation.json", evaluation)
    metadata["evaluation_sha256"] = hashlib.sha256((temporary / "evaluation.json").read_bytes()).hexdigest()
    _json(temporary / "metadata.json", metadata)
    temporary.rename(ARTIFACT_DIR / model_id)
    return metadata


def _registry():
    result = {}
    for root in (ROOT / "artifacts" / "demo", ARTIFACT_DIR):
        if not root.is_dir():
            continue
        for directory in sorted(root.iterdir()):
            if directory.is_symlink() or not directory.is_dir() or not MODEL_ID.fullmatch(directory.name):
                continue
            meta_file = directory / "metadata.json"
            if not meta_file.is_file() or meta_file.is_symlink():
                continue
            try:
                metadata = json.loads(meta_file.read_text())
                if metadata.get("id") == directory.name and metadata.get("schema_version") == SCHEMA_VERSION:
                    result[directory.name] = (directory, metadata)
            except (ValueError, OSError):
                continue
    return result


def list_model_artifacts():
    return sorted([metadata for _, metadata in _registry().values()], key=lambda item: (item["symbol"], item["kind"], item["created_at"]), reverse=True)


def _load_model(model_id):
    if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id):
        raise ValueError("Invalid model id")
    entry = _registry().get(model_id)
    if entry is None:
        raise ValueError("Unknown model version")
    directory, metadata = entry
    weights_path = directory / "weights.npz"
    if weights_path.is_symlink() or not weights_path.is_file() or hashlib.sha256(weights_path.read_bytes()).hexdigest() != metadata["weights_sha256"]:
        raise ValueError("Model weight integrity check failed")
    model = QuantileRNN(len(metadata["feature_names"]), metadata["hidden_size"], metadata["kind"])
    with np.load(weights_path, allow_pickle=False) as weights:
        state = {name: torch.from_numpy(weights[name].copy()) for name in weights.files}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, metadata, directory


def predict(model_id, symbol, cutoff=None, perturbation=None, record_telemetry=True, override_frame=None):
    model, metadata, _ = _load_model(model_id)
    if symbol != metadata["symbol"]:
        raise ValueError("Model and instrument must match")
    full_frame = load_frame(symbol) if override_frame is None else validate_frame(override_frame)
    cutoff = cutoff or str(full_frame.date.iloc[-1])
    if cutoff not in set(full_frame.date):
        raise ValueError("Input cutoff must be a completed session in the historical snapshot")
    if cutoff < metadata["validation_cutoff"]:
        raise ValueError("Input cutoff precedes model-selection cutoff; retrospective leakage prohibited")
    frame = full_frame.loc[full_frame.date <= cutoff].copy().reset_index(drop=True)
    original_snapshot = snapshot_id(frame)
    changes = []
    if perturbation is not None:
        feature, percent = perturbation.get("feature"), perturbation.get("percent")
        if feature not in {"close", "volume"} or isinstance(percent, bool) or not isinstance(percent, (float, int)) or not np.isfinite(percent) or abs(percent) > 20:
            raise ValueError("What-if feature must be close or volume, percent must be finite within −20..20")
        idx = frame.index[-1]
        before = frame.loc[idx].copy()
        frame[feature] = frame[feature].astype(float)
        frame.loc[idx, feature] = float(frame.loc[idx, feature]) * (1 + percent / 100)
        if feature == "close":
            frame.loc[idx, "high"] = max(frame.loc[idx, "high"], frame.loc[idx, "open"], frame.loc[idx, "close"])
            frame.loc[idx, "low"] = min(frame.loc[idx, "low"], frame.loc[idx, "open"], frame.loc[idx, "close"])
        for field in ("open", "high", "low", "close", "volume"):
            if float(before[field]) != float(frame.loc[idx, field]):
                changes.append({"date": cutoff, "feature": field, "original": float(before[field]), "changed": float(frame.loc[idx, field]), "perturbed": float(frame.loc[idx, field])})
        validate_frame(frame)
    lookback = metadata["lookback"]
    raw = make_features(frame).tail(lookback).to_numpy()
    if raw.shape != (lookback, len(FEATURE_NAMES)) or not np.isfinite(raw).all():
        raise ValueError("Insufficient complete observations for this input window")
    mean = np.array(metadata["normalization"]["mean"])
    scale = np.array(metadata["normalization"]["scale"])
    normalized = ((raw - mean) / scale).astype(np.float32)
    with torch.inference_mode():
        quantiles, states = model(torch.from_numpy(normalized)[None, :, :], record=record_telemetry)
    values = quantiles[0].numpy().astype(float)
    base = float(frame.close.iloc[-1])
    prices = base * (1 + values)
    returns = dict(zip(("lower", "median", "upper"), values.tolist()))
    prices_dict = dict(zip(("lower", "median", "upper"), prices.tolist()))
    target = next_session(cutoff)
    actual_rows = full_frame.loc[full_frame.date == target]
    actual = float(actual_rows.close.iloc[0]) if len(actual_rows) and perturbation is None else None
    forecast = {"id": str(uuid.uuid4()), "symbol": symbol, "model_id": model_id,
                "model_kind": metadata["kind"], "model_version": metadata["version"],
                "issue_time": datetime.now(timezone.utc).isoformat(), "target_time": target,
                "input_cutoff": cutoff, "data_snapshot_id": snapshot_id(frame), "original_snapshot_id": original_snapshot,
                "mode": "Replay", "base_price": base, **prices_dict, "returns": returns,
                "lookback": lookback, "training_cutoff": metadata["training_cutoff"],
                "validation_cutoff": metadata["validation_cutoff"], "actual": actual,
                "horizon": "next trading session", "changes": changes, "sensitivity_only": perturbation is not None}
    if actual is not None:
        forecast["outcome"] = {"actual": actual, "error": float(prices[1] - actual),
                               "covered": bool(prices[0] <= actual <= prices[2]), "date": target,
                               "status": "historical_observation", "is_forward_running": False}
    if states is not None:
        telemetry = {"feature_names": FEATURE_NAMES, "input_raw": raw.tolist(),
                     "input_normalized": normalized.tolist(), "dates": frame.date.tail(lookback).tolist(),
                     "input_candles": frame.tail(lookback).to_dict(orient="records"),
                     "hidden": states["hidden"][0].tolist(), "hidden_size": model.hidden_size,
                     "model_id": model_id, "model_kind": metadata["kind"], "units": "normalized",
                     "normalization": metadata["normalization"], "output": {"returns": returns, "prices": prices_dict},
                     "head": {"raw": states["head_raw"][0].tolist(), "weight": model.head.weight.detach().tolist(),
                              "bias": model.head.bias.detach().tolist(), "parameterization": "median = raw[0]/100; lower = (raw[0] − softplus(raw[1]))/100; upper = (raw[0] + softplus(raw[2]))/100"},
                     "weights": {"input_to_hidden": model.cell.weight_ih.detach().tolist(),
                                 "hidden_to_hidden": model.cell.weight_hh.detach().tolist(),
                                 "gate_order": ["input", "forget", "candidate", "output"] if model.kind == "lstm" else ["reset", "update", "candidate"]}}
        if "cell" in states:
            telemetry["cell"] = states["cell"][0].tolist()
        for key in ("gates", "gru_gates"):
            if key in states:
                telemetry[key] = {gate: value[0].tolist() for gate, value in states[key].items()}
        forecast["telemetry"] = telemetry
    return forecast


def evaluate_model(model_id):
    _, metadata, directory = _load_model(model_id)
    path = directory / "evaluation.json"
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("evaluation_sha256"):
        raise ValueError("Evaluation artifact integrity check failed")
    result = json.loads(path.read_text())
    if result.get("model_id") != model_id:
        raise ValueError("Evaluation artifact belongs to another model version")
    return result

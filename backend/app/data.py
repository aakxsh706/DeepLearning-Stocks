"""Pinned historical observations and causal daily features; never synthetic prices."""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
FEATURE_NAMES = ["return_1d", "open_gap", "high_relative", "low_relative",
                 "close_open_return", "log_volume", "volatility_10d", "ma_gap_10d"]
INSTRUMENTS = [{"symbol": "AAPL", "name": "Apple Inc.", "exchange": "XNAS", "timezone": "America/New_York"},
               {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "XNAS", "timezone": "America/New_York"}]
DATA_SOURCE = {"name": "Public historical OHLCV snapshot", "mode": "Replay",
               "url": "https://github.com/plotly/datasets", "delay_seconds": None,
               "price_convention": "Unadjusted historical OHLC; see data/provenance.json"}


def ensure_demo_data():
    paths = [DATA_DIR / f"{item['symbol']}.csv" for item in INSTRUMENTS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("Bundled historical data missing: " + ", ".join(missing) + ". Run scripts/acquire_data.py.")
    return paths


def load_frame(symbol: str) -> pd.DataFrame:
    if symbol not in {item["symbol"] for item in INSTRUMENTS}:
        raise ValueError("Unsupported instrument")
    frame = pd.read_csv(DATA_DIR / f"{symbol}.csv")
    return validate_frame(frame)


def validate_frame(frame):
    required = ["date", "open", "high", "low", "close", "volume"]
    if not set(required).issubset(frame.columns):
        raise ValueError("Historical data requires date and OHLCV")
    frame = frame[required].copy()
    if frame.empty:
        raise ValueError("Historical data must contain at least one completed session")
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    if not frame["date"].is_monotonic_increasing or frame["date"].duplicated().any():
        raise ValueError("Observations must have unique ascending dates")
    numeric = frame[required[1:]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (frame[["open", "high", "low", "close"]] <= 0).any().any() or (frame.volume < 0).any():
        raise ValueError("Observations must be finite with positive prices and nonnegative volume")
    if (frame.high < frame[["open", "close", "low"]].max(axis=1) - 1e-6).any() or (frame.low > frame[["open", "close", "high"]].min(axis=1) + 1e-6).any():
        raise ValueError("Invalid OHLC ordering")
    # Missing sessions must never turn a next-session label into a multi-day label.
    expected = calendar().sessions_in_range(frame.date.iloc[0], frame.date.iloc[-1]).strftime("%Y-%m-%d").tolist()
    if frame.date.tolist() != expected:
        raise ValueError("Daily observations must be consecutive exchange sessions; missing or nontrading dates found")
    return frame.reset_index(drop=True)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    previous = frame.close.shift(1)
    returns = frame.close / previous - 1
    return pd.DataFrame({
        "return_1d": returns,
        "open_gap": frame.open / previous - 1,
        "high_relative": frame.high / previous - 1,
        "low_relative": frame.low / previous - 1,
        "close_open_return": frame.close / frame.open - 1,
        "log_volume": np.log1p(frame.volume),
        "volatility_10d": returns.rolling(10, min_periods=10).std(ddof=0),
        "ma_gap_10d": frame.close / frame.close.rolling(10, min_periods=10).mean() - 1,
    })[FEATURE_NAMES]


@lru_cache(maxsize=1)
def calendar():
    # Explicit bounds cover the pinned demo and avoid host-time-dependent calendars.
    return xcals.get_calendar("XNYS", start="2000-01-01", end="2035-12-31")


def next_session(date: str) -> str:
    value = pd.Timestamp(date)
    if value.tzinfo:
        value = value.tz_convert("America/New_York").tz_localize(None)
    following = value.normalize() + pd.Timedelta(days=1)
    return calendar().date_to_session(following, direction="next").strftime("%Y-%m-%d")


def snapshot_id(frame: pd.DataFrame) -> str:
    # Hash exact numeric observations and dates; future rows do not affect a snapshot.
    encoded = frame.to_csv(index=False, float_format="%.12g").encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def provenance(symbol):
    path = DATA_DIR / "provenance.json"
    if path.exists():
        return json.loads(path.read_text()).get(symbol, DATA_SOURCE)
    return DATA_SOURCE


def list_instruments():
    return INSTRUMENTS


def data_source_info(symbol="AAPL"):
    return {**DATA_SOURCE, **provenance(symbol)}

"""Daily provider boundary. External quotes never enter a trained daily model.

Alpha Vantage GLOBAL_QUOTE is end-of-day by default; a successful HTTP response
alone is not evidence of realtime entitlement. This integration intentionally
labels those observations Delayed and does not guess the age in seconds.
See https://www.alphavantage.co/documentation/#latestprice .
"""
from __future__ import annotations

import math
import os
import threading
import time
from typing import Protocol

import httpx

from .data import load_frame
from .db import utcnow


class ProviderUnavailable(RuntimeError):
    pass


class MarketProvider(Protocol):
    def historical(self, symbol: str, limit: int = 260) -> dict: ...
    def quote(self, symbol: str) -> dict: ...
    def completed_candle(self, symbol: str) -> dict: ...
    def status(self) -> dict: ...


def frame_records(frame):
    records = []
    for row in frame.to_dict("records"):
        time_value = row.get("date", row.get("time"))
        records.append({"time": str(time_value)[:10], **{
            key: float(row[key]) for key in ("open", "high", "low", "close", "volume")
        }})
    return records


class ReplayProvider:
    def historical(self, symbol: str, limit: int = 260) -> dict:
        frame = load_frame(symbol)
        candles = frame_records(frame.tail(limit))
        return {"symbol": symbol, "candles": candles, "source": "Bundled historical daily observations",
                "mode": "Replay", "delay_seconds": None, "timezone": "America/New_York",
                "session_status": "Historical replay", "as_of": candles[-1]["time"] if candles else None}

    def quote(self, symbol: str) -> dict:
        market = self.historical(symbol, 2)
        candles = market.pop("candles")
        last = candles[-1]
        previous = candles[-2]["close"] if len(candles) > 1 else last["close"]
        return {**market, "price": last["close"], "volume": last["volume"],
                "change": last["close"] - previous, "change_percent": 100 * (last["close"] / previous - 1)}

    def completed_candle(self, symbol: str) -> dict:
        return self.historical(symbol, 1)

    def status(self) -> dict:
        return {"provider": "historical", "mode": "Replay", "connected": True,
                "frequency": "daily", "delay_seconds": None, "requires_credentials": False}


class AlphaVantageProvider:
    """Server-only credential, fixed trusted origin, bounded timeout and cache."""
    def __init__(self):
        self._cache: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()
        self._last_request = 0.0

    def status(self) -> dict:
        configured = bool(os.getenv("ALPHA_VANTAGE_API_KEY", ""))
        recent = [item for item in self._cache.values() if time.monotonic() - item[0] < 300]
        connected = bool(configured and recent)
        return {"provider": "alpha_vantage", "configured": configured, "connected": connected,
                "mode": "Delayed" if connected else "Disconnected", "frequency": "end-of-day", "delay_seconds": None,
                "requires_credentials": True, "replay_available": True,
                "message": ("Last external quote request succeeded; data is end-of-day." if connected else
                            "Request a quote to check connectivity." if configured else
                            "Set ALPHA_VANTAGE_API_KEY on the server to enable external quotes.")}

    def _fetch(self, symbol: str, function: str) -> dict:
        key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        if not key:
            raise ProviderUnavailable("External provider is not configured. Historical replay is available.")
        try:
            response = httpx.get("https://www.alphavantage.co/query",
                                 params={"function": function, "symbol": symbol, "apikey": key},
                                 timeout=10.0, follow_redirects=False)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            # Never expose upstream request URLs: those contain server credentials.
            raise ProviderUnavailable("External provider request failed. Historical replay is available.") from None
        if not isinstance(payload, dict) or any(name in payload for name in ("Error Message", "Note", "Information")):
            raise ProviderUnavailable("External provider rejected the request or rate limit was reached. Historical replay is available.")
        return payload

    def quote(self, symbol: str) -> dict:
        with self._lock:
            now = time.monotonic()
            cached = self._cache.get(symbol)
            if cached and now - cached[0] < 300:
                return {**cached[1], "cached": True}
            if now - self._last_request < 60:
                raise ProviderUnavailable("External quote requests are limited to one per minute. Historical replay is available.")
            self._last_request = now
            raw = self._fetch(symbol, "GLOBAL_QUOTE").get("Global Quote", {})
            try:
                result = {"symbol": symbol, "price": float(raw["05. price"]), "volume": int(raw["06. volume"]),
                          "as_of": raw["07. latest trading day"], "change": float(raw["09. change"]),
                          "change_percent": float(raw["10. change percent"].rstrip("%"))}
                if not all(math.isfinite(result[field]) for field in ("price", "volume", "change", "change_percent")) or result["price"] <= 0:
                    raise ValueError("Invalid price")
            except (KeyError, ValueError, TypeError, AttributeError):
                raise ProviderUnavailable("External provider returned no valid quote. Historical replay is available.") from None
            result.update({"source": "Alpha Vantage GLOBAL_QUOTE", "mode": "Delayed", "delay_seconds": None,
                           "frequency": "end-of-day", "timezone": "America/New_York", "connected": True,
                           "session_status": "Unknown", "received_at": utcnow(), "cached": False,
                           "note": "End-of-day quote; exact delay is not supplied. Separate from historical model inputs."})
            self._cache[symbol] = (now, result)
            return result

    def historical(self, symbol: str, limit: int = 100) -> dict:
        # An explicit retrieval API for import tools; never silently substitutes a
        # differently adjusted feed into the bundled model's historical inputs.
        raw = self._fetch(symbol, "TIME_SERIES_DAILY").get("Time Series (Daily)", {})
        if not raw:
            raise ProviderUnavailable("External provider returned no daily history.")
        try:
            candles = [{"time": date, "open": float(row["1. open"]), "high": float(row["2. high"]),
                        "low": float(row["3. low"]), "close": float(row["4. close"]), "volume": float(row["5. volume"])}
                       for date, row in sorted(raw.items())][-limit:]
        except (ValueError, KeyError, TypeError):
            raise ProviderUnavailable("External daily history was malformed.") from None
        return {"symbol": symbol, "candles": candles, "source": "Alpha Vantage TIME_SERIES_DAILY",
                "mode": "Delayed", "delay_seconds": None, "timezone": "America/New_York",
                "session_status": "Unknown", "as_of": candles[-1]["time"], "price_convention": "unadjusted"}

    def completed_candle(self, symbol: str) -> dict:
        # A daily response may include today's incomplete bar. Conservatively
        # expose the prior dated bar only; no automatic model inference follows.
        history = self.historical(symbol, 2)
        history["candles"] = history["candles"][:1]
        history["as_of"] = history["candles"][-1]["time"]
        return history


replay_provider = ReplayProvider()
external_provider = AlphaVantageProvider()

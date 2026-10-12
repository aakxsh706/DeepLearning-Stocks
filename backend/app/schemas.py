from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class ForecastRequest(StrictRequest):
    symbol: str = Field(min_length=1, max_length=16, pattern=r"^[A-Z][A-Z0-9.-]*$")
    model_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    cutoff: date | None = None


class WhatIfRequest(StrictRequest):
    forecast_id: str = Field(min_length=1, max_length=64)
    feature: Literal["close", "volume"]
    percent: float = Field(ge=-20, le=20)

    @field_validator("percent", mode="before")
    @classmethod
    def numeric_percent(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Percent must be a finite number")
        return value


class TrainingRequest(StrictRequest):
    symbol: str = Field(default="AAPL", min_length=1, max_length=16, pattern=r"^[A-Z][A-Z0-9.-]*$")
    kind: Literal["lstm", "gru"] = "lstm"
    lookback: int = Field(default=30, ge=10, le=60, strict=True)
    epochs: int = Field(default=8, ge=1, le=20, strict=True)
    max_samples: int = Field(default=1200, ge=200, le=2000, strict=True)


class WatchlistRequest(StrictRequest):
    symbols: list[str] = Field(max_length=20)

    @field_validator("symbols")
    @classmethod
    def unique_symbols(cls, values):
        if len(set(values)) != len(values):
            raise ValueError("Watchlist symbols must be unique")
        return values

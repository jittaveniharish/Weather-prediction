"""
Utilities to fetch and prepare weather data for modelling.
"""
from __future__ import annotations

import datetime as dt
from typing import Tuple

import numpy as np
import pandas as pd
import requests

from . import config


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def _read_cache() -> pd.DataFrame | None:
    if not config.CACHE_PATH.exists():
        return None
    try:
        df = pd.read_parquet(config.CACHE_PATH)
    except Exception:
        return None
    max_age = dt.timedelta(hours=config.CACHE_MAX_AGE_HOURS)
    if df["timestamp"].max() < (pd.Timestamp.utcnow() - max_age):
        return None
    return df


def _write_cache(df: pd.DataFrame) -> None:
    try:
        df.to_parquet(config.CACHE_PATH, index=False)
    except Exception:
        pass


def fetch_weather() -> pd.DataFrame:
    """Fetch recent hourly weather data (temperature in °C)."""
    cached = _read_cache()
    if cached is not None and len(cached) >= config.HOURS_BACK // 2:
        return cached

    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "hourly": "temperature_2m",
        "past_days": min(config.HOURS_BACK // 24 + 1, 7),
        "timezone": config.TIMEZONE,
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()
    timestamps = pd.to_datetime(payload["hourly"]["time"])
    temps = payload["hourly"]["temperature_2m"]
    df = pd.DataFrame({"timestamp": timestamps, "temperature": temps})
    # Keep only the most recent HOURS_BACK entries
    df = df.sort_values("timestamp").tail(config.HOURS_BACK).reset_index(drop=True)
    _write_cache(df)
    return df


def make_supervised(df: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Construct sliding-window supervised dataset."""
    values = df["temperature"].values.astype(np.float32)
    X, y = [], []
    for start_idx in range(len(values) - horizon):
        X.append(values[start_idx : start_idx + horizon])
        y.append(values[start_idx + horizon])
    return np.array(X), np.array(y)



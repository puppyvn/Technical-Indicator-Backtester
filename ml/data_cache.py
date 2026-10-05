"""Download OHLCV once and cache it on disk: model experiments rerun hundreds of times."""

import os
import time

import pandas as pd


def load_cached_ohlcv(ticker: str, period: str = "max", interval: str = "1d",
                      cache_dir: str = "data_cache", max_age_days: float = 1.0) -> pd.DataFrame:
    from data.collector import fetch_price_history  # lazy: keeps yfinance optional for tests
    from data.validator import clean_ohlcv, validate_ohlcv

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{ticker.upper()}_{period}_{interval}.csv")

    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_days * 86400:
        return pd.read_csv(path, parse_dates=["Date"])

    df = fetch_price_history(ticker, period=period, interval=interval)
    ok, msg = validate_ohlcv(df)
    if not ok:
        raise ValueError(f"{ticker}: {msg}")
    df = clean_ohlcv(df)
    df.to_csv(path, index=False)
    return df

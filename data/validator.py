"""
Small validation utilities. Keeping this separate from collector.py
so the fetch functions stay focused purely on I/O.
"""

import pandas as pd

REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume"}


def validate_ohlcv(df: pd.DataFrame, min_rows: int = 30) -> tuple[bool, str]:
    """
    Returns (is_valid, message). Checks structure, not correctness of values —
    yfinance data is generally trustworthy, but rows can be missing/NaN-heavy
    around splits, halts, or new listings.
    """
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"

    if len(df) < min_rows:
        return False, f"Only {len(df)} rows available; need at least {min_rows}"

    null_fraction = df[["Open", "High", "Low", "Close"]].isna().mean().mean()
    if null_fraction > 0.05:
        return False, f"Too many missing price values ({null_fraction:.1%})"

    if not df["Date"].is_monotonic_increasing:
        return False, "Dates are not sorted ascending"

    return True, "OK"


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill small gaps, drop rows that are still incomplete."""
    df = df.copy()
    price_cols = ["Open", "High", "Low", "Close"]
    df[price_cols] = df[price_cols].ffill()
    df = df.dropna(subset=price_cols)
    return df.reset_index(drop=True)

"""
Feature engineering (Phase M1).

RULE: every feature at row t may only use prices/volumes up to and including
row t. No negative shifts, no full-sample statistics. tests/test_ml.py checks
this by truncating the data and confirming earlier rows do not change.

Raw prices are non-stationary, so features are returns, ratios and z-score-like
quantities that are comparable across time and across tickers.
"""

import numpy as np
import pandas as pd

from indicators.technical import add_bollinger_bands, add_macd, add_rsi


def build_features(
    df: pd.DataFrame,
    price_col: str = "Close",
    include_calendar: bool = False,
) -> pd.DataFrame:
    """Return a feature DataFrame with the same index as `df` (NaN during warm-up)."""
    close = df[price_col]
    ret1 = close.pct_change()

    # Reuse the indicator functions from the base project.
    tmp = add_rsi(df, 14, price_col)
    tmp = add_macd(tmp, 12, 26, 9, price_col)
    tmp = add_bollinger_bands(tmp, 20, 2, price_col)

    f = pd.DataFrame(index=df.index)

    # --- momentum: multi-horizon returns
    for n in (1, 5, 10, 20):
        f[f"ret_{n}"] = close.pct_change(n)

    # --- trend: distance from moving averages, and the short/long gap
    sma = {n: close.rolling(n).mean() for n in (10, 20, 50)}
    for n in (10, 20, 50):
        f[f"dist_sma_{n}"] = close / sma[n] - 1
    f["sma_gap_20_50"] = sma[20] / sma[50] - 1

    # --- oscillators, rescaled so they are unit-free
    f["rsi_14"] = tmp["RSI_14"] / 100.0
    f["macd_hist_pct"] = tmp["MACD_Hist"] / close  # MACD is in price units; divide by price

    # --- volatility
    f["vol_5"] = ret1.rolling(5).std()
    f["vol_20"] = ret1.rolling(20).std()
    f["vol_ratio"] = f["vol_5"] / f["vol_20"].replace(0, np.nan)
    width = (tmp["BB_Upper"] - tmp["BB_Lower"]).replace(0, np.nan)
    f["bb_pctb"] = (close - tmp["BB_Lower"]) / width  # where price sits inside the band
    f["range_pct"] = (df["High"] - df["Low"]) / close

    # --- volume relative to its own recent average
    if "Volume" in df.columns:
        v = df["Volume"].astype(float).replace(0, np.nan)
        f["vol_rel"] = (v / v.rolling(20).mean() - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if include_calendar and "Date" in df.columns:
        f["dow"] = pd.to_datetime(df["Date"]).dt.dayofweek / 4.0

    return f.replace([np.inf, -np.inf], np.nan)

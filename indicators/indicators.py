"""
Pure functions: DataFrame in, DataFrame (with new columns) out.
No plotting, no fetching. Easy to unit test in isolation.
"""

import pandas as pd
import numpy as np


def add_sma(df: pd.DataFrame, window: int, column: str = "Close") -> pd.DataFrame:
    df = df.copy()
    df[f"SMA_{window}"] = df[column].rolling(window=window).mean()
    return df


def add_ema(df: pd.DataFrame, span: int, column: str = "Close") -> pd.DataFrame:
    df = df.copy()
    df[f"EMA_{span}"] = df[column].ewm(span=span, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14, column: str = "Close") -> pd.DataFrame:
    df = df.copy()
    delta = df[column].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "Close",
) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()

    df["MACD_line"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD_line"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD_line"] - df["MACD_signal"]
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2,
    column: str = "Close",
) -> pd.DataFrame:
    df = df.copy()
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()

    df["BB_Middle"] = rolling_mean
    df["BB_Upper"] = rolling_mean + (rolling_std * num_std)
    df["BB_Lower"] = rolling_mean - (rolling_std * num_std)
    return df


def add_all_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Convenience wrapper that applies every indicator using a params dict,
    e.g. as produced by the Streamlit sidebar. Keeps app.py from having to
    know the individual function signatures.
    """
    df = add_sma(df, params["sma_short"])
    df = add_sma(df, params["sma_long"])
    df = add_ema(df, params["ema_short"])
    df = add_ema(df, params["ema_long"])
    df = add_rsi(df, params["rsi_period"])
    df = add_macd(df, params["ema_short"], params["ema_long"], params["macd_signal"])
    df = add_bollinger_bands(df, params["bb_window"], params["bb_std"])
    return df

"""
Converts indicator values into a `position` column: 1 = long, 0 = flat.
Kept separate from engine.py so you can swap strategies without touching
the simulation math.
"""

import pandas as pd


def sma_crossover_signal(df: pd.DataFrame, short_col: str, long_col: str) -> pd.DataFrame:
    """
    Classic trend-following signal: go long when the short SMA is above
    the long SMA, flat otherwise. Position is shifted by 1 day to avoid
    lookahead bias (you trade on the next bar's open after a signal fires).
    """
    df = df.copy()
    df["signal"] = (df[short_col] > df[long_col]).astype(int)
    df["position"] = df["signal"].shift(1).fillna(0)
    return df


def rsi_mean_reversion_signal(
    df: pd.DataFrame,
    rsi_col: str,
    oversold: int = 30,
    overbought: int = 70,
) -> pd.DataFrame:
    """
    Go long when RSI dips below `oversold` (expecting a bounce), exit when
    it climbs back above `overbought`. Position persists between signals
    (i.e. it's a state machine, not a one-bar flag).
    """
    df = df.copy()
    position = []
    holding = 0

    for rsi in df[rsi_col]:
        if pd.isna(rsi):
            position.append(0)
            continue
        if rsi < oversold:
            holding = 1
        elif rsi > overbought:
            holding = 0
        position.append(holding)

    df["signal"] = position
    df["position"] = pd.Series(position, index=df.index).shift(1).fillna(0)
    return df


def macd_crossover_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Long when MACD line is above its signal line."""
    df = df.copy()
    df["signal"] = (df["MACD"] > df["MACD_Signal"]).astype(int)
    df["position"] = df["signal"].shift(1).fillna(0)
    return df

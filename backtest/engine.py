"""
Turns a `position` column into an equity curve and summary stats.
Assumes `signals.py` has already populated df["position"] with 0/1 values.
"""

import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 10_000,
    fee_bps: float = 5,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Simulates:
      - `strategy_equity`: following the position column
      - `buy_hold_equity`: simply holding from day 1

    A trading fee (in basis points) is applied whenever the position changes,
    to keep the strategy from looking unrealistically good on paper.
    """
    df = df.copy()
    df["daily_return"] = df[price_col].pct_change().fillna(0)

    # Fee applied on days the position flips (entry or exit)
    position_change = df["position"].diff().abs().fillna(0)
    fee_drag = position_change * (fee_bps / 10_000)

    df["strategy_return"] = (df["position"] * df["daily_return"]) - fee_drag
    df["buy_hold_return"] = df["daily_return"]

    df["strategy_equity"] = initial_capital * (1 + df["strategy_return"]).cumprod()
    df["buy_hold_equity"] = initial_capital * (1 + df["buy_hold_return"]).cumprod()

    # Drawdowns, for the performance chart
    df["strategy_peak"] = df["strategy_equity"].cummax()
    df["strategy_drawdown"] = (df["strategy_equity"] - df["strategy_peak"]) / df["strategy_peak"]

    return df


def compute_summary_stats(df: pd.DataFrame, trading_days_per_year: int = 252) -> dict:
    """Headline metrics for both the strategy and buy-and-hold, side by side."""

    def _stats(returns: pd.Series, equity: pd.Series) -> dict:
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        n_years = len(returns) / trading_days_per_year
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan

        ann_vol = returns.std() * np.sqrt(trading_days_per_year)
        sharpe = (returns.mean() * trading_days_per_year) / ann_vol if ann_vol > 0 else np.nan

        peak = equity.cummax()
        max_dd = ((equity - peak) / peak).min()

        return {
            "Total Return": total_return,
            "CAGR": cagr,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
        }

    return {
        "Strategy": _stats(df["strategy_return"], df["strategy_equity"]),
        "Buy & Hold": _stats(df["buy_hold_return"], df["buy_hold_equity"]),
    }

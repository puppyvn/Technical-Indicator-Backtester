"""
Evaluation (Phase M8). Two separate questions, two separate sets of metrics:

  classification_metrics : does the model PREDICT well?   (accuracy vs baseline, AUC ...)
  backtest_signal        : does it MAKE MONEY?            (existing engine: Sharpe, drawdown ...)

A model can score 56% accuracy and lose money, or trade profitably on mediocre
accuracy, so always report both.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backtest.engine import compute_summary_stats, run_backtest


def classification_metrics(y_true: pd.Series, p: pd.Series, threshold: float = 0.5) -> dict:
    """Out-of-sample classification quality over rows where both label and prediction exist."""
    m = y_true.notna() & p.notna()
    yt, pp = y_true[m].astype(int), p[m]
    if len(yt) == 0:
        return {"n": 0}
    pred = (pp > threshold).astype(int)
    return {
        "n": int(len(yt)),
        "accuracy": float((pred == yt).mean()),
        "baseline_accuracy": float(max(yt.mean(), 1 - yt.mean())),  # always predict the majority class
        "always_up_accuracy": float(yt.mean()),
        "auc": float(roc_auc_score(yt, pp)) if yt.nunique() == 2 else float("nan"),
        "precision_up": float(yt[pred == 1].mean()) if pred.sum() > 0 else float("nan"),
        "pct_predicted_up": float(pred.mean()),
    }


def attach_ml_position(
    df: pd.DataFrame,
    signal: pd.Series,
    start: int | None = None,
    end: int | None = None,
) -> pd.DataFrame:
    """
    Add `signal` and `position` columns and cut the frame to the out-of-sample window.

    position[t] = signal[t-1]  (decide at the close of t-1, hold through t: the same
    lookahead rule as backtest/signals.py). The window starts at the first non-NaN
    signal so strategy and buy-and-hold are measured over exactly the same dates.
    `start` / `end` are index labels (inclusive) to narrow the window further.
    """
    d = df.copy()
    sig = signal.reindex(d.index)
    first = sig.first_valid_index()
    if first is None:
        raise ValueError("Signal has no valid values")
    lo = first if start is None else max(first, start)
    hi = d.index[-1] if end is None else end
    d["signal"] = sig
    d["position"] = sig.shift(1)
    d = d.loc[lo:hi].copy()
    d["position"] = d["position"].fillna(0.0)
    return d.reset_index(drop=True)


def backtest_signal(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 10_000,
    fee_bps: float = 5,
    start: int | None = None,
    end: int | None = None,
):
    """Run the EXISTING engine on an ML signal. Returns (backtest_df, stats).
    `stats` = {"Strategy": {...}, "Buy & Hold": {...}}; the strategy dict also gets
    Trades / Avg Exposure / Turnover, which matter because fees punish over-trading."""
    d = attach_ml_position(df, signal, start, end)
    bt = run_backtest(d, initial_capital=initial_capital, fee_bps=fee_bps)
    stats = compute_summary_stats(bt)
    change = bt["position"].diff().abs().fillna(0)
    stats["Strategy"]["Trades"] = int((change > 1e-9).sum())
    stats["Strategy"]["Avg Exposure"] = float(bt["position"].mean())
    stats["Strategy"]["Turnover"] = float(change.sum())
    return bt, stats


def timing_test(
    df: pd.DataFrame,
    signal: pd.Series,
    fee_bps: float = 5,
    n_shifts: int = 500,
    start: int | None = None,
    end: int | None = None,
    min_shift: int = 20,
    seed: int = 0,
    periods: int = 252,
) -> dict:
    """
    Circular-shift test: is the strategy's Sharpe better than what the SAME exposure and the
    SAME trading frequency would earn if its timing were unrelated to returns?

    The position series is rolled by a random number of days (wrapping around). That keeps
    the average exposure, the turnover and the autocorrelation of the signal exactly the
    same, but breaks any alignment with returns. `p_value` = share of shifted versions whose
    Sharpe is >= the real one. Small (< 0.05) means the timing carries information.

    Why it matters: in a rising market ANY strategy that is 60% invested has a positive
    Sharpe. Profit alone is not skill; beating its own shifted copies is closer to it.
    (Still not proof: testing many configurations on many tickers guarantees some small
    p-values by luck. Treat them as a filter, not a verdict.)
    """
    d = attach_ml_position(df, signal, start, end)
    # The first row of every window is flat by construction (no earlier signal exists to act on).
    # Rolling would move that artefact around and blur the null, so the test skips it.
    pos = d["position"].to_numpy(float)[1:]
    ret = d["Close"].pct_change().to_numpy()[1:]
    n = len(pos)

    def sharpe(p):
        change = np.abs(np.diff(p, prepend=p[0]))
        r = p * ret - change * fee_bps / 10_000
        sd = r.std(ddof=1)
        return float(r.mean() * periods / (sd * np.sqrt(periods))) if sd > 0 else float("nan")

    actual = sharpe(pos)
    if n <= 2 * min_shift + 1 or np.isnan(actual):
        return {"sharpe": actual, "null_mean": float("nan"), "null_p95": float("nan"), "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    null = np.array([sharpe(np.roll(pos, int(k))) for k in rng.integers(min_shift, n - min_shift, n_shifts)])
    null = null[~np.isnan(null)]
    return {
        "sharpe": actual,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "p_value": float((np.sum(null >= actual - 1e-12) + 1) / (len(null) + 1)),
    }


def ablation_table(rows: list[tuple[str, dict]]) -> pd.DataFrame:
    """rows = [(name, flat_stats_dict), ...] -> tidy comparison table."""
    cols = ["Total Return", "CAGR", "Annualized Volatility", "Sharpe Ratio",
            "Max Drawdown", "Trades", "Avg Exposure", "Timing p-value"]
    data = {name: {c: s.get(c, np.nan) for c in cols} for name, s in rows}
    return pd.DataFrame(data).T.rename_axis("Configuration")

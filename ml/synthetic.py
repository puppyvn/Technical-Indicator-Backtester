"""
Synthetic OHLCV generator so the whole ML pipeline can be developed and tested
offline, and so tests can plant known structure.

Returns follow a 3-state Markov chain (calm / choppy / turbulent) with different
drift and volatility, plus an AR(1) term (`momentum`) that creates a small amount
of genuinely predictable structure. Real markets have far less signal than a
large `momentum` value implies: use big values only to prove the machinery works.
"""

import numpy as np
import pandas as pd


def make_synthetic_ohlcv(
    n: int = 3000,
    seed: int = 0,
    momentum: float = 0.08,
    start: str = "2010-01-04",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # regime parameters: (daily drift, daily volatility)
    params = [(0.0007, 0.008), (0.0000, 0.012), (-0.0010, 0.028)]
    transition = np.array([
        [0.985, 0.012, 0.003],
        [0.020, 0.970, 0.010],
        [0.030, 0.050, 0.920],
    ])

    state = 0
    states = np.empty(n, dtype=int)
    for i in range(n):
        states[i] = state
        state = rng.choice(3, p=transition[state])

    mu = np.array([params[s][0] for s in states])
    sigma = np.array([params[s][1] for s in states])
    shocks = rng.standard_normal(n)

    returns = np.empty(n)
    prev = 0.0
    for i in range(n):
        returns[i] = mu[i] + momentum * prev + sigma[i] * shocks[i]
        prev = returns[i] - mu[i]  # AR term acts on the surprise, not the drift

    close = 100 * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.002, n))
    spread = np.abs(rng.normal(0, 0.006, n)) + 0.5 * sigma
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = (2e6 * np.exp(rng.normal(0, 0.3, n)) * (1 + 20 * np.abs(returns))).astype(int)

    return pd.DataFrame({
        "Date": pd.bdate_range(start, periods=n),
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    })


def make_era_shift_ohlcv(
    n_first: int = 2200,
    n_second: int = 3800,
    momentum_first: float = 0.0,
    momentum_second: float = 0.35,
    seed: int = 0,
    start: str = "1990-01-02",
) -> pd.DataFrame:
    """
    Two spliced eras with continuous prices: era 1 has no learnable structure, era 2 has a lot.

    Purpose: reproduce the failure mode where something decided on an early era (e.g. a regime
    gate learned on the validation slice) is applied unchanged to a later era that behaves
    differently. This is an ENGINEERED scenario to demonstrate a mechanism, not evidence
    about real markets.
    """
    a = make_synthetic_ohlcv(n_first, seed=seed, momentum=momentum_first)
    b = make_synthetic_ohlcv(n_second, seed=seed + 101, momentum=momentum_second)
    k = a["Close"].iloc[-1] / b["Open"].iloc[0]
    for col in ("Open", "High", "Low", "Close"):
        b[col] = b[col] * k
    out = pd.concat([a, b], ignore_index=True)
    out["Date"] = pd.bdate_range(start, periods=len(out))
    return out

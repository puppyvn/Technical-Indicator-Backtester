"""
Combining models (Phase M7).

Layers, applied in this order (each returns a Series aligned to the input):
    combine_probabilities   average P(up) from several models
    probabilities_to_signal (in tree_model.py) threshold -> 0/1
    apply_regime_gate       zero the signal in regimes where it should not trade
    size_position           scale exposure by target_vol / recent_vol
    smooth_position         suppress tiny daily changes (fees)

Convention: NaN = "no information yet". Backtesting treats NaN as flat AFTER the
first valid signal date, and uses that first valid date as the start of the
out-of-sample window.
"""

import numpy as np
import pandas as pd

from ml.regime import performance_by_regime


def combine_probabilities(ps: list[pd.Series], weights: list[float] | None = None) -> pd.Series:
    """Weighted mean of probabilities. NaN wherever ANY model has no prediction, so the
    ensemble only exists where every member is out-of-sample."""
    if not ps:
        raise ValueError("Need at least one probability series")
    w = np.ones(len(ps)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(ps):
        raise ValueError("weights must match the number of probability series")
    frame = pd.concat(ps, axis=1)
    ok = frame.notna().all(axis=1)
    out = pd.Series(np.nan, index=frame.index, name="p_ens")
    out[ok] = (frame[ok].to_numpy() * (w / w.sum())).sum(axis=1)
    return out


def apply_regime_gate(signal: pd.Series, regime: pd.Series, allowed: set[int]) -> pd.Series:
    """Keep the signal only in `allowed` regimes; flat (0) elsewhere. NaN signal stays NaN."""
    gated = signal.where(regime.isin(list(allowed)), 0.0)
    gated[signal.isna()] = np.nan
    return gated.rename(signal.name)


def learn_allowed_regimes(
    strategy_returns: pd.Series,
    regime: pd.Series,
    min_days: int = 20,
    min_sharpe: float = 0.0,
    allow_unknown: bool = False,
    n_regimes: int | None = None,
) -> set[int]:
    """
    Regimes in which the strategy earned a Sharpe above `min_sharpe`.
    !! Call this on a VALIDATION slice only, then apply the result to the test period.
    Learning it on the same data you report on is lookahead.

    allow_unknown=False (default): a regime with fewer than `min_days` observations is EXCLUDED
        (no evidence it works). allow_unknown=True: it is ALLOWED instead (no evidence it fails).
        Use True with short, rolling windows, where a regime may simply be absent from the window.
        When True, pass `n_regimes` so regimes that never appear are known about too.
    """
    table = performance_by_regime(strategy_returns, regime)
    if table.empty:
        return set(range(n_regimes)) if (allow_unknown and n_regimes) else set()
    good = table[(table["days"] >= min_days) & (table["sharpe"] > min_sharpe)]
    allowed = {int(k) for k in good.index}
    if allow_unknown:
        seen_enough = {int(k) for k in table[table["days"] >= min_days].index}
        universe = set(range(n_regimes)) if n_regimes else {int(k) for k in table.index}
        allowed |= universe - seen_enough
    return allowed


def rolling_regime_gate(
    strategy_returns: pd.Series,
    regime: pd.Series,
    refresh: int = 252,
    window: int = 756,
    min_days: int = 30,
    min_sharpe: float = 0.0,
    n_regimes: int = 3,
    start: int = 0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Time-varying regime gate: re-decide which regimes may trade every `refresh` rows.

    For each block [s, s + refresh), the allowed regimes are learned from the trailing
    `window` rows STRICTLY BEFORE s (rows [s - window, s)), so no decision ever uses data from
    the block it is applied to (or anything later). A regime with fewer than `min_days`
    observations in the window is allowed (no evidence against it), so the first blocks,
    which have no history yet, trade everything.

    Why not one fixed decision? A single set learned on an early slice is applied to decades
    that may behave very differently, and can switch a good strategy off for most of the sample.

    Parameters
    ----------
    strategy_returns  daily returns of the UNGATED strategy (NaN where it had no position yet)
    regime            walk-forward regime id per row (NaN = unknown)
    start             first row position at which blocks begin (rows before it are not gated)

    Returns
    -------
    gate_ok   bool Series, True where the row's regime is allowed in its block
    schedule  DataFrame [block_start, block_end, allowed, n_allowed] (positions, end exclusive)
    """
    n = len(strategy_returns)
    gate_ok = pd.Series(True, index=strategy_returns.index, name="gate_ok")
    records = []
    for s in range(start, n, refresh):
        e = min(s + refresh, n)
        lo = max(0, s - window)
        allowed = learn_allowed_regimes(
            strategy_returns.iloc[lo:s], regime.iloc[lo:s],
            min_days=min_days, min_sharpe=min_sharpe,
            allow_unknown=True, n_regimes=n_regimes,
        )
        block_regime = regime.iloc[s:e]
        gate_ok.iloc[s:e] = block_regime.isin(list(allowed)).to_numpy()  # unknown regime (NaN) -> closed, like the static gate
        records.append({"block_start": s, "block_end": e,
                        "allowed": tuple(sorted(allowed)), "n_allowed": len(allowed)})
    return gate_ok, pd.DataFrame(records)


def apply_gate_mask(signal: pd.Series, gate_ok: pd.Series) -> pd.Series:
    """Zero the signal where `gate_ok` is False; NaN signal stays NaN."""
    gated = signal.where(gate_ok.reindex(signal.index).fillna(True).astype(bool), 0.0)
    gated[signal.isna()] = np.nan
    return gated.rename(signal.name)


def size_position(
    signal: pd.Series,
    daily_vol: pd.Series,
    target_vol: float = 0.15,
    max_leverage: float = 1.0,
    periods: int = 252,
) -> pd.Series:
    """Volatility targeting: exposure = signal * min(max_leverage, target_vol / annualised_vol).
    Scale-down only by default (max_leverage=1) so it can cut drawdowns but never adds leverage.
    NaN volatility gives NaN exposure (treated as flat later)."""
    ann = daily_vol * np.sqrt(periods)
    scale = (target_vol / ann.replace(0, np.nan)).clip(upper=max_leverage)
    return (signal * scale).rename(signal.name)


def smooth_position(position: pd.Series, min_change: float = 0.1) -> pd.Series:
    """Only update exposure when it moves by at least `min_change` (exits to 0 always allowed).
    Sizing changes the position every day, which multiplies fees; this removes the noise."""
    vals = position.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    cur = np.nan
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        if np.isnan(cur) or abs(v - cur) >= min_change or v == 0.0:
            cur = v
        out[i] = cur
    return pd.Series(out, index=position.index, name=position.name)

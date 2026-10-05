"""
Ablation study (Phase M8): add one layer at a time and measure what it really adds.

    python -m experiments.run_ablation --tickers AAPL SPY MSFT
    python -m experiments.run_ablation --synthetic            # offline demo, no network
    python -m experiments.run_ablation --tickers AAPL --lstm  # include the sequence model (slow)

Honesty protocol built into the script
--------------------------------------
* Every ML prediction is walk-forward out-of-sample.
* The out-of-sample window is split in two: a VALIDATION slice (first `val_frac`) and a
  TEST slice (the rest). Everything that is tuned or learned (probability threshold,
  which rule-based strategy is "best", which regimes are allowed to trade) uses the
  validation slice ONLY. Every reported number comes from the TEST slice.
* All configurations are scored on identical dates, with identical fees.
"""

import argparse
import os
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from backtest.signals import macd_crossover_signal, rsi_mean_reversion_signal, sma_crossover_signal
from indicators.technical import add_macd, add_rsi, add_sma
from ml.ensemble import (apply_gate_mask, apply_regime_gate, combine_probabilities,
                         learn_allowed_regimes, rolling_regime_gate, size_position, smooth_position)
from ml.evaluate import ablation_table, backtest_signal, classification_metrics, timing_test
from ml.features import build_features
from ml.io import save_predictions
from ml.labels import build_labels
from ml.lstm_model import make_sequence_model
from ml.regime import performance_by_regime, regime_features, walk_forward_regimes
from ml.tree_model import make_tree_model, probabilities_to_signal
from ml.walkforward import walk_forward_predict


@dataclass
class Config:
    horizon: int = 5
    kind: str = "xgboost"
    initial_train: int = 1000
    step: int = 63
    use_lstm: bool = False
    lstm_step: int = 250
    lstm_train_window: int | None = 2000
    lookback: int = 30
    lstm_epochs: int = 30
    n_seeds: int = 1
    device: str = "auto"   # "auto" | "cpu" | "cuda"  (passed to the sequence model)
    n_regimes: int = 3
    val_frac: float = 0.4
    fee_bps: float = 5
    target_vol: float = 0.15
    thresholds: tuple = (0.50, 0.52, 0.54, 0.56, 0.58, 0.60)
    hysteresis: float = 0.0   # >0: exit only when P(up) falls this far BELOW the entry threshold
    gate_mode: str = "static"  # "static": decide once on validation | "rolling": refresh over time | "off"
    gate_refresh: int = 252    # rolling: re-decide every this many rows (~1 year)
    gate_window: int = 756     # rolling: learn from this many trailing rows (~3 years)
    gate_min_days: int = 30    # rolling: minimum days of evidence per regime
    seed: int = 0


def rule_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Unshifted 0/1 signals from the base project's rule-based strategies."""
    d = add_sma(add_sma(add_sma(df, 20), 50), 200)
    d = add_macd(add_rsi(d, 14))
    return {
        "SMA 20/50": sma_crossover_signal(d, "SMA_20", "SMA_50")["signal"].astype(float),
        "SMA 50/200": sma_crossover_signal(d, "SMA_50", "SMA_200")["signal"].astype(float),
        "MACD": macd_crossover_signal(d)["signal"].astype(float),
        "RSI reversion": rsi_mean_reversion_signal(d, "RSI_14")["signal"].astype(float),
    }


def to_signal(p: pd.Series, threshold: float, cfg: Config) -> pd.Series:
    exit_t = threshold - cfg.hysteresis if cfg.hysteresis > 0 else None
    return probabilities_to_signal(p, threshold, exit_threshold=exit_t)


def _sharpe(df, sig, cfg, start, end) -> float:
    _, st = backtest_signal(df, sig, fee_bps=cfg.fee_bps, start=start, end=end)
    s = st["Strategy"]["Sharpe Ratio"]
    return -np.inf if (s is None or np.isnan(s)) else float(s)


def pick_threshold(df, p, cfg, val_start, val_end) -> float:
    """Choose the probability threshold that maximises VALIDATION Sharpe."""
    scores = {t: _sharpe(df, to_signal(p, t, cfg), cfg, val_start, val_end) for t in cfg.thresholds}
    best = max(scores, key=scores.get)
    return best if np.isfinite(scores[best]) else cfg.thresholds[len(cfg.thresholds) // 2]


def exposure_warnings(table: pd.DataFrame, drop: float = 0.5, min_ref: float = 0.05) -> list[str]:
    """Name the layer that switches the strategy off.

    Compares Avg Exposure along the pipeline (members -> ensemble -> gate -> sizing) and flags
    any step that removes more than `drop` of the exposure it received. A flat equity curve
    almost always has exactly one such step.
    """
    exp = table["Avg Exposure"].dropna()
    members = [i for i in exp.index if i.startswith("Level ")]
    chain = []
    if "Ensemble (avg P)" in exp.index and members:
        chain.append(("Ensemble (avg P)", float(exp[members].mean()), "the average of its member models"))
    gate = [i for i in exp.index if i.startswith("+ regime gate")]
    sizing = [i for i in exp.index if i.startswith("+ vol sizing")]
    prev_name = "Ensemble (avg P)" if "Ensemble (avg P)" in exp.index else (members[0] if members else None)
    msgs = []
    if prev_name is not None:
        for name, ref, what in chain:
            if ref > min_ref and exp[name] < (1 - drop) * ref:
                msgs.append(f"'{name}' exposure {exp[name]:.3f} vs {ref:.3f} for {what}")
        for nxt in gate + sizing:
            ref = float(exp[prev_name])
            if ref > min_ref and exp[nxt] < (1 - drop) * ref:
                msgs.append(f"'{nxt}' exposure {exp[nxt]:.3f} vs {ref:.3f} for '{prev_name}'")
            prev_name = nxt
    return msgs


def print_diagnostics(res: dict) -> None:
    """Everything needed to see WHY a layer did what it did."""
    i = res["info"]
    n = i["n_rows"]
    print(f"\n  rows: validation {n['validation']}, test {n['test']}   "
          f"(thresholds: models {i['thresholds']}, ensemble {i['ensemble_threshold']})")

    perf = i["regime_perf_val"].copy()
    if not perf.empty:
        perf["test_share"] = i["regime_share_test"].reindex(perf.index)
        if i["gate_mode"] == "static":
            perf["gate"] = ["allowed" if r in i["allowed_regimes"] else "BLOCKED" for r in perf.index]
        elif i["gate_mode"] == "rolling" and i["gate_schedule"] is not None:
            sch = i["gate_schedule"]
            test_blocks = sch[sch["block_end"] > i["test_pos"]]   # blocks that overlap the test slice
            perf["gate"] = [f"allowed in {np.mean([r in a for a in test_blocks['allowed']]):.0%} of test blocks"
                            for r in perf.index]
        else:
            perf["gate"] = "n/a (off)"
        show = perf.rename(columns={"share": "val_share"})[
            ["days", "val_share", "test_share", "ann_return", "ann_vol", "sharpe", "hit_rate", "gate"]]
        print(f"  Ensemble strategy by regime on the VALIDATION slice (regime 0 = calmest ... highest = most volatile):")
        print("  " + show.to_string(float_format=lambda v: f"{v:,.3f}").replace("\n", "\n  "))
        gap = (perf["val_share"] if "val_share" in perf else perf["share"]).sub(perf["test_share"].fillna(0)).abs().max()
        if gap > 0.25:
            print(f"  NOTE: regime shares differ by up to {gap:.0%} between validation and test. A gate learned on "
                  f"validation may not describe the test era (try --gate rolling).")
    if i["gate_mode"] == "rolling" and i["gate_schedule"] is not None:
        sch = i["gate_schedule"]
        print(f"  Rolling gate: {len(sch)} blocks, allowed-set sizes {sch['n_allowed'].value_counts().sort_index().to_dict()}")
    for m in exposure_warnings(res["table"]):
        print(f"  WARNING exposure collapse: {m}")
    t = res["table"]
    base = "Baseline: vol-targeted buy & hold (no model)"
    if base in t.index and "+ vol sizing" in t.index:
        a, b = t.loc["+ vol sizing"], t.loc[base]
        print(f"  Model vs model-free volatility targeting: Sharpe {a['Sharpe Ratio']:.3f} vs {b['Sharpe Ratio']:.3f} "
              f"(diff {a['Sharpe Ratio'] - b['Sharpe Ratio']:+.3f}), max drawdown {a['Max Drawdown']:.3f} vs {b['Max Drawdown']:.3f}, "
              f"timing p {a['Timing p-value']:.3f} vs {b['Timing p-value']:.3f}.")
        print("  A gain the baseline matches is not evidence of model skill.")


def run_ticker(df: pd.DataFrame, name: str, cfg: Config) -> dict:
    X = build_features(df)
    y = build_labels(df, cfg.horizon)

    # ---- Level 1: tree model, walk-forward out-of-sample
    p_tree = walk_forward_predict(X, y, partial(make_tree_model, cfg.kind),
                                  cfg.initial_train, cfg.step, cfg.horizon)
    probs = {"tree": p_tree}

    # ---- Level 3: sequence model (optional, slow)
    if cfg.use_lstm:
        factory = partial(make_sequence_model, lookback=cfg.lookback, epochs=cfg.lstm_epochs,
                          n_seeds=cfg.n_seeds, seed=cfg.seed, device=cfg.device)
        probs["seq"] = walk_forward_predict(X, y, factory, cfg.initial_train, cfg.lstm_step,
                                            cfg.horizon, train_window=cfg.lstm_train_window)

    # ---- Level 2: regimes
    regime = walk_forward_regimes(regime_features(df), cfg.n_regimes, cfg.initial_train, cfg.step)

    p_ens = combine_probabilities(list(probs.values()))

    # ---- validation / test split of the out-of-sample window
    oos = p_ens.dropna().index
    split = int(len(oos) * cfg.val_frac)
    val_start, test_start = oos[0], oos[split]
    val_end = test_start - 1

    close_ret = df["Close"].pct_change()

    def val_returns(sig):
        pos = sig.shift(1).fillna(0.0)
        return (pos * close_ret).loc[val_start:val_end]

    # ---- signals (thresholds chosen on validation only)
    th = {k: pick_threshold(df, p, cfg, val_start, val_end) for k, p in probs.items()}
    th_ens = pick_threshold(df, p_ens, cfg, val_start, val_end)
    signals = {k: to_signal(p, th[k], cfg) for k, p in probs.items()}
    sig_ens = to_signal(p_ens, th_ens, cfg)

    # ---- regime gate (three modes, see --gate)
    gate_schedule = None
    if cfg.gate_mode == "static":
        # one decision, learned on the validation slice, applied to everything after it
        allowed = learn_allowed_regimes(val_returns(sig_ens), regime.loc[val_start:val_end])
        sig_gate = apply_regime_gate(sig_ens, regime, allowed)
    elif cfg.gate_mode == "rolling":
        # re-decided every `gate_refresh` rows from the trailing `gate_window` rows BEFORE each block
        ungated_returns = sig_ens.shift(1) * close_ret            # NaN where no position existed yet
        gate_ok, gate_schedule = rolling_regime_gate(
            ungated_returns, regime, refresh=cfg.gate_refresh, window=cfg.gate_window,
            min_days=cfg.gate_min_days, n_regimes=cfg.n_regimes,
            start=int(df.index.get_loc(val_start)))
        sig_gate = apply_gate_mask(sig_ens, gate_ok)
        allowed = set(gate_schedule.loc[gate_schedule["block_end"] > df.index.get_loc(test_start),
                                        "allowed"].explode().dropna().astype(int))
    elif cfg.gate_mode == "off":
        sig_gate, allowed = sig_ens, set(range(cfg.n_regimes))
    else:
        raise ValueError(f"gate_mode must be static, rolling or off, got '{cfg.gate_mode}'")
    sig_size = smooth_position(size_position(sig_gate, X["vol_20"], cfg.target_vol), 0.1)

    # Model-free baseline: always long, sized by the SAME volatility rule and smoothing. Volatility targeting
    # alone lowers drawdowns and raises Sharpe (and earns small timing p-values) with no model at all, so any
    # gain of '+ vol sizing' that this baseline matches is NOT evidence that the model has skill.
    always_long = pd.Series(np.where(p_ens.notna(), 1.0, np.nan), index=df.index)
    sig_volbh = smooth_position(size_position(always_long, X["vol_20"], cfg.target_vol), 0.1)

    rules = rule_signals(df)
    best_rule = max(rules, key=lambda k: _sharpe(df, rules[k], cfg, val_start, val_end))

    # ---- score everything on the TEST slice
    configs = [(f"Best rule, picked on val ({best_rule})", rules[best_rule]),
               (f"Level 1: {cfg.kind}", signals["tree"])]
    if cfg.use_lstm:
        configs += [("Level 3: sequence model", signals["seq"]),
                    ("Ensemble (avg P)", sig_ens)]
    gate_label = {"static": "+ regime gate (static)", "rolling": "+ regime gate (rolling)", "off": "+ regime gate (off)"}[cfg.gate_mode]
    configs += [(gate_label, sig_gate), ("+ vol sizing", sig_size),
                ("Baseline: vol-targeted buy & hold (no model)", sig_volbh)]

    rows, bh = [], None
    for label, sig in configs:
        _, st = backtest_signal(df, sig, fee_bps=cfg.fee_bps, start=test_start)
        st["Strategy"]["Timing p-value"] = timing_test(df, sig, cfg.fee_bps, start=test_start)["p_value"]
        rows.append((label, st["Strategy"]))
        bh = st["Buy & Hold"]
    table = ablation_table([("Buy & hold", bh)] + rows)

    test_mask = y.index >= test_start
    cls = {k: classification_metrics(y[test_mask], p[test_mask], th[k]) for k, p in probs.items()}

    frame = pd.DataFrame({f"p_{k}": p for k, p in probs.items()})
    frame["p_ens"] = p_ens
    frame["regime"] = regime
    frame["signal_ml"] = sig_size          # final pipeline signal, decided at close of t
    frame["is_test"] = frame.index >= test_start

    return {
        "table": table, "predictions": frame, "classification": cls,
        "info": {"ticker": name, "val_start": val_start, "test_start": test_start,
                 "thresholds": th, "ensemble_threshold": th_ens, "allowed_regimes": sorted(allowed),
                 "best_rule": best_rule, "gate_mode": cfg.gate_mode, "gate_schedule": gate_schedule,
                 "regime_perf_val": performance_by_regime(val_returns(sig_ens), regime.loc[val_start:val_end]),
                 "regime_share_val": regime.loc[val_start:val_end].value_counts(normalize=True).sort_index(),
                 "regime_share_test": regime.loc[test_start:].value_counts(normalize=True).sort_index(),
                 "test_pos": int(df.index.get_loc(test_start)),
                 "n_rows": {"validation": int(val_end - val_start + 1), "test": int(len(df) - test_start)}},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "SPY"])
    ap.add_argument("--synthetic", action="store_true", help="use generated data (offline)")
    ap.add_argument("--lstm", action="store_true", help="include the sequence model (slow)")
    ap.add_argument("--kind", default="xgboost", choices=["logistic", "random_forest", "xgboost", "hist_gbm"])
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--initial-train", type=int, default=1000)
    ap.add_argument("--step", type=int, default=63)
    ap.add_argument("--fee-bps", type=float, default=5)
    ap.add_argument("--hysteresis", type=float, default=0.0,
                    help="exit only when P(up) drops this far below the entry threshold (fewer trades)")
    ap.add_argument("--gate", default="static", choices=["static", "rolling", "off"],
                    help="regime gate: static = decide once on validation, rolling = re-decide over time, off = skip it")
    ap.add_argument("--gate-refresh", type=int, default=252, help="rolling gate: re-decide every N rows")
    ap.add_argument("--gate-window", type=int, default=756, help="rolling gate: learn from the trailing N rows")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="device for the LSTM (--lstm). auto picks a GPU if PyTorch can see one")
    ap.add_argument("--period", default="max")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    cfg = Config(horizon=a.horizon, kind=a.kind, initial_train=a.initial_train, step=a.step,
                 use_lstm=a.lstm, fee_bps=a.fee_bps, hysteresis=a.hysteresis, device=a.device,
                 gate_mode=a.gate, gate_refresh=a.gate_refresh, gate_window=a.gate_window)
    if a.lstm:
        import torch
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
        print(f"[LSTM device] requested='{a.device}' -> using '{resolved}'"
              + ("" if torch.cuda.is_available() else "  (no GPU visible to PyTorch)"))
    os.makedirs(a.out, exist_ok=True)
    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.width = 200
    pd.options.display.max_columns = None

    if a.synthetic:
        from ml.synthetic import make_synthetic_ohlcv
        data = {f"SYN{i}": make_synthetic_ohlcv(3500, seed=i, momentum=0.05 + 0.03 * i) for i in range(3)}
    else:
        from ml.data_cache import load_cached_ohlcv
        data = {t: load_cached_ohlcv(t, period=a.period) for t in a.tickers}

    tables = {}
    for name, df in data.items():
        print(f"\n=== {name}: {len(df)} rows ===")
        res = run_ticker(df, name, cfg)
        tables[name] = res["table"]
        i = res["info"]
        print(f"validation from {i['val_start']}, test from {i['test_start']}, gate '{i['gate_mode']}', "
              f"best rule on validation: {i['best_rule']}")
        for k, c in res["classification"].items():
            print(f"  [{k}] test accuracy {c.get('accuracy', float('nan')):.3f} "
                  f"vs majority baseline {c.get('baseline_accuracy', float('nan')):.3f}, "
                  f"AUC {c.get('auc', float('nan')):.3f}")
        print(res["table"])
        print_diagnostics(res)
        save_predictions(name, df, res["predictions"], directory="predictions")
        res["table"].to_csv(os.path.join(a.out, f"ablation_{name}.csv"))

    if len(tables) > 1:
        allrows = pd.concat(tables, names=["Ticker", "Configuration"]).reset_index()
        allrows["Configuration"] = allrows["Configuration"].str.replace(r"Best rule.*", "Best rule (picked on val)", regex=True)
        summary = allrows.groupby("Configuration", sort=False)[
            ["Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown", "Trades"]].mean()
        sig_count = allrows.assign(sig=allrows["Timing p-value"] < 0.05).groupby("Configuration", sort=False)["sig"].sum()
        summary["Tickers with timing p<0.05"] = sig_count.astype(int)
        bh_sharpe = allrows[allrows["Configuration"] == "Buy & hold"].set_index("Ticker")["Sharpe Ratio"]
        wins = {}
        for cfg_name, g in allrows.groupby("Configuration", sort=False):
            g = g.set_index("Ticker")["Sharpe Ratio"]
            wins[cfg_name] = int((g > bh_sharpe.reindex(g.index)).sum())
        summary["Tickers beating B&H (Sharpe)"] = pd.Series(wins)
        print("\n=== Mean across tickers (test slice) ===")
        print(summary)
        summary.to_csv(os.path.join(a.out, "ablation_summary.csv"))


if __name__ == "__main__":
    main()

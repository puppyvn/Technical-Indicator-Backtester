# ML extension: classifier + regime detection + LSTM, combined

Everything here plugs into the existing backtester: every model outputs a **signal column**, and the
existing `run_backtest()` scores it against buy-and-hold with the same metrics as before.

```
features.py -> X      labels.py -> y      walkforward.py -> out-of-sample P(up)
     |                                        |
     |         tree_model.py  (Level 1)  ----+
     |         lstm_model.py  (Level 3)  ----+--> ensemble.py: average -> regime gate -> vol sizing
     +-------> regime.py      (Level 2)  ----+                                  |
                                                                                v
                                              evaluate.py: signal.shift(1) -> run_backtest()
```

## Quick start

```bash
pip install -r requirements.txt -r requirements-ml.txt

python -m pytest tests -q                                   # 42 tests, ~10 s
python -m experiments.run_ablation --synthetic              # full pipeline, no network needed
python -m experiments.run_ablation --tickers AAPL SPY MSFT  # real data (cached in data_cache/)
python -m experiments.run_ablation --tickers AAPL --lstm    # add the sequence model (slow)
python -m experiments.run_ablation --tickers AAPL --hysteresis 0.03   # fewer trades
streamlit run app.py                                        # pick "ML Ensemble (offline)"
```

Outputs: `results/ablation_<ticker>.csv`, `results/ablation_summary.csv`, `predictions/<ticker>.csv`.
The app only *loads* `predictions/`, it never trains anything.

## Adding it to your own (already customised) project

Copy the folders `ml/`, `experiments/`, `tests/` and `requirements-ml.txt` next to your existing code.
Then apply two small patches instead of replacing your files:
`ml/patch_app.diff` (new radio option + ML branch) and `ml/patch_price_charts.diff`
(buy/sell markers that also work for fractional positions). Apply with `patch -p0 app.py < ml/patch_app.diff`
or make the ~15 edits by hand; they are short.

## Reading the ablation table

| Column | Meaning |
|---|---|
| Sharpe / Max Drawdown / Total Return | computed by the existing engine on the **test slice only** |
| Trades | position changes. Hundreds of trades means fees matter; try `--hysteresis` |
| Avg Exposure | average fraction invested. Explains a lot of "performance" |
| **Timing p-value** | circular-shift test: how often the same exposure with unrelated timing does as well. **Small (<0.05) = timing carries information.** |

**Profit is not skill.** In a rising market any strategy that is 60% invested has a positive Sharpe.
The timing p-value is there to catch exactly that. Even so, testing many configurations on many
tickers guarantees a few small p-values by luck, so treat it as a filter, not a verdict.

## Honesty protocol (built in)

* All ML predictions are walk-forward out-of-sample, with a purge of `horizon` rows before each test block.
* The out-of-sample window is split into **validation** (first 40%) and **test**. The probability
  threshold, the "best rule-based strategy" and the regimes allowed to trade are chosen on validation
  only. Every reported number comes from the test slice, on identical dates and fees for all configurations.

## What was verified, and how

* **Leakage tests**: features are causal; training never touches the purge zone; mangling every price
  after day t0 leaves all predictions at or before t0 unchanged (trees, LSTM and regimes).
* **The leak tests can fail**: I planted three deliberate leaks (a feature that peeks a day ahead, no purge,
  training on the test block) in scratch copies. Each was caught by several tests.
* **Controls**: shuffled labels and random probabilities show no skill; a planted signal is found
  (AUC 0.70, timing p = 0.002); on weak-signal data no configuration is credited (p = 0.10 to 0.91).
* **App**: the Streamlit app was run headlessly with the new option (renders, warns gracefully when no
  predictions exist, rule-based strategies still work).

## What was NOT verified (be aware)

* The real Yahoo Finance path (`data_cache.py` -> `fetch_price_history`) was not exercised: my sandbox
  cannot reach Yahoo. It reuses the same `fetch_price_history` as the base project.
* Results on synthetic data say nothing about real markets. They only prove the machinery works.
* The LSTM was trained on CPU with small data. Expect it to need more data, and often to tie or lose to
  XGBoost on daily prices. On the planted-signal test, averaging it with the tree model did not beat the
  tree model alone: an ensemble is only as good as its weakest useful member.

## Design choices worth knowing

* `xgboost` defaults to `n_jobs=1` so runs are reproducible; raise it for speed.
* Regimes use a Gaussian mixture (causal per day). An HMM's `predict` uses the whole sequence you pass in,
  which would leak the future; a causal HMM must decode using data up to t only.
* Vol sizing is scale-down only (`max_leverage=1`): it can cut drawdowns but never adds leverage.
* The first row of every backtest window is flat by construction (no earlier signal exists to act on).

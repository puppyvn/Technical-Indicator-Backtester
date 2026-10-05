"""
Runs the real Streamlit app headlessly (streamlit.testing) with synthetic data, and checks that
the new "ML Ensemble (offline)" strategy renders, uses only the out-of-sample window,
and fails gracefully when no predictions exist.
"""

import os

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import data.collector as collector  # noqa: E402
from ml.io import save_predictions  # noqa: E402
from ml.synthetic import make_synthetic_ohlcv  # noqa: E402
import numpy as np  # noqa: E402


@pytest.fixture()
def synthetic_market(monkeypatch, tmp_path):
    df = make_synthetic_ohlcv(1200, seed=7)
    monkeypatch.setattr(collector, "fetch_price_history", lambda t, p="2y", i="1d", **k: df.copy())
    monkeypatch.setattr(collector, "fetch_company_info", lambda t: {
        "name": "Synthetic Corp", "sector": "Test", "industry": "Test", "market_cap": None, "currency": "USD"})
    monkeypatch.chdir(tmp_path)  # predictions/ lives relative to the working directory
    # need the project on sys.path for AppTest when cwd changes
    return df


def _run_app(strategy: str, ticker: str = "SYN"):
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app.py"), default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(ticker)
    at.sidebar.radio[0].set_value(strategy)
    at.run()
    return at


def test_rule_based_strategy_still_works(synthetic_market):
    at = _run_app("SMA Crossover")
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error


def test_ml_strategy_without_saved_predictions_warns_gracefully(synthetic_market):
    at = _run_app("ML Ensemble (offline)")
    assert not at.exception
    assert any("No saved ML predictions" in w.value for w in at.warning)


def test_ml_strategy_renders_and_uses_only_the_out_of_sample_window(synthetic_market):
    df = synthetic_market
    sig = pd.Series(np.nan, index=df.index)
    sig.iloc[600:] = np.clip(np.random.default_rng(0).normal(0.5, 0.4, len(df) - 600), 0, 1)  # fractional exposure
    save_predictions("SYN", df, pd.DataFrame({"signal_ml": sig}), directory="predictions")

    at = _run_app("ML Ensemble (offline)")
    assert not at.exception, [e.value for e in at.exception]
    assert any("out-of-sample" in i.value for i in at.info)
    rows = [m for m in at.metric if m.label == "Rows Loaded"][0]
    assert int(rows.value) == 1200  # metric shows the loaded data, backtest below is windowed
    # both Plotly figures and the stats table were produced
    assert len(at.tabs) == 2 and len(at.dataframe) >= 1

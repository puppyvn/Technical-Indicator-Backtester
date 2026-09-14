import pandas as pd

from app import apply_strategy
from backtest.engine import run_backtest


def test_run_backtest_accepts_fee_keyword():
    df = pd.DataFrame({
        "Close": [100.0, 101.0, 102.0],
        "position": [1, 1, 0],
    })

    result = run_backtest(df, "position", fee=0.001)

    assert "equity" in result.columns
    assert result["equity"].notna().all()


def test_apply_strategy_uses_macd_line_and_signal_columns():
    df = pd.DataFrame({
        "Close": [100.0, 101.0, 102.0],
        "MACD_line": [1.0, 2.0, 3.0],
        "MACD_signal": [0.5, 1.5, 2.5],
    })

    result = apply_strategy(df, {"strategy": "MACD Crossover"})

    assert "position" in result.columns
    assert result["position"].notna().all()

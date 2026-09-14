# Stock Technical Indicator Backtester

A modular yfinance project split into clean layers: **collect → process → visualize**,
wrapped in an interactive Streamlit dashboard.

## Project Structure

```
Technical_Indicator_Backtester/
├── app.py                     # Streamlit entrypoint (UI glue only)
├── config.py                  # Constants, defaults, ticker lists
├── requirements.txt
│
├── data/
│   ├── __init__.py
│   ├── collector.py            # yfinance calls + caching
│   └── validator.py            # sanity checks on raw data
│
├── indicators/
│   ├── __init__.py
│   └── indicators.py            # SMA, EMA, RSI, MACD, Bollinger Bands
│
├── backtest/
│   ├── __init__.py
│   ├── signals.py               # turn indicators into buy/sell signals
│   └── engine.py                 # simulate strategy vs buy-and-hold
│
└── viz/
    ├── __init__.py
    ├── price_charts.py           # candlestick + indicator overlays
    └── performance_charts.py     # equity curve, drawdown, metrics table
```

## Design principles

1. **`data/`** never plots or computes indicators — it only fetches and validates.
2. **`indicators/`** and **`backtest/`** are pure pandas/numpy functions — no Streamlit,
   no I/O. This makes them independently testable and reusable in a notebook.
3. **`viz/`** takes DataFrames in, returns Plotly figures out — no data fetching.
4. **`app.py`** is intentionally thin: it just wires sidebar inputs → pipeline → charts.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

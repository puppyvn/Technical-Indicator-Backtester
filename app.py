import streamlit as st

import config 
from data.collector import fetch_price_history, fetch_company_info, DataCollectionError
from data.validator import validate_ohlcv, clean_ohlcv
from indicators.indicators import add_all_indicators
from backtest.signals import sma_crossover_signal, rsi_mean_reversion_signal, macd_crossover_signal
from backtest.engine import run_backtest, compute_summary_stats
from viz.price_charts import candlestick_with_overlays, rsi_chart, macd_chart, signal_markers_overlay
from viz.performance_charts import equity_curve_chart, drawdown_chart, stats_to_dataframe

st.set_page_config(page_title="Technical Indicator Backtester", layout="wide")

@st.cache_data(ttl=config.CACHE_TTL_SECONDS, show_spinner=False)
def load_data(ticker: str, period: str, interval: str):
    df = fetch_price_history(ticker, period, interval)
    info = fetch_company_info(ticker)
    return df, info    

def sidebar_inputs() -> dict:
    st.sidebar.header("Ticker & Range")

    preset_choice = st.sidebar.selectbox(
        "Quick pick a list (optional)", ["-- custom --"] + list(config.TICKER_PRESETS.keys())
    )
    if preset_choice != "-- custom --":
        ticker = st.sidebar.selectbox("Ticker", config.TICKER_PRESETS[preset_choice])
    else:
        ticker = st.sidebar.text_input("Ticker", value=config.DEFAULT_TICKER).upper().strip()

    period = st.sidebar.selectbox("Period", config.PERIOD_OPTIONS, index=config.PERIOD_OPTIONS.index(config.DEFAULT_PERIOD))
    interval = st.sidebar.selectbox("Interval", config.INTERVAL_OPTIONS, index=0)

    st.sidebar.header("Indicators")
    sma_short = st.sidebar.slider("SMA short window", 5, 50, config.SMA_SHORT_DEFAULT)
    sma_long = st.sidebar.slider("SMA long window", 20, 200, config.SMA_LONG_DEFAULT)
    ema_short = st.sidebar.slider("EMA short span", 5, 50, config.EMA_SHORT_DEFAULT)
    ema_long = st.sidebar.slider("EMA long span", 20, 100, config.EMA_LONG_DEFAULT)
    rsi_period = st.sidebar.slider("RSI period", 5, 30, config.RSI_PERIOD_DEFAULT)
    macd_signal = st.sidebar.slider("MACD signal span", 5, 20, config.MACD_SIGNAL_DEFAULT)
    bb_window = st.sidebar.slider("Bollinger window", 10, 50, config.BOLLINGER_WINDOW_DEFAULT)
    bb_std = st.sidebar.slider("Bollinger std dev", 1.0, 3.0, float(config.BOLLINGER_STD_DEFAULT), step=0.5)

    st.sidebar.header("Strategy")
    strategy = st.sidebar.radio(
        "Signal to backtest",
        ["SMA Crossover", "RSI Mean Reversion", "MACD Crossover"],
    )
    st.sidebar.header("Backtest Settings")
    capital = st.sidebar.number_input("Initial capital ($)", value=config.INITIAL_CAPITAL, step=1000)
    fee_bps = st.sidebar.slider("Trading fee (bps per trade)", 0, 50, config.TRADING_FEE_BPS)

    return dict(
        ticker=ticker, period=period, interval=interval,
        sma_short=sma_short, sma_long=sma_long,
        ema_short=ema_short, ema_long=ema_long,
        rsi_period=rsi_period, macd_signal=macd_signal,
        bb_window=bb_window, bb_std=bb_std,
        strategy=strategy, capital=capital, fee_bps=fee_bps,
    )

def apply_strategy(df, params):
    if params["strategy"] == "SMA Crossover":
        return sma_crossover_signal(df, f"SMA_{params['sma_short']}", f"SMA_{params['sma_long']}")
    elif params["strategy"] == "RSI Mean Reversion":
        return rsi_mean_reversion_signal(
            df,
            f"RSI_{params['rsi_period']}",
            config.RSI_OVERFSOLD,
            config.RSI_OVERBOUGHT,
        )
    else:
        return macd_crossover_signal(df, "MACD_line", "MACD_signal")

def main():
    st.title("Technical Indicator Backtester")
    st.caption("Data via yfinance - Indicators computed with pandas - Charts with Plotly")

    params = sidebar_inputs()

    if not params["ticker"]:
        st.info("Enter a ticker in the sidebar to get started.")
        return

    with st.spinner(f"Fetching {params["ticker"]}..."):
        try:
            raw_df, info = load_data(params["ticker"], params["period"], params["interval"])
        except DataCollectionError as e:
            st.error(f"Couldn't fetch data: {e}")
            return

    is_valid, message = validate_ohlcv(raw_df)
    if not is_valid:
        st.error(f"Data validation failed: {message}")
        return
    df = clean_ohlcv(raw_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Company", info["name"])
    col2.metric("Sector", info["sector"])
    col3.metric("Last Close", f"${df["Close"].iloc[-1]:.2f}")
    col4.metric("Rows Loaded", len(df))

    df = add_all_indicators(df, params)
    df = apply_strategy(df, params)
    df = run_backtest(df, "position", fee=params["fee_bps"] / 10_000)
    df["strategy_equity"] = df["equity"]
    df["buy_hold_equity"] = df["bh_equity"]
    df["strategy_drawdown"] = (df["equity"] / df["equity"].cummax()) - 1
    stats = compute_summary_stats(df["equity"])

    tab_price, tab_backtest = st.tabs(["Price & Indicators", "Backtest Results"])

    with tab_price:
        sma_cols = [f"SMA_{params['sma_short']}", f"SMA_{params['sma_long']}"]
        fig = candlestick_with_overlays(
            df,
            sma_cols=sma_cols,
            show_bollinger=True,
            title=f"{params['ticker']} Price",
        )
        fig = signal_markers_overlay(fig, df)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                rsi_chart(df, f"RSI_{params['rsi_period']}", config.RSI_OVERBOUGHT, config.RSI_OVERFSOLD),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(macd_chart(df), use_container_width=True)

    with tab_backtest:
        st.plotly_chart(equity_curve_chart(df), use_container_width=True)
        st.plotly_chart(drawdown_chart(df), use_container_width=True)

        st.subheader("Summary Stats")
        st.dataframe(stats_to_dataframe(stats), use_container_width=True)

        st.caption(
            "Note: This is a simplified backtest for educational purposes - it ignores "
            "slippage beyond the flat fee assumption, taxes and position sizing beyond "
            "100% in/out. Not investment advice."
        )


if __name__ == "__main__":
    main()

import pandas as pd
import plotly.graph_objects as go


def equity_curve_chart(df: pd.DataFrame) -> go.Figure:
    x_values = df["Date"] if "Date" in df.columns else df.index
    strategy_equity = df["strategy_equity"] if "strategy_equity" in df.columns else df["equity"]
    buy_hold_equity = df["buy_hold_equity"] if "buy_hold_equity" in df.columns else df["bh_equity"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_values, y=strategy_equity, mode="lines", name="Strategy"))
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=buy_hold_equity,
            mode="lines",
            name="Buy & Hold",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(
        title="Strategy vs. Buy & Hold - Equity Curve",
        yaxis_title="Portfolio Value ($)",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def drawdown_chart(df: pd.DataFrame) -> go.Figure:
    x_values = df["Date"] if "Date" in df.columns else df.index
    if "strategy_drawdown" in df.columns:
        drawdown = df["strategy_drawdown"]
    elif "equity" in df.columns:
        drawdown = (df["equity"] / df["equity"].cummax()) - 1
    else:
        raise KeyError("No drawdown-ready equity column found in DataFrame")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=drawdown * 100,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="crimson"),
        )
    )
    fig.update_layout(
        title="Strategy Drawdown",
        yaxis_title="Drawdown (%)",
        height=250,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig

def stats_to_dataframe(stats: dict) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Metric": list(stats.keys()),
            "Value": [float(v) for v in stats.values()],
        },
        dtype=object,
    )

    pct_metrics = {"CAGR", "Volatility", "Max Drawdown"}
    for idx, metric in enumerate(table["Metric"]):
        value = float(table.at[idx, "Value"])
        if metric in pct_metrics:
            table.at[idx, "Value"] = f"{value:.2%}"
        elif metric == "Sharpe Ratio":
            table.at[idx, "Value"] = f"{value:.2f}"
        else:
            table.at[idx, "Value"] = f"{value:.2f}"

    return table
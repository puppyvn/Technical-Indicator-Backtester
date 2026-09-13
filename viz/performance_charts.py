import pandas as pd
import plotly.graph_objects as go

def equity_cruve_chart(df : pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["strategy_equity"], mode="line", name="Strategy"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["buy_hold_equity"], mode="line", name="Buy & Hold",
                             line=dict(dash="dash")))
    fig.update_layout(
        title="Strategy vs. Buy & Hold - Equity Curve",
        yaxis_title="Portfolio Value ($)",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig

def drawdown_charts(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Data"], y = df["strategy_drawdown"] * 100, mode="lines", name="Drawdown",
        fill = "tozeroy", line=dict(color="crimson"),
    ))
    fig.updatge_layout(
        title="Strategy Drawdown",
        yaxis_title="Drawdown (%)",
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

def stats_to_dataframe(stats: dict) -> pd.DataFrame:
    table = pd.DataFrame(stats).astype(object)
    pct_rows = ["Total Return", "CAGR", "Annualized Volatility", "Max Drawdown"]
    for row in pct_rows:
        if row in table.index:
            table.loc[row]= table.loc[row].apply(lambda x: f"{x:.2%}")
    if "Sharpe Ratio" in table.index:
        table.loc["Sharpe Ratio"] = table.loc["Sharpe Ratio"].apply(lambda x: f"{x:.2f}")
    return table
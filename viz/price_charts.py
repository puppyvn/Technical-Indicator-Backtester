import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def candlestick_with_overlays(
    df: pd.DataFrame,
    sma_cols: list[str] | None = None,
    show_bollinger: bool = False,
    title: str = "Price",
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )

    for col in sma_cols or []:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode="lines", line=dict(width=1.5)))

    if show_bollinger:
        bb_upper = next((c for c in df.columns if c.startswith("BB_upper_")), None)
        bb_lower = next((c for c in df.columns if c.startswith("BB_lower_")), None)
        if bb_upper and bb_lower:
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df[bb_upper],
                    mode="lines",
                    name="BB Upper",
                    line=dict(width=1, dash="dot"),
                    showlegend=True,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df[bb_lower],
                    mode="lines",
                    name="BB Lower",
                    line=dict(width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(100,100,255,0.08)",
                    showlegend=True,
                )
            )
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig

def rsi_chart(df: pd.DataFrame, rsi_col: str, overbought: int= 70, oversold: int = 30) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df[rsi_col], mode="lines", name=rsi_col))
    fig.add_hline(y=overbought, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=oversold, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=20, b=10), yaxis_range=[0, 100])
    return fig

def macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_hist"], name="Histogram", marker_color="lightgray"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_line"], mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_signal"], mode="lines", name="Signal"))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig

def signal_markers_overlay(fig: go.Figure, df: pd.DataFrame, price_col: str = "Close") -> go.Figure:
    position_diff = df["position"].diff()
    buys = df[position_diff == 1]
    sells = df[position_diff == -1]

    fig.add_trace(go.Scatter(
        x=buys["Date"], y=buys[price_col], mode="markers", name="Buy",
        marker=dict(symbol="triangle-up", size=11, color="green"),
    ))
    fig.add_trace(go.Scatter(
        x=sells["Date"], y=sells[price_col], mode="markers", name="Sell",
        marker=dict(symbol="triangle-down", size=11, color="red"),
    ))
    return fig
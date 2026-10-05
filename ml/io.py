"""Save / load predictions so the Streamlit app never trains anything itself."""

import os

import pandas as pd


def save_predictions(ticker: str, df: pd.DataFrame, frame: pd.DataFrame, directory: str = "predictions") -> str:
    """`frame` is indexed like `df` and holds columns such as p_tree, p_seq, p_ens, regime, signal_ml."""
    os.makedirs(directory, exist_ok=True)
    out = frame.copy()
    out.insert(0, "Date", df["Date"].to_numpy())
    path = os.path.join(directory, f"{ticker.upper()}.csv")
    out.to_csv(path, index=False)
    return path


def load_saved_signal(ticker: str, df: pd.DataFrame, column: str = "signal_ml",
                      directory: str = "predictions") -> pd.Series | None:
    """Align a saved signal to `df` by Date. Returns None if no file exists for the ticker."""
    path = os.path.join(directory, f"{ticker.upper()}.csv")
    if not os.path.exists(path):
        return None
    saved = pd.read_csv(path, parse_dates=["Date"])
    if column not in saved.columns:
        return None
    s = saved.set_index("Date")[column]
    aligned = pd.Series(pd.to_datetime(df["Date"]).map(s).to_numpy(), index=df.index, name=column)
    return aligned

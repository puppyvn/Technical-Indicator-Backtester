"""
Labels (Phase M2). This is the ONE place where looking forward is required.

label[t] = 1 if Close[t + horizon] / Close[t] - 1 > threshold else 0
The last `horizon` rows have no label (NaN). The walk-forward loop purges
training rows whose labels overlap the test block.
"""

import numpy as np
import pandas as pd


def forward_returns(df: pd.DataFrame, horizon: int = 5, price_col: str = "Close") -> pd.Series:
    """Return over the next `horizon` rows (NaN for the last `horizon` rows)."""
    return df[price_col].shift(-horizon) / df[price_col] - 1


def build_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.0,
    price_col: str = "Close",
) -> pd.Series:
    fwd = forward_returns(df, horizon, price_col)
    y = (fwd > threshold).astype(float)
    y[fwd.isna()] = np.nan
    y.name = f"up_{horizon}d"
    return y

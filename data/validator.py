"""
Phase 2 — Data Validation & Cleaning

Validates that an OHLCV DataFrame is usable (right shape, not too messy,
properly ordered), and cleans up minor issues like scattered NaNs.
"""

import pandas as pd

REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume"}
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
MAX_NAN_FRACTION = 0.05  # more than 5% missing in price columns is too messy


def validate_ohlcv(df: pd.DataFrame, min_rows: int = 30) -> tuple[bool, str]:
    """
    Validate an OHLCV DataFrame.

    Checks (accumulated, not short-circuited, so the caller gets the full
    picture of what's wrong in one pass rather than fixing issues one at a
    time):
      - required columns are present
      - enough rows to be useful
      - price columns aren't too riddled with NaNs
      - dates are sorted ascending

    Returns:
        (is_valid, message) — message is "OK" when valid, otherwise a
        semicolon-separated list of every problem found.
    """
    problems: list[str] = []

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        problems.append(f"missing required columns: {sorted(missing)}")

    if len(df) < min_rows:
        problems.append(f"only {len(df)} rows, need at least {min_rows}")

    # Only check NaN fraction on price columns that actually exist —
    # otherwise a missing-column problem cascades into a KeyError here.
    present_price_cols = [c for c in PRICE_COLUMNS if c in df.columns]
    if present_price_cols:
        nan_fraction = df[present_price_cols].isna().mean().mean()
        if nan_fraction > MAX_NAN_FRACTION:
            problems.append(
                f"price columns are {nan_fraction:.1%} NaN "
                f"(max allowed is {MAX_NAN_FRACTION:.0%})"
            )

    if "Date" in df.columns:
        if not df["Date"].is_monotonic_increasing:
            problems.append("dates are not sorted in ascending order")

    if problems:
        return False, "; ".join(problems)
    return True, "OK"


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean an OHLCV DataFrame: forward-fill gaps in price columns and drop
    any rows that still have no valid data.

    Volume is deliberately NOT forward-filled — a missing volume reading
    means "we don't know how many shares traded that period," and carrying
    yesterday's volume forward would fabricate a number rather than
    reasonably estimate a price (which price forward-fill does, under the
    assumption the price didn't move).

    This function does not re-sort by date — sort order is a validation
    concern (see validate_ohlcv), not something cleaning should silently
    paper over.
    """
    cleaned = df.copy()

    present_price_cols = [c for c in PRICE_COLUMNS if c in cleaned.columns]
    cleaned[present_price_cols] = cleaned[present_price_cols].ffill()

    # Leading NaNs (no prior value to forward-fill from) can't be fixed —
    # drop those rows.
    cleaned = cleaned.dropna(subset=present_price_cols)

    return cleaned.reset_index(drop=True)

"""
Phase 1 — Data Collection Layer

Fetches OHLCV price history from yfinance with retry logic, and batches
fetches across multiple tickers while isolating per-ticker failures.
"""

import time
import logging
import pandas as pd
import yfinance as yf


class DataCollectionError(Exception):
    """Raised when a ticker's price history cannot be fetched after all retries."""
    pass


def fetch_price_history(
    ticker: str, period: str, interval: str, max_retries: int = 3, retry_delay: float = 1.5,
) -> pd.DataFrame:
    """
    Fetch OHLCV price history for a single ticker, retrying on failure.

    A "failure" includes both raised exceptions and a silently-empty
    DataFrame (yfinance does this on bad tickers or transient network
    issues instead of always raising).

    Raises:
        DataCollectionError: if all `max_retries` attempts fail.
    """
    last_error: str | None = None

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval)

            if df.empty:
                last_error = "received an empty DataFrame"
            else:
                # Date currently lives in the index; promote it to a real
                # column so downstream code (e.g. Plotly) can use it directly.
                df = df.reset_index()
                return df

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < max_retries:
            # Growing backoff: 1x, 2x, 3x... the base delay.
            base_delay = 1.5
            time.sleep(attempt * base_delay)

    raise DataCollectionError(
        f"Failed to fetch price history for '{ticker}' after {max_retries} "
        f"attempts. Last error: {last_error}"
    )


def fetch_multiple(
    tickers: list[str], period: str, interval: str
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV price history for multiple tickers.

    Failures on individual tickers are caught and skipped so that one bad
    ticker doesn't abort the whole batch. Only successful fetches are
    included in the returned dict.
    """
    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        try:
            results[ticker] = fetch_price_history(ticker, period, interval)
        except DataCollectionError as exc:
            print(f"Skipping '{ticker}': {exc}")

    return results

def fetch_company_info(ticker: str) -> dict:
    """Lightweight metadata fetch (name, sector, market cap) for display purposes."""
    try:
        info = yf.Ticker(ticker).get_info()
        return {
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not fetch info for %s: %s", ticker, e)
        return {"name": ticker, "sector": "N/A", "industry": "N/A", "market_cap": None, "currency": "USD"}

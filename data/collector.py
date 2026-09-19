"""
Collection layer. Only responsibility: talk to yfinance, return clean-ish
raw DataFrames. No indicator math, no plotting happens here.
"""

import time
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataCollectionError(Exception):
    """Raised when yfinance data can't be retrieved after retries."""
\

def fetch_price_history(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    max_retries: int = 3,
    retry_delay: float = 1.5, 
) -> pd.DataFrame:
    """
    Fetch OHLCV history for a single ticker with basic retry handling.

    yfinance occasionally returns empty frames on transient failures
    (rate limiting, network hiccups) rather than raising — so we treat
    an empty frame as a retryable failure too.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval)
            if df is None or df.empty:
                raise DataCollectionError(f"Empty data returned for '{ticker}'")

            df = df.reset_index()
            # yfinance names the date column "Date" for daily/weekly/monthly;
            # normalize just in case.
            date_col = "Date" if "Date" in df.columns else df.columns[0]
            df = df.rename(columns={date_col: "Date"})
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            return df

        except Exception as e:  # noqa: BLE001 - we want to retry broadly here
            last_error = e
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, max_retries, ticker, e)
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)  # simple backoff

    raise DataCollectionError(
        f"Failed to fetch data for '{ticker}' after {max_retries} attempts"
    ) from last_error


def fetch_multiple(
    tickers: list[str],
    period: str = "2y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch several tickers, skipping ones that fail rather than aborting all."""
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = fetch_price_history(ticker, period, interval)
        except DataCollectionError as e:
            logger.error("Skipping %s: %s", ticker, e)
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

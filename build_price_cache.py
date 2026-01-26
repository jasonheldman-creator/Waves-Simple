#!/usr/bin/env python3
"""
build_price_cache.py

Canonical price cache builder for WAVES.
This script MUST either:
- produce a fully valid cache + metadata
- OR fail loudly and exit non-zero

There is no degraded success path.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List

import pandas as pd
import yfinance as yf

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
TICKERS_FILE = "data/wave_weights.csv"  # Source tickers from wave_weights.csv
CACHE_DIR = "data/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "prices_cache.parquet")
META_FILE = os.path.join(CACHE_DIR, "prices_cache_meta.json")
MIN_TRADING_DAYS = 252  # Ensure at least 1 trading year of data

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("build_price_cache")


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def hard_fail(msg: str) -> None:
    log.error(msg)
    sys.exit(1)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------------------------
# Ticker Loading
# ------------------------------------------------------------------------------
def load_tickers_from_weights() -> List[str]:
    """
    Load tickers deterministically from wave_weights.csv.
    """
    log.info("Loading tickers from wave_weights.csv...")
    if not os.path.exists(TICKERS_FILE):
        hard_fail("wave_weights.csv not found")

    df = pd.read_csv(TICKERS_FILE)
    required_columns = {"wave_id", "ticker", "weight"}
    if not required_columns.issubset(df.columns):
        hard_fail(f"wave_weights.csv missing required columns: {required_columns - set(df.columns)}")

    tickers = (
        df["ticker"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )
    tickers = sorted(set(tickers))  # De-duplicate and sort

    if not tickers:
        hard_fail("No tickers found in wave_weights.csv")

    log.info(f"Loaded {len(tickers)} tickers from wave_weights.csv")
    return tickers


# ------------------------------------------------------------------------------
# Price Data Fetching
# ------------------------------------------------------------------------------
def fetch_price_data(tickers: List[str], lookback_years: int = 2) -> pd.DataFrame:
    """
    Fetch historical price data for the given tickers.
    Ensures sufficient lookback depth (≥ MIN_TRADING_DAYS).

    :param tickers: List of ticker symbols to fetch prices for.
    :param lookback_years: Number of years back to fetch data.
    :return: Wide-form DataFrame with tickers as columns and DatetimeIndex.
    """
    start_date = (datetime.now() - pd.DateOffset(years=lookback_years)).date()
    log.info(f"Fetching price data starting from {start_date}...")

    all_data = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date)
            if "Close" in data.columns:
                all_data[ticker] = data["Close"]
            else:
                log.warning(f"No 'Close' column found for ticker {ticker}, skipping...")
        except Exception as e:
            log.warning(f"Failed to fetch data for ticker {ticker}: {e}")

    price_data = pd.DataFrame(all_data)
    price_data = price_data.dropna(how="all")  # Drop rows with all NaN

    if price_data.empty or len(price_data.index) < MIN_TRADING_DAYS:
        hard_fail("Insufficient trading data fetched. Ensure tickers and dates are valid.")

    log.info(f"Fetched price data with shape {price_data.shape}")
    return price_data


# ------------------------------------------------------------------------------
# Cache Validation
# ------------------------------------------------------------------------------
def validate_cache(cache: pd.DataFrame, tickers: List[str]) -> None:
    """
    Validate the built cache to ensure trading days, tickers, and recent data.

    :param cache: Pandas DataFrame of the price cache.
    :param tickers: List of expected tickers.
    """
    log.info("Validating cache integrity...")

    missing_tickers = [ticker for ticker in tickers if ticker not in cache.columns]
    if missing_tickers:
        hard_fail(f"Cache validation failed. Missing tickers: {missing_tickers}")

    if cache.index.max() < pd.Timestamp.now().normalize():
        hard_fail("Cache max date is stale. Ensure the data is current.")

    if len(cache.index) < MIN_TRADING_DAYS:
        hard_fail(f"Cache has insufficient trading days ({len(cache.index)} < {MIN_TRADING_DAYS})")

    log.info("Cache validation passed.")


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    ensure_dir(CACHE_DIR)
    tickers = load_tickers_from_weights()
    price_data = fetch_price_data(tickers)
    validate_cache(price_data, tickers)

    # Save to parquet file
    price_data.to_parquet(CACHE_FILE)
    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "tickers_fetched": len(tickers),
        "max_date": price_data.index.max().strftime("%Y-%m-%d"),
        "min_date": price_data.index.min().strftime("%Y-%m-%d"),
        "trading_days": len(price_data.index),
    }
    with open(META_FILE, "w") as metafile:
        json.dump(metadata, metafile, indent=4)

    log.info(f"Price cache successfully built: {CACHE_FILE}")
    log.info(f"Metadata written: {META_FILE}")


if __name__ == "__main__":
    main()
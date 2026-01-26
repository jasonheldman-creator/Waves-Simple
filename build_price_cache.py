#!/usr/bin/env python3
"""
build_price_cache.py

Canonical price cache builder for WAVES.

CI-safe behavior:
- Batch-downloads ALL tickers in one Yahoo request
- Skips bad tickers safely
- Fails only if resulting cache is invalid
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
TICKERS_FILE = "data/wave_weights.csv"
CACHE_DIR = "data/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "prices_cache.parquet")
META_FILE = os.path.join(CACHE_DIR, "prices_cache_meta.json")

LOOKBACK_YEARS = 2
MIN_TRADING_DAYS = 252

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
# Load tickers
# ------------------------------------------------------------------------------
def load_tickers_from_weights() -> List[str]:
    log.info("Loading tickers from wave_weights.csv...")

    if not os.path.exists(TICKERS_FILE):
        hard_fail("wave_weights.csv not found")

    df = pd.read_csv(TICKERS_FILE)

    required = {"wave_id", "ticker", "weight"}
    if not required.issubset(df.columns):
        hard_fail(f"wave_weights.csv missing columns: {required - set(df.columns)}")

    tickers = (
        df["ticker"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )

    if not tickers:
        hard_fail("No tickers found")

    tickers = sorted(set(tickers))
    log.info(f"Loaded {len(tickers)} tickers")
    return tickers

# ------------------------------------------------------------------------------
# Fetch prices (BATCH, CI-safe)
# ------------------------------------------------------------------------------
def fetch_price_data(tickers: List[str]) -> (pd.DataFrame, List[str]):
    start_date = (datetime.utcnow() - pd.DateOffset(years=LOOKBACK_YEARS)).date()
    log.info(f"Batch fetching prices from {start_date}")

    ticker_str = " ".join(tickers)

    data = yf.download(
        ticker_str,
        start=start_date,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if data.empty:
        hard_fail("Yahoo returned empty dataset (rate limit / outage)")

    series_list = []
    failed = []

    for ticker in tickers:
        try:
            if ticker not in data.columns.levels[0]:
                raise ValueError("Ticker missing in batch response")

            close = data[ticker]["Close"].dropna()

            if not isinstance(close.index, pd.DatetimeIndex):
                raise ValueError("Invalid index")

            if len(close) < MIN_TRADING_DAYS:
                raise ValueError("Insufficient history")

            close.name = ticker
            series_list.append(close)

        except Exception as e:
            log.warning(f"Skipping {ticker}: {e}")
            failed.append(ticker)

    if not series_list:
        hard_fail("ALL tickers invalid after batch download")

    prices = pd.concat(series_list, axis=1).sort_index()

    if prices.empty or len(prices) < MIN_TRADING_DAYS:
        hard_fail("Final cache invalid")

    log.info(f"Final cache: {prices.shape[0]} days × {prices.shape[1]} tickers")

    return prices, failed

# ------------------------------------------------------------------------------
# Validate cache
# ------------------------------------------------------------------------------
def validate_cache(prices: pd.DataFrame) -> None:
    latest = prices.index.max()
    today = pd.Timestamp.utcnow().normalize()

    if (today - latest).days > 5:
        hard_fail(f"Cache stale: latest date {latest.date()}")

    if len(prices) < MIN_TRADING_DAYS:
        hard_fail("Insufficient trading history")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    ensure_dir(CACHE_DIR)

    tickers = load_tickers_from_weights()
    prices, failed = fetch_price_data(tickers)
    validate_cache(prices)

    prices.to_parquet(CACHE_FILE)

    meta = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "tickers_requested": len(tickers),
        "tickers_used": prices.shape[1],
        "tickers_failed": failed,
        "min_date": prices.index.min().strftime("%Y-%m-%d"),
        "max_date": prices.index.max().strftime("%Y-%m-%d"),
        "trading_days": len(prices),
    }

    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("✅ Price cache build SUCCESS")

if __name__ == "__main__":
    main()
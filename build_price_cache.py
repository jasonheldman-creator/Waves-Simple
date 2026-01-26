#!/usr/bin/env python3
"""
build_price_cache.py

Canonical price cache builder for WAVES.

This script MUST either:
- produce a fully valid cache + metadata
- OR fail loudly and exit non-zero

There is NO degraded success path.
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

MIN_TRADING_DAYS = 252
LOOKBACK_YEARS = 2

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
    log.info("Loading tickers from wave_weights.csv...")

    if not os.path.exists(TICKERS_FILE):
        hard_fail("wave_weights.csv not found")

    df = pd.read_csv(TICKERS_FILE)
    required = {"wave_id", "ticker", "weight"}

    missing = required - set(df.columns)
    if missing:
        hard_fail(f"wave_weights.csv missing required columns: {missing}")

    tickers = (
        df["ticker"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )

    tickers = sorted(set(tickers))

    if not tickers:
        hard_fail("No tickers found in wave_weights.csv")

    log.info("Loaded %d tickers", len(tickers))
    return tickers

# ------------------------------------------------------------------------------
# Price Fetching
# ------------------------------------------------------------------------------
def fetch_price_data(tickers: List[str]) -> pd.DataFrame:
    start_date = (datetime.utcnow() - pd.DateOffset(years=LOOKBACK_YEARS)).date()
    log.info("Fetching prices starting from %s", start_date)

    series_list = {}

    for ticker in tickers:
        log.info("Fetching %s", ticker)
        data = yf.download(ticker, start=start_date, progress=False)

        if data.empty:
            hard_fail(f"{ticker}: no data returned from Yahoo Finance")

        if "Close" not in data.columns:
            hard_fail(f"{ticker}: missing Close column")

        close = data["Close"].dropna()

        if len(close) < MIN_TRADING_DAYS:
            hard_fail(
                f"{ticker}: insufficient history "
                f"({len(close)} < {MIN_TRADING_DAYS})"
            )

        series_list[ticker] = close

    if not series_list:
        hard_fail("No valid price series fetched")

    price_df = pd.concat(series_list, axis=1)
    price_df.index = pd.to_datetime(price_df.index)

    log.info(
        "Price cache assembled: %d dates × %d tickers",
        price_df.shape[0],
        price_df.shape[1],
    )

    return price_df

# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------
def validate_cache(cache: pd.DataFrame, tickers: List[str]) -> None:
    log.info("Validating cache integrity...")

    missing = [t for t in tickers if t not in cache.columns]
    if missing:
        hard_fail(f"Cache missing tickers: {missing}")

    max_date = cache.index.max()
    if max_date < pd.Timestamp.utcnow().normalize():
        hard_fail(f"Cache stale: max date {max_date}")

    if len(cache.index) < MIN_TRADING_DAYS:
        hard_fail("Cache has insufficient trading days")

    log.info("Cache validation PASSED")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    ensure_dir(CACHE_DIR)

    tickers = load_tickers_from_weights()
    prices = fetch_price_data(tickers)
    validate_cache(prices, tickers)

    prices.to_parquet(CACHE_FILE)

    meta = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "tickers": len(tickers),
        "min_date": prices.index.min().strftime("%Y-%m-%d"),
        "max_date": prices.index.max().strftime("%Y-%m-%d"),
        "trading_days": len(prices.index),
    }

    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Price cache written: %s", CACHE_FILE)
    log.info("Metadata written: %s", META_FILE)


if __name__ == "__main__":
    main()
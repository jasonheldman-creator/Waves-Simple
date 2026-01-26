#!/usr/bin/env python3
"""
build_price_cache.py

Purpose:
- Build a canonical prices_cache.parquet for WAVES
- Be CI-safe and robust to bad tickers
- Skip bad/missing/delisted tickers instead of failing
- Validate the final cache for recency and trading day depth

Requirements:
1. Always normalize timestamps to UTC tz-naive before any date arithmetic.
2. Skip bad/missing/delisted tickers and continue building the cache.
3. Validate that the cache:
   - Contains tz-naive UTC timestamps.
   - Has at least 252 trading days.
   - Is less than 5 days old.
4. Output:
   - data/cache/prices_cache.parquet
   - data/cache/prices_cache_meta.json
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import yfinance as yf

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
WAVE_WEIGHTS_FILE = "data/wave_weights.csv"
CACHE_DIR = "data/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "prices_cache.parquet")
META_FILE = os.path.join(CACHE_DIR, "prices_cache_meta.json")
LOOKBACK_YEARS = 2
MIN_TRADING_DAYS = 252
MAX_CACHE_AGE_DAYS = 5

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
def load_tickers() -> List[str]:
    log.info(f"Loading tickers from {WAVE_WEIGHTS_FILE}...")

    if not os.path.exists(WAVE_WEIGHTS_FILE):
        hard_fail(f"wave_weights.csv not found at {WAVE_WEIGHTS_FILE}")

    df = pd.read_csv(WAVE_WEIGHTS_FILE)
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

    if not tickers:
        hard_fail("No tickers found in wave_weights.csv")

    log.info(f"Loaded {len(tickers)} unique tickers")
    return tickers


# ------------------------------------------------------------------------------
# Fetch price data
# ------------------------------------------------------------------------------
def fetch_price_data(tickers: List[str]) -> pd.DataFrame:
    start_date = (datetime.utcnow() - timedelta(days=LOOKBACK_YEARS * 365)).strftime("%Y-%m-%d")
    log.info(f"Fetching price data for {len(tickers)} tickers starting from {start_date}...")

    try:
        all_data = yf.download(
            tickers,
            start=start_date,
            auto_adjust=True,
            group_by="ticker",
            threads=True
        )
    except Exception as e:
        hard_fail(f"yfinance batch download failed: {e}")

    # --- Normalize single-ticker edge case ---
    if not isinstance(all_data.columns, pd.MultiIndex):
        if len(tickers) == 1:
            all_data = pd.concat({tickers[0]: all_data}, axis=1)
        else:
            log.warning("yfinance returned unexpected structure; proceeding cautiously")

    price_data = {}

    for ticker in tickers:
        try:
            series = all_data[ticker]["Close"]
            if not isinstance(series, pd.Series):
                log.warning(f"{ticker}: no Close series found, skipping")
                continue

            series = series.dropna()
            if len(series) < MIN_TRADING_DAYS:
                log.warning(f"{ticker}: insufficient history ({len(series)} days), skipping")
                continue

            price_data[ticker] = series

        except Exception as e:
            log.warning(f"{ticker}: failed to process ({e}), skipping")

    if not price_data:
        hard_fail("No valid tickers produced usable price data")

    price_df = pd.DataFrame(price_data).dropna(how="all")
    log.info(f"Final price cache shape: {price_df.shape}")
    return price_df


# ------------------------------------------------------------------------------
# Validate cache
# ------------------------------------------------------------------------------
def validate_cache(cache: pd.DataFrame) -> None:
    if cache.empty:
        hard_fail("Cache validation failed: empty DataFrame")

    # Normalize timestamps
    cache.index = pd.to_datetime(cache.index, utc=True).tz_convert(None)

    today = pd.to_datetime(datetime.utcnow())
    max_date = cache.index.max()
    trading_days = len(cache.index)

    if (today - max_date).days > MAX_CACHE_AGE_DAYS:
        hard_fail(
            f"Cache stale: max date {max_date.strftime('%Y-%m-%d')} "
            f"is older than {MAX_CACHE_AGE_DAYS} days"
        )

    if trading_days < MIN_TRADING_DAYS:
        hard_fail(
            f"Cache insufficient history: {trading_days} < {MIN_TRADING_DAYS}"
        )

    log.info(
        f"Cache validation passed | max_date={max_date.date()} | trading_days={trading_days}"
    )


# ------------------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------------------
def write_metadata(cache: pd.DataFrame, requested: List[str]) -> None:
    used = list(cache.columns)
    failed = sorted(set(requested) - set(used))

    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "tickers_requested": len(requested),
        "tickers_used": len(used),
        "tickers_failed": failed,
        "min_date": cache.index.min().strftime("%Y-%m-%d"),
        "max_date": cache.index.max().strftime("%Y-%m-%d"),
        "trading_days": len(cache.index),
    }

    with open(META_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    log.info(f"Metadata written to {META_FILE}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    ensure_dir(CACHE_DIR)

    tickers = load_tickers()
    cache = fetch_price_data(tickers)
    validate_cache(cache)

    cache.to_parquet(CACHE_FILE)
    write_metadata(cache, tickers)

    log.info(f"Price cache successfully written to {CACHE_FILE}")


if __name__ == "__main__":
    main()
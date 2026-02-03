#!/usr/bin/env python3
"""
build_price_cache.py

Canonical price cache builder for WAVES.

Behavior:
- Loads tickers from data/wave_weights.csv
- Fetches prices via Yahoo Finance
- Skips bad / missing / delisted tickers
- Normalizes timestamps to UTC tz-naive
- Requires >=252 trading days
- Requires cache freshness <=5 days
- Writes prices_cache.parquet + metadata
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
    if not os.path.exists(WAVE_WEIGHTS_FILE):
        hard_fail(f"Missing {WAVE_WEIGHTS_FILE}")

    df = pd.read_csv(WAVE_WEIGHTS_FILE)
    required = {"wave_id", "ticker", "weight"}
    if not required.issubset(df.columns):
        hard_fail("wave_weights.csv missing required columns")

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

    log.info(f"Loaded {len(tickers)} tickers")
    return tickers

# ------------------------------------------------------------------------------
# Fetch prices
# ------------------------------------------------------------------------------
def fetch_price_data(tickers: List[str]) -> pd.DataFrame:
    start_date = (datetime.utcnow() - timedelta(days=LOOKBACK_YEARS * 365)).strftime("%Y-%m-%d")
    log.info(f"Fetching prices since {start_date}")

    try:
        raw = yf.download(
            tickers,
            start=start_date,
            auto_adjust=True,
            group_by="ticker",
            threads=True
        )
    except Exception as e:
        hard_fail(f"Yahoo fetch failed: {e}")

    prices = {}

    for ticker in tickers:
        try:
            s = raw[ticker]["Close"]
            if isinstance(s, pd.Series) and len(s) >= MIN_TRADING_DAYS:
                prices[ticker] = s
            else:
                log.warning(f"Skipping {ticker}: insufficient data")
        except Exception:
            log.warning(f"Skipping {ticker}: no data")

    if not prices:
        hard_fail("All tickers failed — no price data")

    df = pd.DataFrame(prices).dropna(how="all")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

    log.info(f"Final cache shape: {df.shape}")
    return df

# ------------------------------------------------------------------------------
# Validate cache
# ------------------------------------------------------------------------------
def validate_cache(df: pd.DataFrame) -> None:
    if df.empty:
        hard_fail("Cache empty")

    today = pd.Timestamp.utcnow().tz_convert(None)
    max_date = df.index.max()

    if (today - max_date).days > MAX_CACHE_AGE_DAYS:
        hard_fail(f"Cache stale: max date {max_date}")

    if len(df) < MIN_TRADING_DAYS:
        hard_fail("Insufficient trading days")

    log.info("Cache validation passed")

# ------------------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------------------
def write_metadata(df: pd.DataFrame, requested: List[str]) -> None:
    failed = sorted(set(requested) - set(df.columns))

    meta = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "tickers_requested": len(requested),
        "tickers_used": len(df.columns),
        "tickers_failed": failed,
        "min_date": df.index.min().strftime("%Y-%m-%d"),
        "max_date": df.index.max().strftime("%Y-%m-%d"),
        "trading_days": len(df),
    }

    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    ensure_dir(CACHE_DIR)

    tickers = load_tickers()
    prices = fetch_price_data(tickers)
    validate_cache(prices)

    prices.to_parquet(CACHE_FILE)
    write_metadata(prices, tickers)

    log.info("Price cache written successfully")

if __name__ == "__main__":
    main()
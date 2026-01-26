#!/usr/bin/env python3
"""
build_price_cache.py

Purpose:
- Build a canonical prices_cache.parquet for WAVES
- Be CI-safe and robust to bad tickers
- Skip bad/missing/delisted tickers instead of failing
- Validate the final cache for recency and trading day depth

Requirements:
1. Load tickers exclusively from data/wave_weights.csv.
2. Fetch prices using Yahoo Finance in batch requests.
3. Handle bad/missing tickers gracefully (log and skip).
4. Validate the resulting cache:
   - Timestamps must be UTC tz-naive.
   - Cache must be <5 days old and ≥252 trading days.
5. Output:
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
def hard_fail(message: str) -> None:
    log.error(message)
    sys.exit(1)


def ensure_dir(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

# ------------------------------------------------------------------------------
# Ticker Loading
# ------------------------------------------------------------------------------
def load_tickers() -> List[str]:
    log.info(f"Loading tickers from {WAVE_WEIGHTS_FILE}...")

    if not os.path.exists(WAVE_WEIGHTS_FILE):
        hard_fail(f"wave_weights.csv not found at {WAVE_WEIGHTS_FILE}")

    df = pd.read_csv(WAVE_WEIGHTS_FILE)
    required_columns = {"wave_id", "ticker", "weight"}
    missing = required_columns - set(df.columns)
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

    if not tickers:
        hard_fail("No tickers found in wave_weights.csv")

    log.info(f"Loaded {len(tickers)} unique tickers")
    return tickers

# ------------------------------------------------------------------------------
# Price Fetching
# ------------------------------------------------------------------------------
def fetch_price_data(tickers: List[str]) -> pd.DataFrame:
    start_date = (datetime.utcnow() - timedelta(days=LOOKBACK_YEARS * 365)).strftime("%Y-%m-%d")
    log.info(f"Fetching price data starting from {start_date}")

    try:
        raw = yf.download(
            tickers,
            start=start_date,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as e:
        hard_fail(f"Yahoo Finance batch download failed: {e}")

    price_series = {}
    for ticker in tickers:
        try:
            s = raw[ticker]["Close"]
            if isinstance(s, pd.Series) and s.notna().sum() >= MIN_TRADING_DAYS:
                price_series[ticker] = s
            else:
                log.warning(f"Skipping {ticker}: insufficient data")
        except Exception:
            log.warning(f"Skipping {ticker}: no usable price series")

    if not price_series:
        hard_fail("All tickers failed — no valid price data produced")

    price_df = pd.DataFrame(price_series).dropna(how="all")

    # Ensure tz-naive DatetimeIndex (critical)
    price_df.index = pd.to_datetime(price_df.index).tz_localize(None)

    log.info(f"Final price cache shape: {price_df.shape}")
    return price_df

# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------
def validate_cache(cache: pd.DataFrame) -> None:
    if cache.empty:
        hard_fail("Cache validation failed: cache is empty")

    max_date = cache.index.max()
    today = pd.Timestamp.utcnow().normalize()

    age_days = (today - max_date).days
    if age_days > MAX_CACHE_AGE_DAYS:
        hard_fail(
            f"Cache validation failed: max date {max_date.date()} "
            f"is {age_days} days old (> {MAX_CACHE_AGE_DAYS})"
        )

    if len(cache.index) < MIN_TRADING_DAYS:
        hard_fail(
            f"Cache validation failed: only {len(cache.index)} trading days "
            f"(min required {MIN_TRADING_DAYS})"
        )

    log.info(
        f"Cache validated successfully "
        f"(max_date={max_date.date()}, trading_days={len(cache.index)})"
    )

# ------------------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------------------
def write_metadata(
    cache: pd.DataFrame,
    tickers_requested: List[str],
) -> None:
    used = set(cache.columns)
    failed = sorted(set(tickers_requested) - used)

    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "tickers_requested": len(tickers_requested),
        "tickers_used": len(used),
        "tickers_failed": failed,
        "min_date": cache.index.min().strftime("%Y-%m-%d"),
        "max_date": cache.index.max().strftime("%Y
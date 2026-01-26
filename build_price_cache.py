#!/usr/bin/env python3
"""
build_price_cache.py

Purpose:
- Build a canonical prices_cache.parquet for WAVES
- Be CI-safe and robust to bad tickers
- Skip bad/missing/delisted tickers instead of failing
- Validate the final cache for recency and trading day depth
"""

import os
import sys
import json
import logging
from datetime import timedelta
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
# Ticker Loading
# ------------------------------------------------------------------------------
def load_tickers() -> List[str]:
    log.info("Loading tickers from wave_weights.csv")

    if not os.path.exists(WAVE_WEIGHTS_FILE):
        hard_fail("wave_weights.csv not found")

    df = pd.read_csv(WAVE_WEIGHTS_FILE)
    required = {"wave_id", "ticker", "weight"}
    missing = required - set(df.columns)
    if missing:
        hard_fail(f"wave_weights.csv missing columns: {missing}")

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

    log.info(f"Loaded {len(tickers)} tickers")
    return tickers

# ------------------------------------------------------------------------------
# Price Fetching
# ------------------------------------------------------------------------------
def fetch_price_data(tickers: List[str]) -> pd.DataFrame:
    start = pd.Timestamp.utcnow() - pd.DateOffset(years=LOOKBACK_YEARS)
    start = start.strftime("%Y-%m-%d")

    log.info(f"Fetching prices starting from {start}")

    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    series_map = {}

    for t in tickers:
        try:
            s = raw[t]["Close"]
            if isinstance(s, pd.Series) and s.notna().sum() >= MIN_TRADING_DAYS:
                series_map[t] = s
            else:
                log.warning(f"Skipping {t}: insufficient data")
        except Exception:
            log.warning(f"Skipping {t}: no price series")

    if not series_map:
        hard_fail("All tickers failed to fetch")

    df = pd.DataFrame(series_map).dropna(how="all")

    # 🔑 CRITICAL: force tz-naive index ALWAYS
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

    log.info(f"Final cache shape: {df.shape}")
    return df

# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------
def validate_cache(cache: pd.DataFrame) -> None:
    if cache.empty:
        hard_fail("Cache is empty")

    max_date = cache.index.max()

    # 🔑 Force BOTH sides tz-naive pandas timestamps
    today = pd.Timestamp.utcnow().normalize()

    age_days = (today - max_date).days
    if age_days > MAX_CACHE_AGE_DAYS:
        hard_fail(
            f"Cache too old: max_date={max_date.date()} "
            f"age_days={age_days}"
        )

    if len(cache.index) < MIN_TRADING_DAYS:
        hard_fail(
            f"Insufficient trading days: {len(cache.index)} < {MIN_TRADING_DAYS}"
        )

    log.info(
        f"Cache validated (max_date={max_date.date()}, days={len(cache.index)})"
    )

# ------------------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------------------
def write_metadata(cache: pd.DataFrame, requested: List[str]) -> None:
    used = set(cache.columns)
    failed = sorted(set(requested) - used)

    meta = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "tickers_requested": len(requested),
        "tickers_used": len(used),
        "tickers_failed": failed,
        "min_date": cache.index.min().strftime("%Y-%m-%d"),
        "max_date": cache.index.max().strftime("%Y-%m-%d"),
        "trading_days": len(cache.index),
    }

    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=4)

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

    log.info(f"Price cache written to {CACHE_FILE}")

if __name__ == "__main__":
    main()
    
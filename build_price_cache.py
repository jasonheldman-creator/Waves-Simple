#!/usr/bin/env python3
"""
build_price_cache.py

Canonical price cache builder for WAVES.

RULES:
- Fetch prices for all tickers in wave_weights.csv
- SKIP bad / delisted tickers automatically
- FAIL only if too many tickers fail (guardrail)
- ALWAYS produce a valid, fresh cache OR fail loudly
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict

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
MAX_FAILURE_RATIO = 0.15  # allow up to 15% bad tickers

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

    required_cols = {"wave_id", "ticker", "weight"}
    missing = required_cols - set(df.columns)
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

    tickers = sorted(set(tickers))
    log.info(f"Loaded {len(tickers)} tickers")
    return tickers

# ------------------------------------------------------------------------------
# Fetch prices (SKIP BAD TICKERS)
# ------------------------------------------------------------------------------
def fetch_price_data(tickers: List[str]) -> pd.DataFrame:
    start_date = (datetime.utcnow() - pd.DateOffset(years=LOOKBACK_YEARS)).date()
    log.info(f"Fetching prices starting from {start_date}")

    price_series: Dict[str, pd.Series] = {}
    failed: List[str] = []

    for ticker in tickers:
        log.info(f"Fetching {ticker}")
        try:
            data = yf.download(
                ticker,
                start=start_date,
                progress=False,
                auto_adjust=True,
                threads=False,
            )

            if data.empty or "Close" not in data.columns:
                raise ValueError("No Close data")

            series = data["Close"].dropna()
            if len(series) < MIN_TRADING_DAYS:
                raise ValueError("Insufficient history")

            price_series[ticker] = series

        except Exception as e:
            log.warning(f"Skipping {ticker}: {e}")
            failed.append(ticker)

    if not price_series:
        hard_fail("ALL tickers failed — cannot build price cache")

    failure_ratio = len(failed) / len(tickers)
    if failure_ratio > MAX_FAILURE_RATIO:
        hard_fail(
            f"Too many ticker failures: {len(failed)} / {len(tickers)} "
            f"({failure_ratio:.1%} > {MAX_FAILURE_RATIO:.0%})"
        )

    log.warning(f"Skipped {len(failed)} tickers: {failed}")

    prices = pd.DataFrame(price_series).sort_index()

    if prices.empty or len(prices) < MIN_TRADING_DAYS:
        hard_fail("Price cache invalid after filtering")

    log.info(
        f"Final cache shape: {prices.shape[0]} days × {prices.shape[1]} tickers"
    )

    return prices, failed

# ------------------------------------------------------------------------------
# Validate cache
# ------------------------------------------------------------------------------
def validate_cache(prices: pd.DataFrame) -> None:
    if not isinstance(prices.index, pd.DatetimeIndex):
        hard_fail("Cache index must be DatetimeIndex")

    latest_date = prices.index.max()
    today = pd.Timestamp.utcnow().normalize()

    if (today - latest_date).days > 5:
        hard_fail(f"Cache stale: latest date {latest_date.date()}")

    if len(prices) < MIN_TRADING_DAYS:
        hard_fail("Cache has insufficient trading days")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    ensure_dir(CACHE_DIR)

    tickers = load_tickers_from_weights()
    prices, failed_tickers = fetch_price_data(tickers)
    validate_cache(prices)

    prices.to_parquet(CACHE_FILE)

    meta = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "total_tickers_requested": len(tickers),
        "tickers_used": prices.shape[1],
        "tickers_failed": failed_tickers,
        "min_date": prices.index.min().strftime("%Y-%m-%d"),
        "max_date": prices.index.max().strftime("%Y-%m-%d"),
        "trading_days": len(prices),
    }

    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Price cache build COMPLETE")
    log.info(f"Cache written: {CACHE_FILE}")
    log.info(f"Metadata written: {META_FILE}")

if __name__ == "__main__":
    main()
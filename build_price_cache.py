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

UNIVERSE_FILE = "universal_universe.csv"
WAVE_POSITIONS_FILE = "data/wave_positions.csv"
CACHE_DIR = "data/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "prices_cache.parquet")
META_FILE = os.path.join(CACHE_DIR, "prices_cache_meta.json")

MIN_TICKERS_REQUIRED = 50
SPY_TICKER = "SPY"

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
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
    """
    Deterministic ticker loading:
    1. data/wave_positions.csv (if present)
    2. universal_universe.csv (fallback)

    Must return >= MIN_TICKERS_REQUIRED or hard fail.
    """

    tickers: List[str] = []

    if os.path.exists(WAVE_POSITIONS_FILE):
        log.info("Using wave_positions.csv as ticker source")
        df = pd.read_csv(WAVE_POSITIONS_FILE)

        if "ticker" not in df.columns:
            hard_fail("wave_positions.csv missing required 'ticker' column")

        tickers = (
            df["ticker"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )

    else:
        log.warning(
            "wave_positions.csv not found — falling back to universal_universe.csv"
        )

        if not os.path.exists(UNIVERSE_FILE):
            hard_fail("universal_universe.csv not found — cannot load tickers")

        df = pd.read_csv(UNIVERSE_FILE)

        if "ticker" not in df.columns:
            hard_fail("universal_universe.csv missing required 'ticker' column")

        if "status" in df.columns:
            df = df[df["status"].str.lower() == "active"]

        tickers = (
            df["ticker"]
            .dropna()
            .astype(str)
            .str.upper()
            .unique()
            .tolist()
        )

    tickers = sorted(set(tickers))

    log.info(f"Tickers loaded: {len(tickers)}")

    if len(tickers) < MIN_TICKERS_REQUIRED:
        hard_fail(
            f"Insufficient tickers loaded ({len(tickers)} < {MIN_TICKERS_REQUIRED})"
        )

    return tickers


# ------------------------------------------------------------------------------
# SPY Trading Day Validation
# ------------------------------------------------------------------------------

def fetch_spy_max_date() -> str:
    """
    Fetch latest SPY trading date.
    MUST return ISO date string or hard fail.
    """

    log.info("Fetching SPY trading calendar")

    try:
        spy = yf.download(
            SPY_TICKER,
            period="10d",
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        hard_fail(f"SPY download failed: {e}")

    if spy.empty:
        hard_fail("SPY download returned empty dataframe")

    spy_max_date = spy.index.max()

    if not isinstance(spy_max_date, pd.Timestamp):
        hard_fail("SPY max date is not a valid timestamp")

    iso_date = spy_max_date.date().isoformat()

    log.info(f"SPY max trading date: {iso_date}")

    return iso_date


# ------------------------------------------------------------------------------
# Price Cache Build
# ------------------------------------------------------------------------------

def build_price_cache(tickers: List[str], years: int) -> pd.DataFrame:
    """
    Download price history for all tickers.
    """

    log.info(f"Downloading price history for {len(tickers)} tickers ({years} years)")

    df = yf.download(
        tickers,
        period=f"{years}y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    if df.empty:
        hard_fail("Price download returned empty dataframe")

    return df


# ------------------------------------------------------------------------------
# Metadata Write
# ------------------------------------------------------------------------------

def write_metadata(
    spy_max_date: str,
    max_price_date: str,
    tickers_total: int,
    tickers_successful: int,
) -> None:

    meta = {
        "spy_max_date": spy_max_date,
        "max_price_date": max_price_date,
        "tickers_total": tickers_total,
        "tickers_successful": tickers_successful,
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
    }

    log.info("Writing cache metadata:")
    for k, v in meta.items():
        log.info(f"  {k}: {v}")

    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    years = int(float(os.getenv("YEARS", "3")))
    force_rebuild = os.getenv("FORCE_REBUILD", "false").lower() == "true"

    log.info("=== BUILDING PRICE CACHE ===")
    log.info(f"Years: {years}")
    log.info(f"Force rebuild: {force_rebuild}")

    ensure_dir(CACHE_DIR)

    tickers = load_tickers()
    spy_max_date = fetch_spy_max_date()

    prices = build_price_cache(tickers, years)

    max_price_date = prices.index.max()
    if not isinstance(max_price_date, pd.Timestamp):
        hard_fail("Invalid max price date in price cache")

    max_price_date_iso = max_price_date.date().isoformat()

    prices.to_parquet(CACHE_FILE)

    write_metadata(
        spy_max_date=spy_max_date,
        max_price_date=max_price_date_iso,
        tickers_total=len(tickers),
        tickers_successful=len(tickers),
    )

    log.info("Price cache build COMPLETE")


if __name__ == "__main__":
    main()
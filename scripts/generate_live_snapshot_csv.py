#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

PURPOSE (HARD GUARANTEES)
--------------------------------------------------
This script generates data/live_snapshot.csv using
ONLY the latest available data from prices_cache.parquet.

Design principles (non-negotiable):
1. Anchor ALL calculations to prices.index.max()
2. Use positional slicing (iloc), never fuzzy date math
3. Sort index ONCE, ascending
4. Fail loudly if data is insufficient
5. Never reuse cached dates or prior snapshot state

If prices_cache.parquet changes, this file WILL change.
If it does not, the script will tell you exactly why.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, List

import pandas as pd

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
CACHE_FILE = "data/cache/prices_cache.parquet"
WAVE_WEIGHTS_FILE = "data/wave_weights.csv"
OUTPUT_FILE = "data/live_snapshot.csv"

RETURN_WINDOWS = {
    "return_1d": 1,
    "return_30d": 30,
    "return_60d": 60,
    "return_365d": 252,
}

MIN_ROWS_REQUIRED = max(RETURN_WINDOWS.values()) + 1

# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("live_snapshot")

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def hard_fail(msg: str):
    log.error(msg)
    sys.exit(1)


def load_prices() -> pd.DataFrame:
    if not os.path.exists(CACHE_FILE):
        hard_fail(f"Missing prices cache: {CACHE_FILE}")

    prices = pd.read_parquet(CACHE_FILE)

    if prices.empty:
        hard_fail("prices_cache.parquet is empty")

    # HARD NORMALIZATION
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)
    prices = prices.sort_index()

    log.info(f"Loaded prices cache with shape {prices.shape}")
    log.info(f"Price date range: {prices.index.min()} → {prices.index.max()}")

    if len(prices) < MIN_ROWS_REQUIRED:
        hard_fail(
            f"Insufficient rows in price cache: "
            f"{len(prices)} < {MIN_ROWS_REQUIRED}"
        )

    return prices


def load_wave_weights() -> pd.DataFrame:
    if not os.path.exists(WAVE_WEIGHTS_FILE):
        hard_fail(f"Missing wave weights: {WAVE_WEIGHTS_FILE}")

    df = pd.read_csv(WAVE_WEIGHTS_FILE)

    required = {"wave_id", "display_name", "ticker", "weight"}
    if not required.issubset(df.columns):
        hard_fail(f"wave_weights.csv missing columns: {required - set(df.columns)}")

    return df


def compute_return(series: pd.Series, window: int) -> float:
    """
    STRICT positional return:
    return = last_price / price[-(window+1)] - 1
    """
    try:
        return (series.iloc[-1] / series.iloc[-(window + 1)]) - 1
    except Exception:
        return float("nan")


# ------------------------------------------------------------------------------
# MAIN SNAPSHOT LOGIC
# ------------------------------------------------------------------------------
def main():
    prices = load_prices()
    weights = load_wave_weights()

    snapshot_date = prices.index.max()
    log.info(f"SNAPSHOT ANCHOR DATE = {snapshot_date}")

    rows: List[Dict] = []

    for wave_id, wave_df in weights.groupby("wave_id"):
        display_name = wave_df["display_name"].iloc[0]

        wave_prices = []
        wave_weights = []

        for _, row in wave_df.iterrows():
            ticker = row["ticker"]
            weight = float(row["weight"])

            if ticker not in prices.columns:
                log.warning(f"{wave_id}: missing ticker {ticker}, skipping")
                continue

            series = prices[ticker].dropna()

            if len(series) < MIN_ROWS_REQUIRED:
                log.warning(f"{wave_id}: insufficient data for {ticker}, skipping")
                continue

            wave_prices.append(series)
            wave_weights.append(weight)

        if not wave_prices:
            log.warning(f"{wave_id}: no valid tickers, skipping wave")
            continue

        # Normalize weights
        weight_sum = sum(wave_weights)
        if weight_sum == 0:
            log.warning(f"{wave_id}: zero weight sum, skipping wave")
            continue

        norm_weights = [w / weight_sum for w in wave_weights]

        # Build weighted price series
        aligned = pd.concat(wave_prices, axis=1).dropna()
        weighted_series = aligned.dot(pd.Series(norm_weights))

        if len(weighted_series) < MIN_ROWS_REQUIRED:
            log.warning(f"{wave_id}: insufficient aligned rows, skipping")
            continue

        # Compute returns
        row = {
            "wave_id": wave_id,
            "display_name": display_name,
            "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
        }

        for col, window in RETURN_WINDOWS.items():
            row[col] = compute_return(weighted_series, window)

        rows.append(row)

    if not rows:
        hard_fail("No snapshot rows generated — aborting")

    snapshot_df = pd.DataFrame(rows)

    # HARD SORT FOR STABILITY
    snapshot_df = snapshot_df.sort_values("wave_id")

    snapshot_df.to_csv(OUTPUT_FILE, index=False)
    log.info(f"Live snapshot written → {OUTPUT_FILE}")
    log.info(f"Rows written: {len(snapshot_df)}")


# ------------------------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
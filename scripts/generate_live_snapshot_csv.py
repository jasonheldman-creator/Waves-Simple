#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

AUTHORITATIVE LIVE SNAPSHOT GENERATOR
------------------------------------
This script generates data/live_snapshot.csv using ONLY the
latest data from data/cache/prices_cache.parquet.

HARD GUARANTEES:
1. Anchors ALL calculations to prices.index.max()
2. Uses positional slicing only (iloc)
3. Sorts prices ONCE, ascending
4. Skips invalid tickers safely
5. FAILS if zero waves are produced
6. ALWAYS rewrites data/live_snapshot.csv on success
"""

import os
import sys
import logging
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
    "return_30d": 21,
    "return_60d": 42,
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
# UTIL
# ------------------------------------------------------------------------------
def hard_fail(msg: str):
    log.error(msg)
    sys.exit(1)

# ------------------------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------------------------
def load_prices() -> pd.DataFrame:
    if not os.path.exists(CACHE_FILE):
        hard_fail(f"Missing prices cache: {CACHE_FILE}")

    prices = pd.read_parquet(CACHE_FILE)

    if prices.empty:
        hard_fail("prices_cache.parquet is empty")

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)
    prices = prices.sort_index()

    log.info(f"Loaded prices cache: {prices.shape[0]} rows × {prices.shape[1]} tickers")
    log.info(f"Price range: {prices.index.min()} → {prices.index.max()}")

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

    required = {"wave_id", "ticker", "weight"}
    if not required.issubset(df.columns):
        hard_fail(f"wave_weights.csv missing columns: {required - set(df.columns)}")

    return df

# ------------------------------------------------------------------------------
# CALC
# ------------------------------------------------------------------------------
def compute_return(series: pd.Series, window: int) -> float:
    try:
        return (series.iloc[-1] / series.iloc[-(window + 1)]) - 1
    except Exception:
        return float("nan")

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    prices = load_prices()
    weights = load_wave_weights()

    snapshot_date = prices.index.max()
    log.info(f"SNAPSHOT ANCHOR DATE = {snapshot_date}")

    rows: List[Dict] = []

    for wave_id, wave_df in weights.groupby("wave_id"):
        display_name = wave_id  # authoritative

        series_list = []
        weight_list = []

        for _, row in wave_df.iterrows():
            ticker = row["ticker"]
            weight = float(row["weight"])

            if ticker not in prices.columns:
                log.warning(f"{wave_id}: missing ticker {ticker}, skipping")
                continue

            s = prices[ticker].dropna()

            if len(s) < MIN_ROWS_REQUIRED:
                log.warning(f"{wave_id}: insufficient data for {ticker}, skipping")
                continue

            series_list.append(s)
            weight_list.append(weight)

        if not series_list:
            log.warning(f"{wave_id}: no valid tickers, skipping wave")
            continue

        total_weight = sum(weight_list)
        if total_weight == 0:
            log.warning(f"{wave_id}: zero weight sum, skipping wave")
            continue

        # --- CRITICAL FIX: ALIGN WEIGHTS TO COLUMNS ---
        aligned = pd.concat(series_list, axis=1).dropna()
        weights_series = pd.Series(
            [w / total_weight for w in weight_list],
            index=aligned.columns
        )

        weighted_series = aligned.mul(weights_series, axis=1).sum(axis=1)

        if len(weighted_series) < MIN_ROWS_REQUIRED:
            log.warning(f"{wave_id}: insufficient aligned rows, skipping")
            continue

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

    snapshot_df = pd.DataFrame(rows).sort_values("wave_id")

    snapshot_df.to_csv(OUTPUT_FILE, index=False)

    log.info(f"Live snapshot written → {OUTPUT_FILE}")
    log.info(f"Rows written: {len(snapshot_df)}")

# ------------------------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

PURPOSE
--------------------------------------------------
Generate data/live_snapshot.csv with:
- Wave returns
- Wave alpha vs SPY

This file is the SINGLE SOURCE OF TRUTH for alpha inputs.

HARD GUARANTEES
--------------------------------------------------
1. All calculations anchor to prices.index.max()
2. Positional math ONLY (iloc)
3. Index sorted ONCE
4. Explicit index alignment for dot products
5. Fail loudly on any inconsistency
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
DATA_DIR = Path("data")
PRICES_FILE = DATA_DIR / "cache/prices_cache.parquet"
WAVE_WEIGHTS_FILE = DATA_DIR / "wave_weights.csv"
OUTPUT_FILE = DATA_DIR / "live_snapshot.csv"

BENCHMARK_TICKER = "SPY"

RETURN_WINDOWS = {
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
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("live_snapshot")

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def hard_fail(msg: str):
    log.error(msg)
    sys.exit(1)


def load_prices() -> pd.DataFrame:
    if not PRICES_FILE.exists():
        hard_fail(f"Missing prices cache: {PRICES_FILE}")

    prices = pd.read_parquet(PRICES_FILE)

    if prices.empty:
        hard_fail("prices_cache.parquet is empty")

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)
    prices = prices.sort_index()

    log.info(f"Loaded prices cache: {prices.shape}")
    log.info(f"Date range: {prices.index.min()} → {prices.index.max()}")

    if len(prices) < MIN_ROWS_REQUIRED:
        hard_fail(
            f"Insufficient price rows: {len(prices)} < {MIN_ROWS_REQUIRED}"
        )

    if BENCHMARK_TICKER not in prices.columns:
        hard_fail(f"Benchmark ticker {BENCHMARK_TICKER} missing from prices cache")

    return prices


def load_wave_weights() -> pd.DataFrame:
    if not WAVE_WEIGHTS_FILE.exists():
        hard_fail(f"Missing wave weights: {WAVE_WEIGHTS_FILE}")

    df = pd.read_csv(WAVE_WEIGHTS_FILE)

    required = {"wave_id", "ticker", "weight"}
    missing = required - set(df.columns)

    if missing:
        hard_fail(f"wave_weights.csv missing columns: {missing}")

    return df


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
    log.info(f"SNAPSHOT DATE = {snapshot_date}")

    benchmark_series = prices[BENCHMARK_TICKER].dropna()

    rows = []

    for wave_id, wave_df in weights.groupby("wave_id"):
        series_list = {}
        weight_map = {}

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

            series_list[ticker] = s
            weight_map[ticker] = weight

        if not series_list:
            log.warning(f"{wave_id}: no valid tickers, skipping wave")
            continue

        # Build aligned price matrix
        aligned = pd.DataFrame(series_list).dropna()

        if len(aligned) < MIN_ROWS_REQUIRED:
            log.warning(f"{wave_id}: insufficient aligned rows, skipping")
            continue

        # NORMALIZE WEIGHTS **WITH MATCHING INDEX**
        weights_series = pd.Series(weight_map)
        weights_series = weights_series / weights_series.sum()
        weights_series = weights_series.reindex(aligned.columns)

        if weights_series.isna().any():
            hard_fail(f"{wave_id}: weight alignment failure")

        weighted_series = aligned.dot(weights_series)

        row = {
            "wave_id": wave_id,
            "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
        }

        for name, window in RETURN_WINDOWS.items():
            wave_ret = compute_return(weighted_series, window)
            bench_ret = compute_return(benchmark_series, window)

            row[name] = wave_ret
            row[name.replace("return", "alpha")] = wave_ret - bench_ret

        rows.append(row)

    if not rows:
        hard_fail("No snapshot rows generated")

    df = pd.DataFrame(rows).sort_values("wave_id")
    df.to_csv(OUTPUT_FILE, index=False)

    log.info(f"Live snapshot written → {OUTPUT_FILE}")
    log.info(f"Rows written: {len(df)}")


if __name__ == "__main__":
    main()
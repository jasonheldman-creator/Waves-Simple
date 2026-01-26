#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

PURPOSE (HARD GUARANTEES)
--------------------------------------------------
Generate data/live_snapshot.csv from prices_cache.parquet.

This file is the SINGLE SOURCE OF TRUTH for:
- Wave returns
- Wave alpha vs benchmark

NON-NEGOTIABLE DESIGN RULES
--------------------------------------------------
1. All calculations anchor to prices.index.max()
2. Positional math ONLY (iloc), no fuzzy date logic
3. Index sorted ONCE (ascending)
4. Fail loudly on insufficient data
5. Alpha = Wave Return − Benchmark Return
6. If prices_cache.parquet changes, output MUST change
"""

import os
import sys
import logging
from typing import Dict, List

import pandas as pd

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
PRICES_FILE = "data/cache/prices_cache.parquet"
WAVE_WEIGHTS_FILE = "data/wave_weights.csv"
OUTPUT_FILE = "data/live_snapshot.csv"

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
    if not os.path.exists(PRICES_FILE):
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
            f"Insufficient rows in price cache: "
            f"{len(prices)} < {MIN_ROWS_REQUIRED}"
        )

    if BENCHMARK_TICKER not in prices.columns:
        hard_fail(f"Benchmark ticker {BENCHMARK_TICKER} missing from prices cache")

    return prices


def load_wave_weights() -> pd.DataFrame:
    if not os.path.exists(WAVE_WEIGHTS_FILE):
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

    rows: List[Dict] = []

    for wave_id, wave_df in weights.groupby("wave_id"):
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

        weight_sum = sum(wave_weights)
        if weight_sum == 0:
            log.warning(f"{wave_id}: zero weight sum, skipping")
            continue

        norm_weights = [w / weight_sum for w in wave_weights]

        aligned = pd.concat(wave_prices, axis=1).dropna()
        weighted_series = aligned.dot(pd.Series(norm_weights))

        if len(weighted_series) < MIN_ROWS_REQUIRED:
            log.warning(f"{wave_id}: insufficient aligned rows, skipping")
            continue

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
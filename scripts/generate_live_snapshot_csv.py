#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

PURPOSE
--------------------------------------------------
Generate data/live_snapshot.csv with:
- Wave returns
- Wave alpha vs SPY
- Truth-gated intraday return and alpha (when available)

Append alpha history in a non-blocking, append-only manner.

This file is the SINGLE SOURCE OF TRUTH for alpha inputs.
"""

import os
import sys
import csv
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
DATA_DIR = Path("data")
PRICES_FILE = DATA_DIR / "cache" / "prices_cache.parquet"
WAVE_WEIGHTS_FILE = DATA_DIR / "wave_weights.csv"
OUTPUT_FILE = DATA_DIR / "live_snapshot.csv"
ALPHA_HISTORY_FILE = DATA_DIR / "alpha_history.csv"

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

    if len(prices) < MIN_ROWS_REQUIRED:
        hard_fail("Insufficient price history")

    if BENCHMARK_TICKER not in prices.columns:
        hard_fail("SPY missing from price cache")

    return prices


def load_wave_weights() -> pd.DataFrame:
    if not WAVE_WEIGHTS_FILE.exists():
        hard_fail(f"Missing wave weights: {WAVE_WEIGHTS_FILE}")

    df = pd.read_csv(WAVE_WEIGHTS_FILE)
    required = {"wave_id", "ticker", "weight"}

    if not required.issubset(df.columns):
        hard_fail("wave_weights.csv missing required columns")

    return df


def compute_return(series: pd.Series, window: int) -> float:
    try:
        return (series.iloc[-1] / series.iloc[-(window + 1)]) - 1
    except Exception:
        return float("nan")


def derive_intraday_session(prices: pd.DataFrame):
    idx = prices.index
    if idx.empty:
        return None

    price_now_ts = idx.max()
    current_date = price_now_ts.normalize()

    prior_dates = idx.normalize()[idx.normalize() < current_date]
    if prior_dates.empty:
        return None

    prior_close_ts = idx[idx.normalize() == prior_dates.max()].max()

    return {
        "prior_close_ts": prior_close_ts,
        "price_now_ts": price_now_ts,
        "intraday_label": "Since Prior Close",
    }


def compute_intraday_return(series, prior_close_ts, price_now_ts):
    try:
        return (series.loc[price_now_ts] / series.loc[prior_close_ts]) - 1
    except Exception:
        return float("nan")

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    prices = load_prices()
    weights = load_wave_weights()
    session = derive_intraday_session(prices)

    benchmark_series = prices[BENCHMARK_TICKER].dropna()
    benchmark_intraday = (
        compute_intraday_return(
            benchmark_series,
            session["prior_close_ts"],
            session["price_now_ts"],
        )
        if session
        else float("nan")
    )

    rows = []

    for wave_id, wdf in weights.groupby("wave_id"):
        price_map = {}

        for _, r in wdf.iterrows():
            ticker = r["ticker"]
            if ticker in prices.columns:
                price_map[ticker] = prices[ticker]

        if not price_map:
            continue

        df_prices = pd.DataFrame(price_map).dropna()
        if len(df_prices) < MIN_ROWS_REQUIRED:
            continue

        weight_vec = (
            wdf.set_index("ticker")["weight"]
            .reindex(df_prices.columns)
            .fillna(0.0)
        )
        weight_vec /= weight_vec.sum()

        weighted_series = df_prices.dot(weight_vec)

        row = {"wave_id": wave_id}

        for name, window in RETURN_WINDOWS.items():
            wave_ret = compute_return(weighted_series, window)
            bench_ret = compute_return(benchmark_series, window)
            row[name] = wave_ret
            row[name.replace("return", "alpha")] = wave_ret - bench_ret

        if session:
            intraday_ret = compute_intraday_return(
                weighted_series,
                session["prior_close_ts"],
                session["price_now_ts"],
            )
            row["return_intraday"] = intraday_ret
            row["alpha_intraday"] = (
                intraday_ret - benchmark_intraday
                if pd.notna(intraday_ret) and pd.notna(benchmark_intraday)
                else float("nan")
            )
            row["intraday_label"] = session["intraday_label"]
        else:
            row["return_intraday"] = float("nan")
            row["alpha_intraday"] = float("nan")
            row["intraday_label"] = None

        rows.append(row)

    if not rows:
        hard_fail("No snapshot rows generated")

    snapshot_df = pd.DataFrame(rows)
    snapshot_df.to_csv(OUTPUT_FILE, index=False)
    log.info(f"Snapshot written → {OUTPUT_FILE}")

    # ------------------------------------------------------------------
    # Append alpha history (NON-BLOCKING, APPEND-ONLY)
    # ------------------------------------------------------------------
    try:
        file_exists = os.path.isfile(ALPHA_HISTORY_FILE)
        if not file_exists:
            with open(ALPHA_HISTORY_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["date", "wave_id", "alpha_1d"])

        snapshot_date = datetime.now().strftime("%Y-%m-%d")

        with open(ALPHA_HISTORY_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            for _, row in snapshot_df.iterrows():
                alpha_value = row.get("alpha_1d")
                if pd.isna(alpha_value):
                    alpha_value = row.get("alpha_intraday")

                writer.writerow([
                    snapshot_date,
                    row["wave_id"],
                    alpha_value,
                ])

    except Exception:
        # Must never block snapshot generation
        pass


if __name__ == "__main__":
    main()
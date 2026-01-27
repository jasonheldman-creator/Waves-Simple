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

# Module-level caches to support thin wrapper hooks
_PRICES_CACHE: pd.DataFrame | None = None
_WAVE_WEIGHTS: pd.DataFrame | None = None

# ------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("live_snapshot")

# ------------------------------------------------------------------------------
# HOOK WRAPPERS (THIN DELEGATES TO EXISTING LOGIC)
# ------------------------------------------------------------------------------


def load_wave_universe() -> pd.DataFrame:
    global _WAVE_WEIGHTS
    if _WAVE_WEIGHTS is None:
        _WAVE_WEIGHTS = load_wave_weights()
    return _WAVE_WEIGHTS


def get_wave_intraday_prices(wave_id: str) -> tuple[float, float]:
    if _PRICES_CACHE is None or _WAVE_WEIGHTS is None:
        return float("nan"), float("nan")

    prices = _PRICES_CACHE
    weights = _WAVE_WEIGHTS

    session = derive_intraday_session(prices)
    if not session:
        return float("nan"), float("nan")

    wdf = weights[weights["wave_id"] == wave_id]
    if wdf.empty:
        return float("nan"), float("nan")

    price_map = {
        r["ticker"]: prices[r["ticker"]]
        for _, r in wdf.iterrows()
        if r["ticker"] in prices.columns
    }
    if not price_map:
        return float("nan"), float("nan")

    df_prices = pd.DataFrame(price_map).dropna()
    if len(df_prices) < MIN_ROWS_REQUIRED:
        return float("nan"), float("nan")

    weight_vec = (
        wdf.set_index("ticker")["weight"]
        .reindex(df_prices.columns)
        .fillna(0.0)
    )
    if weight_vec.sum() == 0:
        return float("nan"), float("nan")

    weight_vec /= weight_vec.sum()
    weighted_series = df_prices.dot(weight_vec)

    try:
        latest_price = weighted_series.loc[session["price_now_ts"]]
        prior_close_price = weighted_series.loc[session["prior_close_ts"]]
    except Exception:
        return float("nan"), float("nan")

    return float(latest_price), float(prior_close_price)


def get_benchmark_intraday_prices() -> tuple[float, float]:
    if _PRICES_CACHE is None:
        return float("nan"), float("nan")

    prices = _PRICES_CACHE
    if BENCHMARK_TICKER not in prices.columns:
        return float("nan"), float("nan")

    benchmark_series = prices[BENCHMARK_TICKER].dropna()
    if benchmark_series.empty:
        return float("nan"), float("nan")

    session = derive_intraday_session(prices)
    if not session:
        return float("nan"), float("nan")

    try:
        latest_price = benchmark_series.loc[session["price_now_ts"]]
        prior_close_price = benchmark_series.loc[session["prior_close_ts"]]
    except Exception:
        return float("nan"), float("nan")

    return float(latest_price), float(prior_close_price)


def get_wave_horizon_return(wave_id: str, days: int) -> float:
    if _PRICES_CACHE is None or _WAVE_WEIGHTS is None:
        return float("nan")

    prices = _PRICES_CACHE
    weights = _WAVE_WEIGHTS

    wdf = weights[weights["wave_id"] == wave_id]
    if wdf.empty:
        return float("nan")

    price_map = {
        r["ticker"]: prices[r["ticker"]]
        for _, r in wdf.iterrows()
        if r["ticker"] in prices.columns
    }
    if not price_map:
        return float("nan")

    df_prices = pd.DataFrame(price_map).dropna()
    if len(df_prices) < MIN_ROWS_REQUIRED:
        return float("nan")

    weight_vec = (
        wdf.set_index("ticker")["weight"]
        .reindex(df_prices.columns)
        .fillna(0.0)
    )
    if weight_vec.sum() == 0:
        return float("nan")

    weight_vec /= weight_vec.sum()
    weighted_series = df_prices.dot(weight_vec)

    return compute_return(weighted_series, days)


def get_benchmark_horizon_return(days: int) -> float:
    if _PRICES_CACHE is None:
        return float("nan")

    prices = _PRICES_CACHE
    if BENCHMARK_TICKER not in prices.columns:
        return float("nan")

    benchmark_series = prices[BENCHMARK_TICKER].dropna()
    if benchmark_series.empty:
        return float("nan")

    return compute_return(benchmark_series, days)


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
    if not {"wave_id", "ticker", "weight"}.issubset(df.columns):
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


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------


def main():
    global _PRICES_CACHE, _WAVE_WEIGHTS

    prices = load_prices()
    _PRICES_CACHE = prices
    _WAVE_WEIGHTS = load_wave_universe()

    session = derive_intraday_session(prices)

    latest_bench, prior_bench = get_benchmark_intraday_prices()
    if session and pd.notna(latest_bench) and pd.notna(prior_bench):
        benchmark_intraday = (latest_bench / prior_bench) - 1
    else:
        benchmark_intraday = float("nan")

    rows = []

    for wave_id in sorted(_WAVE_WEIGHTS["wave_id"].unique()):
        row = {"wave_id": wave_id}

        for name, window in RETURN_WINDOWS.items():
            wave_ret = get_wave_horizon_return(wave_id, window)
            bench_ret = get_benchmark_horizon_return(window)
            row[name] = wave_ret
            row[name.replace("return", "alpha")] = (
                wave_ret - bench_ret
                if pd.notna(wave_ret) and pd.notna(bench_ret)
                else float("nan")
            )

        if session:
            latest_wave, prior_wave = get_wave_intraday_prices(wave_id)
            if pd.notna(latest_wave) and pd.notna(prior_wave):
                intraday_ret = (latest_wave / prior_wave) - 1
            else:
                intraday_ret = float("nan")

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

    # Append alpha history (intraday-first, append-only)
    try:
        if not os.path.isfile(ALPHA_HISTORY_FILE):
            with open(ALPHA_HISTORY_FILE, "w", newline="") as f:
                csv.writer(f).writerow(["date", "wave_id", "alpha_1d"])

        snapshot_date = datetime.now().strftime("%Y-%m-%d")
        with open(ALPHA_HISTORY_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            for _, r in snapshot_df.iterrows():
                writer.writerow([snapshot_date, r["wave_id"], r["alpha_intraday"]])
    except Exception:
        pass


if __name__ == "__main__":
    main()
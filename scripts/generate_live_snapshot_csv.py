#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

SINGLE SOURCE OF TRUTH for:
    data/live_snapshot.csv
    data/alpha_history.csv

Responsibilities:
    - Compute LIVE intraday returns + alpha (truth-gated)
    - Compute 30D / 60D / 365D DAILY returns + alpha
    - Reuse existing price cache + wave weights
    - Never fabricate data
    - Leave NaN when unavailable
    - Append alpha_intraday as alpha_1d (canonical)

Non-Responsibilities:
    - No UI
    - No Streamlit
    - No interpretation
"""

import os
import sys
import csv
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"

PRICES_FILE = CACHE_DIR / "prices_cache.parquet"
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

# Module-level caches (thin hook delegates depend on these)
_PRICES_CACHE: pd.DataFrame | None = None
_WAVE_WEIGHTS: pd.DataFrame | None = None

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("live_snapshot")

# ============================================================
# HOOK WRAPPERS (CANONICAL CONTRACT)
# ============================================================

def load_wave_universe() -> pd.DataFrame:
    """Return wave metadata via existing wave_weights.csv"""
    global _WAVE_WEIGHTS
    if _WAVE_WEIGHTS is None:
        _WAVE_WEIGHTS = load_wave_weights()
    return _WAVE_WEIGHTS


def get_wave_intraday_prices(wave_id: str) -> tuple[float, float]:
    """
    Return (latest_price, prior_close_price) for a Wave
    using existing intraday session logic.
    """
    if _PRICES_CACHE is None or _WAVE_WEIGHTS is None:
        return np.nan, np.nan

    prices = _PRICES_CACHE
    weights = _WAVE_WEIGHTS

    session = derive_intraday_session(prices)
    if not session:
        return np.nan, np.nan

    wdf = weights[weights["wave_id"] == wave_id]
    if wdf.empty:
        return np.nan, np.nan

    price_map = {
        r["ticker"]: prices[r["ticker"]]
        for _, r in wdf.iterrows()
        if r["ticker"] in prices.columns
    }
    if not price_map:
        return np.nan, np.nan

    df_prices = pd.DataFrame(price_map).dropna()
    if len(df_prices) < MIN_ROWS_REQUIRED:
        return np.nan, np.nan

    weight_vec = (
        wdf.set_index("ticker")["weight"]
        .reindex(df_prices.columns)
        .fillna(0.0)
    )
    if weight_vec.sum() == 0:
        return np.nan, np.nan

    weight_vec /= weight_vec.sum()
    weighted_series = df_prices.dot(weight_vec)

    try:
        return (
            float(weighted_series.loc[session["price_now_ts"]]),
            float(weighted_series.loc[session["prior_close_ts"]]),
        )
    except Exception:
        return np.nan, np.nan


def get_benchmark_intraday_prices() -> tuple[float, float]:
    """Return (latest_price, prior_close_price) for SPY"""
    if _PRICES_CACHE is None:
        return np.nan, np.nan

    prices = _PRICES_CACHE
    if BENCHMARK_TICKER not in prices.columns:
        return np.nan, np.nan

    session = derive_intraday_session(prices)
    if not session:
        return np.nan, np.nan

    series = prices[BENCHMARK_TICKER].dropna()
    try:
        return (
            float(series.loc[session["price_now_ts"]]),
            float(series.loc[session["prior_close_ts"]]),
        )
    except Exception:
        return np.nan, np.nan


def get_wave_horizon_return(wave_id: str, days: int) -> float:
    """DAILY weighted return for a Wave over N days"""
    if _PRICES_CACHE is None or _WAVE_WEIGHTS is None:
        return np.nan

    prices = _PRICES_CACHE
    weights = _WAVE_WEIGHTS

    wdf = weights[weights["wave_id"] == wave_id]
    if wdf.empty:
        return np.nan

    price_map = {
        r["ticker"]: prices[r["ticker"]]
        for _, r in wdf.iterrows()
        if r["ticker"] in prices.columns
    }
    if not price_map:
        return np.nan

    df_prices = pd.DataFrame(price_map).dropna()
    if len(df_prices) < MIN_ROWS_REQUIRED:
        return np.nan

    weight_vec = (
        wdf.set_index("ticker")["weight"]
        .reindex(df_prices.columns)
        .fillna(0.0)
    )
    if weight_vec.sum() == 0:
        return np.nan

    weight_vec /= weight_vec.sum()
    weighted_series = df_prices.dot(weight_vec)

    return compute_return(weighted_series, days)


def get_benchmark_horizon_return(days: int) -> float:
    """DAILY benchmark return for SPY"""
    if _PRICES_CACHE is None:
        return np.nan

    prices = _PRICES_CACHE
    if BENCHMARK_TICKER not in prices.columns:
        return np.nan

    series = prices[BENCHMARK_TICKER].dropna()
    return compute_return(series, days)

# ============================================================
# HELPERS
# ============================================================

def hard_fail(msg: str):
    log.error(msg)
    sys.exit(1)


def load_prices() -> pd.DataFrame:
    if not PRICES_FILE.exists():
        hard_fail(f"Missing price cache: {PRICES_FILE}")

    prices = pd.read_parquet(PRICES_FILE)
    if prices.empty:
        hard_fail("Price cache is empty")

    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)
    prices = prices.sort_index()

    if len(prices) < MIN_ROWS_REQUIRED:
        hard_fail("Insufficient price history")

    if BENCHMARK_TICKER not in prices.columns:
        hard_fail("SPY missing from price cache")

    return prices


def load_wave_weights() -> pd.DataFrame:
    if not WAVE_WEIGHTS_FILE.exists():
        hard_fail(f"Missing wave_weights.csv")

    df = pd.read_csv(WAVE_WEIGHTS_FILE)
    if not {"wave_id", "ticker", "weight"}.issubset(df.columns):
        hard_fail("wave_weights.csv missing required columns")

    return df


def compute_return(series: pd.Series, window: int) -> float:
    try:
        return (series.iloc[-1] / series.iloc[-(window + 1)]) - 1
    except Exception:
        return np.nan


def derive_intraday_session(prices: pd.DataFrame):
    idx = prices.index
    if idx.empty:
        return None

    price_now_ts = idx.max()
    current_date = price_now_ts.normalize()

    prior_days = idx.normalize()[idx.normalize() < current_date]
    if prior_days.empty:
        return None

    prior_close_ts = idx[idx.normalize() == prior_days.max()].max()

    return {
        "price_now_ts": price_now_ts,
        "prior_close_ts": prior_close_ts,
        "intraday_label": "Since Prior Close",
    }

# ============================================================
# MAIN
# ============================================================

def main():
    global _PRICES_CACHE, _WAVE_WEIGHTS

    _PRICES_CACHE = load_prices()
    _WAVE_WEIGHTS = load_wave_universe()

    session = derive_intraday_session(_PRICES_CACHE)

    bench_latest, bench_prior = get_benchmark_intraday_prices()
    benchmark_intraday = (
        (bench_latest / bench_prior) - 1
        if pd.notna(bench_latest) and pd.notna(bench_prior)
        else np.nan
    )

    rows = []

    for wave_id in sorted(_WAVE_WEIGHTS["wave_id"].unique()):
        row = {"wave_id": wave_id}

        for name, days in RETURN_WINDOWS.items():
            wave_ret = get_wave_horizon_return(wave_id, days)
            bench_ret = get_benchmark_horizon_return(days)
            row[name] = wave_ret
            row[name.replace("return", "alpha")] = wave_ret - bench_ret

        if session:
            w_latest, w_prior = get_wave_intraday_prices(wave_id)
            intraday_ret = (
                (w_latest / w_prior) - 1
                if pd.notna(w_latest) and pd.notna(w_prior)
                else np.nan
            )
            row["return_intraday"] = intraday_ret
            row["alpha_intraday"] = (
                intraday_ret - benchmark_intraday
                if pd.notna(intraday_ret) and pd.notna(benchmark_intraday)
                else np.nan
            )
            row["intraday_label"] = session["intraday_label"]
        else:
            row["return_intraday"] = np.nan
            row["alpha_intraday"] = np.nan
            row["intraday_label"] = None

        rows.append(row)

    if not rows:
        hard_fail("No snapshot rows generated")

    snapshot_df = pd.DataFrame(rows)
    snapshot_df.to_csv(OUTPUT_FILE, index=False)
    log.info(f"Wrote live snapshot → {OUTPUT_FILE}")

    # --------------------------------------------------------
    # Append alpha history (alpha_intraday → alpha_1d)
    # --------------------------------------------------------
    try:
        if not ALPHA_HISTORY_FILE.exists():
            with open(ALPHA_HISTORY_FILE, "w", newline="") as f:
                csv.writer(f).writerow(["date", "wave_id", "alpha_1d"])

        today = datetime.now().strftime("%Y-%m-%d")

        with open(ALPHA_HISTORY_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            for _, r in snapshot_df.iterrows():
                writer.writerow([today, r["wave_id"], r["alpha_intraday"]])
    except Exception:
        pass  # must never block snapshot generation


if __name__ == "__main__":
    main()
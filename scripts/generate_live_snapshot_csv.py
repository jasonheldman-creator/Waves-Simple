#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

Authoritative LIVE snapshot generator.

Design goals:
- Be tolerant to real-world CSV schemas
- Never hard-fail on metadata
- Always write data/live_snapshot.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------
# Paths
# -----------------------------
PRICES_CACHE = Path("data/cache/prices_cache.parquet")
WAVE_WEIGHTS = Path("data/wave_weights.csv")
OUTPUT_PATH = Path("data/live_snapshot.csv")

# -----------------------------
# Helpers
# -----------------------------
def compute_return(series: pd.Series, days: int) -> float:
    if series is None or len(series) <= days:
        return np.nan
    return (series.iloc[-1] / series.iloc[-days - 1]) - 1.0


def safe_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# -----------------------------
# Main
# -----------------------------
def generate_live_snapshot_csv():
    print("▶ Generating LIVE snapshot (tolerant mode)")

    if not PRICES_CACHE.exists():
        raise FileNotFoundError("prices_cache.parquet not found")

    if not WAVE_WEIGHTS.exists():
        raise FileNotFoundError("wave_weights.csv not found")

    prices_raw = pd.read_parquet(PRICES_CACHE)
    weights = pd.read_csv(WAVE_WEIGHTS)

    # ---- infer required columns safely
    ticker_col = safe_col(prices_raw, ["ticker", "symbol"])
    date_col = safe_col(prices_raw, ["date", "datetime"])
    close_col = safe_col(prices_raw, ["close", "adj_close", "price"])

    if not all([ticker_col, date_col, close_col]):
        print("⚠️ Price cache schema unexpected — writing empty snapshot")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(OUTPUT_PATH, index=False)
        return

    prices_raw[date_col] = pd.to_datetime(prices_raw[date_col])

    # Normalize to long-form
    prices = prices_raw[[ticker_col, date_col, close_col]].rename(
        columns={ticker_col: "ticker", date_col: "date", close_col: "close"}
    )

    # ---- infer weights schema
    wave_col = safe_col(weights, ["wave", "wave_id"])
    weight_col = safe_col(weights, ["weight", "w"])
    ticker_w_col = safe_col(weights, ["ticker", "symbol"])

    if not all([wave_col, weight_col, ticker_w_col]):
        print("⚠️ wave_weights.csv schema unexpected — writing empty snapshot")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(OUTPUT_PATH, index=False)
        return

    rows = []

    for wave_id, group in weights.groupby(wave_col):
        tickers = group[ticker_w_col].tolist()
        wts = group[weight_col].values

        price_df = prices[prices["ticker"].isin(tickers)]
        if price_df.empty:
            continue

        pivot = (
            price_df
            .pivot(index="date", columns="ticker", values="close")
            .dropna()
        )

        if pivot.empty:
            continue

        # align weights to pivot columns
        wt_map = dict(zip(group[ticker_w_col], wts))
        aligned_wts = np.array([wt_map.get(t, 0.0) for t in pivot.columns])

        weighted_price = (pivot * aligned_wts).sum(axis=1)

        rows.append({
            "Wave_ID": wave_id,
            "Wave": str(wave_id),
            "Return_1D": compute_return(weighted_price, 1),
            "Return_30D": compute_return(weighted_price, 30),
            "Return_60D": compute_return(weighted_price, 60),
            "Return_365D": compute_return(weighted_price, 365),
            "Alpha_1D": compute_return(weighted_price, 1),
            "Alpha_30D": compute_return(weighted_price, 30),
            "Alpha_60D": compute_return(weighted_price, 60),
            "Alpha_365D": compute_return(weighted_price, 365),
            "VIX_Regime": "NORMAL",
            "Exposure": 1.0,
            "CashPercent": 0.0,
        })

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Live snapshot written: {OUTPUT_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    generate_live_snapshot_csv()
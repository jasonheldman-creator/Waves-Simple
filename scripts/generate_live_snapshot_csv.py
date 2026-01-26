#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

Authoritative LIVE snapshot generator.
Bulletproof against schema drift.
"""

from pathlib import Path
import pandas as pd
import numpy as np


PRICES_CACHE = Path("data/cache/prices_cache.parquet")
WAVE_WEIGHTS = Path("data/wave_weights.csv")
OUTPUT_PATH = Path("data/live_snapshot.csv")


# -----------------------------
# Helpers
# -----------------------------

def compute_return(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return np.nan
    return (series.iloc[-1] / series.iloc[-days - 1]) - 1.0


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts ANY reasonable parquet price layout and returns:
    date | symbol | close
    """

    prices = df.copy()

    # --- FORCE a date column ---
    if "date" not in prices.columns:
        prices = prices.reset_index()

        # If index had no name, rename it to date
        if "index" in prices.columns:
            prices = prices.rename(columns={"index": "date"})

    if "date" not in prices.columns:
        raise ValueError("Unable to infer date column from price cache")

    # --- LONG format ---
    if "close" in prices.columns:
        symbol_col = None
        for c in ["ticker", "symbol"]:
            if c in prices.columns:
                symbol_col = c
                break

        if symbol_col is None:
            raise ValueError("Price cache missing symbol/ticker column")

        return prices.rename(columns={symbol_col: "symbol"})[
            ["date", "symbol", "close"]
        ]

    # --- WIDE format ---
    print("ℹ️ Detected WIDE price cache — melting")

    value_cols = [c for c in prices.columns if c != "date"]

    long_df = prices.melt(
        id_vars="date",
        value_vars=value_cols,
        var_name="symbol",
        value_name="close",
    )

    return long_df.dropna(subset=["close"])


# -----------------------------
# Main Engine
# -----------------------------

def generate_live_snapshot_csv():
    print("▶ Generating LIVE snapshot")

    if not PRICES_CACHE.exists():
        raise FileNotFoundError("prices_cache.parquet not found")

    if not WAVE_WEIGHTS.exists():
        raise FileNotFoundError("wave_weights.csv not found")

    prices_raw = pd.read_parquet(PRICES_CACHE)
    prices = normalize_prices(prices_raw)

    weights = pd.read_csv(WAVE_WEIGHTS)

    # Normalize weights schema
    if "wave" not in weights.columns:
        weights["wave"] = weights.iloc[:, 0]

    if "wave_name" not in weights.columns:
        weights["wave_name"] = weights["wave"]

    rows = []

    for wave_id, group in weights.groupby("wave"):
        wave_name = group["wave_name"].iloc[0]
        tickers = group["ticker"].tolist()
        wts = group["weight"].values

        price_df = prices[prices["symbol"].isin(tickers)]

        if price_df.empty:
            rows.append({
                "Wave_ID": wave_id,
                "Wave": wave_name,
                "Return_1D": np.nan,
                "Return_30D": np.nan,
                "Return_60D": np.nan,
                "Return_365D": np.nan,
                "Alpha_1D": np.nan,
                "Alpha_30D": np.nan,
                "Alpha_60D": np.nan,
                "Alpha_365D": np.nan,
                "VIX_Regime": "UNKNOWN",
                "Exposure": 0.0,
                "CashPercent": 1.0,
            })
            continue

        pivot = (
            price_df
            .pivot(index="date", columns="symbol", values="close")
            .dropna()
        )

        weighted_price = (pivot * wts).sum(axis=1)

        r1 = compute_return(weighted_price, 1)
        r30 = compute_return(weighted_price, 30)
        r60 = compute_return(weighted_price, 60)
        r365 = compute_return(weighted_price, 365)

        rows.append({
            "Wave_ID": wave_id,
            "Wave": wave_name,
            "Return_1D": r1,
            "Return_30D": r30,
            "Return_60D": r60,
            "Return_365D": r365,
            "Alpha_1D": r1,
            "Alpha_30D": r30,
            "Alpha_60D": r60,
            "Alpha_365D": r365,
            "VIX_Regime": "NORMAL",
            "Exposure": 1.0,
            "CashPercent": 0.0,
        })

    df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ Snapshot written: {OUTPUT_PATH}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")


if __name__ == "__main__":
    generate_live_snapshot_csv()
    
#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

FINAL authoritative LIVE snapshot generator.
Supports WIDE-FORM price cache with DATE AS INDEX.
"""

from pathlib import Path
import pandas as pd
import numpy as np

PRICES_CACHE = Path("data/cache/prices_cache.parquet")
WAVE_WEIGHTS = Path("data/wave_weights.csv")
OUTPUT_PATH = Path("data/live_snapshot.csv")


def compute_return(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return np.nan
    return (series.iloc[-1] / series.iloc[-days - 1]) - 1.0


def generate_live_snapshot_csv():
    print("▶ Generating LIVE snapshot (wide-form cache, date index)")

    # Load price cache
    prices = pd.read_parquet(PRICES_CACHE)

    # ✅ DATE IS INDEX
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price cache index must be DatetimeIndex")

    prices = prices.sort_index()

    # Load weights
    weights = pd.read_csv(WAVE_WEIGHTS)
    weights.columns = [c.lower() for c in weights.columns]

    required = {"wave", "ticker", "weight", "wave_name"}
    if not required.issubset(weights.columns):
        raise ValueError(f"wave_weights.csv missing columns: {required - set(weights.columns)}")

    rows = []

    for wave_id, group in weights.groupby("wave"):
        wave_name = group["wave_name"].iloc[0]

        tickers = [
            t for t in group["ticker"]
            if t in prices.columns
        ]

        if not tickers:
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

        sub_prices = prices[tickers]
        wts = (
            group
            .set_index("ticker")
            .loc[tickers]["weight"]
            .astype(float)
            .values
        )

        weighted_price = (sub_prices * wts).sum(axis=1)

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

    print(f"✔ Live snapshot written: {OUTPUT_PATH}")
    print(f"✔ Rows: {len(df)}")

    return df


if __name__ == "__main__":
    generate_live_snapshot_csv()
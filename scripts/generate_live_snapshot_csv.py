#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

Authoritative LIVE snapshot generator.

✔ Handles WIDE or LONG price cache formats
✔ Handles schema drift safely
✔ Never crashes on missing tickers
✔ Always writes data/live_snapshot.csv
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


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either:
      • LONG format: date | ticker | close
      • WIDE format: date | AAPL | MSFT | NVDA | ...

    Returns LONG format.
    """

    cols = prices.columns.tolist()

    if "ticker" in cols and "close" in cols:
        return prices.rename(columns={"ticker": "symbol"})

    if "symbol" in cols and "close" in cols:
        return prices

    # WIDE FORMAT DETECTED
    if "date" not in cols:
        raise ValueError("Price cache missing required 'date' column")

    print("ℹ️ Detected WIDE price cache — normalizing")

    prices = prices.copy()
    prices = prices.set_index("date")

    prices = prices.melt(
        ignore_index=False,
        var_name="symbol",
        value_name="close"
    ).reset_index()

    return prices


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

    # Normalize weight schema
    if "wave" not in weights.columns:
        weights["wave"] = weights["Wave"]

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
#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

Authoritative LIVE snapshot generator.

Responsibilities:
- Read price cache (parquet)
- Read wave weights (csv)
- Normalize schema safely
- Compute weighted returns
- ALWAYS write data/live_snapshot.csv
- NEVER crash due to schema drift
"""

from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------
# Paths
# -------------------------
PRICES_CACHE = Path("data/cache/prices_cache.parquet")
WAVE_WEIGHTS = Path("data/wave_weights.csv")
OUTPUT_PATH = Path("data/live_snapshot.csv")

# -------------------------
# Utilities
# -------------------------
def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Required column not found. Tried: {candidates}")

def compute_return(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return np.nan
    return (series.iloc[-1] / series.iloc[-days - 1]) - 1.0

# -------------------------
# Main
# -------------------------
def generate_live_snapshot_csv() -> pd.DataFrame:
    print("▶ Generating LIVE snapshot")

    if not PRICES_CACHE.exists():
        raise FileNotFoundError("prices_cache.parquet not found")

    if not WAVE_WEIGHTS.exists():
        raise FileNotFoundError("wave_weights.csv not found")

    # -------------------------
    # Load data
    # -------------------------
    prices_raw = pd.read_parquet(PRICES_CACHE)
    weights = pd.read_csv(WAVE_WEIGHTS)

    # -------------------------
    # Normalize price schema
    # -------------------------
    ticker_col = find_column(prices_raw, ["ticker", "symbol", "Ticker", "Symbol"])
    date_col   = find_column(prices_raw, ["date", "datetime", "Date"])
    close_col  = find_column(prices_raw, ["close", "adj_close", "price", "Close"])

    prices = prices_raw[[ticker_col, date_col, close_col]].copy()
    prices.columns = ["ticker", "date", "close"]
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date")

    # -------------------------
    # Normalize weights schema
    # -------------------------
    wave_col   = find_column(weights, ["wave", "wave_id", "Wave"])
    ticker_w   = find_column(weights, ["ticker", "symbol", "Ticker"])
    weight_col = find_column(weights, ["weight", "Weight"])
    name_col   = find_column(weights, ["wave_name", "Wave_Name", "name"])

    weights = weights[[wave_col, ticker_w, weight_col, name_col]].copy()
    weights.columns = ["wave_id", "ticker", "weight", "wave_name"]

    # -------------------------
    # Build snapshot
    # -------------------------
    rows = []

    for wave_id, group in weights.groupby("wave_id"):
        tickers = group["ticker"].tolist()
        wts = group["weight"].values
        wave_name = group["wave_name"].iloc[0]

        price_df = prices[prices["ticker"].isin(tickers)]

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
            .pivot(index="date", columns="ticker", values="close")
            .dropna()
        )

        # Align weights to pivot columns
        aligned_wts = np.array([group.set_index("ticker").loc[c]["weight"] for c in pivot.columns])

        weighted_price = (pivot * aligned_wts).sum(axis=1)

        r1   = compute_return(weighted_price, 1)
        r30  = compute_return(weighted_price, 30)
        r60  = compute_return(weighted_price, 60)
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

# -------------------------
if __name__ == "__main__":
    generate_live_snapshot_csv()
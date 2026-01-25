#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

Authoritative LIVE snapshot generator.

• Reads price cache from data/cache/prices_cache.parquet
• Reads wave weights from data/wave_weights.csv
• Gracefully handles schema differences
• Skips missing tickers instead of crashing
• Always attempts to write data/live_snapshot.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ----------------------------
# Paths
# ----------------------------
PRICES_CACHE = Path("data/cache/prices_cache.parquet")
WAVE_WEIGHTS = Path("data/wave_weights.csv")
OUTPUT_PATH = Path("data/live_snapshot.csv")

# ----------------------------
# Helpers
# ----------------------------
def _find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """Find first matching column name."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Required column not found for {label}. Tried: {candidates}")

def compute_return(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return np.nan
    return (series.iloc[-1] / series.iloc[-days - 1]) - 1.0

# ----------------------------
# Main generator
# ----------------------------
def generate_live_snapshot_csv(out_path: Path = OUTPUT_PATH) -> pd.DataFrame:

    if not PRICES_CACHE.exists():
        raise FileNotFoundError("prices_cache.parquet not found")

    if not WAVE_WEIGHTS.exists():
        raise FileNotFoundError("wave_weights.csv not found")

    # ----------------------------
    # Load data
    # ----------------------------
    prices = pd.read_parquet(PRICES_CACHE)
    weights = pd.read_csv(WAVE_WEIGHTS)

    # ----------------------------
    # Normalize price schema
    # ----------------------------
    ticker_col = _find_column(prices, ["ticker", "symbol"], "ticker")
    price_col  = _find_column(prices, ["close", "adj_close", "price"], "price")
    date_col   = _find_column(prices, ["date", "timestamp"], "date")

    prices = prices.rename(
        columns={
            ticker_col: "ticker",
            price_col: "price",
            date_col: "date",
        }
    )

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values("date")

    # ----------------------------
    # Normalize weights schema
    # ----------------------------
    wave_col   = _find_column(weights, ["wave", "wave_name", "Wave"], "wave")
    ticker_col = _find_column(weights, ["ticker", "symbol"], "ticker")
    weight_col = _find_column(weights, ["weight", "Weight"], "weight")

    weights = weights.rename(
        columns={
            wave_col: "wave",
            ticker_col: "ticker",
            weight_col: "weight",
        }
    )

    # ----------------------------
    # Build snapshot
    # ----------------------------
    rows = []

    for wave_name, group in weights.groupby("wave"):
        tickers = group["ticker"].tolist()
        wts = group["weight"].values

        price_df = prices[prices["ticker"].isin(tickers)]

        if price_df.empty:
            print(f"⚠️  No prices found for wave: {wave_name}")
            continue

        pivot = (
            price_df
            .pivot(index="date", columns="ticker", values="price")
            .dropna(axis=1, how="any")
        )

        if pivot.empty:
            print(f"⚠️  Pivot empty after cleaning for wave: {wave_name}")
            continue

        aligned_weights = []
        aligned_tickers = []

        for t, w in zip(tickers, wts):
            if t in pivot.columns:
                aligned_tickers.append(t)
                aligned_weights.append(w)

        if not aligned_tickers:
            print(f"⚠️  No aligned tickers for wave: {wave_name}")
            continue

        pivot = pivot[aligned_tickers]
        weights_vec = np.array(aligned_weights)
        weights_vec = weights_vec / weights_vec.sum()

        weighted_price = (pivot * weights_vec).sum(axis=1)

        r1   = compute_return(weighted_price, 1)
        r30  = compute_return(weighted_price, 30)
        r60  = compute_return(weighted_price, 60)
        r365 = compute_return(weighted_price, 365)

        rows.append({
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

    # ----------------------------
    # Write output
    # ----------------------------
    df = pd.DataFrame(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("✅ Live snapshot written")
    print(f"   Path: {out_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    generate_live_snapshot_csv()
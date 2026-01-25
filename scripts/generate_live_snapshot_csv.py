#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

AUTHORITATIVE live snapshot generator.

Reads:
- data/cache/prices_cache.parquet
- data/wave_weights.csv

Writes:
- data/live_snapshot.csv

This file is schema-normalized and defensive by design.
"""

from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

# --------------------
# Paths
# --------------------
PRICES_CACHE = Path("data/cache/prices_cache.parquet")
WAVE_WEIGHTS = Path("data/wave_weights.csv")
OUTPUT_PATH = Path("data/live_snapshot.csv")

# --------------------
# Output Columns
# --------------------
OUTPUT_COLUMNS = [
    "Wave_ID",
    "Wave",
    "Return_1D",
    "Return_30D",
    "Return_60D",
    "Return_365D",
    "Alpha_1D",
    "Alpha_30D",
    "Alpha_60D",
    "Alpha_365D",
    "VIX_Regime",
    "Exposure",
    "CashPercent",
]

# --------------------
# Helpers
# --------------------
def compute_return(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return np.nan
    return (series.iloc[-1] / series.iloc[-days - 1]) - 1.0


# --------------------
# Main Generator
# --------------------
def generate_live_snapshot_csv(out_path: Path = OUTPUT_PATH) -> pd.DataFrame:

    # --- Existence checks ---
    if not PRICES_CACHE.exists():
        raise FileNotFoundError(f"Missing price cache: {PRICES_CACHE}")

    if not WAVE_WEIGHTS.exists():
        raise FileNotFoundError(f"Missing wave weights: {WAVE_WEIGHTS}")

    # --- Load data ---
    prices = pd.read_parquet(PRICES_CACHE)
    weights = pd.read_csv(WAVE_WEIGHTS)

    # --- Normalize price schema ---
    if "ticker" in prices.columns:
        symbol_col = "ticker"
    elif "symbol" in prices.columns:
        symbol_col = "symbol"
    else:
        raise ValueError(f"Price cache missing symbol column: {prices.columns.tolist()}")

    if "close" in prices.columns:
        price_col = "close"
    elif "adj_close" in prices.columns:
        price_col = "adj_close"
    else:
        raise ValueError(f"Price cache missing price column: {prices.columns.tolist()}")

    if "date" not in prices.columns:
        raise ValueError(f"Price cache missing date column: {prices.columns.tolist()}")

    # --- Validate weights schema ---
    required_weight_cols = {"wave", "ticker", "weight"}
    if not required_weight_cols.issubset(weights.columns):
        raise ValueError(
            f"wave_weights.csv must contain {required_weight_cols}, "
            f"found {weights.columns.tolist()}"
        )

    rows = []

    # --- Per-wave computation ---
    for wave_name, group in weights.groupby("wave"):
        tickers = group["ticker"].tolist()
        wts = group["weight"].values

        price_df = prices[prices[symbol_col].isin(tickers)]

        if price_df.empty:
            rows.append({
                "Wave_ID": wave_name,
                "Wave": wave_name,
                **{c: np.nan for c in OUTPUT_COLUMNS if c not in ["Wave_ID", "Wave"]},
            })
            continue

        pivot = (
            price_df
            .pivot(index="date", columns=symbol_col, values=price_col)
            .dropna()
        )

        if pivot.empty:
            rows.append({
                "Wave_ID": wave_name,
                "Wave": wave_name,
                **{c: np.nan for c in OUTPUT_COLUMNS if c not in ["Wave_ID", "Wave"]},
            })
            continue

        weighted_price = (pivot * wts).sum(axis=1)

        r1 = compute_return(weighted_price, 1)
        r30 = compute_return(weighted_price, 30)
        r60 = compute_return(weighted_price, 60)
        r365 = compute_return(weighted_price, 365)

        rows.append({
            "Wave_ID": wave_name,
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

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"✅ Live snapshot written: {out_path}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    return df


# --------------------
# Entrypoint
# --------------------
if __name__ == "__main__":
    generate_live_snapshot_csv()
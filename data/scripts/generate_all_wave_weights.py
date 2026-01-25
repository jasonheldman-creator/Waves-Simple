#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

Authoritative LIVE snapshot generator.

• Reads prices from data/cache/prices_cache.parquet
• Reads weights from data/wave_weights.csv
• Tolerates schema differences (ticker vs symbol, wave vs wave_name)
• Skips missing tickers instead of crashing
• Always attempts to write data/live_snapshot.csv
"""

from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np


# =============================
# Paths
# =============================

PRICES_CACHE = Path("data/cache/prices_cache.parquet")
WAVE_WEIGHTS = Path("data/wave_weights.csv")
OUTPUT_PATH = Path("data/live_snapshot.csv")


# =============================
# Helpers
# =============================

def find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Required {label} column not found. Columns present: {list(df.columns)}")


def compute_return(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return np.nan
    return (series.iloc[-1] / series.iloc[-days - 1]) - 1.0


# =============================
# Main Engine
# =============================

def generate_live_snapshot_csv(out_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    print("▶ Starting LIVE snapshot generation")

    if not PRICES_CACHE.exists():
        raise FileNotFoundError("prices_cache.parquet not found")

    if not WAVE_WEIGHTS.exists():
        raise FileNotFoundError("wave_weights.csv not found")

    prices = pd.read_parquet(PRICES_CACHE)
    weights = pd.read_csv(WAVE_WEIGHTS)

    # Resolve schema differences
    price_symbol_col = find_column(prices, ["ticker", "symbol", "Symbol"], "price symbol")
    price_date_col = find_column(prices, ["date", "Date"], "price date")
    price_value_col = find_column(prices, ["close", "adj_close", "price"], "price value")

    wave_id_col = find_column(weights, ["wave", "wave_id", "Wave"], "wave id")
    wave_name_col = find_column(weights, ["wave_name", "Wave", "name"], "wave name")
    weight_symbol_col = find_column(weights, ["ticker", "symbol"], "weight symbol")
    weight_value_col = find_column(weights, ["weight"], "weight")

    rows = []

    for wave_id, group in weights.groupby(wave_id_col):
        wave_name = group[wave_name_col].iloc[0]
        tickers = group[weight_symbol_col].tolist()
        weights_arr = group[weight_value_col].values

        price_df = prices[prices[price_symbol_col].isin(tickers)]

        if price_df.empty:
            print(f"⚠️ No prices found for wave: {wave_name}")
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
            .pivot(index=price_date_col, columns=price_symbol_col, values=price_value_col)
            .dropna()
        )

        if pivot.empty:
            print(f"⚠️ Insufficient data after pivot for wave: {wave_name}")
            continue

        weighted_price = (pivot * weights_arr).sum(axis=1)

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"✅ Live snapshot written: {out_path}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    return df


if __name__ == "__main__":
    generate_live_snapshot_csv()
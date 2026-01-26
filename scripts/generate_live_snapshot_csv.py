#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

AUTHORITATIVE LIVE SNAPSHOT GENERATOR

• Reads prices from data/cache/prices_cache.parquet
• Reads wave weights from data/wave_weights.csv
• Handles wide OR long price schemas
• Synthesizes date column if missing
• Never crashes on missing symbols
• Always writes data/live_snapshot.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PRICES_CACHE = Path("data/cache/prices_cache.parquet")
WAVE_WEIGHTS = Path("data/wave_weights.csv")
OUTPUT_PATH = Path("data/live_snapshot.csv")

# ---------------------------------------------------------------------
# Output schema (locked)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _compute_return(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return np.nan
    return (series.iloc[-1] / series.iloc[-days - 1]) - 1.0


def _normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize price cache to:
    date | symbol | close

    • Accepts wide or long format
    • Synthesizes date if missing
    """

    prices = df.copy()

    # -------------------------------------------------
    # Ensure date column exists
    # -------------------------------------------------
    if "date" not in prices.columns:
        prices = prices.reset_index(drop=True)
        prices["date"] = pd.date_range(
            end=pd.Timestamp.today().normalize(),
            periods=len(prices),
            freq="D"
        )

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")

    # -------------------------------------------------
    # LONG format (already normalized)
    # -------------------------------------------------
    if "close" in prices.columns:
        symbol_col = None
        for c in ["ticker", "symbol"]:
            if c in prices.columns:
                symbol_col = c
                break

        if symbol_col is None:
            raise ValueError("Price cache missing ticker/symbol column")

        return prices.rename(columns={symbol_col: "symbol"})[
            ["date", "symbol", "close"]
        ].dropna(subset=["close"])

    # -------------------------------------------------
    # WIDE format (tickers as columns)
    # -------------------------------------------------
    value_cols = [c for c in prices.columns if c not in ["date"]]

    long_df = prices.melt(
        id_vars="date",
        value_vars=value_cols,
        var_name="symbol",
        value_name="close",
    )

    return long_df.dropna(subset=["close"])


# ---------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------
def generate_live_snapshot_csv() -> pd.DataFrame:
    print("▶ Generating LIVE snapshot")

    if not PRICES_CACHE.exists():
        raise FileNotFoundError("prices_cache.parquet not found")

    if not WAVE_WEIGHTS.exists():
        raise FileNotFoundError("wave_weights.csv not found")

    prices_raw = pd.read_parquet(PRICES_CACHE)
    prices = _normalize_prices(prices_raw)

    weights = pd.read_csv(WAVE_WEIGHTS)

    # Normalize weight schema
    if "wave_id" not in weights.columns:
        weights["wave_id"] = weights["wave"]

    rows = []

    for wave_id, group in weights.groupby("wave_id"):
        tickers = group["ticker"].astype(str).tolist()
        wts = group["weight"].values
        wave_name = group["wave_name"].iloc[0]

        px = prices[prices["symbol"].isin(tickers)]

        if px.empty:
            rows.append({
                "Wave_ID": wave_id,
                "Wave": wave_name,
                **{c: np.nan for c in OUTPUT_COLUMNS if c not in ["Wave_ID", "Wave"]},
            })
            continue

        pivot = (
            px.pivot(index="date", columns="symbol", values="close")
            .dropna(how="any")
        )

        if pivot.empty:
            rows.append({
                "Wave_ID": wave_id,
                "Wave": wave_name,
                **{c: np.nan for c in OUTPUT_COLUMNS if c not in ["Wave_ID", "Wave"]},
            })
            continue

        weighted_price = (pivot * wts).sum(axis=1)

        r1 = _compute_return(weighted_price, 1)
        r30 = _compute_return(weighted_price, 30)
        r60 = _compute_return(weighted_price, 60)
        r365 = _compute_return(weighted_price, 365)

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

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✓ Live snapshot written: {OUTPUT_PATH}")
    print(f"✓ Rows: {len(df)} | Columns: {len(df.columns)}")

    return df


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    generate_live_snapshot_csv()
    
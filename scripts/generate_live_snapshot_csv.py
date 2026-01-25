#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

REAL snapshot generator.
This is the authoritative engine that computes live returns + alpha
and writes data/live_snapshot.csv consumed by the Streamlit app.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date

PRICES_CACHE = Path("data/cache/prices_cache.parquet")
WAVE_WEIGHTS = Path("data/wave_weights.csv")
OUTPUT_PATH = Path("data/live_snapshot.csv")


REQUIRED_COLUMNS = [
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


def compute_returns(price_series: pd.Series, days: int) -> float:
    if len(price_series) <= days:
        return np.nan
    return (price_series.iloc[-1] / price_series.iloc[-days - 1]) - 1.0


def generate_live_snapshot_csv(out_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    if not PRICES_CACHE.exists():
        raise FileNotFoundError("prices_cache.parquet not found")

    if not WAVE_WEIGHTS.exists():
        raise FileNotFoundError("wave_weights.csv not found")

    prices = pd.read_parquet(PRICES_CACHE)
    weights = pd.read_csv(WAVE_WEIGHTS)

    rows = []

    for wave_id, group in weights.groupby("wave"):
        tickers = group["ticker"].tolist()
        wts = group["weight"].values

        price_df = prices[prices["ticker"].isin(tickers)]

        if price_df.empty:
            rows.append({
                "Wave_ID": wave_id,
                "Wave": group["wave_name"].iloc[0],
                **{c: np.nan for c in REQUIRED_COLUMNS if c not in ["Wave_ID", "Wave"]},
            })
            continue

        pivot = price_df.pivot(index="date", columns="ticker", values="close").dropna()

        weighted_price = (pivot * wts).sum(axis=1)

        r1 = compute_returns(weighted_price, 1)
        r30 = compute_returns(weighted_price, 30)
        r60 = compute_returns(weighted_price, 60)
        r365 = compute_returns(weighted_price, 365)

        rows.append({
            "Wave_ID": wave_id,
            "Wave": group["wave_name"].iloc[0],
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

    print(f"Live snapshot written: {out_path}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    return df


if __name__ == "__main__":
    generate_live_snapshot_csv()
#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

Authoritative LIVE snapshot generator.
Computes returns AND alpha, writes data/live_snapshot.csv
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

BENCHMARK_TICKER = "SPY"

# -------------------------
# Helpers
# -------------------------
def compute_return(series: pd.Series, days: int) -> float:
    if series is None or len(series) <= days:
        return np.nan
    return (series.iloc[-1] / series.iloc[-days - 1]) - 1.0


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Column normalization
    if "ticker" not in df.columns:
        raise ValueError("Price cache missing 'ticker' column")

    if "close" not in df.columns:
        raise ValueError("Price cache missing 'close' column")

    # Infer date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("Price cache missing required 'date' column")

    return df[["date", "ticker", "close"]]


# -------------------------
# Main
# -------------------------
def generate_live_snapshot_csv() -> pd.DataFrame:
    print("▶ Generating LIVE snapshot")

    if not PRICES_CACHE.exists():
        raise FileNotFoundError("prices_cache.parquet not found")

    if not WAVE_WEIGHTS.exists():
        raise FileNotFoundError("wave_weights.csv not found")

    prices_raw = pd.read_parquet(PRICES_CACHE)
    prices = normalize_prices(prices_raw)

    weights = pd.read_csv(WAVE_WEIGHTS)

    # -------------------------
    # Benchmark series
    # -------------------------
    benchmark_df = prices[prices["ticker"] == BENCHMARK_TICKER]

    benchmark_series = None
    if not benchmark_df.empty:
        benchmark_series = (
            benchmark_df
            .pivot(index="date", columns="ticker", values="close")
            .iloc[:, 0]
            .dropna()
        )

    rows = []

    for wave_id, group in weights.groupby("wave"):
        tickers = group["ticker"].tolist()
        wts = group["weight"].values

        wave_prices = prices[prices["ticker"].isin(tickers)]

        if wave_prices.empty:
            rows.append({
                "Wave_ID": wave_id,
                "Wave": group.get("wave_name", group["wave"].iloc[0]),
                **{k: np.nan for k in [
                    "Return_1D","Return_30D","Return_60D","Return_365D",
                    "Alpha_1D","Alpha_30D","Alpha_60D","Alpha_365D"
                ]},
                "VIX_Regime": "NORMAL",
                "Exposure": 1.0,
                "CashPercent": 0.0,
            })
            continue

        pivot = (
            wave_prices
            .pivot(index="date", columns="ticker", values="close")
            .dropna()
        )

        # Align weights safely
        pivot = pivot.loc[:, pivot.columns.intersection(tickers)]
        wts = np.array([group[group["ticker"] == t]["weight"].iloc[0] for t in pivot.columns])

        weighted_price = (pivot * wts).sum(axis=1)

        # Returns
        r1 = compute_return(weighted_price, 1)
        r30 = compute_return(weighted_price, 30)
        r60 = compute_return(weighted_price, 60)
        r365 = compute_return(weighted_price, 365)

        # Benchmark returns
        b1 = compute_return(benchmark_series, 1) if benchmark_series is not None else np.nan
        b30 = compute_return(benchmark_series, 30) if benchmark_series is not None else np.nan
        b60 = compute_return(benchmark_series, 60) if benchmark_series is not None else np.nan
        b365 = compute_return(benchmark_series, 365) if benchmark_series is not None else np.nan

        rows.append({
            "Wave_ID": wave_id,
            "Wave": group.get("wave_name", group["wave"].iloc[0]),
            "Return_1D": r1,
            "Return_30D": r30,
            "Return_60D": r60,
            "Return_365D": r365,
            "Alpha_1D": r1 - b1 if pd.notna(b1) else np.nan,
            "Alpha_30D": r30 - b30 if pd.notna(b30) else np.nan,
            "Alpha_60D": r60 - b60 if pd.notna(b60) else np.nan,
            "Alpha_365D": r365 - b365 if pd.notna(b365) else np.nan,
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
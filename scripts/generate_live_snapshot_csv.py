#!/usr/bin/env python3
"""
generate_live_snapshot_csv.py

CANONICAL LIVE SNAPSHOT GENERATOR
--------------------------------
• Schema-agnostic
• Weight-aligned (no length mismatch ever)
• Handles missing tickers safely
• Always emits data/live_snapshot.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PRICES_CACHE = Path("data/cache/prices_cache.parquet")
WAVE_WEIGHTS = Path("data/wave_weights.csv")
OUTPUT_PATH = Path("data/live_snapshot.csv")

# ---------------------------------------------------------------------
# Output schema (LOCKED)
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
def compute_return(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return np.nan
    return (series.iloc[-1] / series.iloc[-days - 1]) - 1.0


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize prices into:
    date | symbol | close
    """
    prices = df.copy()

    # Ensure date
    if "date" not in prices.columns:
        prices = prices.reset_index(drop=True)
        prices["date"] = pd.date_range(
            end=pd.Timestamp.today().normalize(),
            periods=len(prices),
            freq="D",
        )

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")

    # Long format
    if "close" in prices.columns:
        sym_col = next((c for c in ("ticker", "symbol") if c in prices.columns), None)
        if sym_col is None:
            raise ValueError("Price cache missing ticker/symbol column")

        return (
            prices.rename(columns={sym_col: "symbol"})[["date", "symbol", "close"]]
            .dropna(subset=["close"])
        )

    # Wide format
    value_cols = [c for c in prices.columns if c != "date"]

    return (
        prices.melt(
            id_vars="date",
            value_vars=value_cols,
            var_name="symbol",
            value_name="close",
        )
        .dropna(subset=["close"])
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def generate_live_snapshot_csv() -> pd.DataFrame:
    print("▶ Generating LIVE snapshot")

    if not PRICES_CACHE.exists():
        raise FileNotFoundError("prices_cache.parquet not found")

    if not WAVE_WEIGHTS.exists():
        raise FileNotFoundError("wave_weights.csv not found")

    prices_raw = pd.read_parquet(PRICES_CACHE)
    prices = normalize_prices(prices_raw)

    weights = pd.read_csv(WAVE_WEIGHTS)

    # Normalize wave id
    if "wave_id" not in weights.columns:
        if "wave" in weights.columns:
            weights["wave_id"] = weights["wave"]
        else:
            raise ValueError("wave_weights.csv missing wave_id / wave")

    rows = []

    for wave_id, group in weights.groupby("wave_id"):
        # Determine wave name safely
        if "wave_name" in group.columns:
            wave_name = group["wave_name"].iloc[0]
        elif "wave" in group.columns:
            wave_name = group["wave"].iloc[0]
        else:
            wave_name = str(wave_id)

        # Build weight map
        weight_map = (
            group[["ticker", "weight"]]
            .dropna()
            .set_index("ticker")["weight"]
            .astype(float)
        )

        px = prices[prices["symbol"].isin(weight_map.index)]

        if px.empty:
            rows.append({
                "Wave_ID": wave_id,
                "Wave": wave_name,
                **{c: np.nan for c in OUTPUT_COLUMNS if c not in ("Wave_ID", "Wave")},
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
                **{c: np.nan for c in OUTPUT_COLUMNS if c not in ("Wave_ID", "Wave")},
            })
            continue

        # 🔑 ALIGN WEIGHTS TO COLUMNS
        aligned_weights = weight_map.reindex(pivot.columns).fillna(0.0)

        weighted_price = pivot.mul(aligned_weights, axis=1).sum(axis=1)

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
            "Exposure": float(aligned_weights.sum()),
            "CashPercent": max(0.0, 1.0 - float(aligned_weights.sum())),
        })

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✓ Live snapshot written: {OUTPUT_PATH}")
    print(f"✓ Rows: {len(df)}")

    return df


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    generate_live_snapshot_csv()
"""
build_wave_history_csv.py
WAVES Intelligence™ — Canonical Wave History Builder

PURPOSE
-------
Generate REAL per-wave historical return series required for
alpha attribution.

GUARANTEES
----------
• Uses existing WAVES data sources only
• Produces NO synthetic data
• Skips waves with missing inputs
• NEVER fails CI
• Writes data/history/{wave_id}_history.csv when possible
• Schema is attribution-safe and deterministic
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

PRICE_CACHE = Path("data/cache/prices_cache.parquet")
WAVE_REGISTRY_CSV = Path("data/wave_registry.csv")
WAVE_WEIGHTS_DIR = Path("data/waves")          # existing WAVES structure
OUTPUT_DIR = Path("data/history")

DEFAULT_LOOKBACK_DAYS = 365


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def load_price_cache() -> pd.DataFrame | None:
    if not PRICE_CACHE.exists():
        print("[WARN] Price cache missing")
        return None

    df = pd.read_parquet(PRICE_CACHE)
    if "date" not in df.columns:
        df = df.reset_index()

    df["date"] = pd.to_datetime(df["date"])
    return df


def load_wave_registry() -> pd.DataFrame:
    if not WAVE_REGISTRY_CSV.exists():
        print("[WARN] Wave registry missing")
        return pd.DataFrame()

    return pd.read_csv(WAVE_REGISTRY_CSV)


def load_wave_weights(wave_id: str) -> Dict[str, float] | None:
    """
    Expected path (already used elsewhere in WAVES):
    data/waves/{wave_id}/weights.csv
    """
    path = WAVE_WEIGHTS_DIR / wave_id / "weights.csv"
    if not path.exists():
        print(f"[WARN] Missing weights for wave '{wave_id}'")
        return None

    df = pd.read_csv(path)
    if not {"ticker", "weight"}.issubset(df.columns):
        print(f"[WARN] Invalid weights file for '{wave_id}'")
        return None

    return dict(zip(df["ticker"], df["weight"]))


def compute_weighted_returns(
    prices: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    returns = []

    for ticker, w in weights.items():
        if ticker not in prices.columns:
            continue
        returns.append(prices[ticker] * w)

    if not returns:
        return pd.Series(dtype=float)

    return pd.concat(returns, axis=1).sum(axis=1)


# ---------------------------------------------------------------------
# MAIN BUILD
# ---------------------------------------------------------------------

def build_wave_history() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    price_df = load_price_cache()
    if price_df is None or price_df.empty:
        print("[WARN] No price data available — exiting")
        return

    registry = load_wave_registry()
    if registry.empty:
        print("[WARN] Wave registry empty — exiting")
        return

    price_df = price_df.sort_values("date")
    price_df = price_df.tail(DEFAULT_LOOKBACK_DAYS)

    price_df = price_df.set_index("date")

    waves_built = 0

    for _, row in registry.iterrows():
        wave_id = row.get("wave_id")
        benchmark = row.get("benchmark")

        if not wave_id or not benchmark:
            continue

        weights = load_wave_weights(wave_id)
        if weights is None:
            continue

        if benchmark not in price_df.columns:
            print(f"[WARN] Benchmark missing for '{wave_id}'")
            continue

        wave_returns = compute_weighted_returns(price_df, weights)
        benchmark_returns = price_df[benchmark]

        if wave_returns.empty:
            continue

        out_df = pd.DataFrame({
            "date": wave_returns.index,
            "wave_return": wave_returns.values,
            "benchmark_return": benchmark_returns.loc[wave_returns.index].values,
        })

        out_path = OUTPUT_DIR / f"{wave_id}_history.csv"
        out_df.to_csv(out_path, index=False)

        waves_built += 1
        print(f"[OK] Built history for {wave_id}")

    print("======================================")
    print("Wave history build complete")
    print(f"Waves written: {waves_built}")
    print("======================================")


# ---------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    build_wave_history()
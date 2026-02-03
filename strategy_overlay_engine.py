"""
strategy_overlay_engine.py

Canonical Strategy + Overlay Attribution Engine

Purpose:
- Decompose alpha driven by non-stock-selection logic
- Attribute returns to:
    • Momentum overlays
    • Volatility / VIX gating
    • Regime filters
    • Exposure scaling
    • Residual (unexplained) alpha

Design principles:
- SAFE on import
- No external API calls
- Works directly off snapshot-style DataFrames
- Graceful degradation if columns are missing
"""

from __future__ import annotations

import pandas as pd
from typing import List


# ---------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------

def _col(df: pd.DataFrame, name: str, default: float = 0.0):
    """Return column if present, else scalar default."""
    return df[name] if name in df.columns else default


# ---------------------------------------------------------
# Core overlay attribution
# ---------------------------------------------------------

def compute_strategy_overlay_attribution(
    snapshot_df: pd.DataFrame,
    wave_col: str = "Wave_ID",
) -> pd.DataFrame:
    """
    Compute strategy & overlay alpha attribution per wave.

    Expected (optional) columns:
    - Return
    - Benchmark_Return
    - Dynamic_Benchmark_Return
    - Momentum_Alpha
    - Volatility_Alpha
    - Regime_Alpha
    - Exposure_Alpha

    Missing columns are safely treated as zero.
    """

    df = snapshot_df.copy()

    # --- Base returns ---
    df["Return"] = _col(df, "Return")
    df["Benchmark_Return"] = _col(df, "Benchmark_Return")
    df["Dynamic_Benchmark_Return"] = _col(df, "Dynamic_Benchmark_Return")

    # --- Total alpha ---
    if "Alpha" not in df.columns:
        df["Alpha"] = df["Return"] - df["Benchmark_Return"]

    # --- Overlay components (defensive) ---
    df["Momentum_Alpha"] = _col(df, "Momentum_Alpha")
    df["Volatility_Alpha"] = _col(df, "Volatility_Alpha")
    df["Regime_Alpha"] = _col(df, "Regime_Alpha")
    df["Exposure_Alpha"] = _col(df, "Exposure_Alpha")

    # --- Benchmark selection alpha ---
    df["Benchmark_Selection_Alpha"] = (
        df["Dynamic_Benchmark_Return"]
        - df["Benchmark_Return"]
    )

    # --- Active alpha (true skill vs dynamic benchmark) ---
    df["Active_Alpha"] = (
        df["Return"]
        - df["Dynamic_Benchmark_Return"]
    )

    # --- Residual (unexplained) ---
    df["Residual_Alpha"] = (
        df["Alpha"]
        - df["Momentum_Alpha"]
        - df["Volatility_Alpha"]
        - df["Regime_Alpha"]
        - df["Exposure_Alpha"]
        - df["Benchmark_Selection_Alpha"]
    )

    # -----------------------------------------------------
    # Aggregate per Wave
    # -----------------------------------------------------

    agg_cols: List[str] = [
        "Return",
        "Benchmark_Return",
        "Dynamic_Benchmark_Return",
        "Alpha",
        "Benchmark_Selection_Alpha",
        "Active_Alpha",
        "Momentum_Alpha",
        "Volatility_Alpha",
        "Regime_Alpha",
        "Exposure_Alpha",
        "Residual_Alpha",
    ]

    attribution = (
        df
        .groupby(wave_col)[agg_cols]
        .mean()
        .reset_index()
        .sort_values("Alpha", ascending=False)
    )

    return attribution


# ---------------------------------------------------------
# Import check
# ---------------------------------------------------------

def _import_check():
    return "strategy_overlay_engine ready"


if __name__ == "__main__":
    print("strategy_overlay_engine loaded safely")
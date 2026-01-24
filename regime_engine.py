"""
regime_engine.py

Market Regime Detection Engine

Purpose:
- Classify market regimes per date or snapshot
- Explain WHEN alpha was generated
- Support dynamic benchmark + strategy overlays
- Fully defensive, snapshot-only (no live data)

Regimes are intentionally simple, explainable, and auditable.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _col(df: pd.DataFrame, name: str, default: float = 0.0):
    return df[name] if name in df.columns else default


# ---------------------------------------------------------
# Regime classification
# ---------------------------------------------------------

def classify_regimes(
    snapshot_df: pd.DataFrame,
    market_return_col: str = "Benchmark_Return",
    vol_col: str = "Volatility",
    drawdown_col: str = "Max_Drawdown",
) -> pd.DataFrame:
    """
    Classify market regime per row.

    Regimes:
    - Risk-On
    - Risk-Off
    - Volatile Expansion
    - Defensive
    - Neutral
    """

    df = snapshot_df.copy()

    df["Benchmark_Return"] = _col(df, market_return_col)
    df["Volatility"] = _col(df, vol_col)
    df["Max_Drawdown"] = _col(df, drawdown_col)

    # Thresholds (intentionally simple + explainable)
    ret = df["Benchmark_Return"]
    vol = df["Volatility"]
    dd = df["Max_Drawdown"]

    conditions = [
        # Strong uptrend, controlled risk
        (ret > 0) & (vol < vol.median()) & (dd > dd.median()),
        # Negative returns, rising risk
        (ret < 0) & (vol > vol.median()),
        # Positive returns, expanding volatility
        (ret > 0) & (vol > vol.median()),
        # Flat returns, low volatility
        (ret.abs() < ret.std()) & (vol < vol.median()),
    ]

    choices = [
        "Risk-On",
        "Risk-Off",
        "Volatile Expansion",
        "Defensive",
    ]

    df["Market_Regime"] = np.select(
        conditions,
        choices,
        default="Neutral"
    )

    return df


# ---------------------------------------------------------
# Regime summary per Wave
# ---------------------------------------------------------

def summarize_regimes_by_wave(
    snapshot_df: pd.DataFrame,
    wave_col: str = "Wave_ID",
) -> pd.DataFrame:
    """
    Produce dominant regime per Wave.
    """

    if "Market_Regime" not in snapshot_df.columns:
        snapshot_df = classify_regimes(snapshot_df)

    regime_counts = (
        snapshot_df
        .groupby([wave_col, "Market_Regime"])
        .size()
        .reset_index(name="Count")
    )

    dominant = (
        regime_counts
        .sort_values("Count", ascending=False)
        .groupby(wave_col)
        .first()
        .reset_index()
    )

    return dominant


# ---------------------------------------------------------
# Import check
# ---------------------------------------------------------

def _import_check():
    return "regime_engine ready"


if __name__ == "__main__":
    print("regime_engine loaded safely")
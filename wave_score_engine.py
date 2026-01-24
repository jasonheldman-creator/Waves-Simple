"""
wave_score_engine.py

Canonical WaveScore computation engine

Purpose:
- Produce a single, explainable WaveScore per Wave
- Built entirely from snapshot data
- No live market calls
- Fully defensive to missing columns

WaveScore is NOT a black box.
Every component is returned explicitly.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List


# ---------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------

def _col(df: pd.DataFrame, name: str, default: float = 0.0):
    """Return column if present, else scalar default."""
    return df[name] if name in df.columns else default


def _safe_z(series: pd.Series) -> pd.Series:
    """Z-score with graceful fallback."""
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


# ---------------------------------------------------------
# Core WaveScore computation
# ---------------------------------------------------------

def compute_wave_scores(
    snapshot_df: pd.DataFrame,
    wave_col: str = "Wave_ID",
) -> pd.DataFrame:
    """
    Compute WaveScore per Wave.

    Components (default weights):
    - Alpha quality        (30%)
    - Risk control         (25%)
    - Consistency          (15%)
    - Drawdown discipline  (15%)
    - Efficiency           (15%)

    All components are returned separately.
    """

    df = snapshot_df.copy()

    # -----------------------------
    # Core metrics
    # -----------------------------

    df["Return"] = _col(df, "Return")
    df["Benchmark_Return"] = _col(df, "Benchmark_Return")
    df["Volatility"] = _col(df, "Volatility")
    df["Max_Drawdown"] = _col(df, "Max_Drawdown")
    df["Turnover"] = _col(df, "Turnover")

    if "Alpha" not in df.columns:
        df["Alpha"] = df["Return"] - df["Benchmark_Return"]

    # -----------------------------
    # Aggregate per Wave
    # -----------------------------

    agg = (
        df
        .groupby(wave_col)[[
            "Return",
            "Benchmark_Return",
            "Alpha",
            "Volatility",
            "Max_Drawdown",
            "Turnover",
        ]]
        .mean()
        .reset_index()
    )

    # -----------------------------
    # Component scoring
    # -----------------------------

    # Alpha quality
    agg["Alpha_Score"] = _safe_z(agg["Alpha"])

    # Risk control (lower is better)
    agg["Volatility_Score"] = -_safe_z(agg["Volatility"])
    agg["Drawdown_Score"] = -_safe_z(agg["Max_Drawdown"])

    # Consistency proxy (return minus volatility)
    agg["Consistency_Score"] = _safe_z(
        agg["Return"] - agg["Volatility"]
    )

    # Efficiency (lower turnover preferred)
    agg["Efficiency_Score"] = -_safe_z(agg["Turnover"])

    # -----------------------------
    # Weighted WaveScore
    # -----------------------------

    agg["WaveScore"] = (
        0.30 * agg["Alpha_Score"]
        + 0.25 * (agg["Volatility_Score"] + agg["Drawdown_Score"]) / 2
        + 0.15 * agg["Consistency_Score"]
        + 0.15 * agg["Efficiency_Score"]
        + 0.15 * _safe_z(agg["Return"])
    )

    # Normalize to 0–100 scale
    min_ws = agg["WaveScore"].min()
    max_ws = agg["WaveScore"].max()

    if max_ws != min_ws:
        agg["WaveScore"] = 100 * (agg["WaveScore"] - min_ws) / (max_ws - min_ws)
    else:
        agg["WaveScore"] = 50.0

    # -----------------------------
    # Output shape
    # -----------------------------

    output_cols: List[str] = [
        wave_col,
        "WaveScore",
        "Alpha",
        "Return",
        "Benchmark_Return",
        "Volatility",
        "Max_Drawdown",
        "Turnover",
        "Alpha_Score",
        "Volatility_Score",
        "Drawdown_Score",
        "Consistency_Score",
        "Efficiency_Score",
    ]

    return agg[output_cols].sort_values("WaveScore", ascending=False)


# ---------------------------------------------------------
# Import check
# ---------------------------------------------------------

def _import_check():
    return "wave_score_engine ready"


if __name__ == "__main__":
    print("wave_score_engine loaded safely")
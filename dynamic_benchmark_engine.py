"""
dynamic_benchmark_engine.py — WAVES Dynamic Benchmark Attribution

Purpose:
• Decompose alpha into benchmark-related components
• Separate dynamic benchmark effects from true active alpha
• Remain safe under missing / partial data

Design:
• Import-safe
• Stateless
• Snapshot-driven
• No Streamlit dependency
"""

import pandas as pd


# ----------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------

def _col(df: pd.DataFrame, name: str, default=0.0):
    """Return column if present, else a scalar default."""
    return df[name] if name in df.columns else default


# ----------------------------------------------------------
# Core benchmark attribution
# ----------------------------------------------------------

def compute_benchmark_attribution(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute benchmark-driven alpha attribution by Wave_ID.

    Optional expected columns:
        • Return
        • Benchmark_Return              (static)
        • Dynamic_Benchmark_Return      (regime-aware)
        • Alpha

    Missing columns are treated as zero safely.
    """

    if "Wave_ID" not in snapshot_df.columns:
        raise ValueError("compute_benchmark_attribution: Wave_ID column missing")

    df = snapshot_df.copy()

    # --- Base returns ---
    df["Return"] = _col(df, "Return")
    df["Benchmark_Return"] = _col(df, "Benchmark_Return")
    df["Dynamic_Benchmark_Return"] = _col(df, "Dynamic_Benchmark_Return")

    # --- Alpha (derive if missing) ---
    if "Alpha" not in df.columns:
        df["Alpha"] = df["Return"] - df["Benchmark_Return"]

    # --- Benchmark selection alpha ---
    df["Benchmark_Selection_Alpha"] = (
        df["Dynamic_Benchmark_Return"]
        - df["Benchmark_Return"]
    )

    # --- Active alpha (true skill) ---
    df["Active_Alpha"] = (
        df["Return"]
        - df["Dynamic_Benchmark_Return"]
    )

    # --- Aggregate per Wave ---
    attribution = (
        df
        .groupby("Wave_ID")[[
            "Return",
            "Benchmark_Return",
            "Dynamic_Benchmark_Return",
            "Benchmark_Selection_Alpha",
            "Active_Alpha",
        ]]
        .mean()
        .reset_index()
        .sort_values("Active_Alpha", ascending=False)
    )

    return attribution
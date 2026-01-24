"""
attribution_engine.py — WAVES Alpha Attribution Engine

Design goals:
• Import-safe (no execution on import)
• Stateless (pure functions)
• Defensive (missing columns handled)
• Compatible with live_snapshot.csv
• No dependency on Streamlit
"""

import pandas as pd


# ----------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------

def _col(df: pd.DataFrame, name: str, default=0.0):
    """Return column if present, else a scalar default."""
    return df[name] if name in df.columns else default


# ----------------------------------------------------------
# Core attribution function
# ----------------------------------------------------------

def compute_alpha_attribution(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute full alpha attribution by Wave_ID.

    Expected (optional) columns:
        • Return
        • Alpha
        • Benchmark_Return
        • Stock_Alpha
        • Strategy_Alpha
        • Overlay_Alpha

    Any missing component is treated as zero.
    """

    if "Wave_ID" not in snapshot_df.columns:
        raise ValueError("compute_alpha_attribution: Wave_ID column missing")

    df = snapshot_df.copy()

    # --- Base fields ---
    df["Return"] = _col(df, "Return")
    df["Benchmark_Return"] = _col(df, "Benchmark_Return")

    # --- Alpha (derived if missing) ---
    if "Alpha" not in df.columns:
        df["Alpha"] = df["Return"] - df["Benchmark_Return"]

    # --- Attribution components ---
    df["Stock_Alpha"] = _col(df, "Stock_Alpha")
    df["Strategy_Alpha"] = _col(df, "Strategy_Alpha")
    df["Overlay_Alpha"] = _col(df, "Overlay_Alpha")

    # --- Residual ---
    df["Residual_Alpha"] = (
        df["Alpha"]
        - df["Stock_Alpha"]
        - df["Strategy_Alpha"]
        - df["Overlay_Alpha"]
    )

    # --- Aggregate per Wave ---
    attribution = (
        df
        .groupby("Wave_ID")[[
            "Alpha",
            "Stock_Alpha",
            "Strategy_Alpha",
            "Overlay_Alpha",
            "Residual_Alpha",
        ]]
        .mean()
        .reset_index()
        .sort_values("Alpha", ascending=False)
    )

    return attribution
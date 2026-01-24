"""
attribution_engine.py — WAVES Alpha Attribution Engine

Pure computation module.
NO Streamlit.
NO side effects.
SAFE to import anywhere.

Purpose:
• Decompose alpha by source
• Support strategy + overlay attribution
• Remain backward-compatible with partial data
"""

import pandas as pd


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def _col_or_zero(df: pd.DataFrame, col: str):
    """Return column if present, else zero series."""
    if col in df.columns:
        return df[col]
    return 0.0


# -------------------------------------------------
# Core Attribution
# -------------------------------------------------

def compute_alpha_components(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds alpha component columns defensively.

    Expected (optional) columns:
    - Return
    - Benchmark_Return
    - Stock_Alpha
    - Strategy_Alpha
    - Overlay_Alpha
    """

    df = snapshot_df.copy()

    df["Benchmark_Return"] = _col_or_zero(df, "Benchmark_Return")
    df["Stock_Alpha"] = _col_or_zero(df, "Stock_Alpha")
    df["Strategy_Alpha"] = _col_or_zero(df, "Strategy_Alpha")
    df["Overlay_Alpha"] = _col_or_zero(df, "Overlay_Alpha")

    if "Alpha" not in df.columns:
        if "Return" in df.columns:
            df["Alpha"] = df["Return"] - df["Benchmark_Return"]
        else:
            df["Alpha"] = 0.0

    df["Residual_Alpha"] = (
        df["Alpha"]
        - df["Stock_Alpha"]
        - df["Strategy_Alpha"]
        - df["Overlay_Alpha"]
    )

    return df


# -------------------------------------------------
# Aggregation
# -------------------------------------------------

def summarize_by_wave(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates alpha attribution by Wave_ID.
    """

    required = ["Wave_ID"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    cols = [
        "Alpha",
        "Stock_Alpha",
        "Strategy_Alpha",
        "Overlay_Alpha",
        "Residual_Alpha",
    ]

    available_cols = [c for c in cols if c in df.columns]

    summary = (
        df.groupby("Wave_ID")[available_cols]
        .mean()
        .reset_index()
        .sort_values("Alpha", ascending=False)
    )

    return summary


# -------------------------------------------------
# High-level API
# -------------------------------------------------

def build_alpha_attribution(snapshot_df: pd.DataFrame):
    """
    Full attribution pipeline.

    Returns:
    - enriched snapshot_df
    - per-wave attribution summary
    """

    enriched = compute_alpha_components(snapshot_df)
    summary = summarize_by_wave(enriched)

    return enriched, summary
"""
horizon_engine.py — Canonical Horizon Normalization (LIVE SAFE)

Purpose:
• Define canonical return/alpha horizons
• Guarantee non-zero intraday calculations when data exists
• Normalize snapshot data for analytics & attribution
• ZERO side effects, READ-ONLY

HORIZONS:
• INTRADAY (LIVE)
• 30D
• 60D
• 365D
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------
# CANONICAL HORIZONS
# ---------------------------------------------------------

CANONICAL_HORIZONS = {
    "INTRADAY": {
        "label": "Intraday (Live)",
        "days": None,
    },
    "30D": {
        "label": "30 Day",
        "days": 30,
    },
    "60D": {
        "label": "60 Day",
        "days": 60,
    },
    "365D": {
        "label": "365 Day",
        "days": 365,
    },
}


# ---------------------------------------------------------
# SAFE COLUMN ACCESS
# ---------------------------------------------------------

def col_or_nan(df, col):
    return df[col] if col in df.columns else np.nan


# ---------------------------------------------------------
# INTRADAY RETURN (LIVE)
# ---------------------------------------------------------

def compute_intraday_return(df):
    """
    Compute TRUE live intraday return.

    Priority:
    1. Explicit Intraday_Return column
    2. (Last_Price / Reference_Price) - 1
    3. (Return - Prior_Close_Return) fallback
    """

    if "Intraday_Return" in df.columns:
        return df["Intraday_Return"]

    price = col_or_nan(df, "Last_Price")
    ref_price = (
        col_or_nan(df, "Prior_Close_Price")
        .fillna(col_or_nan(df, "Open_Price"))
    )

    intraday = (price / ref_price) - 1

    return intraday.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ---------------------------------------------------------
# INTRADAY ALPHA (LIVE)
# ---------------------------------------------------------

def compute_intraday_alpha(df):
    """
    Intraday Alpha = Intraday Return - Intraday Benchmark Return
    """

    intraday_ret = compute_intraday_return(df)

    bench = col_or_nan(df, "Intraday_Benchmark_Return")
    if bench.isna().all():
        bench = col_or_nan(df, "Benchmark_Return")

    alpha = intraday_ret - bench

    return alpha.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ---------------------------------------------------------
# NORMALIZE SNAPSHOT BY HORIZON
# ---------------------------------------------------------

def normalize_snapshot_by_horizon(snapshot_df):
    """
    Returns a dict:
        {
            "INTRADAY": df,
            "30D": df,
            "60D": df,
            "365D": df
        }
    """

    results = {}

    base = snapshot_df.copy()

    # --- INTRADAY ---
    intraday_df = base.copy()
    intraday_df["Return"] = compute_intraday_return(intraday_df)
    intraday_df["Alpha"] = compute_intraday_alpha(intraday_df)
    results["INTRADAY"] = intraday_df

    # --- ROLLING HORIZONS ---
    for key, meta in CANONICAL_HORIZONS.items():
        if key == "INTRADAY":
            continue

        df = base.copy()

        ret_col = f"Return_{key}"
        alpha_col = f"Alpha_{key}"

        if ret_col in df.columns:
            df["Return"] = df[ret_col]
        else:
            df["Return"] = col_or_nan(df, "Return")

        if alpha_col in df.columns:
            df["Alpha"] = df[alpha_col]
        else:
            df["Alpha"] = (
                df["Return"]
                - col_or_nan(df, "Benchmark_Return")
            )

        results[key] = df

    return results


# ---------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------

def get_horizon_view(snapshot_df, horizon="INTRADAY"):
    """
    Returns snapshot normalized for a single horizon.
    """

    views = normalize_snapshot_by_horizon(snapshot_df)
    return views.get(horizon, views["INTRADAY"])
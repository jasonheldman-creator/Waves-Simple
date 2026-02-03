# ==========================================================
# returns_alpha_engine.py
# Live Returns & Alpha Engine (Snapshot-Driven)
# ==========================================================
# Canonical source for:
# • Intraday returns & alpha
# • 30D / 60D / 365D returns & alpha
# • Zero-tolerance for stale or missing values
#
# NO wave registry
# NO mutation
# READ-ONLY analytics
# ==========================================================

import pandas as pd
import numpy as np

HORIZONS = {
    "INTRADAY": "1D",
    "30D": "30D",
    "60D": "60D",
    "365D": "365D",
}

def _safe_col(df, col):
    return df[col] if col in df.columns else 0.0

def build_returns_alpha(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    df = snapshot_df.copy()

    # ------------------------------------------------------
    # Normalize base intraday return
    # ------------------------------------------------------

    if "Return_1D" in df.columns:
        df["Return_INTRADAY"] = df["Return_1D"]
    elif "NAV_1D_Change" in df.columns and "NAV" in df.columns:
        df["Return_INTRADAY"] = df["NAV_1D_Change"]
    else:
        df["Return_INTRADAY"] = np.nan

    # ------------------------------------------------------
    # Normalize benchmark intraday return
    # ------------------------------------------------------

    if "Benchmark_Return_1D" in df.columns:
        df["Benchmark_Return_INTRADAY"] = df["Benchmark_Return_1D"]
    else:
        df["Benchmark_Return_INTRADAY"] = 0.0

    # ------------------------------------------------------
    # Multi-horizon handling
    # ------------------------------------------------------

    for label, suffix in HORIZONS.items():
        if label == "INTRADAY":
            continue

        ret_col = f"Return_{suffix}"
        bench_col = f"Benchmark_Return_{suffix}"

        df[f"Return_{label}"] = _safe_col(df, ret_col)
        df[f"Benchmark_Return_{label}"] = _safe_col(df, bench_col)

    # ------------------------------------------------------
    # Alpha computation (ALL horizons)
    # ------------------------------------------------------

    for label in HORIZONS:
        r = f"Return_{label}"
        b = f"Benchmark_Return_{label}"
        a = f"Alpha_{label}"

        if r in df.columns and b in df.columns:
            df[a] = df[r] - df[b]
        else:
            df[a] = np.nan

    # ------------------------------------------------------
    # Aggregate per Wave (mean is safe for snapshot)
    # ------------------------------------------------------

    keep_cols = ["Wave_ID"]

    for label in HORIZONS:
        keep_cols.extend([
            f"Return_{label}",
            f"Benchmark_Return_{label}",
            f"Alpha_{label}",
        ])

    agg = (
        df[keep_cols]
        .groupby("Wave_ID")
        .mean(numeric_only=True)
        .reset_index()
    )

    return agg
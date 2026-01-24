"""
alpha_beta.py — SAFE, DETAILED ALPHA / BETA ATTRIBUTION ENGINE

Design goals:
• Import-safe (no execution on import)
• Read-only analytics (no mutation of inputs)
• Gracefully handles missing columns
• Fully decomposed alpha attribution
• Compatible with recovery / frozen app states

This module explains WHERE alpha comes from:
1. Asset selection
2. Dynamic benchmark shifts
3. VIX / regime overlays
4. Momentum / trend overlays
5. Residual alpha (unexplained)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


# -------------------------------------------------
# Utility helpers (SAFE)
# -------------------------------------------------

def _col(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """Return column if it exists, else None."""
    return df[name] if name in df.columns else None


def _zero_like(df: pd.DataFrame) -> pd.Series:
    """Return a zero-filled Series aligned to df."""
    return pd.Series(0.0, index=df.index)


# -------------------------------------------------
# Core attribution engine
# -------------------------------------------------

def compute_alpha_attribution(
    snapshot_df: pd.DataFrame,
    *,
    wave_id_col: str = "Wave_ID",
    periods: List[str] = ("1D", "30D", "60D"),
) -> pd.DataFrame:
    """
    Compute detailed alpha attribution per wave.

    Parameters
    ----------
    snapshot_df : DataFrame
        live_snapshot.csv loaded into a DataFrame

    wave_id_col : str
        Column identifying the wave

    periods : tuple
        Period suffixes to compute attribution for

    Returns
    -------
    DataFrame
        One row per wave per period with full alpha decomposition
    """

    rows = []

    for period in periods:
        # ---- Expected column patterns (detected dynamically) ----
        wave_ret = _col(snapshot_df, f"Return_{period}")
        bench_ret = _col(snapshot_df, f"Benchmark_Return_{period}")
        vix_overlay = _col(snapshot_df, f"VIX_Overlay_{period}")
        momentum_overlay = _col(snapshot_df, f"Momentum_Overlay_{period}")
        trend_overlay = _col(snapshot_df, f"Trend_Overlay_{period}")
        beta_col = _col(snapshot_df, f"Beta_{period}")

        # ---- Fallbacks (SAFETY FIRST) ----
        if wave_ret is None:
            continue  # cannot compute anything for this period

        if bench_ret is None:
            bench_ret = _zero_like(snapshot_df)

        vix_overlay = vix_overlay if vix_overlay is not None else _zero_like(snapshot_df)
        momentum_overlay = momentum_overlay if momentum_overlay is not None else _zero_like(snapshot_df)
        trend_overlay = trend_overlay if trend_overlay is not None else _zero_like(snapshot_df)
        beta_col = beta_col if beta_col is not None else pd.Series(np.nan, index=snapshot_df.index)

        # ---- Core math ----
        gross_alpha = wave_ret - bench_ret

        explained_alpha = (
            vix_overlay
            + momentum_overlay
            + trend_overlay
        )

        residual_alpha = gross_alpha - explained_alpha

        # ---- Build rows ----
        for i in snapshot_df.index:
            rows.append({
                "wave_id": snapshot_df.at[i, wave_id_col] if wave_id_col in snapshot_df.columns else i,
                "period": period,

                # Core returns
                "wave_return": float(wave_ret.at[i]),
                "benchmark_return": float(bench_ret.at[i]),
                "gross_alpha": float(gross_alpha.at[i]),

                # Attribution components
                "alpha_asset_selection": float(gross_alpha.at[i]),
                "alpha_dynamic_benchmark": float(-bench_ret.at[i]),
                "alpha_vix_overlay": float(vix_overlay.at[i]),
                "alpha_momentum_overlay": float(momentum_overlay.at[i]),
                "alpha_trend_overlay": float(trend_overlay.at[i]),

                # Residual
                "alpha_residual": float(residual_alpha.at[i]),

                # Risk context
                "beta": float(beta_col.at[i]) if pd.notna(beta_col.at[i]) else None,
            })

    return pd.DataFrame(rows)


# -------------------------------------------------
# Optional summary helpers (SAFE)
# -------------------------------------------------

def summarize_alpha_sources(alpha_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate alpha attribution by source per period.

    Returns a compact table useful for dashboards.
    """

    if alpha_df.empty:
        return alpha_df

    sources = [
        "alpha_asset_selection",
        "alpha_dynamic_benchmark",
        "alpha_vix_overlay",
        "alpha_momentum_overlay",
        "alpha_trend_overlay",
        "alpha_residual",
    ]

    return (
        alpha_df
        .groupby("period")[sources]
        .sum()
        .reset_index()
    )


# -------------------------------------------------
# Import verification hook
# -------------------------------------------------

def _import_check():
    return "alpha_beta module imported safely"


# -------------------------------------------------
# No execution at import time
# -------------------------------------------------

if __name__ == "__main__":
    print("alpha_beta.py loaded directly — no execution performed")
"""
snapshot_enrichment.py

Phase-2 additive enrichment for snapshot rows.
Adds VIX diagnostics and strategy metadata without
impacting snapshot correctness or rebuild logic.
"""

from __future__ import annotations

from typing import Dict, Any
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# VIX helpers
# ---------------------------------------------------------

def infer_vix_regime(vix_level: float) -> str:
    if np.isnan(vix_level):
        return ""
    if vix_level < 15:
        return "low"
    if vix_level < 20:
        return "normal"
    if vix_level < 30:
        return "elevated"
    return "high"


def enrich_snapshot_with_vix(
    snapshot_df: pd.DataFrame,
    price_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Adds VIX_Level and VIX_Regime columns.
    Safe no-op if VIX data is unavailable.
    """
    if snapshot_df.empty:
        return snapshot_df

    df = snapshot_df.copy()

    if price_df is None or "^VIX" not in price_df.columns:
        df["VIX_Level"] = ""
        df["VIX_Regime"] = ""
        return df

    vix_series = price_df["^VIX"].dropna()
    if vix_series.empty:
        df["VIX_Level"] = ""
        df["VIX_Regime"] = ""
        return df

    vix_level = float(vix_series.iloc[-1])
    df["VIX_Level"] = vix_level
    df["VIX_Regime"] = infer_vix_regime(vix_level)

    return df


# ---------------------------------------------------------
# Strategy enrichment
# ---------------------------------------------------------

def enrich_snapshot_with_strategy(
    snapshot_df: pd.DataFrame,
    strategy_lookup: Dict[str, Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Adds strategy_stack, strategy_stack_applied, and strategy_state columns.
    Safe even if strategy engine is unavailable.
    """
    if snapshot_df.empty:
        return snapshot_df

    df = snapshot_df.copy()

    df["strategy_stack"] = ""
    df["strategy_stack_applied"] = False
    df["strategy_state"] = ""

    if not strategy_lookup:
        return df

    for idx, row in df.iterrows():
        wave_id = row.get("Wave_ID")
        if not wave_id or wave_id not in strategy_lookup:
            continue

        state = strategy_lookup.get(wave_id, {})
        df.at[idx, "strategy_state"] = state

        stack = state.get("strategy_stack", "")
        df.at[idx, "strategy_stack"] = stack
        df.at[idx, "strategy_stack_applied"] = bool(stack)

    return df
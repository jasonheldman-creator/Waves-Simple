"""
snapshot_ledger.py

Canonical snapshot assembly logic.
Responsible for producing the base snapshot rows,
then applying safe, additive enrichment layers.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd

# --- Enrichment imports (Phase-2 additive) ---
from helpers.snapshot_enrichment import (
    enrich_snapshot_with_vix,
    enrich_snapshot_with_strategy,
)


def build_snapshot(
    snapshot_df: pd.DataFrame,
    price_df: Optional[pd.DataFrame] = None,
    strategy_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Build and enrich the snapshot.

    Flow:
    1. Assume snapshot_df already contains core Wave rows
    2. Apply VIX enrichment (safe no-op)
    3. Apply strategy enrichment (safe no-op)
    4. Return enriched snapshot
    """

    if snapshot_df is None or snapshot_df.empty:
        return snapshot_df

    df = snapshot_df.copy()

    # --- Phase 2 enrichment layers (SAFE) ---
    df = enrich_snapshot_with_vix(df, price_df=price_df)
    df = enrich_snapshot_with_strategy(df, strategy_lookup=strategy_lookup)

    return df
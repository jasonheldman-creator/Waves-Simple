"""
snapshot_ledger.py

Canonical snapshot construction layer.

Responsibilities:
- Assemble per-wave snapshot rows
- Merge returns and metadata
- Output snapshot-ready DataFrame

This file DOES NOT:
- Compute VIX logic
- Apply strategies
- Perform attribution

Those are additive layers handled elsewhere.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd


def build_snapshot(
    snapshot_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Core snapshot builder.

    At this stage, this function is intentionally minimal.
    It guarantees snapshot integrity and shape without enrichment.
    """

    if snapshot_df is None or snapshot_df.empty:
        return snapshot_df

    df = snapshot_df.copy()

    # Ensure required columns exist (defensive)
    if "Wave_ID" not in df.columns:
        raise ValueError("Snapshot missing Wave_ID")

    if "Return" not in df.columns:
        df["Return"] = 0.0

    return df
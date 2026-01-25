"""
snapshot_ledger.py

Canonical snapshot assembly logic.

Responsible for:
- Loading the base snapshot
- Applying safe, additive enrichment layers
- Returning a fully populated snapshot for rebuild_snapshot.py

This file is the SINGLE source of truth for snapshot generation.
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Base snapshot loader
# ---------------------------------------------------------

def load_snapshot() -> pd.DataFrame:
    """
    Load the base live snapshot from disk.

    This function does NOT apply enrichment.
    """
    snapshot_path = Path("data/live_snapshot.csv")

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    df = pd.read_csv(snapshot_path)
    return df


# ---------------------------------------------------------
# Snapshot generator (called by rebuild_snapshot.py)
# ---------------------------------------------------------

def generate_snapshot(
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate the canonical snapshot.

    This function:
    - Loads the base snapshot
    - Applies enrichment layers (VIX, strategy)
    - Optionally writes the result to disk
    """

    # 1. Load base snapshot
    df = load_snapshot()

    # 2. Apply enrichment layers (SAFE / NON-FATAL)
    try:
        from helpers.snapshot_enrichment import (
            enrich_snapshot_with_vix,
            enrich_snapshot_with_strategy,
        )

        # VIX enrichment
        try:
            from helpers.price_book import get_price_book
            price_df = get_price_book()
            df = enrich_snapshot_with_vix(df, price_df)
        except Exception:
            # VIX enrichment is optional
            pass

        # Strategy enrichment
        try:
            from helpers.strategy_registry import STRATEGY_LOOKUP
            df = enrich_snapshot_with_strategy(df, STRATEGY_LOOKUP)
        except Exception:
            # Strategy enrichment is optional
            pass

    except Exception:
        # Snapshot must still return even if enrichment fails
        pass

    # 3. Write output if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

__all__ = [
    "load_snapshot",
    "generate_snapshot",
]
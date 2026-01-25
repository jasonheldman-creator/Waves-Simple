"""
snapshot_ledger.py

Canonical snapshot assembly logic.
This file is the SINGLE source of truth for snapshot loading.

CRITICAL:
- live_snapshot.csv DOES NOT contain a header row
- Column names MUST be injected here
- Alpha Attribution depends on this schema
"""

from __future__ import annotations

from typing import Optional
from pathlib import Path
import pandas as pd


# -------------------------------------------------------------------
# Snapshot schema (THIS is what Alpha Attribution depends on)
# -------------------------------------------------------------------

SNAPSHOT_COLUMNS = [
    "Wave_ID",
    "Wave_Name",
    "Asset_Class",
    "Mode",
    "Snapshot_Date",

    "Return_1D",
    "Return_30D",
    "Return_60D",
    "Return_365D",

    "Alpha_1D",
    "Alpha_30D",
    "Alpha_60D",
    "Alpha_365D",

    "Benchmark_Return_1D",
    "Benchmark_Return_30D",
    "Benchmark_Return_60D",
    "Benchmark_Return_365D",

    "VIX_Regime",
    "Exposure",
    "CashPercent",
]


# -------------------------------------------------------------------
# Base snapshot loader (schema enforced)
# -------------------------------------------------------------------

def load_snapshot() -> pd.DataFrame:
    """
    Load the base live snapshot from disk.

    IMPORTANT:
    - live_snapshot.csv has NO HEADER ROW
    - Column names MUST be injected here
    """

    snapshot_path = Path("data/canonical_snapshot.csv")

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    df = pd.read_csv(
        snapshot_path,
        header=None,
        names=SNAPSHOT_COLUMNS,
    )

    return df


# -------------------------------------------------------------------
# Snapshot generator (called by scripts/rebuild_snapshot.py)
# -------------------------------------------------------------------

def generate_snapshot(
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate (or reload) the canonical snapshot.

    NOTE:
    - This function does NOT enrich
    - Enrichment layers are applied elsewhere
    """

    df = load_snapshot()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

__all__ = [
    "load_snapshot",
    "generate_snapshot",
]
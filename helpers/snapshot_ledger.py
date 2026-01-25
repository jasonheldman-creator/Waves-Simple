"""
snapshot_ledger.py

Canonical snapshot assembly logic.

Responsibilities:
- Build the base Wave snapshot
- Compute returns and alpha inputs
- Serve as the single snapshot entrypoint
- Be called by scripts/rebuild_snapshot.py

IMPORTANT:
- NO VIX logic
- NO strategy logic
- NO enrichment logic

All enrichment happens in helpers/snapshot_enrichment.py
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

DATA_DIR = Path("data")
SNAPSHOT_PATH = DATA_DIR / "live_snapshot.csv"


# -------------------------------------------------------------------
# Core snapshot loader
# -------------------------------------------------------------------

def load_snapshot(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the most recent snapshot from disk.

    This function is intentionally simple and stable.
    """
    snapshot_path = path or SNAPSHOT_PATH

    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    df = pd.read_csv(snapshot_path)

    if df.empty:
        raise ValueError("Loaded snapshot is empty")

    return df


# -------------------------------------------------------------------
# Snapshot generator (called by rebuild_snapshot.py)
# -------------------------------------------------------------------

def generate_snapshot(
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate (or reload) the canonical snapshot.

    NOTE:
    - This function does NOT enrich
    - It simply loads and returns the snapshot
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
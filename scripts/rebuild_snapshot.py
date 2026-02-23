#!/usr/bin/env python3
"""
Rebuild snapshot script for GitHub Actions workflow.

This script ensures that a canonical snapshot exists before attempting
to load or enrich it. If data/canonical_snapshot.csv is missing or empty,
it will be generated using the fallback generator.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.snapshot_ledger import generate_snapshot
from helpers.helpers.build_canonical_snapshot import build_canonical_snapshot

SEPARATOR = "=" * 80
CANONICAL_PATH = Path("data/canonical_snapshot.csv")


def ensure_canonical_snapshot() -> None:
    """
    Ensure canonical_snapshot.csv exists and contains rows.
    If missing or empty, generate placeholder rows.
    """
    if not CANONICAL_PATH.exists():
        print("⚠ canonical_snapshot.csv missing - generating fallback snapshot")
        build_canonical_snapshot()
        return

    try:
        df = pd.read_csv(CANONICAL_PATH, header=None)
        if df.empty:
            print("⚠ canonical_snapshot.csv empty - regenerating fallback snapshot")
            build_canonical_snapshot()
    except Exception as e:
        print(f"⚠ Error reading canonical_snapshot.csv ({e}) - regenerating")
        build_canonical_snapshot()


def main() -> int:
    try:
        print("\n" + SEPARATOR)
        print("REBUILD SNAPSHOT WORKFLOW (CANONICAL SAFE)")
        print(SEPARATOR)

        # 🔑 THIS WAS THE MISSING STEP
        ensure_canonical_snapshot()

        # Load + normalize snapshot
        df = generate_snapshot()

        print("\n" + SEPARATOR)
        print("FINAL SUMMARY")
        print(SEPARATOR)
        print(f"✓ Rows loaded: {len(df)}")
        print(f"✓ Columns: {len(df.columns)}")
        print(SEPARATOR + "\n")

        return 0

    except Exception as e:
        print(f"\n✗ Snapshot rebuild failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
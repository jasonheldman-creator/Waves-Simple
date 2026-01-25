#!/usr/bin/env python3
"""
Rebuild snapshot script for GitHub Actions workflow.

This script rebuilds the live snapshot by calling generate_snapshot()
from helpers.snapshot_ledger.

IMPORTANT:
- This file is responsible for WRITING data/live_snapshot.csv
- helpers/snapshot_ledger.py only LOADS / FORMATS the snapshot
"""

import sys
import os
from pathlib import Path

# ------------------------------------------------------------------
# Ensure project root is on PYTHONPATH
# ------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------------
# CORRECT IMPORT (THIS WAS THE BUG)
# ------------------------------------------------------------------

from helpers.snapshot_ledger import generate_snapshot

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

SEPARATOR = "=" * 80
OUTPUT_PATH = Path("data/live_snapshot.csv")

REQUIRED_COLUMNS = [
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

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    try:
        print("\n" + SEPARATOR)
        print("REBUILD SNAPSHOT WORKFLOW")
        print(SEPARATOR)

        # Generate snapshot and WRITE CSV
        df = generate_snapshot(output_path=OUTPUT_PATH)

        print("\n" + SEPARATOR)
        print("FINAL SNAPSHOT SUMMARY")
        print(SEPARATOR)

        print(f"✓ Rows written: {len(df)}")
        print(f"✓ Columns written: {len(df.columns)}")

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"⚠ Missing columns: {missing}")
        else:
            print("✓ All required columns present")

        print(SEPARATOR + "\n")
        return 0

    except Exception as e:
        print("\n✗ SNAPSHOT REBUILD FAILED")
        print(str(e))
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
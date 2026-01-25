#!/usr/bin/env python3
"""
Rebuild snapshot script for GitHub Actions workflow.

This script rebuilds the live snapshot by loading the canonical
live_snapshot.csv via snapshot_ledger and re-emitting it with
a guaranteed schema.

IMPORTANT:
- This script DOES NOT compute returns or alpha
- It validates and rewrites the snapshot so the app can consume it
"""

import sys
import os
from pathlib import Path

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snapshot_ledger import generate_snapshot

SEPARATOR = "=" * 80

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


def main() -> int:
    try:
        print("\n" + SEPARATOR)
        print("REBUILD SNAPSHOT WORKFLOW — CANONICAL SCHEMA ENFORCEMENT")
        print(SEPARATOR)

        output_path = Path("data/live_snapshot.csv")

        df = generate_snapshot(output_path=output_path)

        print("\nFINAL SUMMARY")
        print(SEPARATOR)
        print(f"✓ Row count: {len(df)}")
        print(f"✓ Column count: {len(df.columns)}")

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"✗ Missing columns: {missing}")
        else:
            print("✓ All required columns present")

        print(SEPARATOR + "\n")
        return 0

    except Exception as e:
        print(f"\n✗ Snapshot rebuild failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
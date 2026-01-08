#!/usr/bin/env python3
"""
Rebuild snapshot script for GitHub Actions workflow.

This script rebuilds the live snapshot by calling generate_live_snapshot_csv()
from analytics_truth, then WRITES the artifact to data/live_snapshot.csv.

The workflow should only be considered successful if the CSV file is written.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics_truth import generate_live_snapshot_csv


def main() -> int:
    try:
        print("\n" + "=" * 80)
        print("REBUILD SNAPSHOT WORKFLOW")
        print("=" * 80)

        # Generate snapshot dataframe
        print("\nRebuilding live snapshot (generating DataFrame)...")
        df = generate_live_snapshot_csv()

        # Write artifact to the exact file the app/workflow expects
        out_path = Path("data") / "live_snapshot.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(out_path, index=False)

        # Hard proof in logs
        print(f"\n✓ Wrote snapshot: {out_path.resolve()}")
        print(f"✓ Rows: {len(df)} | Cols: {len(df.columns)}")
        print("\nFirst 5 lines of CSV:")
        with out_path.open("r", encoding="utf-8") as f:
            for i in range(5):
                line = f.readline()
                if not line:
                    break
                print(line.rstrip("\n"))

        # Validate file exists and is non-empty
        size = out_path.stat().st_size
        if size <= 0:
            raise AssertionError(f"Snapshot file is empty: {out_path} (size={size})")

        print("\n" + "=" * 80)
        return 0

    except Exception as e:
        print(f"\n✗ Error rebuilding snapshot: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
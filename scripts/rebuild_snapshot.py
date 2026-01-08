#!/usr/bin/env python3
"""
Rebuild snapshot script for GitHub Actions workflow.

This script rebuilds the live snapshot by calling generate_live_snapshot_csv()
from analytics_truth. The canonical generator writes the artifact to data/live_snapshot.csv.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics_truth import generate_live_snapshot_csv


def main() -> int:
    try:
        print("\n" + "=" * 80)
        print("REBUILD SNAPSHOT WORKFLOW")
        print("=" * 80)

        # Call canonical snapshot generator
        # This function writes to data/live_snapshot.csv and prints diagnostics
        df = generate_live_snapshot_csv()

        # Print essential diagnostics from returned DataFrame
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"✓ Row count: {len(df)}")
        print(f"✓ Column count: {len(df.columns)}")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        print(f"\n✗ Error rebuilding snapshot: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
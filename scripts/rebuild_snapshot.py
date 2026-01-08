#!/usr/bin/env python3
"""
Rebuild snapshot script for GitHub Actions workflow.

This script rebuilds the live snapshot by calling generate_live_snapshot_csv()
from analytics_truth with explicit paths, then validates the output.

The workflow should only be considered successful if the CSV file is written
and passes all validation checks.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics_truth import generate_live_snapshot_csv


def main() -> None:
    try:
        print("\n" + "=" * 80)
        print("REBUILD SNAPSHOT WORKFLOW")
        print("=" * 80)

        # Generate snapshot dataframe with explicit paths
        print("\nRebuilding live snapshot (generating DataFrame)...")
        df = generate_live_snapshot_csv(
            out_path="data/live_snapshot.csv",
            weights_path="wave_weights.csv"
        )

        # Validate file exists
        out_path = Path("data/live_snapshot.csv")
        if not out_path.exists():
            raise AssertionError(f"Snapshot file does not exist: {out_path}")

        # Validate file size
        size = out_path.stat().st_size
        if size <= 0:
            raise AssertionError(f"Snapshot file is empty: {out_path} (size={size})")

        # Print diagnostic information
        print("\n" + "=" * 80)
        print("VALIDATION DIAGNOSTICS")
        print("=" * 80)
        print(f"\n✓ File exists: {out_path.resolve()}")
        print(f"✓ File size: {size} bytes")
        print(f"✓ rows: {len(df)}")

        # Print status counts if status column exists
        if 'status' in df.columns:
            ok_count = (df['status'] == 'OK').sum()
            no_data_count = (df['status'] == 'NO DATA').sum()
            print(f"\n📊 Status Counts:")
            print(f"  status == \"OK\":      {ok_count}")
            print(f"  status == \"NO DATA\": {no_data_count}")
        
        # Print first line (CSV header) to confirm schema
        print("\n📄 CSV Header (first line):")
        with out_path.open("r", encoding="utf-8") as f:
            first_line = f.readline().rstrip("\n")
            print(f"  {first_line}")

        print("\n" + "=" * 80)
        print("✓ SNAPSHOT REBUILD SUCCESSFUL")
        print("=" * 80 + "\n")
        
        sys.exit(0)

    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ SNAPSHOT REBUILD FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
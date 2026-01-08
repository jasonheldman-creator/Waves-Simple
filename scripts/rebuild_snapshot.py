#!/usr/bin/env python3
"""
Rebuild snapshot script for GitHub Actions workflow.

This script rebuilds the live snapshot by calling the generate_live_snapshot_csv
function from analytics_truth. The generated snapshot and any cache artifacts
in data/cache/ are ignored by .gitignore and will not be committed.

The workflow succeeds based on successful execution of this script only.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from analytics_truth import generate_live_snapshot_csv


def main():
    """
    Main function to rebuild the live snapshot.
    
    This function:
    1. Calls generate_live_snapshot_csv() to rebuild the snapshot
    2. Validates that the snapshot was created successfully
    3. Returns exit code 0 on success, 1 on failure
    
    Any cache artifacts generated in data/cache/ will be ignored by .gitignore.
    """
    try:
        print("\n" + "=" * 80)
        print("REBUILD SNAPSHOT WORKFLOW")
        print("=" * 80)
        
        # Generate the live snapshot
        print("\nRebuilding live snapshot...")
        df = generate_live_snapshot_csv()
        
        # Validate the result
        if df is not None and len(df) > 0:
            print(f"\n✓ Successfully rebuilt snapshot with {len(df)} rows")
            print("\nNote: Generated snapshot and cache artifacts in data/cache/")
            print("      are ignored by .gitignore and will not be committed.")
            print("\n" + "=" * 80)
            return 0
        else:
            print("\n✗ Failed to rebuild snapshot: Empty or None result")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error rebuilding snapshot: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

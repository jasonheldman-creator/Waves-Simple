#!/usr/bin/env python3
"""
Rebuild snapshot script for GitHub Actions workflow.

This script rebuilds the live snapshot by calling generate_snapshot()
from snapshot_ledger. The snapshot generator applies the full strategy pipeline
including momentum calculations, VIX overlay adjustments, exposure adjustments,
and computes alpha metrics against the benchmark.

The output is written to data/live_snapshot.csv with all required fields.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snapshot_ledger import generate_snapshot

# Formatting constant
SEPARATOR = "=" * 80


def main() -> int:
    try:
        print("\n" + SEPARATOR)
        print("REBUILD SNAPSHOT WORKFLOW - STRATEGY-AWARE")
        print(SEPARATOR)

        # Call snapshot ledger generator with full strategy pipeline
        # This function:
        # - Applies complete strategy pipeline (momentum, VIX overlay, exposure adjustments)
        # - Computes returns: return_1d, return_30d, return_60d, return_365d
        # - Computes alpha metrics against benchmark: alpha_1d, alpha_30d, alpha_60d, alpha_365d
        # - Populates VIX regime, exposure, and cash fields
        # - Writes to data/live_snapshot.csv
        df = generate_snapshot(force_refresh=True, generation_reason='rebuild_workflow')

        # Print essential diagnostics from returned DataFrame
        print("\n" + SEPARATOR)
        print("FINAL SUMMARY")
        print(SEPARATOR)
        print(f"✓ Row count: {len(df)}")
        print(f"✓ Column count: {len(df.columns)}")
        
        # Validate required columns are present
        required_cols = ['Return_1D', 'Return_30D', 'Return_60D', 'Return_365D',
                        'Alpha_1D', 'Alpha_30D', 'Alpha_60D', 'Alpha_365D',
                        'VIX_Regime', 'Exposure', 'CashPercent']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠ Warning: Missing columns: {', '.join(missing_cols)}")
        else:
            print(f"✓ All required columns present")
        
        # Show sample data for clean_transit_infrastructure_wave if present
        if 'Wave_ID' in df.columns:
            transit_wave = df[df['Wave_ID'] == 'clean_transit_infrastructure_wave']
            if not transit_wave.empty:
                print(f"\n✓ Sample: clean_transit_infrastructure_wave")
                row = transit_wave.iloc[0]
                print(f"  Return_1D: {row.get('Return_1D', 'N/A')}")
                print(f"  Alpha_1D: {row.get('Alpha_1D', 'N/A')}")
                print(f"  VIX_Regime: {row.get('VIX_Regime', 'N/A')}")
                print(f"  Exposure: {row.get('Exposure', 'N/A')}")
        
        print(SEPARATOR + "\n")

        return 0

    except Exception as e:
        print(f"\n✗ Error rebuilding snapshot: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
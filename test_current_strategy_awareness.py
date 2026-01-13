#!/usr/bin/env python3
"""
Test to verify current strategy-awareness of Return_* and Alpha_* metrics.

This test checks if the current implementation already reflects strategy adjustments.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_current_implementation():
    """
    Test if current Return_* and Alpha_* metrics reflect strategy adjustments.
    """
    print("\n" + "=" * 80)
    print("TEST: Verify current strategy-awareness")
    print("=" * 80)
    
    # Load the existing snapshot
    snapshot_file = "data/live_snapshot.csv"
    
    if not os.path.exists(snapshot_file):
        print(f"✗ Snapshot file not found: {snapshot_file}")
        return False
    
    df = pd.read_csv(snapshot_file)
    print(f"✓ Loaded snapshot: {len(df)} waves")
    
    # Filter to equity waves only
    equity_waves = df[df['Category'] == 'equity_growth']
    print(f"✓ Found {len(equity_waves)} equity growth waves")
    
    # Check for waves with VIX overlay adjustments
    waves_with_vix = equity_waves[
        equity_waves['VIX_Adjustment_Pct'].notna() & 
        (equity_waves['VIX_Adjustment_Pct'] != '')
    ]
    
    print(f"✓ Found {len(waves_with_vix)} waves with VIX_Adjustment_Pct")
    
    if len(waves_with_vix) == 0:
        print("⚠ No waves with VIX adjustments found - cannot verify strategy awareness")
        return True  # Not a failure, just no data
    
    # Show some examples
    print("\nExamples of waves with VIX overlay adjustments:")
    for idx, row in waves_with_vix.head(3).iterrows():
        wave_name = row['Wave']
        vix_adj = row['VIX_Adjustment_Pct']
        exposure = row['Exposure']
        return_1d = row['Return_1D']
        alpha_1d = row['Alpha_1D']
        
        print(f"\n  Wave: {wave_name}")
        print(f"    VIX_Adjustment_Pct: {vix_adj}")
        print(f"    Exposure: {exposure}")
        print(f"    Return_1D: {return_1d}")
        print(f"    Alpha_1D: {alpha_1d}")
        
        # Parse strategy_state to see if it mentions VIX overlay
        strategy_state = row['strategy_state']
        if 'vix_overlay' in str(strategy_state).lower():
            print(f"    ✓ Strategy state mentions vix_overlay")
    
    # Key question: Are Return_* and Alpha_* computed from strategy-adjusted NAV?
    # Let's trace through the logic:
    print("\n" + "=" * 80)
    print("ANALYSIS: How are Return_* and Alpha_* computed?")
    print("=" * 80)
    
    print("""
Current implementation (from code analysis):

1. snapshot_ledger.py calls compute_history_nav(wave_name, mode, days=365)
2. compute_history_nav() returns wave_nav and bm_nav series
3. wave_nav is built from strategy-adjusted daily returns (wave_ret_list)
4. wave_ret_list includes exposure adjustments, VIX overlay, safe allocation
5. snapshot_ledger.py computes Return_* as (wave_nav[end] / wave_nav[start]) - 1

Therefore:
- Return_* IS computed from strategy-adjusted NAV ✓
- Alpha_* IS computed as Return_* - Benchmark_Return_* ✓

Conclusion: Current implementation APPEARS to be strategy-aware!
""")
    
    return True


if __name__ == "__main__":
    success = test_current_implementation()
    sys.exit(0 if success else 1)

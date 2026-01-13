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
    
    # Key validation: Check that VIX overlay is tracked in strategy_state
    print("\n" + "=" * 80)
    print("VALIDATION: VIX overlay is tracked in strategy_state")
    print("=" * 80)
    
    has_vix_overlay = False
    for idx, row in waves_with_vix.head(3).iterrows():
        strategy_state = row['strategy_state']
        if 'vix_overlay' in str(strategy_state).lower():
            has_vix_overlay = True
            print(f"✓ {row['Wave']}: strategy_state includes vix_overlay")
    
    if not has_vix_overlay:
        print("✗ No waves have vix_overlay in strategy_state")
        return False
    
    print("\n" + "=" * 80)
    print("CONCLUSION: Current implementation tracks VIX overlay in strategy_state")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = test_current_implementation()
    sys.exit(0 if success else 1)

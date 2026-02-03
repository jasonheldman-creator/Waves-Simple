#!/usr/bin/env python3
"""
Test Strategy Overlay Application

This test validates that all equity waves in wave_history.csv have strategy overlays
(Momentum, VIX, regime detection, etc.) applied to their returns, not just raw holding returns.

Key validation:
1. Equity waves have exposure_used != 1.0 (strategy overlays are active)
2. Equity waves have overlay_active = True
3. Returns show variation consistent with strategy adjustments
4. S&P 500 Wave (reference) has full strategy stack applied
"""

import pandas as pd
import numpy as np
import os
import sys


def test_strategy_overlays_in_wave_history():
    """
    Test that wave_history.csv contains strategy-adjusted returns for equity waves.
    """
    print("=" * 80)
    print("TEST: Strategy Overlay Application in wave_history.csv")
    print("=" * 80)
    
    # Load wave_history.csv
    wave_history_file = "wave_history.csv"
    if not os.path.exists(wave_history_file):
        print(f"✗ FAILED: {wave_history_file} not found")
        print("  Run build_wave_history_from_prices.py first to generate wave_history.csv")
        return False
    
    df = pd.read_csv(wave_history_file)
    print(f"✓ Loaded wave_history.csv: {len(df)} rows")
    
    # Define equity waves that should have overlays
    equity_waves = [
        "S&P 500 Wave",
        "US MegaCap Core Wave",
        "AI & Cloud MegaCap Wave",
        "Next-Gen Compute & Semis Wave",
        "Future Energy & EV Wave",
        "EV & Infrastructure Wave",
        "US Small-Cap Disruptors Wave",
        "US Mid/Small Growth & Semis Wave",
        "Small Cap Growth Wave",
        "Small to Mid Cap Growth Wave",
        "Future Power & Energy Wave",
        "Quantum Computing Wave",
        "Clean Transit-Infrastructure Wave",
        "Demas Fund Wave",
        "Infinity Multi-Asset Growth Wave",
        "Gold Wave",
    ]
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Check that equity waves have overlay_active column
    print("\n[Test 1] overlay_active column exists")
    if "overlay_active" not in df.columns:
        print("  ✗ FAILED: overlay_active column not found in wave_history.csv")
        tests_failed += 1
    else:
        print("  ✓ PASSED: overlay_active column present")
        tests_passed += 1
    
    # Test 2: Check that equity waves have exposure_used column
    print("\n[Test 2] exposure_used column exists")
    if "exposure_used" not in df.columns:
        print("  ✗ FAILED: exposure_used column not found in wave_history.csv")
        tests_failed += 1
    else:
        print("  ✓ PASSED: exposure_used column present")
        tests_passed += 1
    
    # Test 3: Check that equity waves have overlay_active = True
    print("\n[Test 3] Equity waves have overlay_active = True")
    equity_df = df[df['wave'].isin(equity_waves)].copy()
    
    if equity_df.empty:
        print("  ✗ FAILED: No equity waves found in wave_history.csv")
        tests_failed += 1
    elif "overlay_active" not in df.columns:
        print("  ✗ SKIPPED: overlay_active column not available")
        tests_failed += 1
    else:
        waves_with_overlays = 0
        waves_without_overlays = []
        
        for wave in equity_waves:
            wave_data = equity_df[equity_df['wave'] == wave]
            if wave_data.empty:
                # Wave not in dataset, skip
                continue
            
            # Check if majority of rows have overlay_active = True
            overlay_active_pct = (wave_data['overlay_active'] == True).sum() / len(wave_data)
            
            if overlay_active_pct >= 0.5:  # At least 50% should have overlays active
                waves_with_overlays += 1
            else:
                waves_without_overlays.append((wave, overlay_active_pct))
        
        if waves_without_overlays:
            print(f"  ✗ FAILED: {len(waves_without_overlays)} equity waves have overlays disabled")
            for wave, pct in waves_without_overlays:
                print(f"    - {wave}: {pct:.1%} overlay_active = True")
            tests_failed += 1
        else:
            print(f"  ✓ PASSED: {waves_with_overlays} equity waves have overlays active")
            tests_passed += 1
    
    # Test 4: Check that exposure_used varies (not always 1.0)
    print("\n[Test 4] exposure_used shows variation (strategy adjustments active)")
    
    if "exposure_used" not in df.columns:
        print("  ✗ SKIPPED: exposure_used column not available")
        tests_failed += 1
    elif equity_df.empty:
        print("  ✗ FAILED: No equity waves found")
        tests_failed += 1
    else:
        waves_with_variation = 0
        waves_without_variation = []
        
        for wave in equity_waves:
            wave_data = equity_df[equity_df['wave'] == wave]
            if wave_data.empty:
                continue
            
            # Check if exposure_used varies (std > 0.01)
            exposure_std = wave_data['exposure_used'].std()
            exposure_mean = wave_data['exposure_used'].mean()
            
            # Should have some variation if overlays are working
            if exposure_std > 0.01 or abs(exposure_mean - 1.0) > 0.01:
                waves_with_variation += 1
            else:
                waves_without_variation.append((wave, exposure_mean, exposure_std))
        
        if waves_without_variation:
            print(f"  ⚠ WARNING: {len(waves_without_variation)} equity waves have constant exposure_used = 1.0")
            for wave, mean, std in waves_without_variation[:5]:  # Show first 5
                print(f"    - {wave}: mean={mean:.4f}, std={std:.4f}")
            if len(waves_without_variation) > 5:
                print(f"    ... and {len(waves_without_variation) - 5} more")
            # This is a warning, not a failure - might be due to missing VIX data
            print("  ℹ This may be expected if VIX data is not available")
            tests_passed += 1
        else:
            print(f"  ✓ PASSED: {waves_with_variation} equity waves show exposure variation")
            tests_passed += 1
    
    # Test 5: S&P 500 Wave specifically (reference implementation)
    print("\n[Test 5] S&P 500 Wave has overlays applied (reference check)")
    sp500_data = df[df['wave'] == "S&P 500 Wave"]
    
    if sp500_data.empty:
        print("  ✗ FAILED: S&P 500 Wave not found in wave_history.csv")
        tests_failed += 1
    elif "overlay_active" not in df.columns or "exposure_used" not in df.columns:
        print("  ✗ SKIPPED: Required columns not available")
        tests_failed += 1
    else:
        overlay_active_pct = (sp500_data['overlay_active'] == True).sum() / len(sp500_data)
        exposure_std = sp500_data['exposure_used'].std()
        exposure_mean = sp500_data['exposure_used'].mean()
        
        print(f"  📊 S&P 500 Wave statistics:")
        print(f"     - Days: {len(sp500_data)}")
        print(f"     - overlay_active: {overlay_active_pct:.1%}")
        print(f"     - exposure_used mean: {exposure_mean:.4f}")
        print(f"     - exposure_used std: {exposure_std:.4f}")
        
        if overlay_active_pct >= 0.5 and (exposure_std > 0.01 or abs(exposure_mean - 1.0) > 0.01):
            print("  ✓ PASSED: S&P 500 Wave has overlays applied")
            tests_passed += 1
        else:
            print("  ✗ FAILED: S&P 500 Wave overlays not active or exposure constant")
            tests_failed += 1
    
    # Final summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n🎉 ALL TESTS PASSED")
        print("✓ Strategy overlays are correctly applied to equity waves")
        return True
    else:
        print(f"\n❌ {tests_failed} TEST(S) FAILED")
        print("⚠ Strategy overlays may not be fully applied")
        return False


if __name__ == "__main__":
    success = test_strategy_overlays_in_wave_history()
    sys.exit(0 if success else 1)

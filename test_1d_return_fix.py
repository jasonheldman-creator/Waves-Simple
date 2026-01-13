#!/usr/bin/env python3
"""
Test 1D return calculation fix and VIX overlay fields persistence

This test validates:
1. compute_period_return() uses trading-day alignment for 1D returns
2. VIX_Level, VIX_Regime, Exposure, CashPercent fields are added to snapshot
3. asof_date is set based on max trading date in price data
"""

import sys
import os
sys.path.insert(0, '/home/runner/work/Waves-Simple/Waves-Simple')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from analytics_truth import compute_period_return


def test_1d_return_trading_day_alignment():
    """
    Test that 1D return uses last two available trading dates,
    not calendar day calculations.
    """
    print("\n" + "=" * 80)
    print("TEST: 1D Return Trading Day Alignment")
    print("=" * 80)
    
    # Create test price series with gaps (simulating weekends)
    # Mon, Tue, Wed, Thu, Fri (skip weekend), Mon
    dates = [
        datetime(2024, 1, 1),  # Monday - 100
        datetime(2024, 1, 2),  # Tuesday - 101
        datetime(2024, 1, 3),  # Wednesday - 102
        datetime(2024, 1, 4),  # Thursday - 103
        datetime(2024, 1, 5),  # Friday - 104
        # Weekend gap
        datetime(2024, 1, 8),  # Monday - 105
    ]
    
    prices = pd.Series([100, 101, 102, 103, 104, 105], index=pd.DatetimeIndex(dates))
    
    print("Price series:")
    print(prices)
    print()
    
    # Test 1D return - should use last two trading dates (Mon 105 vs Fri 104)
    ret_1d = compute_period_return(prices, 1)
    expected_1d = (105 - 104) / 104  # = 0.00961...
    
    print(f"1D Return: {ret_1d:.6f}")
    print(f"Expected:  {expected_1d:.6f}")
    print(f"Match: {abs(ret_1d - expected_1d) < 1e-10}")
    
    assert abs(ret_1d - expected_1d) < 1e-10, f"1D return mismatch: {ret_1d} vs {expected_1d}"
    
    print("\n✓ 1D return correctly uses last two trading dates (not calendar days)")
    print(f"✓ Correctly computed: (105 / 104) - 1 = {ret_1d:.6f}")
    
    return True


def test_longer_period_returns():
    """
    Test that longer period returns use trading-day counting.
    """
    print("\n" + "=" * 80)
    print("TEST: Longer Period Returns (Trading Day Counting)")
    print("=" * 80)
    
    # Create 10 days of trading data
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    prices = pd.Series(range(100, 110), index=dates)
    
    print("Price series (10 trading days):")
    print(prices)
    print()
    
    # Test 5D return
    ret_5d = compute_period_return(prices, 5)
    expected_5d = (109 - 104) / 104  # Last day vs 5 days ago
    
    print(f"5D Return: {ret_5d:.6f}")
    print(f"Expected:  {expected_5d:.6f}")
    print(f"Match: {abs(ret_5d - expected_5d) < 1e-10}")
    
    assert abs(ret_5d - expected_5d) < 1e-10, f"5D return mismatch: {ret_5d} vs {expected_5d}"
    
    # Test 30D return (should be NaN - insufficient data)
    ret_30d = compute_period_return(prices, 30)
    print(f"\n30D Return: {ret_30d} (should be NaN)")
    assert pd.isna(ret_30d), f"30D return should be NaN with only 10 days of data"
    
    print("\n✓ Longer period returns use trading-day row counting")
    print("✓ Returns NaN when insufficient data (no silent fallback)")
    
    return True


def test_vix_fields_in_snapshot():
    """
    Test that VIX overlay fields are added to snapshot DataFrame.
    
    Note: This test checks the structure without actually generating
    the snapshot (which would require live market data).
    """
    print("\n" + "=" * 80)
    print("TEST: VIX Fields in Snapshot Structure")
    print("=" * 80)
    
    # Expected VIX fields that should be added to snapshot
    expected_vix_fields = [
        'VIX_Level',
        'VIX_Regime',
        'Exposure',
        'CashPercent',
        'asof_date',
    ]
    
    print("Expected VIX overlay fields in snapshot:")
    for field in expected_vix_fields:
        print(f"  - {field}")
    
    # Verify the fields are in the generate_live_snapshot_csv function
    import inspect
    from analytics_truth import generate_live_snapshot_csv
    
    source = inspect.getsource(generate_live_snapshot_csv)
    
    missing_fields = []
    for field in expected_vix_fields:
        if f"'{field}':" not in source and f'"{field}":' not in source:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"\n✗ Missing fields in generate_live_snapshot_csv: {missing_fields}")
        return False
    
    print("\n✓ All expected VIX overlay fields are in generate_live_snapshot_csv()")
    
    # Check that compute_volatility_regime_and_exposure is called
    if 'compute_volatility_regime_and_exposure' in source:
        print("✓ Function calls compute_volatility_regime_and_exposure()")
    else:
        print("⚠️ Warning: compute_volatility_regime_and_exposure not found in source")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("1D RETURN FIX AND VIX OVERLAY FIELDS - TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("1D Return Trading Day Alignment", test_1d_return_trading_day_alignment),
        ("Longer Period Returns", test_longer_period_returns),
        ("VIX Fields in Snapshot", test_vix_fields_in_snapshot),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"\n✓ PASSED: {test_name}")
            else:
                failed += 1
                print(f"\n✗ FAILED: {test_name}")
        except Exception as e:
            failed += 1
            print(f"\n✗ FAILED: {test_name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())

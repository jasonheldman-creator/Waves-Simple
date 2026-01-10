"""
Unit tests for period_returns helper module.

This test module validates the canonical period return calculation logic
using synthetic data with known, predictable returns.

Tests cover:
1. Basic return computation with exact trading days
2. 30D, 60D, and 252D (365D) trading day windows
3. Insufficient data handling
4. Edge cases (NaN, zero prices, empty series)
5. Benchmark alignment for alpha computation
6. Alpha calculation with aligned series
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module we're testing
from helpers.period_returns import (
    compute_period_return,
    align_series_for_alpha,
    compute_alpha,
    TRADING_DAYS_MAP
)


def create_synthetic_prices(
    start_date: str,
    num_days: int,
    daily_return: float = 0.01,
    start_price: float = 100.0
) -> pd.Series:
    """
    Create synthetic price series with known daily return.
    
    Args:
        start_date: Starting date (YYYY-MM-DD)
        num_days: Number of trading days
        daily_return: Daily return (decimal, e.g., 0.01 for 1%)
        start_price: Starting price
    
    Returns:
        Series with DatetimeIndex and synthetic prices
    """
    dates = pd.date_range(start=start_date, periods=num_days, freq='B')  # 'B' = business days
    prices = [start_price]
    
    for i in range(1, num_days):
        prices.append(prices[-1] * (1 + daily_return))
    
    return pd.Series(prices, index=dates)


def test_basic_return_computation():
    """Test basic return computation with simple known values."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Return Computation")
    print("=" * 70)
    
    # Create simple price series: 100 -> 105 over 5 days
    prices = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 105.0],
        index=pd.date_range('2024-01-01', periods=5, freq='B')
    )
    
    # Test 1-day return (last 2 points)
    ret_1d = compute_period_return(prices, 1)
    expected_1d = (105.0 / 103.0) - 1.0  # ~0.0194
    print(f"  1D return: {ret_1d:.6f} (expected: {expected_1d:.6f})")
    assert abs(ret_1d - expected_1d) < 1e-6, f"1D return mismatch: {ret_1d} vs {expected_1d}"
    
    # Test 3-day return (last 4 points)
    ret_3d = compute_period_return(prices, 3)
    expected_3d = (105.0 / 101.0) - 1.0  # ~0.0396
    print(f"  3D return: {ret_3d:.6f} (expected: {expected_3d:.6f})")
    assert abs(ret_3d - expected_3d) < 1e-6, f"3D return mismatch: {ret_3d} vs {expected_3d}"
    
    # Test 4-day return (all 5 points)
    ret_4d = compute_period_return(prices, 4)
    expected_4d = (105.0 / 100.0) - 1.0  # 0.05
    print(f"  4D return: {ret_4d:.6f} (expected: {expected_4d:.6f})")
    assert abs(ret_4d - expected_4d) < 1e-6, f"4D return mismatch: {ret_4d} vs {expected_4d}"
    
    print("  ✓ Basic return computation PASSED")


def test_30_day_trading_window():
    """Test 30 trading day return calculation."""
    print("\n" + "=" * 70)
    print("TEST 2: 30 Trading Day Return")
    print("=" * 70)
    
    # Create 35 days of data with 1% daily return
    # This gives us enough data for a 30-day window
    daily_return = 0.01  # 1% per day
    prices = create_synthetic_prices('2024-01-01', 35, daily_return=daily_return)
    
    # Compute 30-day return
    ret_30d = compute_period_return(prices, 30)
    
    # Expected return: (1.01)^30 - 1 ≈ 0.3478
    expected_30d = (1 + daily_return) ** 30 - 1.0
    
    print(f"  30D return: {ret_30d:.6f}")
    print(f"  Expected:   {expected_30d:.6f}")
    print(f"  Difference: {abs(ret_30d - expected_30d):.8f}")
    
    # Should be very close (within floating point precision)
    assert abs(ret_30d - expected_30d) < 1e-6, f"30D return mismatch: {ret_30d} vs {expected_30d}"
    
    print("  ✓ 30 trading day return PASSED")


def test_60_day_trading_window():
    """Test 60 trading day return calculation."""
    print("\n" + "=" * 70)
    print("TEST 3: 60 Trading Day Return")
    print("=" * 70)
    
    # Create 65 days of data with 0.5% daily return
    daily_return = 0.005  # 0.5% per day
    prices = create_synthetic_prices('2024-01-01', 65, daily_return=daily_return)
    
    # Compute 60-day return
    ret_60d = compute_period_return(prices, 60)
    
    # Expected return: (1.005)^60 - 1 ≈ 0.3494
    expected_60d = (1 + daily_return) ** 60 - 1.0
    
    print(f"  60D return: {ret_60d:.6f}")
    print(f"  Expected:   {expected_60d:.6f}")
    print(f"  Difference: {abs(ret_60d - expected_60d):.8f}")
    
    assert abs(ret_60d - expected_60d) < 1e-6, f"60D return mismatch: {ret_60d} vs {expected_60d}"
    
    print("  ✓ 60 trading day return PASSED")


def test_252_day_trading_window():
    """Test 252 trading day (1 year) return calculation."""
    print("\n" + "=" * 70)
    print("TEST 4: 252 Trading Day Return (365D)")
    print("=" * 70)
    
    # Create 260 days of data with 0.05% daily return
    daily_return = 0.0005  # 0.05% per day
    prices = create_synthetic_prices('2024-01-01', 260, daily_return=daily_return)
    
    # Compute 252-day return (standard 1-year trading days)
    ret_252d = compute_period_return(prices, 252)
    
    # Expected return: (1.0005)^252 - 1 ≈ 0.1338
    expected_252d = (1 + daily_return) ** 252 - 1.0
    
    print(f"  252D return: {ret_252d:.6f}")
    print(f"  Expected:    {expected_252d:.6f}")
    print(f"  Difference:  {abs(ret_252d - expected_252d):.8f}")
    
    assert abs(ret_252d - expected_252d) < 1e-6, f"252D return mismatch: {ret_252d} vs {expected_252d}"
    
    print("  ✓ 252 trading day return PASSED")


def test_insufficient_data():
    """Test behavior with insufficient data."""
    print("\n" + "=" * 70)
    print("TEST 5: Insufficient Data Handling")
    print("=" * 70)
    
    # Create only 10 days of data
    prices = create_synthetic_prices('2024-01-01', 10)
    
    # Try to compute 30-day return (need 31 points, have only 10)
    ret = compute_period_return(prices, 30)
    
    print(f"  10 days of data, requesting 30-day return")
    print(f"  Result: {ret}")
    print(f"  Expected: 0.0 (insufficient data)")
    
    assert ret == 0.0, f"Should return 0.0 for insufficient data, got {ret}"
    
    # Test with return_none_on_insufficient_data=True
    ret_none = compute_period_return(prices, 30, return_none_on_insufficient_data=True)
    print(f"  With return_none_on_insufficient_data=True: {ret_none}")
    
    assert ret_none is None, f"Should return None for insufficient data, got {ret_none}"
    
    print("  ✓ Insufficient data handling PASSED")


def test_edge_cases():
    """Test edge cases: NaN, zero prices, empty series."""
    print("\n" + "=" * 70)
    print("TEST 6: Edge Cases")
    print("=" * 70)
    
    # Test empty series
    empty_series = pd.Series([], dtype=float)
    ret_empty = compute_period_return(empty_series, 30)
    print(f"  Empty series: {ret_empty} (expected: 0.0)")
    assert ret_empty == 0.0, "Empty series should return 0.0"
    
    # Test series with NaN values
    prices_with_nan = pd.Series(
        [100.0, np.nan, 105.0, np.nan, 110.0],
        index=pd.date_range('2024-01-01', periods=5, freq='B')
    )
    ret_nan = compute_period_return(prices_with_nan, 1)
    # After dropping NaN, we have [100, 105, 110], so 1-day return is (110/105)-1
    expected_nan = (110.0 / 105.0) - 1.0
    print(f"  Series with NaN: {ret_nan:.6f} (expected: {expected_nan:.6f})")
    assert abs(ret_nan - expected_nan) < 1e-6, "Should drop NaN and compute correctly"
    
    # Test series with zero start price
    prices_with_zero = pd.Series(
        [0.0, 100.0, 105.0],
        index=pd.date_range('2024-01-01', periods=3, freq='B')
    )
    ret_zero = compute_period_return(prices_with_zero, 2)
    print(f"  Series with zero start: {ret_zero} (expected: 0.0)")
    assert ret_zero == 0.0, "Zero start price should return 0.0"
    
    # Test None input
    ret_none = compute_period_return(None, 30)
    print(f"  None input: {ret_none} (expected: 0.0)")
    assert ret_none == 0.0, "None input should return 0.0"
    
    print("  ✓ Edge cases PASSED")


def test_benchmark_alignment():
    """Test alignment of wave and benchmark series."""
    print("\n" + "=" * 70)
    print("TEST 7: Benchmark Alignment")
    print("=" * 70)
    
    # Create wave prices (10 days)
    wave_prices = pd.Series(
        [100.0, 102.0, 105.0, 107.0, 110.0, 112.0, 115.0, 118.0, 120.0, 122.0],
        index=pd.date_range('2024-01-01', periods=10, freq='B')
    )
    
    # Create benchmark prices (12 days - starts earlier and ends later)
    benchmark_prices = pd.Series(
        [98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        index=pd.date_range('2023-12-28', periods=12, freq='B')
    )
    
    print(f"  Wave dates: {wave_prices.index[0].date()} to {wave_prices.index[-1].date()}")
    print(f"  Benchmark dates: {benchmark_prices.index[0].date()} to {benchmark_prices.index[-1].date()}")
    
    # Align series
    aligned_wave, aligned_bench = align_series_for_alpha(wave_prices, benchmark_prices)
    
    print(f"  Aligned length: {len(aligned_wave)} (both wave and benchmark)")
    print(f"  Aligned dates: {aligned_wave.index[0].date()} to {aligned_wave.index[-1].date()}")
    
    # Both should have same length and same index
    assert len(aligned_wave) == len(aligned_bench), "Aligned series should have same length"
    assert (aligned_wave.index == aligned_bench.index).all(), "Aligned series should have same index"
    
    # Should be 10 days (the intersection)
    assert len(aligned_wave) == 10, f"Should have 10 aligned days, got {len(aligned_wave)}"
    
    print("  ✓ Benchmark alignment PASSED")


def test_alpha_computation():
    """Test alpha calculation with aligned series."""
    print("\n" + "=" * 70)
    print("TEST 8: Alpha Computation")
    print("=" * 70)
    
    # Create wave with 2% daily return
    wave_prices = create_synthetic_prices('2024-01-01', 35, daily_return=0.02)
    
    # Create benchmark with 1% daily return
    bench_prices = create_synthetic_prices('2024-01-01', 35, daily_return=0.01)
    
    # Compute 30-day alpha
    alpha_30d = compute_alpha(wave_prices, bench_prices, 30)
    
    # Expected:
    # Wave return: (1.02)^30 - 1 ≈ 0.8114
    # Bench return: (1.01)^30 - 1 ≈ 0.3478
    # Alpha: 0.8114 - 0.3478 ≈ 0.4636
    expected_wave = (1.02 ** 30) - 1.0
    expected_bench = (1.01 ** 30) - 1.0
    expected_alpha = expected_wave - expected_bench
    
    print(f"  Wave 30D return:  {expected_wave:.6f}")
    print(f"  Bench 30D return: {expected_bench:.6f}")
    print(f"  Expected alpha:   {expected_alpha:.6f}")
    print(f"  Computed alpha:   {alpha_30d:.6f}")
    print(f"  Difference:       {abs(alpha_30d - expected_alpha):.8f}")
    
    assert abs(alpha_30d - expected_alpha) < 1e-6, f"Alpha mismatch: {alpha_30d} vs {expected_alpha}"
    
    print("  ✓ Alpha computation PASSED")


def test_alpha_with_misaligned_dates():
    """Test alpha computation when wave and benchmark have different date ranges."""
    print("\n" + "=" * 70)
    print("TEST 9: Alpha with Misaligned Dates")
    print("=" * 70)
    
    # Wave: 40 days starting 2024-01-01
    wave_prices = create_synthetic_prices('2024-01-01', 40, daily_return=0.015)
    
    # Benchmark: 40 days starting 2023-12-20 (starts earlier)
    bench_prices = create_synthetic_prices('2023-12-20', 40, daily_return=0.008)
    
    print(f"  Wave dates: {wave_prices.index[0].date()} to {wave_prices.index[-1].date()}")
    print(f"  Bench dates: {bench_prices.index[0].date()} to {bench_prices.index[-1].date()}")
    
    # Compute alpha - should automatically align to common dates
    alpha_30d = compute_alpha(wave_prices, bench_prices, 30)
    
    # The alignment should use the intersection of dates
    # Both should be aligned before computing returns
    print(f"  Computed alpha: {alpha_30d:.6f}")
    
    # Should not be None or 0.0 if there's sufficient overlap
    assert alpha_30d != 0.0, "Alpha should be computed from aligned dates"
    assert alpha_30d is not None, "Alpha should not be None when sufficient data exists"
    
    print("  ✓ Alpha with misaligned dates PASSED")


def test_trading_days_map():
    """Test that TRADING_DAYS_MAP constants are correct."""
    print("\n" + "=" * 70)
    print("TEST 10: Trading Days Map Constants")
    print("=" * 70)
    
    print(f"  TRADING_DAYS_MAP = {TRADING_DAYS_MAP}")
    
    # Verify expected values
    assert TRADING_DAYS_MAP['1D'] == 1, "1D should be 1 trading day"
    assert TRADING_DAYS_MAP['30D'] == 30, "30D should be 30 trading days"
    assert TRADING_DAYS_MAP['60D'] == 60, "60D should be 60 trading days"
    assert TRADING_DAYS_MAP['365D'] == 252, "365D should be 252 trading days (standard year)"
    
    print("  ✓ Trading days map constants PASSED")


def run_all_tests():
    """Run all test functions."""
    print("\n" + "=" * 70)
    print("PERIOD RETURNS UNIT TESTS")
    print("=" * 70)
    
    test_functions = [
        test_basic_return_computation,
        test_30_day_trading_window,
        test_60_day_trading_window,
        test_252_day_trading_window,
        test_insufficient_data,
        test_edge_cases,
        test_benchmark_alignment,
        test_alpha_computation,
        test_alpha_with_misaligned_dates,
        test_trading_days_map,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n  ✗ TEST FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Passed: {passed}/{len(test_functions)}")
    print(f"  Failed: {failed}/{len(test_functions)}")
    
    if failed == 0:
        print("\n  ✓ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n  ✗ {failed} TEST(S) FAILED")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)

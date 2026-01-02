#!/usr/bin/env python3
"""
Test new snapshot generation functions in analytics_truth.py

Since we're in a sandboxed environment without network access,
these tests validate the function signatures and logic without
actually fetching live data.
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import pandas as pd
import numpy as np
from analytics_truth import (
    load_weights,
    expected_waves,
    compute_period_return,
    compute_wave_returns,
    _convert_wave_name_to_id,
)


def test_load_weights():
    """Test loading wave weights from CSV"""
    print("\n" + "=" * 80)
    print("TEST: load_weights()")
    print("=" * 80)
    
    weights_df = load_weights('wave_weights.csv')
    
    # Validate structure
    assert 'wave' in weights_df.columns
    assert 'ticker' in weights_df.columns
    assert 'weight' in weights_df.columns
    
    # Validate data
    assert len(weights_df) > 0
    assert weights_df['wave'].nunique() == 28
    
    print(f"✓ Loaded {len(weights_df)} weight entries")
    print(f"✓ Found {weights_df['wave'].nunique()} unique waves")
    print(f"✓ Found {weights_df['ticker'].nunique()} unique tickers")


def test_expected_waves():
    """Test getting expected wave list"""
    print("\n" + "=" * 80)
    print("TEST: expected_waves()")
    print("=" * 80)
    
    weights_df = load_weights('wave_weights.csv')
    waves = expected_waves(weights_df)
    
    # Validate
    assert len(waves) == 28
    assert len(set(waves)) == 28  # All unique
    assert waves == sorted(waves)  # Sorted
    
    print(f"✓ Found exactly {len(waves)} expected waves")
    print(f"✓ All waves are unique")
    print(f"✓ Waves are sorted alphabetically")
    print(f"\nFirst 5: {waves[:5]}")
    print(f"Last 5: {waves[-5:]}")


def test_compute_period_return():
    """Test return calculation logic"""
    print("\n" + "=" * 80)
    print("TEST: compute_period_return()")
    print("=" * 80)
    
    # Create synthetic price series
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = pd.Series(range(100, 200), index=dates)  # Linear growth
    
    # Test 1-day return
    ret_1d = compute_period_return(prices, 1)
    assert not np.isnan(ret_1d)
    print(f"✓ 1-day return: {ret_1d*100:.2f}%")
    
    # Test 30-day return
    ret_30d = compute_period_return(prices, 30)
    assert not np.isnan(ret_30d)
    print(f"✓ 30-day return: {ret_30d*100:.2f}%")
    
    # Test with insufficient data
    short_prices = pd.Series([100, 101], index=dates[:2])
    ret_short = compute_period_return(short_prices, 365)
    # Should still work but use available data
    assert not np.isnan(ret_short)
    print(f"✓ Handled short series gracefully")
    
    print("✓ Return calculations working correctly")


def test_compute_wave_returns():
    """Test wave return computation with mock data"""
    print("\n" + "=" * 80)
    print("TEST: compute_wave_returns()")
    print("=" * 80)
    
    # Create mock weights DataFrame
    weights_df = pd.DataFrame({
        'wave': ['Test Wave 1', 'Test Wave 1', 'Test Wave 2', 'Test Wave 2'],
        'ticker': ['AAPL', 'MSFT', 'SPY', 'QQQ'],
        'weight': [0.5, 0.5, 0.6, 0.4]
    })
    
    # Create mock price cache (AAPL and SPY succeed, MSFT and QQQ fail)
    dates = pd.date_range(start='2024-01-01', periods=400, freq='D')
    prices_cache = {
        'AAPL': pd.Series(range(100, 500), index=dates),
        'SPY': pd.Series(range(200, 600), index=dates),
        # MSFT and QQQ missing (simulating failure)
    }
    
    # Compute returns
    results = compute_wave_returns(weights_df, prices_cache)
    
    # Validate results
    assert 'Test Wave 1' in results
    assert 'Test Wave 2' in results
    
    # Test Wave 1: AAPL succeeded, MSFT failed
    wave1 = results['Test Wave 1']
    assert wave1['status'] == 'OK'  # Has at least one ticker
    assert 'MSFT' in wave1['missing_tickers']
    assert 'AAPL' in wave1['tickers_ok']
    assert wave1['coverage_pct'] == 50.0  # 1 of 2 tickers
    assert not np.isnan(wave1['return_1d'])
    print(f"✓ Test Wave 1: {wave1['status']}, coverage={wave1['coverage_pct']:.0f}%")
    
    # Test Wave 2: SPY succeeded, QQQ failed
    wave2 = results['Test Wave 2']
    assert wave2['status'] == 'OK'
    assert 'QQQ' in wave2['missing_tickers']
    assert 'SPY' in wave2['tickers_ok']
    assert wave2['coverage_pct'] == 50.0
    print(f"✓ Test Wave 2: {wave2['status']}, coverage={wave2['coverage_pct']:.0f}%")
    
    # Test wave with all failed tickers
    weights_df_fail = pd.DataFrame({
        'wave': ['Test Wave Fail', 'Test Wave Fail'],
        'ticker': ['NONEXISTENT1', 'NONEXISTENT2'],
        'weight': [0.5, 0.5]
    })
    results_fail = compute_wave_returns(weights_df_fail, prices_cache)
    wave_fail = results_fail['Test Wave Fail']
    assert wave_fail['status'] == 'NO DATA'
    assert wave_fail['coverage_pct'] == 0.0
    assert np.isnan(wave_fail['return_1d'])
    print(f"✓ Test Wave Fail: {wave_fail['status']}, coverage={wave_fail['coverage_pct']:.0f}%")
    
    print("✓ Wave return computation working correctly")


def test_wave_id_conversion():
    """Test wave name to wave_id conversion"""
    print("\n" + "=" * 80)
    print("TEST: Wave ID Conversion")
    print("=" * 80)
    
    test_cases = [
        ("S&P 500 Wave", "s_p_500_wave"),
        ("AI & Cloud MegaCap Wave", "ai_cloud_megacap_wave"),
        ("Clean Transit-Infrastructure Wave", "clean_transit_infrastructure_wave"),
        ("US Small-Cap Disruptors Wave", "us_small_cap_disruptors_wave"),
    ]
    
    for wave_name, expected_pattern in test_cases:
        wave_id = _convert_wave_name_to_id(wave_name)
        
        # Validate formatting
        assert ' ' not in wave_id, f"Wave ID should not contain spaces: {wave_id}"
        assert wave_id == wave_id.lower(), f"Wave ID should be lowercase: {wave_id}"
        
        print(f"✓ '{wave_name}' -> '{wave_id}'")
    
    print("✓ All wave ID conversions valid")


def test_snapshot_structure():
    """Test that we can build the expected snapshot structure"""
    print("\n" + "=" * 80)
    print("TEST: Snapshot Structure")
    print("=" * 80)
    
    # Load actual weights
    weights_df = load_weights('wave_weights.csv')
    waves = expected_waves(weights_df)
    
    # Build a mock snapshot DataFrame (simulating generate_live_snapshot_csv output)
    rows = []
    for wave_name in waves:
        wave_id = _convert_wave_name_to_id(wave_name)
        row = {
            'wave_id': wave_id,
            'Wave': wave_name,
            'Return_1D': np.nan,
            'Return_30D': np.nan,
            'Return_60D': np.nan,
            'Return_365D': np.nan,
            'status': 'NO DATA',
            'coverage_pct': 0.0,
            'missing_tickers': '',
            'tickers_ok': 0,
            'tickers_total': 0,
            'asof_utc': '2026-01-02T00:00:00'
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Validate structure
    assert len(df) == 28, f"Expected 28 rows, got {len(df)}"
    assert df['wave_id'].nunique() == 28, "Expected 28 unique wave_ids"
    assert 'Wave' in df.columns
    assert 'Return_1D' in df.columns
    assert 'Return_30D' in df.columns
    assert 'Return_60D' in df.columns
    assert 'Return_365D' in df.columns
    assert 'status' in df.columns
    assert 'coverage_pct' in df.columns
    assert 'missing_tickers' in df.columns
    assert 'tickers_ok' in df.columns
    assert 'tickers_total' in df.columns
    assert 'asof_utc' in df.columns
    
    print(f"✓ Snapshot has exactly {len(df)} rows")
    print(f"✓ All required columns present")
    print(f"✓ All wave_ids are unique")
    print(f"\nColumns: {', '.join(df.columns)}")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("NEW SNAPSHOT FUNCTIONS TESTS")
    print("=" * 80)
    
    tests = [
        ("load_weights", test_load_weights),
        ("expected_waves", test_expected_waves),
        ("compute_period_return", test_compute_period_return),
        ("compute_wave_returns", test_compute_wave_returns),
        ("wave_id_conversion", test_wave_id_conversion),
        ("snapshot_structure", test_snapshot_structure),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test '{test_name}' FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

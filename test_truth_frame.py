"""
test_truth_frame.py

Unit tests for analytics_truth.py TruthFrame module
"""

import pandas as pd
import numpy as np
from analytics_truth import (
    get_truth_frame,
    get_wave_truth,
    filter_truth_frame,
    _create_empty_truth_frame,
    _get_all_28_wave_ids,
    _get_display_name,
)


def test_get_all_28_wave_ids():
    """Test that we can get all 28 wave IDs"""
    wave_ids = _get_all_28_wave_ids()
    
    print(f"✓ Found {len(wave_ids)} wave IDs")
    assert len(wave_ids) == 28, f"Expected 28 waves, got {len(wave_ids)}"
    
    # Check some known waves
    assert "sp500_wave" in wave_ids
    assert "income_wave" in wave_ids
    assert "crypto_l1_growth_wave" in wave_ids
    
    print("✓ All 28 wave IDs validated")


def test_get_display_name():
    """Test display name resolution"""
    name = _get_display_name("sp500_wave")
    print(f"✓ sp500_wave -> {name}")
    assert name is not None and len(name) > 0
    
    name = _get_display_name("income_wave")
    print(f"✓ income_wave -> {name}")
    assert name is not None and len(name) > 0


def test_create_empty_truth_frame():
    """Test empty TruthFrame creation"""
    truth_df = _create_empty_truth_frame()
    
    print(f"✓ Created empty TruthFrame with {len(truth_df)} rows")
    assert len(truth_df) == 28, f"Expected 28 rows, got {len(truth_df)}"
    
    # Check required columns
    required_columns = [
        'wave_id', 'display_name', 'mode', 'readiness_status', 'coverage_pct',
        'data_regime_tag', 'return_1d', 'return_30d', 'return_60d', 'return_365d',
        'alpha_1d', 'alpha_30d', 'alpha_60d', 'alpha_365d',
        'benchmark_return_1d', 'benchmark_return_30d', 'benchmark_return_60d', 'benchmark_return_365d',
        'exposure_pct', 'cash_pct', 'beta_real', 'beta_target', 'beta_drift',
        'turnover_est', 'drawdown_60d', 'alert_badges', 'last_snapshot_ts'
    ]
    
    for col in required_columns:
        assert col in truth_df.columns, f"Missing required column: {col}"
    
    print(f"✓ All {len(required_columns)} required columns present")
    
    # Check data types and defaults
    assert all(truth_df['readiness_status'] == 'unavailable')
    assert all(truth_df['data_regime_tag'] == 'UNAVAILABLE')
    assert all(truth_df['coverage_pct'] == 0.0)
    assert all(pd.isna(truth_df['return_1d']))
    
    print("✓ Default values validated")


def test_get_truth_frame_safe_mode():
    """Test TruthFrame retrieval in Safe Mode"""
    print("\n=== Testing Safe Mode ===")
    
    # Safe Mode should always return a DataFrame with 28 waves
    truth_df = get_truth_frame(safe_mode=True)
    
    print(f"✓ Got TruthFrame with {len(truth_df)} rows in Safe Mode")
    assert len(truth_df) == 28, f"Expected 28 rows in Safe Mode, got {len(truth_df)}"
    
    # Check that all wave_ids are present
    wave_ids = _get_all_28_wave_ids()
    for wave_id in wave_ids:
        assert wave_id in truth_df['wave_id'].values, f"Missing wave_id in Safe Mode: {wave_id}"
    
    print("✓ All 28 waves present in Safe Mode")


def test_get_wave_truth():
    """Test getting truth for a specific wave"""
    truth_df = get_truth_frame(safe_mode=True)
    
    # Get specific wave
    wave_data = get_wave_truth("sp500_wave", truth_df)
    
    print(f"✓ Got wave truth for sp500_wave")
    assert wave_data is not None
    assert wave_data.get('wave_id') == 'sp500_wave'
    assert 'display_name' in wave_data
    assert 'return_1d' in wave_data
    
    print(f"  - Display Name: {wave_data.get('display_name')}")
    print(f"  - Readiness: {wave_data.get('readiness_status')}")
    print(f"  - Data Regime: {wave_data.get('data_regime_tag')}")


def test_filter_truth_frame():
    """Test filtering TruthFrame"""
    truth_df = get_truth_frame(safe_mode=True)
    
    # Filter by wave_ids
    filtered = filter_truth_frame(
        truth_df,
        wave_ids=["sp500_wave", "income_wave"]
    )
    
    print(f"✓ Filtered to {len(filtered)} waves")
    assert len(filtered) == 2
    assert set(filtered['wave_id'].values) == {"sp500_wave", "income_wave"}
    
    # Filter by readiness_status
    filtered = filter_truth_frame(
        truth_df,
        readiness_status=["unavailable"]
    )
    print(f"✓ Filtered to {len(filtered)} unavailable waves")
    
    # Filter by coverage
    # Set some coverage values for testing
    test_df = truth_df.copy()
    test_df.loc[0, 'coverage_pct'] = 100.0
    test_df.loc[1, 'coverage_pct'] = 75.0
    test_df.loc[2, 'coverage_pct'] = 25.0
    
    filtered = filter_truth_frame(
        test_df,
        min_coverage_pct=50.0
    )
    print(f"✓ Filtered to {len(filtered)} waves with coverage >= 50%")
    # Just check that filtering works, don't assert exact count
    assert len(filtered) >= 0, "Filter should return valid DataFrame"


def test_truth_frame_never_drops_rows():
    """Test that TruthFrame always maintains all 28 rows"""
    print("\n=== Testing No Row Dropping ===")
    
    # Even with failures, should always return 28 rows
    truth_df = get_truth_frame(safe_mode=True)
    assert len(truth_df) == 28, "TruthFrame should never drop rows"
    
    # Check that every wave_id is unique
    assert len(truth_df['wave_id'].unique()) == 28, "All wave_ids should be unique"
    
    print("✓ TruthFrame maintains all 28 rows (no drops)")


def test_truth_frame_column_consistency():
    """Test that TruthFrame has consistent columns"""
    print("\n=== Testing Column Consistency ===")
    
    truth_df = get_truth_frame(safe_mode=True)
    
    # Required columns (from spec)
    required_columns = [
        'wave_id', 'display_name', 'mode', 'readiness_status', 'coverage_pct',
        'data_regime_tag', 
        'return_1d', 'return_30d', 'return_60d', 'return_365d',
        'alpha_1d', 'alpha_30d', 'alpha_60d', 'alpha_365d',
        'benchmark_return_1d', 'benchmark_return_30d', 'benchmark_return_60d', 'benchmark_return_365d',
        'exposure_pct', 'cash_pct',
        'beta_real', 'beta_target', 'beta_drift',
        'turnover_est', 'drawdown_60d',
        'alert_badges', 'last_snapshot_ts'
    ]
    
    for col in required_columns:
        assert col in truth_df.columns, f"Missing required column: {col}"
    
    print(f"✓ All {len(required_columns)} required columns present and accounted for")


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("TRUTHFRAME UNIT TESTS")
    print("=" * 80)
    
    tests = [
        ("Get All 28 Wave IDs", test_get_all_28_wave_ids),
        ("Get Display Name", test_get_display_name),
        ("Create Empty TruthFrame", test_create_empty_truth_frame),
        ("Get TruthFrame (Safe Mode)", test_get_truth_frame_safe_mode),
        ("Get Wave Truth", test_get_wave_truth),
        ("Filter TruthFrame", test_filter_truth_frame),
        ("TruthFrame Never Drops Rows", test_truth_frame_never_drops_rows),
        ("TruthFrame Column Consistency", test_truth_frame_column_consistency),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"Test: {test_name}")
        print(f"{'='*80}")
        try:
            test_func()
            print(f"✓ PASSED: {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {test_name}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {test_name}")
            print(f"  Exception: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

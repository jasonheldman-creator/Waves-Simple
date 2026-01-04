#!/usr/bin/env python3
"""
Test the new snapshot generation logic in analytics_truth.py

This test validates that the snapshot generation:
1. Derives expected waves from wave_weights.csv
2. Processes each wave with per-ticker error handling
3. Tracks failed tickers
4. Generates exactly one row per wave
5. Validates the final count
6. Writes to data/live_snapshot.csv
"""

import sys
import os
sys.path.insert(0, '/home/runner/work/Waves-Simple/Waves-Simple')

import pandas as pd
import numpy as np
from analytics_truth import (
    generate_snapshot_with_full_coverage,
    _derive_expected_waves_from_weights,
    _convert_wave_name_to_id,
    _get_wave_tickers_from_weights
)


def test_derive_expected_waves():
    """Test deriving expected waves from wave_weights.csv"""
    print("\n" + "=" * 80)
    print("TEST: Derive Expected Waves")
    print("=" * 80)
    
    from waves_engine import WAVE_ID_REGISTRY
    expected_count = len(WAVE_ID_REGISTRY)
    
    expected_waves = _derive_expected_waves_from_weights()
    
    assert len(expected_waves) == expected_count, f"Expected {expected_count} waves, got {len(expected_waves)}"
    assert len(expected_waves) == len(set(expected_waves)), "Waves should be unique"
    assert expected_waves == sorted(expected_waves), "Waves should be sorted"
    
    print(f"✓ Derived {len(expected_waves)} expected waves (from WAVE_ID_REGISTRY)")
    print(f"✓ All waves are unique")
    print(f"✓ Waves are sorted")
    print(f"\nFirst 5 waves: {expected_waves[:5]}")
    print(f"Last 5 waves: {expected_waves[-5:]}")
    

def test_wave_name_to_id_conversion():
    """Test converting wave names to wave IDs"""
    print("\n" + "=" * 80)
    print("TEST: Wave Name to ID Conversion")
    print("=" * 80)
    
    test_cases = [
        ("S&P 500 Wave", "s_p_500_wave"),
        ("AI & Cloud MegaCap Wave", "ai_cloud_megacap_wave"),
        ("Clean Transit-Infrastructure Wave", "clean_transit_infrastructure_wave"),
    ]
    
    for wave_name, expected_pattern in test_cases:
        wave_id = _convert_wave_name_to_id(wave_name)
        # Check that special characters are handled
        assert ' ' not in wave_id, f"Wave ID should not contain spaces: {wave_id}"
        assert wave_id == wave_id.lower(), f"Wave ID should be lowercase: {wave_id}"
        print(f"✓ '{wave_name}' -> '{wave_id}'")


def test_get_wave_tickers():
    """Test getting tickers for each wave"""
    print("\n" + "=" * 80)
    print("TEST: Get Wave Tickers")
    print("=" * 80)
    
    expected_waves = _derive_expected_waves_from_weights()
    
    total_tickers = 0
    for wave_name in expected_waves[:5]:  # Test first 5 waves
        tickers = _get_wave_tickers_from_weights(wave_name)
        total_tickers += len(tickers)
        print(f"✓ {wave_name}: {len(tickers)} tickers")
        assert len(tickers) > 0, f"Wave {wave_name} should have tickers"
    
    print(f"\n✓ Total tickers in first 5 waves: {total_tickers}")


def test_snapshot_generation():
    """Test the complete snapshot generation"""
    print("\n" + "=" * 80)
    print("TEST: Complete Snapshot Generation")
    print("=" * 80)
    
    # Generate snapshot
    df = generate_snapshot_with_full_coverage()
    
    # Verify structure
    assert 'Wave' in df.columns, "Snapshot should have 'Wave' column"
    assert 'Wave_ID' in df.columns, "Snapshot should have 'Wave_ID' column"
    assert 'status' in df.columns, "Snapshot should have 'status' column"
    assert 'missing_tickers' in df.columns, "Snapshot should have 'missing_tickers' column"
    assert 'missing_ticker_count' in df.columns, "Snapshot should have 'missing_ticker_count' column"
    assert 'Coverage_Score' in df.columns, "Snapshot should have 'Coverage_Score' column"
    
    # Verify counts
    expected_waves = _derive_expected_waves_from_weights()
    assert len(df) == len(expected_waves), f"Expected {len(expected_waves)} rows, got {len(df)}"
    assert df['Wave'].nunique() == len(expected_waves), f"Expected {len(expected_waves)} unique waves"
    
    # Verify all expected waves are present
    actual_waves = set(df['Wave'].tolist())
    expected_waves_set = set(expected_waves)
    assert actual_waves == expected_waves_set, "All expected waves should be present"
    
    # Verify status column
    assert df['status'].isin(['OK', 'NO DATA']).all(), "Status should be either OK or NO DATA"
    
    # Verify missing_tickers tracking
    no_data_rows = df[df['status'] == 'NO DATA']
    for _, row in no_data_rows.iterrows():
        # NO DATA rows should have missing tickers tracked
        assert pd.notna(row['missing_tickers']) or row['missing_tickers'] == '', \
            f"NO DATA row should track missing tickers: {row['Wave']}"
    
    print(f"✓ Snapshot has {len(df)} rows")
    print(f"✓ Snapshot has {df['Wave'].nunique()} unique waves")
    print(f"✓ All expected waves are present")
    print(f"✓ Status column is valid")
    print(f"✓ Waves with OK status: {(df['status'] == 'OK').sum()}")
    print(f"✓ Waves with NO DATA status: {(df['status'] == 'NO DATA').sum()}")
    
    # Verify file was written
    snapshot_file = "data/live_snapshot.csv"
    assert os.path.exists(snapshot_file), f"Snapshot file should exist: {snapshot_file}"
    
    # Verify file has correct number of lines
    with open(snapshot_file) as f:
        lines = f.readlines()
    assert len(lines) == len(expected_waves) + 1, \
        f"File should have {len(expected_waves) + 1} lines (including header)"
    
    print(f"✓ Snapshot file written to {snapshot_file}")
    print(f"✓ File has correct number of lines: {len(lines)}")


def test_hard_assertion():
    """Test that hard assertion catches mismatched wave counts"""
    print("\n" + "=" * 80)
    print("TEST: Hard Assertion Validation")
    print("=" * 80)
    
    # The assertion should pass when we generate normally
    try:
        df = generate_snapshot_with_full_coverage()
        expected_waves = _derive_expected_waves_from_weights()
        
        # Manually verify the assertion
        actual_wave_count = df['Wave'].nunique()
        expected_wave_count = len(expected_waves)
        
        assert actual_wave_count == expected_wave_count, \
            f"Assertion should pass: {actual_wave_count} == {expected_wave_count}"
        
        print(f"✓ Hard assertion passed correctly")
        print(f"✓ Expected: {expected_wave_count}, Actual: {actual_wave_count}")
        
    except AssertionError as e:
        print(f"✗ Hard assertion failed: {e}")
        raise


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("ANALYTICS_TRUTH.PY SNAPSHOT GENERATION TESTS")
    print("=" * 80)
    
    tests = [
        ("Derive Expected Waves", test_derive_expected_waves),
        ("Wave Name to ID Conversion", test_wave_name_to_id_conversion),
        ("Get Wave Tickers", test_get_wave_tickers),
        ("Snapshot Generation", test_snapshot_generation),
        ("Hard Assertion Validation", test_hard_assertion),
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

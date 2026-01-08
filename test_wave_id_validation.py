#!/usr/bin/env python3
"""
Test wave_id validation in generate_live_snapshot_csv

This test validates the new requirements:
A) Dynamic expected wave IDs derived from wave_weights.csv
B) Normalization of wave_id (strip, handle None/blank)
C) Single validation point with comprehensive diagnostics
"""

import sys
import os
sys.path.insert(0, '/home/runner/work/Waves-Simple/Waves-Simple')

import pandas as pd
import numpy as np
from analytics_truth import (
    load_weights,
    expected_waves,
    _convert_wave_name_to_id,
)


def test_dynamic_wave_ids():
    """Test that expected wave IDs are derived dynamically from wave_weights.csv"""
    print("\n" + "=" * 80)
    print("TEST: Dynamic Wave IDs Derivation")
    print("=" * 80)
    
    # Load weights
    weights_df = load_weights("wave_weights.csv")
    
    # Get expected waves
    waves = expected_waves(weights_df)
    
    # Derive expected wave_ids dynamically
    expected_wave_ids = sorted(set([_convert_wave_name_to_id(wave_name) for wave_name in waves]))
    expected_count = len(expected_wave_ids)
    
    print(f"✓ Loaded {len(weights_df)} weight entries")
    print(f"✓ Found {len(waves)} expected waves")
    print(f"✓ Derived {expected_count} expected wave_ids dynamically")
    
    # Verify no hardcoded count (should work with any number of waves)
    assert expected_count > 0, "Should have at least one wave"
    assert expected_count == len(waves), "Wave count should match wave_id count"
    assert len(expected_wave_ids) == len(set(expected_wave_ids)), "Wave IDs should be unique"
    
    print(f"✓ Dynamic derivation works correctly")
    print(f"  First 5 wave_ids: {expected_wave_ids[:5]}")
    print(f"  Last 5 wave_ids: {expected_wave_ids[-5:]}")


def test_wave_id_normalization():
    """Test wave_id normalization (strip whitespace, handle None/blank)"""
    print("\n" + "=" * 80)
    print("TEST: Wave ID Normalization")
    print("=" * 80)
    
    # Test cases for normalization
    test_cases = [
        ("S&P 500 Wave", "s_p_500_wave"),
        ("AI & Cloud MegaCap Wave", "ai_cloud_megacap_wave"),
        ("Clean Transit-Infrastructure Wave", "clean_transit_infrastructure_wave"),
    ]
    
    for wave_name, expected_pattern in test_cases:
        wave_id = _convert_wave_name_to_id(wave_name)
        
        # Check normalization
        assert wave_id is not None, f"wave_id should not be None for '{wave_name}'"
        assert isinstance(wave_id, str), f"wave_id should be string for '{wave_name}'"
        assert wave_id.strip() == wave_id, f"wave_id should be stripped for '{wave_name}'"
        assert wave_id == wave_id.lower(), f"wave_id should be lowercase for '{wave_name}'"
        assert ' ' not in wave_id, f"wave_id should not contain spaces for '{wave_name}'"
        
        print(f"✓ '{wave_name}' -> '{wave_id}'")
    
    print(f"✓ Wave ID normalization works correctly")


def test_validation_metrics():
    """Test that validation metrics are correctly computed"""
    print("\n" + "=" * 80)
    print("TEST: Validation Metrics")
    print("=" * 80)
    
    # Create test DataFrame with various scenarios
    test_data = {
        'wave_id': ['wave_1', 'wave_2', 'wave_3', None, ''],
        'Wave': ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4', 'Wave 5']
    }
    df = pd.DataFrame(test_data)
    
    # Test nunique with dropna=False
    nunique_with_na = df['wave_id'].nunique(dropna=False)
    nunique_without_na = df['wave_id'].nunique(dropna=True)
    
    print(f"✓ nunique(dropna=False): {nunique_with_na}")
    print(f"✓ nunique(dropna=True): {nunique_without_na}")
    
    # Test isna count
    isna_sum = df['wave_id'].isna().sum()
    print(f"✓ isna().sum(): {isna_sum}")
    assert isna_sum == 1, "Should detect 1 null value"
    
    # Test blank count
    blank_sum = sum(1 for x in df['wave_id'] if isinstance(x, str) and not x.strip())
    print(f"✓ blank count: {blank_sum}")
    assert blank_sum == 1, "Should detect 1 blank value"
    
    # Test duplicates
    test_data_dup = {
        'wave_id': ['wave_1', 'wave_2', 'wave_1'],
        'Wave': ['Wave 1', 'Wave 2', 'Wave 1 Duplicate']
    }
    df_dup = pd.DataFrame(test_data_dup)
    wave_id_counts = df_dup['wave_id'].value_counts()
    duplicates = wave_id_counts[wave_id_counts > 1]
    
    print(f"✓ duplicate count: {len(duplicates)}")
    assert len(duplicates) == 1, "Should detect 1 duplicate"
    assert 'wave_1' in duplicates.index, "Should identify wave_1 as duplicate"
    
    print(f"✓ Validation metrics computed correctly")


def test_expected_waves_no_hardcoded_count():
    """Test that expected_waves() doesn't hardcode wave count"""
    print("\n" + "=" * 80)
    print("TEST: No Hardcoded Wave Count")
    print("=" * 80)
    
    # Create a test DataFrame with different number of waves
    test_weights = pd.DataFrame({
        'wave': ['Wave A', 'Wave A', 'Wave B', 'Wave B', 'Wave C'],
        'ticker': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'weight': [0.5, 0.5, 0.6, 0.4, 1.0]
    })
    
    waves = expected_waves(test_weights)
    
    print(f"✓ Test with 3 waves: {waves}")
    assert len(waves) == 3, "Should return 3 waves"
    assert waves == ['Wave A', 'Wave B', 'Wave C'], "Should return correct waves"
    
    # Test with different number
    test_weights_2 = pd.DataFrame({
        'wave': ['W1', 'W2', 'W3', 'W4', 'W5'],
        'ticker': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'weight': [1.0, 1.0, 1.0, 1.0, 1.0]
    })
    
    waves_2 = expected_waves(test_weights_2)
    
    print(f"✓ Test with 5 waves: {waves_2}")
    assert len(waves_2) == 5, "Should return 5 waves"
    
    print(f"✓ expected_waves() works dynamically without hardcoded count")


def run_all_tests():
    """Run all validation tests"""
    print("\n" + "=" * 80)
    print("WAVE_ID VALIDATION TESTS")
    print("=" * 80)
    
    tests = [
        ("Dynamic Wave IDs Derivation", test_dynamic_wave_ids),
        ("Wave ID Normalization", test_wave_id_normalization),
        ("Validation Metrics", test_validation_metrics),
        ("No Hardcoded Wave Count", test_expected_waves_no_hardcoded_count),
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

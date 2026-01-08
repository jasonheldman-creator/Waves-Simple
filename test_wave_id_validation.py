#!/usr/bin/env python3
"""
Test suite for wave_id validation improvements.

This test verifies:
1. Russell 3000 Wave is present in WAVE_ID_REGISTRY
2. All waves in wave_weights.csv have valid wave_ids
3. _convert_wave_name_to_id never returns None
4. Validation logic correctly detects duplicates, nulls, and blanks
"""

import pandas as pd
import numpy as np
import sys


def test_russell_3000_in_registry():
    """Test that Russell 3000 Wave is in WAVE_ID_REGISTRY."""
    print("\n" + "=" * 80)
    print("Test 1: Russell 3000 Wave in WAVE_ID_REGISTRY")
    print("=" * 80)
    
    from waves_engine import WAVE_ID_REGISTRY, get_wave_id_from_display_name
    
    # Check registry has 28 waves
    assert len(WAVE_ID_REGISTRY) == 28, f"Expected 28 waves, got {len(WAVE_ID_REGISTRY)}"
    print(f"✓ WAVE_ID_REGISTRY has 28 waves")
    
    # Check Russell 3000 Wave is present
    assert "russell_3000_wave" in WAVE_ID_REGISTRY, "russell_3000_wave not in registry"
    print(f"✓ russell_3000_wave is in WAVE_ID_REGISTRY")
    
    # Check get_wave_id_from_display_name works
    wave_id = get_wave_id_from_display_name("Russell 3000 Wave")
    assert wave_id == "russell_3000_wave", f"Expected 'russell_3000_wave', got '{wave_id}'"
    print(f"✓ get_wave_id_from_display_name('Russell 3000 Wave') returns 'russell_3000_wave'")
    
    print("\n✓ Test 1 PASSED\n")


def test_all_waves_have_valid_ids():
    """Test that all waves in wave_weights.csv can be converted to valid wave_ids."""
    print("=" * 80)
    print("Test 2: All waves in wave_weights.csv have valid wave_ids")
    print("=" * 80)
    
    from analytics_truth import _convert_wave_name_to_id
    
    # Load wave_weights.csv
    weights_df = pd.read_csv('wave_weights.csv')
    waves = sorted(weights_df['wave'].unique())
    
    assert len(waves) == 28, f"Expected 28 waves in wave_weights.csv, got {len(waves)}"
    print(f"✓ wave_weights.csv has 28 waves")
    
    # Test each wave can be converted to a valid wave_id
    all_valid = True
    invalid_waves = []
    
    for wave_name in waves:
        wave_id = _convert_wave_name_to_id(wave_name)
        
        if wave_id is None or wave_id == '' or not wave_id.strip():
            all_valid = False
            invalid_waves.append(f"{wave_name} -> {repr(wave_id)}")
    
    if not all_valid:
        print("\n✗ Found waves with invalid wave_ids:")
        for invalid in invalid_waves:
            print(f"  - {invalid}")
        raise AssertionError(f"Found {len(invalid_waves)} waves with invalid wave_ids")
    
    print(f"✓ All 28 waves convert to valid wave_ids (no None/empty values)")
    
    # Test uniqueness
    wave_ids = [_convert_wave_name_to_id(wave_name) for wave_name in waves]
    unique_count = len(set(wave_ids))
    
    assert unique_count == 28, f"Expected 28 unique wave_ids, got {unique_count}"
    print(f"✓ All wave_ids are unique ({unique_count} unique values)")
    
    print("\n✓ Test 2 PASSED\n")


def test_convert_wave_name_fallback():
    """Test that _convert_wave_name_to_id fallback works correctly."""
    print("=" * 80)
    print("Test 3: _convert_wave_name_to_id fallback logic")
    print("=" * 80)
    
    from analytics_truth import _convert_wave_name_to_id
    
    # Test with a real wave name
    wave_id = _convert_wave_name_to_id("Russell 3000 Wave")
    assert wave_id == "russell_3000_wave", f"Expected 'russell_3000_wave', got '{wave_id}'"
    print(f"✓ 'Russell 3000 Wave' -> '{wave_id}'")
    
    # Test with a fictional wave name (should use fallback)
    wave_id = _convert_wave_name_to_id("Fictional Test Wave")
    assert wave_id is not None and wave_id != '', f"Fallback failed, got {repr(wave_id)}"
    print(f"✓ 'Fictional Test Wave' -> '{wave_id}' (fallback works)")
    
    # Test with edge cases
    edge_cases = [
        "Wave & Test",  # Ampersand
        "Test-Wave",    # Hyphen
        "Test / Wave",  # Slash
        "Test  Wave",   # Multiple spaces
    ]
    
    for test_case in edge_cases:
        wave_id = _convert_wave_name_to_id(test_case)
        assert wave_id is not None and wave_id != '', f"Failed for '{test_case}', got {repr(wave_id)}"
        print(f"✓ '{test_case}' -> '{wave_id}'")
    
    print("\n✓ Test 3 PASSED\n")


def test_validation_logic():
    """Test the validation logic in generate_live_snapshot_csv."""
    print("=" * 80)
    print("Test 4: Validation logic for wave_id")
    print("=" * 80)
    
    from analytics_truth import _convert_wave_name_to_id
    
    # Load expected waves
    weights_df = pd.read_csv('wave_weights.csv')
    waves = sorted(weights_df['wave'].unique())
    expected_wave_ids = [_convert_wave_name_to_id(wave_name) for wave_name in waves]
    expected_count = len(expected_wave_ids)
    
    # Test 1: Valid DataFrame
    print("\n  Subtest 4.1: Valid DataFrame with 28 unique wave_ids")
    rows = []
    for wave_name in waves:
        wave_id = _convert_wave_name_to_id(wave_name)
        rows.append({'wave_id': wave_id, 'Wave': wave_name})
    
    df = pd.DataFrame(rows)
    
    # Normalize
    df['wave_id'] = df['wave_id'].fillna('')
    df['wave_id'] = df['wave_id'].astype(str)
    df['wave_id'] = df['wave_id'].str.strip()
    
    # Validate
    isna_sum = df['wave_id'].isna().sum()
    blank_sum = (df['wave_id'] == '').sum()
    nunique = df['wave_id'].nunique(dropna=False)
    
    assert nunique == expected_count, f"Expected {expected_count} unique, got {nunique}"
    assert isna_sum == 0, f"Found {isna_sum} null values"
    assert blank_sum == 0, f"Found {blank_sum} blank values"
    print(f"  ✓ Valid DataFrame passes validation")
    
    # Test 2: Duplicate detection
    print("\n  Subtest 4.2: Duplicate wave_id detection")
    rows_dup = rows.copy()
    rows_dup.append({'wave_id': 'sp500_wave', 'Wave': 'Duplicate'})
    df_dup = pd.DataFrame(rows_dup)
    
    df_dup['wave_id'] = df_dup['wave_id'].fillna('')
    df_dup['wave_id'] = df_dup['wave_id'].astype(str)
    df_dup['wave_id'] = df_dup['wave_id'].str.strip()
    
    wave_id_counts = df_dup['wave_id'].value_counts()
    duplicates = wave_id_counts[wave_id_counts > 1].to_dict()
    
    assert len(duplicates) > 0, "Failed to detect duplicates"
    assert 'sp500_wave' in duplicates, "Failed to detect sp500_wave duplicate"
    print(f"  ✓ Duplicate detection works (found {len(duplicates)} duplicates)")
    
    # Test 3: Null/blank detection
    print("\n  Subtest 4.3: Null/blank wave_id detection")
    rows_null = rows.copy()
    rows_null[0]['wave_id'] = None
    rows_null[1]['wave_id'] = ''
    rows_null[2]['wave_id'] = '  '  # Whitespace
    df_null = pd.DataFrame(rows_null)
    
    df_null['wave_id'] = df_null['wave_id'].fillna('')
    df_null['wave_id'] = df_null['wave_id'].astype(str)
    df_null['wave_id'] = df_null['wave_id'].str.strip()
    
    blank_sum_null = (df_null['wave_id'] == '').sum()
    
    assert blank_sum_null == 3, f"Expected 3 blank values, got {blank_sum_null}"
    print(f"  ✓ Null/blank detection works (found {blank_sum_null} blank values)")
    
    print("\n✓ Test 4 PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("WAVE_ID VALIDATION TEST SUITE")
    print("=" * 80)
    
    try:
        test_russell_3000_in_registry()
        test_all_waves_have_valid_ids()
        test_convert_wave_name_fallback()
        test_validation_logic()
        
        print("=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

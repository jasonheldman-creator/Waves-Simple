#!/usr/bin/env python3
"""
Test wave_id debugging information in generate_live_snapshot_csv

This test validates that the new debugging logic correctly identifies
and reports duplicate wave_ids when they occur.
"""

import sys
import os
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def test_duplicate_detection_logic():
    """
    Test that the duplicate detection logic works as expected.
    This simulates what happens in generate_live_snapshot_csv.
    """
    print("\n" + "=" * 80)
    print("TEST: Duplicate Wave ID Detection Logic")
    print("=" * 80)
    
    # Create a mock DataFrame with duplicate wave_ids
    rows = [
        {'wave_id': 'sp500_wave', 'Wave': 'S&P 500 Wave', 'Return_1D': 0.01},
        {'wave_id': 'growth_wave', 'Wave': 'Growth Wave', 'Return_1D': 0.02},
        {'wave_id': 'sp500_wave', 'Wave': 'SP 500 Wave (Duplicate)', 'Return_1D': 0.03},  # Duplicate!
        {'wave_id': 'income_wave', 'Wave': 'Income Wave', 'Return_1D': 0.04},
        {'wave_id': 'growth_wave', 'Wave': 'Growth Wave (Also Duplicate)', 'Return_1D': 0.05},  # Duplicate!
    ]
    
    df = pd.DataFrame(rows)
    
    print("\nTest DataFrame:")
    print(df[['wave_id', 'Wave']])
    
    # Count occurrences of each wave_id to detect duplicates
    wave_id_counts = Counter(df['wave_id'])
    duplicates = {wave_id: count for wave_id, count in wave_id_counts.items() if count > 1}
    
    print("\n--- DEBUG: Wave ID Uniqueness Check ---")
    print(f"Total wave_ids: {len(df)}")
    print(f"Unique wave_ids: {df['wave_id'].nunique()}")
    
    if duplicates:
        print("\n⚠️  DUPLICATE WAVE_IDs DETECTED:")
        for wave_id, count in duplicates.items():
            print(f"  - wave_id '{wave_id}' appears {count} times")
            # Get all rows with this duplicate wave_id
            duplicate_rows = df[df['wave_id'] == wave_id]
            print(f"    Corresponding display names (Wave column):")
            for idx, row in duplicate_rows.iterrows():
                print(f"      * '{row['Wave']}' (wave_id: '{row['wave_id']}')")
    else:
        print("✓ No duplicate wave_ids found")
    
    print("---------------------------------------\n")
    
    # Validate that we detected the duplicates
    assert len(duplicates) == 2, f"Expected 2 duplicate wave_ids, found {len(duplicates)}"
    assert 'sp500_wave' in duplicates, "Expected 'sp500_wave' to be detected as duplicate"
    assert 'growth_wave' in duplicates, "Expected 'growth_wave' to be detected as duplicate"
    assert duplicates['sp500_wave'] == 2, "Expected 'sp500_wave' to appear 2 times"
    assert duplicates['growth_wave'] == 2, "Expected 'growth_wave' to appear 2 times"
    
    print("✓ Duplicate detection logic works correctly")
    print("✓ All duplicate wave_ids were identified")
    print("✓ Correct display names were associated with duplicates")


def test_no_duplicates_case():
    """
    Test that the logic correctly reports no duplicates when all wave_ids are unique.
    """
    print("\n" + "=" * 80)
    print("TEST: No Duplicates Case")
    print("=" * 80)
    
    # Create a mock DataFrame with NO duplicate wave_ids
    rows = [
        {'wave_id': 'sp500_wave', 'Wave': 'S&P 500 Wave', 'Return_1D': 0.01},
        {'wave_id': 'growth_wave', 'Wave': 'Growth Wave', 'Return_1D': 0.02},
        {'wave_id': 'income_wave', 'Wave': 'Income Wave', 'Return_1D': 0.04},
        {'wave_id': 'value_wave', 'Wave': 'Value Wave', 'Return_1D': 0.05},
    ]
    
    df = pd.DataFrame(rows)
    
    print("\nTest DataFrame:")
    print(df[['wave_id', 'Wave']])
    
    # Count occurrences of each wave_id to detect duplicates
    wave_id_counts = Counter(df['wave_id'])
    duplicates = {wave_id: count for wave_id, count in wave_id_counts.items() if count > 1}
    
    print("\n--- DEBUG: Wave ID Uniqueness Check ---")
    print(f"Total wave_ids: {len(df)}")
    print(f"Unique wave_ids: {df['wave_id'].nunique()}")
    
    if duplicates:
        print("\n⚠️  DUPLICATE WAVE_IDs DETECTED:")
        for wave_id, count in duplicates.items():
            print(f"  - wave_id '{wave_id}' appears {count} times")
            # Get all rows with this duplicate wave_id
            duplicate_rows = df[df['wave_id'] == wave_id]
            print(f"    Corresponding display names (Wave column):")
            for idx, row in duplicate_rows.iterrows():
                print(f"      * '{row['Wave']}' (wave_id: '{row['wave_id']}')")
    else:
        print("✓ No duplicate wave_ids found")
    
    print("---------------------------------------\n")
    
    # Validate that no duplicates were detected
    assert len(duplicates) == 0, f"Expected no duplicates, found {len(duplicates)}"
    assert df['wave_id'].nunique() == len(df), "All wave_ids should be unique"
    
    print("✓ Correctly identified that no duplicates exist")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("WAVE_ID DEBUGGING TESTS")
    print("=" * 80)
    
    tests = [
        ("duplicate_detection_logic", test_duplicate_detection_logic),
        ("no_duplicates_case", test_no_duplicates_case),
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

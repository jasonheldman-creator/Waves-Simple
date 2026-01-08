#!/usr/bin/env python3
"""
Integration test for generate_live_snapshot_csv validation logic

This test simulates the validation logic without fetching real market data.
"""

import sys
import os
sys.path.insert(0, '/home/runner/work/Waves-Simple/Waves-Simple')

import pandas as pd
import numpy as np
from datetime import datetime


def test_validation_logic_simulation():
    """Simulate the validation logic from generate_live_snapshot_csv"""
    print("\n" + "=" * 80)
    print("SIMULATED VALIDATION TEST")
    print("=" * 80)
    
    # Simulate loading wave_weights.csv
    from analytics_truth import load_weights, expected_waves, _convert_wave_name_to_id
    
    weights_df = load_weights("wave_weights.csv")
    waves = expected_waves(weights_df)
    
    # Derive expected_wave_ids dynamically (as in the updated code)
    expected_wave_ids = sorted(set([_convert_wave_name_to_id(wave_name) for wave_name in waves]))
    expected_count = len(expected_wave_ids)
    
    print(f"✓ Expected count (dynamic): {expected_count}")
    print(f"✓ Expected wave_ids sample: {expected_wave_ids[:3]}")
    
    # Simulate creating DataFrame with wave_ids
    rows = []
    for wave_name in waves:
        wave_id_raw = _convert_wave_name_to_id(wave_name)
        
        # Normalize wave_id (as in updated code)
        if wave_id_raw is None or (isinstance(wave_id_raw, str) and not wave_id_raw.strip()):
            wave_id = _convert_wave_name_to_id(wave_name) if wave_name else 'unknown_wave'
        else:
            wave_id = wave_id_raw.strip() if isinstance(wave_id_raw, str) else str(wave_id_raw)
        
        rows.append({
            'wave_id': wave_id,
            'Wave': wave_name,
        })
    
    df = pd.DataFrame(rows)
    
    # Normalize wave_id column (as in updated code)
    df['wave_id'] = df['wave_id'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    # === VALIDATION LOGIC (copied from implementation) ===
    print("\n" + "=" * 80)
    print("WAVE_ID VALIDATION")
    print("=" * 80)
    
    # Count metrics for validation
    nunique_with_na = df['wave_id'].nunique(dropna=False)
    nunique_without_na = df['wave_id'].nunique(dropna=True)
    isna_sum = df['wave_id'].isna().sum()
    
    # Count blank wave_ids
    blank_sum = sum(1 for x in df['wave_id'] if isinstance(x, str) and not x.strip()) if not df['wave_id'].isna().all() else 0
    
    # Get duplicates
    wave_id_counts = df['wave_id'].value_counts()
    duplicates = wave_id_counts[wave_id_counts > 1]
    
    # Check for validation failures
    validation_passed = True
    error_messages = []
    
    # Check 1: nunique(dropna=False) should equal expected_count
    if nunique_with_na != expected_count:
        validation_passed = False
        error_messages.append(
            f"FAILED: nunique(dropna=False) = {nunique_with_na}, expected {expected_count}"
        )
    
    # Check 2: No null wave_ids
    if isna_sum > 0:
        validation_passed = False
        error_messages.append(
            f"FAILED: Found {isna_sum} null wave_id(s)"
        )
    
    # Check 3: No blank wave_ids
    if blank_sum > 0:
        validation_passed = False
        error_messages.append(
            f"FAILED: Found {blank_sum} blank wave_id(s) (empty after strip)"
        )
    
    # Check 4: No duplicates
    if len(duplicates) > 0:
        validation_passed = False
        duplicate_details = []
        for wave_id, count in duplicates.items():
            duplicate_rows = df[df['wave_id'] == wave_id]
            wave_names = duplicate_rows['Wave'].tolist()
            duplicate_details.append(
                f"  - wave_id '{wave_id}' appears {count} times in waves: {wave_names}"
            )
        error_messages.append(
            f"FAILED: Found {len(duplicates)} duplicate wave_id(s):\n" + "\n".join(duplicate_details)
        )
    
    # Report results
    if validation_passed:
        print(f"✓ wave_id validation PASSED")
        print(f"  - Expected count: {expected_count}")
        print(f"  - Unique wave_ids (dropna=False): {nunique_with_na}")
        print(f"  - Null wave_ids: {isna_sum}")
        print(f"  - Blank wave_ids: {blank_sum}")
        print(f"  - Duplicate wave_ids: {len(duplicates)}")
    else:
        print("✗ wave_id validation FAILED")
        for msg in error_messages:
            print(f"  {msg}")
    
    print("=" * 80 + "\n")
    
    return validation_passed


if __name__ == "__main__":
    success = test_validation_logic_simulation()
    if success:
        print("✓ INTEGRATION TEST PASSED")
    else:
        print("✗ INTEGRATION TEST FAILED")
    sys.exit(0 if success else 1)

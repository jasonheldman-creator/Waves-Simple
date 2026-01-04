#!/usr/bin/env python3
"""
End-to-End Test: Wave ID System Integration

Tests the complete wave_id system from waves_engine through app.py
to ensure all components work together correctly.
"""

import pandas as pd
import sys
from datetime import datetime

def test_wave_engine():
    """Test waves_engine.py wave_id functions."""
    print("=" * 70)
    print("TEST 1: Wave Engine Functions")
    print("=" * 70)
    
    from waves_engine import (
        get_all_wave_ids,
        get_wave_id_from_display_name,
        get_display_name_from_wave_id,
        validate_wave_id_registry,
        WAVE_ID_REGISTRY
    )
    
    # Test 1.1: Get all wave IDs
    wave_ids = get_all_wave_ids()
    print(f"\n1.1 Get all wave_ids: {len(wave_ids)} wave_ids found")
    print(f"    Examples: {wave_ids[:3]}")
    assert len(wave_ids) > 0, "No wave_ids found"
    
    # Test 1.2: Display name to wave_id conversion
    display_name = "S&P 500 Wave"
    wave_id = get_wave_id_from_display_name(display_name)
    print(f"\n1.2 Display name to wave_id:")
    print(f"    '{display_name}' -> '{wave_id}'")
    assert wave_id == "sp500_wave", f"Expected 'sp500_wave', got '{wave_id}'"
    
    # Test 1.3: Wave_id to display name conversion
    wave_id = "income_wave"
    display_name = get_display_name_from_wave_id(wave_id)
    print(f"\n1.3 Wave_id to display name:")
    print(f"    '{wave_id}' -> '{display_name}'")
    assert display_name == "Income Wave", f"Expected 'Income Wave', got '{display_name}'"
    
    # Test 1.4: Validate registry
    warnings = validate_wave_id_registry()
    print(f"\n1.4 Validate wave_id registry:")
    if warnings:
        print(f"    ⚠️  {len(warnings)} warnings found:")
        for w in warnings:
            print(f"       - {w}")
    else:
        print(f"    ✓ No warnings - registry is valid")
    
    # Test 1.5: Legacy alias support
    legacy_name = "Growth Wave"
    wave_id = get_wave_id_from_display_name(legacy_name)
    print(f"\n1.5 Legacy alias support:")
    print(f"    '{legacy_name}' -> '{wave_id}'")
    assert wave_id is not None, f"Legacy alias '{legacy_name}' not mapped"
    
    print("\n✅ Wave Engine Tests Passed\n")


def test_wave_history_csv():
    """Test wave_history.csv structure."""
    print("=" * 70)
    print("TEST 2: Wave History CSV Structure")
    print("=" * 70)
    
    # Test 2.1: Load CSV
    df = pd.read_csv('wave_history.csv')
    print(f"\n2.1 Load wave_history.csv: {len(df)} rows loaded")
    print(f"    Columns: {list(df.columns)}")
    
    # Test 2.2: Required columns
    required_cols = ['wave_id', 'display_name', 'date', 'portfolio_return', 'benchmark_return']
    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' not found"
    print(f"\n2.2 Required columns: ✓ All present")
    
    # Test 2.3: Wave_id values
    unique_wave_ids = df['wave_id'].unique()
    print(f"\n2.3 Unique wave_ids: {len(unique_wave_ids)}")
    print(f"    Examples: {list(unique_wave_ids[:3])}")
    
    # Test 2.4: No null wave_ids
    null_count = df['wave_id'].isna().sum()
    print(f"\n2.4 Null wave_ids: {null_count}")
    assert null_count == 0, f"Found {null_count} null wave_ids"
    
    # Test 2.5: Display names match wave_ids
    from waves_engine import get_display_name_from_wave_id
    mismatches = []
    for idx, row in df.head(10).iterrows():
        expected_display = get_display_name_from_wave_id(row['wave_id'])
        if expected_display and expected_display != row['display_name']:
            # Check if it's a legacy alias
            if row['display_name'] not in ['Growth Wave', 'Small-Mid Cap Growth Wave']:
                mismatches.append((row['wave_id'], row['display_name'], expected_display))
    
    print(f"\n2.5 Display name consistency: {len(mismatches)} mismatches in sample")
    if mismatches:
        for wid, actual, expected in mismatches[:3]:
            print(f"    - {wid}: '{actual}' != '{expected}'")
    
    print("\n✅ Wave History CSV Tests Passed\n")


def test_app_integration():
    """Test app.py integration with wave_id."""
    print("=" * 70)
    print("TEST 3: App.py Integration")
    print("=" * 70)
    
    from app import (
        safe_load_wave_history,
        get_wave_data_filtered,
        get_wave_universe_with_data
    )
    
    # Test 3.1: Load wave history via app.py
    print("\n3.1 Load wave history via app.py...")
    df = safe_load_wave_history()
    assert df is not None, "Failed to load wave history"
    assert 'wave_id' in df.columns, "wave_id column not found"
    print(f"    ✓ Loaded {len(df)} rows with wave_id column")
    
    # Test 3.2: Get waves with data
    print("\n3.2 Get waves with data...")
    waves = get_wave_universe_with_data(period_days=30)
    print(f"    ✓ Found {len(waves)} waves with data")
    print(f"    Examples: {waves[:3]}")
    
    # Test 3.3: Filter by display name (should use wave_id internally)
    print("\n3.3 Filter by display name...")
    for wave_name in waves[:2]:
        wave_data = get_wave_data_filtered(wave_name=wave_name, days=30)
        if wave_data is not None:
            assert 'wave_id' in wave_data.columns, f"wave_id column not in filtered data for {wave_name}"
            wave_id = wave_data['wave_id'].iloc[0]
            print(f"    ✓ '{wave_name}' -> wave_id: '{wave_id}' ({len(wave_data)} rows)")
        else:
            print(f"    ⚠️  No data for '{wave_name}'")
    
    print("\n✅ App Integration Tests Passed\n")


def test_backward_compatibility():
    """Test backward compatibility with legacy code."""
    print("=" * 70)
    print("TEST 4: Backward Compatibility")
    print("=" * 70)
    
    from waves_engine import get_all_waves
    from app import get_wave_data_filtered
    
    # Test 4.1: get_all_waves() still returns display names
    print("\n4.1 get_all_waves() returns display names...")
    waves = get_all_waves()
    print(f"    ✓ {len(waves)} wave names returned")
    print(f"    Examples: {waves[:3]}")
    assert all(' ' in w or '&' in w or '-' in w for w in waves[:5]), \
        "Wave names should be display names, not wave_ids"
    
    # Test 4.2: Can still filter by display name
    print("\n4.2 Filter by display name still works...")
    wave_data = get_wave_data_filtered(wave_name="Income Wave", days=10)
    if wave_data is not None:
        print(f"    ✓ 'Income Wave' filtering works ({len(wave_data)} rows)")
    else:
        print(f"    ⚠️  No data for 'Income Wave'")
    
    print("\n✅ Backward Compatibility Tests Passed\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("WAVE_ID SYSTEM END-TO-END TESTS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    try:
        test_wave_engine()
        test_wave_history_csv()
        test_app_integration()
        test_backward_compatibility()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

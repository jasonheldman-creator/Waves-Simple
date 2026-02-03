#!/usr/bin/env python3
"""
Test wave status filtering functionality.

Tests:
1. Wave registry has status field
2. Wave status can be filtered (ACTIVE vs STAGING)
3. Live snapshot includes wave_status
4. Filtering works correctly
"""

import sys
import os
import pandas as pd

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.wave_registry import (
    get_wave_registry,
    get_active_status_waves,
    check_wave_data_readiness,
    get_wave_by_id
)


def test_wave_registry_has_status():
    """Test that wave registry has status field."""
    print("\nTest 1: Wave registry has status field")
    registry = get_wave_registry()
    
    assert not registry.empty, "Registry should not be empty"
    assert 'status' in registry.columns, "Registry should have 'status' column"
    
    # Check that all rows have a status
    null_count = registry['status'].isna().sum()
    assert null_count == 0, f"All waves should have a status, found {null_count} nulls"
    
    # Check that status values are valid
    valid_statuses = {'ACTIVE', 'STAGING'}
    invalid_statuses = set(registry['status'].unique()) - valid_statuses
    assert len(invalid_statuses) == 0, f"Found invalid statuses: {invalid_statuses}"
    
    print(f"  ✓ Registry has {len(registry)} waves with status field")
    print(f"  ✓ Status distribution: {registry['status'].value_counts().to_dict()}")


def test_filtering_by_status():
    """Test filtering waves by status."""
    print("\nTest 2: Filtering by status")
    
    # Get all waves
    registry = get_wave_registry()
    total_waves = len(registry)
    
    # Get only ACTIVE waves
    active_waves = get_active_status_waves(include_staging=False)
    active_count = len(active_waves)
    
    # Get ACTIVE + STAGING waves
    all_waves = get_active_status_waves(include_staging=True)
    all_count = len(all_waves)
    
    # Verify counts
    assert active_count <= total_waves, "Active waves should be <= total waves"
    assert all_count <= total_waves, "All waves should be <= total waves"
    assert active_count <= all_count, "Active waves should be <= all waves"
    
    # Verify filtering works
    active_only_status = active_waves['status'].unique()
    assert len(active_only_status) == 1 and active_only_status[0] == 'ACTIVE', \
        f"Active-only filter should only return ACTIVE waves, got: {active_only_status}"
    
    print(f"  ✓ Total waves: {total_waves}")
    print(f"  ✓ ACTIVE waves only: {active_count}")
    print(f"  ✓ ACTIVE + STAGING waves: {all_count}")
    print(f"  ✓ STAGING waves: {all_count - active_count}")


def test_live_snapshot_has_wave_status():
    """Test that live_snapshot.csv has wave_status column."""
    print("\nTest 3: Live snapshot has wave_status column")
    
    snapshot_path = 'data/live_snapshot.csv'
    assert os.path.exists(snapshot_path), f"Snapshot file should exist at {snapshot_path}"
    
    snapshot = pd.read_csv(snapshot_path)
    
    assert not snapshot.empty, "Snapshot should not be empty"
    assert 'wave_status' in snapshot.columns, "Snapshot should have 'wave_status' column"
    
    # Check that all rows have a wave_status
    null_count = snapshot['wave_status'].isna().sum()
    assert null_count == 0, f"All snapshot rows should have wave_status, found {null_count} nulls"
    
    # Check that wave_status values are valid
    valid_statuses = {'ACTIVE', 'STAGING'}
    invalid_statuses = set(snapshot['wave_status'].unique()) - valid_statuses
    assert len(invalid_statuses) == 0, f"Found invalid wave_status: {invalid_statuses}"
    
    print(f"  ✓ Snapshot has {len(snapshot)} rows with wave_status field")
    print(f"  ✓ Wave status distribution: {snapshot['wave_status'].value_counts().to_dict()}")


def test_snapshot_filtering():
    """Test filtering snapshot by wave_status."""
    print("\nTest 4: Snapshot filtering by wave_status")
    
    snapshot = pd.read_csv('data/live_snapshot.csv')
    total_rows = len(snapshot)
    
    # Filter to ACTIVE only
    active_snapshot = snapshot[snapshot['wave_status'] == 'ACTIVE']
    active_count = len(active_snapshot)
    
    # Filter to STAGING only
    staging_snapshot = snapshot[snapshot['wave_status'] == 'STAGING']
    staging_count = len(staging_snapshot)
    
    # Verify counts
    assert active_count + staging_count == total_rows, \
        f"ACTIVE ({active_count}) + STAGING ({staging_count}) should equal total ({total_rows})"
    
    print(f"  ✓ Total snapshot rows: {total_rows}")
    print(f"  ✓ ACTIVE rows: {active_count}")
    print(f"  ✓ STAGING rows: {staging_count}")
    
    if staging_count > 0:
        print(f"  ✓ STAGING waves: {staging_snapshot['wave'].tolist()}")


def test_wave_data_readiness_check():
    """Test wave data readiness checking."""
    print("\nTest 5: Wave data readiness check")
    
    registry = get_wave_registry()
    
    # Test a few waves
    test_waves = registry['wave_id'].head(5).tolist()
    
    for wave_id in test_waves:
        is_ready = check_wave_data_readiness(wave_id)
        wave_info = get_wave_by_id(wave_id)
        wave_name = wave_info.get('wave_name', wave_id) if wave_info else wave_id
        status = wave_info.get('status', 'UNKNOWN') if wave_info else 'UNKNOWN'
        
        print(f"  - {wave_name}: ready={is_ready}, status={status}")
        
        # If status is ACTIVE, readiness should ideally be True (though this can vary)
        # If status is STAGING, readiness should be False
        if status == 'STAGING':
            assert is_ready == False, f"STAGING wave {wave_name} should not be ready"
    
    print("  ✓ Readiness check function works")


def main():
    """Run all tests."""
    print("=" * 80)
    print("WAVE STATUS FILTERING TESTS")
    print("=" * 80)
    
    try:
        test_wave_registry_has_status()
        test_filtering_by_status()
        test_live_snapshot_has_wave_status()
        test_snapshot_filtering()
        test_wave_data_readiness_check()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Test script to validate get_all_waves_universe() and diagnostic functions.

This script tests:
1. get_all_waves_universe() returns all waves from registry (dynamic count)
2. get_wave_readiness_diagnostic_summary() provides root cause visibility
3. Wave universe is correctly sourced from the registry
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_get_all_waves_universe():
    """Test that get_all_waves_universe() returns complete wave universe."""
    print("=" * 80)
    print("TEST: get_all_waves_universe() - Canonical Registry Source")
    print("=" * 80)
    
    from waves_engine import get_all_waves_universe, WAVE_ID_REGISTRY
    
    # Get wave universe
    universe = get_all_waves_universe()
    
    # Dynamic expected count from registry
    expected_count = len(WAVE_ID_REGISTRY)
    
    # Validate structure
    assert 'waves' in universe, "Missing 'waves' key"
    assert 'wave_ids' in universe, "Missing 'wave_ids' key"
    assert 'count' in universe, "Missing 'count' key"
    assert 'source' in universe, "Missing 'source' key"
    assert 'version' in universe, "Missing 'version' key"
    
    print(f"✓ Universe structure is valid")
    
    # Validate count (dynamic)
    assert universe['count'] == expected_count, f"Expected {expected_count} waves, got {universe['count']}"
    print(f"✓ Universe contains exactly {expected_count} waves (from WAVE_ID_REGISTRY)")
    
    # Validate source
    assert universe['source'] == 'wave_registry', f"Expected source 'wave_registry', got '{universe['source']}'"
    print(f"✓ Wave universe sourced from: {universe['source']}")
    
    # Validate wave names (dynamic)
    assert len(universe['waves']) == expected_count, f"Expected {expected_count} wave names, got {len(universe['waves'])}"
    print(f"✓ All {expected_count} wave display names present")
    
    # Validate wave IDs (dynamic)
    assert len(universe['wave_ids']) == expected_count, f"Expected {expected_count} wave IDs, got {len(universe['wave_ids'])}"
    print(f"✓ All {expected_count} wave IDs present")
    
    # Show sample
    print(f"\nSample wave names (first 5):")
    for wave in sorted(universe['waves'])[:5]:
        print(f"  - {wave}")
    
    print(f"\nSample wave IDs (first 5):")
    for wave_id in sorted(universe['wave_ids'])[:5]:
        print(f"  - {wave_id}")
    
    print(f"\n✓ get_all_waves_universe() test PASSED!")
    return True


def test_diagnostic_summary():
    """Test that get_wave_readiness_diagnostic_summary() provides root cause visibility."""
    print("\n" + "=" * 80)
    print("TEST: get_wave_readiness_diagnostic_summary() - Root Cause Visibility")
    print("=" * 80)
    
    from analytics_pipeline import get_wave_readiness_diagnostic_summary
    from waves_engine import WAVE_ID_REGISTRY
    
    # Get diagnostic summary
    diag = get_wave_readiness_diagnostic_summary()
    
    # Dynamic expected count from registry
    expected_count = len(WAVE_ID_REGISTRY)
    
    # Validate structure
    required_keys = [
        'total_waves_in_registry',
        'total_waves_rendered',
        'readiness_by_status',
        'unavailable_waves_detail',
        'wave_universe_source',
        'is_complete',
        'warnings'
    ]
    
    for key in required_keys:
        assert key in diag, f"Missing required key: {key}"
    
    print(f"✓ Diagnostic summary structure is valid")
    
    # Validate registry count (dynamic)
    assert diag['total_waves_in_registry'] == expected_count, \
        f"Expected {expected_count} waves in registry, got {diag['total_waves_in_registry']}"
    print(f"✓ Total waves in registry: {diag['total_waves_in_registry']}")
    
    # Validate all waves rendered (dynamic)
    assert diag['total_waves_rendered'] == expected_count, \
        f"Expected {expected_count} waves rendered, got {diag['total_waves_rendered']} (NO SILENT EXCLUSIONS!)"
    print(f"✓ Total waves rendered: {diag['total_waves_rendered']} (NO SILENT EXCLUSIONS)")
    
    # Validate source
    assert diag['wave_universe_source'] == 'wave_registry', \
        f"Expected source 'wave_registry', got '{diag['wave_universe_source']}'"
    print(f"✓ Wave universe sourced from: {diag['wave_universe_source']}")
    
    # Validate completeness (dynamic)
    assert diag['is_complete'], f"Wave universe should be complete ({expected_count} waves)"
    print(f"✓ Wave universe is complete: {diag['is_complete']}")
    
    # Show readiness breakdown
    print(f"\nGraded Readiness Breakdown:")
    for status, count in diag['readiness_by_status'].items():
        percentage = (count / diag['total_waves_in_registry'] * 100) if diag['total_waves_in_registry'] > 0 else 0
        print(f"  - {status.capitalize()}: {count} waves ({percentage:.1f}%)")
    
    # Show usable count
    usable = sum([
        diag['readiness_by_status'].get('full', 0),
        diag['readiness_by_status'].get('partial', 0),
        diag['readiness_by_status'].get('operational', 0)
    ])
    usable_pct = (usable / diag['total_waves_in_registry'] * 100) if diag['total_waves_in_registry'] > 0 else 0
    print(f"\n  Usable Waves (operational or better): {usable} ({usable_pct:.1f}%)")
    
    # Show warnings if any
    if diag['warnings']:
        print(f"\n⚠ Warnings ({len(diag['warnings'])}):")
        for warning in diag['warnings']:
            print(f"  - {warning}")
    else:
        print(f"\n✓ No warnings")
    
    # Show sample of unavailable waves with blocking reasons
    if diag['unavailable_waves_detail']:
        print(f"\nUnavailable Waves with Root Cause (sample of {min(3, len(diag['unavailable_waves_detail']))}):")
        for wave_detail in diag['unavailable_waves_detail'][:3]:
            print(f"  • {wave_detail['display_name']}")
            print(f"    - Coverage: {wave_detail['coverage_pct']:.1f}%")
            print(f"    - Blocking: {', '.join(wave_detail['top_blocking_reasons'])}")
            if wave_detail['suggested_actions']:
                print(f"    - Action: {wave_detail['suggested_actions'][0]}")
    
    print(f"\n✓ get_wave_readiness_diagnostic_summary() test PASSED!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("WAVE UNIVERSE & DIAGNOSTICS TEST SUITE")
    print("Testing new functions for 28-wave visibility")
    print("=" * 80)
    
    try:
        # Test 1: get_all_waves_universe()
        test_get_all_waves_universe()
        
        # Test 2: get_wave_readiness_diagnostic_summary()
        test_diagnostic_summary()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED - Wave Universe Functions Working Correctly!")
        print("=" * 80)
        print("\nKey Findings:")
        print("  • All 28 waves from registry are accessible")
        print("  • Wave universe correctly sourced from 'wave_registry'")
        print("  • Diagnostic summary provides root cause visibility")
        print("  • NO SILENT EXCLUSIONS - All waves are rendered")
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

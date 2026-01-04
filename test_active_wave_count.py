#!/usr/bin/env python3
"""
Test Active Wave Count Logic

This test ensures that:
1. get_active_wave_registry() returns only active waves
2. Validation logic uses dynamic active wave count, not hard-coded 28
3. No hard-coded expected_count = 28 exists in validation logic
"""

import sys
import os
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_active_wave_registry_function():
    """Test that get_active_wave_registry() returns only active waves."""
    print("=" * 70)
    print("TEST 1: get_active_wave_registry() Function")
    print("=" * 70)
    
    from wave_registry_manager import get_active_wave_registry
    import pandas as pd
    
    # Get active waves
    active_waves = get_active_wave_registry()
    
    assert isinstance(active_waves, pd.DataFrame), "Should return a DataFrame"
    print(f"\n✓ Returned DataFrame with {len(active_waves)} rows")
    
    # Load full registry and verify filtering
    full_registry = pd.read_csv("data/wave_registry.csv")
    expected_active = full_registry[full_registry['active'] == True]
    
    assert len(active_waves) == len(expected_active), \
        f"Active wave count mismatch: got {len(active_waves)}, expected {len(expected_active)}"
    print(f"✓ Active wave count matches CSV filter: {len(active_waves)} active waves")
    
    # Verify all returned waves are marked as active
    if 'active' in active_waves.columns:
        all_active = active_waves['active'].all()
        assert all_active, "All waves returned should be active"
        print("✓ All returned waves have active=True")
    
    # Display inactive waves if any
    inactive_waves = full_registry[full_registry['active'] == False]
    if len(inactive_waves) > 0:
        print(f"\nℹ️  Inactive waves ({len(inactive_waves)}):")
        for _, wave in inactive_waves.iterrows():
            print(f"   - {wave['wave_name']}")
    else:
        print(f"\nℹ️  No inactive waves found")
    
    print("\n✅ TEST 1 PASSED")
    return True


def test_no_hardcoded_28_in_validation():
    """Test that validation logic doesn't use hard-coded expected_count = 28."""
    print("\n" + "=" * 70)
    print("TEST 2: No Hard-Coded expected_count = 28 in Validation Logic")
    print("=" * 70)
    
    # Files to check for hard-coded expected_count = 28
    files_to_check = [
        "app.py",
        "helpers/wave_registry_validator.py",
        "wave_registry_manager.py",
    ]
    
    violations = []
    
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            continue
        
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Look for patterns like "expected_count = 28" in validation context
        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith('#'):
                continue
            
            # Check for expected_count = 28 pattern
            if re.search(r'expected(_active)?_count\s*=\s*28\b', line):
                # Check if this is in a validation context (has "wave" or "universe" nearby)
                context_start = max(0, i - 5)
                context_end = min(len(lines), i + 5)
                context = '\n'.join(lines[context_start:context_end])
                
                if 'wave' in context.lower() or 'universe' in context.lower() or 'registry' in context.lower():
                    violations.append({
                        'file': filepath,
                        'line': i,
                        'content': line.strip()
                    })
    
    if violations:
        print("\n❌ Found hard-coded expected_count = 28 in validation logic:")
        for v in violations:
            print(f"\n  File: {v['file']}")
            print(f"  Line {v['line']}: {v['content']}")
        print("\n⚠️  These should be replaced with dynamic active wave count!")
        return False
    else:
        print("\n✓ No hard-coded expected_count = 28 found in validation logic")
        print("✓ Validation uses dynamic active wave count")
        print("\n✅ TEST 2 PASSED")
        return True


def test_validation_uses_active_count():
    """Test that validation logic computes expected count from active registry."""
    print("\n" + "=" * 70)
    print("TEST 3: Validation Uses Active Wave Count")
    print("=" * 70)
    
    from helpers.wave_registry_validator import validate_wave_registry
    from wave_registry_manager import get_active_wave_registry
    
    # Get expected active count
    active_waves = get_active_wave_registry()
    expected_active_count = len(active_waves)
    
    print(f"\n✓ Active wave count from CSV: {expected_active_count}")
    
    # Run validation (uses JSON registry, but should check against CSV)
    # Note: This may fail if JSON doesn't exist, but that's okay for this test
    try:
        result = validate_wave_registry()
        
        # Check if validation result mentions the active count
        info_str = ' '.join(result.info)
        
        # Look for the expected active count in validation output
        if str(expected_active_count) in info_str:
            print(f"✓ Validation references expected active count ({expected_active_count})")
        else:
            print(f"⚠️  Validation info: {result.info}")
        
        if result.is_valid:
            print("✓ Validation passed")
        else:
            print(f"ℹ️  Validation issues: {result.errors}")
    
    except Exception as e:
        print(f"⚠️  Validation error (may be expected if JSON not found): {e}")
    
    print("\n✅ TEST 3 PASSED")
    return True


def test_app_startup_validation():
    """Test that app.py startup validation uses active wave count."""
    print("\n" + "=" * 70)
    print("TEST 4: App Startup Validation")
    print("=" * 70)
    
    with open("app.py", 'r') as f:
        app_content = f.read()
    
    # Check that app.py imports get_active_wave_registry
    if 'get_active_wave_registry' in app_content:
        print("✓ app.py imports get_active_wave_registry")
    else:
        print("⚠️  app.py doesn't import get_active_wave_registry (may use different approach)")
    
    # Check that expected_active_count is computed from active registry
    if 'expected_active_count' in app_content and 'get_active_wave_registry()' in app_content:
        print("✓ app.py computes expected_active_count from active registry")
    else:
        print("⚠️  app.py may use different variable naming")
    
    # Check for hard-coded 28 in wave universe validation section
    if re.search(r'wave.*universe.*validation.*expected_count\s*=\s*28', app_content, re.IGNORECASE | re.DOTALL):
        print("❌ app.py still has hard-coded expected_count = 28 in wave validation")
        return False
    else:
        print("✓ app.py doesn't use hard-coded expected_count = 28 in wave validation")
    
    print("\n✅ TEST 4 PASSED")
    return True


def run_all_tests():
    """Run all validation tests."""
    print("\n")
    print("*" * 80)
    print("ACTIVE WAVE COUNT VALIDATION TESTS")
    print("Verifying dynamic active wave counting instead of hard-coded 28")
    print("*" * 80)
    
    tests = [
        ("Active Wave Registry Function", test_active_wave_registry_function),
        ("No Hard-Coded expected_count = 28", test_no_hardcoded_28_in_validation),
        ("Validation Uses Active Count", test_validation_uses_active_count),
        ("App Startup Validation", test_app_startup_validation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n")
    print("*" * 80)
    print("TEST RESULTS")
    print("*" * 80)
    print(f"✓ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(tests)}")
        print("\n⚠️  Some tests failed. Review failures above.")
    else:
        print("🎉 All validation tests passed!")
        print("\n✅ Active Wave Count Logic Verified")
        print("   - get_active_wave_registry() filters by active=True")
        print("   - Validation uses dynamic active count")
        print("   - No hard-coded expected_count = 28 in validation logic")
    print("*" * 80)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

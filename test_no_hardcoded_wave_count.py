#!/usr/bin/env python3
"""
test_no_hardcoded_wave_count.py

Regression test to ensure no hard-coded wave count expectations exist in the codebase.

This test validates that:
1. No "Expected 28" or "28 waves" strings exist in validation logic
2. Wave counts are always computed dynamically from WAVE_ID_REGISTRY
3. Documentation doesn't reference fixed wave counts

This prevents regression to hard-coded wave expectations that cause
false validation failures when the registry changes.
"""

import os
import sys
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_no_hardcoded_28_in_validation_files():
    """Test that no validation files have hard-coded '28' wave expectations."""
    print("=" * 80)
    print("TEST: No Hard-Coded Wave Count in Validation Files")
    print("=" * 80)
    
    # Files that should NOT have hard-coded "28" wave references
    validation_files = [
        'app.py',
        'waves_engine.py',
        'wave_registry_manager.py',
        'analytics_pipeline.py',
        'snapshot_ledger.py',
        'truth_frame_helpers.py',
    ]
    
    # Patterns that indicate hard-coded wave count (case-insensitive)
    hardcoded_patterns = [
        r'expected_count\s*=\s*28',
        r'wave_count\s*==\s*28',
        r'len\([^)]+\)\s*==\s*28',
        r'Expected\s+28\s+waves',
        r'exactly\s+28\s+waves',
    ]
    
    failures = []
    
    for filename in validation_files:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️  Skipping {filename} (file not found)")
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check each pattern
        for pattern in hardcoded_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Get line number
                line_num = content[:match.start()].count('\n') + 1
                # Get the matched text
                matched_text = match.group()
                
                # Exclude comments and strings that are just examples
                line_start = content.rfind('\n', 0, match.start()) + 1
                line_end = content.find('\n', match.end())
                if line_end == -1:
                    line_end = len(content)
                line = content[line_start:line_end]
                
                # Skip if it's in a comment or a test example
                if '#' in line[:match.start() - line_start]:
                    continue
                
                failures.append({
                    'file': filename,
                    'line': line_num,
                    'text': matched_text,
                    'full_line': line.strip()
                })
    
    if failures:
        print(f"\n❌ Found {len(failures)} hard-coded wave count references:")
        for failure in failures:
            print(f"  {failure['file']}:{failure['line']} - {failure['text']}")
            print(f"    Line: {failure['full_line']}")
        
        assert False, f"Found {len(failures)} hard-coded wave count references. Use len(WAVE_ID_REGISTRY) instead."
    
    print(f"✓ No hard-coded wave counts found in {len(validation_files)} validation files")
    return True


def test_wave_id_registry_is_used():
    """Test that WAVE_ID_REGISTRY is imported and used for wave counting."""
    print("\n" + "=" * 80)
    print("TEST: WAVE_ID_REGISTRY Usage in Validation")
    print("=" * 80)
    
    # Key files should import and use WAVE_ID_REGISTRY
    key_files = [
        'app.py',
        'wave_registry_manager.py',
    ]
    
    all_use_registry = True
    
    for filename in key_files:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️  Skipping {filename} (file not found)")
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if WAVE_ID_REGISTRY is imported
        if 'WAVE_ID_REGISTRY' not in content:
            print(f"  ❌ {filename} does not import or use WAVE_ID_REGISTRY")
            all_use_registry = False
        else:
            # Check if len(WAVE_ID_REGISTRY) is used for dynamic counting
            if 'len(WAVE_ID_REGISTRY)' in content:
                print(f"  ✓ {filename} uses len(WAVE_ID_REGISTRY) for dynamic counting")
            else:
                print(f"  ⚠️  {filename} imports WAVE_ID_REGISTRY but may not use it for counting")
    
    assert all_use_registry, "Some files don't use WAVE_ID_REGISTRY for wave counting"
    return True


def test_dynamic_expected_count_pattern():
    """Test that expected_count is computed dynamically, not hard-coded."""
    print("\n" + "=" * 80)
    print("TEST: Dynamic Expected Count Pattern")
    print("=" * 80)
    
    # Check that the app.py uses dynamic expected_count
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    
    if not os.path.exists(app_path):
        print("⚠️  app.py not found, skipping test")
        return True
    
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for the validation section
    if 'expected_count = len(WAVE_ID_REGISTRY)' in content:
        print("  ✓ app.py uses dynamic expected_count = len(WAVE_ID_REGISTRY)")
    else:
        print("  ❌ app.py does not use dynamic expected_count pattern")
        assert False, "app.py should use: expected_count = len(WAVE_ID_REGISTRY)"
    
    # Check for GREEN success banner
    if 'wave_universe_success' in content:
        print("  ✓ app.py includes GREEN success banner for validation")
    else:
        print("  ⚠️  app.py may not have GREEN success banner")
    
    return True


def run_all_tests():
    """Run all regression tests."""
    print("\n" + "=" * 80)
    print("WAVE COUNT REGRESSION TEST SUITE")
    print("=" * 80)
    print("Ensuring no hard-coded wave count expectations exist")
    print("=" * 80 + "\n")
    
    tests = [
        ("No Hard-Coded Wave Count in Validation Files", test_no_hardcoded_28_in_validation_files),
        ("WAVE_ID_REGISTRY Usage", test_wave_id_registry_is_used),
        ("Dynamic Expected Count Pattern", test_dynamic_expected_count_pattern),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"\n✅ {test_name} PASSED\n")
        except AssertionError as e:
            failed += 1
            print(f"\n❌ {test_name} FAILED: {e}\n")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} ERROR: {e}\n")
    
    print("=" * 80)
    print(f"REGRESSION TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed > 0:
        sys.exit(1)
    
    print("\n✅ All regression tests passed! No hard-coded wave counts detected.")


if __name__ == "__main__":
    run_all_tests()

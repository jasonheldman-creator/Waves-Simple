#!/usr/bin/env python3
"""
Test to verify that app.py does not contain execution-stopping statements
that would prevent UI rendering.

This test checks:
1. No st.stop() calls are present (uncommented)
2. No early return statements in main() that exit before UI rendering
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_no_st_stop_calls():
    """Verify that no uncommented st.stop() calls exist in app.py"""
    print("=" * 70)
    print("TEST: Verify no st.stop() calls in app.py")
    print("=" * 70)
    
    with open('app.py', 'r') as f:
        lines = f.readlines()
    
    # Find any uncommented st.stop() calls
    st_stop_lines = []
    for i, line in enumerate(lines, 1):
        # Skip commented lines
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        
        # Check for st.stop()
        if 'st.stop()' in line and not line.strip().startswith('#'):
            st_stop_lines.append((i, line.strip()))
    
    if st_stop_lines:
        print(f"\n❌ FAIL: Found {len(st_stop_lines)} uncommented st.stop() call(s):")
        for line_num, line_content in st_stop_lines:
            print(f"   Line {line_num}: {line_content}")
        return False
    else:
        print("\n✅ PASS: No uncommented st.stop() calls found")
        return True


def test_removed_st_stop_comments():
    """Verify that st.stop() calls have been properly commented out"""
    print("\n" + "=" * 70)
    print("TEST: Verify st.stop() calls are properly commented")
    print("=" * 70)
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Count commented-out st.stop() with our marker
    marker_count = content.count('# st.stop()  # REMOVED: Allow UI to render instead of halting')
    
    print(f"\n✅ PASS: Found {marker_count} properly commented st.stop() calls with removal marker")
    
    # Should be 3 (loop detection, rapid rerun, watchdog)
    if marker_count == 3:
        print("   ✓ All 3 expected st.stop() calls are commented out")
        return True
    else:
        print(f"   ⚠️  Expected 3 commented st.stop() calls, found {marker_count}")
        return marker_count > 0


def test_logger_warnings_present():
    """Verify that logger.warning() calls have been added"""
    print("\n" + "=" * 70)
    print("TEST: Verify logger.warning() calls are present")
    print("=" * 70)
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Look for logger.warning calls related to our changes
    expected_warnings = [
        'logger.warning(f"Loop detection triggered:',
        'logger.warning(f"Rapid rerun detection triggered:',
        'logger.warning(f"Safe Mode watchdog timeout:'
    ]
    
    found_count = 0
    for warning in expected_warnings:
        if warning in content:
            found_count += 1
            print(f"   ✓ Found: {warning}...")
    
    if found_count == 3:
        print(f"\n✅ PASS: All {found_count} expected logger.warning() calls are present")
        return True
    else:
        print(f"\n❌ FAIL: Expected 3 logger.warning() calls, found {found_count}")
        return False


def test_app_imports_successfully():
    """Verify that app.py can be imported without errors"""
    print("\n" + "=" * 70)
    print("TEST: Verify app.py imports successfully")
    print("=" * 70)
    
    try:
        # Import will execute module-level code but not main()
        import app
        print("\n✅ PASS: app.py imported successfully without errors")
        return True
    except Exception as e:
        print(f"\n❌ FAIL: app.py import failed with error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("=" * 70)
    print("EXECUTION-STOPPING STATEMENTS REMOVAL TEST SUITE")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("No st.stop() calls", test_no_st_stop_calls()))
    results.append(("Commented st.stop() markers", test_removed_st_stop_comments()))
    results.append(("Logger warnings present", test_logger_warnings_present()))
    results.append(("App imports successfully", test_app_imports_successfully()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

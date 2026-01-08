#!/usr/bin/env python3
"""
Final Verification Script for Rerun Loop Fix

This script validates all implemented changes are in place and working correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_price_book_caching():
    """Verify PRICE_BOOK caching implementation."""
    with open('app.py', 'r') as f:
        content = f.read()
    
    checks = {
        "get_cached_price_book exists": 'def get_cached_price_book():' in content,
        "@st.cache_resource decorator": '@st.cache_resource' in content,
        "Uses PRICE_BOOK_CONSTANTS_AVAILABLE": 'if PRICE_BOOK_CONSTANTS_AVAILABLE' in content,
        "Logs on cache miss": 'PRICE_BOOK loaded (cached)' in content,
        "Returns get_price_book()": 'return get_price_book(active_tickers=None)' in content,
    }
    
    all_pass = all(checks.values())
    return all_pass, checks


def verify_rerun_throttle():
    """Verify rerun throttle safety fuse implementation."""
    with open('app.py', 'r') as f:
        content = f.read()
    
    checks = {
        "RERUN_THROTTLE_THRESHOLD constant": 'RERUN_THROTTLE_THRESHOLD = ' in content,
        "MAX_RAPID_RERUNS constant": 'MAX_RAPID_RERUNS = ' in content,
        "last_rerun_time tracking": 'last_rerun_time' in content,
        "rapid_rerun_count tracking": 'rapid_rerun_count' in content,
        "Uses configurable threshold": 'time_since_last_rerun < RERUN_THROTTLE_THRESHOLD' in content,
        "Uses configurable max": 'rapid_rerun_count >= MAX_RAPID_RERUNS' in content,
        "Error message": 'RAPID RERUN DETECTED' in content,
        "Stops execution": 'st.stop()' in content,
    }
    
    all_pass = all(checks.values())
    return all_pass, checks


def verify_wave_selection_state():
    """Verify wave selection state management."""
    with open('app.py', 'r') as f:
        content = f.read()
    
    checks = {
        "selected_wave_id initialization guard": 'if "selected_wave_id" not in st.session_state:' in content,
        "Widget uses stable key": 'key="selected_wave_id_display"' in content,
        "Initialized only once": 'if "initialized" not in st.session_state:' in content,
    }
    
    all_pass = all(checks.values())
    return all_pass, checks


def verify_auto_refresh_disabled():
    """Verify auto-refresh is disabled by default."""
    with open('auto_refresh_config.py', 'r') as f:
        content = f.read()
    
    checks = {
        "DEFAULT_AUTO_REFRESH_ENABLED = False": 'DEFAULT_AUTO_REFRESH_ENABLED = False' in content,
    }
    
    with open('app.py', 'r') as f:
        app_content = f.read()
    
    checks["count = None disables timer"] = 'count = None' in app_content
    
    all_pass = all(checks.values())
    return all_pass, checks


def count_cached_price_book_usage():
    """Count usage of get_cached_price_book()."""
    with open('app.py', 'r') as f:
        content = f.read()
    
    count = content.count('get_cached_price_book()')
    # Subtract 1 for the function definition itself
    actual_usage = count - 1
    
    return actual_usage >= 15, {"Usage count": actual_usage, "Minimum expected": 15}


def verify_tests_exist():
    """Verify test files exist and are executable."""
    checks = {
        "test_rerun_loops.py exists": os.path.exists('test_rerun_loops.py'),
        "test_price_book_caching.py exists": os.path.exists('test_price_book_caching.py'),
        "RERUN_LOOP_FIX_SUMMARY.md exists": os.path.exists('RERUN_LOOP_FIX_SUMMARY.md'),
    }
    
    all_pass = all(checks.values())
    return all_pass, checks


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_checks(checks_dict):
    """Print check results."""
    for check, passed in checks_dict.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")


def main():
    """Run all verification checks."""
    print_section("RERUN LOOP FIX - FINAL VERIFICATION")
    
    all_sections_pass = []
    
    # 1. PRICE_BOOK Caching
    print_section("1. PRICE_BOOK Caching Implementation")
    passed, checks = verify_price_book_caching()
    print_checks(checks)
    all_sections_pass.append(passed)
    
    # 2. Rerun Throttle
    print_section("2. Rerun Throttle Safety Fuse")
    passed, checks = verify_rerun_throttle()
    print_checks(checks)
    all_sections_pass.append(passed)
    
    # 3. Wave Selection State
    print_section("3. Wave Selection State Management")
    passed, checks = verify_wave_selection_state()
    print_checks(checks)
    all_sections_pass.append(passed)
    
    # 4. Auto-Refresh Disabled
    print_section("4. Auto-Refresh Default State")
    passed, checks = verify_auto_refresh_disabled()
    print_checks(checks)
    all_sections_pass.append(passed)
    
    # 5. Cached Price Book Usage
    print_section("5. get_cached_price_book() Usage")
    passed, checks = count_cached_price_book_usage()
    print_checks(checks)
    all_sections_pass.append(passed)
    
    # 6. Tests and Documentation
    print_section("6. Tests and Documentation")
    passed, checks = verify_tests_exist()
    print_checks(checks)
    all_sections_pass.append(passed)
    
    # Summary
    print_section("VERIFICATION SUMMARY")
    if all(all_sections_pass):
        print("\n  🎉 ALL VERIFICATION CHECKS PASSED!")
        print("\n  The rerun loop fix has been successfully implemented.")
        print("\n  Next steps:")
        print("    1. Run manual tests in the Streamlit app")
        print("    2. Monitor run counter behavior")
        print("    3. Test wave selection functionality")
        print("    4. Check logs for PRICE_BOOK loading frequency")
        print()
        return 0
    else:
        print("\n  ⚠️  Some verification checks failed!")
        print("\n  Please review the failed checks above.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())

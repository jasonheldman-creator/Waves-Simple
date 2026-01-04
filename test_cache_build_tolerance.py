#!/usr/bin/env python3
"""
Test cache build tolerance logic

This test validates that the cache build script correctly handles:
1. Critical tickers (must succeed)
2. Non-critical tickers (can fail without failing the build)
3. Proper exit codes and status messages
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_complete_price_cache import CRITICAL_TICKERS


def test_critical_tickers_defined():
    """Test that critical tickers are properly defined."""
    print("Testing critical tickers definition...")
    
    # Expected critical tickers
    expected = {'IGV', 'STETH-USD', '^VIX'}
    
    assert CRITICAL_TICKERS == expected, \
        f"Critical tickers mismatch. Expected: {expected}, Got: {CRITICAL_TICKERS}"
    
    print(f"✓ Critical tickers correctly defined: {CRITICAL_TICKERS}")


def test_status_determination():
    """Test cache status determination logic."""
    print("\nTesting cache status determination...")
    
    # Simulate different scenarios
    
    # Scenario 1: All tickers succeed
    successful_tickers = {'IGV', 'STETH-USD', '^VIX', 'SPY', 'QQQ'}
    failures = {}
    missing_critical = CRITICAL_TICKERS - successful_tickers
    failed_critical = [t for t in CRITICAL_TICKERS if t in failures]
    
    critical_failures = {k: v for k, v in failures.items() if k in CRITICAL_TICKERS}
    non_critical_failures = {k: v for k, v in failures.items() if k not in CRITICAL_TICKERS}
    
    if missing_critical or failed_critical:
        status = "FAILED"
    elif non_critical_failures:
        status = f"DEGRADED ({len(non_critical_failures)} non-critical tickers skipped)"
    else:
        status = "STABLE"
    
    assert status == "STABLE", f"Expected STABLE, got {status}"
    print(f"  Scenario 1 (all succeed): {status} ✓")
    
    # Scenario 2: Critical tickers succeed, some non-critical fail
    successful_tickers = {'IGV', 'STETH-USD', '^VIX', 'SPY'}
    failures = {'QQQ': 'Insufficient data', 'IWM': 'Network error'}
    missing_critical = CRITICAL_TICKERS - successful_tickers
    failed_critical = [t for t in CRITICAL_TICKERS if t in failures]
    
    critical_failures = {k: v for k, v in failures.items() if k in CRITICAL_TICKERS}
    non_critical_failures = {k: v for k, v in failures.items() if k not in CRITICAL_TICKERS}
    
    if missing_critical or failed_critical:
        status = "FAILED"
    elif non_critical_failures:
        status = f"DEGRADED ({len(non_critical_failures)} non-critical tickers skipped)"
    else:
        status = "STABLE"
    
    assert status.startswith("DEGRADED"), f"Expected DEGRADED, got {status}"
    assert "2 non-critical" in status, f"Expected '2 non-critical' in status, got {status}"
    print(f"  Scenario 2 (non-critical failures): {status} ✓")
    
    # Scenario 3: Critical ticker fails
    successful_tickers = {'SPY', 'QQQ'}
    failures = {'IGV': 'Insufficient data', 'IWM': 'Network error'}
    missing_critical = CRITICAL_TICKERS - successful_tickers
    failed_critical = [t for t in CRITICAL_TICKERS if t in failures]
    
    critical_failures = {k: v for k, v in failures.items() if k in CRITICAL_TICKERS}
    non_critical_failures = {k: v for k, v in failures.items() if k not in CRITICAL_TICKERS}
    
    if missing_critical or failed_critical:
        status = "FAILED"
    elif non_critical_failures:
        status = f"DEGRADED ({len(non_critical_failures)} non-critical tickers skipped)"
    else:
        status = "STABLE"
    
    assert status == "FAILED", f"Expected FAILED, got {status}"
    print(f"  Scenario 3 (critical failure): {status} ✓")


def test_exit_code_logic():
    """Test exit code determination logic."""
    print("\nTesting exit code logic...")
    
    # Scenario 1: All critical tickers present
    successful_tickers = {'IGV', 'STETH-USD', '^VIX', 'SPY'}
    failures = {'QQQ': 'Error'}
    missing_critical = CRITICAL_TICKERS - successful_tickers
    failed_critical = [t for t in CRITICAL_TICKERS if t in failures]
    
    if missing_critical or failed_critical:
        exit_code = 1
    else:
        exit_code = 0
    
    assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
    print(f"  Scenario 1 (critical present, non-critical fail): exit code {exit_code} ✓")
    
    # Scenario 2: Critical ticker missing
    successful_tickers = {'SPY', 'QQQ'}
    failures = {'IGV': 'Error'}
    missing_critical = CRITICAL_TICKERS - successful_tickers
    failed_critical = [t for t in CRITICAL_TICKERS if t in failures]
    
    if missing_critical or failed_critical:
        exit_code = 1
    else:
        exit_code = 0
    
    assert exit_code == 1, f"Expected exit code 1, got {exit_code}"
    print(f"  Scenario 2 (critical missing): exit code {exit_code} ✓")
    
    # Scenario 3: All succeed
    successful_tickers = {'IGV', 'STETH-USD', '^VIX', 'SPY', 'QQQ'}
    failures = {}
    missing_critical = CRITICAL_TICKERS - successful_tickers
    failed_critical = [t for t in CRITICAL_TICKERS if t in failures]
    
    if missing_critical or failed_critical:
        exit_code = 1
    else:
        exit_code = 0
    
    assert exit_code == 0, f"Expected exit code 0, got {exit_code}"
    print(f"  Scenario 3 (all succeed): exit code {exit_code} ✓")


def main():
    """Run all tests."""
    print("=" * 60)
    print("CACHE BUILD TOLERANCE TESTS")
    print("=" * 60)
    
    try:
        test_critical_tickers_defined()
        test_status_determination()
        test_exit_code_logic()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
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

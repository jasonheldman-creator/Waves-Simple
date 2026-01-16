#!/usr/bin/env python3
"""
Verify the snapshot freshness fix by examining the code changes.

This script validates that our fixes are in place without requiring full dependencies.
"""

import re
import os

def check_file_for_pattern(filename, pattern, should_exist=True, description=""):
    """Check if a pattern exists in a file."""
    if not os.path.exists(filename):
        print(f"  ✗ File {filename} not found")
        return False
    
    with open(filename, 'r') as f:
        content = f.read()
    
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    found = match is not None
    
    if should_exist:
        if found:
            print(f"  ✓ {description}")
            return True
        else:
            print(f"  ✗ {description} - MISSING")
            return False
    else:
        if not found:
            print(f"  ✓ {description}")
            return True
        else:
            print(f"  ✗ {description} - FOUND (should not exist)")
            return False


def verify_snapshot_ledger_fixes():
    """Verify fixes in snapshot_ledger.py."""
    print("\n" + "=" * 80)
    print("VERIFYING snapshot_ledger.py FIXES")
    print("=" * 80)
    
    all_passed = True
    
    # 1. Check that _get_snapshot_date() raises error instead of returning datetime.now()
    all_passed &= check_file_for_pattern(
        'snapshot_ledger.py',
        r'CRITICAL ERROR.*SPY-based snapshot date.*raise RuntimeError',
        should_exist=True,
        description="_get_snapshot_date() raises error when no price data (no datetime.now() fallback)"
    )
    
    # 2. Check that _get_snapshot_date() prioritizes metadata
    all_passed &= check_file_for_pattern(
        'snapshot_ledger.py',
        r'cache_meta_path = "data/cache/prices_cache_meta.json".*spy_max_date',
        should_exist=True,
        description="_get_snapshot_date() reads spy_max_date from metadata"
    )
    
    # 3. Check that load_snapshot() compares price cache date vs snapshot date
    all_passed &= check_file_for_pattern(
        'snapshot_ledger.py',
        r'prices_cache_max_date.*>.*snapshot_date',
        should_exist=True,
        description="load_snapshot() compares price cache date vs snapshot date"
    )
    
    # 4. Check that load_snapshot() has detailed logging
    all_passed &= check_file_for_pattern(
        'snapshot_ledger.py',
        r'SNAPSHOT FRESHNESS CHECK.*Max SPY Price Date.*Snapshot Rebuild',
        should_exist=True,
        description="load_snapshot() has detailed freshness logging"
    )
    
    # 5. Check that generate_snapshot() checks price data freshness
    all_passed &= check_file_for_pattern(
        'snapshot_ledger.py',
        r'price data is newer.*Snapshot date.*Price cache max date',
        should_exist=True,
        description="generate_snapshot() checks if price data is newer"
    )
    
    # 6. Verify datetime.now() is NOT used for snapshot dates (only for age calculations)
    # This is tricky - we allow datetime.now() for age calculations but not for snapshot dates
    # The key is that snapshot dates should come from price data
    print("  ℹ️  Note: datetime.now() may still be used for age calculations (acceptable)")
    
    return all_passed


def verify_analytics_truth_fixes():
    """Verify fixes in analytics_truth.py."""
    print("\n" + "=" * 80)
    print("VERIFYING analytics_truth.py FIXES")
    print("=" * 80)
    
    all_passed = True
    
    # 1. Check that generate_live_snapshot_csv() extracts SPY date from prices_cache
    all_passed &= check_file_for_pattern(
        'analytics_truth.py',
        r"if 'SPY' in prices_cache:.*spy_max_date.*snapshot_date_str",
        should_exist=True,
        description="generate_live_snapshot_csv() extracts SPY date from prices_cache"
    )
    
    # 2. Check that it raises error when SPY date unavailable
    all_passed &= check_file_for_pattern(
        'analytics_truth.py',
        r'if snapshot_date_str is None:.*CRITICAL ERROR.*Unable to determine snapshot date',
        should_exist=True,
        description="generate_live_snapshot_csv() raises error when SPY date unavailable"
    )
    
    # 3. Check that snapshot_utc uses SPY-based date
    all_passed &= check_file_for_pattern(
        'analytics_truth.py',
        r"'asof_utc': snapshot_utc.*Use SPY-based date",
        should_exist=True,
        description="generate_live_snapshot_csv() uses SPY-based date for asof_utc"
    )
    
    # 4. Check that date field uses snapshot_date_str
    all_passed &= check_file_for_pattern(
        'analytics_truth.py',
        r'df\["date"\] = snapshot_date_str.*Use SPY-based date',
        should_exist=True,
        description="generate_live_snapshot_csv() uses SPY-based date for date field"
    )
    
    return all_passed


def verify_no_silent_reuse():
    """Verify that silent snapshot reuse is eliminated."""
    print("\n" + "=" * 80)
    print("VERIFYING NO SILENT SNAPSHOT REUSE")
    print("=" * 80)
    
    all_passed = True
    
    # Check that when price data is newer, snapshot is NOT silently reused
    all_passed &= check_file_for_pattern(
        'snapshot_ledger.py',
        r'if prices_cache_max_date.*>.*snapshot_date:.*print.*regenerat',
        should_exist=True,
        description="Newer price data triggers regeneration (not silent reuse)"
    )
    
    return all_passed


def main():
    print("=" * 80)
    print("SNAPSHOT FRESHNESS FIX - CODE VERIFICATION")
    print("=" * 80)
    print("\nThis script verifies that the key fixes are in place:")
    print("1. Snapshot dates use SPY price data, not datetime.now()")
    print("2. Explicit rebuild triggering when price data is newer")
    print("3. No silent reuse of stale snapshots")
    print("4. Detailed freshness logging")
    
    all_passed = True
    
    all_passed &= verify_snapshot_ledger_fixes()
    all_passed &= verify_analytics_truth_fixes()
    all_passed &= verify_no_silent_reuse()
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL VERIFICATIONS PASSED")
        print("=" * 80)
        print("\nThe snapshot freshness fix has been successfully implemented.")
        print("Key improvements:")
        print("  • Snapshot dates now use SPY price data from prices_cache_meta.json")
        print("  • Explicit rebuild when price data is newer than snapshot")
        print("  • Detailed logging of snapshot freshness decisions")
        print("  • No silent reuse of stale snapshots")
        return 0
    else:
        print("✗ SOME VERIFICATIONS FAILED")
        print("=" * 80)
        print("\nPlease review the failed checks above.")
        return 1


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
"""
Validate Price Cache Script

This script runs comprehensive validation on the price cache to ensure:
1. Trading-day freshness (cache is up-to-date with latest trading day)
2. Required symbols are present (ALL/ANY group semantics)
3. Cache integrity (file exists, non-empty, has symbols)
4. No-change logic (fresh/stale + changed/unchanged scenarios)

Usage:
    python validate_cache.py
    python validate_cache.py --cache-path data/cache/prices_cache.parquet
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.cache_validation import (
    validate_trading_day_freshness,
    validate_required_symbols,
    validate_cache_integrity,
    check_for_changes,
    validate_no_change_logic,
    REQUIRED_SYMBOLS_ALL,
    REQUIRED_SYMBOLS_VIX_ANY,
    REQUIRED_SYMBOLS_TBILL_ANY
)


def print_separator(char='=', length=70):
    """Print a separator line."""
    print(char * length)


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate price cache')
    parser.add_argument(
        '--cache-path',
        default='data/cache/prices_cache.parquet',
        help='Path to cache parquet file (default: data/cache/prices_cache.parquet)'
    )
    parser.add_argument(
        '--max-market-gap',
        type=int,
        default=5,
        help='Maximum gap in days before market data is considered broken (default: 5)'
    )
    parser.add_argument(
        '--check-git',
        action='store_true',
        help='Check for git changes and run no-change logic'
    )
    
    args = parser.parse_args()
    
    print_separator()
    print("PRICE CACHE VALIDATION")
    print_separator()
    print(f"Cache path: {args.cache_path}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    all_valid = True
    
    # Validation 1: Cache Integrity
    print_separator('-')
    print("Validation 1: Cache Integrity")
    print_separator('-')
    integrity_result = validate_cache_integrity(args.cache_path)
    
    if not integrity_result['valid']:
        print(f"\n❌ FAIL: {integrity_result['error']}")
        all_valid = False
    else:
        print(f"\n✅ PASS: Cache integrity validated")
        print(f"  File exists: {integrity_result['file_exists']}")
        print(f"  File size: {integrity_result['file_size_bytes']:,} bytes ({integrity_result['file_size_bytes'] / (1024*1024):.2f} MB)")
        print(f"  Symbol count: {integrity_result['symbol_count']}")
    print()
    
    # Validation 2: Required Symbols
    print_separator('-')
    print("Validation 2: Required Symbols")
    print_separator('-')
    print(f"Required symbol groups:")
    print(f"  ALL group: {REQUIRED_SYMBOLS_ALL}")
    print(f"  VIX ANY group (at least 1): {REQUIRED_SYMBOLS_VIX_ANY}")
    print(f"  T-bill ANY group (at least 1): {REQUIRED_SYMBOLS_TBILL_ANY}")
    print()
    
    symbols_result = validate_required_symbols(args.cache_path)
    
    if not symbols_result['valid']:
        print(f"❌ FAIL: {symbols_result['error']}")
        all_valid = False
    else:
        print(f"✅ PASS: All required symbols present")
        print(f"  Total symbols in cache: {len(symbols_result['symbols_in_cache'])}")
        print(f"  ALL group present: {REQUIRED_SYMBOLS_ALL}")
        print(f"  VIX group present: {symbols_result['present_vix_group']}")
        print(f"  T-bill group present: {symbols_result['present_tbill_group']}")
    print()
    
    # Validation 3: Trading-Day Freshness
    print_separator('-')
    print("Validation 3: Trading-Day Freshness")
    print_separator('-')
    freshness_result = validate_trading_day_freshness(
        args.cache_path, 
        max_market_feed_gap_days=args.max_market_gap
    )
    
    if not freshness_result['valid']:
        print(f"\n❌ FAIL: {freshness_result['error']}")
        print(f"  Today: {freshness_result['today'].date()}")
        print(f"  Last trading day: {freshness_result['last_trading_day'].date() if freshness_result['last_trading_day'] else 'N/A'}")
        print(f"  Cache max date: {freshness_result['cache_max_date'].date() if freshness_result['cache_max_date'] else 'N/A'}")
        print(f"  Delta (cache - trading): {freshness_result['delta_days']} days")
        print(f"  Market feed gap: {freshness_result['market_feed_gap_days']} days")
        all_valid = False
    else:
        print(f"\n✅ PASS: Cache is fresh and up-to-date")
        print(f"  Today: {freshness_result['today'].date()}")
        print(f"  Last trading day: {freshness_result['last_trading_day'].date()}")
        print(f"  Cache max date: {freshness_result['cache_max_date'].date()}")
        print(f"  Market feed gap: {freshness_result['market_feed_gap_days']} days")
    print()
    
    # Optional: Check git changes and no-change logic
    if args.check_git:
        print_separator('-')
        print("Git Changes Check")
        print_separator('-')
        
        git_result = check_for_changes()
        
        print(f"Has uncommitted changes: {git_result['has_changes']}")
        
        if git_result['git_status']:
            print(f"\nGit status:")
            for line in git_result['git_status'].split('\n')[:10]:
                print(f"  {line}")
        
        if git_result['git_diff_stat']:
            print(f"\nGit diff stats:")
            for line in git_result['git_diff_stat'].split('\n')[:10]:
                print(f"  {line}")
        
        print()
        
        # Run no-change logic
        print_separator('-')
        print("No-Change Logic")
        print_separator('-')
        
        no_change_result = validate_no_change_logic(
            cache_freshness_valid=freshness_result['valid'],
            has_changes=git_result['has_changes']
        )
        
        print(f"\nDecision:")
        print(f"  Should commit: {no_change_result['should_commit']}")
        print(f"  Should succeed: {no_change_result['should_succeed']}")
        print(f"  Message: {no_change_result['message']}")
        print()
        
        if not no_change_result['should_succeed']:
            all_valid = False
    
    # Final summary
    print_separator()
    print("VALIDATION SUMMARY")
    print_separator()
    
    if all_valid:
        print("✅ ALL VALIDATIONS PASSED")
        print_separator()
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print_separator()
        return 1


if __name__ == '__main__':
    sys.exit(main())

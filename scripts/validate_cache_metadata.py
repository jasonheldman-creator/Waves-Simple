#!/usr/bin/env python3
"""
Validate Price Cache Metadata

This script validates the prices_cache_meta.json file to ensure:
1. spy_max_date is not missing or null
2. tickers_total is at least 50
3. spy_max_date is not more than 5 calendar days stale compared to UTC today

Exits with code 1 if any validation fails, code 0 if all validations pass.
"""

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path


def is_valid_spy_date(spy_max_date):
    """
    Check if spy_max_date is valid (not None, not empty, not whitespace-only).
    
    Args:
        spy_max_date: The spy_max_date value to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if spy_max_date is None:
        return False
    if isinstance(spy_max_date, str) and not spy_max_date.strip():
        return False
    return True


def validate_cache_metadata(metadata_path='data/cache/prices_cache_meta.json'):
    """
    Validate cache metadata file.
    
    Args:
        metadata_path: Path to metadata JSON file
        
    Returns:
        bool: True if all validations pass, False otherwise
    """
    print("=" * 70)
    print("CACHE METADATA VALIDATION")
    print("=" * 70)
    
    # Check if file exists
    if not Path(metadata_path).exists():
        print(f"✗ ERROR: Metadata file not found: {metadata_path}")
        return False
    
    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"✗ ERROR: Failed to load metadata file: {e}")
        return False
    
    # Log all relevant metadata values
    print("\nMetadata Values:")
    print(f"  spy_max_date: {metadata.get('spy_max_date', 'MISSING')}")
    print(f"  max_price_date: {metadata.get('max_price_date', 'MISSING')}")
    print(f"  tickers_total: {metadata.get('tickers_total', 'MISSING')}")
    print(f"  tickers_successful: {metadata.get('tickers_successful', 'MISSING')}")
    print(f"  generated_at_utc: {metadata.get('generated_at_utc', 'MISSING')}")
    print()
    
    all_valid = True
    
    # Validation 1: spy_max_date is not missing or null
    print("Validation 1: spy_max_date exists and is not null")
    spy_max_date = metadata.get('spy_max_date')
    if not is_valid_spy_date(spy_max_date):
        print(f"✗ FAIL: spy_max_date is missing or null")
        all_valid = False
    else:
        print(f"✓ PASS: spy_max_date = {spy_max_date}")
    print()
    
    # Validation 2: tickers_total >= 50
    print("Validation 2: tickers_total >= 50")
    tickers_total = metadata.get('tickers_total')
    if tickers_total is None:
        print(f"✗ FAIL: tickers_total is missing")
        all_valid = False
    elif not isinstance(tickers_total, (int, float)):
        print(f"✗ FAIL: tickers_total is not a number: {tickers_total}")
        all_valid = False
    elif tickers_total < 50:
        print(f"✗ FAIL: tickers_total ({tickers_total}) is less than 50")
        all_valid = False
    else:
        print(f"✓ PASS: tickers_total = {tickers_total}")
    print()
    
    # Validation 3: spy_max_date is not more than 5 calendar days stale
    print("Validation 3: spy_max_date is not more than 5 calendar days stale")
    if not is_valid_spy_date(spy_max_date):
        print(f"✗ FAIL: Cannot validate staleness - spy_max_date is missing")
        all_valid = False
    else:
        try:
            # Parse the spy_max_date
            spy_date = datetime.strptime(spy_max_date, '%Y-%m-%d').date()
            
            # Get current UTC date
            utc_now = datetime.now(timezone.utc).date()
            
            # Calculate days difference
            days_old = (utc_now - spy_date).days
            
            print(f"  spy_max_date: {spy_date}")
            print(f"  UTC today: {utc_now}")
            print(f"  Days old: {days_old}")
            
            if days_old > 5:
                print(f"✗ FAIL: spy_max_date is {days_old} days old (max allowed: 5)")
                all_valid = False
            else:
                print(f"✓ PASS: spy_max_date is {days_old} days old (within 5-day threshold)")
        except ValueError as e:
            print(f"✗ FAIL: Invalid date format for spy_max_date: {spy_max_date} (expected YYYY-MM-DD format) - {e}")
            all_valid = False
    print()
    
    # Summary
    print("=" * 70)
    if all_valid:
        print("✓ ALL VALIDATIONS PASSED")
        print("=" * 70)
        return True
    else:
        print("✗ VALIDATION FAILED")
        print("=" * 70)
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate price cache metadata')
    parser.add_argument('metadata_path', nargs='?', 
                        default='data/cache/prices_cache_meta.json',
                        help='Path to metadata JSON file (default: data/cache/prices_cache_meta.json)')
    
    args = parser.parse_args()
    
    success = validate_cache_metadata(args.metadata_path)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

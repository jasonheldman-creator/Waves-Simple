#!/usr/bin/env python3
"""
Validate Price Cache Metadata

This script validates the prices_cache_meta.json file to ensure:
1. spy_max_date is not missing or null
2. tickers_total is at least 50
3. spy_max_date matches the latest SPY trading day (with optional 1-day grace period)

The staleness check now dynamically fetches SPY trading history to determine
the actual latest trading day, accounting for weekends and holidays.

Exits with code 1 if any validation fails, code 0 if all validations pass.
"""

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple


def fetch_spy_latest_trading_day(calendar_days: int = 15) -> Tuple[Optional[datetime], list]:
    """
    Fetch SPY price history to determine the latest trading day.
    
    Args:
        calendar_days: Number of calendar days to look back (default: 15)
        
    Returns:
        Tuple of (latest_trading_day, all_trading_days):
        - latest_trading_day: Most recent trading day from SPY data (or None if unavailable)
        - all_trading_days: List of all trading day dates in the period
    """
    try:
        import yfinance as yf
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=calendar_days)
        
        print(f"  Fetching SPY data from {start_date.date()} to {end_date.date()}...")
        
        # Fetch SPY data
        spy_data = yf.download(
            tickers="SPY",
            start=start_date.strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False,
            timeout=15
        )
        
        if spy_data.empty:
            print("  ⚠ Warning: No SPY data returned from yfinance")
            return None, []
        
        # Extract trading days from index
        trading_days = [dt.date() for dt in spy_data.index]
        
        if not trading_days:
            print("  ⚠ Warning: No trading days found in SPY data")
            return None, []
        
        latest_trading_day = max(trading_days)
        
        print(f"  Found {len(trading_days)} trading days")
        print(f"  Latest trading day: {latest_trading_day}")
        
        return latest_trading_day, trading_days
        
    except ImportError:
        print("  ⚠ Warning: yfinance not available - cannot fetch SPY data")
        return None, []
    except Exception as e:
        print(f"  ⚠ Warning: Error fetching SPY trading days: {e}")
        return None, []


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


def validate_cache_metadata(metadata_path='data/cache/prices_cache_meta.json', grace_period_days: int = 1):
    """
    Validate cache metadata file.
    
    Args:
        metadata_path: Path to metadata JSON file
        grace_period_days: Allow cache to be this many trading days behind (default: 1)
        
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
    
    # Validation 3: spy_max_date matches latest trading day
    print("Validation 3: spy_max_date matches latest SPY trading day")
    if not is_valid_spy_date(spy_max_date):
        print(f"✗ FAIL: Cannot validate trading day freshness - spy_max_date is missing")
        all_valid = False
    else:
        try:
            # Parse the spy_max_date
            spy_date = datetime.strptime(spy_max_date, '%Y-%m-%d').date()
            
            print(f"  spy_max_date from metadata: {spy_date}")
            print()
            
            # Fetch latest trading day from SPY
            latest_trading_day, trading_days = fetch_spy_latest_trading_day(calendar_days=15)
            
            if latest_trading_day is None:
                print(f"⚠ WARNING: Could not fetch SPY trading data")
                print(f"  Falling back to basic staleness check (5 calendar days)")
                print()
                
                # Fallback to basic calendar day check
                utc_now = datetime.now(timezone.utc).date()
                days_old = (utc_now - spy_date).days
                
                print(f"  UTC today: {utc_now}")
                print(f"  Days old: {days_old}")
                
                if days_old > 5:
                    print(f"✗ FAIL: spy_max_date is {days_old} days old (max allowed: 5 calendar days)")
                    all_valid = False
                else:
                    print(f"✓ PASS: spy_max_date is {days_old} days old (within 5-day threshold)")
            else:
                # Compare with latest trading day
                print(f"  latest_trading_day from SPY: {latest_trading_day}")
                
                # Calculate the difference
                day_diff = (latest_trading_day - spy_date).days
                
                # Determine how many trading days behind
                if spy_date in trading_days:
                    sorted_trading_days = sorted(trading_days, reverse=True)
                    try:
                        spy_index = sorted_trading_days.index(spy_date)
                        sessions_behind = spy_index
                    except ValueError:
                        sessions_behind = None
                else:
                    sessions_behind = None
                
                print(f"  Difference: {day_diff} calendar days")
                if sessions_behind is not None:
                    print(f"  Sessions behind: {sessions_behind} trading days")
                
                # Validation logic: MATCH or within grace period
                if spy_date == latest_trading_day:
                    print(f"  Comparison: MATCH ✓")
                    print(f"✓ PASS: spy_max_date matches latest trading day")
                elif sessions_behind is not None and sessions_behind <= grace_period_days:
                    print(f"  Comparison: BEHIND by {sessions_behind} trading day(s)")
                    print(f"✓ PASS: spy_max_date is within {grace_period_days}-day grace period")
                else:
                    if sessions_behind is not None:
                        print(f"  Comparison: BEHIND by {sessions_behind} trading day(s)")
                        print(f"✗ FAIL: spy_max_date is {sessions_behind} trading days behind latest trading day")
                        print(f"  (Grace period: {grace_period_days} trading day(s))")
                    else:
                        print(f"  Comparison: BEHIND (spy_max_date not in recent trading days)")
                        print(f"✗ FAIL: spy_max_date ({spy_date}) is not a recent trading day")
                    all_valid = False
                    
        except ValueError as e:
            print(f"✗ FAIL: Invalid date format for spy_max_date: {spy_max_date}. Expected format: YYYY-MM-DD")
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
    
    parser = argparse.ArgumentParser(
        description='Validate price cache metadata with trading-day awareness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate with default 1-day grace period
  python validate_cache_metadata.py
  
  # Require exact match (no grace period)
  python validate_cache_metadata.py --grace-period 0
  
  # Allow up to 2 trading days behind
  python validate_cache_metadata.py --grace-period 2
        """
    )
    parser.add_argument('metadata_path', nargs='?', 
                        default='data/cache/prices_cache_meta.json',
                        help='Path to metadata JSON file (default: data/cache/prices_cache_meta.json)')
    parser.add_argument('--grace-period', type=int, default=1,
                        help='Number of trading days the cache can be behind (default: 1)')
    
    args = parser.parse_args()
    
    # Validate grace period is non-negative
    if args.grace_period < 0:
        print(f"Error: grace-period must be non-negative (got: {args.grace_period})")
        sys.exit(1)
    
    success = validate_cache_metadata(args.metadata_path, grace_period_days=args.grace_period)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

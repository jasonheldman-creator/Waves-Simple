#!/usr/bin/env python3
"""
Canonical Price Loading Mechanism

This script implements the canonical price-loading mechanism for all waves.
It dynamically fetches and caches prices for each wave's tickers, generating
data/waves/<wave_id>/prices.csv for ALL waves (except SmartSafe cash waves).

Key Features:
- Automatically generates prices.csv for all waves that need it
- Skips SmartSafe cash waves (they don't need price data)
- Uses existing analytics_pipeline functions for consistency
- Provides detailed progress and error reporting

Usage:
    # Generate prices for all waves
    python generate_all_wave_prices.py
    
    # Generate prices for specific waves only
    python generate_all_wave_prices.py --waves sp500_wave gold_wave
    
    # Use dummy data for testing (no network calls)
    python generate_all_wave_prices.py --dummy
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics_pipeline import (
    get_all_wave_ids,
    generate_prices_csv,
    generate_benchmark_prices_csv,
    generate_positions_csv,
    generate_trades_csv,
    generate_nav_csv,
    compute_data_ready_status,
    DEFAULT_LOOKBACK_DAYS,
    MIN_REQUIRED_TRADING_DAYS,
)
from waves_engine import (
    is_smartsafe_cash_wave,
    get_display_name_from_wave_id,
)


def generate_prices_for_wave(
    wave_id: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    use_dummy_data: bool = False,
    skip_existing: bool = False
) -> Dict[str, Any]:
    """
    Generate price data and related files for a single wave.
    
    Args:
        wave_id: Wave identifier
        lookback_days: Number of days of historical data to fetch
        use_dummy_data: If True, generate dummy data instead of fetching
        skip_existing: If True, skip waves that already have prices.csv
        
    Returns:
        Dictionary with generation results
    """
    result = {
        'wave_id': wave_id,
        'display_name': get_display_name_from_wave_id(wave_id),
        'is_smartsafe': False,
        'skipped': False,
        'success': True,
        'errors': [],
        'files_generated': []
    }
    
    # Check if this is a SmartSafe cash wave
    if is_smartsafe_cash_wave(wave_id):
        result['is_smartsafe'] = True
        result['skipped'] = True
        result['skip_reason'] = 'SmartSafe cash wave (no price data needed)'
        return result
    
    # Check if prices already exist and we should skip
    if skip_existing:
        from analytics_pipeline import get_wave_analytics_dir
        wave_dir = get_wave_analytics_dir(wave_id)
        prices_path = os.path.join(wave_dir, 'prices.csv')
        if os.path.exists(prices_path):
            result['skipped'] = True
            result['skip_reason'] = 'prices.csv already exists'
            return result
    
    # Generate prices.csv
    try:
        if generate_prices_csv(wave_id, lookback_days, use_dummy_data):
            result['files_generated'].append('prices.csv')
        else:
            result['errors'].append('Failed to generate prices.csv')
            result['success'] = False
    except Exception as e:
        result['errors'].append(f'Error generating prices.csv: {str(e)}')
        result['success'] = False
    
    # Generate benchmark_prices.csv
    try:
        if generate_benchmark_prices_csv(wave_id, lookback_days, use_dummy_data):
            result['files_generated'].append('benchmark_prices.csv')
        else:
            result['errors'].append('Failed to generate benchmark_prices.csv')
            # Note: benchmark failure is not fatal
    except Exception as e:
        result['errors'].append(f'Error generating benchmark_prices.csv: {str(e)}')
        # Note: benchmark failure is not fatal
    
    # Generate positions.csv
    try:
        if generate_positions_csv(wave_id):
            result['files_generated'].append('positions.csv')
        else:
            result['errors'].append('Failed to generate positions.csv')
            # Note: positions failure is not fatal
    except Exception as e:
        result['errors'].append(f'Error generating positions.csv: {str(e)}')
    
    # Generate trades.csv
    try:
        if generate_trades_csv(wave_id):
            result['files_generated'].append('trades.csv')
        else:
            result['errors'].append('Failed to generate trades.csv')
            # Note: trades failure is not fatal
    except Exception as e:
        result['errors'].append(f'Error generating trades.csv: {str(e)}')
    
    # Generate nav.csv
    try:
        if generate_nav_csv(wave_id, lookback_days):
            result['files_generated'].append('nav.csv')
        else:
            result['errors'].append('Failed to generate nav.csv')
            # Note: nav failure is not fatal
    except Exception as e:
        result['errors'].append(f'Error generating nav.csv: {str(e)}')
    
    return result


def generate_all_prices(
    wave_ids: List[str] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    use_dummy_data: bool = False,
    skip_existing: bool = False
) -> Dict[str, Any]:
    """
    Generate prices for all waves (or specified waves).
    
    This is the canonical price-loading mechanism for the application.
    
    Args:
        wave_ids: List of specific wave IDs to process (None = all waves)
        lookback_days: Number of days of historical data to fetch
        use_dummy_data: If True, generate dummy data instead of fetching
        skip_existing: If True, skip waves that already have prices.csv
        
    Returns:
        Dictionary with summary results
    """
    print("=" * 80)
    print("CANONICAL PRICE LOADING MECHANISM")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Lookback days: {lookback_days}")
    print(f"Dummy data mode: {use_dummy_data}")
    print(f"Skip existing: {skip_existing}")
    print("=" * 80)
    print()
    
    # Determine which waves to process
    if wave_ids is None:
        wave_ids = get_all_wave_ids()
        print(f"Processing ALL waves: {len(wave_ids)} total")
    else:
        print(f"Processing specific waves: {len(wave_ids)} total")
    
    print()
    
    # Initialize counters
    summary = {
        'total_waves': len(wave_ids),
        'successful': 0,
        'failed': 0,
        'skipped_smartsafe': 0,
        'skipped_existing': 0,
        'results': []
    }
    
    # Process each wave
    for i, wave_id in enumerate(wave_ids, 1):
        display_name = get_display_name_from_wave_id(wave_id) or wave_id
        
        print(f"[{i}/{len(wave_ids)}] {wave_id} ({display_name})")
        print("-" * 80)
        
        result = generate_prices_for_wave(
            wave_id,
            lookback_days=lookback_days,
            use_dummy_data=use_dummy_data,
            skip_existing=skip_existing
        )
        
        summary['results'].append(result)
        
        # Update counters
        if result['skipped']:
            if result['is_smartsafe']:
                summary['skipped_smartsafe'] += 1
                print(f"⊘ SKIPPED: {result['skip_reason']}")
            else:
                summary['skipped_existing'] += 1
                print(f"⊘ SKIPPED: {result['skip_reason']}")
        elif result['success']:
            summary['successful'] += 1
            print(f"✓ SUCCESS: Generated {len(result['files_generated'])} files")
            if result['errors']:
                print(f"  ⚠️  Non-fatal warnings: {len(result['errors'])}")
        else:
            summary['failed'] += 1
            print(f"✗ FAILED: {len(result['errors'])} error(s)")
            for error in result['errors'][:3]:  # Show first 3 errors
                print(f"  - {error}")
        
        print()
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total waves: {summary['total_waves']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Skipped (SmartSafe): {summary['skipped_smartsafe']}")
    print(f"Skipped (existing): {summary['skipped_existing']}")
    print("=" * 80)
    
    # Show failed waves if any
    if summary['failed'] > 0:
        print("\nFailed waves:")
        for result in summary['results']:
            if not result['success'] and not result['skipped']:
                print(f"  - {result['wave_id']}: {', '.join(result['errors'][:2])}")
    
    # Validate readiness after generation
    print("\n" + "=" * 80)
    print("READINESS VALIDATION")
    print("=" * 80)
    
    ready_count = 0
    partial_count = 0
    operational_count = 0
    unavailable_count = 0
    
    for wave_id in wave_ids:
        if is_smartsafe_cash_wave(wave_id):
            continue  # SmartSafe waves are always ready
        
        status = compute_data_ready_status(wave_id)
        readiness = status.get('readiness_status', 'unavailable')
        
        if readiness == 'full':
            ready_count += 1
            icon = '🟢'
        elif readiness == 'partial':
            partial_count += 1
            icon = '🟡'
        elif readiness == 'operational':
            operational_count += 1
            icon = '🟠'
        else:
            unavailable_count += 1
            icon = '🔴'
        
        coverage = status.get('coverage_pct', 0)
        print(f"{icon} {wave_id}: {readiness.upper()} (coverage: {coverage:.1f}%)")
    
    print("\n" + "=" * 80)
    print(f"Full: {ready_count}")
    print(f"Partial: {partial_count}")
    print(f"Operational: {operational_count}")
    print(f"Unavailable: {unavailable_count}")
    print(f"SmartSafe (exempt): {summary['skipped_smartsafe']}")
    print("=" * 80)
    
    return summary


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Canonical price loading mechanism for all waves'
    )
    parser.add_argument(
        '--waves',
        nargs='+',
        help='Specific wave IDs to process (default: all waves)'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f'Number of days to look back (default: {DEFAULT_LOOKBACK_DAYS})'
    )
    parser.add_argument(
        '--dummy',
        action='store_true',
        help='Use dummy data instead of fetching from yfinance'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip waves that already have prices.csv'
    )
    
    args = parser.parse_args()
    
    # Run the price generation
    summary = generate_all_prices(
        wave_ids=args.waves,
        lookback_days=args.lookback,
        use_dummy_data=args.dummy,
        skip_existing=args.skip_existing
    )
    
    # Exit with appropriate code
    if summary['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()

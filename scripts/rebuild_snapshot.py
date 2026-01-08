#!/usr/bin/env python3
"""
Rebuild snapshot script for GitHub Actions workflow.

This script rebuilds the live snapshot by:
1. Validating the price cache exists and contains required tickers
2. Building the cache if missing or invalid
3. Calling generate_live_snapshot_csv with validated cache data

The generated snapshot and any cache artifacts in data/cache/ are ignored by
.gitignore and will not be committed.

The workflow succeeds based on successful execution of this script only.
"""

import sys
import os
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics_truth import generate_live_snapshot_csv


def validate_price_cache():
    """
    Validate that price cache exists and contains required tickers.
    
    Required tickers:
    - SPY, QQQ, IWM (all required)
    - At least one of: ^VIX, VIXY, VXX
    
    Returns:
        Tuple of (is_valid: bool, diagnostics: dict)
    """
    import pandas as pd
    import json
    from datetime import datetime
    
    CACHE_PATH = "data/cache/prices_cache.parquet"
    CACHE_META_PATH = "data/cache/prices_cache_meta.json"
    
    # Required symbols
    REQUIRED_BENCHMARKS = ['SPY', 'QQQ', 'IWM']
    REQUIRED_VOLATILITY = ['^VIX', 'VIXY', 'VXX']  # Need at least one
    
    print("\n" + "=" * 80)
    print("PRICE CACHE VALIDATION")
    print("=" * 80)
    
    diagnostics = {
        'cache_exists': False,
        'cache_size_mb': 0,
        'ticker_count': 0,
        'max_date': None,
        'required_benchmarks_present': [],
        'required_benchmarks_missing': [],
        'required_volatility_present': [],
        'required_volatility_missing': [],
        'is_valid': False
    }
    
    # Check if cache file exists
    if not os.path.exists(CACHE_PATH):
        print(f"\n✗ Cache file not found: {CACHE_PATH}")
        print("  Cache must be built before generating snapshot.")
        return False, diagnostics
    
    diagnostics['cache_exists'] = True
    
    # Get cache file size
    cache_size = os.path.getsize(CACHE_PATH)
    diagnostics['cache_size_mb'] = cache_size / (1024 * 1024)
    print(f"\n✓ Cache file exists: {CACHE_PATH}")
    print(f"  Size: {diagnostics['cache_size_mb']:.2f} MB")
    
    # Check if cache is empty
    if cache_size == 0:
        print("\n✗ Cache file is empty (0 bytes)")
        return False, diagnostics
    
    # Load cache to inspect contents
    try:
        cache_df = pd.read_parquet(CACHE_PATH)
        diagnostics['ticker_count'] = len(cache_df.columns)
        diagnostics['max_date'] = str(cache_df.index.max())
        
        print(f"  Ticker count: {diagnostics['ticker_count']}")
        print(f"  Date range: {cache_df.index.min()} to {cache_df.index.max()}")
        print(f"  Maximum date: {diagnostics['max_date']}")
        
    except Exception as e:
        print(f"\n✗ Error reading cache file: {e}")
        return False, diagnostics
    
    # Load metadata if available
    if os.path.exists(CACHE_META_PATH):
        try:
            with open(CACHE_META_PATH, 'r') as f:
                meta = json.load(f)
            print(f"\n  Cache metadata:")
            print(f"    Generated: {meta.get('generated_at_utc', 'N/A')}")
            print(f"    Success rate: {meta.get('success_rate', 'N/A')}")
            print(f"    Tickers successful: {meta.get('tickers_successful', 'N/A')}/{meta.get('tickers_total', 'N/A')}")
        except Exception as e:
            print(f"  Warning: Could not read metadata: {e}")
    
    # Validate required tickers
    cache_symbols = set(cache_df.columns)
    
    print("\n  Required ticker validation:")
    
    # Check benchmarks (all required)
    for ticker in REQUIRED_BENCHMARKS:
        if ticker in cache_symbols:
            diagnostics['required_benchmarks_present'].append(ticker)
        else:
            diagnostics['required_benchmarks_missing'].append(ticker)
    
    if diagnostics['required_benchmarks_missing']:
        print(f"    ✗ Missing required benchmarks: {diagnostics['required_benchmarks_missing']}")
    else:
        print(f"    ✓ All required benchmarks present: {diagnostics['required_benchmarks_present']}")
    
    # Check volatility regime (at least one required)
    for ticker in REQUIRED_VOLATILITY:
        if ticker in cache_symbols:
            diagnostics['required_volatility_present'].append(ticker)
        else:
            diagnostics['required_volatility_missing'].append(ticker)
    
    if diagnostics['required_volatility_present']:
        print(f"    ✓ Volatility indicators present: {diagnostics['required_volatility_present']}")
    else:
        print(f"    ✗ Missing ALL volatility indicators: {REQUIRED_VOLATILITY}")
    
    # Validate non-empty histories for required tickers
    required_tickers_with_data = []
    required_tickers_without_data = []
    
    all_required = REQUIRED_BENCHMARKS + diagnostics['required_volatility_present']
    for ticker in all_required:
        if ticker in cache_symbols:
            if not cache_df[ticker].isna().all():
                required_tickers_with_data.append(ticker)
            else:
                required_tickers_without_data.append(ticker)
    
    if required_tickers_without_data:
        print(f"    ✗ Required tickers with empty histories: {required_tickers_without_data}")
    
    # Overall validation
    is_valid = (
        len(diagnostics['required_benchmarks_missing']) == 0 and
        len(diagnostics['required_volatility_present']) > 0 and
        len(required_tickers_without_data) == 0
    )
    
    diagnostics['is_valid'] = is_valid
    
    if is_valid:
        print("\n✓ Cache validation PASSED")
    else:
        print("\n✗ Cache validation FAILED")
        if diagnostics['required_benchmarks_missing']:
            print(f"  Missing benchmarks: {diagnostics['required_benchmarks_missing']}")
        if not diagnostics['required_volatility_present']:
            print(f"  Missing all volatility indicators: {REQUIRED_VOLATILITY}")
        if required_tickers_without_data:
            print(f"  Empty histories: {required_tickers_without_data}")
    
    print("=" * 80)
    
    return is_valid, diagnostics


def build_price_cache():
    """
    Build the price cache by running build_price_cache.py.
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print("BUILDING PRICE CACHE")
    print("=" * 80)
    print("\nRunning build_price_cache.py...")
    
    try:
        result = subprocess.run(
            [sys.executable, "build_price_cache.py", "--force"],
            capture_output=False,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        if result.returncode == 0:
            print("\n✓ Price cache built successfully")
            return True
        else:
            print(f"\n✗ Price cache build failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n✗ Error building price cache: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main function to rebuild the live snapshot.
    
    This function:
    1. Validates price cache exists and contains required tickers
    2. Builds cache if missing or invalid
    3. Calls generate_live_snapshot_csv() to rebuild the snapshot
    4. Validates that the snapshot was created successfully
    5. Returns exit code 0 on success, 1 on failure
    
    Any cache artifacts generated in data/cache/ will be ignored by .gitignore.
    """
    try:
        print("\n" + "=" * 80)
        print("REBUILD SNAPSHOT WORKFLOW")
        print("=" * 80)
        
        # Step 1: Validate price cache
        is_valid, diagnostics = validate_price_cache()
        
        # Step 2: Build cache if needed
        if not is_valid:
            print("\n⚠ Price cache validation failed. Attempting to build cache...")
            if not build_price_cache():
                print("\n✗ Failed to build price cache")
                return 1
            
            # Re-validate after build
            print("\n⚠ Re-validating cache after build...")
            is_valid, diagnostics = validate_price_cache()
            
            if not is_valid:
                print("\n✗ Cache validation still failing after build")
                print("\nMissing tickers:")
                if diagnostics['required_benchmarks_missing']:
                    print(f"  Benchmarks: {diagnostics['required_benchmarks_missing']}")
                if not diagnostics['required_volatility_present']:
                    print(f"  Volatility: Need at least one of [^VIX, VIXY, VXX]")
                return 1
        
        # Step 3: Generate the live snapshot
        print("\n" + "=" * 80)
        print("GENERATING LIVE SNAPSHOT")
        print("=" * 80)
        print("\nRebuilding live snapshot from validated cache...")
        df = generate_live_snapshot_csv()
        
        # Validate the result (generate_live_snapshot_csv always returns a DataFrame
        # with exactly 28 rows or raises AssertionError)
        print(f"\n✓ Successfully rebuilt snapshot with {len(df)} rows")
        print("\nNote: Generated snapshot and cache artifacts in data/cache/")
        print("      are ignored by .gitignore and will not be committed.")
        print("\n" + "=" * 80)
        return 0
            
    except Exception as e:
        print(f"\n✗ Error rebuilding snapshot: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

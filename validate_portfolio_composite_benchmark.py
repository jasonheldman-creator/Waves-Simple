"""
Portfolio Composite Benchmark Validation Script

Validates that the portfolio composite benchmark:
1. Builds without errors
2. Is not empty or all-NaN
3. Has at least 60 rows of data
4. Reports range, rows, and any missing waves

Usage:
    python validate_portfolio_composite_benchmark.py
"""

import sys
import os
import pandas as pd
import logging
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waves_engine import (
    compute_history_nav,
    build_portfolio_composite_benchmark_returns,
    get_all_waves_universe
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
MIN_REQUIRED_DAYS = 60
DEFAULT_HISTORY_DAYS = 365  # Default lookback period for wave history computation


def load_price_cache():
    """Load cached price data."""
    cache_path = os.path.join(
        os.path.dirname(__file__),
        "data", "cache", "prices_cache.parquet"
    )
    if os.path.exists(cache_path):
        logger.info(f"Loading price cache from {cache_path}")
        return pd.read_parquet(cache_path)
    else:
        logger.error(f"Price cache not found at {cache_path}")
        return None


def validate_portfolio_composite_benchmark():
    """
    Validate portfolio composite benchmark implementation.
    
    Returns:
        bool: True if validation passes, False otherwise
    """
    print("=" * 80)
    print("PORTFOLIO COMPOSITE BENCHMARK VALIDATION")
    print("=" * 80)
    print()
    
    # Load price cache
    price_df = load_price_cache()
    if price_df is None:
        logger.error("✗ Failed to load price cache")
        return False
    
    logger.info(f"✓ Loaded price cache with {len(price_df)} rows and {len(price_df.columns)} tickers")
    print()
    
    # Get all waves
    try:
        universe = get_all_waves_universe()
        all_waves = universe.get('waves', [])
        logger.info(f"✓ Found {len(all_waves)} waves in universe")
    except Exception as e:
        logger.error(f"✗ Failed to get wave universe: {e}")
        return False
    
    # Compute wave results for all waves
    print()
    print("Computing wave benchmark returns...")
    print("-" * 80)
    
    wave_results = {}
    successful_waves = []
    failed_waves = []
    
    for wave_name in all_waves:
        try:
            # Compute history for this wave
            wave_df = compute_history_nav(
                wave_name=wave_name,
                mode='Standard',
                days=DEFAULT_HISTORY_DAYS,
                include_diagnostics=False,
                price_df=price_df
            )
            
            if wave_df.empty:
                logger.warning(f"  Wave '{wave_name}': empty result")
                failed_waves.append(wave_name)
                continue
            
            if 'bm_ret' not in wave_df.columns:
                logger.warning(f"  Wave '{wave_name}': missing 'bm_ret' column")
                failed_waves.append(wave_name)
                continue
            
            # Check if benchmark has any valid data
            if wave_df['bm_ret'].isna().all():
                logger.warning(f"  Wave '{wave_name}': benchmark returns all NaN")
                failed_waves.append(wave_name)
                continue
            
            wave_results[wave_name] = wave_df
            successful_waves.append(wave_name)
            logger.info(f"  ✓ Wave '{wave_name}': {len(wave_df)} days, {wave_df['bm_ret'].notna().sum()} non-NaN benchmark returns")
            
        except Exception as e:
            logger.error(f"  ✗ Wave '{wave_name}': {e}")
            failed_waves.append(wave_name)
    
    print()
    logger.info(f"Successfully computed benchmark returns for {len(successful_waves)}/{len(all_waves)} waves")
    
    if failed_waves:
        print()
        print("Failed waves:")
        for wave_name in failed_waves:
            print(f"  - {wave_name}")
    
    # Build portfolio composite benchmark
    print()
    print("Building portfolio composite benchmark...")
    print("-" * 80)
    
    try:
        composite_benchmark = build_portfolio_composite_benchmark_returns(
            wave_results=wave_results,
            wave_weights=None  # Use equal weights
        )
        logger.info(f"✓ Portfolio composite benchmark built successfully")
    except Exception as e:
        logger.error(f"✗ Failed to build portfolio composite benchmark: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Validate composite benchmark
    print()
    print("Validating composite benchmark...")
    print("-" * 80)
    
    validation_passed = True
    
    # Check 1: Not empty
    if composite_benchmark.empty:
        logger.error("✗ Composite benchmark is empty")
        validation_passed = False
    else:
        logger.info(f"✓ Composite benchmark is not empty ({len(composite_benchmark)} rows)")
    
    # Check 2: Not all NaN
    if not composite_benchmark.empty and composite_benchmark.isna().all():
        logger.error("✗ Composite benchmark returns are all NaN")
        validation_passed = False
    else:
        non_nan_count = composite_benchmark.notna().sum()
        logger.info(f"✓ Composite benchmark has {non_nan_count} non-NaN values")
    
    # Check 3: At least 60 rows
    if len(composite_benchmark) < MIN_REQUIRED_DAYS:
        logger.error(f"✗ Composite benchmark has insufficient data ({len(composite_benchmark)} < {MIN_REQUIRED_DAYS} days)")
        validation_passed = False
    else:
        logger.info(f"✓ Composite benchmark has sufficient data ({len(composite_benchmark)} >= {MIN_REQUIRED_DAYS} days)")
    
    # Report statistics
    if not composite_benchmark.empty:
        print()
        print("Composite Benchmark Statistics:")
        print("-" * 80)
        print(f"  Total days:           {len(composite_benchmark)}")
        print(f"  Non-NaN values:       {composite_benchmark.notna().sum()}")
        print(f"  Date range:           {composite_benchmark.index[0]} to {composite_benchmark.index[-1]}")
        print(f"  Mean daily return:    {composite_benchmark.mean():.6f} ({composite_benchmark.mean()*100:.4f}%)")
        print(f"  Std daily return:     {composite_benchmark.std():.6f} ({composite_benchmark.std()*100:.4f}%)")
        print(f"  Min daily return:     {composite_benchmark.min():.6f} ({composite_benchmark.min()*100:.4f}%)")
        print(f"  Max daily return:     {composite_benchmark.max():.6f} ({composite_benchmark.max()*100:.4f}%)")
        
        # Compute cumulative return
        cumulative = (1 + composite_benchmark).cumprod()
        total_return = (cumulative.iloc[-1] - 1) * 100
        print(f"  Total return:         {total_return:.2f}%")
    
    # Summary
    print()
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"  Waves processed:      {len(all_waves)}")
    print(f"  Waves with data:      {len(successful_waves)}")
    print(f"  Waves failed:         {len(failed_waves)}")
    print(f"  Composite benchmark:  {'✓ PASSED' if validation_passed else '✗ FAILED'}")
    print()
    
    if validation_passed:
        logger.info("✓✓✓ VALIDATION PASSED ✓✓✓")
        return True
    else:
        logger.error("✗✗✗ VALIDATION FAILED ✗✗✗")
        return False


if __name__ == '__main__':
    success = validate_portfolio_composite_benchmark()
    sys.exit(0 if success else 1)

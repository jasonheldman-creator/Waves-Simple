"""
Build Initial Price Cache

This script builds the initial price cache from existing price data files
or fetches fresh data from yfinance if needed.

It performs the following steps:
1. Collect all unique tickers from all waves
2. Load existing price data (if available)
3. Identify missing/stale tickers
4. Fetch missing data from yfinance
5. Save consolidated cache

Usage:
    python build_price_cache.py [--force] [--years 5]
    
    --force: Force rebuild cache even if it exists
    --years: Number of years of history to keep (default: 5)
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Threshold configuration - hardened parsing with clamping
DEFAULT_MIN_SUCCESS_RATE = 0.90
try:
    MIN_SUCCESS_RATE = min(1.0, max(0.0, float(os.getenv("MIN_SUCCESS_RATE", str(DEFAULT_MIN_SUCCESS_RATE)))))
except ValueError:
    MIN_SUCCESS_RATE = DEFAULT_MIN_SUCCESS_RATE

from helpers.price_loader import (
    deduplicate_tickers,
    load_cache,
    save_cache,
    fetch_prices_batch,
    merge_cache_and_new_data,
    trim_cache_to_date_range,
    get_cache_info,
    CACHE_PATH,
    CACHE_DIR,
    DEFAULT_CACHE_YEARS,
    BATCH_SIZE
)
from waves_engine import get_all_wave_ids, WAVE_WEIGHTS
from analytics_pipeline import resolve_wave_tickers, resolve_wave_benchmarks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metadata file path
METADATA_PATH = os.path.join(CACHE_DIR, "prices_cache_meta.json")


def collect_all_tickers():
    """
    Collect all unique tickers from all waves.
    
    Returns:
        Tuple of (wave_tickers, benchmark_tickers)
    """
    logger.info("Collecting tickers from all waves...")
    
    wave_tickers = set()
    benchmark_tickers = set()
    
    all_waves = get_all_wave_ids()
    
    for wave_id in all_waves:
        # Get wave tickers
        tickers = resolve_wave_tickers(wave_id)
        wave_tickers.update(tickers)
        
        # Get benchmark tickers
        benchmarks = resolve_wave_benchmarks(wave_id)
        for ticker, _ in benchmarks:
            benchmark_tickers.add(ticker)
    
    # Deduplicate and sort
    wave_tickers = deduplicate_tickers(list(wave_tickers))
    benchmark_tickers = deduplicate_tickers(list(benchmark_tickers))
    
    # Combine for total
    all_tickers = deduplicate_tickers(wave_tickers + benchmark_tickers)
    
    logger.info(f"Found {len(wave_tickers)} wave tickers")
    logger.info(f"Found {len(benchmark_tickers)} benchmark tickers")
    logger.info(f"Total unique tickers: {len(all_tickers)}")
    
    return all_tickers, wave_tickers, benchmark_tickers


def load_existing_price_files():
    """
    Load existing price data from various CSV files in the repository.
    
    Returns:
        DataFrame with dates as index and tickers as columns
    """
    logger.info("Looking for existing price data...")
    
    # Possible locations for price data
    price_files = [
        'data/prices.csv',
        'prices.csv',
        'data/waves/*/prices.csv'
    ]
    
    all_prices = []
    
    for pattern in price_files:
        import glob
        for filepath in glob.glob(pattern):
            if not os.path.exists(filepath):
                continue
            
            logger.info(f"Loading {filepath}...")
            
            try:
                df = pd.read_csv(filepath)
                
                # Check format
                if 'date' in df.columns and 'ticker' in df.columns and 'close' in df.columns:
                    # Long format: date, ticker, close
                    df['date'] = pd.to_datetime(df['date'])
                    df_pivot = df.pivot(index='date', columns='ticker', values='close')
                    all_prices.append(df_pivot)
                    logger.info(f"  Loaded {len(df_pivot)} days, {len(df_pivot.columns)} tickers")
                
                elif df.columns[0] == 'date' or df.columns[0] == 'Date':
                    # Wide format with date column
                    df = df.rename(columns={df.columns[0]: 'date'})
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    all_prices.append(df)
                    logger.info(f"  Loaded {len(df)} days, {len(df.columns)} tickers")
                
                else:
                    # Try assuming first column is date index
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    if isinstance(df.index, pd.DatetimeIndex):
                        all_prices.append(df)
                        logger.info(f"  Loaded {len(df)} days, {len(df.columns)} tickers")
                
            except Exception as e:
                logger.warning(f"  Could not load {filepath}: {e}")
                continue
    
    if not all_prices:
        logger.info("No existing price data found")
        return pd.DataFrame()
    
    # Merge all price data
    logger.info("Merging price data from multiple sources...")
    merged = all_prices[0]
    
    for df in all_prices[1:]:
        # Combine on index (dates) and columns (tickers)
        merged = merged.combine_first(df)
    
    logger.info(f"Merged data: {len(merged)} days, {len(merged.columns)} tickers")
    
    return merged


def save_metadata(total_tickers, successful_tickers, failed_tickers, success_rate, max_price_date):
    """
    Save metadata file next to the cache file.
    
    Args:
        total_tickers: Total number of tickers requested
        successful_tickers: Number of successfully downloaded tickers
        failed_tickers: Number of failed tickers
        success_rate: Success rate (0.0 to 1.0)
        max_price_date: Latest date in the cache (datetime or string)
    """
    try:
        # Ensure cache directory exists
        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        
        # Convert max_price_date to string if it's a datetime
        if isinstance(max_price_date, datetime):
            max_price_date_str = max_price_date.strftime('%Y-%m-%d')
        elif hasattr(max_price_date, 'strftime'):
            max_price_date_str = max_price_date.strftime('%Y-%m-%d')
        else:
            max_price_date_str = str(max_price_date) if max_price_date else None
        
        metadata = {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "success_rate": success_rate,
            "min_success_rate": MIN_SUCCESS_RATE,
            "tickers_total": total_tickers,
            "tickers_successful": successful_tickers,
            "tickers_failed": failed_tickers,
            "max_price_date": max_price_date_str,
            "cache_file": CACHE_PATH
        }
        
        with open(METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {METADATA_PATH}")
        
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")


def build_initial_cache(force_rebuild=False, years=DEFAULT_CACHE_YEARS):
    """
    Build the initial price cache.
    
    Args:
        force_rebuild: Force rebuild even if cache exists
        years: Number of years of history to keep
        
    Returns:
        Tuple of (success: bool, success_rate: float)
    """
    logger.info("=" * 70)
    logger.info("BUILD PRICE CACHE")
    logger.info("=" * 70)
    
    # Check if cache exists
    if not force_rebuild and os.path.exists(CACHE_PATH):
        info = get_cache_info()
        logger.info(f"Cache already exists:")
        logger.info(f"  Path: {info['path']}")
        logger.info(f"  Size: {info['size_mb']:.2f} MB")
        logger.info(f"  Tickers: {info['num_tickers']}")
        logger.info(f"  Days: {info['num_days']}")
        logger.info(f"  Date range: {info['date_range'][0]} to {info['date_range'][1]}")
        
        user_input = input("\nCache exists. Rebuild? (y/N): ").strip().lower()
        if user_input != 'y':
            logger.info("Keeping existing cache")
            return True, 1.0
    
    # Step 1: Collect all tickers
    all_tickers, wave_tickers, benchmark_tickers = collect_all_tickers()
    total_requested = len(all_tickers)
    
    # Step 2: Load existing price data
    existing_prices = load_existing_price_files()
    
    # Step 3: Load current cache (if exists)
    current_cache = load_cache()
    
    # Merge existing data with current cache
    if current_cache is not None and not current_cache.empty:
        logger.info("Merging existing cache...")
        cache_df = merge_cache_and_new_data(existing_prices, current_cache)
    else:
        cache_df = existing_prices
    
    # Step 4: Identify missing tickers
    if cache_df is not None and not cache_df.empty:
        available_tickers = set(cache_df.columns)
        missing_tickers = [t for t in all_tickers if t not in available_tickers or cache_df[t].isna().all()]
    else:
        missing_tickers = all_tickers
        cache_df = pd.DataFrame()
    
    logger.info(f"Missing tickers: {len(missing_tickers)}/{len(all_tickers)}")
    
    # Step 5: Fetch missing data (if network is available)
    all_failures = {}
    if missing_tickers:
        logger.info("Attempting to fetch missing tickers from yfinance...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Fetch in batches
        all_new_data = []
        
        for i in range(0, len(missing_tickers), BATCH_SIZE):
            batch = missing_tickers[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(missing_tickers) - 1) // BATCH_SIZE + 1
            
            logger.info(f"Fetching batch {batch_num}/{total_batches} ({len(batch)} tickers)...")
            
            try:
                batch_data, batch_failures = fetch_prices_batch(batch, start_date, end_date)
                
                if not batch_data.empty:
                    all_new_data.append(batch_data)
                    logger.info(f"  Fetched {len(batch_data)} days for {len(batch_data.columns)} tickers")
                
                all_failures.update(batch_failures)
                
            except Exception as e:
                logger.error(f"  Batch {batch_num} failed: {e}")
                for ticker in batch:
                    all_failures[ticker] = str(e)
        
        # Merge new data with cache
        if all_new_data:
            logger.info("Merging fetched data with cache...")
            
            new_data_df = all_new_data[0]
            for df in all_new_data[1:]:
                new_data_df = pd.concat([new_data_df, df], axis=1)
            
            cache_df = merge_cache_and_new_data(cache_df, new_data_df)
        
        # Log failures with ticker + reason
        if all_failures:
            logger.warning(f"Failed to fetch {len(all_failures)} tickers:")
            for ticker, reason in sorted(all_failures.items()):
                logger.warning(f"  {ticker}: {reason}")
    
    # Step 6: Trim cache to date range
    if not cache_df.empty:
        cache_df = trim_cache_to_date_range(cache_df, years)
    
    # Calculate success metrics
    successful_downloads = total_requested - len(all_failures)
    success_rate = successful_downloads / total_requested if total_requested > 0 else 0.0
    
    # Get the latest date in cache for metadata
    max_price_date = None
    if not cache_df.empty:
        max_price_date = cache_df.index[-1]
    
    # Step 7: Save cache and metadata
    if not cache_df.empty:
        logger.info("Saving cache...")
        save_cache(cache_df)
        
        # Save metadata file
        save_metadata(
            total_tickers=total_requested,
            successful_tickers=successful_downloads,
            failed_tickers=len(all_failures),
            success_rate=success_rate,
            max_price_date=max_price_date
        )
        
        # Print summary
        info = get_cache_info()
        logger.info("=" * 70)
        logger.info("CACHE BUILD SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Path: {info['path']}")
        logger.info(f"  Size: {info['size_mb']:.2f} MB")
        logger.info(f"  Tickers: {info['num_tickers']}")
        logger.info(f"  Days: {info['num_days']}")
        logger.info(f"  Date range: {info['date_range'][0]} to {info['date_range'][1]}")
        logger.info("")
        logger.info(f"  Total tickers requested: {total_requested}")
        logger.info(f"  Successful tickers: {successful_downloads}")
        logger.info(f"  Failed tickers: {len(all_failures)}")
        logger.info(f"  Success rate: {success_rate * 100:.2f}%")
        logger.info(f"  Threshold: {MIN_SUCCESS_RATE * 100:.2f}%")
        logger.info(f"  Latest price date: {max_price_date.strftime('%Y-%m-%d') if max_price_date else 'N/A'}")
        logger.info("=" * 70)
        
        # Determine success based on threshold
        meets_threshold = success_rate >= MIN_SUCCESS_RATE
        if meets_threshold:
            logger.info(f"✓ SUCCESS: Success rate {success_rate * 100:.2f}% meets threshold {MIN_SUCCESS_RATE * 100:.2f}%")
        else:
            logger.error(f"✗ FAILURE: Success rate {success_rate * 100:.2f}% below threshold {MIN_SUCCESS_RATE * 100:.2f}%")
        
        return meets_threshold, success_rate
    else:
        logger.error("No data available to build cache")
        # total_requested is defined earlier in the function (Step 1: Collect all tickers)
        # Still write metadata even on failure
        save_metadata(
            total_tickers=total_requested,
            successful_tickers=0,
            failed_tickers=total_requested,
            success_rate=0.0,
            max_price_date=None
        )
        return False, 0.0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Build initial price cache')
    parser.add_argument('--force', action='store_true', help='Force rebuild even if cache exists')
    parser.add_argument('--years', type=int, default=DEFAULT_CACHE_YEARS, help='Number of years of history')
    parser.add_argument('--skip-validation', action='store_true', help='Skip strict validation checks')
    
    args = parser.parse_args()
    
    success, success_rate = build_initial_cache(force_rebuild=args.force, years=args.years)
    
    # Strict exit codes:
    # - Exit 0 if success rate >= MIN_SUCCESS_RATE AND cache file exists AND validations pass
    # - Exit 1 otherwise
    cache_exists = os.path.exists(CACHE_PATH) and os.path.getsize(CACHE_PATH) > 0
    
    if not success or not cache_exists:
        if not cache_exists:
            logger.error(f"✗ Exiting with code 1 (cache file missing or empty)")
        else:
            logger.error(f"✗ Exiting with code 1 (success rate: {success_rate * 100:.2f}% below threshold)")
        sys.exit(1)
    
    # Run comprehensive validations (unless skipped)
    if not args.skip_validation:
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING COMPREHENSIVE VALIDATIONS")
        logger.info("=" * 70)
        
        try:
            from helpers.cache_validation import (
                validate_trading_day_freshness,
                validate_required_symbols,
                validate_cache_integrity
            )
            
            # Validation 1: Cache integrity
            integrity_result = validate_cache_integrity(CACHE_PATH)
            if not integrity_result['valid']:
                logger.error(f"✗ Cache integrity validation FAILED: {integrity_result['error']}")
                sys.exit(1)
            
            # Validation 2: Required symbols
            symbols_result = validate_required_symbols(CACHE_PATH)
            if not symbols_result['valid']:
                logger.error(f"✗ Required symbols validation FAILED: {symbols_result['error']}")
                sys.exit(1)
            
            # Validation 3: Trading-day freshness
            freshness_result = validate_trading_day_freshness(CACHE_PATH, max_market_feed_gap_days=5)
            if not freshness_result['valid']:
                logger.error(f"✗ Trading-day freshness validation FAILED: {freshness_result['error']}")
                sys.exit(1)
            
            logger.info("\n" + "=" * 70)
            logger.info("✓ ALL VALIDATIONS PASSED")
            logger.info("=" * 70)
            
        except ImportError as e:
            logger.warning(f"⚠️  Validation module not available: {e}")
            logger.warning("Skipping comprehensive validations")
        except Exception as e:
            logger.error(f"✗ Validation error: {e}")
            sys.exit(1)
    
    logger.info(f"✓ Exiting with code 0 (success rate: {success_rate * 100:.2f}%, cache valid)")
    sys.exit(0)


if __name__ == '__main__':
    main()

"""
Price Loader and Caching Module

This module provides a canonical price data loader with intelligent caching.
It replaces the scattered price loading logic across the codebase with a
single, reliable source of truth for price data.

Key Features:
- Centralized price cache (data/cache/prices_cache.parquet)
- Intelligent cache updates (only fetch missing/stale data)
- Ticker normalization and deduplication
- Graceful error handling (failed tickers get NaN values)
- Forward-filling limited to small gaps
- Configurable date range limits (default: last 5 years)
- Comprehensive logging of operations and failures

Usage:
    from helpers.price_loader import load_or_fetch_prices
    
    # Load prices for a list of tickers
    prices_df = load_or_fetch_prices(
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        start='2024-01-01',
        end='2024-12-31'
    )
"""

from __future__ import annotations

import os
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

# Try to import Streamlit for caching
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None

# Constants
CACHE_DIR = "data/cache"
CACHE_FILE = "prices_cache.parquet"
CACHE_PATH = os.path.join(CACHE_DIR, CACHE_FILE)
FAILED_TICKERS_FILE = "failed_tickers.csv"
FAILED_TICKERS_PATH = os.path.join(CACHE_DIR, FAILED_TICKERS_FILE)

# Cache configuration
DEFAULT_CACHE_YEARS = 5  # Keep last 5 years of data
MAX_FORWARD_FILL_DAYS = 3  # Maximum gap to forward-fill (CHANGED from 5 to 3)
MIN_REQUIRED_DAYS = 60  # Minimum trading days for readiness (CHANGED from 10 to 60)
MAX_STALE_DAYS = 2  # Data older than this many trading days is considered stale (CHANGED to use trading days)

# Download configuration
BATCH_SIZE = 50  # Maximum tickers per batch download
RETRY_ATTEMPTS = 1  # Number of retry attempts for failed downloads (CHANGED from 2 to 1)
RETRY_DELAY = 1.0  # Initial delay between retries (seconds)
REQUEST_TIMEOUT = 15  # Timeout for yfinance requests (seconds)

# Check environment variable for force cache refresh
FORCE_CACHE_REFRESH = os.environ.get('FORCE_CACHE_REFRESH', '0') == '1'

# Option B: Staleness threshold configuration (read from environment variables)
# OK: ≤14 days, DEGRADED: 15-30 days, STALE: >30 days
try:
    PRICE_CACHE_OK_DAYS = int(os.environ.get('PRICE_CACHE_OK_DAYS', '14'))
except (ValueError, TypeError):
    logger.warning("Invalid PRICE_CACHE_OK_DAYS environment variable, using default: 14")
    PRICE_CACHE_OK_DAYS = 14

try:
    PRICE_CACHE_DEGRADED_DAYS = int(os.environ.get('PRICE_CACHE_DEGRADED_DAYS', '30'))
except (ValueError, TypeError):
    logger.warning("Invalid PRICE_CACHE_DEGRADED_DAYS environment variable, using default: 30")
    PRICE_CACHE_DEGRADED_DAYS = 30

# Validate threshold constraint: DEGRADED_DAYS must be greater than OK_DAYS
if PRICE_CACHE_DEGRADED_DAYS <= PRICE_CACHE_OK_DAYS:
    logger.warning(
        f"PRICE_CACHE_DEGRADED_DAYS ({PRICE_CACHE_DEGRADED_DAYS}) must be greater than "
        f"PRICE_CACHE_OK_DAYS ({PRICE_CACHE_OK_DAYS}). Using defaults: OK=14, DEGRADED=30"
    )
    PRICE_CACHE_OK_DAYS = 14
    PRICE_CACHE_DEGRADED_DAYS = 30


def ensure_cache_directory() -> None:
    """Ensure the cache directory exists."""
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


def normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbols to match yfinance conventions.
    
    Examples:
        BRK.B -> BRK-B
        BF.B -> BF-B
        stETH-USD -> stETH-USD (already normalized)
    
    Args:
        ticker: Raw ticker symbol
        
    Returns:
        Normalized ticker symbol
    """
    if not ticker:
        return ticker
    
    # Replace dots with hyphens for class shares (e.g., BRK.B -> BRK-B)
    normalized = ticker.replace('.', '-')
    
    return normalized.strip().upper()


def deduplicate_tickers(tickers: List[str]) -> List[str]:
    """
    Deduplicate and normalize ticker list.
    
    Args:
        tickers: List of ticker symbols (may contain duplicates)
        
    Returns:
        Sorted list of unique, normalized ticker symbols
    """
    if not tickers:
        return []
    
    # Normalize and deduplicate
    normalized = set()
    for ticker in tickers:
        if ticker and ticker.strip():
            normalized.add(normalize_ticker(ticker))
    
    return sorted(list(normalized))


def collect_required_tickers(active_only: bool = True) -> List[str]:
    """
    Collect ONLY required tickers for active waves and their benchmarks.
    
    This is a deterministic and scoped function that gathers:
    - All tickers in wave holdings for active waves ONLY
    - All benchmark tickers for those active waves ONLY
    - Essential market indicators (SPY, ^VIX, BTC-USD)
    - EXCLUDES:
      * Universe tickers (e.g., top 200 crypto) unless explicitly in an active wave
      * Optional watchlist tickers
      * Inactive wave tickers
      * Unnecessary safe sleeve tickers not actually used by active waves
    
    SmartSafe cash waves are properly exempt from price requirements.
    
    Args:
        active_only: If True, only include tickers from active waves (default: True)
        
    Returns:
        Sorted list of unique, normalized ticker symbols
    """
    tickers = set()
    
    try:
        # Import waves_engine to get wave definitions
        from waves_engine import (
            get_all_wave_ids, 
            get_display_name_from_wave_id,
            WAVE_WEIGHTS, 
            BENCHMARK_WEIGHTS_STATIC,
            is_smartsafe_cash_wave
        )
        
        # Get all wave IDs
        all_wave_ids = get_all_wave_ids()
        
        # Filter to active waves if requested
        if active_only:
            # Read wave_registry.csv to get active wave_ids
            wave_registry_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'data', 
                'wave_registry.csv'
            )
            if os.path.exists(wave_registry_path):
                registry_df = pd.read_csv(wave_registry_path)
                # Filter to active waves only
                active_wave_ids = set(
                    registry_df[registry_df['active']]['wave_id'].tolist()
                )
                all_wave_ids = [wid for wid in all_wave_ids if wid in active_wave_ids]
                logger.info(f"Filtered to {len(all_wave_ids)} active waves")
                # Debug: Log first 10 active wave IDs
                logger.info(f"First 10 active wave IDs: {all_wave_ids[:10]}")
        
        # Collect tickers from wave holdings (ONLY from active waves)
        ticker_count_per_wave = {}  # Track tickers per wave for debugging
        for wave_id in all_wave_ids:
            # Skip SmartSafe cash waves - they don't need price data
            if is_smartsafe_cash_wave(wave_id):
                logger.debug(f"Skipping SmartSafe cash wave: {wave_id}")
                continue
            
            # Convert wave_id to display_name to look up in WAVE_WEIGHTS
            display_name = get_display_name_from_wave_id(wave_id)
            if not display_name:
                logger.warning(f"Could not get display name for wave_id: {wave_id}")
                continue
            
            wave_tickers_before = len(tickers)
            
            # Get wave weights (holdings) using display_name
            wave_weights = WAVE_WEIGHTS.get(display_name, [])
            for holding in wave_weights:
                if hasattr(holding, 'ticker'):
                    tickers.add(holding.ticker)
                elif isinstance(holding, dict) and 'ticker' in holding:
                    tickers.add(holding['ticker'])
            
            # Get benchmark weights for this wave using display_name
            benchmark_weights = BENCHMARK_WEIGHTS_STATIC.get(display_name, [])
            for benchmark in benchmark_weights:
                # Benchmark can be Holding object, (ticker, weight) tuple, or dict
                if hasattr(benchmark, 'ticker'):
                    # It's a Holding object
                    tickers.add(benchmark.ticker)
                elif isinstance(benchmark, tuple) and len(benchmark) >= 1:
                    # It's a (ticker, weight) tuple
                    tickers.add(benchmark[0])
                elif isinstance(benchmark, dict) and 'ticker' in benchmark:
                    # It's a dict
                    tickers.add(benchmark['ticker'])
            
            # Track how many tickers this wave contributed
            wave_tickers_count = len(tickers) - wave_tickers_before
            ticker_count_per_wave[wave_id] = wave_tickers_count
        
        # Log ticker counts per wave for debugging (first 10 waves)
        if ticker_count_per_wave:
            sample_waves = list(ticker_count_per_wave.items())[:10]
            logger.info(f"Ticker counts for first 10 waves: {sample_waves}")
        
        # Add essential market indicators (always included for system health)
        # These are used across multiple components for diagnostics
        essential_indicators = ['SPY', '^VIX', 'BTC-USD']
        tickers.update(essential_indicators)
        
        logger.info(f"Collected {len(tickers)} required tickers from {len(all_wave_ids)} active waves")
        
    except Exception as e:
        logger.error(f"Error collecting required tickers: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}", exc_info=True)
        # Fallback to a minimal set of essential indicators only
        tickers = {'SPY', 'QQQ', '^VIX', 'BTC-USD'}
    
    return deduplicate_tickers(list(tickers))


def save_failed_tickers(failures: Dict[str, str]) -> None:
    """
    Save failed tickers to CSV file for tracking.
    
    Args:
        failures: Dictionary mapping failed tickers to error reasons
    """
    if not failures:
        return
    
    ensure_cache_directory()
    
    try:
        # Create DataFrame from failures
        failed_df = pd.DataFrame([
            {
                'ticker': ticker,
                'reason': reason,
                'timestamp': datetime.utcnow().isoformat()
            }
            for ticker, reason in failures.items()
        ])
        
        # Append to existing file if it exists
        if os.path.exists(FAILED_TICKERS_PATH):
            existing_df = pd.read_csv(FAILED_TICKERS_PATH)
            failed_df = pd.concat([existing_df, failed_df], ignore_index=True)
            
            # Keep only the last 1000 entries to avoid file bloat
            if len(failed_df) > 1000:
                failed_df = failed_df.tail(1000)
        
        # Save to CSV
        failed_df.to_csv(FAILED_TICKERS_PATH, index=False)
        
        logger.info(f"Saved {len(failures)} failed tickers to {FAILED_TICKERS_PATH}")
        
    except Exception as e:
        logger.error(f"Error saving failed tickers: {e}")


def load_cache() -> Optional[pd.DataFrame]:
    """
    Load the price cache from disk.
    
    Returns:
        DataFrame with dates as index and tickers as columns, or None if cache doesn't exist
    """
    if not os.path.exists(CACHE_PATH):
        logger.info(f"Cache file not found: {CACHE_PATH}")
        return None
    
    try:
        cache_df = pd.read_parquet(CACHE_PATH)
        
        # Ensure index is datetime
        if not isinstance(cache_df.index, pd.DatetimeIndex):
            cache_df.index = pd.to_datetime(cache_df.index)
        
        # Sort by date
        cache_df = cache_df.sort_index()
        
        logger.info(
            f"Loaded cache: {len(cache_df)} days, {len(cache_df.columns)} tickers, "
            f"range: {cache_df.index[0].date()} to {cache_df.index[-1].date()}"
        )
        
        return cache_df
        
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        return None


def save_cache(cache_df: pd.DataFrame) -> None:
    """
    Save the price cache to disk.
    
    Args:
        cache_df: DataFrame with dates as index and tickers as columns
    """
    ensure_cache_directory()
    
    try:
        # Ensure index is datetime
        if not isinstance(cache_df.index, pd.DatetimeIndex):
            cache_df.index = pd.to_datetime(cache_df.index)
        
        # Sort by date
        cache_df = cache_df.sort_index()
        
        # Save to parquet
        cache_df.to_parquet(CACHE_PATH)
        
        logger.info(
            f"Saved cache: {len(cache_df)} days, {len(cache_df.columns)} tickers, "
            f"range: {cache_df.index[0].date()} to {cache_df.index[-1].date()}"
        )
        
    except Exception as e:
        logger.error(f"Error saving cache: {e}")
        raise


def trim_cache_to_date_range(
    cache_df: pd.DataFrame,
    years: int = DEFAULT_CACHE_YEARS
) -> pd.DataFrame:
    """
    Trim cache to keep only the last N years of data.
    
    Args:
        cache_df: DataFrame with dates as index
        years: Number of years to keep (default: 5)
        
    Returns:
        Trimmed DataFrame
    """
    if cache_df.empty:
        return cache_df
    
    # Calculate cutoff date
    cutoff_date = datetime.utcnow() - timedelta(days=365 * years)
    
    # Filter to dates after cutoff
    trimmed = cache_df[cache_df.index >= cutoff_date]
    
    if len(trimmed) < len(cache_df):
        logger.info(
            f"Trimmed cache from {len(cache_df)} to {len(trimmed)} days "
            f"(keeping last {years} years)"
        )
    
    return trimmed


def fetch_prices_batch(
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    retry: bool = True
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Fetch prices for a batch of tickers from yfinance with retry logic.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for data
        end_date: End date for data
        retry: Whether to retry on failure (default: True)
        
    Returns:
        Tuple of (prices_df, failures_dict):
        - prices_df: DataFrame with dates as index and tickers as columns
        - failures_dict: Dict mapping failed tickers to error messages
    """
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance is not available")
        return pd.DataFrame(), {ticker: "yfinance not available" for ticker in tickers}
    
    if not tickers:
        return pd.DataFrame(), {}
    
    failures = {}
    
    # First attempt
    prices, failures = _fetch_prices_batch_impl(tickers, start_date, end_date)
    
    # Retry failed tickers once if requested
    if retry and failures and RETRY_ATTEMPTS > 0:
        failed_tickers = list(failures.keys())
        logger.info(f"Retrying {len(failed_tickers)} failed tickers...")
        
        import time
        time.sleep(RETRY_DELAY)
        
        retry_prices, retry_failures = _fetch_prices_batch_impl(
            failed_tickers, start_date, end_date
        )
        
        # Merge successful retries into prices
        if not retry_prices.empty:
            if prices.empty:
                prices = retry_prices
            else:
                prices = pd.concat([prices, retry_prices], axis=1)
            
            # Remove successfully retried tickers from failures
            for ticker in retry_prices.columns:
                if ticker in failures:
                    del failures[ticker]
        
        # Update failures with retry failures
        failures.update(retry_failures)
    
    return prices, failures


def _fetch_prices_batch_impl(
    tickers: List[str],
    start_date: datetime,
    end_date: datetime
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Internal implementation of fetch_prices_batch.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Tuple of (prices_df, failures_dict)
    """
    failures = {}
    
    try:
        # Download data
        logger.info(f"Fetching {len(tickers)} tickers from {start_date.date()} to {end_date.date()}")
        
        data = yf.download(
            tickers=tickers,
            start=start_date.strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False,
            group_by='ticker',
            timeout=REQUEST_TIMEOUT
        )
        
        if data.empty:
            logger.warning(f"No data returned for {len(tickers)} tickers")
            return pd.DataFrame(), {ticker: "No data returned" for ticker in tickers}
        
        # Extract Close prices
        if len(tickers) == 1:
            # Single ticker case
            if 'Close' in data.columns:
                prices = data[['Close']].rename(columns={'Close': tickers[0]})
            else:
                return pd.DataFrame(), {tickers[0]: "No Close column in data"}
        else:
            # Multiple tickers case
            frames = []
            for ticker in tickers:
                try:
                    if (ticker, 'Close') in data.columns:
                        frames.append(data[(ticker, 'Close')].rename(ticker))
                    elif ticker in data.columns and 'Close' in data[ticker].columns:
                        frames.append(data[ticker]['Close'].rename(ticker))
                    else:
                        failures[ticker] = "Missing Close column"
                except (KeyError, AttributeError) as e:
                    failures[ticker] = f"Data extraction error: {str(e)}"
            
            if frames:
                prices = pd.concat(frames, axis=1)
            else:
                return pd.DataFrame(), failures
        
        # Ensure datetime index
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        
        # Check for failed tickers
        for ticker in tickers:
            if ticker not in prices.columns and ticker not in failures:
                failures[ticker] = "Ticker not in result"
        
        logger.info(
            f"Fetched {len(prices)} days for {len(prices.columns)}/{len(tickers)} tickers, "
            f"{len(failures)} failed"
        )
        
        return prices, failures
        
    except Exception as e:
        logger.error(f"Batch download error: {e}")
        return pd.DataFrame(), {ticker: f"Download error: {str(e)}" for ticker in tickers}


def apply_forward_fill(
    df: pd.DataFrame,
    max_gap_days: int = MAX_FORWARD_FILL_DAYS
) -> pd.DataFrame:
    """
    Apply forward-filling with a maximum gap limit.
    
    Args:
        df: DataFrame with price data
        max_gap_days: Maximum number of days to forward-fill (default: 5)
        
    Returns:
        DataFrame with limited forward-filling applied
    """
    if df.empty:
        return df
    
    # Forward fill with limit
    filled = df.ffill(limit=max_gap_days)
    
    return filled


def identify_missing_and_stale_tickers(
    cache_df: Optional[pd.DataFrame],
    tickers: List[str],
    start_date: datetime,
    end_date: datetime,
    max_stale_days: int = MAX_STALE_DAYS
) -> Tuple[List[str], List[str]]:
    """
    Identify which tickers are missing or have stale data in the cache.
    
    Args:
        cache_df: Cached price data (or None if no cache)
        tickers: Requested ticker symbols
        start_date: Requested start date
        end_date: Requested end date
        max_stale_days: Maximum age before data is considered stale
        
    Returns:
        Tuple of (missing_tickers, stale_tickers)
    """
    if cache_df is None or cache_df.empty:
        return tickers, []
    
    missing = []
    stale = []
    
    # Calculate stale cutoff date (relative to end_date or now)
    reference_date = min(end_date, datetime.now())
    stale_cutoff = reference_date - timedelta(days=max_stale_days)
    
    for ticker in tickers:
        if ticker not in cache_df.columns:
            # Ticker not in cache
            missing.append(ticker)
        else:
            # Check if ticker has data in requested range
            ticker_data = cache_df[ticker].dropna()
            
            if ticker_data.empty:
                # No valid data for this ticker
                missing.append(ticker)
            elif ticker_data.index[0] > start_date:
                # Data doesn't cover requested start date
                missing.append(ticker)
            elif ticker_data.index[-1] < stale_cutoff:
                # Latest data is too old
                stale.append(ticker)
    
    return missing, stale


def merge_cache_and_new_data(
    cache_df: Optional[pd.DataFrame],
    new_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge existing cache with newly fetched data.
    
    Args:
        cache_df: Existing cache (or None)
        new_df: Newly fetched data
        
    Returns:
        Merged DataFrame
    """
    if cache_df is None or cache_df.empty:
        return new_df
    
    if new_df.empty:
        return cache_df
    
    # Combine on index (dates)
    merged = cache_df.combine_first(new_df)
    
    # Update with new data (overwrites cache for overlapping dates/tickers)
    for ticker in new_df.columns:
        if ticker in merged.columns:
            # Update existing ticker with new data
            new_dates = new_df.index
            merged.loc[new_dates, ticker] = new_df[ticker]
    
    return merged


def get_canonical_prices(
    tickers: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None
) -> pd.DataFrame:
    """
    Load price data from the canonical cache ONLY. Never fetches from network.
    
    This is the single source of truth for price data in the system.
    - Loads ONLY from data/cache/prices_cache.parquet
    - Never performs downloads under any circumstances
    - Returns what is available in cache or NaN for missing tickers
    
    Args:
        tickers: List of ticker symbols (optional - if None, returns all cached tickers)
        start: Start date (YYYY-MM-DD) or None for all available dates
        end: End date (YYYY-MM-DD) or None for all available dates
        
    Returns:
        DataFrame with:
        - Index: Datetime (trading days)
        - Columns: Ticker symbols
        - Values: Adjusted close prices (NaN for missing tickers)
    
    Example:
        >>> prices = get_canonical_prices(['AAPL', 'MSFT'])
        >>> prices = get_canonical_prices()  # Get all cached tickers
    """
    logger.info("=" * 70)
    logger.info("CANONICAL PRICE GETTER: Loading from cache only")
    logger.info("=" * 70)
    
    # Load cache
    cache_df = load_cache()
    
    if cache_df is None or cache_df.empty:
        logger.warning("Cache is empty or does not exist")
        # Return empty DataFrame
        result = pd.DataFrame()
        if tickers:
            result = pd.DataFrame(columns=deduplicate_tickers(tickers))
        result.index.name = 'Date'
        return result
    
    # Parse date range if provided
    if start is not None:
        start_date = pd.to_datetime(start)
        cache_df = cache_df[cache_df.index >= start_date]
    
    if end is not None:
        end_date = pd.to_datetime(end)
        cache_df = cache_df[cache_df.index <= end_date]
    
    # If tickers specified, filter to those tickers
    if tickers:
        tickers = deduplicate_tickers(tickers)
        result = cache_df.copy()
        
        # Add NaN columns for requested tickers not in cache
        for ticker in tickers:
            if ticker not in result.columns:
                result[ticker] = np.nan
        
        # Keep only requested tickers
        result = result[tickers]
    else:
        result = cache_df
    
    logger.info(f"Loaded from cache: {len(result)} days, {len(result.columns)} tickers")
    logger.info("=" * 70)
    
    return result


def load_or_fetch_prices(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    force_fetch: bool = False
) -> pd.DataFrame:
    """
    Load price data from cache (with optional explicit fetch).
    
    IMPORTANT: This function NO LONGER automatically fetches missing data.
    - By default, loads ONLY from cache (same as get_canonical_prices)
    - Only fetches if force_fetch=True is explicitly set
    - Network fetching should ONLY occur via:
      1. User clicking "Refresh Prices Cache" button
      2. Explicit FORCE_CACHE_REFRESH=1 environment variable
    
    This is a compatibility wrapper around get_canonical_prices() that allows
    explicit fetching when needed.
    
    Args:
        tickers: List of ticker symbols (may contain duplicates)
        start: Start date (YYYY-MM-DD) or None for auto-calculate
        end: End date (YYYY-MM-DD) or None for today
        force_fetch: If True, fetch data even if available in cache (default: False)
        
    Returns:
        DataFrame with:
        - Index: Datetime (trading days)
        - Columns: Ticker symbols
        - Values: Adjusted close prices (NaN for failed tickers)
    
    Example:
        >>> prices = load_or_fetch_prices(['AAPL', 'MSFT'])  # Loads from cache only
        >>> prices = load_or_fetch_prices(['AAPL'], force_fetch=True)  # Explicitly fetches
    """
    logger.info("=" * 70)
    logger.info("PRICE LOADER: Starting load_or_fetch_prices")
    logger.info("=" * 70)
    
    # Step 1: Deduplicate and normalize tickers
    tickers = deduplicate_tickers(tickers)
    
    if not tickers:
        logger.warning("No valid tickers provided")
        return pd.DataFrame()
    
    logger.info(f"Requested tickers: {len(tickers)}")
    logger.debug(f"Tickers: {tickers}")
    
    # Step 2: Parse and validate date range
    if end is None:
        end_date = datetime.now()
    else:
        end_date = pd.to_datetime(end)
    
    if start is None:
        # Default to 5 years of data
        start_date = end_date - timedelta(days=365 * DEFAULT_CACHE_YEARS)
    else:
        start_date = pd.to_datetime(start)
    
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Step 3: Check if explicit fetch is requested
    # ONLY fetch if force_fetch=True OR FORCE_CACHE_REFRESH env var is set
    should_fetch = force_fetch or FORCE_CACHE_REFRESH
    
    if not should_fetch:
        # NO IMPLICIT FETCHING - just load from cache
        logger.info("Implicit fetching disabled - loading from cache only")
        result = get_canonical_prices(tickers, start=start, end=end)
        
        # Apply limited forward-filling
        result = apply_forward_fill(result, MAX_FORWARD_FILL_DAYS)
        
        logger.info(f"Returning: {len(result)} days, {len(result.columns)} tickers")
        logger.info(
            f"Coverage: {result.notna().any(axis=1).sum()} days with data, "
            f"{result.isna().all().sum()} tickers fully missing"
        )
        logger.info("=" * 70)
        
        return result
    
    # EXPLICIT FETCH PATH - only when force_fetch=True or FORCE_CACHE_REFRESH=1
    logger.warning("EXPLICIT FETCH requested - will download missing/stale data")
    
    # Step 4: Load existing cache
    cache_df = load_cache()
    
    # Step 5: Identify missing and stale tickers
    to_fetch = []
    if force_fetch or FORCE_CACHE_REFRESH:
        logger.info("Force fetch enabled - will refresh all tickers")
        to_fetch = tickers
    else:
        missing_tickers, stale_tickers = identify_missing_and_stale_tickers(
            cache_df, tickers, start_date, end_date
        )
        
        to_fetch = list(set(missing_tickers + stale_tickers))
        
        logger.info(f"Cache status: {len(missing_tickers)} missing, {len(stale_tickers)} stale")
        if to_fetch:
            logger.info(f"Will fetch {len(to_fetch)} ticker(s): {to_fetch}")
        else:
            logger.info("All requested data available in cache")
    
    # Step 6: Fetch missing/stale data
    failures = {}
    new_data = pd.DataFrame()
    
    if to_fetch:
        # Fetch in batches to avoid overwhelming the API
        for i in range(0, len(to_fetch), BATCH_SIZE):
            batch = to_fetch[i:i + BATCH_SIZE]
            logger.info(f"Fetching batch {i//BATCH_SIZE + 1}/{(len(to_fetch)-1)//BATCH_SIZE + 1}")
            
            batch_data, batch_failures = fetch_prices_batch(batch, start_date, end_date)
            
            if not batch_data.empty:
                if new_data.empty:
                    new_data = batch_data
                else:
                    new_data = pd.concat([new_data, batch_data], axis=1)
            
            failures.update(batch_failures)
        
        # Save failed tickers to CSV
        if failures:
            save_failed_tickers(failures)
            logger.warning(f"Failed to fetch {len(failures)} ticker(s):")
            for ticker, reason in list(failures.items())[:10]:  # Show first 10
                logger.warning(f"  {ticker}: {reason}")
            if len(failures) > 10:
                logger.warning(f"  ... and {len(failures) - 10} more")
    
    # Step 7: Merge cache and new data
    if not new_data.empty:
        merged = merge_cache_and_new_data(cache_df, new_data)
        
        # Trim to date range
        merged = trim_cache_to_date_range(merged, DEFAULT_CACHE_YEARS)
        
        # Save updated cache
        save_cache(merged)
        cache_df = merged
    
    # Step 8: Extract requested data from cache
    if cache_df is None or cache_df.empty:
        logger.warning("No data available")
        # Return empty DataFrame with requested tickers as columns (all NaN)
        result = pd.DataFrame(columns=tickers)
        result.index.name = 'Date'
        return result
    
    # Filter to requested date range
    result = cache_df[
        (cache_df.index >= start_date) & (cache_df.index <= end_date)
    ].copy()
    
    # Ensure all requested tickers are present (add NaN columns for failed tickers)
    for ticker in tickers:
        if ticker not in result.columns:
            result[ticker] = np.nan
            if ticker not in failures:
                failures[ticker] = "Not available in cache or fetch failed"
    
    # Keep only requested tickers and sort columns
    result = result[tickers]
    
    # Apply limited forward-filling
    result = apply_forward_fill(result, MAX_FORWARD_FILL_DAYS)
    
    logger.info(f"Returning: {len(result)} days, {len(result.columns)} tickers")
    logger.info(
        f"Coverage: {result.notna().any(axis=1).sum()} days with data, "
        f"{result.isna().all().sum()} tickers fully missing"
    )
    logger.info("=" * 70)
    
    return result


def refresh_price_cache(active_only: bool = True) -> Dict[str, Any]:
    """
    Refresh the price cache by fetching all required tickers.
    
    This function should be called explicitly by the user via UI button
    or when FORCE_CACHE_REFRESH environment variable is set.
    
    Args:
        active_only: If True, only fetch tickers for active waves (default: True)
        
    Returns:
        Dictionary with:
        - success: bool
        - tickers_fetched: int
        - tickers_failed: int
        - failures: Dict[str, str]
        - cache_info: Dict with cache metadata
    """
    logger.info("=" * 70)
    logger.info("REFRESH PRICE CACHE: Starting explicit cache refresh")
    logger.info("=" * 70)
    
    # Collect required tickers
    tickers = collect_required_tickers(active_only=active_only)
    
    logger.info(f"Collected {len(tickers)} required tickers (active_only={active_only})")
    
    # Fetch prices with force_fetch=True
    prices_df = load_or_fetch_prices(tickers, force_fetch=True)
    
    # Get cache info
    cache_info = get_cache_info()
    
    # Load failed tickers from CSV
    failures = {}
    if os.path.exists(FAILED_TICKERS_PATH):
        try:
            failed_df = pd.read_csv(FAILED_TICKERS_PATH)
            # Get most recent failure for each ticker
            if not failed_df.empty:
                latest_failures = failed_df.sort_values('timestamp').groupby('ticker').last()
                failures = dict(zip(latest_failures.index, latest_failures['reason']))
        except Exception as e:
            logger.error(f"Error loading failed tickers: {e}")
    
    result = {
        'success': not prices_df.empty,
        'tickers_requested': len(tickers),
        'tickers_fetched': len(prices_df.columns) if not prices_df.empty else 0,
        'tickers_failed': len(failures),
        'failures': failures,
        'cache_info': cache_info
    }
    
    logger.info(f"Cache refresh complete: {result['tickers_fetched']}/{result['tickers_requested']} tickers fetched")
    logger.info("=" * 70)
    
    return result


# Streamlit cached version
if STREAMLIT_AVAILABLE:
    @st.cache_data(ttl=3600, show_spinner=False)
    def load_or_fetch_prices_cached(
        tickers: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Streamlit-cached version of load_or_fetch_prices.
        
        Cache TTL: 1 hour (3600 seconds)
        
        Args:
            tickers: List of ticker symbols
            start: Start date (YYYY-MM-DD) or None
            end: End date (YYYY-MM-DD) or None
            
        Returns:
            DataFrame with price data
        """
        return load_or_fetch_prices(tickers, start, end)
else:
    # Fallback when Streamlit is not available
    load_or_fetch_prices_cached = load_or_fetch_prices


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about the current canonical price cache.
    
    This provides metadata about the single source of truth for price data.
    
    Returns:
        Dictionary with cache metadata:
        - exists: bool - Cache file exists
        - path: str - Canonical cache file path
        - size_mb: float - File size in MB
        - num_tickers: int - Number of tickers in cache
        - num_days: int - Number of trading days in cache
        - date_range: tuple - (start_date, end_date) as strings
        - last_updated: str - Most recent date in cache
        - tickers: list - All ticker symbols in cache
        - is_stale: bool - Whether cache is older than MAX_STALE_DAYS
        - days_stale: int - Days since last update
    """
    info = {
        'exists': os.path.exists(CACHE_PATH),
        'path': CACHE_PATH,
        'size_mb': 0.0,
        'num_tickers': 0,
        'num_days': 0,
        'date_range': (None, None),
        'last_updated': None,
        'tickers': [],
        'is_stale': False,
        'days_stale': 0
    }
    
    if not info['exists']:
        return info
    
    try:
        # Get file size
        info['size_mb'] = os.path.getsize(CACHE_PATH) / (1024 * 1024)
        
        # Load cache
        cache_df = load_cache()
        
        if cache_df is not None and not cache_df.empty:
            info['num_tickers'] = len(cache_df.columns)
            info['num_days'] = len(cache_df)
            
            start_date = cache_df.index[0].strftime('%Y-%m-%d')
            end_date = cache_df.index[-1].strftime('%Y-%m-%d')
            
            info['date_range'] = (start_date, end_date)
            info['last_updated'] = end_date
            info['tickers'] = sorted(cache_df.columns.tolist())
            
            # Calculate staleness
            days_since_update = (datetime.utcnow() - cache_df.index[-1]).days
            info['days_stale'] = days_since_update
            info['is_stale'] = days_since_update > MAX_STALE_DAYS
    
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
    
    return info


def clear_cache() -> bool:
    """
    Clear the price cache.
    
    Returns:
        True if cache was cleared successfully, False otherwise
    """
    if not os.path.exists(CACHE_PATH):
        logger.info("Cache does not exist, nothing to clear")
        return True
    
    try:
        os.remove(CACHE_PATH)
        logger.info(f"Cleared cache: {CACHE_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return False


def get_trading_days_ago(target_date: datetime, trading_days_back: int = 1) -> datetime:
    """
    Calculate a date that is N trading days before the target date.
    Trading days exclude weekends (Saturday and Sunday).
    
    Args:
        target_date: The reference date
        trading_days_back: Number of trading days to go back (default: 1)
        
    Returns:
        datetime object representing the date N trading days ago
    
    Example:
        >>> # If today is Monday, 1 trading day ago is Friday
        >>> # If today is Sunday, 1 trading day ago is Friday (2 calendar days)
    """
    current = target_date
    trading_days_counted = 0
    
    # Move back day by day until we've counted enough trading days
    while trading_days_counted < trading_days_back:
        current = current - timedelta(days=1)
        # Check if it's a weekday (Monday=0, Sunday=6)
        if current.weekday() < 5:  # Monday-Friday
            trading_days_counted += 1
    
    return current


def check_cache_readiness(
    min_trading_days: int = MIN_REQUIRED_DAYS,
    max_stale_days: int = MAX_STALE_DAYS,
    active_only: bool = True
) -> Dict[str, Any]:
    """
    Check if the canonical price cache is ready for use.
    
    This provides a single source of truth for system readiness diagnostics.
    All execution logic should use this same cache and readiness determination.
    
    Readiness criteria:
    - Cache file exists
    - Has minimum trading days of data
    - Data staleness is evaluated using Option B thresholds (calendar days):
      * OK: ≤14 days (PRICE_CACHE_OK_DAYS)
      * DEGRADED: 15-30 days (PRICE_CACHE_DEGRADED_DAYS)
      * STALE: >30 days
    - Contains required tickers for active waves
    
    Args:
        min_trading_days: Minimum trading days required (default: 60)
        max_stale_days: Legacy parameter, retained for compatibility (default: 2)
        active_only: If True, only check tickers for active waves (default: True)
        
    Returns:
        Dictionary with:
        - ready: bool - Overall readiness status
        - exists: bool - Cache file exists
        - num_days: int - Number of trading days in cache
        - num_tickers: int - Number of tickers in cache
        - max_date: str - Most recent date in cache
        - days_stale: int - Calendar days since most recent data
        - required_tickers: int - Number of tickers required for active waves
        - missing_tickers: List[str] - Required tickers not in cache
        - extra_tickers: List[str] - Tickers in cache but not required (harmless)
        - failed_tickers: List[str] - Tickers that failed to download (from failed_tickers.csv)
        - status: str - Human-readable status message
        - status_code: str - Status code (READY, MISSING, EMPTY, INSUFFICIENT, STALE, DEGRADED)
    """
    cache_df = load_cache()
    required_tickers = collect_required_tickers(active_only=active_only)
    
    result = {
        'ready': False,
        'exists': os.path.exists(CACHE_PATH),
        'num_days': 0,
        'num_tickers': 0,
        'max_date': None,
        'days_stale': None,
        'required_tickers': len(required_tickers),
        'missing_tickers': [],
        'extra_tickers': [],
        'failed_tickers': [],
        'status': 'Unknown',
        'status_code': 'UNKNOWN'
    }
    
    if not result['exists']:
        result['status'] = 'MISSING - Cache file does not exist'
        result['status_code'] = 'MISSING'
        result['missing_tickers'] = required_tickers
        return result
    
    if cache_df is None or cache_df.empty:
        result['status'] = 'EMPTY - Cache file is empty'
        result['status_code'] = 'EMPTY'
        result['missing_tickers'] = required_tickers
        return result
    
    result['num_days'] = len(cache_df)
    result['num_tickers'] = len(cache_df.columns)
    result['max_date'] = cache_df.index[-1].strftime('%Y-%m-%d')
    
    # Calculate staleness (calendar days)
    max_date = cache_df.index[-1].to_pydatetime()
    now = datetime.utcnow()
    
    # Calculate calendar days for reporting purposes
    days_since_update = (now - max_date).days
    result['days_stale'] = days_since_update
    
    # Check minimum trading days
    if result['num_days'] < min_trading_days:
        result['status'] = f'INSUFFICIENT - Only {result["num_days"]} trading days (need {min_trading_days})'
        result['status_code'] = 'INSUFFICIENT'
        return result
    
    # Option B: Check staleness using three-tier system (calendar days)
    # OK: ≤14 days, DEGRADED: 15-30 days, STALE: >30 days
    # Use environment-configurable thresholds
    staleness_status = None
    if days_since_update > PRICE_CACHE_DEGRADED_DAYS:
        # Data is older than 30 days (DEGRADED threshold) → STALE
        result['status'] = f'STALE - Data is {days_since_update} days old (>{PRICE_CACHE_DEGRADED_DAYS} days)'
        result['status_code'] = 'STALE'
        staleness_status = 'STALE'
    elif days_since_update > PRICE_CACHE_OK_DAYS:
        # Data is 15-30 days old → DEGRADED (don't return yet, check for missing tickers)
        staleness_status = 'DEGRADED'
    # else: Data is ≤14 days old → OK (no staleness issue)
    
    # Check required tickers
    cached_tickers = set(cache_df.columns)
    required_set = set(required_tickers)
    
    missing = sorted(list(required_set - cached_tickers))
    extra = sorted(list(cached_tickers - required_set))
    
    result['missing_tickers'] = missing
    result['extra_tickers'] = extra
    
    # Load failed tickers from CSV to provide context
    if os.path.exists(FAILED_TICKERS_PATH):
        try:
            failed_df = pd.read_csv(FAILED_TICKERS_PATH)
            if not failed_df.empty:
                # Get unique failed tickers (use ticker_normalized column)
                if 'ticker_normalized' in failed_df.columns:
                    failed_tickers = failed_df['ticker_normalized'].unique().tolist()
                elif 'ticker' in failed_df.columns:
                    failed_tickers = failed_df['ticker'].unique().tolist()
                else:
                    failed_tickers = []
                
                # Only include failed tickers that are in the required set
                result['failed_tickers'] = sorted(list(set(failed_tickers) & required_set))
        except Exception as e:
            logger.error(f"Error loading failed tickers: {e}")
    
    # Determine final readiness based on missing tickers and staleness
    if missing:
        result['status'] = f'DEGRADED - Missing {len(missing)} required tickers'
        result['status_code'] = 'DEGRADED'
        result['ready'] = False
        return result
    
    # If data is STALE (>30 days), mark as not ready
    if staleness_status == 'STALE':
        result['ready'] = False
        return result
    
    # If data is DEGRADED (15-30 days), mark as ready but with degraded status
    if staleness_status == 'DEGRADED':
        result['ready'] = True  # Still operational
        result['status'] = f'DEGRADED - Data is {days_since_update} days old ({PRICE_CACHE_OK_DAYS + 1}-{PRICE_CACHE_DEGRADED_DAYS} days)'
        result['status_code'] = 'DEGRADED'
        return result
    
    # All checks passed - data is OK (≤14 days old)
    result['ready'] = True
    result['status'] = 'READY'
    result['status_code'] = 'READY'
    
    return result

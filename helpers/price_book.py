"""
PRICE_BOOK - Single Canonical Price Source of Truth

This module provides THE authoritative price data loader for the entire application.
All price data must flow through this module to ensure consistency and prevent 
"two truths" problems.

Key Principles:
- ONE canonical cache file: data/cache/prices_cache.parquet
- NO implicit fetching - all fetches are explicit and controlled (ALLOW_NETWORK_FETCH=False by default)
- PRICE_BOOK is a DataFrame: index=dates, columns=tickers, values=close prices
- All readiness, health, execution, and diagnostics use the SAME PRICE_BOOK

Usage:
    from helpers.price_book import get_price_book, get_price_book_meta, PRICE_BOOK
    
    # Load prices for active tickers (cache-only, no fetching)
    prices = get_price_book(active_tickers=['SPY', 'QQQ', 'NVDA'], mode='Standard')
    
    # Or use the singleton PRICE_BOOK (loads on first access)
    from helpers.price_book import PRICE_BOOK
    
    # Get metadata about the price book
    meta = get_price_book_meta(prices)
    print(f"Date range: {meta['date_min']} to {meta['date_max']}")
"""

import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Canonical cache configuration
CACHE_DIR = "data/cache"
CACHE_FILE = "prices_cache.parquet"
CANONICAL_CACHE_PATH = os.path.join(CACHE_DIR, CACHE_FILE)

# Environment variable to control fetching
# IMPORTANT: Set to False in production/cloud to prevent automatic fetching
PRICE_FETCH_ENABLED = os.environ.get('PRICE_FETCH_ENABLED', 'false').lower() in ('true', '1', 'yes')

# Alias for consistency with problem statement
ALLOW_NETWORK_FETCH = PRICE_FETCH_ENABLED

# Import from price_loader for supporting functions
try:
    from helpers.price_loader import (
        load_cache,
        save_cache,
        collect_required_tickers,
        normalize_ticker,
        deduplicate_tickers,
        load_or_fetch_prices,
        get_cache_info,
        check_cache_readiness,
        CACHE_PATH
    )
except ImportError as e:
    logger.warning(f"Could not import from price_loader: {e}")
    # Set all to None for proper error handling
    load_cache = None
    save_cache = None
    collect_required_tickers = None
    normalize_ticker = None
    deduplicate_tickers = None
    load_or_fetch_prices = None
    get_cache_info = None
    check_cache_readiness = None
    CACHE_PATH = CANONICAL_CACHE_PATH  # Use our own constant as fallback


def get_price_book(
    active_tickers: Optional[List[str]] = None,
    mode: str = 'Standard',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Get the canonical PRICE_BOOK - the single source of truth for all price data.
    
    This function NEVER fetches from network. It only loads from the canonical cache.
    Use rebuild_price_cache() to explicitly update the cache.
    
    Args:
        active_tickers: List of tickers to include (if None, returns all cached tickers)
        mode: Wave mode (not used for cache loading, kept for compatibility)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        
    Returns:
        DataFrame with:
        - Index: DatetimeIndex (trading days)
        - Columns: Ticker symbols
        - Values: Close prices (NaN for missing data)
        
    Example:
        >>> prices = get_price_book(active_tickers=['SPY', 'QQQ'])
        >>> prices = get_price_book()  # All cached tickers
    """
    logger.info("=" * 70)
    logger.info("PRICE_BOOK: Loading canonical price data (cache-only)")
    logger.info(f"Source: {CANONICAL_CACHE_PATH}")
    logger.info("=" * 70)
    
    # Check if required functions are available
    if load_cache is None or deduplicate_tickers is None:
        logger.error("Required price_loader functions not available")
        result = pd.DataFrame()
        if active_tickers:
            result = pd.DataFrame(columns=active_tickers)  # Use raw tickers if deduplicate not available
        result.index.name = 'Date'
        return result
    
    # Load from canonical cache (never fetches)
    cache_df = load_cache()
    
    if cache_df is None or cache_df.empty:
        logger.warning("PRICE_BOOK is empty - cache does not exist or is empty")
        logger.warning("Use rebuild_price_cache() to populate the cache")
        # Return empty DataFrame with requested tickers as columns
        result = pd.DataFrame()
        if active_tickers:
            result = pd.DataFrame(columns=deduplicate_tickers(active_tickers))
        result.index.name = 'Date'
        return result
    
    # Apply date filters if provided
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        cache_df = cache_df[cache_df.index >= start_dt]
    
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        cache_df = cache_df[cache_df.index <= end_dt]
    
    # Filter to requested tickers if provided
    if active_tickers:
        tickers = deduplicate_tickers(active_tickers)
        result = cache_df.copy()
        
        # Add NaN columns for requested tickers not in cache
        for ticker in tickers:
            if ticker not in result.columns:
                result[ticker] = np.nan
                logger.warning(f"Ticker {ticker} not found in PRICE_BOOK")
        
        # Keep only requested tickers
        result = result[tickers]
    else:
        result = cache_df
    
    logger.info(f"PRICE_BOOK loaded: {len(result)} days × {len(result.columns)} tickers")
    if not result.empty:
        logger.info(f"Date range: {result.index[0].date()} to {result.index[-1].date()}")
    logger.info("=" * 70)
    
    return result


def get_price_book_meta(price_book: pd.DataFrame) -> Dict[str, Any]:
    """
    Get metadata about the PRICE_BOOK DataFrame.
    
    This provides diagnostic information about the price data without
    loading it again. All diagnostics should use this function to ensure
    they report on the actual PRICE_BOOK used for execution.
    
    Args:
        price_book: The PRICE_BOOK DataFrame returned by get_price_book()
        
    Returns:
        Dictionary with:
        - date_min: str - Earliest date (YYYY-MM-DD)
        - date_max: str - Latest date (YYYY-MM-DD)
        - rows: int - Number of trading days
        - cols: int - Number of tickers
        - tickers_count: int - Same as cols (for clarity)
        - tickers: List[str] - All ticker symbols
        - is_empty: bool - Whether PRICE_BOOK is empty
        - cache_path: str - Canonical cache file path
    """
    if price_book is None or price_book.empty:
        return {
            'date_min': None,
            'date_max': None,
            'rows': 0,
            'cols': 0,
            'tickers_count': 0,
            'tickers': [],
            'is_empty': True,
            'cache_path': CANONICAL_CACHE_PATH
        }
    
    return {
        'date_min': price_book.index[0].strftime('%Y-%m-%d'),
        'date_max': price_book.index[-1].strftime('%Y-%m-%d'),
        'rows': len(price_book),
        'cols': len(price_book.columns),
        'tickers_count': len(price_book.columns),
        'tickers': sorted(price_book.columns.tolist()),
        'is_empty': False,
        'cache_path': CANONICAL_CACHE_PATH
    }


def rebuild_price_cache(active_only: bool = True) -> Dict[str, Any]:
    """
    Rebuild the canonical price cache by fetching data for active tickers.
    
    This is the ONLY function that should trigger network fetching.
    It must be called explicitly (e.g., via button click).
    
    Workflow:
    1. Check PRICE_FETCH_ENABLED environment variable
    2. Collect required tickers (active waves only)
    3. Fetch prices for those tickers
    4. Write to canonical cache file
    5. Return summary of what was fetched
    
    Args:
        active_only: If True, only fetch tickers for active waves (default: True)
        
    Returns:
        Dictionary with:
        - allowed: bool - Whether fetching is enabled
        - success: bool - Whether rebuild succeeded
        - tickers_requested: int - Number of tickers requested
        - tickers_fetched: int - Number of tickers successfully fetched
        - tickers_failed: int - Number of tickers that failed
        - failures: Dict[str, str] - Failed tickers with reasons
        - date_max: str - Most recent date in cache after rebuild
        - cache_info: Dict - Cache metadata after rebuild
    """
    logger.info("=" * 70)
    logger.info("REBUILD PRICE CACHE: Explicit cache rebuild requested")
    logger.info("=" * 70)
    
    # Check if required functions are available
    if collect_required_tickers is None or load_or_fetch_prices is None or get_cache_info is None:
        logger.error("Required price_loader functions not available for rebuild")
        return {
            'allowed': False,
            'success': False,
            'tickers_requested': 0,
            'tickers_fetched': 0,
            'tickers_failed': 0,
            'failures': {},
            'date_max': None,
            'cache_info': {},
            'message': 'Price loader functions not available. Check imports.'
        }
    
    # Check if fetching is enabled
    if not PRICE_FETCH_ENABLED:
        logger.warning("PRICE_FETCH_ENABLED is False - fetching is disabled")
        return {
            'allowed': False,
            'success': False,
            'tickers_requested': 0,
            'tickers_fetched': 0,
            'tickers_failed': 0,
            'failures': {},
            'date_max': None,
            'cache_info': {},
            'message': 'Fetching is disabled. Set PRICE_FETCH_ENABLED=true to enable.'
        }
    
    logger.info("PRICE_FETCH_ENABLED is True - proceeding with fetch")
    
    # Collect required tickers
    required_tickers = collect_required_tickers(active_only=active_only)
    logger.info(f"Collected {len(required_tickers)} required tickers (active_only={active_only})")
    
    # Fetch prices with force_fetch=True to update cache
    prices_df = load_or_fetch_prices(required_tickers, force_fetch=True)
    
    # Get updated cache info
    cache_info = get_cache_info()
    
    # Count successes and failures
    tickers_fetched = len(prices_df.columns) if not prices_df.empty else 0
    tickers_failed = len(required_tickers) - tickers_fetched
    
    # Load failed tickers details
    failures = {}
    failed_tickers_path = os.path.join(CACHE_DIR, 'failed_tickers.csv')
    if os.path.exists(failed_tickers_path):
        try:
            failed_df = pd.read_csv(failed_tickers_path)
            if not failed_df.empty:
                # Get most recent failure for each ticker
                latest_failures = failed_df.sort_values('timestamp').groupby('ticker').last()
                failures = dict(zip(latest_failures.index, latest_failures['reason']))
        except Exception as e:
            logger.error(f"Error loading failed tickers: {e}")
    
    result = {
        'allowed': True,
        'success': not prices_df.empty,
        'tickers_requested': len(required_tickers),
        'tickers_fetched': tickers_fetched,
        'tickers_failed': tickers_failed,
        'failures': failures,
        'date_max': cache_info.get('last_updated'),
        'cache_info': cache_info
    }
    
    logger.info(f"Rebuild complete: {tickers_fetched}/{len(required_tickers)} tickers fetched")
    if failures:
        logger.warning(f"Failed to fetch {len(failures)} tickers")
    logger.info("=" * 70)
    
    return result


def get_active_required_tickers() -> List[str]:
    """
    Get the list of required tickers for active waves.
    
    This is a thin wrapper around collect_required_tickers that explicitly
    requests active-only tickers. This makes the intent clear in calling code.
    
    Returns:
        Sorted list of unique ticker symbols required for active waves
    """
    if collect_required_tickers is None:
        logger.error("collect_required_tickers function not available")
        return []
    return collect_required_tickers(active_only=True)


def get_required_tickers_active_waves() -> List[str]:
    """
    Alias for get_active_required_tickers() for consistency with problem statement.
    
    Get the list of required tickers for active waves only.
    Excludes inactive waves, redundant crypto tickers, and old test datasets.
    
    Returns:
        Sorted list of unique ticker symbols required for active waves
    """
    return get_active_required_tickers()


def compute_missing_and_extra_tickers(price_book: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute which tickers are missing from or extra in the PRICE_BOOK.
    
    This diagnostic function compares the PRICE_BOOK to the required tickers
    for active waves to identify gaps and excesses.
    
    Args:
        price_book: The PRICE_BOOK DataFrame
        
    Returns:
        Dictionary with:
        - required_tickers: List[str] - Tickers needed for active waves
        - cached_tickers: List[str] - Tickers in PRICE_BOOK
        - missing_tickers: List[str] - Required but not in cache
        - extra_tickers: List[str] - In cache but not required
        - required_count: int
        - cached_count: int
        - missing_count: int
        - extra_count: int
    """
    required = get_active_required_tickers()
    cached = [] if price_book is None or price_book.empty else price_book.columns.tolist()
    
    required_set = set(required)
    cached_set = set(cached)
    
    missing = sorted(list(required_set - cached_set))
    extra = sorted(list(cached_set - required_set))
    
    return {
        'required_tickers': required,
        'cached_tickers': cached,
        'missing_tickers': missing,
        'extra_tickers': extra,
        'required_count': len(required),
        'cached_count': len(cached),
        'missing_count': len(missing),
        'extra_count': len(extra)
    }


# ============================================================================
# PRICE_BOOK Singleton
# ============================================================================
# Global PRICE_BOOK instance - lazy loaded on first access
# This ensures all parts of the application use the same price data
_PRICE_BOOK_CACHE: Optional[pd.DataFrame] = None
_PRICE_BOOK_LOADED: bool = False


def get_price_book_singleton(force_reload: bool = False) -> pd.DataFrame:
    """
    Get the singleton PRICE_BOOK instance.
    
    This ensures all parts of the application (execution, readiness, health, 
    diagnostics) use the exact same price data loaded from the canonical cache.
    
    Args:
        force_reload: If True, reload the PRICE_BOOK from disk (default: False)
        
    Returns:
        The singleton PRICE_BOOK DataFrame
    """
    global _PRICE_BOOK_CACHE, _PRICE_BOOK_LOADED
    
    if force_reload or not _PRICE_BOOK_LOADED:
        logger.info("Loading PRICE_BOOK singleton from canonical cache")
        _PRICE_BOOK_CACHE = get_price_book(active_tickers=None)  # Load all cached tickers
        _PRICE_BOOK_LOADED = True
        logger.info(f"PRICE_BOOK singleton loaded: {len(_PRICE_BOOK_CACHE)} rows × {len(_PRICE_BOOK_CACHE.columns)} cols")
    
    return _PRICE_BOOK_CACHE


# Expose PRICE_BOOK as a module-level variable for convenience
# Note: This is a function that returns the singleton, not the DataFrame itself
# Usage: from helpers.price_book import get_price_book_singleton as PRICE_BOOK
# Or: PRICE_BOOK = get_price_book_singleton()
PRICE_BOOK = get_price_book_singleton


def compute_system_health(price_book: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Compute system health status based on PRICE_BOOK and active wave requirements.
    
    This provides a unified health assessment that combines:
    - Ticker coverage (missing vs required)
    - Data staleness (age of latest prices)
    - Data sufficiency (number of trading days)
    
    Health Levels:
    - OK: All required tickers present, data fresh (< 5 days old)
    - DEGRADED: Missing some required tickers OR data slightly stale (5-10 days)
    - STALE: Data very stale (> 10 days) OR missing many required tickers
    
    Args:
        price_book: Optional PRICE_BOOK DataFrame. If None, loads from singleton.
        
    Returns:
        Dictionary with:
        - health_status: str - "OK", "DEGRADED", or "STALE"
        - health_emoji: str - Visual indicator
        - missing_count: int - Number of missing required tickers
        - total_required: int - Total required tickers
        - coverage_pct: float - Percentage of required tickers present
        - days_stale: int - Days since latest data
        - num_days: int - Number of trading days in cache
        - details: str - Human-readable explanation
    """
    if price_book is None:
        price_book = get_price_book_singleton()
    
    # Get ticker analysis
    ticker_analysis = compute_missing_and_extra_tickers(price_book)
    
    # Get cache info
    if check_cache_readiness is not None:
        readiness = check_cache_readiness(active_only=True)
        days_stale = readiness.get('days_stale', 0) or 0
        num_days = readiness.get('num_days', 0)
    else:
        days_stale = 0
        num_days = len(price_book) if not price_book.empty else 0
    
    missing_count = ticker_analysis['missing_count']
    total_required = ticker_analysis['required_count']
    
    # Calculate coverage percentage
    if total_required > 0:
        coverage_pct = ((total_required - missing_count) / total_required) * 100
    else:
        coverage_pct = 100.0
    
    # Determine health status
    health_status = "OK"
    health_emoji = "✅"
    details = "All systems nominal"
    
    # Check for critical issues
    if price_book is None or price_book.empty:
        health_status = "STALE"
        health_emoji = "❌"
        details = "PRICE_BOOK is empty - cache needs to be built"
    elif missing_count > total_required * 0.5:  # More than 50% missing
        health_status = "STALE"
        health_emoji = "❌"
        details = f"Critical: {missing_count}/{total_required} required tickers missing ({coverage_pct:.1f}% coverage)"
    elif days_stale > 10:  # More than 10 days old
        health_status = "STALE"
        health_emoji = "❌"
        details = f"Data is {days_stale} days stale - needs refresh"
    elif missing_count > 0:  # Some tickers missing
        health_status = "DEGRADED"
        health_emoji = "⚠️"
        details = f"Missing {missing_count}/{total_required} required tickers ({coverage_pct:.1f}% coverage)"
    elif days_stale > 5:  # Slightly stale
        health_status = "DEGRADED"
        health_emoji = "⚠️"
        details = f"Data is {days_stale} days old - consider refresh"
    else:
        health_status = "OK"
        health_emoji = "✅"
        details = f"All {total_required} required tickers present, data fresh ({days_stale} days old)"
    
    return {
        'health_status': health_status,
        'health_emoji': health_emoji,
        'missing_count': missing_count,
        'total_required': total_required,
        'coverage_pct': coverage_pct,
        'days_stale': days_stale,
        'num_days': num_days,
        'details': details
    }


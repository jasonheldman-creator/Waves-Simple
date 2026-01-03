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

# Cache configuration
DEFAULT_CACHE_YEARS = 5  # Keep last 5 years of data
MAX_FORWARD_FILL_DAYS = 5  # Maximum gap to forward-fill
MIN_REQUIRED_DAYS = 10  # Minimum trading days for readiness
MAX_STALE_DAYS = 3  # Data older than this is considered stale

# Download configuration
BATCH_SIZE = 50  # Maximum tickers per batch download
RETRY_ATTEMPTS = 2  # Number of retry attempts for failed downloads
RETRY_DELAY = 1.0  # Initial delay between retries (seconds)
REQUEST_TIMEOUT = 15  # Timeout for yfinance requests (seconds)


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
    cutoff_date = datetime.now() - timedelta(days=365 * years)
    
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
    end_date: datetime
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Fetch prices for a batch of tickers from yfinance.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for data
        end_date: End date for data
        
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


def load_or_fetch_prices(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None
) -> pd.DataFrame:
    """
    Load or fetch price data with intelligent caching.
    
    This is the main entry point for price data. It:
    1. Deduplicates and normalizes tickers
    2. Loads existing cache
    3. Identifies missing/stale data
    4. Fetches only what's needed
    5. Updates cache
    6. Returns requested data
    
    Args:
        tickers: List of ticker symbols (may contain duplicates)
        start: Start date (YYYY-MM-DD) or None for auto-calculate
        end: End date (YYYY-MM-DD) or None for today
        
    Returns:
        DataFrame with:
        - Index: Datetime (trading days)
        - Columns: Ticker symbols
        - Values: Adjusted close prices (NaN for failed tickers)
    
    Example:
        >>> prices = load_or_fetch_prices(['AAPL', 'MSFT'], start='2024-01-01')
        >>> print(prices.head())
                    AAPL   MSFT
        2024-01-02  185.64  374.58
        2024-01-03  184.25  373.85
        ...
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
    
    # Step 3: Load existing cache
    cache_df = load_cache()
    
    # Step 4: Identify missing and stale tickers
    missing_tickers, stale_tickers = identify_missing_and_stale_tickers(
        cache_df, tickers, start_date, end_date
    )
    
    to_fetch = list(set(missing_tickers + stale_tickers))
    
    logger.info(f"Cache status: {len(missing_tickers)} missing, {len(stale_tickers)} stale")
    if to_fetch:
        logger.info(f"Will fetch {len(to_fetch)} ticker(s): {to_fetch}")
    else:
        logger.info("All requested data available in cache")
    
    # Step 5: Fetch missing/stale data
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
        
        # Log failures
        if failures:
            logger.warning(f"Failed to fetch {len(failures)} ticker(s):")
            for ticker, reason in failures.items():
                logger.warning(f"  {ticker}: {reason}")
    
    # Step 6: Merge cache and new data
    if not new_data.empty:
        merged = merge_cache_and_new_data(cache_df, new_data)
        
        # Trim to date range
        merged = trim_cache_to_date_range(merged, DEFAULT_CACHE_YEARS)
        
        # Save updated cache
        save_cache(merged)
        cache_df = merged
    
    # Step 7: Extract requested data from cache
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
    Get information about the current price cache.
    
    Returns:
        Dictionary with cache metadata:
        - exists: bool
        - path: str
        - size_mb: float
        - num_tickers: int
        - num_days: int
        - date_range: tuple of (start, end) dates
        - tickers: list of ticker symbols
    """
    info = {
        'exists': os.path.exists(CACHE_PATH),
        'path': CACHE_PATH,
        'size_mb': 0.0,
        'num_tickers': 0,
        'num_days': 0,
        'date_range': (None, None),
        'tickers': []
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
            info['date_range'] = (
                cache_df.index[0].strftime('%Y-%m-%d'),
                cache_df.index[-1].strftime('%Y-%m-%d')
            )
            info['tickers'] = sorted(cache_df.columns.tolist())
    
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

"""
Live Data Engine - Real-time market data fetching and PRICE_BOOK building

This module provides the live data engine for fetching real-time market prices
from Yahoo Finance (with Alpaca as a potential alternative). It builds the
PRICE_BOOK dynamically at runtime, ensuring Portfolio Snapshot values reflect
true live market conditions.

Key Features:
- Live market data fetching via Yahoo Finance API
- Fallback to forced PRICE_BOOK rebuild on API failure
- Timestamp tracking for diagnostics
- Memory reference tracking for verification
- NO caching - pure runtime data fetching

Usage:
    from helpers.live_data_engine import build_live_price_book
    
    # Build PRICE_BOOK from live market data
    price_book, metadata = build_live_price_book(tickers=['SPY', 'QQQ', 'NVDA'])
    
    # Access diagnostics
    print(f"Data source: {metadata['data_source']}")
    print(f"Timestamp: {metadata['timestamp_utc']}")
    print(f"Memory ref: {metadata['memory_id']}")
"""

import os
import logging
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Live data API selection (Yahoo Finance as primary, Alpaca as alternative)
LIVE_DATA_PROVIDER = os.environ.get('LIVE_DATA_PROVIDER', 'yahoo').lower()
LIVE_DATA_ENABLED = os.environ.get('LIVE_DATA_ENABLED', 'true').lower() in ('true', '1', 'yes')

# Import data providers
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    logger.warning("yfinance not available - live data fetching disabled")
    YFINANCE_AVAILABLE = False

# For fallback to cache when live fetch fails
try:
    from helpers.price_book import CANONICAL_CACHE_PATH
    CACHE_PATH_AVAILABLE = True
except ImportError:
    CANONICAL_CACHE_PATH = "data/cache/prices_cache.parquet"
    CACHE_PATH_AVAILABLE = True


def fetch_live_prices_yahoo(tickers: List[str], period: str = "2y", interval: str = "1d") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetch live market prices from Yahoo Finance API.
    
    This function retrieves real-time historical price data for the specified
    tickers using the yfinance library. It returns a DataFrame suitable for
    PRICE_BOOK construction.
    
    Args:
        tickers: List of ticker symbols to fetch
        period: Time period for historical data (default: "2y" for 2 years)
        interval: Data interval (default: "1d" for daily)
        
    Returns:
        Tuple of (prices_df, metadata):
        - prices_df: DataFrame with index=dates, columns=tickers, values=close prices
        - metadata: Dict with fetch details (success, failures, timestamp, etc.)
    """
    logger.info("=" * 70)
    logger.info(f"LIVE DATA ENGINE: Fetching prices from Yahoo Finance")
    logger.info(f"Tickers: {len(tickers)} symbols")
    logger.info(f"Period: {period}, Interval: {interval}")
    logger.info("=" * 70)
    
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance is not available - cannot fetch live data")
        return pd.DataFrame(), {
            'success': False,
            'data_source': 'Yahoo Finance (FAILED)',
            'error': 'yfinance library not available',
            'tickers_fetched': 0,
            'tickers_failed': len(tickers),
            'timestamp_utc': datetime.now(timezone.utc).isoformat()
        }
    
    fetch_timestamp = datetime.now(timezone.utc)
    successful_tickers = []
    failed_tickers = {}
    price_data = {}
    
    # Fetch data for each ticker
    for ticker in tickers:
        try:
            logger.info(f"Fetching {ticker}...")
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period, interval=interval)
            
            if hist.empty:
                logger.warning(f"No data returned for {ticker}")
                failed_tickers[ticker] = "No data returned"
                continue
            
            # Extract close prices
            if 'Close' in hist.columns:
                price_data[ticker] = hist['Close']
                successful_tickers.append(ticker)
                logger.info(f"✓ {ticker}: {len(hist)} days fetched")
            else:
                logger.warning(f"No 'Close' column for {ticker}")
                failed_tickers[ticker] = "Missing 'Close' column"
                
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            failed_tickers[ticker] = str(e)
    
    # Build unified DataFrame
    if price_data:
        prices_df = pd.DataFrame(price_data)
        prices_df.index.name = 'Date'
        
        # Ensure datetime index
        if not isinstance(prices_df.index, pd.DatetimeIndex):
            prices_df.index = pd.to_datetime(prices_df.index)
        
        # Sort by date
        prices_df = prices_df.sort_index()
        
        logger.info(f"Live fetch complete: {len(successful_tickers)}/{len(tickers)} tickers")
        logger.info(f"Date range: {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
        logger.info(f"Shape: {prices_df.shape[0]} days × {prices_df.shape[1]} tickers")
    else:
        prices_df = pd.DataFrame()
        logger.error("All tickers failed to fetch")
    
    metadata = {
        'success': len(successful_tickers) > 0,
        'data_source': 'Yahoo Finance (Live API)',
        'tickers_requested': len(tickers),
        'tickers_fetched': len(successful_tickers),
        'tickers_failed': len(failed_tickers),
        'failed_tickers': failed_tickers,
        'timestamp_utc': fetch_timestamp.isoformat(),
        'period': period,
        'interval': interval
    }
    
    logger.info("=" * 70)
    
    return prices_df, metadata


def load_fallback_cache() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load PRICE_BOOK from canonical cache as fallback when live fetch fails.
    
    This function provides a fallback mechanism to ensure the application
    can still function even when live data APIs are unavailable.
    
    Returns:
        Tuple of (prices_df, metadata):
        - prices_df: DataFrame from cache
        - metadata: Dict with fallback details
    """
    logger.warning("=" * 70)
    logger.warning("FALLBACK: Loading PRICE_BOOK from canonical cache")
    logger.warning(f"Cache path: {CANONICAL_CACHE_PATH}")
    logger.warning("=" * 70)
    
    try:
        if os.path.exists(CANONICAL_CACHE_PATH):
            prices_df = pd.read_parquet(CANONICAL_CACHE_PATH)
            logger.info(f"Fallback cache loaded: {prices_df.shape[0]} days × {prices_df.shape[1]} tickers")
            
            metadata = {
                'success': True,
                'data_source': 'Forced Rebuild (Cache Fallback)',
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'note': 'Live API failed - using cached data as fallback'
            }
        else:
            logger.error(f"Cache file not found: {CANONICAL_CACHE_PATH}")
            prices_df = pd.DataFrame()
            metadata = {
                'success': False,
                'data_source': 'Forced Rebuild (FAILED)',
                'error': 'Cache file not found',
                'timestamp_utc': datetime.now(timezone.utc).isoformat()
            }
    except Exception as e:
        logger.error(f"Error loading fallback cache: {e}")
        prices_df = pd.DataFrame()
        metadata = {
            'success': False,
            'data_source': 'Forced Rebuild (FAILED)',
            'error': str(e),
            'timestamp_utc': datetime.now(timezone.utc).isoformat()
        }
    
    logger.warning("=" * 70)
    return prices_df, metadata


def build_live_price_book(
    tickers: Optional[List[str]] = None,
    use_cache_fallback: bool = True,
    period: str = "2y"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build PRICE_BOOK dynamically from live market data at runtime.
    
    This is the main entry point for the live data engine. It attempts to
    fetch live market data and falls back to cache on failure.
    
    This function NEVER caches - it fetches fresh data on every call to
    ensure Portfolio Snapshot values are truly dynamic and reflect current
    market conditions.
    
    Args:
        tickers: List of tickers to fetch. If None, loads from cache fallback.
        use_cache_fallback: If True, fall back to cache when live fetch fails
        period: Historical data period (default: "2y")
        
    Returns:
        Tuple of (price_book, metadata):
        - price_book: DataFrame with index=dates, columns=tickers, values=close prices
        - metadata: Dict with comprehensive diagnostics:
            - success: bool
            - data_source: str (e.g., "Yahoo Finance (Live API)" or "Forced Rebuild")
            - timestamp_utc: str (ISO format)
            - memory_id: str (hex ID for object identity verification)
            - latest_price_date: str (most recent price date)
            - render_utc: str (UTC time of render)
            - tickers_count: int
            - days_count: int
    """
    render_timestamp = datetime.now(timezone.utc)
    
    # Check if live data is enabled
    if not LIVE_DATA_ENABLED:
        logger.warning("LIVE_DATA_ENABLED is False - using cache fallback only")
        if use_cache_fallback:
            prices_df, fallback_meta = load_fallback_cache()
        else:
            prices_df = pd.DataFrame()
            fallback_meta = {
                'success': False,
                'data_source': 'Live Data DISABLED',
                'error': 'LIVE_DATA_ENABLED=false'
            }
        
        # Add metadata
        metadata = {
            **fallback_meta,
            'memory_id': hex(id(prices_df)),
            'render_utc': render_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'latest_price_date': prices_df.index[-1].strftime('%Y-%m-%d') if not prices_df.empty else None,
            'tickers_count': len(prices_df.columns) if not prices_df.empty else 0,
            'days_count': len(prices_df) if not prices_df.empty else 0
        }
        return prices_df, metadata
    
    # If no tickers provided, use fallback
    if tickers is None or len(tickers) == 0:
        logger.warning("No tickers provided - using cache fallback")
        if use_cache_fallback:
            prices_df, fallback_meta = load_fallback_cache()
            metadata = {
                **fallback_meta,
                'memory_id': hex(id(prices_df)),
                'render_utc': render_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                'latest_price_date': prices_df.index[-1].strftime('%Y-%m-%d') if not prices_df.empty else None,
                'tickers_count': len(prices_df.columns) if not prices_df.empty else 0,
                'days_count': len(prices_df) if not prices_df.empty else 0
            }
            return prices_df, metadata
        else:
            return pd.DataFrame(), {
                'success': False,
                'data_source': 'No Tickers Provided',
                'memory_id': hex(id(pd.DataFrame())),
                'render_utc': render_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
            }
    
    # Attempt live fetch
    if LIVE_DATA_PROVIDER == 'yahoo':
        prices_df, fetch_meta = fetch_live_prices_yahoo(tickers, period=period)
    else:
        logger.error(f"Unknown live data provider: {LIVE_DATA_PROVIDER}")
        prices_df = pd.DataFrame()
        fetch_meta = {
            'success': False,
            'data_source': f'Unknown Provider ({LIVE_DATA_PROVIDER})',
            'error': f'Provider {LIVE_DATA_PROVIDER} not implemented'
        }
    
    # Fall back to cache if live fetch failed
    if (prices_df.empty or not fetch_meta.get('success', False)) and use_cache_fallback:
        logger.warning("Live fetch failed - falling back to cache")
        prices_df, fallback_meta = load_fallback_cache()
        metadata = {
            **fallback_meta,
            'live_fetch_attempted': True,
            'live_fetch_error': fetch_meta.get('error', 'Unknown error'),
            'memory_id': hex(id(prices_df)),
            'render_utc': render_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'latest_price_date': prices_df.index[-1].strftime('%Y-%m-%d') if not prices_df.empty else None,
            'tickers_count': len(prices_df.columns) if not prices_df.empty else 0,
            'days_count': len(prices_df) if not prices_df.empty else 0
        }
    else:
        # Live fetch succeeded
        metadata = {
            **fetch_meta,
            'memory_id': hex(id(prices_df)),
            'render_utc': render_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'latest_price_date': prices_df.index[-1].strftime('%Y-%m-%d') if not prices_df.empty else None,
            'tickers_count': len(prices_df.columns) if not prices_df.empty else 0,
            'days_count': len(prices_df) if not prices_df.empty else 0
        }
    
    return prices_df, metadata


def get_required_tickers_from_cache() -> List[str]:
    """
    Get list of required tickers from the canonical cache.
    
    This is a helper function to determine which tickers should be fetched
    from live APIs by examining what's already in the cache.
    
    Returns:
        List of ticker symbols found in cache
    """
    try:
        if os.path.exists(CANONICAL_CACHE_PATH):
            cache_df = pd.read_parquet(CANONICAL_CACHE_PATH)
            tickers = cache_df.columns.tolist()
            logger.info(f"Found {len(tickers)} tickers in cache")
            return sorted(tickers)
        else:
            logger.warning(f"Cache not found: {CANONICAL_CACHE_PATH}")
            return []
    except Exception as e:
        logger.error(f"Error reading tickers from cache: {e}")
        return []

"""
Global Price Data Caching System

This module provides centralized price data management with:
- Batch downloading to avoid rate limits
- TTL-based caching using Streamlit
- Comprehensive error handling and failure tracking
- Session state persistence for rate-limit recovery

Usage:
    from data_cache import get_global_price_cache
    
    cache = get_global_price_cache(wave_registry, days=365, ttl_seconds=7200)
    price_df = cache["price_df"]
    failures = cache["failures"]
"""

from __future__ import annotations

import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
import streamlit as st

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None


# Constants for download configuration
BUFFER_DAYS = 260  # Extra days to fetch for moving average calculations
BATCH_PAUSE_MIN = 0.5  # Minimum pause between batches (seconds)
BATCH_PAUSE_MAX = 1.5  # Maximum pause between batches (seconds)
DEFAULT_TTL_SECONDS = 7200  # Default cache TTL (2 hours)


def collect_all_required_tickers(
    wave_registry: Dict[str, List[Any]],
    include_benchmarks: bool = True,
    include_safe_assets: bool = True,
    active_only: bool = False
) -> List[str]:
    """
    Collect all tickers required across the wave registry.
    
    Args:
        wave_registry: Dictionary mapping wave_id to holdings
        include_benchmarks: Whether to include benchmark tickers
        include_safe_assets: Whether to include safe asset tickers
        active_only: If True, only include tickers from active waves (filters using wave_registry.csv)
    
    Returns:
        Sorted list of unique tickers
    """
    tickers = set()
    
    # Filter wave_registry to only active waves if requested
    filtered_registry = wave_registry
    if active_only:
        try:
            import os
            # Read wave_registry.csv to get active wave_ids
            wave_registry_path = os.path.join(os.path.dirname(__file__), 'data', 'wave_registry.csv')
            if os.path.exists(wave_registry_path):
                import pandas as pd
                registry_df = pd.read_csv(wave_registry_path)
                # Filter to active waves only
                active_wave_ids = set(registry_df[registry_df['active'] == True]['wave_id'].tolist())
                # Filter wave_registry dict to only include active wave_ids
                filtered_registry = {k: v for k, v in wave_registry.items() if k in active_wave_ids}
            else:
                # If wave_registry.csv doesn't exist, fall back to using all waves
                # This maintains backward compatibility
                pass
        except Exception as e:
            # If filtering fails, fall back to using all waves to avoid breaking functionality
            # Log the error but continue
            import warnings
            warnings.warn(f"Failed to filter waves by active status: {e}. Using all waves.")
    
    # Extract tickers from wave holdings
    for wave_name, holdings in filtered_registry.items():
        for holding in holdings:
            # Holdings can be Holding objects or dicts
            if hasattr(holding, 'ticker'):
                tickers.add(holding.ticker)
            elif isinstance(holding, dict) and 'ticker' in holding:
                tickers.add(holding['ticker'])
    
    # Add standard market and volatility tickers (always included for system health monitoring)
    tickers.update(['SPY', '^VIX', 'BTC-USD'])
    
    # Add benchmark tickers if requested
    if include_benchmarks:
        # Common benchmark tickers
        benchmark_tickers = [
            'SPY', 'QQQ', 'IWM', 'IWV', 'DIA', 'VTI',
            'EFA', 'EEM', 'AGG', 'TLT', 'GLD'
        ]
        tickers.update(benchmark_tickers)
    
    # Add safe asset tickers if requested
    if include_safe_assets:
        safe_asset_tickers = [
            'SGOV', 'BIL', 'SHY', 'SUB', 'SHM', 'MUB',
            'USDC-USD', 'USDT-USD', 'DAI-USD', 'USDP-USD',
            '^IRX', '^FVX', '^TNX', 'IEF', 'TLT',
            'HYG', 'LQD'
        ]
        tickers.update(safe_asset_tickers)
    
    return sorted(list(tickers))


def download_prices_batched(
    tickers: List[str],
    period_days: int,
    chunk_size: int = 100
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Download price data in batches to avoid rate limits.
    
    Features:
    - Chunks downloads into manageable sizes
    - Pauses between batches
    - Per-ticker failure tracking
    - Continues on error (graceful degradation)
    
    Args:
        tickers: List of ticker symbols to download
        period_days: Number of days of history to fetch
        chunk_size: Number of tickers per batch (default: 100)
    
    Returns:
        Tuple of (price_df, failures_dict):
        - price_df: Wide DataFrame with date index and ticker columns
        - failures_dict: Dict mapping failed tickers to error reasons
    """
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame(), {"error": "yfinance not available"}
    
    if not tickers:
        return pd.DataFrame(), {}
    
    failures = {}
    all_prices = pd.DataFrame()
    
    # Calculate date range with buffer
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=period_days + BUFFER_DAYS)  # Extra buffer for calculations
    
    # Split into batches
    num_batches = (len(tickers) + chunk_size - 1) // chunk_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * chunk_size
        end_idx = min((batch_idx + 1) * chunk_size, len(tickers))
        batch_tickers = tickers[start_idx:end_idx]
        
        try:
            # Download this batch
            batch_data = yf.download(
                tickers=batch_tickers,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="column"
            )
            
            if batch_data is None or len(batch_data) == 0:
                # Record failure for all tickers in batch
                for ticker in batch_tickers:
                    failures[ticker] = "Empty response from yfinance"
                continue
            
            # Extract Adj Close or Close prices
            if isinstance(batch_data.columns, pd.MultiIndex):
                if "Adj Close" in batch_data.columns.get_level_values(0):
                    batch_prices = batch_data["Adj Close"]
                elif "Close" in batch_data.columns.get_level_values(0):
                    batch_prices = batch_data["Close"]
                else:
                    batch_prices = batch_data[batch_data.columns.levels[0][0]]
            else:
                batch_prices = batch_data
            
            # Handle single ticker case (Series instead of DataFrame)
            if isinstance(batch_prices, pd.Series):
                batch_prices = batch_prices.to_frame()
            
            # Flatten multi-index columns if needed
            if isinstance(batch_prices.columns, pd.MultiIndex):
                batch_prices = batch_prices.droplevel(0, axis=1)
            
            # Sort index and forward/backward fill
            batch_prices = batch_prices.sort_index().ffill().bfill()
            
            # Merge into all_prices
            if len(all_prices) == 0:
                all_prices = batch_prices.copy()
            else:
                # Align indices and concatenate
                all_prices = pd.concat([all_prices, batch_prices], axis=1)
            
            # Check which tickers succeeded and which failed
            for ticker in batch_tickers:
                if ticker not in batch_prices.columns:
                    failures[ticker] = "Ticker not in downloaded data"
                elif batch_prices[ticker].isna().all():
                    failures[ticker] = "All NaN values"
            
        except Exception as e:
            # Record failure for all tickers in batch
            error_msg = str(e)
            for ticker in batch_tickers:
                failures[ticker] = f"Batch download error: {error_msg}"
        
        # Pause between batches to avoid rate limits (except for last batch)
        if batch_idx < num_batches - 1:
            pause_time = random.uniform(BATCH_PAUSE_MIN, BATCH_PAUSE_MAX)
            time.sleep(pause_time)
    
    return all_prices, failures


@st.cache_data(ttl=DEFAULT_TTL_SECONDS, show_spinner=False)
def _download_prices_cached(
    tickers_tuple: Tuple[str, ...],
    period_days: int,
    chunk_size: int,
    _cache_key: str
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Cached wrapper around download_prices_batched.
    
    Note: Uses tuple for tickers to make it hashable for caching.
    _cache_key is a timestamp used to force cache invalidation.
    The decorator TTL provides automatic expiration, while _cache_key
    allows manual cache invalidation via get_global_price_cache().
    """
    tickers = list(tickers_tuple)
    return download_prices_batched(tickers, period_days, chunk_size)


def get_global_price_cache(
    wave_registry: Dict[str, List[Any]],
    days: int = 365,
    ttl_seconds: int = 7200,
    active_only: bool = False
) -> Dict[str, Any]:
    """
    Get global price cache with TTL-based caching and session state persistence.
    
    This is the main entry point for getting cached price data.
    
    Features:
    - Uses Streamlit cache_data with configurable TTL
    - Stores last known good data in session_state for rate-limit recovery
    - Returns comprehensive failure tracking
    
    Args:
        wave_registry: Dictionary mapping wave names to holdings
        days: Number of days of history (default: 365)
        ttl_seconds: Cache TTL in seconds (default: 7200 = 2 hours)
        active_only: If True, only fetch prices for tickers from active waves (default: False for backward compatibility)
    
    Returns:
        Dictionary with:
        - price_df: Wide DataFrame (date index, ticker columns)
        - failures: Dict mapping failed tickers to error reasons
        - asof: Timestamp when data was fetched
        - ticker_count: Number of tickers requested
        - success_count: Number of tickers successfully downloaded
    """
    # Collect all required tickers
    # Note: For backward compatibility, default is to include all waves (active_only=False)
    # System health monitoring should use active_only=True to avoid false positives from inactive waves
    tickers = collect_all_required_tickers(
        wave_registry,
        include_benchmarks=True,
        include_safe_assets=True,
        active_only=active_only
    )
    
    if not tickers:
        return {
            "price_df": pd.DataFrame(),
            "failures": {"error": "No tickers to fetch"},
            "asof": datetime.utcnow(),
            "ticker_count": 0,
            "success_count": 0
        }
    
    # Check for force rebuild flag
    force_rebuild = st.session_state.get("force_price_cache_rebuild", False)
    
    # Generate cache key (timestamp-based for force rebuild)
    if force_rebuild:
        cache_key = datetime.utcnow().isoformat()
        # Clear the force rebuild flag
        st.session_state.force_price_cache_rebuild = False
    else:
        # Use a cache key that changes based on TTL
        cache_key = str(int(time.time() / ttl_seconds))
    
    try:
        # Download with caching
        # Note: We pass ttl to the decorator, but also use cache_key for manual invalidation
        price_df, failures = _download_prices_cached(
            tickers_tuple=tuple(tickers),
            period_days=days,
            chunk_size=100,
            _cache_key=cache_key
        )
        
        # Trim to requested period
        if not price_df.empty and len(price_df) > days:
            price_df = price_df.iloc[-days:]
        
        # Store in session state as last known good
        st.session_state.last_known_good_prices = {
            "price_df": price_df.copy() if not price_df.empty else pd.DataFrame(),
            "failures": failures.copy(),
            "asof": datetime.utcnow(),
            "ticker_count": len(tickers),
            "success_count": len(tickers) - len(failures)
        }
        
        result = {
            "price_df": price_df,
            "failures": failures,
            "asof": datetime.utcnow(),
            "ticker_count": len(tickers),
            "success_count": len(tickers) - len(failures)
        }
        
    except Exception as e:
        # On error, try to use last known good data from session state
        last_known_good = st.session_state.get("last_known_good_prices")
        
        if last_known_good:
            # Use cached data with warning
            result = last_known_good.copy()
            result["failures"]["cache_error"] = f"Using stale cache: {str(e)}"
        else:
            # No fallback available
            result = {
                "price_df": pd.DataFrame(),
                "failures": {"cache_error": str(e)},
                "asof": datetime.utcnow(),
                "ticker_count": len(tickers),
                "success_count": 0
            }
    
    return result

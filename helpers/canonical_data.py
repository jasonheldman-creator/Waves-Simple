"""
Canonical Data Access Helper

This module provides a single, standardized entry point for accessing price data
across all modules in the application. It enforces a cache-first strategy and
provides consistent error handling.

Key Principles:
- Single canonical function for data access: get_canonical_price_data()
- Cache-first strategy: no network fetches
- Consistent error handling and logging
- Returns standardized structure

Usage:
    from helpers.canonical_data import get_canonical_price_data
    
    # Get price data for specific tickers
    prices = get_canonical_price_data(tickers=['SPY', 'QQQ'])
    
    # Get all cached data
    prices = get_canonical_price_data()
"""

import logging
from typing import List, Optional, Dict, Any
import pandas as pd
import sys
import os

# Add parent helpers directory to path to avoid __init__.py imports
helpers_dir = os.path.dirname(os.path.abspath(__file__))
if helpers_dir not in sys.path:
    sys.path.insert(0, helpers_dir)

import price_book
get_price_book = price_book.get_price_book
get_price_book_meta = price_book.get_price_book_meta

logger = logging.getLogger(__name__)


def get_canonical_price_data(
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Get canonical price data with cache-first strategy.
    
    This is the single entry point for all price data access in the application.
    It wraps get_price_book with standardized error handling and logging.
    
    Args:
        tickers: Optional list of tickers to include (if None, returns all cached tickers)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        
    Returns:
        DataFrame with:
        - Index: DatetimeIndex (trading days)
        - Columns: Ticker symbols
        - Values: Close prices (NaN for missing data)
        
    Example:
        >>> prices = get_canonical_price_data(tickers=['SPY', 'QQQ'])
        >>> prices = get_canonical_price_data()  # All cached tickers
    """
    try:
        logger.info("Accessing canonical price data (cache-first)")
        
        # Call get_price_book with cache-first strategy
        price_data = get_price_book(
            active_tickers=tickers,
            start_date=start_date,
            end_date=end_date
        )
        
        if price_data.empty:
            logger.warning("No price data available - cache may be empty")
        else:
            meta = get_price_book_meta(price_data)
            logger.info(
                f"Retrieved price data: {meta['rows']} days × {meta['cols']} tickers"
            )
        
        return price_data
        
    except Exception as e:
        logger.error(f"Error accessing canonical price data: {e}", exc_info=True)
        # Return empty DataFrame on error
        return pd.DataFrame()

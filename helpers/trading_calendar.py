"""
Trading Calendar Helper

This module provides canonical helper functions for extracting trading-day aligned
dates from price data. This ensures all return calculations use actual trading dates
instead of calendar dates, fixing the "0.00% 1D returns" issue on weekends/holidays.

Key Functions:
- get_asof_dates(price_df): Extract asof_date and prev_date from price cache
  
Usage:
    from helpers.trading_calendar import get_asof_dates
    
    asof_date, prev_date = get_asof_dates(price_df)
    # asof_date = most recent trading date in cache
    # prev_date = previous trading date (for 1D calculations)
"""

from typing import Tuple, Optional
import pandas as pd
from datetime import datetime


def get_asof_dates(price_df: Optional[pd.DataFrame]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Extract as-of date and previous trading date from price DataFrame.
    
    This is the canonical function for determining the trading-day aligned
    date range for return calculations. It uses the actual trading dates
    present in the price cache, not calendar dates.
    
    Args:
        price_df: DataFrame with DatetimeIndex representing trading dates
        
    Returns:
        Tuple of (asof_date, prev_date):
        - asof_date: Most recent trading date (price_df.index.max())
        - prev_date: Previous trading date (price_df.index[-2])
        
        Both are None if price_df is None or has insufficient data.
        
    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-01', periods=5, freq='D')
        >>> price_df = pd.DataFrame({'SPY': [100, 101, 102, 103, 104]}, index=dates)
        >>> asof, prev = get_asof_dates(price_df)
        >>> asof == dates[-1]
        True
        >>> prev == dates[-2]
        True
        
        >>> # Insufficient data
        >>> small_df = pd.DataFrame({'SPY': [100]}, index=[dates[0]])
        >>> asof, prev = get_asof_dates(small_df)
        >>> asof is None and prev is None
        True
    """
    # Handle None or empty DataFrame
    if price_df is None or len(price_df) == 0:
        return None, None
    
    # Require at least 2 rows for meaningful calculations
    if len(price_df) < 2:
        return None, None
    
    try:
        # Extract as-of date (most recent trading date)
        asof_date = price_df.index.max()
        
        # Extract previous trading date (for 1D calculations)
        prev_date = price_df.index[-2]
        
        # Convert Timestamp to datetime if needed
        if isinstance(asof_date, pd.Timestamp):
            asof_date = asof_date.to_pydatetime()
        if isinstance(prev_date, pd.Timestamp):
            prev_date = prev_date.to_pydatetime()
        
        return asof_date, prev_date
        
    except (IndexError, KeyError, AttributeError) as e:
        # Log error and return None
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to extract asof_dates from price_df: {e}")
        return None, None


def get_asof_date_str(price_df: Optional[pd.DataFrame], fmt: str = '%Y-%m-%d') -> str:
    """
    Get as-of date as formatted string.
    
    Convenience function for formatting the as-of date for display/storage.
    
    Args:
        price_df: DataFrame with DatetimeIndex
        fmt: strftime format string (default: '%Y-%m-%d')
        
    Returns:
        Formatted date string, or 'N/A' if unavailable
        
    Examples:
        >>> import pandas as pd
        >>> dates = pd.date_range('2024-01-15', periods=5, freq='D')
        >>> price_df = pd.DataFrame({'SPY': [100, 101, 102, 103, 104]}, index=dates)
        >>> get_asof_date_str(price_df)
        '2024-01-19'
        
        >>> get_asof_date_str(None)
        'N/A'
    """
    asof_date, _ = get_asof_dates(price_df)
    
    if asof_date is None:
        return 'N/A'
    
    try:
        return asof_date.strftime(fmt)
    except Exception:
        return 'N/A'

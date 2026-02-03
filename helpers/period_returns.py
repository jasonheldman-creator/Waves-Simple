"""
Period Return Calculation Helper

This module provides the canonical helper function for computing period returns
using trading-day windows. This ensures consistency across all return calculations
in the application.

Key Features:
- Uses trading days, not calendar days
- Requires minimum data points for valid computation
- Returns 0.0 for insufficient data (consistent with UI expectations)
- Handles edge cases (NaN, zero prices, missing data)

Trading Day Conventions:
- 1D = 1 trading session
- 30D = 30 trading sessions
- 60D = 60 trading sessions
- 365D = 252 trading sessions

Usage:
    from helpers.period_returns import compute_period_return, TRADING_DAYS_MAP
    
    # Compute 30-day return
    ret = compute_period_return(price_series, TRADING_DAYS_MAP['30D'])
    
    # Compute 252-day (1 year) return
    ret = compute_period_return(price_series, TRADING_DAYS_MAP['365D'])
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Trading day mapping for standard periods
TRADING_DAYS_MAP = {
    '1D': 1,      # 1 trading day
    '30D': 30,    # 30 trading days (~1.5 months)
    '60D': 60,    # 60 trading days (~3 months)
    '365D': 252,  # 252 trading days (~1 year, standard market convention)
}


def compute_period_return(
    price_series: pd.Series,
    trading_days: int,
    return_none_on_insufficient_data: bool = False
) -> Optional[float]:
    """
    Compute return over a specified number of trading days.
    
    This is the canonical helper used everywhere for period returns.
    It uses adjusted close/close series from the cached price dataframe.
    
    Algorithm:
    1. Drop NaN values from price series
    2. Sort index ascending
    3. Require at least (trading_days + 1) data points
    4. Compute return: (end_price / start_price) - 1
       where:
       - start_price = series.iloc[-(trading_days + 1)]
       - end_price = series.iloc[-1]
    
    Args:
        price_series: Pandas Series with DatetimeIndex and price values
        trading_days: Number of trading days for the period (e.g., 30, 60, 252)
        return_none_on_insufficient_data: If True, return None instead of 0.0
                                          when insufficient data (default: False)
    
    Returns:
        float: Computed return as decimal (e.g., 0.05 for 5% return)
               Returns 0.0 (or None) if insufficient data
    
    Edge Cases:
        - If price_series is None or empty: returns 0.0 (or None)
        - If insufficient data points: returns 0.0 (or None)
        - If start_price is 0 or NaN: returns 0.0 and logs warning
        - If end_price is NaN: returns 0.0 and logs warning
    
    Examples:
        >>> prices = pd.Series([100, 102, 105, 103, 107], 
        ...                    index=pd.date_range('2024-01-01', periods=5))
        >>> ret = compute_period_return(prices, 3)  # 3 trading days
        >>> # Computes: (107 / 102) - 1 = 0.049
        >>> round(ret, 3)
        0.049
        
        >>> # Insufficient data
        >>> ret = compute_period_return(prices, 10)
        >>> ret
        0.0
    """
    # Handle None or empty series
    if price_series is None or len(price_series) == 0:
        logger.debug(f"compute_period_return: Empty price series for {trading_days} days")
        return None if return_none_on_insufficient_data else 0.0
    
    # Drop NaN values and sort index ascending
    clean_series = price_series.dropna().sort_index()
    
    # Check for sufficient data
    # We need trading_days + 1 points (start point + trading_days worth of data)
    required_points = trading_days + 1
    available_points = len(clean_series)
    
    if available_points < required_points:
        logger.debug(
            f"compute_period_return: Insufficient data for {trading_days} days "
            f"(need {required_points}, have {available_points})"
        )
        return None if return_none_on_insufficient_data else 0.0
    
    # Get start and end prices
    # Start price is at position -(trading_days + 1) from the end
    # End price is the last value
    try:
        start_price = clean_series.iloc[-(trading_days + 1)]
        end_price = clean_series.iloc[-1]
    except (IndexError, KeyError) as e:
        logger.warning(
            f"compute_period_return: Error accessing prices for {trading_days} days: {e}"
        )
        return None if return_none_on_insufficient_data else 0.0
    
    # Handle edge cases
    if pd.isna(start_price) or pd.isna(end_price):
        logger.warning(
            f"compute_period_return: NaN price encountered "
            f"(start={start_price}, end={end_price}) for {trading_days} days"
        )
        return None if return_none_on_insufficient_data else 0.0
    
    if start_price == 0:
        logger.warning(
            f"compute_period_return: Start price is zero for {trading_days} days "
            f"(cannot compute return)"
        )
        return None if return_none_on_insufficient_data else 0.0
    
    # Compute return
    period_return = (end_price / start_price) - 1.0
    
    return float(period_return)


def align_series_for_alpha(
    wave_prices: pd.Series,
    benchmark_prices: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """
    Align wave and benchmark price series to common date range.
    
    This ensures that alpha computation (wave_return - benchmark_return)
    uses the exact same time window for both series.
    
    Args:
        wave_prices: Wave price series with DatetimeIndex
        benchmark_prices: Benchmark price series with DatetimeIndex
    
    Returns:
        Tuple of (aligned_wave_prices, aligned_benchmark_prices)
        Both series will have the same index (intersection of both indices)
    
    Example:
        >>> wave_prices = pd.Series([100, 105, 110], 
        ...                          index=pd.date_range('2024-01-01', periods=3))
        >>> bench_prices = pd.Series([200, 202, 205, 208], 
        ...                           index=pd.date_range('2024-01-01', periods=4))
        >>> w, b = align_series_for_alpha(wave_prices, bench_prices)
        >>> len(w) == len(b)
        True
        >>> (w.index == b.index).all()
        True
    """
    # Find common index (intersection)
    common_index = wave_prices.index.intersection(benchmark_prices.index)
    
    # Slice both series to common index
    aligned_wave = wave_prices.loc[common_index].sort_index()
    aligned_benchmark = benchmark_prices.loc[common_index].sort_index()
    
    return aligned_wave, aligned_benchmark


def compute_alpha(
    wave_prices: pd.Series,
    benchmark_prices: pd.Series,
    trading_days: int,
    return_none_on_insufficient_data: bool = False
) -> Optional[float]:
    """
    Compute alpha (excess return) over a specified period.
    
    Alpha = wave_return - benchmark_return, where both returns are computed
    over the same aligned date range and same trading day window.
    
    Args:
        wave_prices: Wave price series with DatetimeIndex
        benchmark_prices: Benchmark price series with DatetimeIndex
        trading_days: Number of trading days for the period
        return_none_on_insufficient_data: If True, return None instead of 0.0
                                          when insufficient data (default: False)
    
    Returns:
        float: Alpha as decimal (e.g., 0.02 for 2% outperformance)
               Returns 0.0 (or None) if insufficient data
    
    Example:
        >>> wave = pd.Series([100, 105, 110], 
        ...                  index=pd.date_range('2024-01-01', periods=3))
        >>> bench = pd.Series([100, 102, 104], 
        ...                   index=pd.date_range('2024-01-01', periods=3))
        >>> alpha = compute_alpha(wave, bench, 2)
        >>> # Wave return: (110/105)-1 = 0.0476
        >>> # Bench return: (104/102)-1 = 0.0196
        >>> # Alpha: 0.0476 - 0.0196 = 0.028
        >>> round(alpha, 3)
        0.028
    """
    # Align series to common date range
    aligned_wave, aligned_benchmark = align_series_for_alpha(wave_prices, benchmark_prices)
    
    # Compute returns on aligned series
    wave_return = compute_period_return(
        aligned_wave, 
        trading_days,
        return_none_on_insufficient_data=return_none_on_insufficient_data
    )
    benchmark_return = compute_period_return(
        aligned_benchmark, 
        trading_days,
        return_none_on_insufficient_data=return_none_on_insufficient_data
    )
    
    # Handle None returns
    if wave_return is None or benchmark_return is None:
        return None if return_none_on_insufficient_data else 0.0
    
    # Compute alpha
    alpha = wave_return - benchmark_return
    
    return float(alpha)

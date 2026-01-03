"""
Wave Performance Computation - PRICE_BOOK-based Returns Calculator

This module provides functions to compute wave performance metrics (returns)
directly from the canonical PRICE_BOOK (prices_cache.parquet).

Key Features:
- Computes 1D/30D/60D/365D returns for any wave
- Uses wave tickers and weights from waves_engine
- Returns N/A with explicit failure_reason for any issues
- Single source of truth: PRICE_BOOK only

Usage:
    from helpers.wave_performance import compute_wave_returns, compute_all_waves_performance
    from helpers.price_book import get_price_book
    
    price_book = get_price_book()
    result = compute_wave_returns(
        wave_name='S&P 500 Wave',
        price_book=price_book,
        periods=[1, 30, 60, 365]
    )
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Import wave definitions
try:
    from waves_engine import WAVE_WEIGHTS, get_all_waves_universe
    WAVES_ENGINE_AVAILABLE = True
except ImportError:
    WAVES_ENGINE_AVAILABLE = False
    WAVE_WEIGHTS = {}
    logger.warning("waves_engine not available - wave performance computation will be limited")


def compute_wave_returns(
    wave_name: str,
    price_book: pd.DataFrame,
    periods: List[int] = [1, 30, 60, 365]
) -> Dict[str, Any]:
    """
    Compute returns for a single wave using PRICE_BOOK.
    
    This function:
    1. Gets wave tickers and weights from WAVE_WEIGHTS
    2. Filters PRICE_BOOK to those tickers
    3. Computes weighted portfolio returns for specified periods
    4. Returns N/A with failure_reason if any issues occur
    
    Args:
        wave_name: Name of the wave (must exist in WAVE_WEIGHTS)
        price_book: PRICE_BOOK DataFrame (index=dates, columns=tickers, values=prices)
        periods: List of lookback periods in days [1, 30, 60, 365]
        
    Returns:
        Dictionary with:
        - wave_name: str - Name of the wave
        - success: bool - Whether computation succeeded
        - failure_reason: str - Reason for failure (if success=False)
        - returns: Dict[str, float] - Returns for each period (e.g., {'1D': 0.05, '30D': 0.12})
        - tickers: List[str] - Tickers used in computation
        - missing_tickers: List[str] - Required tickers not in PRICE_BOOK
        - date_range: Tuple[str, str] - Date range used for computation
        - coverage_pct: float - Percentage of required tickers present
    """
    result = {
        'wave_name': wave_name,
        'success': False,
        'failure_reason': None,
        'returns': {f'{p}D': None for p in periods},
        'tickers': [],
        'missing_tickers': [],
        'date_range': (None, None),
        'coverage_pct': 0.0
    }
    
    # Check if PRICE_BOOK is valid
    if price_book is None or price_book.empty:
        result['failure_reason'] = 'PRICE_BOOK is empty'
        return result
    
    # Check if waves_engine is available
    if not WAVES_ENGINE_AVAILABLE or not WAVE_WEIGHTS:
        result['failure_reason'] = 'waves_engine not available'
        return result
    
    # Get wave holdings (tickers and weights)
    if wave_name not in WAVE_WEIGHTS:
        result['failure_reason'] = f'Wave "{wave_name}" not found in WAVE_WEIGHTS'
        return result
    
    wave_holdings = WAVE_WEIGHTS[wave_name]
    
    if not wave_holdings:
        result['failure_reason'] = 'Wave has no holdings defined'
        return result
    
    # Extract tickers and weights from holdings
    # Holdings are Holding objects with .ticker and .weight attributes
    try:
        tickers = [h.ticker for h in wave_holdings]
        weights = [h.weight for h in wave_holdings]
    except (AttributeError, TypeError) as e:
        result['failure_reason'] = f'Invalid holdings format: {str(e)}'
        return result
    
    # Normalize weights to sum to 1.0
    total_weight = sum(weights)
    if total_weight == 0:
        result['failure_reason'] = 'Total weight is zero'
        return result
    
    normalized_weights = [w / total_weight for w in weights]
    
    # Check which tickers are in PRICE_BOOK
    available_tickers = []
    available_weights = []
    missing_tickers = []
    
    for ticker, weight in zip(tickers, normalized_weights):
        if ticker in price_book.columns:
            available_tickers.append(ticker)
            available_weights.append(weight)
        else:
            missing_tickers.append(ticker)
    
    result['tickers'] = available_tickers
    result['missing_tickers'] = missing_tickers
    
    # Calculate coverage percentage
    coverage_pct = (len(available_tickers) / len(tickers)) * 100 if tickers else 0.0
    result['coverage_pct'] = coverage_pct
    
    # Check if we have sufficient coverage (at least one ticker)
    if not available_tickers:
        result['failure_reason'] = 'No tickers found in PRICE_BOOK'
        return result
    
    # Renormalize weights for available tickers
    total_available_weight = sum(available_weights)
    if total_available_weight == 0:
        result['failure_reason'] = 'Total available weight is zero'
        return result
    
    renormalized_weights = [w / total_available_weight for w in available_weights]
    
    # Get price data for available tickers
    try:
        ticker_prices = price_book[available_tickers].copy()
    except KeyError as e:
        result['failure_reason'] = f'Error accessing ticker prices: {str(e)}'
        return result
    
    # Check for sufficient data
    if len(ticker_prices) < 2:
        result['failure_reason'] = 'Insufficient price history (need at least 2 days)'
        return result
    
    # Record date range
    result['date_range'] = (
        ticker_prices.index[0].strftime('%Y-%m-%d'),
        ticker_prices.index[-1].strftime('%Y-%m-%d')
    )
    
    # Compute weighted portfolio prices
    # For each day, compute weighted average price (normalized by first day to get index)
    try:
        # Create portfolio index (start at 100)
        first_prices = ticker_prices.iloc[0]
        
        # Handle NaN values in first row
        valid_first_prices = first_prices.notna()
        if not valid_first_prices.any():
            result['failure_reason'] = 'No valid prices on first day'
            return result
        
        # Normalize each ticker to start at 100
        normalized_prices = ticker_prices.div(first_prices, axis=1) * 100
        
        # Compute weighted portfolio value
        portfolio_values = pd.Series(0.0, index=ticker_prices.index)
        for i, ticker in enumerate(available_tickers):
            weight = renormalized_weights[i]
            portfolio_values += normalized_prices[ticker].ffill() * weight
    except Exception as e:
        result['failure_reason'] = f'Error computing portfolio values: {str(e)}'
        return result
    
    # Compute returns for each period
    returns = {}
    max_available_days = len(portfolio_values)
    
    for period in periods:
        try:
            if period >= max_available_days:
                # Not enough history for this period
                returns[f'{period}D'] = None
                continue
            
            # Get value from 'period' days ago and current value
            current_value = portfolio_values.iloc[-1]
            past_value = portfolio_values.iloc[-(period + 1)]
            
            # Handle NaN values
            if pd.isna(current_value) or pd.isna(past_value) or past_value == 0:
                returns[f'{period}D'] = None
                continue
            
            # Calculate return
            ret = (current_value - past_value) / past_value
            returns[f'{period}D'] = ret
            
        except (IndexError, ValueError) as e:
            logger.warning(f"Error computing {period}D return for {wave_name}: {e}")
            returns[f'{period}D'] = None
    
    result['returns'] = returns
    
    # Check if we got at least one valid return
    valid_returns = [v for v in returns.values() if v is not None]
    if not valid_returns:
        result['failure_reason'] = 'No valid returns computed (insufficient date overlap or all NaN)'
        return result
    
    # Success!
    result['success'] = True
    result['failure_reason'] = None
    
    return result


def _get_return_column_name(period: int) -> str:
    """
    Get the display column name for a return period.
    
    Helper function to ensure consistent column naming across the module.
    
    Args:
        period: Number of days (e.g., 1, 30, 60, 365)
        
    Returns:
        Column name (e.g., "1D Return" for 1, "30D" for 30)
    """
    return f'{period}D Return' if period == 1 else f'{period}D'


def compute_all_waves_performance(
    price_book: pd.DataFrame,
    periods: List[int] = [1, 30, 60, 365]
) -> pd.DataFrame:
    """
    Compute performance metrics for all 28 waves using PRICE_BOOK.
    
    This is the main function used by the UI to build the Performance Overview table.
    
    Args:
        price_book: PRICE_BOOK DataFrame (index=dates, columns=tickers, values=prices)
        periods: List of lookback periods in days [1, 30, 60, 365]
        
    Returns:
        DataFrame with columns:
        - Wave: str - Wave name
        - 1D Return, 30D, 60D, 365D: str - Formatted returns (e.g., "+5%") or "N/A"
        - Status/Confidence: str - Data quality indicator
        - Failure_Reason: str - Reason for N/A (if applicable)
        - Coverage_Pct: float - Ticker coverage percentage
    """
    # Get all waves
    if not WAVES_ENGINE_AVAILABLE:
        logger.error("waves_engine not available - cannot compute performance")
        return pd.DataFrame()
    
    try:
        universe = get_all_waves_universe()
        all_waves = universe.get('waves', [])
    except Exception as e:
        logger.error(f"Error getting wave universe: {e}")
        return pd.DataFrame()
    
    if not all_waves:
        logger.error("No waves found in universe")
        return pd.DataFrame()
    
    # Compute performance for each wave
    results = []
    
    for wave_name in all_waves:
        wave_result = compute_wave_returns(wave_name, price_book, periods)
        
        # Build row for display table
        row = {
            'Wave': wave_name,
            'Coverage_Pct': wave_result['coverage_pct'],
            'Failure_Reason': wave_result['failure_reason'] if not wave_result['success'] else None
        }
        
        # Format returns as percentages
        for period in periods:
            key = f'{period}D'
            ret_value = wave_result['returns'].get(key)
            col_name = _get_return_column_name(period)
            
            if ret_value is not None and not pd.isna(ret_value):
                # Format as percentage: +5% or -3%
                row[col_name] = f"{ret_value * 100:+.0f}%"
            else:
                row[col_name] = 'N/A'
        
        # Determine status/confidence based on coverage and success
        if wave_result['success']:
            coverage = wave_result['coverage_pct']
            if coverage >= 95:
                row['Status/Confidence'] = 'Full'
            elif coverage >= 75:
                row['Status/Confidence'] = 'Operational'
            elif coverage >= 50:
                row['Status/Confidence'] = 'Partial'
            else:
                row['Status/Confidence'] = 'Degraded'
        else:
            row['Status/Confidence'] = 'Unavailable'
        
        results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns for display - only include columns that exist
    base_columns = ['Wave']
    period_columns = [_get_return_column_name(period) for period in periods]
    footer_columns = ['Status/Confidence', 'Coverage_Pct', 'Failure_Reason']
    
    display_columns = base_columns + period_columns + footer_columns
    
    # Only select columns that actually exist in the dataframe
    display_columns = [col for col in display_columns if col in df.columns]
    df = df[display_columns]
    
    return df


def get_price_book_diagnostics(price_book: pd.DataFrame) -> Dict[str, Any]:
    """
    Get diagnostic information about PRICE_BOOK for the diagnostics panel.
    
    Args:
        price_book: PRICE_BOOK DataFrame
        
    Returns:
        Dictionary with:
        - path: str - Path to prices_cache.parquet
        - shape: Tuple[int, int] - (rows, columns)
        - date_min: str - Earliest date
        - date_max: str - Latest date
        - total_tickers: int - Total tickers in cache
        - total_days: int - Total trading days
    """
    from helpers.price_book import CANONICAL_CACHE_PATH
    
    if price_book is None or price_book.empty:
        return {
            'path': CANONICAL_CACHE_PATH,
            'shape': (0, 0),
            'date_min': 'N/A',
            'date_max': 'N/A',
            'total_tickers': 0,
            'total_days': 0
        }
    
    return {
        'path': CANONICAL_CACHE_PATH,
        'shape': price_book.shape,
        'date_min': price_book.index[0].strftime('%Y-%m-%d'),
        'date_max': price_book.index[-1].strftime('%Y-%m-%d'),
        'total_tickers': len(price_book.columns),
        'total_days': len(price_book)
    }


def compute_wave_readiness(
    wave_name: str,
    price_book: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute readiness metrics for a single wave against PRICE_BOOK.
    
    This replaces the reliance on data_coverage_summary.csv.
    
    Args:
        wave_name: Name of the wave
        price_book: PRICE_BOOK DataFrame
        
    Returns:
        Dictionary with:
        - wave_name: str
        - data_ready: bool - Whether wave has sufficient data
        - coverage_pct: float - Percentage of tickers present
        - missing_tickers: List[str] - Tickers not in PRICE_BOOK
        - total_tickers: int - Total tickers required
        - history_days: int - Days of history available
        - reason: str - Human-readable status/reason
    """
    result = {
        'wave_name': wave_name,
        'data_ready': False,
        'coverage_pct': 0.0,
        'missing_tickers': [],
        'total_tickers': 0,
        'history_days': 0,
        'reason': 'Unknown'
    }
    
    # Use compute_wave_returns to get ticker coverage info
    wave_result = compute_wave_returns(wave_name, price_book, periods=[1])
    
    result['coverage_pct'] = wave_result['coverage_pct']
    result['missing_tickers'] = wave_result['missing_tickers']
    result['total_tickers'] = len(wave_result['tickers']) + len(wave_result['missing_tickers'])
    result['history_days'] = len(price_book) if price_book is not None else 0
    
    # Determine readiness
    # Consider ready if: coverage >= 90% AND at least 30 days of history AND computation succeeded
    if wave_result['success']:
        if wave_result['coverage_pct'] >= 90 and result['history_days'] >= 30:
            result['data_ready'] = True
            result['reason'] = 'OK'
        elif wave_result['coverage_pct'] < 90:
            result['data_ready'] = False
            result['reason'] = f"Insufficient coverage ({wave_result['coverage_pct']:.1f}%)"
        elif result['history_days'] < 30:
            result['data_ready'] = False
            result['reason'] = f"Insufficient history ({result['history_days']} days)"
        else:
            result['data_ready'] = False
            result['reason'] = 'Failed readiness threshold'
    else:
        result['data_ready'] = False
        result['reason'] = wave_result['failure_reason'] or 'Unknown error'
    
    return result


def compute_all_waves_readiness(price_book: pd.DataFrame) -> pd.DataFrame:
    """
    Compute readiness metrics for all waves.
    
    Args:
        price_book: PRICE_BOOK DataFrame
        
    Returns:
        DataFrame with readiness metrics for all waves
    """
    # Get all waves
    if not WAVES_ENGINE_AVAILABLE:
        return pd.DataFrame()
    
    try:
        universe = get_all_waves_universe()
        all_waves = universe.get('waves', [])
    except Exception as e:
        logger.error(f"Error getting wave universe: {e}")
        return pd.DataFrame()
    
    results = []
    for wave_name in all_waves:
        readiness = compute_wave_readiness(wave_name, price_book)
        results.append({
            'wave_name': wave_name,
            'data_ready': readiness['data_ready'],
            'reason': readiness['reason'],
            'coverage_pct': readiness['coverage_pct'],
            'history_days': readiness['history_days'],
            'missing_tickers': ', '.join(readiness['missing_tickers']) if readiness['missing_tickers'] else '',
            'total_tickers': readiness['total_tickers']
        })
    
    return pd.DataFrame(results)

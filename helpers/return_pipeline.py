"""
Return Pipeline

This module provides the single return pipeline function that computes
standardized returns for waves and benchmarks with alpha attribution.

Key Features:
- Standardized output dataframe with wave_return, benchmark_return, alpha
- Placeholder columns for overlay_return components (VIX, etc.)
- Uses canonical data access helper
- Minimal dependencies and clean architecture

Usage:
    from helpers.return_pipeline import compute_wave_returns_pipeline
    
    # Compute returns for a wave
    returns_df = compute_wave_returns_pipeline('sp500_wave')
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np
import sys
import os

# Add parent helpers directory to path to avoid __init__.py imports
helpers_dir = os.path.dirname(os.path.abspath(__file__))
if helpers_dir not in sys.path:
    sys.path.insert(0, helpers_dir)

import canonical_data
import wave_registry
get_canonical_price_data = canonical_data.get_canonical_price_data
get_wave_by_id = wave_registry.get_wave_by_id

logger = logging.getLogger(__name__)


def compute_wave_returns_pipeline(
    wave_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute returns for a wave with benchmark and alpha attribution.
    
    This is the single return pipeline function that produces standardized
    output for all downstream analysis including alpha attribution, dynamic
    benchmarks, and VIX overlays.
    
    Args:
        wave_id: Wave identifier (e.g., 'sp500_wave')
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        
    Returns:
        DataFrame with columns:
        - date: DatetimeIndex
        - wave_return: Daily return of the wave
        - benchmark_return: Daily return of the benchmark portfolio
        - alpha: wave_return - benchmark_return
        - overlay_return_vix: Placeholder (NaN) for VIX overlay component
        - overlay_return_custom: Placeholder (NaN) for custom overlay component
        
    Example:
        >>> df = compute_wave_returns_pipeline('sp500_wave')
        >>> print(df.columns)
        Index(['wave_return', 'benchmark_return', 'alpha', 
               'overlay_return_vix', 'overlay_return_custom'], dtype='object')
    """
    try:
        logger.info(f"Computing returns for wave: {wave_id}")
        
        # Get wave from registry
        wave = get_wave_by_id(wave_id)
        if wave is None:
            logger.error(f"Wave not found in registry: {wave_id}")
            return _empty_returns_dataframe()
        
        # Parse wave tickers
        wave_tickers = _parse_ticker_list(wave.get('ticker_normalized', ''))
        if not wave_tickers:
            logger.error(f"No tickers found for wave: {wave_id}")
            return _empty_returns_dataframe()
        
        # Parse benchmark recipe
        benchmark_recipe = wave.get('benchmark_recipe', {})
        if not benchmark_recipe:
            logger.warning(f"No benchmark recipe found for wave: {wave_id}")
            # Use empty benchmark (returns will be NaN)
            benchmark_tickers = []
        else:
            benchmark_tickers = list(benchmark_recipe.keys())
        
        # Get all required tickers
        all_tickers = list(set(wave_tickers + benchmark_tickers))
        
        # Get price data (cache-first)
        price_data = get_canonical_price_data(
            tickers=all_tickers,
            start_date=start_date,
            end_date=end_date
        )
        
        if price_data.empty:
            logger.error("No price data available")
            return _empty_returns_dataframe()
        
        # Compute wave returns
        wave_returns = _compute_portfolio_returns(
            price_data, wave_tickers, equal_weight=True
        )
        
        # Compute benchmark returns
        if benchmark_tickers and benchmark_recipe:
            benchmark_weights = [benchmark_recipe[ticker] for ticker in benchmark_tickers]
            benchmark_returns = _compute_portfolio_returns(
                price_data, benchmark_tickers, weights=benchmark_weights
            )
        else:
            benchmark_returns = pd.Series(np.nan, index=price_data.index)
        
        # Compute alpha
        alpha = wave_returns - benchmark_returns
        
        # Create output dataframe
        result = pd.DataFrame({
            'wave_return': wave_returns,
            'benchmark_return': benchmark_returns,
            'alpha': alpha,
            'overlay_return_vix': np.nan,  # Placeholder for VIX overlay
            'overlay_return_custom': np.nan  # Placeholder for custom overlay
        })
        
        logger.info(f"Returns computed: {len(result)} days")
        return result
        
    except Exception as e:
        logger.error(f"Error computing returns for {wave_id}: {e}", exc_info=True)
        return _empty_returns_dataframe()


def _parse_ticker_list(ticker_string: str) -> list:
    """Parse comma-separated ticker string into list."""
    if not ticker_string or pd.isna(ticker_string):
        return []
    
    tickers = [t.strip() for t in ticker_string.split(',')]
    return [t for t in tickers if t]  # Filter empty strings


def _compute_portfolio_returns(
    price_data: pd.DataFrame,
    tickers: list,
    weights: Optional[list] = None,
    equal_weight: bool = False
) -> pd.Series:
    """
    Compute portfolio returns from price data.
    
    Args:
        price_data: DataFrame with dates as index and tickers as columns
        tickers: List of tickers to include
        weights: Optional list of weights (must sum to 1.0)
        equal_weight: If True, use equal weights for all tickers
        
    Returns:
        Series of daily portfolio returns
    """
    # Filter to available tickers
    available_tickers = [t for t in tickers if t in price_data.columns]
    
    if not available_tickers:
        logger.warning(f"No available tickers found in price data")
        return pd.Series(np.nan, index=price_data.index)
    
    # Get prices for portfolio
    portfolio_prices = price_data[available_tickers]
    
    # Set weights
    if equal_weight or weights is None:
        n = len(available_tickers)
        portfolio_weights = [1.0 / n] * n
    else:
        # Use provided weights, filtering to available tickers
        portfolio_weights = []
        for ticker in available_tickers:
            idx = tickers.index(ticker)
            portfolio_weights.append(weights[idx])
        
        # Normalize weights to sum to 1.0
        total_weight = sum(portfolio_weights)
        if total_weight > 0:
            portfolio_weights = [w / total_weight for w in portfolio_weights]
    
    # Compute individual ticker returns
    # fill_method=None is explicit to avoid deprecation warnings
    ticker_returns = portfolio_prices.pct_change(fill_method=None)
    
    # Compute weighted portfolio returns
    portfolio_returns = (ticker_returns * portfolio_weights).sum(axis=1)
    
    return portfolio_returns


def _empty_returns_dataframe() -> pd.DataFrame:
    """Return an empty dataframe with the standard return pipeline columns."""
    return pd.DataFrame(columns=[
        'wave_return',
        'benchmark_return', 
        'alpha',
        'overlay_return_vix',
        'overlay_return_custom'
    ])

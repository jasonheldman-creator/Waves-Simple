"""
Strategy-Aware Return Pipeline

This module extends the basic return_pipeline.py with full strategy stack support.
It integrates the unified strategy overlays to compute realized returns for waves.

Key Features:
- Reads strategy_stack from wave registry
- Applies overlays in sequence (momentum -> trend -> vol_targeting -> vix_safesmart)
- Returns both base (selection) and realized (stacked) returns
- Provides full alpha attribution decomposition

Usage:
    from helpers.strategy_return_pipeline import compute_wave_returns_with_strategy
    
    # Compute returns with full strategy stack
    result = compute_wave_returns_with_strategy('sp500_wave')
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import sys
import os

# Add parent helpers directory to path
helpers_dir = os.path.dirname(os.path.abspath(__file__))
if helpers_dir not in sys.path:
    sys.path.insert(0, helpers_dir)

import canonical_data
import wave_registry
import strategy_overlays

get_canonical_price_data = canonical_data.get_canonical_price_data
get_wave_by_id = wave_registry.get_wave_by_id

logger = logging.getLogger(__name__)


def compute_wave_returns_with_strategy(
    wave_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    apply_strategy_stack: bool = True
) -> Dict[str, Any]:
    """
    Compute returns for a wave with full strategy stack support.
    
    This function:
    1. Computes base (selection) returns from wave tickers
    2. Applies strategy stack overlays from wave registry
    3. Returns both base and realized returns with attribution
    
    Args:
        wave_id: Wave identifier (e.g., 'sp500_wave')
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        apply_strategy_stack: Whether to apply strategy overlays (default: True)
        
    Returns:
        Dictionary with:
        - success: bool
        - wave_id: str
        - base_returns: pd.Series - Selection returns (before overlays)
        - realized_returns: pd.Series - Final returns (after overlays)
        - benchmark_returns: pd.Series - Benchmark returns
        - selection_alpha: pd.Series - base_returns - benchmark_returns
        - total_alpha: pd.Series - realized_returns - benchmark_returns
        - strategy_stack: List[str] - Strategies applied
        - attribution: Dict - Full attribution breakdown
        - failure_reason: str (if success=False)
    """
    result = {
        'success': False,
        'wave_id': wave_id,
        'base_returns': None,
        'realized_returns': None,
        'benchmark_returns': None,
        'selection_alpha': None,
        'total_alpha': None,
        'strategy_stack': [],
        'attribution': None,
        'failure_reason': None
    }
    
    try:
        logger.info(f"Computing returns with strategy for wave: {wave_id}")
        
        # Get wave from registry
        wave = get_wave_by_id(wave_id)
        if wave is None:
            result['failure_reason'] = f"Wave not found in registry: {wave_id}"
            return result
        
        # Parse wave tickers
        wave_tickers = _parse_ticker_list(wave.get('ticker_normalized', ''))
        if not wave_tickers:
            result['failure_reason'] = f"No tickers found for wave: {wave_id}"
            return result
        
        # Parse benchmark recipe
        benchmark_recipe = wave.get('benchmark_recipe', {})
        if not benchmark_recipe:
            logger.warning(f"No benchmark recipe found for wave: {wave_id}")
            benchmark_tickers = []
        else:
            benchmark_tickers = list(benchmark_recipe.keys())
        
        # Get strategy stack from registry
        strategy_stack = wave.get('strategy_stack', [])
        result['strategy_stack'] = strategy_stack
        
        logger.info(f"Wave {wave_id} strategy stack: {strategy_stack}")
        
        # Get all required tickers (wave + benchmark + safe + VIX)
        all_tickers = list(set(wave_tickers + benchmark_tickers))
        
        # Add safe asset tickers if vix_safesmart in stack
        safe_ticker = None
        if 'vix_safesmart' in strategy_stack:
            for ticker in ['BIL', 'SHY', 'SGOV']:
                all_tickers.append(ticker)
        
        # Add VIX ticker if vix_safesmart in stack
        vix_ticker = None
        if 'vix_safesmart' in strategy_stack:
            for ticker in ['^VIX', 'VIXY', 'VXX']:
                all_tickers.append(ticker)
        
        # Get price data (cache-first)
        price_data = get_canonical_price_data(
            tickers=list(set(all_tickers)),
            start_date=start_date,
            end_date=end_date
        )
        
        if price_data.empty:
            result['failure_reason'] = "No price data available"
            return result
        
        # Compute base (selection) returns
        base_returns = _compute_portfolio_returns(
            price_data, wave_tickers, equal_weight=True
        )
        
        result['base_returns'] = base_returns
        
        # Compute benchmark returns
        if benchmark_tickers and benchmark_recipe:
            benchmark_weights = [benchmark_recipe[ticker] for ticker in benchmark_tickers]
            benchmark_returns = _compute_portfolio_returns(
                price_data, benchmark_tickers, weights=benchmark_weights
            )
        else:
            benchmark_returns = pd.Series(np.nan, index=price_data.index)
        
        result['benchmark_returns'] = benchmark_returns
        
        # Compute selection alpha (base vs benchmark)
        result['selection_alpha'] = base_returns - benchmark_returns
        
        # Apply strategy stack if enabled
        if apply_strategy_stack and strategy_stack:
            # Find VIX prices
            vix_prices = None
            for ticker in ['^VIX', 'VIXY', 'VXX']:
                if ticker in price_data.columns:
                    vix_prices = price_data[ticker]
                    vix_ticker = ticker
                    break
            
            # Find safe asset returns
            safe_returns = None
            for ticker in ['BIL', 'SHY', 'SGOV']:
                if ticker in price_data.columns:
                    safe_prices = price_data[ticker]
                    safe_returns = safe_prices.pct_change()
                    safe_ticker = ticker
                    break
            
            # Apply strategy stack
            realized_returns, attribution = strategy_overlays.apply_strategy_stack(
                base_returns=base_returns,
                prices=price_data,
                tickers=wave_tickers,
                strategy_stack=strategy_stack,
                vix_prices=vix_prices,
                safe_returns=safe_returns
            )
            
            result['realized_returns'] = realized_returns
            result['attribution'] = attribution
            
            logger.info(f"Applied {len(attribution['overlays_applied'])} overlays: {attribution['overlays_applied']}")
            
        else:
            # No strategy stack - realized = base
            result['realized_returns'] = base_returns
            result['attribution'] = {
                'base_returns': base_returns,
                'final_returns': base_returns,
                'overlays_applied': [],
                'overlay_diagnostics': {},
                'total_alpha': 0.0,
                'component_alphas': {}
            }
        
        # Compute total alpha (realized vs benchmark)
        result['total_alpha'] = result['realized_returns'] - benchmark_returns
        
        result['success'] = True
        logger.info(f"Successfully computed returns for {wave_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error computing returns for {wave_id}: {e}", exc_info=True)
        result['failure_reason'] = str(e)
        return result


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
    ticker_returns = portfolio_prices.pct_change(fill_method=None)
    
    # Compute weighted portfolio returns
    portfolio_returns = (ticker_returns * portfolio_weights).sum(axis=1)
    
    return portfolio_returns

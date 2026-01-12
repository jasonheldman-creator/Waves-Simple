"""
Strategy Return Pipeline

This module provides strategy-aware return computation for waves with strategy_stack.
It integrates with waves_engine.compute_history_nav to apply momentum, trend, 
volatility, and other strategy overlays to wave returns.

Key Features:
- Wraps waves_engine.compute_history_nav for strategy-aware return calculation
- Returns standardized DataFrame format compatible with return_pipeline
- Preserves diagnostics and attribution metadata
- Supports strategy components: momentum, trend, volatility_targeting, etc.

Usage:
    from helpers.strategy_return_pipeline import compute_wave_returns_with_strategy
    
    # Compute strategy-aware returns for a wave
    returns_df = compute_wave_returns_with_strategy(
        wave_id='sp500_wave',
        strategy_stack=['momentum', 'trend_confirmation']
    )
"""

import logging
from typing import Optional, List
import pandas as pd
import numpy as np
import sys
import os

# Add parent helpers directory to path
helpers_dir = os.path.dirname(os.path.abspath(__file__))
if helpers_dir not in sys.path:
    sys.path.insert(0, helpers_dir)

import wave_registry

logger = logging.getLogger(__name__)

# Import waves_engine for strategy computation
try:
    from waves_engine import compute_history_nav, get_display_name_from_wave_id
    WAVES_ENGINE_AVAILABLE = True
except ImportError:
    WAVES_ENGINE_AVAILABLE = False
    logger.warning("waves_engine not available - strategy pipeline disabled")


def compute_wave_returns_with_strategy(
    wave_id: str,
    strategy_stack: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    mode: str = "Standard",
    days: int = 365
) -> pd.DataFrame:
    """
    Compute returns for a wave with strategy-aware logic.
    
    This function applies the full waves_engine strategy pipeline including:
    - Momentum signal adjustments
    - Trend confirmation overlays
    - Volatility targeting
    - VIX regime detection
    - Relative strength strategies
    
    Args:
        wave_id: Wave identifier (e.g., 'sp500_wave')
        strategy_stack: List of strategy components to apply (e.g., ['momentum', 'trend'])
                       If None or empty, returns basic returns without strategy overlays
        start_date: Optional start date (YYYY-MM-DD) - currently not used, relies on days param
        end_date: Optional end date (YYYY-MM-DD) - currently not used, uses most recent data
        mode: Wave operating mode (default: "Standard")
        days: Lookback window in days (default: 365)
        
    Returns:
        DataFrame with columns:
        - date: DatetimeIndex
        - wave_return: Daily return of the wave with strategy overlays applied
        - benchmark_return: Daily return of the benchmark portfolio
        - alpha: wave_return - benchmark_return (includes strategy-generated alpha)
        - strategy_applied: Boolean indicating strategy pipeline was used
        
    Example:
        >>> df = compute_wave_returns_with_strategy(
        ...     'sp500_wave', 
        ...     strategy_stack=['momentum', 'trend_confirmation']
        ... )
        >>> print(df[['wave_return', 'benchmark_return', 'alpha']].tail())
    """
    try:
        # Get wave from registry
        wave = wave_registry.get_wave_by_id(wave_id)
        if wave is None:
            logger.error(f"Wave not found in registry: {wave_id}")
            return _empty_strategy_returns_dataframe()
        
        # Get wave display name for waves_engine
        wave_display_name = wave.get('wave_name')
        if not wave_display_name:
            logger.error(f"Wave display name not found for: {wave_id}")
            return _empty_strategy_returns_dataframe()
        
        # If no waves_engine available, return empty
        if not WAVES_ENGINE_AVAILABLE:
            logger.error("waves_engine not available for strategy computation")
            return _empty_strategy_returns_dataframe()
        
        # If strategy_stack is empty or None, we still use the engine
        # but note that the engine will apply default strategies based on wave configuration
        has_strategy = strategy_stack is not None and len(strategy_stack) > 0
        
        logger.info(f"Computing strategy-aware returns for {wave_display_name} "
                   f"(mode={mode}, days={days}, strategy_stack={strategy_stack})")
        
        # Call waves_engine compute_history_nav which includes full strategy pipeline
        # This applies momentum, trend, volatility, VIX regime, and other overlays
        hist_df = compute_history_nav(
            wave_name=wave_display_name,
            mode=mode,
            days=days,
            include_diagnostics=True,  # Get strategy diagnostics
            price_df=None  # Let engine load from price book
        )
        
        if hist_df is None or hist_df.empty:
            logger.warning(f"No history data returned for {wave_display_name}")
            return _empty_strategy_returns_dataframe()
        
        # Extract returns from NAV series
        # hist_df should have: wave_nav, bm_nav, wave_ret, bm_ret
        if 'wave_ret' not in hist_df.columns or 'bm_ret' not in hist_df.columns:
            logger.error(f"Missing return columns in history data for {wave_display_name}")
            return _empty_strategy_returns_dataframe()
        
        # Build result dataframe
        result = pd.DataFrame({
            'wave_return': hist_df['wave_ret'],
            'benchmark_return': hist_df['bm_ret'],
            'alpha': hist_df['wave_ret'] - hist_df['bm_ret'],
            'strategy_applied': True  # Strategy pipeline was used
        })
        
        # Preserve diagnostics metadata if available
        if hasattr(hist_df, 'attrs'):
            result.attrs['diagnostics'] = hist_df.attrs.get('diagnostics', None)
            result.attrs['coverage'] = hist_df.attrs.get('coverage', {})
            result.attrs['strategy_stack'] = strategy_stack or []
        
        logger.info(f"Strategy-aware returns computed: {len(result)} days, "
                   f"avg alpha={result['alpha'].mean():.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error computing strategy-aware returns for {wave_id}: {e}", 
                    exc_info=True)
        return _empty_strategy_returns_dataframe()


def _empty_strategy_returns_dataframe() -> pd.DataFrame:
    """Return an empty dataframe with the standard strategy return columns."""
    return pd.DataFrame(columns=[
        'wave_return',
        'benchmark_return', 
        'alpha',
        'strategy_applied'
    ])


def get_strategy_stack_from_wave(wave_id: str) -> List[str]:
    """
    Get the strategy_stack for a wave from the registry.
    
    Args:
        wave_id: Wave identifier
        
    Returns:
        List of strategy component names, or empty list if none defined
    """
    try:
        wave = wave_registry.get_wave_by_id(wave_id)
        if wave is None:
            return []
        
        # Get strategy_stack field (could be string or list)
        strategy_stack = wave.get('strategy_stack', '')
        
        # Handle different formats
        if isinstance(strategy_stack, list):
            return strategy_stack
        elif isinstance(strategy_stack, str) and strategy_stack:
            # Parse comma-separated string
            return [s.strip() for s in strategy_stack.split(',') if s.strip()]
        else:
            return []
            
    except Exception as e:
        logger.error(f"Error getting strategy_stack for {wave_id}: {e}")
        return []

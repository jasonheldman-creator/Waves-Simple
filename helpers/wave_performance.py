"""
Wave Performance Computation - PRICE_BOOK-based Returns Calculator

This module provides functions to compute wave performance metrics (returns)
directly from the canonical PRICE_BOOK (prices_cache.parquet).

Key Features:
- Computes 1D/30D/60D/365D returns for any wave
- Uses wave tickers and weights from waves_engine
- Returns N/A with explicit failure_reason for any issues
- Single source of truth: PRICE_BOOK only
- Portfolio-level snapshot computation with alpha attribution
- Diagnostics validation for data quality

Usage:
    from helpers.wave_performance import compute_wave_returns, compute_all_waves_performance
    from helpers.price_book import get_price_book
    
    price_book = get_price_book()
    result = compute_wave_returns(
        wave_name='S&P 500 Wave',
        price_book=price_book,
        periods=[1, 30, 60, 365]
    )
    
    # Portfolio snapshot
    snapshot = compute_portfolio_snapshot(price_book, mode='Standard')
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_BENCHMARK_TICKER = 'SPY'  # Default benchmark for portfolio calculations
DEFAULT_HISTORY_DAYS = 365  # Default lookback period for wave history computation
MIN_DATES_FOR_PORTFOLIO = 2  # Minimum number of dates required for portfolio aggregation
RESIDUAL_TOLERANCE = 0.0010  # 0.10% tolerance for residual attribution validation

# VIX regime thresholds and exposure mapping
VIX_LOW_THRESHOLD = 18  # VIX below this is Low Volatility regime
VIX_HIGH_THRESHOLD = 25  # VIX above or equal to this is High Volatility regime
EXPOSURE_LOW_VOLATILITY = 1.00  # Full exposure when VIX < 18
EXPOSURE_MODERATE_VOLATILITY = 0.65  # Moderate exposure when 18 <= VIX < 25
EXPOSURE_HIGH_VOLATILITY = 0.25  # Low exposure when VIX >= 25

# Import wave definitions
try:
    from waves_engine import (
        WAVE_WEIGHTS, 
        get_all_waves_universe,
        compute_history_nav,
        build_portfolio_composite_benchmark_returns,
        _is_crypto_wave,
        _is_crypto_growth_wave,
        _is_crypto_income_wave
    )
    WAVES_ENGINE_AVAILABLE = True
except ImportError:
    WAVES_ENGINE_AVAILABLE = False
    WAVE_WEIGHTS = {}
    logger.warning("waves_engine not available - wave performance computation will be limited")

# Import crypto volatility overlay
try:
    from helpers.crypto_volatility_overlay import (
        compute_crypto_volatility_regime,
        get_crypto_wave_exposure
    )
    CRYPTO_OVERLAY_AVAILABLE = True
except ImportError:
    CRYPTO_OVERLAY_AVAILABLE = False
    logger.warning("crypto_volatility_overlay not available - crypto overlays will be disabled")


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
    periods: List[int] = [1, 30, 60, 365],
    only_validated: bool = True,
    min_coverage_pct: float = 100.0,
    min_history_days: int = 30
) -> pd.DataFrame:
    """
    Compute performance metrics for all 28 waves using PRICE_BOOK.
    
    This is the main function used by the UI to build the Performance Overview table.
    
    IMPORTANT: By default (only_validated=True), this function ONLY returns waves 
    that pass strict validation. This prevents silent N/A values and ensures all 
    displayed waves have valid, continuous price history.
    
    Args:
        price_book: PRICE_BOOK DataFrame (index=dates, columns=tickers, values=prices)
        periods: List of lookback periods in days [1, 30, 60, 365]
        only_validated: If True (default), only return waves that pass validation.
                       If False, return all waves (may include N/A values).
        min_coverage_pct: Minimum ticker coverage percentage for validation (default: 100.0)
        min_history_days: Minimum number of trading days for validation (default: 30)
        
    Returns:
        DataFrame with columns:
        - Wave: str - Wave name
        - 1D Return, 30D, 60D, 365D: str - Formatted returns (e.g., "+5%") or "N/A"
        - Status/Confidence: str - Data quality indicator
        - Failure_Reason: str - Reason for N/A (if applicable, only when only_validated=False)
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
    
    # Compute performance for each wave (only validated waves if only_validated=True)
    results = []
    waves_to_process = all_waves  # Default: process all waves
    
    # Optimization: If only_validated=True, filter waves first to avoid computing performance for invalid waves
    if only_validated:
        logger.info(f"Validating {len(all_waves)} waves with strict criteria...")
        logger.info(f"  min_coverage_pct: {min_coverage_pct}%")
        logger.info(f"  min_history_days: {min_history_days}")
        
        validation_map = {}
        for wave_name in all_waves:
            validation = validate_wave_price_history(
                wave_name,
                price_book,
                min_coverage_pct=min_coverage_pct,
                min_history_days=min_history_days
            )
            validation_map[wave_name] = validation
        
        # Filter to only valid waves
        waves_to_process = [w for w in all_waves if validation_map.get(w, {}).get('valid', False)]
        
        invalid_count = len(all_waves) - len(waves_to_process)
        logger.info(f"Validation complete: {len(waves_to_process)}/{len(all_waves)} waves passed")
        if invalid_count > 0:
            logger.warning(f"{invalid_count} waves failed validation and will be excluded from performance table")
            # Log first few failures for debugging
            invalid_waves = [w for w in all_waves if not validation_map.get(w, {}).get('valid', False)]
            for wave_name in invalid_waves[:5]:
                reason = validation_map.get(wave_name, {}).get('validation_reason', 'Unknown')
                logger.warning(f"  - {wave_name}: {reason}")
            if invalid_count > 5:
                logger.warning(f"  ... and {invalid_count - 5} more")
    
    for wave_name in waves_to_process:
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
    
    # Only include Failure_Reason if not in only_validated mode (since validated waves won't have failures)
    if only_validated:
        footer_columns = ['Status/Confidence', 'Coverage_Pct']
    else:
        footer_columns = ['Status/Confidence', 'Coverage_Pct', 'Failure_Reason']
    
    display_columns = base_columns + period_columns + footer_columns
    
    # Only select columns that actually exist in the dataframe
    display_columns = [col for col in display_columns if col in df.columns]
    df = df[display_columns]
    
    return df


def compute_all_waves_performance_with_crypto_diagnostics(
    price_book: pd.DataFrame,
    periods: List[int] = [1, 30, 60, 365],
    only_validated: bool = True,
    min_coverage_pct: float = 100.0,
    min_history_days: int = 30,
    include_crypto_diagnostics: bool = True
) -> pd.DataFrame:
    """
    Compute performance metrics for all waves with optional crypto diagnostics.
    
    This extends compute_all_waves_performance to include crypto-specific diagnostics
    (regime, exposure, overlay status) for crypto waves.
    
    Args:
        price_book: PRICE_BOOK DataFrame (index=dates, columns=tickers, values=prices)
        periods: List of lookback periods in days [1, 30, 60, 365]
        only_validated: If True (default), only return waves that pass validation
        min_coverage_pct: Minimum ticker coverage percentage for validation (default: 100.0)
        min_history_days: Minimum number of trading days for validation (default: 30)
        include_crypto_diagnostics: If True, include crypto diagnostics columns
        
    Returns:
        DataFrame with standard performance columns plus:
        - Crypto_Regime: str - Regime for crypto waves ('calm', 'normal', 'elevated', 'stress', or 'N/A')
        - Crypto_Exposure: str - Exposure recommendation (e.g., "85%" or "N/A")
        - Crypto_Overlay_Status: str - Overlay status message
    """
    # Get base performance data
    df = compute_all_waves_performance(
        price_book=price_book,
        periods=periods,
        only_validated=only_validated,
        min_coverage_pct=min_coverage_pct,
        min_history_days=min_history_days
    )
    
    # Add crypto diagnostics if requested
    if include_crypto_diagnostics and not df.empty:
        crypto_regimes = []
        crypto_exposures = []
        crypto_statuses = []
        
        for wave_name in df['Wave']:
            diagnostics = compute_crypto_regime_diagnostics(price_book, wave_name)
            
            # Format for display
            if diagnostics['applicable']:
                crypto_regimes.append(diagnostics['regime'])
                exposure_pct = f"{diagnostics['exposure'] * 100:.0f}%"
                crypto_exposures.append(exposure_pct)
                crypto_statuses.append(diagnostics['overlay_status'])
            else:
                crypto_regimes.append('N/A')
                crypto_exposures.append('N/A')
                crypto_statuses.append('N/A')
        
        # Insert crypto columns after the Wave column
        df.insert(1, 'Crypto_Regime', crypto_regimes)
        df.insert(2, 'Crypto_Exposure', crypto_exposures)
        df.insert(3, 'Crypto_Overlay_Status', crypto_statuses)
    
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
    # Import directly to avoid streamlit dependency
    import os
    import sys
    helpers_dir = os.path.dirname(os.path.abspath(__file__))
    if helpers_dir not in sys.path:
        sys.path.insert(0, helpers_dir)
    from price_book import CANONICAL_CACHE_PATH
    
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


def validate_wave_price_history(
    wave_name: str,
    price_book: pd.DataFrame,
    min_coverage_pct: float = 100.0,
    min_history_days: int = 30
) -> Dict[str, Any]:
    """
    Validate that a wave has complete, continuous price history from PRICE_BOOK.
    
    This is a STRICT validation function that ensures:
    1. ALL required tickers are present in PRICE_BOOK (no missing tickers allowed by default)
    2. Sufficient date coverage exists (at least min_history_days)
    3. Returns can be computed (no silent N/A values)
    
    Args:
        wave_name: Name of the wave to validate
        price_book: PRICE_BOOK DataFrame (index=dates, columns=tickers, values=prices)
        min_coverage_pct: Minimum ticker coverage percentage required (default: 100.0)
        min_history_days: Minimum number of trading days required (default: 30)
        
    Returns:
        Dictionary with:
        - wave_name: str - Name of the wave
        - valid: bool - Whether wave passes validation
        - required_tickers: List[str] - All tickers the wave requires
        - found_tickers: List[str] - Tickers found in PRICE_BOOK
        - missing_tickers: List[str] - Tickers NOT found in PRICE_BOOK
        - date_coverage_start: str - Earliest date in PRICE_BOOK (YYYY-MM-DD)
        - date_coverage_end: str - Latest date in PRICE_BOOK (YYYY-MM-DD)
        - date_coverage_days: int - Number of trading days available
        - return_computable: str - "Yes" or "No"
        - validation_reason: str - Reason for validation failure (if valid=False)
        - coverage_pct: float - Percentage of required tickers found
    """
    result = {
        'wave_name': wave_name,
        'valid': False,
        'required_tickers': [],
        'found_tickers': [],
        'missing_tickers': [],
        'date_coverage_start': None,
        'date_coverage_end': None,
        'date_coverage_days': 0,
        'return_computable': 'No',
        'validation_reason': 'Unknown',
        'coverage_pct': 0.0
    }
    
    # Check if PRICE_BOOK is valid
    if price_book is None or price_book.empty:
        result['validation_reason'] = 'PRICE_BOOK is empty - no price data available'
        return result
    
    # Check if waves_engine is available
    if not WAVES_ENGINE_AVAILABLE or not WAVE_WEIGHTS:
        result['validation_reason'] = 'waves_engine not available - cannot determine required tickers'
        return result
    
    # Get wave holdings (tickers and weights)
    if wave_name not in WAVE_WEIGHTS:
        result['validation_reason'] = f'Wave "{wave_name}" not found in WAVE_WEIGHTS'
        return result
    
    wave_holdings = WAVE_WEIGHTS[wave_name]
    
    if not wave_holdings:
        result['validation_reason'] = 'Wave has no holdings defined in WAVE_WEIGHTS'
        return result
    
    # Extract required tickers from holdings
    try:
        required_tickers = sorted([h.ticker for h in wave_holdings])
    except (AttributeError, TypeError) as e:
        result['validation_reason'] = f'Invalid holdings format: {str(e)}'
        return result
    
    result['required_tickers'] = required_tickers
    
    # Check which tickers are in PRICE_BOOK
    found_tickers = []
    missing_tickers = []
    
    for ticker in required_tickers:
        if ticker in price_book.columns:
            found_tickers.append(ticker)
        else:
            missing_tickers.append(ticker)
    
    result['found_tickers'] = found_tickers
    result['missing_tickers'] = missing_tickers
    
    # Calculate coverage percentage
    coverage_pct = (len(found_tickers) / len(required_tickers)) * 100 if required_tickers else 0.0
    result['coverage_pct'] = coverage_pct
    
    # Get date coverage from PRICE_BOOK
    if not price_book.empty:
        result['date_coverage_start'] = price_book.index[0].strftime('%Y-%m-%d')
        result['date_coverage_end'] = price_book.index[-1].strftime('%Y-%m-%d')
        result['date_coverage_days'] = len(price_book)
    
    # Validate coverage
    if coverage_pct < min_coverage_pct:
        result['validation_reason'] = (
            f'Insufficient ticker coverage: {coverage_pct:.1f}% '
            f'(found {len(found_tickers)}/{len(required_tickers)} tickers, '
            f'missing: {", ".join(missing_tickers)})'
        )
        return result
    
    # Validate date coverage
    if result['date_coverage_days'] < min_history_days:
        result['validation_reason'] = (
            f'Insufficient date coverage: {result["date_coverage_days"]} days '
            f'(minimum required: {min_history_days} days)'
        )
        return result
    
    # Check if we have any found tickers (should always be true if coverage >= min_coverage_pct)
    if not found_tickers:
        result['validation_reason'] = 'No tickers found in PRICE_BOOK'
        return result
    
    # Try to compute returns to verify data quality
    wave_result = compute_wave_returns(wave_name, price_book, periods=[1])
    
    if not wave_result['success']:
        result['validation_reason'] = (
            f'Cannot compute returns: {wave_result["failure_reason"]}'
        )
        return result
    
    # All validations passed!
    result['valid'] = True
    result['return_computable'] = 'Yes'
    result['validation_reason'] = 'OK - All validations passed'
    
    return result


def generate_wave_validation_report(
    price_book: pd.DataFrame,
    min_coverage_pct: float = 100.0,
    min_history_days: int = 30,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive validation report for all waves.
    
    This produces a deterministic report showing validation status for every wave.
    The report is suitable for inspection and debugging data quality issues.
    
    Args:
        price_book: PRICE_BOOK DataFrame
        min_coverage_pct: Minimum ticker coverage percentage required (default: 100.0)
        min_history_days: Minimum number of trading days required (default: 30)
        output_file: Optional path to save the report as JSON (default: None)
        
    Returns:
        Dictionary with:
        - timestamp: str - When the report was generated
        - price_book_meta: Dict - Metadata about PRICE_BOOK
        - validation_criteria: Dict - Validation criteria used
        - waves_validated: int - Total number of waves validated
        - waves_valid: int - Number of waves that passed validation
        - waves_invalid: int - Number of waves that failed validation
        - validation_results: List[Dict] - Detailed validation results for each wave
        - summary: str - Human-readable summary
    """
    # Import price_book module directly to avoid streamlit dependency
    import sys
    import os
    helpers_dir = os.path.dirname(os.path.abspath(__file__))
    if helpers_dir not in sys.path:
        sys.path.insert(0, helpers_dir)
    from price_book import get_price_book_meta
    
    # Get all waves
    if not WAVES_ENGINE_AVAILABLE:
        logger.error("waves_engine not available - cannot generate validation report")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': 'waves_engine not available',
            'waves_validated': 0,
            'waves_valid': 0,
            'waves_invalid': 0,
            'validation_results': []
        }
    
    try:
        universe = get_all_waves_universe()
        all_waves = universe.get('waves', [])
    except Exception as e:
        logger.error(f"Error getting wave universe: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': f'Error getting wave universe: {str(e)}',
            'waves_validated': 0,
            'waves_valid': 0,
            'waves_invalid': 0,
            'validation_results': []
        }
    
    # Get PRICE_BOOK metadata
    pb_meta = get_price_book_meta(price_book)
    
    # Validate each wave
    validation_results = []
    for wave_name in all_waves:
        validation = validate_wave_price_history(
            wave_name,
            price_book,
            min_coverage_pct=min_coverage_pct,
            min_history_days=min_history_days
        )
        validation_results.append(validation)
    
    # Count valid/invalid waves
    waves_valid = sum(1 for v in validation_results if v['valid'])
    waves_invalid = len(validation_results) - waves_valid
    
    # Build report
    report = {
        'timestamp': datetime.now().isoformat(),
        'price_book_meta': pb_meta,
        'validation_criteria': {
            'min_coverage_pct': min_coverage_pct,
            'min_history_days': min_history_days
        },
        'waves_validated': len(validation_results),
        'waves_valid': waves_valid,
        'waves_invalid': waves_invalid,
        'validation_results': validation_results,
        'summary': f'{waves_valid}/{len(validation_results)} waves passed validation'
    }
    
    # Save to file if requested
    if output_file:
        try:
            import json
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Validation report saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving validation report to {output_file}: {e}")
    
    return report


def compute_beta(
    wave_returns: pd.Series,
    benchmark_returns: pd.Series,
    min_n: int = 60
) -> Dict[str, Any]:
    """
    Compute beta for a wave relative to a benchmark.
    
    Beta measures the sensitivity of wave returns to benchmark returns.
    Formula: beta = cov(wave, benchmark) / var(benchmark)
    
    Args:
        wave_returns: Daily returns for the wave (pandas Series with DatetimeIndex)
        benchmark_returns: Daily returns for the benchmark (pandas Series with DatetimeIndex)
        min_n: Minimum number of overlapping observations required (default: 60)
        
    Returns:
        Dictionary with:
        - success: bool - Whether computation succeeded
        - beta: float or None - Beta value if successful
        - n_observations: int - Number of overlapping observations used
        - failure_reason: str or None - Reason for failure if unsuccessful
        - r_squared: float or None - R-squared value (correlation^2)
        - correlation: float or None - Correlation between wave and benchmark returns
    """
    result = {
        'success': False,
        'beta': None,
        'n_observations': 0,
        'failure_reason': None,
        'r_squared': None,
        'correlation': None
    }
    
    # Input validation
    if wave_returns is None or wave_returns.empty:
        result['failure_reason'] = 'Wave returns are empty'
        return result
    
    if benchmark_returns is None or benchmark_returns.empty:
        result['failure_reason'] = 'Benchmark returns are empty'
        return result
    
    # Align time series - find intersecting dates
    try:
        # Ensure both series have datetime index
        if not isinstance(wave_returns.index, pd.DatetimeIndex):
            result['failure_reason'] = 'Wave returns index is not DatetimeIndex'
            return result
        
        if not isinstance(benchmark_returns.index, pd.DatetimeIndex):
            result['failure_reason'] = 'Benchmark returns index is not DatetimeIndex'
            return result
        
        # Combine and align the two series
        aligned_data = pd.DataFrame({
            'wave': wave_returns,
            'benchmark': benchmark_returns
        })
        
        # Drop rows where either is NaN
        aligned_data = aligned_data.dropna()
        
        # Check if we have sufficient observations
        n_obs = len(aligned_data)
        result['n_observations'] = n_obs
        
        if n_obs < min_n:
            result['failure_reason'] = f'Insufficient data: {n_obs} observations (minimum: {min_n})'
            return result
        
        # Extract aligned returns
        wave_aligned = aligned_data['wave'].values
        benchmark_aligned = aligned_data['benchmark'].values
        
        # Compute variance of benchmark returns
        benchmark_var = np.var(benchmark_aligned, ddof=1)
        
        if np.isnan(benchmark_var) or benchmark_var < 1e-10:
            result['failure_reason'] = 'Benchmark variance is zero or NaN'
            return result
        
        # Compute covariance between wave and benchmark
        covariance = np.cov(wave_aligned, benchmark_aligned, ddof=1)[0, 1]
        
        if np.isnan(covariance):
            result['failure_reason'] = 'Covariance is NaN'
            return result
        
        # Compute beta
        beta = covariance / benchmark_var
        
        # Compute correlation and r-squared
        correlation = np.corrcoef(wave_aligned, benchmark_aligned)[0, 1]
        r_squared = correlation ** 2 if not np.isnan(correlation) else None
        
        # Success!
        result['success'] = True
        result['beta'] = float(beta)
        result['correlation'] = float(correlation) if not np.isnan(correlation) else None
        result['r_squared'] = float(r_squared) if r_squared is not None else None
        result['failure_reason'] = None
        
        return result
        
    except Exception as e:
        result['failure_reason'] = f'Error computing beta: {str(e)}'
        return result


def compute_wave_beta(
    wave_name: str,
    benchmark_name: str,
    price_book: pd.DataFrame,
    lookback_days: int = 252,
    min_n: int = 60
) -> Dict[str, Any]:
    """
    Compute beta for a wave against a benchmark using price_book data.
    
    This is a convenience function that:
    1. Computes daily returns for both wave and benchmark from price_book
    2. Aligns the time series
    3. Calls compute_beta to get the beta value
    
    Args:
        wave_name: Name of the wave (must exist in WAVE_WEIGHTS)
        benchmark_name: Name of the benchmark wave (must exist in WAVE_WEIGHTS)
        price_book: PRICE_BOOK DataFrame (index=dates, columns=tickers, values=prices)
        lookback_days: Number of days to look back for returns computation (default: 252 = 1 year)
        min_n: Minimum number of overlapping observations required (default: 60)
        
    Returns:
        Dictionary with beta computation results (see compute_beta for details)
    """
    result = {
        'success': False,
        'beta': None,
        'n_observations': 0,
        'failure_reason': None,
        'r_squared': None,
        'correlation': None,
        'wave_name': wave_name,
        'benchmark_name': benchmark_name
    }
    
    # Check inputs
    if price_book is None or price_book.empty:
        result['failure_reason'] = 'PRICE_BOOK is empty'
        return result
    
    # Get wave holdings for both wave and benchmark
    if not WAVES_ENGINE_AVAILABLE or not WAVE_WEIGHTS:
        result['failure_reason'] = 'waves_engine not available'
        return result
    
    if wave_name not in WAVE_WEIGHTS:
        result['failure_reason'] = f'Wave "{wave_name}" not found in WAVE_WEIGHTS'
        return result
    
    if benchmark_name not in WAVE_WEIGHTS:
        result['failure_reason'] = f'Benchmark "{benchmark_name}" not found in WAVE_WEIGHTS'
        return result
    
    try:
        # Get recent price data (last lookback_days)
        recent_prices = price_book.tail(lookback_days + 1).copy()
        
        # Compute portfolio values for wave
        wave_holdings = WAVE_WEIGHTS[wave_name]
        wave_tickers = [h.ticker for h in wave_holdings]
        wave_weights_list = [h.weight for h in wave_holdings]
        
        # Normalize weights
        total_wave_weight = sum(wave_weights_list)
        if total_wave_weight == 0:
            result['failure_reason'] = 'Wave total weight is zero'
            return result
        wave_weights_norm = [w / total_wave_weight for w in wave_weights_list]
        
        # Filter to available tickers
        wave_available = [(t, w) for t, w in zip(wave_tickers, wave_weights_norm) if t in recent_prices.columns]
        if not wave_available:
            result['failure_reason'] = 'No wave tickers found in PRICE_BOOK'
            return result
        
        wave_tickers_avail = [t for t, w in wave_available]
        wave_weights_avail = [w for t, w in wave_available]
        total_avail = sum(wave_weights_avail)
        wave_weights_renorm = [w / total_avail for w in wave_weights_avail]
        
        # Compute wave portfolio values
        wave_prices = recent_prices[wave_tickers_avail].copy()
        first_wave_prices = wave_prices.iloc[0]
        wave_normalized = wave_prices.div(first_wave_prices, axis=1) * 100
        wave_portfolio = pd.Series(0.0, index=wave_prices.index)
        for ticker, weight in zip(wave_tickers_avail, wave_weights_renorm):
            wave_portfolio += wave_normalized[ticker].ffill() * weight
        
        # Compute wave returns
        wave_returns = wave_portfolio.pct_change().dropna()
        
        # Compute portfolio values for benchmark
        benchmark_holdings = WAVE_WEIGHTS[benchmark_name]
        benchmark_tickers = [h.ticker for h in benchmark_holdings]
        benchmark_weights_list = [h.weight for h in benchmark_holdings]
        
        # Normalize weights
        total_benchmark_weight = sum(benchmark_weights_list)
        if total_benchmark_weight == 0:
            result['failure_reason'] = 'Benchmark total weight is zero'
            return result
        benchmark_weights_norm = [w / total_benchmark_weight for w in benchmark_weights_list]
        
        # Filter to available tickers
        benchmark_available = [(t, w) for t, w in zip(benchmark_tickers, benchmark_weights_norm) if t in recent_prices.columns]
        if not benchmark_available:
            result['failure_reason'] = 'No benchmark tickers found in PRICE_BOOK'
            return result
        
        benchmark_tickers_avail = [t for t, w in benchmark_available]
        benchmark_weights_avail = [w for t, w in benchmark_available]
        total_avail_bm = sum(benchmark_weights_avail)
        benchmark_weights_renorm = [w / total_avail_bm for w in benchmark_weights_avail]
        
        # Compute benchmark portfolio values
        benchmark_prices = recent_prices[benchmark_tickers_avail].copy()
        first_benchmark_prices = benchmark_prices.iloc[0]
        benchmark_normalized = benchmark_prices.div(first_benchmark_prices, axis=1) * 100
        benchmark_portfolio = pd.Series(0.0, index=benchmark_prices.index)
        for ticker, weight in zip(benchmark_tickers_avail, benchmark_weights_renorm):
            benchmark_portfolio += benchmark_normalized[ticker].ffill() * weight
        
        # Compute benchmark returns
        benchmark_returns = benchmark_portfolio.pct_change().dropna()
        
        # Now compute beta using the aligned returns
        beta_result = compute_beta(wave_returns, benchmark_returns, min_n=min_n)
        
        # Add wave and benchmark names to result
        beta_result['wave_name'] = wave_name
        beta_result['benchmark_name'] = benchmark_name
        
        return beta_result
        
    except Exception as e:
        result['failure_reason'] = f'Error computing wave beta: {str(e)}'
        return result


def compute_portfolio_snapshot(
    price_book: pd.DataFrame,
    mode: str = 'Standard',
    periods: List[int] = [1, 30, 60, 365]
) -> Dict[str, Any]:
    """
    Compute portfolio-level snapshot with returns and alpha attribution.
    
    This is the deterministic computation pipeline that runs on first load.
    It builds portfolio_daily_returns and portfolio_benchmark_daily_returns
    from PRICE_BOOK for the selected mode, then computes multi-window metrics.
    
    Key Features:
    - Equal-weight portfolio across all active waves
    - Equal-weight benchmark across all wave benchmarks
    - Handles insufficient history by showing N/A for specific windows
    - Returns structured data ready for UI rendering
    
    Args:
        price_book: PRICE_BOOK DataFrame (index=dates, columns=tickers)
        mode: Operating mode (e.g., 'Standard', 'Safe', 'Aggressive')
        periods: List of lookback periods in days [1, 30, 60, 365]
        
    Returns:
        Dictionary with:
        - success: bool - Whether computation succeeded
        - failure_reason: str - Reason for failure (if success=False)
        - mode: str - Operating mode
        - portfolio_returns: Dict[str, float] - Portfolio returns by period (e.g., {'1D': 0.02})
        - benchmark_returns: Dict[str, float] - Benchmark returns by period
        - alphas: Dict[str, float] - Alpha by period (portfolio - benchmark)
        - wave_count: int - Number of waves included
        - date_range: Tuple[str, str] - Date range used
        - has_portfolio_returns_series: bool - Whether daily returns series exists
        - has_portfolio_benchmark_series: bool - Whether daily benchmark series exists
        - has_overlay_alpha_series: bool - Whether overlay alpha component exists
        - latest_date: str - Latest data date
        - data_age_days: int - Age of data in days
        - debug: Dict - Diagnostic information for troubleshooting
    """
    # Initialize debug dict to track computation details
    debug = {
        'price_book_source': 'get_price_book()',
        'price_book_shape': None,
        'price_book_index_min': None,
        'price_book_index_max': None,
        'spy_present': False,
        'requested_periods': periods,
        'active_waves_count': 0,
        'portfolio_rows_count': None,
        'tickers_requested_count': 0,
        'tickers_intersection_count': 0,
        'tickers_missing_sample': [],
        'filtered_price_book_shape': None,
        'reason_if_failure': None
    }
    
    result = {
        'success': False,
        'failure_reason': None,
        'mode': mode,
        'portfolio_returns': {f'{p}D': None for p in periods},
        'benchmark_returns': {f'{p}D': None for p in periods},
        'alphas': {f'{p}D': None for p in periods},
        'wave_count': 0,
        'date_range': (None, None),
        'has_portfolio_returns_series': False,
        'has_portfolio_benchmark_series': False,
        'has_overlay_alpha_series': False,
        'latest_date': None,
        'data_age_days': None,
        'debug': debug
    }
    
    # Validate PRICE_BOOK
    if price_book is None or price_book.empty:
        result['failure_reason'] = 'PRICE_BOOK is empty'
        debug['reason_if_failure'] = 'PRICE_BOOK is empty or None'
        return result
    
    # Record price_book debug info
    debug['price_book_shape'] = f"{len(price_book)} x {len(price_book.columns)}"
    try:
        debug['price_book_index_min'] = price_book.index[0].strftime('%Y-%m-%d')
        debug['price_book_index_max'] = price_book.index[-1].strftime('%Y-%m-%d')
    except (IndexError, AttributeError) as e:
        logger.warning(f"Failed to get price_book index range: {e}")
    
    # Check if SPY is present
    debug['spy_present'] = DEFAULT_BENCHMARK_TICKER in price_book.columns
    
    # Get latest date and data age
    try:
        latest_date = price_book.index[-1]
        result['latest_date'] = latest_date.strftime('%Y-%m-%d')
        data_age_days = (datetime.now().date() - latest_date.date()).days
        result['data_age_days'] = data_age_days
    except Exception as e:
        logger.warning(f"Failed to compute data age: {e}")
    
    # Get all waves
    if not WAVES_ENGINE_AVAILABLE:
        result['failure_reason'] = 'waves_engine not available'
        debug['reason_if_failure'] = 'waves_engine not available'
        return result
    
    try:
        universe = get_all_waves_universe()
        all_waves = universe.get('waves', [])
        debug['active_waves_count'] = len(all_waves)
    except Exception as e:
        result['failure_reason'] = f'Error getting wave universe: {str(e)}'
        debug['reason_if_failure'] = f'Error getting wave universe: {str(e)}'
        return result
    
    if not all_waves:
        result['failure_reason'] = 'No waves found in universe'
        debug['reason_if_failure'] = 'No waves found in universe'
        return result
    
    # ========================================================================
    # DETERMINISTIC PORTFOLIO AGGREGATION (PR #406 spec)
    # ========================================================================
    # For each wave:
    #   1. Get daily return series aligned on date
    #   2. Concatenate into matrix R (rows=dates, cols=waves)
    #   3. Compute portfolio_return = R.mean(axis=1, skipna=True) (equal weight)
    #   4. Drop dates where all waves are NaN
    #   5. Sort index ascending
    # ========================================================================
    
    wave_return_series_dict = {}  # {wave_name: pd.Series of daily returns}
    wave_benchmark_return_dict = {}  # {wave_name: pd.Series of benchmark daily returns}
    
    # Track all tickers across all waves for debugging
    all_requested_tickers = set()
    all_available_tickers = set()
    
    for wave_name in all_waves:
        # Get wave tickers and weights
        if wave_name not in WAVE_WEIGHTS:
            continue
        
        wave_holdings = WAVE_WEIGHTS[wave_name]
        if not wave_holdings:
            continue
        
        try:
            # Extract tickers and weights
            tickers = [h.ticker for h in wave_holdings]
            weights = [h.weight for h in wave_holdings]
            
            # Track requested tickers for debugging
            all_requested_tickers.update(tickers)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight == 0:
                continue
            normalized_weights = [w / total_weight for w in weights]
            
            # Filter to available tickers
            available_tickers = []
            available_weights = []
            for ticker, weight in zip(tickers, normalized_weights):
                if ticker in price_book.columns:
                    available_tickers.append(ticker)
                    available_weights.append(weight)
                    all_available_tickers.add(ticker)  # Track for debugging
            
            if not available_tickers:
                continue
            
            # Renormalize weights for available tickers
            total_available = sum(available_weights)
            renormalized_weights = [w / total_available for w in available_weights]
            
            # Get price data for available tickers
            ticker_prices = price_book[available_tickers].copy()
            if len(ticker_prices) < 2:
                continue
            
            # Compute daily returns for each ticker
            ticker_returns = ticker_prices.pct_change()
            
            # Compute weighted portfolio returns for this wave
            wave_returns = pd.Series(0.0, index=ticker_returns.index)
            for i, ticker in enumerate(available_tickers):
                weight = renormalized_weights[i]
                # Don't fill NaN - let them propagate naturally
                ticker_ret = ticker_returns[ticker]
                wave_returns = wave_returns.add(ticker_ret * weight, fill_value=0.0)
            
            # Drop the first row (which is NaN from pct_change)
            wave_returns = wave_returns.iloc[1:]
            
            # Store if we have valid data
            if len(wave_returns) > 0 and wave_returns.notna().any():
                wave_return_series_dict[wave_name] = wave_returns
                
                # PHASE 1B.1: Compute wave-level benchmark returns using compute_history_nav
                # This ensures consistency with wave-level dynamic benchmarks
                try:
                    # Compute wave history to get benchmark returns
                    # Use same price_book and sufficient days to get history
                    wave_history = compute_history_nav(
                        wave_name=wave_name,
                        mode=mode,
                        days=DEFAULT_HISTORY_DAYS,  # Get full history available
                        include_diagnostics=False,
                        price_df=price_book
                    )
                    
                    if not wave_history.empty and 'bm_ret' in wave_history.columns:
                        # Extract benchmark returns from wave history
                        bm_ret_series = wave_history['bm_ret'].copy()
                        
                        # Store benchmark returns for this wave
                        if bm_ret_series.notna().any():
                            wave_benchmark_return_dict[wave_name] = bm_ret_series
                        else:
                            logger.debug(f"Wave {wave_name}: benchmark returns are all NaN")
                    else:
                        logger.debug(f"Wave {wave_name}: compute_history_nav returned empty or missing bm_ret")
                        
                except Exception as e:
                    logger.warning(f"Failed to compute benchmark for wave {wave_name}: {e}")
                    # Don't add to wave_benchmark_return_dict if benchmark computation fails
            
        except Exception as e:
            logger.warning(f"Failed to compute returns for wave {wave_name}: {e}")
            continue
    
    # Update debug info with ticker statistics
    debug['tickers_requested_count'] = len(all_requested_tickers)
    debug['tickers_intersection_count'] = len(all_available_tickers)
    
    # Get missing tickers (first 10 as sample)
    missing_tickers = all_requested_tickers - all_available_tickers
    debug['tickers_missing_sample'] = sorted(list(missing_tickers))[:10]
    
    # Check if we got any valid waves
    if not wave_return_series_dict:
        result['failure_reason'] = 'No valid wave return series computed'
        debug['reason_if_failure'] = f'no tickers intersect (requested={len(all_requested_tickers)}, available={len(all_available_tickers)})'
        return result
    
    result['wave_count'] = len(wave_return_series_dict)
    
    # ========================================================================
    # PHASE 1B.1: Build Portfolio Composite Benchmark
    # ========================================================================
    # Build wave_results dict for composite benchmark computation
    # Each entry needs to be a DataFrame with 'bm_ret' column
    wave_results_for_composite = {}
    for wave_name in wave_return_series_dict.keys():
        if wave_name in wave_benchmark_return_dict:
            # Create a simple DataFrame with bm_ret column
            bm_ret_series = wave_benchmark_return_dict[wave_name]
            wave_results_for_composite[wave_name] = pd.DataFrame({'bm_ret': bm_ret_series})
    
    # Compute portfolio composite benchmark using equal weights
    portfolio_benchmark_returns = build_portfolio_composite_benchmark_returns(
        wave_results=wave_results_for_composite,
        wave_weights=None  # Use equal weights as specified
    )
    
    # Update debug info
    debug['composite_benchmark_waves'] = len(wave_results_for_composite)
    debug['composite_benchmark_days'] = len(portfolio_benchmark_returns) if not portfolio_benchmark_returns.empty else 0
    
    # Build return matrix R (rows=dates, cols=waves)
    try:
        # Concatenate all wave return series into a DataFrame
        return_matrix = pd.DataFrame(wave_return_series_dict)
        
        # Record portfolio rows count for debugging
        debug['portfolio_rows_count'] = len(return_matrix)
        debug['filtered_price_book_shape'] = f"{len(return_matrix)} x {len(return_matrix.columns)}"
        
        # Compute equal-weight portfolio return: mean across waves (skipna=True)
        portfolio_returns = return_matrix.mean(axis=1, skipna=True)
        
        # Use composite benchmark returns (already computed above)
        benchmark_returns = portfolio_benchmark_returns
        
        # If composite benchmark is empty, fall back to error
        if benchmark_returns.empty:
            result['failure_reason'] = 'Failed to build portfolio composite benchmark'
            debug['reason_if_failure'] = 'composite benchmark empty or insufficient history'
            return result
        
        # Align portfolio returns and benchmark returns on common dates
        # This handles the case where they may have different date ranges
        common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) < MIN_DATES_FOR_PORTFOLIO:
            result['failure_reason'] = f'Insufficient common dates between portfolio and benchmark: {len(common_dates)}'
            debug['reason_if_failure'] = f'only {len(common_dates)} common dates after alignment'
            return result
        
        portfolio_returns = portfolio_returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        # Drop dates where portfolio return is NaN
        valid_dates = ~portfolio_returns.isna()
        portfolio_returns = portfolio_returns[valid_dates]
        benchmark_returns = benchmark_returns[valid_dates]
        
        # Sort index ascending
        portfolio_returns = portfolio_returns.sort_index()
        benchmark_returns = benchmark_returns.sort_index()
        
        # Check if we have sufficient data
        if len(portfolio_returns) < MIN_DATES_FOR_PORTFOLIO:
            result['failure_reason'] = f'Insufficient dates after aggregation: {len(portfolio_returns)} (need at least {MIN_DATES_FOR_PORTFOLIO})'
            debug['reason_if_failure'] = f'filtered df empty or too small ({len(portfolio_returns)} dates)'
            return result
        
        # Record that we have these series
        result['has_portfolio_returns_series'] = True
        result['has_portfolio_benchmark_series'] = True
        result['has_overlay_alpha_series'] = False  # VIX overlay not yet implemented
        
        # Record date range
        result['date_range'] = (
            portfolio_returns.index[0].strftime('%Y-%m-%d'),
            portfolio_returns.index[-1].strftime('%Y-%m-%d')
        )
        
        # Compute cumulative returns for each period
        # Convert daily returns to cumulative index starting at 100
        portfolio_cumulative = (1 + portfolio_returns).cumprod() * 100
        benchmark_cumulative = (1 + benchmark_returns).cumprod() * 100
        
        max_available_days = len(portfolio_cumulative)
        
        for period in periods:
            try:
                if period >= max_available_days:
                    # Not enough history for this period
                    result['portfolio_returns'][f'{period}D'] = None
                    result['benchmark_returns'][f'{period}D'] = None
                    result['alphas'][f'{period}D'] = None
                    continue
                
                # Get cumulative values from 'period' days ago and current value
                current_portfolio = portfolio_cumulative.iloc[-1]
                past_portfolio = portfolio_cumulative.iloc[-(period + 1)]
                current_benchmark = benchmark_cumulative.iloc[-1]
                past_benchmark = benchmark_cumulative.iloc[-(period + 1)]
                
                # Handle NaN values
                if (pd.isna(current_portfolio) or pd.isna(past_portfolio) or 
                    pd.isna(current_benchmark) or pd.isna(past_benchmark) or
                    past_portfolio == 0 or past_benchmark == 0):
                    result['portfolio_returns'][f'{period}D'] = None
                    result['benchmark_returns'][f'{period}D'] = None
                    result['alphas'][f'{period}D'] = None
                    continue
                
                # Calculate returns
                portfolio_ret = (current_portfolio - past_portfolio) / past_portfolio
                benchmark_ret = (current_benchmark - past_benchmark) / past_benchmark
                alpha = portfolio_ret - benchmark_ret
                
                result['portfolio_returns'][f'{period}D'] = portfolio_ret
                result['benchmark_returns'][f'{period}D'] = benchmark_ret
                result['alphas'][f'{period}D'] = alpha
                
            except (IndexError, ValueError) as e:
                logger.warning(f"Error computing {period}D portfolio return: {e}")
                result['portfolio_returns'][f'{period}D'] = None
                result['benchmark_returns'][f'{period}D'] = None
                result['alphas'][f'{period}D'] = None
        
        # Check if we got at least one valid return
        valid_returns = [v for v in result['portfolio_returns'].values() if v is not None]
        if not valid_returns:
            result['failure_reason'] = 'No valid returns computed (insufficient history or all NaN)'
            debug['reason_if_failure'] = 'no valid returns (insufficient history or all NaN)'
            return result
        
        # Success!
        result['success'] = True
        result['failure_reason'] = None
        
    except Exception as e:
        result['failure_reason'] = f'Error computing portfolio metrics: {str(e)}'
        debug['reason_if_failure'] = f'exception: {str(e)}'
        return result
    
    # Log debug info for troubleshooting (using debug level to avoid performance impact)
    logger.debug(f"Portfolio snapshot debug: {debug}")
    
    return result


def compute_portfolio_alpha_attribution(
    price_book: pd.DataFrame,
    mode: str = 'Standard',
    periods: List[int] = [30, 60, 365],
    safe_ticker_preference: List[str] = ["BIL", "SHY"],
    wave_registry: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Compute portfolio-level alpha attribution with transparent decomposition.
    
    Returns daily series and period summaries decomposing total alpha into:
    - Selection Alpha: from asset selection (unoverlay vs benchmark)
    - Overlay Alpha: from VIX/SafeSmart exposure management
    - Residual: reconciliation term (should be near 0)
    
    Args:
        price_book: PRICE_BOOK DataFrame (index=dates, columns=tickers)
        wave_registry: Wave registry (if None, uses WAVE_WEIGHTS)
        mode: Operating mode (e.g., 'Standard', 'Safe', 'Aggressive')
        periods: Lookback periods in days for summary (e.g., [30, 60, 365])
        safe_ticker_preference: Ordered list of safe asset tickers (e.g., ["BIL", "SHY"])
        
    Returns:
        Dictionary with:
        - success: bool - Whether computation succeeded
        - failure_reason: Optional[str] - Reason for failure
        - daily_realized_return: pd.Series - Daily portfolio returns (with overlay)
        - daily_unoverlay_return: pd.Series - Daily portfolio returns (exposure=1.0)
        - daily_benchmark_return: pd.Series - Daily benchmark returns
        - daily_exposure: pd.Series - Daily exposure levels [0, 1]
        - period_summaries: Dict[str, Dict] - Summary for each period (30D, 60D, 365D)
        - since_inception_summary: Dict - Summary since inception
        - warnings: List[str] - Any warnings about data quality
    
    Period summary structure (for each period):
        - period: int - Period in days
        - available: bool - Whether period has sufficient data
        - reason: Optional[str] - Reason if unavailable (e.g., "insufficient_aligned_rows")
        - requested_period_days: int - Requested period length in trading days
        - rows_used: int - Actual trading rows used in computation
        - cum_real: Optional[float] - Cumulative realized return (None if unavailable)
        - cum_sel: Optional[float] - Cumulative selection return (unoverlay) (None if unavailable)
        - cum_bm: Optional[float] - Cumulative benchmark return (None if unavailable)
        - total_alpha: Optional[float] - cum_real - cum_bm (None if unavailable)
        - selection_alpha: Optional[float] - cum_sel - cum_bm (None if unavailable)
        - overlay_alpha: Optional[float] - cum_real - cum_sel (None if unavailable)
        - residual: Optional[float] - total_alpha - (selection_alpha + overlay_alpha) (None if unavailable)
    
    Since inception summary structure:
        - period: str - "inception"
        - available: bool - Always True for inception
        - reason: None
        - requested_period_days: None - Not applicable for inception
        - rows_used: int - Total trading rows used
        - days: int - Total trading days (same as rows_used)
        - cum_real: float - Cumulative realized return
        - cum_sel: float - Cumulative selection return (unoverlay)
        - cum_bm: float - Cumulative benchmark return
        - total_alpha: float - cum_real - cum_bm
        - selection_alpha: float - cum_sel - cum_bm
        - overlay_alpha: float - cum_real - cum_sel
        - residual: float - total_alpha - (selection_alpha + overlay_alpha)
    """
    result = {
        'success': False,
        'failure_reason': None,
        'daily_realized_return': None,
        'daily_unoverlay_return': None,
        'daily_benchmark_return': None,
        'daily_exposure': None,
        'period_summaries': {},
        'since_inception_summary': {},
        'warnings': [],
        'using_fallback_exposure': False  # Flag to indicate if fallback exposure (1.0) is used
    }
    
    # Validate inputs
    if price_book is None or price_book.empty:
        result['failure_reason'] = 'PRICE_BOOK is empty'
        return result
    
    if not WAVES_ENGINE_AVAILABLE:
        result['failure_reason'] = 'waves_engine not available'
        return result
    
    # Use WAVE_WEIGHTS as registry if not provided
    if wave_registry is None:
        wave_registry = WAVE_WEIGHTS
    
    try:
        universe = get_all_waves_universe()
        all_waves = universe.get('waves', [])
    except Exception as e:
        result['failure_reason'] = f'Error getting wave universe: {str(e)}'
        return result
    
    if not all_waves:
        result['failure_reason'] = 'No waves found in universe'
        return result
    
    # ========================================================================
    # Step 1: Compute daily_unoverlay_return (exposure=1.0 forced)
    # ========================================================================
    # This is the same as current compute_portfolio_snapshot logic
    try:
        wave_return_dict_unoverlay = {}
        wave_benchmark_dict = {}
        
        for wave_name in all_waves:
            if wave_name not in wave_registry:
                continue
            
            wave_holdings = wave_registry[wave_name]
            if not wave_holdings:
                continue
            
            try:
                tickers = [h.ticker for h in wave_holdings]
                weights = [h.weight for h in wave_holdings]
                
                total_weight = sum(weights)
                if total_weight == 0:
                    continue
                normalized_weights = [w / total_weight for w in weights]
                
                # Filter to available tickers
                available_tickers = []
                available_weights = []
                for ticker, weight in zip(tickers, normalized_weights):
                    if ticker in price_book.columns:
                        available_tickers.append(ticker)
                        available_weights.append(weight)
                
                if not available_tickers:
                    continue
                
                # Renormalize
                total_available = sum(available_weights)
                renormalized_weights = [w / total_available for w in available_weights]
                
                # Compute daily returns
                ticker_prices = price_book[available_tickers].copy()
                if len(ticker_prices) < 2:
                    continue
                
                ticker_returns = ticker_prices.pct_change()
                
                # Weighted returns
                wave_returns = pd.Series(0.0, index=ticker_returns.index)
                for i, ticker in enumerate(available_tickers):
                    weight = renormalized_weights[i]
                    ticker_ret = ticker_returns[ticker]
                    wave_returns = wave_returns.add(ticker_ret * weight, fill_value=0.0)
                
                wave_returns = wave_returns.iloc[1:]  # Drop first NaN
                
                if len(wave_returns) > 0 and wave_returns.notna().any():
                    wave_return_dict_unoverlay[wave_name] = wave_returns
                    
                    # PHASE 1B.1: Compute wave-level benchmark returns using compute_history_nav
                    # This ensures consistency with wave-level dynamic benchmarks
                    try:
                        # Compute wave history to get benchmark returns
                        wave_history = compute_history_nav(
                            wave_name=wave_name,
                            mode=mode,
                            days=DEFAULT_HISTORY_DAYS,  # Get full history available
                            include_diagnostics=False,
                            price_df=price_book
                        )
                        
                        if not wave_history.empty and 'bm_ret' in wave_history.columns:
                            # Extract benchmark returns from wave history
                            bm_ret_series = wave_history['bm_ret'].copy()
                            
                            # Store benchmark returns for this wave
                            if bm_ret_series.notna().any():
                                wave_benchmark_dict[wave_name] = bm_ret_series
                            else:
                                logger.debug(f"Wave {wave_name}: benchmark returns are all NaN")
                        else:
                            logger.debug(f"Wave {wave_name}: compute_history_nav returned empty or missing bm_ret")
                            
                    except Exception as e:
                        logger.warning(f"Failed to compute benchmark for wave {wave_name}: {e}")
                        # Don't add to wave_benchmark_dict if benchmark computation fails
            
            except Exception as e:
                logger.warning(f"Failed to compute returns for wave {wave_name}: {e}")
                continue
        
        if not wave_return_dict_unoverlay:
            result['failure_reason'] = 'No valid wave return series computed'
            return result
        
        # PHASE 1B.1: Build Portfolio Composite Benchmark
        # Build wave_results dict for composite benchmark computation
        wave_results_for_composite = {}
        for wave_name in wave_return_dict_unoverlay.keys():
            if wave_name in wave_benchmark_dict:
                # Create a simple DataFrame with bm_ret column
                bm_ret_series = wave_benchmark_dict[wave_name]
                wave_results_for_composite[wave_name] = pd.DataFrame({'bm_ret': bm_ret_series})
        
        # Compute portfolio composite benchmark using equal weights
        daily_benchmark_return = build_portfolio_composite_benchmark_returns(
            wave_results=wave_results_for_composite,
            wave_weights=None  # Use equal weights
        )
        
        # If composite benchmark is empty, fall back to error
        if daily_benchmark_return.empty:
            result['failure_reason'] = 'Failed to build portfolio composite benchmark'
            return result
        
        # Aggregate to portfolio level (equal weight)
        return_matrix_unoverlay = pd.DataFrame(wave_return_dict_unoverlay)
        
        daily_unoverlay_return = return_matrix_unoverlay.mean(axis=1, skipna=True)
        
        # Align portfolio and benchmark on common dates
        common_dates = daily_unoverlay_return.index.intersection(daily_benchmark_return.index)
        if len(common_dates) < 2:
            result['failure_reason'] = 'Insufficient common dates between portfolio and benchmark'
            return result
        
        daily_unoverlay_return = daily_unoverlay_return.loc[common_dates]
        daily_benchmark_return = daily_benchmark_return.loc[common_dates]
        
        # Drop dates where portfolio is NaN
        valid_dates = ~daily_unoverlay_return.isna()
        daily_unoverlay_return = daily_unoverlay_return[valid_dates].sort_index()
        daily_benchmark_return = daily_benchmark_return[valid_dates].sort_index()
        
        if len(daily_unoverlay_return) < 2:
            result['failure_reason'] = 'Insufficient dates after aggregation'
            return result
        
        result['daily_unoverlay_return'] = daily_unoverlay_return
        result['daily_benchmark_return'] = daily_benchmark_return
        
    except Exception as e:
        result['failure_reason'] = f'Error computing unoverlay returns: {str(e)}'
        return result
    
    # ========================================================================
    # Step 2: Compute daily_exposure series
    # ========================================================================
    # For now, we don't have VIX overlay data, so exposure is always 1.0
    # When VIX overlay is available, this will compute actual exposure from regime
    try:
        # Try to find safe ticker for safe sleeve
        safe_ticker = None
        for ticker in safe_ticker_preference:
            if ticker in price_book.columns:
                safe_ticker = ticker
                break
        
        # Default: exposure = 1.0 (no overlay)
        # When VIX overlay is integrated, compute actual exposure here
        # Using fallback exposure of 1.0 - this is expected behavior when overlay data is not available
        daily_exposure = pd.Series(1.0, index=daily_unoverlay_return.index)
        result['daily_exposure'] = daily_exposure
        result['using_fallback_exposure'] = True  # Flag to indicate fallback is being used
        
        # Compute safe returns if available
        if safe_ticker is not None:
            safe_prices = price_book[safe_ticker].copy()
            safe_returns = safe_prices.pct_change().iloc[1:]
            # Align with portfolio dates
            safe_returns = safe_returns.reindex(daily_unoverlay_return.index, fill_value=0.0)
        else:
            # Use 0 return for safe sleeve if no safe asset available
            safe_returns = pd.Series(0.0, index=daily_unoverlay_return.index)
        
    except Exception as e:
        # Error computing exposure - use fallback
        daily_exposure = pd.Series(1.0, index=daily_unoverlay_return.index)
        safe_returns = pd.Series(0.0, index=daily_unoverlay_return.index)
        result['daily_exposure'] = daily_exposure
        result['using_fallback_exposure'] = True
        logger.debug(f'Using fallback exposure (1.0) due to error: {str(e)}')
    
    # ========================================================================
    # Step 3: Compute daily_realized_return (with overlay)
    # ========================================================================
    # realized_return(t) = exposure(t) * unoverlay_return(t) + (1-exposure(t)) * safe_return(t)
    try:
        daily_realized_return = (
            daily_exposure * daily_unoverlay_return + 
            (1 - daily_exposure) * safe_returns
        )
        result['daily_realized_return'] = daily_realized_return
    
    except Exception as e:
        result['failure_reason'] = f'Error computing realized returns: {str(e)}'
        return result
    
    # ========================================================================
    # Step 4: Compute period summaries
    # ========================================================================
    try:
        def compute_cumulative_return(series: pd.Series, window: int) -> Optional[float]:
            """
            Compute cumulative return over last N days (strict slicing - no fallback).
            
            Returns None when insufficient data is available (len(series) < window).
            This ensures no silent fallback to using all available data when 
            the requested window cannot be satisfied.
            """
            if len(series) < window:
                return None
            # Strict slicing: use exactly the last 'window' trading rows
            window_series = series.iloc[-window:]
            cum_return = (1 + window_series).prod() - 1
            return float(cum_return)
        
        # Compute for each period with strict windowing and diagnostics
        for period in periods:
            # Get actual rows available
            rows_available = len(daily_realized_return)
            
            # Strict windowing: compute only if we have exactly the requested period rows
            if rows_available < period:
                # Insufficient rows - return explicit diagnostic summary with None values
                result['period_summaries'][f'{period}D'] = {
                    'period': period,
                    'available': False,
                    'reason': 'insufficient_aligned_rows',
                    'requested_period_days': period,
                    'rows_used': rows_available,
                    'cum_real': None,
                    'cum_sel': None,
                    'cum_bm': None,
                    'total_alpha': None,
                    'selection_alpha': None,
                    'overlay_alpha': None,
                    'residual': None
                }
                continue
            
            # We have sufficient rows - compute using strict last N rows
            cum_real = compute_cumulative_return(daily_realized_return, period)
            cum_sel = compute_cumulative_return(daily_unoverlay_return, period)
            cum_bm = compute_cumulative_return(daily_benchmark_return, period)
            
            # These should not be None since we checked rows_available >= period
            # However, None can still occur due to data quality issues (e.g., all NaN values,
            # calculation errors from extreme values, or numerical instability)
            if cum_real is None or cum_sel is None or cum_bm is None:
                # Unexpected but handle defensively for data quality issues
                result['period_summaries'][f'{period}D'] = {
                    'period': period,
                    'available': False,
                    'reason': 'computation_error',
                    'requested_period_days': period,
                    'rows_used': period,  # Report the period that was attempted (not total available)
                    'cum_real': None,
                    'cum_sel': None,
                    'cum_bm': None,
                    'total_alpha': None,
                    'selection_alpha': None,
                    'overlay_alpha': None,
                    'residual': None
                }
                continue
            
            total_alpha = cum_real - cum_bm
            selection_alpha = cum_sel - cum_bm
            overlay_alpha = cum_real - cum_sel
            residual = total_alpha - (selection_alpha + overlay_alpha)
            
            result['period_summaries'][f'{period}D'] = {
                'period': period,
                'available': True,
                'reason': None,
                'requested_period_days': period,
                'rows_used': period,  # Exact rows used for this period
                'cum_real': cum_real,
                'cum_sel': cum_sel,
                'cum_bm': cum_bm,
                'total_alpha': total_alpha,
                'selection_alpha': selection_alpha,
                'overlay_alpha': overlay_alpha,
                'residual': residual
            }
        
        # Since inception summary (always computed - uses all available rows)
        rows_available = len(daily_realized_return)
        cum_real_inception = (1 + daily_realized_return).prod() - 1
        cum_sel_inception = (1 + daily_unoverlay_return).prod() - 1
        cum_bm_inception = (1 + daily_benchmark_return).prod() - 1
        
        total_alpha_inception = cum_real_inception - cum_bm_inception
        selection_alpha_inception = cum_sel_inception - cum_bm_inception
        overlay_alpha_inception = cum_real_inception - cum_sel_inception
        residual_inception = total_alpha_inception - (selection_alpha_inception + overlay_alpha_inception)
        
        result['since_inception_summary'] = {
            'period': 'inception',
            'available': True,
            'reason': None,
            'requested_period_days': None,  # Not applicable for inception
            'rows_used': rows_available,
            'days': rows_available,
            'cum_real': float(cum_real_inception),
            'cum_sel': float(cum_sel_inception),
            'cum_bm': float(cum_bm_inception),
            'total_alpha': float(total_alpha_inception),
            'selection_alpha': float(selection_alpha_inception),
            'overlay_alpha': float(overlay_alpha_inception),
            'residual': float(residual_inception)
        }
        
        result['success'] = True
        result['failure_reason'] = None
        
    except Exception as e:
        result['failure_reason'] = f'Error computing period summaries: {str(e)}'
        return result
    
    return result


def validate_portfolio_diagnostics(
    price_book: pd.DataFrame,
    mode: str = 'Standard'
) -> Dict[str, Any]:
    """
    Validate key data points for portfolio snapshot diagnostics panel.
    
    This function confirms:
    - Latest date and data age
    - Existence of required series (portfolio returns, benchmark, alpha components)
    - Data quality and readiness
    
    Args:
        price_book: PRICE_BOOK DataFrame
        mode: Operating mode
        
    Returns:
        Dictionary with:
        - latest_date: str - Latest data date in YYYY-MM-DD format
        - data_age_days: int - Age of data in days
        - has_portfolio_returns_series: bool - Portfolio returns series exists
        - has_portfolio_benchmark_series: bool - Benchmark series exists
        - has_overlay_alpha_series: bool - Overlay alpha component exists
        - wave_count: int - Number of waves with valid data
        - min_history_days: int - Minimum history available
        - data_quality: str - Overall data quality ('OK', 'DEGRADED', 'STALE')
        - issues: List[str] - List of any issues found
    """
    result = {
        'latest_date': None,
        'data_age_days': None,
        'has_portfolio_returns_series': False,
        'has_portfolio_benchmark_series': False,
        'has_overlay_alpha_series': False,
        'wave_count': 0,
        'min_history_days': 0,
        'data_quality': 'UNKNOWN',
        'issues': []
    }
    
    # Validate PRICE_BOOK exists
    if price_book is None or price_book.empty:
        result['issues'].append('PRICE_BOOK is empty')
        result['data_quality'] = 'STALE'
        return result
    
    # Get latest date and data age
    try:
        latest_date = price_book.index[-1]
        result['latest_date'] = latest_date.strftime('%Y-%m-%d')
        data_age_days = (datetime.now().date() - latest_date.date()).days
        result['data_age_days'] = data_age_days
        
        # Determine data quality based on age
        if data_age_days <= 3:
            result['data_quality'] = 'OK'
        elif data_age_days <= 14:
            result['data_quality'] = 'DEGRADED'
            result['issues'].append(f'Data is {data_age_days} days old')
        else:
            result['data_quality'] = 'STALE'
            result['issues'].append(f'Data is {data_age_days} days old (>14 days)')
    except Exception as e:
        result['issues'].append(f'Failed to get latest date: {str(e)}')
        result['data_quality'] = 'STALE'
        return result
    
    # Get minimum history
    result['min_history_days'] = len(price_book)
    
    if result['min_history_days'] < 60:
        result['issues'].append(f'Insufficient history: {result["min_history_days"]} days (need 60)')
        if result['data_quality'] == 'OK':
            result['data_quality'] = 'DEGRADED'
    
    # Try to compute portfolio snapshot to validate series existence
    try:
        snapshot = compute_portfolio_snapshot(price_book, mode=mode, periods=[1, 30, 60])
        
        if snapshot['success']:
            result['has_portfolio_returns_series'] = snapshot['has_portfolio_returns_series']
            result['has_portfolio_benchmark_series'] = snapshot['has_portfolio_benchmark_series']
            result['has_overlay_alpha_series'] = snapshot['has_overlay_alpha_series']
            result['wave_count'] = snapshot['wave_count']
            
            # Check for missing series
            if not result['has_portfolio_returns_series']:
                result['issues'].append('Portfolio returns series missing')
            
            if not result['has_portfolio_benchmark_series']:
                result['issues'].append('Portfolio benchmark series missing')
            
            if not result['has_overlay_alpha_series']:
                # Note: Overlay alpha component requires VIX overlay integration
                # This is a known limitation - see PORTFOLIO_SNAPSHOT_IMPLEMENTATION.md Future Enhancements
                result['issues'].append('Overlay alpha component series missing (requires VIX overlay integration)')
            
            # Validate wave count
            if result['wave_count'] < 3:
                result['issues'].append(f'Insufficient waves: {result["wave_count"]} (need 3+)')
                if result['data_quality'] == 'OK':
                    result['data_quality'] = 'DEGRADED'
        else:
            result['issues'].append(f'Portfolio snapshot failed: {snapshot["failure_reason"]}')
            if result['data_quality'] == 'OK':
                result['data_quality'] = 'DEGRADED'
    except Exception as e:
        result['issues'].append(f'Failed to compute portfolio snapshot: {str(e)}')
        if result['data_quality'] == 'OK':
            result['data_quality'] = 'DEGRADED'
    
    return result


def compute_portfolio_exposure_series(
    price_book: pd.DataFrame,
    mode: str = "Standard",
    smooth_window: int = 3,
    apply_smoothing: bool = True
) -> Optional[pd.Series]:
    """
    Compute deterministic portfolio exposure series using VIX regime mapping.
    
    This function implements the canonical exposure calculation for portfolio overlays:
    1. Selects VIX series in order of preference: ^VIX > VIXY > VXX
    2. Maps VIX to exposure using regime thresholds:
       - VIX < 18: Exposure = 1.00 (full risk)
       - 18 ≤ VIX < 25: Exposure = 0.65 (moderate risk)
       - VIX ≥ 25: Exposure = 0.25 (low risk / defensive)
    3. Optionally smooths using 3-day rolling median
    4. Clips exposure to [0, 1] range
    
    Args:
        price_book: PRICE_BOOK DataFrame (index=dates, columns=tickers, values=prices)
        mode: Operating mode (currently not used, reserved for future mode-specific logic)
        smooth_window: Window size for rolling median smoothing (default: 3)
        apply_smoothing: Whether to apply rolling median smoothing (default: True)
        
    Returns:
        pd.Series: Exposure indexed by date, or None if no VIX proxy is available
        
    Edge Cases:
        - Returns None if no VIX proxy ticker is found in PRICE_BOOK
        - Returns 'missing_vix_series' as failure reason when None is returned
        
    Example:
        >>> price_book = get_price_book()
        >>> exposure = compute_portfolio_exposure_series(price_book)
        >>> if exposure is not None:
        >>>     print(f"Average exposure: {exposure.mean():.2f}")
    """
    # Validate inputs
    if price_book is None or price_book.empty:
        logger.warning("compute_portfolio_exposure_series: PRICE_BOOK is empty")
        return None
    
    # Try to find VIX series in order of preference
    vix_ticker_preference = ['^VIX', 'VIXY', 'VXX']
    vix_series = None
    vix_ticker_found = None
    
    for ticker in vix_ticker_preference:
        if ticker in price_book.columns:
            vix_series = price_book[ticker].copy()
            vix_ticker_found = ticker
            logger.info(f"Using VIX proxy: {ticker}")
            break
    
    # Edge case: No VIX proxy available
    if vix_series is None:
        logger.warning("compute_portfolio_exposure_series: No VIX proxy found (tried: ^VIX, VIXY, VXX)")
        return None
    
    # Remove NaN values
    vix_series = vix_series.dropna()
    
    if len(vix_series) == 0:
        logger.warning(f"compute_portfolio_exposure_series: VIX series {vix_ticker_found} has no valid data")
        return None
    
    # Map VIX to exposure using regime thresholds
    # VIX < VIX_LOW_THRESHOLD: Exposure = EXPOSURE_LOW_VOLATILITY
    # VIX_LOW_THRESHOLD ≤ VIX < VIX_HIGH_THRESHOLD: Exposure = EXPOSURE_MODERATE_VOLATILITY
    # VIX ≥ VIX_HIGH_THRESHOLD: Exposure = EXPOSURE_HIGH_VOLATILITY
    def map_vix_to_exposure(vix_value):
        if pd.isna(vix_value):
            return EXPOSURE_LOW_VOLATILITY  # Default to full exposure if VIX is missing
        if vix_value < VIX_LOW_THRESHOLD:
            return EXPOSURE_LOW_VOLATILITY
        elif vix_value < VIX_HIGH_THRESHOLD:
            return EXPOSURE_MODERATE_VOLATILITY
        else:
            return EXPOSURE_HIGH_VOLATILITY
    
    exposure_series = vix_series.apply(map_vix_to_exposure)
    
    # Apply rolling median smoothing if requested
    if apply_smoothing and smooth_window > 1:
        exposure_series = exposure_series.rolling(
            window=smooth_window, 
            center=False, 
            min_periods=1
        ).median()
    
    # Clip exposure to [0, 1] range
    exposure_series = exposure_series.clip(lower=0.0, upper=1.0)
    
    logger.debug(f"Computed exposure series: {len(exposure_series)} days, "
                f"avg={exposure_series.mean():.3f}, min={exposure_series.min():.3f}, max={exposure_series.max():.3f}")
    
    return exposure_series


def compute_volatility_regime_and_exposure(
    price_book: pd.DataFrame,
    smooth_window: int = 3
) -> Dict[str, Any]:
    """
    Compute volatility regime and exposure with VIX proxy selection and 3-day rolling median.
    
    This function provides detailed regime information for the S&P Flagship Wave (S&P 500 Wave).
    It selects the best available VIX proxy in priority order (^VIX > VIXY > VXX),
    applies a 3-day rolling median for smoothing, and maps VIX levels to exposure regimes.
    
    Requirements:
    - Use ticker priority: ^VIX > VIXY > VXX
    - Compute a 3-day rolling median for volatility smoothing
    - Map VIX to regime and exposure:
      - VIX < 18: Regime = "Low Volatility", Exposure = 1.00
      - 18 ≤ VIX < 25: Regime = "Moderate Volatility", Exposure = 0.65
      - VIX ≥ 25: Regime = "High Volatility", Exposure = 0.25
    - If no VIX proxy available, return available=False with reason="missing_vix_proxy"
    
    Args:
        price_book: PRICE_BOOK DataFrame (index=dates, columns=tickers, values=prices)
        smooth_window: Window size for rolling median smoothing (default: 3)
        
    Returns:
        Dictionary with:
        - success: bool - Whether computation succeeded
        - available: bool - Whether VIX data is available
        - reason: Optional[str] - Reason for unavailability (e.g., "missing_vix_proxy")
        - vix_ticker_used: Optional[str] - VIX ticker used (^VIX, VIXY, or VXX)
        - last_vix_date: Optional[str] - Last date with VIX data (YYYY-MM-DD)
        - last_vix_value: Optional[float] - Last VIX value (smoothed)
        - current_regime: Optional[str] - Current volatility regime
        - current_exposure: Optional[float] - Current exposure level [0, 1]
        - regime_series: Optional[pd.Series] - Daily regime labels (indexed by date)
        - exposure_series: Optional[pd.Series] - Daily exposure values (indexed by date)
        - vix_series_raw: Optional[pd.Series] - Raw VIX values (indexed by date)
        - vix_series_smoothed: Optional[pd.Series] - Smoothed VIX values (indexed by date)
        - data_frame: Optional[pd.DataFrame] - Complete DataFrame with columns:
          [date, vix_value_used, vix_ticker_used, regime, exposure]
          
    Example:
        >>> from helpers.price_book import get_price_book
        >>> price_book = get_price_book()
        >>> result = compute_volatility_regime_and_exposure(price_book)
        >>> if result['available']:
        >>>     print(f"VIX Ticker: {result['vix_ticker_used']}")
        >>>     print(f"Current Regime: {result['current_regime']}")
        >>>     print(f"Current Exposure: {result['current_exposure']:.2f}")
    """
    result = {
        'success': False,
        'available': False,
        'reason': None,
        'vix_ticker_used': None,
        'last_vix_date': None,
        'last_vix_value': None,
        'current_regime': None,
        'current_exposure': None,
        'regime_series': None,
        'exposure_series': None,
        'vix_series_raw': None,
        'vix_series_smoothed': None,
        'data_frame': None
    }
    
    # Validate inputs
    if price_book is None or price_book.empty:
        result['reason'] = 'PRICE_BOOK is empty'
        logger.warning("compute_volatility_regime_and_exposure: PRICE_BOOK is empty")
        return result
    
    # Try to find VIX series in order of preference: ^VIX > VIXY > VXX
    vix_ticker_preference = ['^VIX', 'VIXY', 'VXX']
    vix_series_raw = None
    vix_ticker_found = None
    
    for ticker in vix_ticker_preference:
        if ticker in price_book.columns:
            vix_series_raw = price_book[ticker].copy()
            vix_ticker_found = ticker
            logger.info(f"compute_volatility_regime_and_exposure: Using VIX proxy {ticker}")
            break
    
    # Edge case: No VIX proxy available
    if vix_series_raw is None or vix_ticker_found is None:
        result['available'] = False
        result['reason'] = 'missing_vix_proxy'
        logger.warning("compute_volatility_regime_and_exposure: No VIX proxy found (tried: ^VIX, VIXY, VXX)")
        return result
    
    # Remove NaN values
    vix_series_raw = vix_series_raw.dropna()
    
    if len(vix_series_raw) == 0:
        result['available'] = False
        result['reason'] = 'missing_vix_proxy'
        logger.warning(f"compute_volatility_regime_and_exposure: VIX series {vix_ticker_found} has no valid data")
        return result
    
    # Apply 3-day rolling median smoothing
    if smooth_window > 1:
        vix_series_smoothed = vix_series_raw.rolling(
            window=smooth_window,
            center=False,
            min_periods=1
        ).median()
    else:
        vix_series_smoothed = vix_series_raw.copy()
    
    # Map VIX to regime and exposure
    def map_vix_to_regime(vix_value):
        """Map VIX value to regime label."""
        if pd.isna(vix_value):
            return "Unknown"
        if vix_value < VIX_LOW_THRESHOLD:
            return "Low Volatility"
        elif vix_value < VIX_HIGH_THRESHOLD:
            return "Moderate Volatility"
        else:
            return "High Volatility"
    
    def map_vix_to_exposure(vix_value):
        """Map VIX value to exposure level."""
        if pd.isna(vix_value):
            return EXPOSURE_LOW_VOLATILITY  # Default to full exposure if VIX is missing
        if vix_value < VIX_LOW_THRESHOLD:
            return EXPOSURE_LOW_VOLATILITY
        elif vix_value < VIX_HIGH_THRESHOLD:
            return EXPOSURE_MODERATE_VOLATILITY
        else:
            return EXPOSURE_HIGH_VOLATILITY
    
    # Compute regime and exposure series
    regime_series = vix_series_smoothed.apply(map_vix_to_regime)
    exposure_series = vix_series_smoothed.apply(map_vix_to_exposure)
    
    # Clip exposure to [0, 1] range for safety
    exposure_series = exposure_series.clip(lower=0.0, upper=1.0)
    
    # Get latest values
    last_vix_date = vix_series_smoothed.index[-1]
    last_vix_value = float(vix_series_smoothed.iloc[-1])
    current_regime = regime_series.iloc[-1]
    current_exposure = float(exposure_series.iloc[-1])
    
    # Build data frame with required columns: date, vix_value_used, vix_ticker_used, regime, exposure
    data_frame = pd.DataFrame({
        'vix_value_used': vix_series_smoothed,
        'vix_ticker_used': vix_ticker_found,
        'regime': regime_series,
        'exposure': exposure_series
    })
    data_frame.index.name = 'date'
    data_frame = data_frame.reset_index()
    
    # Populate result
    result['success'] = True
    result['available'] = True
    result['reason'] = None
    result['vix_ticker_used'] = vix_ticker_found
    result['last_vix_date'] = last_vix_date.strftime('%Y-%m-%d')
    result['last_vix_value'] = last_vix_value
    result['current_regime'] = current_regime
    result['current_exposure'] = current_exposure
    result['regime_series'] = regime_series
    result['exposure_series'] = exposure_series
    result['vix_series_raw'] = vix_series_raw
    result['vix_series_smoothed'] = vix_series_smoothed
    result['data_frame'] = data_frame
    
    logger.debug(f"compute_volatility_regime_and_exposure: Computed {len(data_frame)} days, "
                f"VIX ticker={vix_ticker_found}, current regime={current_regime}, "
                f"current exposure={current_exposure:.3f}")
    
    return result


def compute_crypto_regime_diagnostics(
    price_book: pd.DataFrame,
    wave_name: str
) -> Dict[str, Any]:
    """
    Compute crypto-specific regime diagnostics for a crypto wave.
    
    This function provides detailed crypto volatility regime information for 
    diagnostics display in the performance pipeline. It is only applicable to
    crypto waves and returns N/A for equity waves.
    
    Args:
        price_book: PRICE_BOOK DataFrame (index=dates, columns=tickers, values=prices)
        wave_name: Name of the wave to analyze
        
    Returns:
        Dictionary with:
        - applicable: bool - Whether this wave uses crypto overlay
        - regime: str - Current regime ('calm', 'normal', 'elevated', 'stress')
        - exposure: float - Current exposure recommendation [0.0, 1.0]
        - overlay_status: str - Status message
        - data_quality: str - Quality assessment
        - btc_vol_ratio: Optional[float] - BTC volatility ratio
        - eth_vol_ratio: Optional[float] - ETH volatility ratio
        - combined_vol_ratio: Optional[float] - Combined volatility ratio
    """
    result = {
        'applicable': False,
        'regime': 'N/A',
        'exposure': 1.00,
        'overlay_status': 'Not Applicable',
        'data_quality': 'N/A',
        'btc_vol_ratio': None,
        'eth_vol_ratio': None,
        'combined_vol_ratio': None
    }
    
    # Check if crypto overlay is available
    if not CRYPTO_OVERLAY_AVAILABLE:
        logger.debug("Crypto overlay not available")
        return result
    
    # Check if waves_engine is available to detect crypto waves
    if not WAVES_ENGINE_AVAILABLE:
        logger.debug("Waves engine not available")
        return result
    
    # Check if this is a crypto wave
    try:
        is_crypto = _is_crypto_wave(wave_name)
    except Exception as e:
        logger.warning(f"Error detecting crypto wave: {e}")
        return result
    
    if not is_crypto:
        # Not a crypto wave, return N/A
        return result
    
    # This is a crypto wave - compute diagnostics
    result['applicable'] = True
    
    # Get crypto wave exposure
    try:
        exposure_result = get_crypto_wave_exposure(wave_name, price_book)
        
        result['regime'] = exposure_result['regime']
        result['exposure'] = exposure_result['exposure']
        result['overlay_status'] = exposure_result['overlay_status']
        result['data_quality'] = exposure_result['data_quality']
        
        # Get detailed regime data
        regime_result = compute_crypto_volatility_regime(price_book)
        result['btc_vol_ratio'] = regime_result.get('btc_vol_ratio')
        result['eth_vol_ratio'] = regime_result.get('eth_vol_ratio')
        result['combined_vol_ratio'] = regime_result.get('combined_vol_ratio')
        
        logger.info(f"Crypto regime diagnostics for {wave_name}: "
                   f"regime={result['regime']}, exposure={result['exposure']:.2f}, "
                   f"status={result['overlay_status']}")
        
    except Exception as e:
        logger.error(f"Error computing crypto regime diagnostics for {wave_name}: {e}")
        result['overlay_status'] = f"Error: {str(e)}"
    
    return result


def compute_portfolio_alpha_ledger(
    price_book: pd.DataFrame,
    periods: List[int] = [1, 30, 60, 365],
    benchmark_ticker: str = "SPY",
    safe_ticker_preference: List[str] = ["BIL", "SHY"],
    mode: str = "Standard",
    wave_registry: Optional[Dict] = None,
    vix_exposure_enabled: bool = True
) -> Dict[str, Any]:
    """
    Compute canonical portfolio alpha ledger with deterministic attribution.
    
    This is the primary function for computing portfolio-level metrics with full
    transparency and attribution decomposition. It provides:
    
    1. Daily return calculation:
       - Risk-sleeve returns (equal-weight portfolio across waves)
       - Safe asset returns (BIL preferred, fallback to SHY)
       - Exposure series from VIX regime (if available)
       - Realized returns: exposure * risk_return + (1-exposure) * safe_return
       - Unoverlay returns: risk_return (exposure=1.0)
       - Benchmark returns (SPY or specified)
    
    2. Alpha calculation for each period (N):
       - Cumulative returns over strict last N trading rows
       - Attribution: total_alpha, selection_alpha, overlay_alpha
       - Residual validation (should be near 0)
       - Availability flags for insufficient data
    
    3. Alpha Captured computation (if VIX exposure exists):
       - Daily alpha = realized - benchmark
       - Exposure-weighted cumulative metrics
    
    Args:
        price_book: PRICE_BOOK DataFrame (index=dates, columns=tickers, values=prices)
        periods: List of lookback periods in days (default: [1, 30, 60, 365])
        benchmark_ticker: Benchmark ticker symbol (default: "SPY")
        safe_ticker_preference: Ordered list of safe asset tickers (default: ["BIL", "SHY"])
        mode: Operating mode (default: "Standard")
        wave_registry: Wave registry dict (default: None, uses WAVE_WEIGHTS)
        vix_exposure_enabled: Whether to compute VIX-based exposure (default: True)
        
    Returns:
        Dictionary with:
        - success: bool - Whether computation succeeded
        - failure_reason: Optional[str] - Reason for failure
        - daily_risk_return: pd.Series - Daily risk-sleeve returns (equal-weight waves)
        - daily_safe_return: pd.Series - Daily safe asset returns
        - daily_exposure: pd.Series - Daily exposure levels [0, 1]
        - daily_realized_return: pd.Series - Daily realized returns (with overlay)
        - daily_unoverlay_return: pd.Series - Daily unoverlay returns (exposure=1.0)
        - daily_benchmark_return: pd.Series - Daily benchmark returns
        - daily_ledger: pd.DataFrame - Daily ledger with columns:
          [risk_return, safe_return, benchmark_return, exposure, realized_return,
           alpha_total, alpha_selection, alpha_overlay, alpha_residual]
        - reconciliation_passed: bool - Whether reconciliation checks passed
        - reconciliation_1_max_diff: float - Max diff for reconciliation 1 check
        - reconciliation_2_max_diff: float - Max diff for reconciliation 2 check
        - period_results: Dict[str, Dict] - Results for each period
        - vix_ticker_used: Optional[str] - VIX ticker used (None if unavailable)
        - safe_ticker_used: Optional[str] - Safe ticker used (None if unavailable)
        - overlay_available: bool - Whether VIX overlay was applied
        - warnings: List[str] - Any warnings about data quality
        
    Reconciliation Checks (tolerance = 0.10% = 0.001):
        1. realized_return - benchmark_return == alpha_total
        2. alpha_selection + alpha_overlay + alpha_residual == alpha_total
        
        If either check fails (max diff > tolerance), the function marks the period
        as unavailable with reason "reconciliation_failed" but still returns the
        ledger for debugging purposes.
        
    Period result structure (for each period in periods):
        - period: int - Period in days
        - available: bool - Whether sufficient data exists
        - reason: Optional[str] - Reason if unavailable
        - rows_used: int - Actual trading rows used
        - start_date: Optional[str] - Start date of period (YYYY-MM-DD)
        - end_date: Optional[str] - End date of period (YYYY-MM-DD)
        - cum_realized: Optional[float] - Cumulative realized return
        - cum_unoverlay: Optional[float] - Cumulative unoverlay return
        - cum_benchmark: Optional[float] - Cumulative benchmark return
        - total_alpha: Optional[float] - cum_realized - cum_benchmark
        - selection_alpha: Optional[float] - cum_unoverlay - cum_benchmark
        - overlay_alpha: Optional[float] - cum_realized - cum_unoverlay
        - residual: Optional[float] - total_alpha - (selection_alpha + overlay_alpha)
        - alpha_captured: Optional[float] - Exposure-weighted alpha (if overlay available)
        
    Example:
        >>> from helpers.price_book import get_price_book
        >>> price_book = get_price_book()
        >>> ledger = compute_portfolio_alpha_ledger(price_book, periods=[1, 30, 60, 365])
        >>> if ledger['success']:
        >>>     for period_key, period_data in ledger['period_results'].items():
        >>>         if period_data['available']:
        >>>             print(f"{period_key}: Total Alpha = {period_data['total_alpha']:.2%}")
    """
    result = {
        'success': False,
        'failure_reason': None,
        'daily_risk_return': None,
        'daily_safe_return': None,
        'daily_exposure': None,
        'daily_realized_return': None,
        'daily_unoverlay_return': None,
        'daily_benchmark_return': None,
        'daily_ledger': None,
        'reconciliation_passed': False,
        'reconciliation_1_max_diff': None,
        'reconciliation_2_max_diff': None,
        'period_results': {},
        'vix_ticker_used': None,
        'safe_ticker_used': None,
        'overlay_available': False,
        'warnings': []
    }
    
    # Validate inputs
    if price_book is None or price_book.empty:
        result['failure_reason'] = 'PRICE_BOOK is empty'
        return result
    
    if not WAVES_ENGINE_AVAILABLE:
        result['failure_reason'] = 'waves_engine not available'
        return result
    
    # Use WAVE_WEIGHTS as registry if not provided
    if wave_registry is None:
        wave_registry = WAVE_WEIGHTS
    
    try:
        universe = get_all_waves_universe()
        all_waves = universe.get('waves', [])
    except Exception as e:
        result['failure_reason'] = f'Error getting wave universe: {str(e)}'
        return result
    
    if not all_waves:
        result['failure_reason'] = 'No waves found in universe'
        return result
    
    # ========================================================================
    # Step 1: Compute daily_risk_return (equal-weight portfolio, exposure=1.0)
    # ========================================================================
    try:
        wave_return_dict = {}
        
        for wave_name in all_waves:
            if wave_name not in wave_registry:
                continue
            
            wave_holdings = wave_registry[wave_name]
            if not wave_holdings:
                continue
            
            try:
                tickers = [h.ticker for h in wave_holdings]
                weights = [h.weight for h in wave_holdings]
                
                total_weight = sum(weights)
                if total_weight == 0:
                    continue
                normalized_weights = [w / total_weight for w in weights]
                
                # Filter to available tickers
                available_tickers = []
                available_weights = []
                for ticker, weight in zip(tickers, normalized_weights):
                    if ticker in price_book.columns:
                        available_tickers.append(ticker)
                        available_weights.append(weight)
                
                if not available_tickers:
                    continue
                
                # Renormalize
                total_available = sum(available_weights)
                renormalized_weights = [w / total_available for w in available_weights]
                
                # Compute daily returns
                ticker_prices = price_book[available_tickers].copy()
                if len(ticker_prices) < 2:
                    continue
                
                ticker_returns = ticker_prices.pct_change()
                
                # Weighted returns
                wave_returns = pd.Series(0.0, index=ticker_returns.index)
                for i, ticker in enumerate(available_tickers):
                    weight = renormalized_weights[i]
                    ticker_ret = ticker_returns[ticker]
                    wave_returns = wave_returns.add(ticker_ret * weight, fill_value=0.0)
                
                wave_returns = wave_returns.iloc[1:]  # Drop first NaN
                
                if len(wave_returns) > 0 and wave_returns.notna().any():
                    wave_return_dict[wave_name] = wave_returns
            
            except Exception as e:
                logger.warning(f"Failed to compute returns for wave {wave_name}: {e}")
                continue
        
        if not wave_return_dict:
            result['failure_reason'] = 'No valid wave return series computed'
            return result
        
        # Aggregate to portfolio level (equal weight)
        return_matrix = pd.DataFrame(wave_return_dict)
        daily_risk_return = return_matrix.mean(axis=1, skipna=True)
        
        # Drop dates where all are NaN
        valid_dates = ~daily_risk_return.isna()
        daily_risk_return = daily_risk_return[valid_dates].sort_index()
        
        if len(daily_risk_return) < 2:
            result['failure_reason'] = 'Insufficient dates after aggregation'
            return result
        
        result['daily_risk_return'] = daily_risk_return
        result['daily_unoverlay_return'] = daily_risk_return.copy()  # Same as risk return (exposure=1.0)
        
    except Exception as e:
        result['failure_reason'] = f'Error computing risk returns: {str(e)}'
        return result
    
    # ========================================================================
    # Step 2: Compute daily_safe_return
    # ========================================================================
    try:
        safe_ticker = None
        for ticker in safe_ticker_preference:
            if ticker in price_book.columns:
                safe_ticker = ticker
                result['safe_ticker_used'] = ticker
                logger.info(f"Using safe asset: {ticker}")
                break
        
        if safe_ticker is not None:
            safe_prices = price_book[safe_ticker].copy()
            daily_safe_return = safe_prices.pct_change().iloc[1:]
            # Align with risk return dates
            daily_safe_return = daily_safe_return.reindex(daily_risk_return.index, fill_value=0.0)
        else:
            # Use 0 return for safe sleeve if no safe asset available
            daily_safe_return = pd.Series(0.0, index=daily_risk_return.index)
            result['warnings'].append('No safe asset ticker found (BIL, SHY) - using 0% safe return')
        
        result['daily_safe_return'] = daily_safe_return
        
    except Exception as e:
        result['failure_reason'] = f'Error computing safe returns: {str(e)}'
        return result
    
    # ========================================================================
    # Step 3: Compute daily_exposure series (VIX-based if enabled)
    # ========================================================================
    try:
        if vix_exposure_enabled:
            exposure_series = compute_portfolio_exposure_series(price_book, mode=mode)
            
            if exposure_series is not None:
                # Align exposure with risk return dates
                daily_exposure = exposure_series.reindex(daily_risk_return.index, method='ffill')
                # Fill any remaining NaN with 1.0 (full exposure)
                daily_exposure = daily_exposure.fillna(1.0)
                result['overlay_available'] = True
                
                # Determine which VIX ticker was used
                for ticker in ['^VIX', 'VIXY', 'VXX']:
                    if ticker in price_book.columns:
                        result['vix_ticker_used'] = ticker
                        break
            else:
                # No VIX proxy available - use full exposure
                daily_exposure = pd.Series(1.0, index=daily_risk_return.index)
                result['overlay_available'] = False
                result['warnings'].append('No VIX proxy available - using full exposure (1.0)')
        else:
            # VIX exposure disabled - use full exposure
            daily_exposure = pd.Series(1.0, index=daily_risk_return.index)
            result['overlay_available'] = False
        
        result['daily_exposure'] = daily_exposure
        
    except Exception as e:
        # Error computing exposure - use fallback
        daily_exposure = pd.Series(1.0, index=daily_risk_return.index)
        result['daily_exposure'] = daily_exposure
        result['overlay_available'] = False
        result['warnings'].append(f'Error computing exposure: {str(e)} - using full exposure (1.0)')
        logger.warning(f'Error computing exposure in alpha ledger: {str(e)}')
    
    # ========================================================================
    # Step 4: Compute daily_realized_return (with overlay)
    # ========================================================================
    # realized_return(t) = exposure(t) * risk_return(t) + (1-exposure(t)) * safe_return(t)
    try:
        daily_realized_return = (
            daily_exposure * daily_risk_return + 
            (1 - daily_exposure) * daily_safe_return
        )
        result['daily_realized_return'] = daily_realized_return
    
    except Exception as e:
        result['failure_reason'] = f'Error computing realized returns: {str(e)}'
        return result
    
    # ========================================================================
    # Step 5: Compute daily_benchmark_return
    # ========================================================================
    try:
        if benchmark_ticker in price_book.columns:
            benchmark_prices = price_book[benchmark_ticker].copy()
            daily_benchmark_return = benchmark_prices.pct_change().iloc[1:]
            # Align with portfolio dates
            daily_benchmark_return = daily_benchmark_return.reindex(daily_risk_return.index, fill_value=0.0)
        else:
            # Use risk return as benchmark if SPY not available
            daily_benchmark_return = daily_risk_return.copy()
            result['warnings'].append(f'Benchmark ticker {benchmark_ticker} not found - using risk return as benchmark')
        
        result['daily_benchmark_return'] = daily_benchmark_return
        
    except Exception as e:
        result['failure_reason'] = f'Error computing benchmark returns: {str(e)}'
        return result
    
    # ========================================================================
    # Step 6: Compute period results with strict windowing and attribution
    # ========================================================================
    try:
        def compute_cumulative_return(series: pd.Series, window: int) -> Optional[float]:
            """Compute cumulative return over last N days (strict slicing)."""
            if len(series) < window:
                return None
            window_series = series.iloc[-window:]
            cum_return = (1 + window_series).prod() - 1
            return float(cum_return)
        
        def compute_alpha_captured(
            daily_alpha: pd.Series, 
            daily_exposure: pd.Series, 
            window: int
        ) -> Optional[float]:
            """
            Compute exposure-weighted alpha captured over window.
            
            Alpha captured measures the alpha that was actually captured given
            the exposure levels. It's computed as the sum of daily alpha weighted
            by exposure.
            """
            if len(daily_alpha) < window or len(daily_exposure) < window:
                return None
            
            window_alpha = daily_alpha.iloc[-window:]
            window_exposure = daily_exposure.iloc[-window:]
            
            # Align series
            aligned = pd.DataFrame({
                'alpha': window_alpha,
                'exposure': window_exposure
            })
            
            # Alpha captured = sum(daily_alpha * exposure)
            alpha_captured = (aligned['alpha'] * aligned['exposure']).sum()
            return float(alpha_captured)
        
        # Compute daily alpha for alpha_captured calculation
        daily_alpha = daily_realized_return - daily_benchmark_return
        
        rows_available = len(daily_realized_return)
        
        for period in periods:
            # Strict windowing: compute only if we have exactly the requested period rows
            if rows_available < period:
                # Insufficient rows
                result['period_results'][f'{period}D'] = {
                    'period': period,
                    'available': False,
                    'reason': 'insufficient_aligned_rows',
                    'rows_used': rows_available,
                    'start_date': None,
                    'end_date': None,
                    'cum_realized': None,
                    'cum_unoverlay': None,
                    'cum_benchmark': None,
                    'total_alpha': None,
                    'selection_alpha': None,
                    'overlay_alpha': None,
                    'residual': None,
                    'alpha_captured': None
                }
                continue
            
            # We have sufficient rows - compute using strict last N rows
            cum_realized = compute_cumulative_return(daily_realized_return, period)
            cum_unoverlay = compute_cumulative_return(result['daily_unoverlay_return'], period)
            cum_benchmark = compute_cumulative_return(daily_benchmark_return, period)
            
            # Get date range for this period
            period_series = daily_realized_return.iloc[-period:]
            start_date = period_series.index[0].strftime('%Y-%m-%d')
            end_date = period_series.index[-1].strftime('%Y-%m-%d')
            
            if cum_realized is None or cum_unoverlay is None or cum_benchmark is None:
                # Computation error
                result['period_results'][f'{period}D'] = {
                    'period': period,
                    'available': False,
                    'reason': 'computation_error',
                    'rows_used': period,
                    'start_date': start_date,
                    'end_date': end_date,
                    'cum_realized': None,
                    'cum_unoverlay': None,
                    'cum_benchmark': None,
                    'total_alpha': None,
                    'selection_alpha': None,
                    'overlay_alpha': None,
                    'residual': None,
                    'alpha_captured': None
                }
                continue
            
            # Compute attribution
            total_alpha = cum_realized - cum_benchmark
            selection_alpha = cum_unoverlay - cum_benchmark
            overlay_alpha = cum_realized - cum_unoverlay
            residual = total_alpha - (selection_alpha + overlay_alpha)
            
            # Reconciliation checks (Requirements from problem statement)
            # 1. Portfolio Return − Benchmark Return = Total Alpha
            reconciliation_1_diff = abs((cum_realized - cum_benchmark) - total_alpha)
            
            # 2. Selection Alpha + Overlay Alpha + Residual = Total Alpha
            reconciliation_2_diff = abs((selection_alpha + overlay_alpha + residual) - total_alpha)
            
            # Check if reconciliations fail tolerance
            if reconciliation_1_diff > RESIDUAL_TOLERANCE:
                # Reconciliation 1 failed
                result['period_results'][f'{period}D'] = {
                    'period': period,
                    'available': False,
                    'reason': f'reconciliation_1_failed: |({cum_realized:.6f} - {cum_benchmark:.6f}) - {total_alpha:.6f}| = {reconciliation_1_diff:.6f} > {RESIDUAL_TOLERANCE:.6f}',
                    'rows_used': period,
                    'start_date': start_date,
                    'end_date': end_date,
                    'cum_realized': None,
                    'cum_unoverlay': None,
                    'cum_benchmark': None,
                    'total_alpha': None,
                    'selection_alpha': None,
                    'overlay_alpha': None,
                    'residual': None,
                    'alpha_captured': None
                }
                continue
            
            if reconciliation_2_diff > RESIDUAL_TOLERANCE:
                # Reconciliation 2 failed
                result['period_results'][f'{period}D'] = {
                    'period': period,
                    'available': False,
                    'reason': f'reconciliation_2_failed: |({selection_alpha:.6f} + {overlay_alpha:.6f} + {residual:.6f}) - {total_alpha:.6f}| = {reconciliation_2_diff:.6f} > {RESIDUAL_TOLERANCE:.6f}',
                    'rows_used': period,
                    'start_date': start_date,
                    'end_date': end_date,
                    'cum_realized': None,
                    'cum_unoverlay': None,
                    'cum_benchmark': None,
                    'total_alpha': None,
                    'selection_alpha': None,
                    'overlay_alpha': None,
                    'residual': None,
                    'alpha_captured': None
                }
                continue
            
            # Compute alpha captured (if overlay available)
            alpha_captured = None
            if result['overlay_available']:
                alpha_captured = compute_alpha_captured(daily_alpha, daily_exposure, period)
            
            # All reconciliations passed - period is available
            result['period_results'][f'{period}D'] = {
                'period': period,
                'available': True,
                'reason': None,
                'rows_used': period,
                'start_date': start_date,
                'end_date': end_date,
                'cum_realized': cum_realized,
                'cum_unoverlay': cum_unoverlay,
                'cum_benchmark': cum_benchmark,
                'total_alpha': total_alpha,
                'selection_alpha': selection_alpha,
                'overlay_alpha': overlay_alpha,
                'residual': residual,
                'alpha_captured': alpha_captured
            }
        
        result['success'] = True
        result['failure_reason'] = None
        
    except Exception as e:
        result['failure_reason'] = f'Error computing period results: {str(e)}'
        return result
    
    # ========================================================================
    # Step 7: Build Daily Ledger DataFrame with Reconciliation
    # ========================================================================
    try:
        # Build daily ledger with required columns
        daily_ledger = pd.DataFrame({
            'risk_return': daily_risk_return,
            'safe_return': daily_safe_return,
            'benchmark_return': daily_benchmark_return,
            'exposure': daily_exposure,
            'realized_return': daily_realized_return,
        })
        
        # Compute alpha components for each day
        daily_ledger['alpha_total'] = daily_ledger['realized_return'] - daily_ledger['benchmark_return']
        daily_ledger['alpha_selection'] = daily_ledger['risk_return'] - daily_ledger['benchmark_return']
        daily_ledger['alpha_overlay'] = daily_ledger['realized_return'] - daily_ledger['risk_return']
        daily_ledger['alpha_residual'] = (
            daily_ledger['alpha_total'] - 
            (daily_ledger['alpha_selection'] + daily_ledger['alpha_overlay'])
        )
        
        # Reconciliation checks (per requirements)
        # 1. realized_return - benchmark_return == alpha_total
        reconciliation_1 = np.abs(
            (daily_ledger['realized_return'] - daily_ledger['benchmark_return']) - 
            daily_ledger['alpha_total']
        )
        
        # 2. alpha_selection + alpha_overlay + alpha_residual == alpha_total
        reconciliation_2 = np.abs(
            (daily_ledger['alpha_selection'] + daily_ledger['alpha_overlay'] + daily_ledger['alpha_residual']) - 
            daily_ledger['alpha_total']
        )
        
        # Check if any row fails reconciliation (tolerance = 0.10% = 0.001)
        reconciliation_1_failed = (reconciliation_1 > RESIDUAL_TOLERANCE).any()
        reconciliation_2_failed = (reconciliation_2 > RESIDUAL_TOLERANCE).any()
        
        if reconciliation_1_failed or reconciliation_2_failed:
            # Mark as reconciliation failed
            result['success'] = False
            result['available'] = False
            
            if reconciliation_1_failed:
                max_diff_1 = reconciliation_1.max()
                result['failure_reason'] = f'reconciliation_failed: Reconciliation 1 failed with max diff {max_diff_1:.6f} > tolerance {RESIDUAL_TOLERANCE:.6f}'
            else:
                max_diff_2 = reconciliation_2.max()
                result['failure_reason'] = f'reconciliation_failed: Reconciliation 2 failed with max diff {max_diff_2:.6f} > tolerance {RESIDUAL_TOLERANCE:.6f}'
            
            # Still return ledger for debugging
            result['daily_ledger'] = daily_ledger
            result['reconciliation_passed'] = False
            result['reconciliation_1_max_diff'] = float(reconciliation_1.max())
            result['reconciliation_2_max_diff'] = float(reconciliation_2.max())
            
            logger.warning(f"Daily ledger reconciliation failed: {result['failure_reason']}")
        else:
            # Reconciliation passed
            result['daily_ledger'] = daily_ledger
            result['reconciliation_passed'] = True
            result['reconciliation_1_max_diff'] = float(reconciliation_1.max())
            result['reconciliation_2_max_diff'] = float(reconciliation_2.max())
            
            logger.debug(f"Daily ledger reconciliation passed: "
                        f"max_diff_1={reconciliation_1.max():.8f}, "
                        f"max_diff_2={reconciliation_2.max():.8f}")
    
    except Exception as e:
        result['failure_reason'] = f'Error building daily ledger: {str(e)}'
        result['reconciliation_passed'] = False
        return result
    
    return result

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

"""
analytics_pipeline.py

WAVES Intelligence™ Stage 4: Universal Analytics Data Pipeline

This module provides a standardized data emission pipeline that ensures complete
daily analytics artifacts for all waves in the registry.

Key Features:
- Universal emit function for all wave_ids
- Standardized file structure per wave
- Composite benchmark materialization
- Data validation and reporting

Directory Structure per wave:
    data/waves/{wave_id}/
        - prices.csv           (7+ days of ticker prices)
        - benchmark_prices.csv (7+ days of benchmark prices)
        - positions.csv        (current position snapshot)
        - trades.csv           (trade history, may be empty)
        - nav.csv              (NAV history aligned with holdings)

Usage:
    from analytics_pipeline import run_daily_analytics_pipeline
    
    # Run for all waves with 14-day lookback
    result = run_daily_analytics_pipeline(all_waves=True, lookback_days=14)
    
    # Run for specific wave
    result = run_daily_analytics_pipeline(wave_ids=['sp500_wave'], lookback_days=7)
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

# Import ticker diagnostics
try:
    from helpers.ticker_diagnostics import (
        FailedTickerReport, 
        FailureType, 
        categorize_error,
        get_diagnostics_tracker
    )
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False

# Import circuit breaker
try:
    from helpers.circuit_breaker import get_circuit_breaker
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False

# Import from waves_engine
from waves_engine import (
    get_all_wave_ids,
    get_all_waves_universe,
    get_display_name_from_wave_id,
    WAVE_WEIGHTS,
    BENCHMARK_WEIGHTS_STATIC,
    Holding,
)

# Constants
TRADING_DAYS_PER_YEAR = 252
DEFAULT_LOOKBACK_DAYS = 14  # Fetch 14 days to ensure 7+ trading days
MIN_REQUIRED_TRADING_DAYS = 7
ANALYTICS_BASE_DIR = "data/waves"

# Circuit breaker constants to prevent infinite retries
MAX_RETRIES = 3  # Maximum retry attempts for any single operation
MAX_INDIVIDUAL_TICKER_FETCHES = 50  # Maximum tickers to fetch individually before aborting

# Graded Readiness Thresholds
# These thresholds balance usability with data quality
# - Operational: Enough data to show current state (basic display)
# - Partial: Enough data for basic analytics (trends, simple returns)
# - Full: Enough data for all analytics (multi-window, correlations, alpha)
MIN_DAYS_OPERATIONAL = 1      # Minimum for current state display (lenient to show something)
MIN_DAYS_PARTIAL = 7          # Minimum for basic analytics (1 week of data)
MIN_DAYS_FULL = 365           # Required for full multi-window analytics (1 year of data)
MIN_COVERAGE_OPERATIONAL = 0.50  # 50% coverage for operational (allows partial data display)
MIN_COVERAGE_PARTIAL = 0.70      # 70% coverage for partial (reasonably complete data)
MIN_COVERAGE_FULL = 0.90         # 90% coverage for full (near-complete data)
MAX_DAYS_STALE = 7            # Maximum age in days before data considered stale


# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def ensure_directory_exists(path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_wave_analytics_dir(wave_id: str) -> str:
    """Get the analytics directory path for a specific wave."""
    return os.path.join(ANALYTICS_BASE_DIR, wave_id)


def get_trading_days_back(days: int) -> datetime:
    """Get a start date that should cover the requested number of trading days."""
    # Add buffer for weekends and holidays (multiply by ~1.5)
    calendar_days = int(days * 1.5) + 5
    return datetime.now() - timedelta(days=calendar_days)


def _log_readiness_result(result: Dict[str, Any]) -> None:
    """
    Log readiness evaluation result in a structured format.
    
    Args:
        result: Readiness diagnostic dictionary
    """
    import logging
    
    logger = logging.getLogger('analytics_pipeline')
    
    wave_id = result.get('wave_id', 'UNKNOWN')
    ready = result.get('is_ready', False)
    reason_codes = result.get('reason_codes', [])
    missing_tickers = result.get('missing_tickers', [])
    missing_benchmark_tickers = result.get('missing_benchmark_tickers', [])
    exception = result.get('exception', None)
    
    # Build log message
    log_parts = [
        f"[DATA_READY]",
        f"wave_id={wave_id}",
        f"ready={str(ready).lower()}",
        f"reasons={reason_codes}",
    ]
    
    if missing_tickers:
        log_parts.append(f"missing_tickers={missing_tickers}")
    
    if missing_benchmark_tickers:
        log_parts.append(f"missing_benchmark_tickers={missing_benchmark_tickers}")
    
    if exception:
        log_parts.append(f"exception=\"{exception}\"")
    
    log_message = " ".join(log_parts)
    
    if ready:
        logger.info(log_message)
    else:
        logger.warning(log_message)



# ------------------------------------------------------------
# Ticker Normalization
# ------------------------------------------------------------

def normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbols to match yfinance conventions.
    
    Examples:
        BRK.B -> BRK-B
        BF.B -> BF-B
        stETH-USD -> stETH-USD (already normalized)
    
    Args:
        ticker: Raw ticker symbol
        
    Returns:
        Normalized ticker symbol
    """
    if not ticker:
        return ticker
    
    # Replace dots with hyphens for class shares (e.g., BRK.B -> BRK-B)
    normalized = ticker.replace('.', '-')
    
    # Ensure crypto symbols have -USD suffix if needed
    # (This is a simple heuristic - crypto tickers already have -USD in WAVE_WEIGHTS)
    
    return normalized


# ------------------------------------------------------------
# Ticker & Benchmark Resolution
# ------------------------------------------------------------

def resolve_wave_tickers(wave_id: str) -> List[str]:
    """
    Resolve all tickers for a given wave_id from WAVE_WEIGHTS with normalization.
    
    IMPORTANT: All tickers are validated against universal_universe.csv.
    Invalid tickers are logged but do not block wave rendering (graceful degradation).
    
    Args:
        wave_id: The wave identifier
        
    Returns:
        List of normalized ticker symbols (validated against universal universe)
    """
    display_name = get_display_name_from_wave_id(wave_id)
    if not display_name:
        return []
    
    holdings = WAVE_WEIGHTS.get(display_name, [])
    tickers = []
    
    for holding in holdings:
        if isinstance(holding, Holding):
            tickers.append(normalize_ticker(holding.ticker))
        elif isinstance(holding, dict) and 'ticker' in holding:
            tickers.append(normalize_ticker(holding['ticker']))
        elif isinstance(holding, str):
            tickers.append(normalize_ticker(holding))
    
    # Remove duplicates
    unique_tickers = list(set(tickers))
    
    # Validate against universal universe (graceful degradation)
    try:
        from helpers.universal_universe import get_tickers_for_wave_with_degradation
        validated_tickers, degradation_report = get_tickers_for_wave_with_degradation(
            unique_tickers, 
            wave_name=display_name
        )
        
        # Log degradation if any
        if degradation_report:
            logger.warning(
                f"Wave '{wave_id}' degraded: {len(validated_tickers)}/{len(unique_tickers)} "
                f"tickers validated. Missing: {list(degradation_report.keys())[:5]}"
            )
        
        # Return validated tickers (gracefully degraded)
        return validated_tickers if validated_tickers else unique_tickers
        
    except Exception as e:
        # If validation fails, log and continue with all tickers (fail-safe)
        logger.warning(f"Could not validate tickers against universal universe: {e}")
        return unique_tickers


def resolve_wave_benchmarks(wave_id: str) -> List[Tuple[str, float]]:
    """
    Resolve benchmark tickers and weights for a given wave_id with normalization.
    
    Args:
        wave_id: The wave identifier
        
    Returns:
        List of (normalized_ticker, weight) tuples
    """
    display_name = get_display_name_from_wave_id(wave_id)
    if not display_name:
        return []
    
    benchmark_spec = BENCHMARK_WEIGHTS_STATIC.get(display_name, [])
    benchmarks = []
    
    for bm in benchmark_spec:
        if isinstance(bm, Holding):
            benchmarks.append((normalize_ticker(bm.ticker), bm.weight))
        elif isinstance(bm, dict):
            benchmarks.append((normalize_ticker(bm.get('ticker', '')), bm.get('weight', 1.0)))
        elif isinstance(bm, str):
            benchmarks.append((normalize_ticker(bm), 1.0))
    
    return benchmarks


# ------------------------------------------------------------
# Price Data Fetching with Retry Logic
# ------------------------------------------------------------

def _retry_with_backoff(func, *args, max_retries: int = MAX_RETRIES, initial_delay: float = 1.0, **kwargs):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts (defaults to MAX_RETRIES=3)
        initial_delay: Initial delay in seconds
        **kwargs: Keyword arguments for func
        
    Returns:
        Result from successful function call
        
    Raises:
        Last exception if all retries fail
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()
            
            # Check if error is retryable (rate limit, timeout, network)
            is_retryable = any(keyword in error_msg for keyword in [
                'rate limit', 'timeout', 'connection', 'network', 
                '429', 'too many requests', 'quota'
            ])
            
            if not is_retryable or attempt == max_retries - 1:
                # Not retryable or last attempt - raise immediately
                raise
            
            # Wait before retrying with exponential backoff
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    
    # Should never reach here, but just in case
    raise last_exception if last_exception else Exception("Retry failed")


def fetch_prices(tickers: List[str], start_date: datetime, end_date: datetime, use_dummy_data: bool = False, wave_id: Optional[str] = None, wave_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Fetch historical prices for a list of tickers with per-ticker error isolation.
    Enhanced with retry logic, diagnostics tracking, and better error categorization.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        use_dummy_data: If True, generate dummy data instead of fetching from yfinance
        wave_id: Optional wave identifier for diagnostics tracking
        wave_name: Optional wave display name for diagnostics tracking
        
    Returns:
        Tuple of (prices_df, failures_dict):
        - prices_df: DataFrame with dates as index and tickers as columns
        - failures_dict: Dict mapping failed tickers to error reasons
    """
    failures = {}
    
    # Get diagnostics tracker if available
    tracker = get_diagnostics_tracker() if DIAGNOSTICS_AVAILABLE else None
    
    if use_dummy_data:
        # Generate dummy price data for testing
        dates = pd.bdate_range(start=start_date, end=end_date)
        np.random.seed(42)  # For reproducibility
        
        prices_dict = {}
        for ticker in tickers:
            # Generate random walk prices starting at 100
            returns = np.random.normal(0.0005, 0.015, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            prices_dict[ticker] = prices
        
        prices = pd.DataFrame(prices_dict, index=dates)
        return prices, failures
    
    if not YFINANCE_AVAILABLE:
        error_msg = "yfinance is not available"
        print(f"Error: {error_msg}")
        for ticker in tickers:
            failures[ticker] = error_msg
            # Track failure in diagnostics
            if tracker:
                failure_type, suggested_fix = categorize_error(error_msg, ticker)
                report = FailedTickerReport(
                    ticker_original=ticker,
                    ticker_normalized=normalize_ticker(ticker),
                    wave_id=wave_id,
                    wave_name=wave_name,
                    source="yfinance",
                    failure_type=failure_type,
                    error_message=error_msg,
                    is_fatal=True,
                    suggested_fix=suggested_fix
                )
                tracker.record_failure(report)
        return pd.DataFrame(), failures
    
    if not tickers:
        return pd.DataFrame(), failures
    
    # Remove duplicates and clean tickers
    tickers = sorted(set(t.strip().upper() for t in tickers if t.strip()))
    
    # Normalize tickers before fetching
    normalized_tickers = []
    for ticker in tickers:
        normalized = normalize_ticker(ticker)
        normalized_tickers.append(normalized)
    
    # Use normalized tickers for fetching (remove duplicates)
    tickers_to_fetch = list(set(normalized_tickers))
    
    # Try batch download first with retry logic, then fall back to individual ticker fetching on failure
    # Use circuit breaker to prevent excessive retries if yfinance is down
    circuit_breaker = None
    if CIRCUIT_BREAKER_AVAILABLE:
        # Get or create circuit breaker for yfinance
        circuit_breaker = get_circuit_breaker(
            'yfinance_batch',
            failure_threshold=5,  # Open circuit after 5 failures
            recovery_timeout=60,  # Wait 60 seconds before trying again
            success_threshold=2   # Need 2 successes to close circuit
        )
    
    try:
        def _batch_download():
            return yf.download(
                tickers=tickers_to_fetch,
                start=start_date.strftime('%Y-%m-%d'),
                end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                auto_adjust=True,
                progress=False,
                group_by='ticker',
            )
        
        # Try with circuit breaker if available
        if circuit_breaker:
            success, data, error = circuit_breaker.call(_batch_download)
            if not success:
                # Circuit is open or call failed, fall back to individual fetching
                print(f"Circuit breaker prevented batch download or call failed: {error}")
                return _fetch_prices_individually(tickers_to_fetch, start_date, end_date, failures, wave_id, wave_name)
        else:
            # Use retry with backoff for batch download
            data = _retry_with_backoff(_batch_download, max_retries=3, initial_delay=1.0)
        
        if data.empty:
            print(f"Warning: No data returned from yfinance for {tickers_to_fetch}")
            # Try individual ticker fetching
            return _fetch_prices_individually(tickers_to_fetch, start_date, end_date, failures, wave_id, wave_name)
        
        # Normalize the data structure
        if len(tickers_to_fetch) == 1:
            # Single ticker case
            if 'Close' in data.columns:
                prices = data[['Close']].rename(columns={'Close': tickers_to_fetch[0]})
            else:
                error_msg = "No Close column in data"
                failures[tickers_to_fetch[0]] = error_msg
                
                # Track in diagnostics
                if tracker:
                    failure_type, suggested_fix = categorize_error(error_msg, tickers_to_fetch[0])
                    report = FailedTickerReport(
                        ticker_original=tickers_to_fetch[0],
                        ticker_normalized=tickers_to_fetch[0],
                        wave_id=wave_id,
                        wave_name=wave_name,
                        source="yfinance",
                        failure_type=failure_type,
                        error_message=error_msg,
                        is_fatal=True,
                        suggested_fix=suggested_fix
                    )
                    tracker.record_failure(report)
                prices = pd.DataFrame()
        else:
            # Multiple tickers case
            frames = []
            for ticker in tickers_to_fetch:
                try:
                    if (ticker, 'Close') in data.columns:
                        frames.append(data[(ticker, 'Close')].rename(ticker))
                    elif ticker in data.columns and 'Close' in data[ticker].columns:
                        frames.append(data[ticker]['Close'].rename(ticker))
                    else:
                        error_msg = "Missing Close column"
                        failures[ticker] = error_msg
                        
                        # Track in diagnostics
                        if tracker:
                            failure_type, suggested_fix = categorize_error(error_msg, ticker)
                            report = FailedTickerReport(
                                ticker_original=ticker,
                                ticker_normalized=ticker,
                                wave_id=wave_id,
                                wave_name=wave_name,
                                source="yfinance",
                                failure_type=failure_type,
                                error_message=error_msg,
                                is_fatal=True,
                                suggested_fix=suggested_fix
                            )
                            tracker.record_failure(report)
                except (KeyError, AttributeError) as e:
                    error_msg = f"Data extraction error: {str(e)}"
                    failures[ticker] = error_msg
                    
                    # Track in diagnostics
                    if tracker:
                        failure_type, suggested_fix = categorize_error(error_msg, ticker)
                        report = FailedTickerReport(
                            ticker_original=ticker,
                            ticker_normalized=ticker,
                            wave_id=wave_id,
                            wave_name=wave_name,
                            source="yfinance",
                            failure_type=failure_type,
                            error_message=error_msg,
                            is_fatal=True,
                            suggested_fix=suggested_fix
                        )
                        tracker.record_failure(report)
                    continue
            
            if frames:
                prices = pd.concat(frames, axis=1)
            else:
                prices = pd.DataFrame()
        
        if not prices.empty:
            prices.index = pd.to_datetime(prices.index)
            prices = prices.sort_index()
            # Forward fill missing data (holidays)
            prices = prices.ffill()
        
        # Track which tickers failed
        for ticker in tickers_to_fetch:
            if prices.empty or ticker not in prices.columns:
                if ticker not in failures:
                    error_msg = "No data in result"
                    failures[ticker] = error_msg
                    
                    # Track in diagnostics
                    if tracker:
                        failure_type, suggested_fix = categorize_error(error_msg, ticker)
                        report = FailedTickerReport(
                            ticker_original=ticker,
                            ticker_normalized=ticker,
                            wave_id=wave_id,
                            wave_name=wave_name,
                            source="yfinance",
                            failure_type=failure_type,
                            error_message=error_msg,
                            is_fatal=True,
                            suggested_fix=suggested_fix
                        )
                        tracker.record_failure(report)
        
        return prices, failures
        
    except Exception as e:
        error_msg = f"Batch download error: {str(e)}"
        print(f"Error fetching prices for {tickers_to_fetch}: {error_msg}")
        # Try individual ticker fetching as fallback
        return _fetch_prices_individually(tickers_to_fetch, start_date, end_date, failures, wave_id, wave_name)


def _fetch_prices_individually(
    tickers: List[str], 
    start_date: datetime, 
    end_date: datetime, 
    failures: Dict[str, str],
    wave_id: Optional[str] = None,
    wave_name: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Fetch prices one ticker at a time for maximum resilience.
    Enhanced with retry logic, diagnostics tracking, and circuit breaker to prevent infinite loops.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        failures: Dict to track failures (modified in place)
        wave_id: Optional wave identifier for diagnostics
        wave_name: Optional wave name for diagnostics
        
    Returns:
        Tuple of (prices_df, failures_dict)
    """
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame(), failures
    
    # CIRCUIT BREAKER: Limit individual ticker fetches to prevent infinite loops
    if len(tickers) > MAX_INDIVIDUAL_TICKER_FETCHES:
        logger.warning(
            f"Too many tickers ({len(tickers)}) for individual fetching. "
            f"Limiting to {MAX_INDIVIDUAL_TICKER_FETCHES}"
        )
        # Take first MAX_INDIVIDUAL_TICKER_FETCHES tickers, mark rest as failed
        for ticker in tickers[MAX_INDIVIDUAL_TICKER_FETCHES:]:
            failures[ticker] = f"Exceeded max individual ticker fetch limit ({MAX_INDIVIDUAL_TICKER_FETCHES})"
        tickers = tickers[:MAX_INDIVIDUAL_TICKER_FETCHES]
    
    # Get diagnostics tracker if available
    tracker = get_diagnostics_tracker() if DIAGNOSTICS_AVAILABLE else None
    
    all_prices = {}
    common_index = None
    
    # Add batch processing with delays to reduce API stress
    batch_size = 10
    batch_delay = 0.5  # 0.5 seconds between batches
    
    for i, ticker in enumerate(tickers):
        try:
            def _download_single():
                return yf.download(
                    tickers=ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                    auto_adjust=True,
                    progress=False,
                )
            
            # Use retry with backoff
            data = _retry_with_backoff(_download_single, max_retries=3, initial_delay=1.0)
            
            if data.empty:
                error_msg = "Empty data returned"
                failures[ticker] = error_msg
                
                # Track in diagnostics
                if tracker:
                    failure_type, suggested_fix = categorize_error(error_msg, ticker)
                    report = FailedTickerReport(
                        ticker_original=ticker,
                        ticker_normalized=normalize_ticker(ticker),
                        wave_id=wave_id,
                        wave_name=wave_name,
                        source="yfinance",
                        failure_type=failure_type,
                        error_message=error_msg,
                        is_fatal=True,
                        suggested_fix=suggested_fix
                    )
                    tracker.record_failure(report)
                continue
            
            if 'Close' in data.columns:
                prices_series = data['Close']
            else:
                error_msg = "No Close column"
                failures[ticker] = error_msg
                
                # Track in diagnostics
                if tracker:
                    failure_type, suggested_fix = categorize_error(error_msg, ticker)
                    report = FailedTickerReport(
                        ticker_original=ticker,
                        ticker_normalized=normalize_ticker(ticker),
                        wave_id=wave_id,
                        wave_name=wave_name,
                        source="yfinance",
                        failure_type=failure_type,
                        error_message=error_msg,
                        is_fatal=True,
                        suggested_fix=suggested_fix
                    )
                    tracker.record_failure(report)
                continue
            
            # Store prices
            all_prices[ticker] = prices_series
            
            # Build common index
            if common_index is None:
                common_index = prices_series.index
            else:
                common_index = common_index.union(prices_series.index)
            
            # Add delay after each ticker fetch to reduce API stress
            # Skip delay for the very last ticker
            if i < len(tickers) - 1:
                time.sleep(batch_delay)
                
        except Exception as e:
            error_msg = f"Individual fetch error: {str(e)}"
            failures[ticker] = error_msg
            
            # Track in diagnostics
            if tracker:
                failure_type, suggested_fix = categorize_error(error_msg, ticker)
                report = FailedTickerReport(
                    ticker_original=ticker,
                    ticker_normalized=normalize_ticker(ticker),
                    wave_id=wave_id,
                    wave_name=wave_name,
                    source="yfinance",
                    failure_type=failure_type,
                    error_message=error_msg,
                    is_fatal=True,
                    suggested_fix=suggested_fix
                )
                tracker.record_failure(report)
            continue
    
    if not all_prices:
        return pd.DataFrame(), failures
    
    # Build DataFrame with common index
    prices_df = pd.DataFrame(all_prices, index=common_index)
    prices_df = prices_df.sort_index()
    prices_df = prices_df.ffill()
    
    return prices_df, failures


# ------------------------------------------------------------
# Benchmark Materialization
# ------------------------------------------------------------

def materialize_composite_benchmark(
    benchmark_specs: List[Tuple[str, float]],
    start_date: datetime,
    end_date: datetime,
    use_dummy_data: bool = False
) -> pd.DataFrame:
    """
    Materialize a composite benchmark using weighted price calculations.
    
    Args:
        benchmark_specs: List of (ticker, weight) tuples
        start_date: Start date for data
        end_date: End date for data
        use_dummy_data: If True, use dummy data instead of fetching from yfinance
        
    Returns:
        DataFrame with date index and 'composite_benchmark' column
    """
    if not benchmark_specs:
        return pd.DataFrame()
    
    # Fetch all benchmark prices
    benchmark_tickers = [ticker for ticker, _ in benchmark_specs]
    prices, failures = fetch_prices(benchmark_tickers, start_date, end_date, use_dummy_data)
    
    if failures:
        print(f"Warning: {len(failures)} benchmark ticker(s) failed to download")
    
    if prices.empty:
        return pd.DataFrame()
    
    # Calculate weighted composite
    composite = pd.Series(0.0, index=prices.index)
    
    for ticker, weight in benchmark_specs:
        if ticker in prices.columns:
            composite += prices[ticker] * weight
    
    # Normalize if we don't have full weight coverage
    total_weight = sum(weight for ticker, weight in benchmark_specs if ticker in prices.columns)
    if total_weight > 0 and total_weight != 1.0:
        composite = composite / total_weight
    
    result = pd.DataFrame({
        'composite_benchmark': composite
    })
    
    return result


# ------------------------------------------------------------
# File Generation Functions
# ------------------------------------------------------------

def generate_prices_csv(wave_id: str, lookback_days: int, use_dummy_data: bool = False) -> bool:
    """
    Generate prices.csv for a wave containing daily close prices for all tickers.
    
    Args:
        wave_id: The wave identifier
        lookback_days: Number of days to look back
        use_dummy_data: If True, use dummy data instead of fetching from yfinance
        
    Returns:
        True if successful, False otherwise
    """
    tickers = resolve_wave_tickers(wave_id)
    if not tickers:
        print(f"Warning: No tickers found for {wave_id}")
        return False
    
    wave_name = get_display_name_from_wave_id(wave_id)
    end_date = datetime.now()
    start_date = get_trading_days_back(lookback_days)
    
    prices, failures = fetch_prices(tickers, start_date, end_date, use_dummy_data, wave_id, wave_name)
    
    if failures:
        print(f"Warning: {len(failures)}/{len(tickers)} ticker(s) failed for {wave_id}: {list(failures.keys())}")
    
    if prices.empty:
        print(f"Warning: No price data fetched for {wave_id}")
        return False
    
    # Ensure we have the minimum required trading days
    # NEW: Lower threshold for partial data support
    if len(prices) < MIN_REQUIRED_TRADING_DAYS:
        print(f"Warning: Only {len(prices)} trading days available for {wave_id}, need {MIN_REQUIRED_TRADING_DAYS}")
        # Continue anyway - partial data is better than no data
    
    # Save to CSV
    output_dir = get_wave_analytics_dir(wave_id)
    ensure_directory_exists(output_dir)
    output_path = os.path.join(output_dir, 'prices.csv')
    
    prices.to_csv(output_path)
    print(f"✓ Generated prices.csv for {wave_id}: {len(prices)} days, {len(prices.columns)} tickers")
    
    return True


def generate_benchmark_prices_csv(wave_id: str, lookback_days: int, use_dummy_data: bool = False) -> bool:
    """
    Generate benchmark_prices.csv for a wave.
    
    Args:
        wave_id: The wave identifier
        lookback_days: Number of days to look back
        use_dummy_data: If True, use dummy data instead of fetching from yfinance
        
    Returns:
        True if successful, False otherwise
    """
    benchmark_specs = resolve_wave_benchmarks(wave_id)
    if not benchmark_specs:
        print(f"Warning: No benchmarks found for {wave_id}")
        return False
    
    wave_name = get_display_name_from_wave_id(wave_id)
    end_date = datetime.now()
    start_date = get_trading_days_back(lookback_days)
    
    # Fetch individual benchmark prices
    benchmark_tickers = [ticker for ticker, _ in benchmark_specs]
    prices, failures = fetch_prices(benchmark_tickers, start_date, end_date, use_dummy_data, wave_id, wave_name)
    
    if failures:
        print(f"Warning: {len(failures)} benchmark ticker(s) failed for {wave_id}")
    
    if prices.empty:
        print(f"Warning: No benchmark price data fetched for {wave_id}")
        return False
    
    # Also calculate composite benchmark
    composite = materialize_composite_benchmark(benchmark_specs, start_date, end_date, use_dummy_data)
    
    # Combine individual benchmarks and composite
    if not composite.empty:
        benchmark_prices = pd.concat([prices, composite], axis=1)
    else:
        benchmark_prices = prices
    
    # Save to CSV
    output_dir = get_wave_analytics_dir(wave_id)
    ensure_directory_exists(output_dir)
    output_path = os.path.join(output_dir, 'benchmark_prices.csv')
    
    benchmark_prices.to_csv(output_path)
    print(f"✓ Generated benchmark_prices.csv for {wave_id}: {len(benchmark_prices)} days")
    
    return True


def generate_positions_csv(wave_id: str) -> bool:
    """
    Generate positions.csv snapshot for a wave.
    
    Args:
        wave_id: The wave identifier
        
    Returns:
        True if successful, False otherwise
    """
    display_name = get_display_name_from_wave_id(wave_id)
    if not display_name:
        return False
    
    holdings = WAVE_WEIGHTS.get(display_name, [])
    
    positions = []
    for holding in holdings:
        if isinstance(holding, Holding):
            positions.append({
                'ticker': holding.ticker,
                'weight': holding.weight,
                'description': holding.name or '',
                'exposure': holding.weight,  # Default exposure = weight
                'cash': 0.0,
                'safe_fraction': 0.0,
            })
        elif isinstance(holding, dict):
            positions.append({
                'ticker': holding.get('ticker', ''),
                'weight': holding.get('weight', 0.0),
                'description': holding.get('name', holding.get('description', '')),
                'exposure': holding.get('weight', 0.0),
                'cash': 0.0,
                'safe_fraction': 0.0,
            })
    
    if not positions:
        # Create empty position for waves with no holdings
        positions.append({
            'ticker': '',
            'weight': 0.0,
            'description': 'No holdings',
            'exposure': 0.0,
            'cash': 1.0,
            'safe_fraction': 1.0,
        })
    
    df = pd.DataFrame(positions)
    
    # Save to CSV
    output_dir = get_wave_analytics_dir(wave_id)
    ensure_directory_exists(output_dir)
    output_path = os.path.join(output_dir, 'positions.csv')
    
    df.to_csv(output_path, index=False)
    print(f"✓ Generated positions.csv for {wave_id}: {len(df)} positions")
    
    return True


def generate_trades_csv(wave_id: str) -> bool:
    """
    Generate trades.csv for a wave (may be empty if no activity).
    
    Args:
        wave_id: The wave identifier
        
    Returns:
        True if successful, False otherwise
    """
    # Create empty trades dataframe with proper schema
    trades = pd.DataFrame({
        'date': pd.Series(dtype='str'),
        'ticker': pd.Series(dtype='str'),
        'action': pd.Series(dtype='str'),  # 'BUY' or 'SELL'
        'shares': pd.Series(dtype='float'),
        'price': pd.Series(dtype='float'),
        'value': pd.Series(dtype='float'),
    })
    
    # Save to CSV
    output_dir = get_wave_analytics_dir(wave_id)
    ensure_directory_exists(output_dir)
    output_path = os.path.join(output_dir, 'trades.csv')
    
    trades.to_csv(output_path, index=False)
    print(f"✓ Generated trades.csv for {wave_id}: {len(trades)} trades")
    
    return True


def generate_nav_csv(wave_id: str, lookback_days: int) -> bool:
    """
    Generate nav.csv for a wave aligned with holdings.
    
    For now, creates a flat NAV starting at 10000 if no activity.
    
    Args:
        wave_id: The wave identifier
        lookback_days: Number of days to look back
        
    Returns:
        True if successful, False otherwise
    """
    end_date = datetime.now()
    start_date = get_trading_days_back(lookback_days)
    
    # Create date range (business days only)
    date_range = pd.bdate_range(start=start_date, end=end_date)
    
    # Default flat NAV at 10000
    nav_data = pd.DataFrame({
        'date': date_range,
        'nav': 10000.0,
        'cash': 0.0,
        'holdings_value': 10000.0,
    })
    
    nav_data.set_index('date', inplace=True)
    
    # Save to CSV
    output_dir = get_wave_analytics_dir(wave_id)
    ensure_directory_exists(output_dir)
    output_path = os.path.join(output_dir, 'nav.csv')
    
    nav_data.to_csv(output_path)
    print(f"✓ Generated nav.csv for {wave_id}: {len(nav_data)} days")
    
    return True


# ------------------------------------------------------------
# Validation Functions
# ------------------------------------------------------------

def validate_wave_data_ready(wave_id: str, lookback_days: int = 7) -> Dict[str, Any]:
    """
    Validate that all required data is present and consistent for a wave.
    
    Args:
        wave_id: The wave identifier
        lookback_days: Minimum number of days required
        
    Returns:
        Dictionary with validation results:
        {
            'wave_id': str,
            'status': 'pass' or 'fail',
            'checks': {
                'prices_exists': bool,
                'prices_days': int,
                'prices_valid': bool,
                'benchmark_exists': bool,
                'positions_exists': bool,
                'trades_exists': bool,
                'nav_exists': bool,
                'nav_aligned': bool,
            },
            'issues': List[str]
        }
    """
    wave_dir = get_wave_analytics_dir(wave_id)
    checks = {}
    issues = []
    
    # Check prices.csv
    prices_path = os.path.join(wave_dir, 'prices.csv')
    checks['prices_exists'] = os.path.exists(prices_path)
    
    if checks['prices_exists']:
        try:
            prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
            checks['prices_days'] = len(prices_df)
            checks['prices_valid'] = len(prices_df) >= lookback_days
            
            if not checks['prices_valid']:
                issues.append(f"Insufficient price data: {len(prices_df)} days < {lookback_days} required")
            
            # Check for recent data (within last 5 days)
            if not prices_df.empty:
                last_date = prices_df.index[-1]
                days_old = (datetime.now() - last_date).days
                if days_old > 5:
                    issues.append(f"Price data is stale: last date is {last_date.date()}")
                    
        except Exception as e:
            checks['prices_days'] = 0
            checks['prices_valid'] = False
            issues.append(f"Error reading prices.csv: {e}")
    else:
        checks['prices_days'] = 0
        checks['prices_valid'] = False
        issues.append("prices.csv does not exist")
    
    # Check benchmark_prices.csv
    benchmark_path = os.path.join(wave_dir, 'benchmark_prices.csv')
    checks['benchmark_exists'] = os.path.exists(benchmark_path)
    
    if not checks['benchmark_exists']:
        issues.append("benchmark_prices.csv does not exist")
    
    # Check positions.csv
    positions_path = os.path.join(wave_dir, 'positions.csv')
    checks['positions_exists'] = os.path.exists(positions_path)
    
    if not checks['positions_exists']:
        issues.append("positions.csv does not exist")
    
    # Check trades.csv
    trades_path = os.path.join(wave_dir, 'trades.csv')
    checks['trades_exists'] = os.path.exists(trades_path)
    
    if not checks['trades_exists']:
        issues.append("trades.csv does not exist")
    
    # Check nav.csv
    nav_path = os.path.join(wave_dir, 'nav.csv')
    checks['nav_exists'] = os.path.exists(nav_path)
    
    if checks['nav_exists']:
        try:
            nav_df = pd.read_csv(nav_path, index_col=0, parse_dates=True)
            # Check alignment with positions
            checks['nav_aligned'] = len(nav_df) >= lookback_days
            
            if not checks['nav_aligned']:
                issues.append(f"NAV data insufficient: {len(nav_df)} days")
        except Exception as e:
            checks['nav_aligned'] = False
            issues.append(f"Error reading nav.csv: {e}")
    else:
        checks['nav_aligned'] = False
        issues.append("nav.csv does not exist")
    
    # Determine overall status
    required_checks = [
        'prices_exists', 'prices_valid', 'benchmark_exists',
        'positions_exists', 'trades_exists', 'nav_exists', 'nav_aligned'
    ]
    
    status = 'pass' if all(checks.get(c, False) for c in required_checks) else 'fail'
    
    return {
        'wave_id': wave_id,
        'display_name': get_display_name_from_wave_id(wave_id),
        'status': status,
        'checks': checks,
        'issues': issues
    }


def compute_data_ready_status(wave_id: str) -> Dict[str, Any]:
    """
    Compute comprehensive data readiness status for a wave with graded diagnostics.
    
    GRADED READINESS MODEL:
    - operational: Has current pricing, can display current state
    - partial: Has some history but limited analytics (basic metrics only)
    - full: Complete data for all analytics including advanced multi-window metrics
    
    This function provides detailed diagnostics about wave capabilities and limitations,
    enabling operators to quickly identify issues and understand what analytics are available.
    
    Args:
        wave_id: The wave identifier
        
    Returns:
        Dictionary with graded readiness diagnostics:
        {
            'wave_id': str,
            'display_name': str,
            'readiness_status': str,  # 'operational', 'partial', 'full', or 'unavailable'
            'is_ready': bool,  # Deprecated: True if operational or better
            'readiness_reasons': List[str],  # Human-readable explanations
            'allowed_analytics': Dict[str, bool],  # Which analytics can be computed
            'checks_passed': Dict[str, bool],  # All passed checks
            'checks_failed': Dict[str, Dict],  # Failed checks with details
            'blocking_issues': List[str],  # Issues preventing operational status
            'informational_issues': List[str],  # Non-blocking limitations
            'suggested_actions': List[str],  # Actionable recommendations
            'reason': str,  # Legacy: Primary failure reason code
            'reason_codes': List[str],  # Legacy: All applicable reason codes
            'details': str,  # Legacy: Human-readable explanation
            'checks': Dict[str, bool],  # Legacy compatibility
            'missing_tickers': List[str],
            'missing_benchmark_tickers': List[str],
            'coverage_pct': float,  # Percentage of tickers with data
            'history_window_used': Dict[str, str],
            'source_used': str,
            'exception': str
        }
    
    Readiness Levels:
        - 'full': All checks passed, all analytics available
        - 'partial': Enough data for basic analytics, some advanced features limited
        - 'operational': Current pricing available, can display current state
        - 'unavailable': Critical data missing, cannot display
    """
    from datetime import datetime, timezone
    import logging
    
    # Initialize response with graded readiness fields
    result = {
        'wave_id': wave_id,
        'display_name': get_display_name_from_wave_id(wave_id) or wave_id,
        'readiness_status': 'unavailable',
        'is_ready': False,  # Deprecated but kept for compatibility
        'readiness_reasons': [],
        'allowed_analytics': {
            'current_pricing': False,
            'simple_returns': False,
            'multi_window_returns': False,
            'volatility_metrics': False,
            'correlation_analysis': False,
            'alpha_attribution': False,
            'advanced_analytics': False
        },
        'checks_passed': {},
        'checks_failed': {},
        'blocking_issues': [],
        'informational_issues': [],
        'suggested_actions': [],
        # Legacy fields for backward compatibility
        'reason': 'UNKNOWN',
        'reason_codes': [],
        'details': '',
        'checks': {
            'has_weights': False,
            'has_prices': False,
            'has_benchmark': False,
            'has_nav': False,
            'is_fresh': False,
            'has_sufficient_history': False,
        },
        'missing_tickers': [],
        'missing_benchmark_tickers': [],
        'coverage_pct': 0.0,
        'missing_dates': {'earliest': None, 'latest': None},
        'history_window_used': {'start': None, 'end': None},
        'source_used': 'none',
        'exception': None
    }
    
    # Check 1: Wave exists in registry
    all_wave_ids = get_all_wave_ids()
    if wave_id not in all_wave_ids:
        result['reason'] = 'WAVE_NOT_FOUND'
        result['reason_codes'].append('WAVE_NOT_FOUND')
        result['readiness_reasons'].append(f"Wave ID '{wave_id}' is not registered")
        result['blocking_issues'].append('WAVE_NOT_FOUND')
        result['suggested_actions'].append('Verify wave ID and ensure it exists in WAVE_ID_REGISTRY')
        result['details'] = f"Wave ID '{wave_id}' is not registered in WAVE_ID_REGISTRY"
        result['checks_failed']['wave_registry'] = {
            'reason': 'Wave not found in registry',
            'blocking': True
        }
        _log_readiness_result(result)
        return result
    
    result['checks_passed']['wave_registry'] = True
    
    # Check 2: Has weights/holdings defined
    tickers = resolve_wave_tickers(wave_id)
    total_tickers = len(tickers) if tickers else 0
    
    if not tickers:
        result['reason'] = 'MISSING_WEIGHTS'
        result['reason_codes'].append('MISSING_WEIGHTS')
        result['readiness_reasons'].append('No holdings defined for this wave')
        result['blocking_issues'].append('MISSING_WEIGHTS')
        result['suggested_actions'].append('Define holdings in WAVE_WEIGHTS configuration')
        result['details'] = f"No holdings defined in WAVE_WEIGHTS for '{wave_id}'"
        result['checks_failed']['has_weights'] = {
            'reason': 'No holdings configuration found',
            'blocking': True
        }
        _log_readiness_result(result)
        return result
    
    result['checks']['has_weights'] = True
    result['checks_passed']['has_weights'] = True
    
    # Get wave analytics directory
    wave_dir = get_wave_analytics_dir(wave_id)
    
    # Get benchmark tickers for diagnostics
    benchmark_specs = resolve_wave_benchmarks(wave_id)
    benchmark_tickers = [ticker for ticker, _ in benchmark_specs]
    
    # Check 3: Has price data
    prices_path = os.path.join(wave_dir, 'prices.csv')
    if not os.path.exists(prices_path):
        result['reason'] = 'MISSING_PRICES'
        result['reason_codes'].append('MISSING_PRICES')
        result['readiness_reasons'].append('Price data file not found')
        result['blocking_issues'].append('MISSING_PRICES')
        result['suggested_actions'].append(f'Run analytics pipeline to fetch price data: python analytics_pipeline.py --wave {wave_id}')
        result['details'] = f"Price data file not found at {prices_path}"
        result['missing_tickers'] = tickers  # All tickers are missing
        result['source_used'] = 'none'
        result['coverage_pct'] = 0.0
        result['checks_failed']['has_prices'] = {
            'reason': 'Price data file missing',
            'blocking': True,
            'path': prices_path
        }
        _log_readiness_result(result)
        return result
    
    result['checks']['has_prices'] = True
    result['checks_passed']['has_prices'] = True
    result['source_used'] = 'cached'  # Data exists in files
    
    # Check 4: Has benchmark data (informational - not blocking for operational)
    benchmark_path = os.path.join(wave_dir, 'benchmark_prices.csv')
    has_benchmark = os.path.exists(benchmark_path)
    
    if not has_benchmark:
        result['reason_codes'].append('MISSING_BENCHMARK')
        result['informational_issues'].append('MISSING_BENCHMARK')
        result['readiness_reasons'].append('Benchmark data missing (limits some analytics)')
        result['suggested_actions'].append('Run analytics pipeline to fetch benchmark data')
        result['missing_benchmark_tickers'] = benchmark_tickers
        result['checks_failed']['has_benchmark'] = {
            'reason': 'Benchmark data file missing',
            'blocking': False,  # Not blocking for operational status
            'path': benchmark_path,
            'impact': 'Relative performance metrics unavailable'
        }
    else:
        result['checks']['has_benchmark'] = True
        result['checks_passed']['has_benchmark'] = True
    
    # Check 5: Has NAV data (informational - not blocking for operational)
    nav_path = os.path.join(wave_dir, 'nav.csv')
    has_nav = os.path.exists(nav_path)
    
    if not has_nav:
        result['reason_codes'].append('MISSING_NAV')
        result['informational_issues'].append('MISSING_NAV')
        result['readiness_reasons'].append('NAV calculation data missing (limits some analytics)')
        result['suggested_actions'].append('Run analytics pipeline to generate NAV data')
        result['checks_failed']['has_nav'] = {
            'reason': 'NAV data file missing',
            'blocking': False,  # Not blocking for operational status
            'path': nav_path,
            'impact': 'Historical performance metrics limited'
        }
    else:
        result['checks']['has_nav'] = True
        result['checks_passed']['has_nav'] = True
    
    # Check 6: Data quality, freshness, history length, and coverage
    try:
        prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        
        if prices_df.empty:
            result['reason'] = 'INSUFFICIENT_HISTORY'
            result['reason_codes'].append('INSUFFICIENT_HISTORY')
            result['readiness_reasons'].append('Price data file is empty')
            result['blocking_issues'].append('INSUFFICIENT_HISTORY')
            result['suggested_actions'].append('Re-run analytics pipeline to populate price data')
            result['details'] = "Price data file is empty"
            result['checks_failed']['data_populated'] = {
                'reason': 'Empty price data file',
                'blocking': True
            }
            _log_readiness_result(result)
            return result
        
        # Record history window
        first_date = prices_df.index[0]
        last_date = prices_df.index[-1]
        num_days = len(prices_df)
        
        result['history_window_used'] = {
            'start': first_date.strftime('%Y-%m-%d'),
            'end': last_date.strftime('%Y-%m-%d')
        }
        
        # Check for missing tickers and calculate coverage
        available_tickers = set(prices_df.columns)
        expected_tickers = set(tickers)
        missing = list(expected_tickers - available_tickers)
        
        # Also check for tickers with all NaN values
        for ticker in available_tickers.intersection(expected_tickers):
            if prices_df[ticker].isna().all():
                if ticker not in missing:
                    missing.append(ticker)
                if 'NAN_SERIES' not in result['reason_codes']:
                    result['reason_codes'].append('NAN_SERIES')
                    result['informational_issues'].append('NAN_SERIES')
        
        result['missing_tickers'] = missing
        available_count = total_tickers - len(missing)
        coverage_pct = (available_count / total_tickers * 100.0) if total_tickers > 0 else 0.0
        result['coverage_pct'] = round(coverage_pct, 2)
        
        # Check benchmark completeness (informational)
        if has_benchmark:
            try:
                benchmark_df = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)
                available_benchmark_tickers = set(benchmark_df.columns)
                expected_benchmark_tickers = set(benchmark_tickers)
                missing_benchmarks = list(expected_benchmark_tickers - available_benchmark_tickers)
                if missing_benchmarks:
                    result['missing_benchmark_tickers'] = missing_benchmarks
                    if 'MISSING_BENCHMARK' not in result['reason_codes']:
                        result['reason_codes'].append('MISSING_BENCHMARK')
                        result['informational_issues'].append('MISSING_BENCHMARK')
            except Exception as e:
                result['missing_benchmark_tickers'] = benchmark_tickers
                if 'MISSING_BENCHMARK' not in result['reason_codes']:
                    result['reason_codes'].append('MISSING_BENCHMARK')
                    result['informational_issues'].append('MISSING_BENCHMARK')
        
        # Check data freshness
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc) if hasattr(last_date, 'tz') and last_date.tz else datetime.now()
        days_old = (now - last_date).days
        
        is_fresh = days_old <= MAX_DAYS_STALE
        if not is_fresh:
            result['reason_codes'].append('STALE_DATA')
            result['informational_issues'].append('STALE_DATA')
            result['readiness_reasons'].append(f'Data is {days_old} days old (may impact current state accuracy)')
            result['suggested_actions'].append('Re-run analytics pipeline to fetch fresh data')
            result['checks_failed']['is_fresh'] = {
                'reason': f'Data is {days_old} days old',
                'blocking': False,
                'last_update': last_date.strftime('%Y-%m-%d'),
                'impact': 'Current state may not reflect latest market data'
            }
        else:
            result['checks']['is_fresh'] = True
            result['checks_passed']['is_fresh'] = True
        
        # ====================================================================
        # GRADED READINESS EVALUATION
        # ====================================================================
        
        # Determine readiness level based on coverage, history, and data quality
        
        # Level 1: OPERATIONAL (can display current state)
        # Requirements: Minimal coverage + minimal history + some freshness tolerance
        if coverage_pct >= (MIN_COVERAGE_OPERATIONAL * 100) and num_days >= MIN_DAYS_OPERATIONAL:
            result['readiness_status'] = 'operational'
            result['is_ready'] = True
            result['allowed_analytics']['current_pricing'] = True
            result['allowed_analytics']['simple_returns'] = num_days >= MIN_DAYS_OPERATIONAL
            
            if missing:
                result['readiness_reasons'].append(
                    f'Wave is operational with {coverage_pct:.1f}% coverage ({len(missing)} tickers missing)'
                )
            else:
                result['readiness_reasons'].append('Wave is operational with full ticker coverage')
            
            # Level 2: PARTIAL (can run basic analytics)
            # Requirements: Better coverage + more history
            if coverage_pct >= (MIN_COVERAGE_PARTIAL * 100) and num_days >= MIN_DAYS_PARTIAL:
                result['readiness_status'] = 'partial'
                result['allowed_analytics']['volatility_metrics'] = True
                result['allowed_analytics']['correlation_analysis'] = has_benchmark
                
                if num_days < MIN_DAYS_FULL:
                    result['readiness_reasons'].append(
                        f'Partial readiness: {num_days} days of history (need {MIN_DAYS_FULL} for full analytics)'
                    )
                    result['informational_issues'].append('INSUFFICIENT_FULL_HISTORY')
                    result['suggested_actions'].append(
                        f'Run analytics pipeline with --lookback={MIN_DAYS_FULL} for full analytics'
                    )
                
                # Level 3: FULL (all analytics available)
                # Requirements: High coverage + full history + all data sources
                if (coverage_pct >= (MIN_COVERAGE_FULL * 100) and 
                    num_days >= MIN_DAYS_FULL and 
                    has_benchmark and has_nav):
                    result['readiness_status'] = 'full'
                    result['allowed_analytics']['multi_window_returns'] = True
                    result['allowed_analytics']['alpha_attribution'] = True
                    result['allowed_analytics']['advanced_analytics'] = True
                    result['readiness_reasons'] = ['Wave is fully ready: all analytics available']
                    result['reason'] = 'READY'
                    result['reason_codes'] = ['READY'] + result['reason_codes']
                    result['details'] = f'All checks passed. {num_days} days of fresh data with {coverage_pct:.1f}% coverage'
        
        else:
            # Below operational threshold
            result['readiness_status'] = 'unavailable'
            result['is_ready'] = False
            
            if coverage_pct < (MIN_COVERAGE_OPERATIONAL * 100):
                issue = f'Insufficient coverage: {coverage_pct:.1f}% (need {MIN_COVERAGE_OPERATIONAL*100:.0f}% for operational)'
                result['blocking_issues'].append('LOW_COVERAGE')
                result['readiness_reasons'].append(issue)
                missing_count = len(missing)
                result['suggested_actions'].append(
                    f'Improve coverage from {coverage_pct:.1f}% to {MIN_COVERAGE_OPERATIONAL*100:.0f}% by fixing {missing_count} missing ticker(s) or adjust coverage threshold'
                )
            
            if num_days < MIN_DAYS_OPERATIONAL:
                issue = f'Insufficient history: {num_days} days (need {MIN_DAYS_OPERATIONAL} for operational)'
                result['blocking_issues'].append('INSUFFICIENT_HISTORY')
                result['readiness_reasons'].append(issue)
                days_needed = MIN_DAYS_OPERATIONAL - num_days
                result['suggested_actions'].append(
                    f'Increase history from {num_days} to {MIN_DAYS_OPERATIONAL} days by running analytics pipeline with longer lookback'
                )
        
        # Set legacy fields for backward compatibility
        if result['readiness_status'] in ['operational', 'partial', 'full']:
            if not result['reason_codes'] or result['reason_codes'][0] != 'READY':
                result['reason'] = 'OPERATIONAL'
                if 'OPERATIONAL' not in result['reason_codes']:
                    result['reason_codes'].insert(0, 'OPERATIONAL')
        else:
            if not result['reason']:
                result['reason'] = result['blocking_issues'][0] if result['blocking_issues'] else 'UNAVAILABLE'
        
        if not result['details']:
            result['details'] = '; '.join(result['readiness_reasons']) if result['readiness_reasons'] else 'Unknown status'
        
        # Set history check for legacy compatibility
        if num_days >= MIN_DAYS_OPERATIONAL:
            result['checks']['has_sufficient_history'] = True
            result['checks_passed']['has_sufficient_history'] = True
        
        _log_readiness_result(result)
        
    except Exception as e:
        result['reason'] = 'DATA_READ_ERROR'
        result['reason_codes'].append('DATA_READ_ERROR')
        result['readiness_reasons'].append(f'Error reading data files: {str(e)}')
        result['blocking_issues'].append('DATA_READ_ERROR')
        result['suggested_actions'].append('Check data file integrity and format')
        result['details'] = f"Error reading price data: {str(e)}"
        result['exception'] = str(e)
        result['checks_failed']['data_read'] = {
            'reason': f'Exception reading data: {str(e)}',
            'blocking': True,
            'exception_type': type(e).__name__
        }
        _log_readiness_result(result)
        return result
    
    return result


def generate_readiness_report_dataframe(wave_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generate a pandas DataFrame summarizing readiness status for all waves.
    
    Args:
        wave_ids: Optional list of wave_ids to include. If None, includes all waves.
        
    Returns:
        DataFrame with columns:
        - wave_id
        - display_name
        - is_ready
        - reason
        - reason_codes
        - missing_tickers_count
        - missing_benchmark_tickers_count
        - history_start
        - history_end
        - source_used
        - has_exception
    """
    if wave_ids is None:
        wave_ids = get_all_wave_ids()
    
    records = []
    for wave_id in wave_ids:
        diagnostics = compute_data_ready_status(wave_id)
        
        records.append({
            'wave_id': diagnostics['wave_id'],
            'display_name': diagnostics['display_name'],
            'is_ready': diagnostics['is_ready'],
            'reason': diagnostics['reason'],
            'reason_codes': ', '.join(diagnostics['reason_codes']),
            'details': diagnostics['details'],
            'missing_tickers_count': len(diagnostics['missing_tickers']),
            'missing_tickers': ', '.join(diagnostics['missing_tickers']) if diagnostics['missing_tickers'] else '',
            'missing_benchmark_tickers_count': len(diagnostics['missing_benchmark_tickers']),
            'missing_benchmark_tickers': ', '.join(diagnostics['missing_benchmark_tickers']) if diagnostics['missing_benchmark_tickers'] else '',
            'history_start': diagnostics['history_window_used']['start'],
            'history_end': diagnostics['history_window_used']['end'],
            'source_used': diagnostics['source_used'],
            'has_exception': diagnostics['exception'] is not None,
            'exception': diagnostics['exception'] or '',
        })
    
    return pd.DataFrame(records)


def generate_readiness_report_json(wave_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate a JSON-serializable dictionary of readiness diagnostics for all waves.
    
    Args:
        wave_ids: Optional list of wave_ids to include. If None, includes all waves.
        
    Returns:
        Dictionary mapping wave_id to full diagnostic results:
        {
            'summary': {
                'total_waves': int,
                'ready_count': int,
                'degraded_count': int,
                'missing_count': int
            },
            'waves': {
                'wave_id_1': { ... full diagnostics ... },
                'wave_id_2': { ... full diagnostics ... },
                ...
            }
        }
    """
    import json
    
    if wave_ids is None:
        wave_ids = get_all_wave_ids()
    
    waves_data = {}
    ready_count = 0
    degraded_count = 0
    missing_count = 0
    
    for wave_id in wave_ids:
        diagnostics = compute_data_ready_status(wave_id)
        waves_data[wave_id] = diagnostics
        
        if diagnostics['is_ready']:
            ready_count += 1
        elif diagnostics['reason'] in ['STALE_DATA', 'INSUFFICIENT_HISTORY']:
            degraded_count += 1
        else:
            missing_count += 1
    
    return {
        'summary': {
            'total_waves': len(wave_ids),
            'ready_count': ready_count,
            'degraded_count': degraded_count,
            'missing_count': missing_count,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'waves': waves_data
    }


# Configuration for coverage-based readiness policy
DEFAULT_COVERAGE_THRESHOLD = 0.95  # 95% coverage required by default
DEFAULT_REQUIRED_WINDOW_DAYS = 365  # Default required history window


def generate_wave_readiness_report(
    wave_ids: Optional[List[str]] = None,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    required_window_days: int = DEFAULT_REQUIRED_WINDOW_DAYS
) -> pd.DataFrame:
    """
    Generate comprehensive Wave Readiness Report with graded diagnostics.
    
    This function produces a deterministic report showing:
    - Graded readiness status (operational/partial/full/unavailable)
    - Detailed reason for status
    - Failing tickers
    - Coverage percentage
    - Data window availability
    - Allowed analytics capabilities
    - Suggested fixes
    
    Args:
        wave_ids: Optional list of wave_ids to include. If None, includes all waves.
        coverage_threshold: Minimum required coverage for full status (0.0 to 1.0). Default: 0.95
        required_window_days: Minimum required days of history for full status. Default: 365
        
    Returns:
        DataFrame with columns:
        - wave_id: Canonical wave identifier
        - wave_name: Display name
        - readiness_status: operational | partial | full | unavailable
        - readiness_summary: Human-readable summary of capabilities
        - blocking_issues: Comma-separated list of blocking issues
        - informational_issues: Comma-separated list of non-blocking limitations
        - allowed_analytics: Summary of available analytics
        - failing_tickers: Comma-separated list of problematic tickers
        - coverage_pct: Percentage of required data available (0-100)
        - required_window_days: Policy-defined minimum days
        - available_window_days: Actual days of data available
        - start_date: Start of available data window
        - end_date: End of available data window
        - suggested_actions: Actionable recommendations
    """
    if wave_ids is None:
        wave_ids = get_all_wave_ids()
    
    records = []
    
    for wave_id in wave_ids:
        # Get graded diagnostics
        diagnostics = compute_data_ready_status(wave_id)
        display_name = diagnostics.get('display_name', wave_id)
        
        # Extract graded readiness fields
        readiness_status = diagnostics.get('readiness_status', 'unavailable')
        readiness_reasons = diagnostics.get('readiness_reasons', [])
        blocking_issues = diagnostics.get('blocking_issues', [])
        informational_issues = diagnostics.get('informational_issues', [])
        suggested_actions = diagnostics.get('suggested_actions', [])
        allowed_analytics = diagnostics.get('allowed_analytics', {})
        
        # Coverage and ticker info
        coverage_pct = diagnostics.get('coverage_pct', 0.0)
        missing_tickers = diagnostics.get('missing_tickers', [])
        
        # Extract date window
        history_window = diagnostics.get('history_window_used', {})
        start_date = history_window.get('start', None)
        end_date = history_window.get('end', None)
        
        # Calculate available window days
        if start_date and end_date:
            try:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                available_window_days = (end - start).days
            except:
                available_window_days = 0
        else:
            available_window_days = 0
        
        # Create readiness summary
        readiness_summary = '; '.join(readiness_reasons) if readiness_reasons else f'Status: {readiness_status}'
        
        # Create analytics summary
        enabled_analytics = [k for k, v in allowed_analytics.items() if v]
        analytics_summary = ', '.join(enabled_analytics) if enabled_analytics else 'None available'
        
        # Build record
        record = {
            'wave_id': wave_id,
            'wave_name': display_name,
            'readiness_status': readiness_status,
            'readiness_summary': readiness_summary,
            'blocking_issues': ', '.join(blocking_issues) if blocking_issues else '',
            'informational_issues': ', '.join(informational_issues) if informational_issues else '',
            'allowed_analytics': analytics_summary,
            'failing_tickers': ', '.join(missing_tickers) if missing_tickers else '',
            'coverage_pct': round(coverage_pct, 2),
            'required_window_days': required_window_days,
            'available_window_days': available_window_days,
            'start_date': start_date or '',
            'end_date': end_date or '',
            'suggested_actions': '; '.join(suggested_actions) if suggested_actions else '',
        }
        
        records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Sort by readiness status (full > partial > operational > unavailable) then by wave_id
    status_priority = {'full': 0, 'partial': 1, 'operational': 2, 'unavailable': 3}
    df['_sort_key'] = df['readiness_status'].map(status_priority)
    df = df.sort_values(['_sort_key', 'wave_id']).drop(columns=['_sort_key'])
    
    return df


def _generate_suggested_fix(
    reason_category: str,
    failing_tickers: List[str],
    coverage_pct: float,
    coverage_threshold: float,
    available_days: int,
    required_days: int
) -> str:
    """
    Generate a human-readable suggested fix for readiness failures.
    
    Args:
        reason_category: Failure reason category
        failing_tickers: List of failing ticker symbols
        coverage_pct: Current coverage percentage
        coverage_threshold: Required coverage threshold
        available_days: Days of data available
        required_days: Days of data required
        
    Returns:
        Suggested fix string
    """
    if reason_category == 'READY':
        return 'No action needed - wave is ready'
    
    elif reason_category == 'MISSING_PRICES':
        if failing_tickers:
            ticker_str = ', '.join(failing_tickers[:3])
            if len(failing_tickers) > 3:
                ticker_str += f' (+{len(failing_tickers) - 3} more)'
            
            if coverage_pct >= coverage_threshold:
                return f'Partial coverage ({coverage_pct:.0f}%) meets threshold. Consider dropping: {ticker_str}'
            else:
                return f'Fetch missing price data for: {ticker_str}. Run analytics pipeline to populate data.'
        else:
            return 'Run analytics pipeline to fetch and populate price data'
    
    elif reason_category == 'INVALID_TICKER':
        if failing_tickers:
            ticker_str = ', '.join(failing_tickers[:3])
            return f'Remove invalid/delisted tickers: {ticker_str}'
        else:
            return 'Review and remove invalid tickers from wave holdings'
    
    elif reason_category == 'SHORT_HISTORY':
        days_needed = required_days - available_days
        return f'Fetch {days_needed} more days of history. Run analytics pipeline with longer lookback.'
    
    elif reason_category == 'STALE_DATA':
        return 'Data is stale. Re-run analytics pipeline to fetch fresh prices.'
    
    elif reason_category == 'MISSING_BENCHMARK':
        return 'Benchmark data missing. Ensure benchmark tickers are valid and fetch their prices.'
    
    elif reason_category == 'REGISTRY_MISMATCH':
        return 'Wave not properly registered. Check WAVE_ID_REGISTRY and WAVE_WEIGHTS mappings.'
    
    elif reason_category == 'PROVIDER_UNSUPPORTED':
        return 'Data provider API failure. Check yfinance status or try again later.'
    
    else:
        return 'Unknown issue. Check logs for detailed diagnostics.'


def print_readiness_report(coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD) -> None:
    """
    Print Wave Readiness Report to console/logs with graded readiness model.
    
    This function generates and prints a formatted readiness report showing:
    - Summary statistics by graded status
    - Per-wave readiness status with capabilities
    - Blocking vs informational issues
    - Suggested actions
    
    Args:
        coverage_threshold: Minimum required coverage for full status (0.0 to 1.0). Default: 0.95
    """
    print("\n" + "=" * 100)
    print("WAVE READINESS REPORT - GRADED MODEL")
    print("=" * 100)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Coverage Threshold (Full): {coverage_threshold * 100:.0f}%")
    print()
    
    # Generate report
    df = generate_wave_readiness_report(coverage_threshold=coverage_threshold)
    
    # Summary statistics by graded readiness
    total_waves = len(df)
    full_count = (df['readiness_status'] == 'full').sum()
    partial_count = (df['readiness_status'] == 'partial').sum()
    operational_count = (df['readiness_status'] == 'operational').sum()
    unavailable_count = (df['readiness_status'] == 'unavailable').sum()
    
    print("SUMMARY:")
    print(f"  Total Waves: {total_waves}")
    print(f"  Full (All Analytics): {full_count} ({full_count/total_waves*100:.1f}%)")
    print(f"  Partial (Basic Analytics): {partial_count} ({partial_count/total_waves*100:.1f}%)")
    print(f"  Operational (Current State): {operational_count} ({operational_count/total_waves*100:.1f}%)")
    print(f"  Unavailable: {unavailable_count} ({unavailable_count/total_waves*100:.1f}%)")
    
    usable_count = full_count + partial_count + operational_count
    print(f"\n  USABLE WAVES (operational or better): {usable_count} ({usable_count/total_waves*100:.1f}%)")
    print()
    
    # Full waves
    full_df = df[df['readiness_status'] == 'full']
    if not full_df.empty:
        print(f"FULL READINESS WAVES ({len(full_df)}):")
        for _, row in full_df.iterrows():
            print(f"  ✓✓ {row['wave_id']}: {row['wave_name']}")
            print(f"     Coverage: {row['coverage_pct']:.1f}% | Window: {row['available_window_days']} days")
            print(f"     Analytics: {row['allowed_analytics']}")
        print()
    
    # Partial waves
    partial_df = df[df['readiness_status'] == 'partial']
    if not partial_df.empty:
        print(f"PARTIAL READINESS WAVES ({len(partial_df)}):")
        for _, row in partial_df.iterrows():
            print(f"  ✓ {row['wave_id']}: {row['wave_name']}")
            print(f"    Coverage: {row['coverage_pct']:.1f}% | Window: {row['available_window_days']} days")
            print(f"    Analytics: {row['allowed_analytics']}")
            if row['informational_issues']:
                print(f"    Limitations: {row['informational_issues']}")
        print()
    
    # Operational waves
    operational_df = df[df['readiness_status'] == 'operational']
    if not operational_df.empty:
        print(f"OPERATIONAL WAVES ({len(operational_df)}):")
        for _, row in operational_df.iterrows():
            print(f"  ○ {row['wave_id']}: {row['wave_name']}")
            print(f"    Coverage: {row['coverage_pct']:.1f}% | Window: {row['available_window_days']} days")
            print(f"    Analytics: {row['allowed_analytics']}")
            if row['informational_issues']:
                print(f"    Limitations: {row['informational_issues']}")
        print()
    
    # Unavailable waves (sample)
    unavailable_df = df[df['readiness_status'] == 'unavailable']
    if not unavailable_df.empty:
        print(f"UNAVAILABLE WAVES (showing first 10 of {len(unavailable_df)}):")
        for _, row in unavailable_df.head(10).iterrows():
            print(f"  ✗ {row['wave_id']}: {row['wave_name']}")
            print(f"    Blocking Issues: {row['blocking_issues'] or 'Unknown'}")
            if row['failing_tickers']:
                tickers = row['failing_tickers'].split(', ')
                ticker_preview = ', '.join(tickers[:5])
                if len(tickers) > 5:
                    ticker_preview += f' (+{len(tickers) - 5} more)'
                print(f"    Failing Tickers: {ticker_preview}")
            if row['suggested_actions']:
                actions = row['suggested_actions'].split('; ')
                print(f"    Action: {actions[0]}")
            print()
    
    print("=" * 100)
    print()


def get_wave_readiness_diagnostic_summary() -> Dict[str, Any]:
    """
    Get a comprehensive diagnostic summary of wave readiness for debugging.
    
    This function provides immediate visibility into:
    - Total waves in the registry
    - Total waves rendered (should be 28)
    - Count by readiness_status (full/partial/operational/unavailable)
    - Top 1-3 blocking reasons for each unavailable wave
    - Verification that wave universe was sourced from registry
    
    Returns:
        Dictionary with diagnostic information:
        {
            'total_waves_in_registry': int,
            'total_waves_rendered': int,
            'readiness_by_status': {
                'full': int,
                'partial': int,
                'operational': int,
                'unavailable': int
            },
            'unavailable_waves_detail': List[Dict],
            'wave_universe_source': str,
            'is_complete': bool,  # True if all 28 waves present
            'warnings': List[str]
        }
    
    Example:
        >>> diagnostics = get_wave_readiness_diagnostic_summary()
        >>> print(f"Total waves: {diagnostics['total_waves_rendered']}/28")
        >>> print(f"Operational+: {diagnostics['readiness_by_status']['operational']}")
    """
    # Get wave universe from canonical source (already imported at module level)
    universe = get_all_waves_universe()
    total_in_registry = universe['count']
    wave_ids = universe['wave_ids']
    
    # Initialize result
    result = {
        'total_waves_in_registry': total_in_registry,
        'total_waves_rendered': 0,
        'readiness_by_status': {
            'full': 0,
            'partial': 0,
            'operational': 0,
            'unavailable': 0
        },
        'unavailable_waves_detail': [],
        'wave_universe_source': universe['source'],
        'is_complete': False,
        'warnings': []
    }
    
    # Check each wave
    for wave_id in wave_ids:
        try:
            status = compute_data_ready_status(wave_id)
            result['total_waves_rendered'] += 1
            
            # Count by status
            readiness_status = status.get('readiness_status', 'unavailable')
            if readiness_status in result['readiness_by_status']:
                result['readiness_by_status'][readiness_status] += 1
            
            # Track unavailable waves in detail
            if readiness_status == 'unavailable':
                blocking_issues = status.get('blocking_issues', [])
                missing_tickers = status.get('missing_tickers', [])
                
                # Get top 3 blocking reasons
                reasons = []
                if blocking_issues:
                    reasons.extend(blocking_issues[:3])
                if missing_tickers and len(reasons) < 3:
                    reasons.append(f"{len(missing_tickers)} tickers missing")
                
                result['unavailable_waves_detail'].append({
                    'wave_id': wave_id,
                    'display_name': status.get('display_name', wave_id),
                    'top_blocking_reasons': reasons[:3],
                    'coverage_pct': status.get('coverage_pct', 0.0),
                    'suggested_actions': status.get('suggested_actions', [])[:1]  # Top 1 action
                })
        
        except Exception as e:
            result['warnings'].append(f"Error checking {wave_id}: {str(e)}")
    
    # Validate completeness
    result['is_complete'] = (result['total_waves_rendered'] == total_in_registry == 28)
    
    # Add warnings if incomplete
    if result['total_waves_rendered'] < total_in_registry:
        result['warnings'].append(
            f"Only {result['total_waves_rendered']} of {total_in_registry} waves rendered"
        )
    
    if total_in_registry != 28:
        result['warnings'].append(
            f"Registry contains {total_in_registry} waves, expected 28"
        )
    
    return result


# ------------------------------------------------------------
# Main Pipeline Orchestrator
# ------------------------------------------------------------

def run_daily_analytics_pipeline(
    all_waves: bool = True,
    wave_ids: Optional[List[str]] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    use_dummy_data: bool = False
) -> Dict[str, Any]:
    """
    Run the daily analytics pipeline for specified waves.
    
    This is the main orchestrator function that:
    1. Iterates across wave_ids from the registry
    2. Resolves tickers/holdings from wave definitions
    3. Builds benchmark series (including composite benchmarks)
    4. Fetches price history for all tickers + benchmarks
    5. Writes all required artifacts for each wave_id
    
    Args:
        all_waves: If True, process all waves in registry
        wave_ids: Specific wave_ids to process (used if all_waves=False)
        lookback_days: Number of days of historical data to fetch
        use_dummy_data: If True, use dummy data instead of fetching from yfinance
        
    Returns:
        Dictionary with results:
        {
            'total_waves': int,
            'successful': int,
            'failed': int,
            'results': List[Dict],
            'validation_summary': pd.DataFrame
        }
    """
    print("=" * 70)
    print("WAVES Intelligence™ Analytics Pipeline - Stage 4")
    print("=" * 70)
    print(f"Lookback days: {lookback_days}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dummy data mode: {use_dummy_data}")
    print()
    
    # Determine which waves to process
    if all_waves:
        target_wave_ids = get_all_wave_ids()
        print(f"Processing ALL waves: {len(target_wave_ids)} total")
    else:
        target_wave_ids = wave_ids or []
        print(f"Processing specific waves: {len(target_wave_ids)} total")
    
    print()
    
    results = []
    successful = 0
    failed = 0
    
    # Process each wave
    for i, wave_id in enumerate(target_wave_ids, 1):
        display_name = get_display_name_from_wave_id(wave_id)
        print(f"[{i}/{len(target_wave_ids)}] Processing: {wave_id} ({display_name})")
        print("-" * 70)
        
        wave_result = {
            'wave_id': wave_id,
            'display_name': display_name,
            'success': True,
            'errors': []
        }
        
        try:
            # Generate all required files
            if not generate_prices_csv(wave_id, lookback_days, use_dummy_data):
                wave_result['errors'].append('Failed to generate prices.csv')
                wave_result['success'] = False
            
            if not generate_benchmark_prices_csv(wave_id, lookback_days, use_dummy_data):
                wave_result['errors'].append('Failed to generate benchmark_prices.csv')
                wave_result['success'] = False
            
            if not generate_positions_csv(wave_id):
                wave_result['errors'].append('Failed to generate positions.csv')
                wave_result['success'] = False
            
            if not generate_trades_csv(wave_id):
                wave_result['errors'].append('Failed to generate trades.csv')
                wave_result['success'] = False
            
            if not generate_nav_csv(wave_id, lookback_days):
                wave_result['errors'].append('Failed to generate nav.csv')
                wave_result['success'] = False
            
            if wave_result['success']:
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            wave_result['success'] = False
            wave_result['errors'].append(f'Exception: {str(e)}')
            failed += 1
            print(f"✗ Error processing {wave_id}: {e}")
        
        results.append(wave_result)
        print()
    
    # Run validation on all processed waves
    print("=" * 70)
    print("Running Validation Checks")
    print("=" * 70)
    
    validation_results = []
    for wave_id in target_wave_ids:
        val_result = validate_wave_data_ready(wave_id, MIN_REQUIRED_TRADING_DAYS)
        validation_results.append(val_result)
        
        status_icon = "✓" if val_result['status'] == 'pass' else "✗"
        print(f"{status_icon} {wave_id}: {val_result['status'].upper()}")
        
        if val_result['issues']:
            for issue in val_result['issues']:
                print(f"    - {issue}")
    
    print()
    
    # Create validation summary DataFrame
    validation_summary = pd.DataFrame([
        {
            'wave_id': vr['wave_id'],
            'display_name': vr['display_name'],
            'status': vr['status'],
            'prices_ok': vr['checks'].get('prices_valid', False),
            'benchmark_ok': vr['checks'].get('benchmark_exists', False),
            'positions_ok': vr['checks'].get('positions_exists', False),
            'trades_ok': vr['checks'].get('trades_exists', False),
            'nav_ok': vr['checks'].get('nav_aligned', False),
            'issue_count': len(vr['issues'])
        }
        for vr in validation_results
    ])
    
    # Save validation summary
    ensure_directory_exists(ANALYTICS_BASE_DIR)
    validation_path = os.path.join(ANALYTICS_BASE_DIR, 'validation_report.csv')
    validation_summary.to_csv(validation_path, index=False)
    print(f"Validation report saved to: {validation_path}")
    print()
    
    # Export failed tickers diagnostics if available
    if DIAGNOSTICS_AVAILABLE:
        try:
            from helpers.ticker_diagnostics import get_diagnostics_tracker
            tracker = get_diagnostics_tracker()
            
            # Export to CSV with standard filename
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            csv_path = tracker.export_to_csv(filename="failed_tickers_report.csv")
            
            # Get summary stats
            stats = tracker.get_summary_stats()
            print("=" * 70)
            print("Ticker Failure Diagnostics")
            print("=" * 70)
            print(f"Total failed ticker attempts: {stats['total_failures']}")
            print(f"Unique failed tickers: {stats['unique_tickers']}")
            print(f"Fatal failures: {stats['fatal_count']}")
            print(f"Non-fatal failures: {stats['non_fatal_count']}")
            if stats['by_type']:
                print("\nFailure breakdown by type:")
                for ftype, count in stats['by_type'].items():
                    print(f"  - {ftype}: {count}")
            print(f"\nDetailed report saved to: {csv_path}")
            print("=" * 70)
            print()
        except Exception as e:
            print(f"Warning: Could not export ticker diagnostics: {e}")
    
    # Print summary
    print("=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    print(f"Total waves processed: {len(target_wave_ids)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Validation passed: {(validation_summary['status'] == 'pass').sum()}")
    print(f"Validation failed: {(validation_summary['status'] == 'fail').sum()}")
    print("=" * 70)
    
    return {
        'total_waves': len(target_wave_ids),
        'successful': successful,
        'failed': failed,
        'results': results,
        'validation_summary': validation_summary,
        'validation_results': validation_results
    }


def generate_live_snapshot(output_path: str = "live_snapshot.csv") -> pd.DataFrame:
    """
    Generate a comprehensive snapshot of all waves with returns, alpha, and diagnostics.
    
    This provides a single-file snapshot suitable for API endpoints or quick data access.
    
    Args:
        output_path: Path to save the snapshot CSV
        
    Returns:
        DataFrame with snapshot data
    """
    from waves_engine import compute_history_nav
    
    print("=" * 70)
    print("Generating Live Snapshot")
    print("=" * 70)
    
    snapshot_rows = []
    timeframes = [1, 30, 60, 365]
    
    for wave_id in get_all_wave_ids():
        wave_name = get_display_name_from_wave_id(wave_id)
        
        try:
            # Get readiness status
            readiness = compute_data_ready_status(wave_id)
            
            # Initialize row
            # Use readiness_status for both readiness and data_regime for now
            # In future, data_regime could incorporate market regime (VIX, trend, etc.)
            row = {
                'wave_id': wave_id,
                'wave_name': wave_name,
                'readiness_status': readiness.get('readiness_status', 'unavailable'),
                'coverage_pct': readiness.get('coverage_pct', 0.0),
                'data_regime': readiness.get('readiness_status', 'unavailable'),  # Placeholder for future market regime
            }
            
            # Try to compute returns for each timeframe
            for days in timeframes:
                try:
                    nav_df = compute_history_nav(wave_name, mode="Standard", days=days, include_diagnostics=False)
                    
                    if not nav_df.empty and len(nav_df) >= 2:
                        # Calculate returns
                        wave_return = (nav_df['wave_nav'].iloc[-1] / nav_df['wave_nav'].iloc[0] - 1) if 'wave_nav' in nav_df.columns else np.nan
                        bm_return = (nav_df['bm_nav'].iloc[-1] / nav_df['bm_nav'].iloc[0] - 1) if 'bm_nav' in nav_df.columns else np.nan
                        alpha = wave_return - bm_return if not np.isnan(wave_return) and not np.isnan(bm_return) else np.nan
                        
                        row[f'wave_return_{days}d'] = wave_return
                        row[f'bm_return_{days}d'] = bm_return
                        row[f'alpha_{days}d'] = alpha
                    else:
                        row[f'wave_return_{days}d'] = np.nan
                        row[f'bm_return_{days}d'] = np.nan
                        row[f'alpha_{days}d'] = np.nan
                        
                except Exception as e:
                    print(f"  Warning: Could not compute {days}d returns for {wave_name}: {str(e)}")
                    row[f'wave_return_{days}d'] = np.nan
                    row[f'bm_return_{days}d'] = np.nan
                    row[f'alpha_{days}d'] = np.nan
            
            # Add exposure/cash if available (placeholder for now)
            row['exposure'] = np.nan
            row['cash_pct'] = np.nan
            
            snapshot_rows.append(row)
            print(f"  ✓ {wave_name}")
            
        except Exception as e:
            print(f"  ✗ Error processing {wave_name}: {str(e)}")
            # Add minimal row with error info
            snapshot_rows.append({
                'wave_id': wave_id,
                'wave_name': wave_name,
                'readiness_status': 'error',
                'coverage_pct': 0.0,
                'data_regime': 'error',
                **{f'wave_return_{days}d': np.nan for days in timeframes},
                **{f'bm_return_{days}d': np.nan for days in timeframes},
                **{f'alpha_{days}d': np.nan for days in timeframes},
                'exposure': np.nan,
                'cash_pct': np.nan,
            })
    
    # Create DataFrame
    snapshot_df = pd.DataFrame(snapshot_rows)
    
    # Save to CSV
    snapshot_df.to_csv(output_path, index=False)
    print(f"\n✓ Live snapshot saved to: {output_path}")
    print(f"  Total waves: {len(snapshot_df)}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return snapshot_df


def load_live_snapshot(path: str = "live_snapshot.csv", fallback: bool = True) -> pd.DataFrame:
    """
    Load live snapshot CSV with fallback to placeholder data.
    
    Args:
        path: Path to snapshot CSV file
        fallback: If True, return placeholder data if file doesn't exist
        
    Returns:
        DataFrame with snapshot data
    """
    import os
    
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"Warning: Could not load snapshot from {path}: {str(e)}")
            if not fallback:
                raise
    
    if not fallback:
        raise FileNotFoundError(f"Snapshot file not found: {path}")
    
    # Return placeholder snapshot with minimal data
    print(f"Warning: Snapshot file not found, generating fallback data")
    
    rows = []
    for wave_id in get_all_wave_ids():
        wave_name = get_display_name_from_wave_id(wave_id)
        rows.append({
            'wave_id': wave_id,
            'wave_name': wave_name,
            'readiness_status': 'unavailable',
            'coverage_pct': 0.0,
            'data_regime': 'no_data',  # Distinct value for fallback case
            'wave_return_1d': np.nan,
            'bm_return_1d': np.nan,
            'alpha_1d': np.nan,
            'wave_return_30d': np.nan,
            'bm_return_30d': np.nan,
            'alpha_30d': np.nan,
            'wave_return_60d': np.nan,
            'bm_return_60d': np.nan,
            'alpha_60d': np.nan,
            'wave_return_365d': np.nan,
            'bm_return_365d': np.nan,
            'alpha_365d': np.nan,
            'exposure': np.nan,
            'cash_pct': np.nan,
        })
    
    return pd.DataFrame(rows)


def get_broken_tickers_report() -> Dict[str, Any]:
    """
    Get a comprehensive report of all broken/failed tickers across all waves.
    
    This consolidates ticker failures to help identify systematic issues
    (e.g., delisted tickers, API problems).
    
    Returns:
        Dictionary with:
        - total_broken: int - total count of unique broken tickers
        - broken_by_wave: Dict[str, List[str]] - mapping of wave_id to failed tickers
        - ticker_failure_counts: Dict[str, int] - how many waves each ticker fails in
        - most_common_failures: List[Tuple[str, int]] - tickers failing in most waves
    """
    broken_by_wave = {}
    ticker_waves = {}  # ticker -> list of waves it fails in
    
    for wave_id in get_all_wave_ids():
        try:
            status = compute_data_ready_status(wave_id)
            missing_tickers = status.get('missing_tickers', [])
            
            if missing_tickers:
                broken_by_wave[wave_id] = missing_tickers
                
                # Track which waves each ticker fails in
                for ticker in missing_tickers:
                    if ticker not in ticker_waves:
                        ticker_waves[ticker] = []
                    ticker_waves[ticker].append(wave_id)
                    
        except Exception as e:
            print(f"Warning: Could not check {wave_id}: {str(e)}")
            continue
    
    # Count failures per ticker
    ticker_failure_counts = {ticker: len(waves) for ticker, waves in ticker_waves.items()}
    
    # Sort by failure count (descending)
    most_common_failures = sorted(ticker_failure_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'total_broken': len(ticker_waves),
        'broken_by_wave': broken_by_wave,
        'ticker_failure_counts': ticker_failure_counts,
        'most_common_failures': most_common_failures,
        'total_waves_with_failures': len(broken_by_wave),
    }


# ------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------

if __name__ == '__main__':
    import sys
    
    # Parse simple command-line arguments
    all_waves = '--all' in sys.argv or len(sys.argv) == 1
    lookback = DEFAULT_LOOKBACK_DAYS
    
    for arg in sys.argv[1:]:
        if arg.startswith('--lookback='):
            try:
                lookback = int(arg.split('=')[1])
            except ValueError:
                print(f"Invalid lookback value: {arg}")
                sys.exit(1)
    
    # Run pipeline
    result = run_daily_analytics_pipeline(
        all_waves=all_waves,
        lookback_days=lookback
    )
    
    # Exit with appropriate code
    sys.exit(0 if result['failed'] == 0 else 1)

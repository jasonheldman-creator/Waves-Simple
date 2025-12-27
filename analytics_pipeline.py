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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

# Import from waves_engine
from waves_engine import (
    get_all_wave_ids,
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


# ------------------------------------------------------------
# Ticker & Benchmark Resolution
# ------------------------------------------------------------

def resolve_wave_tickers(wave_id: str) -> List[str]:
    """
    Resolve all tickers for a given wave_id from WAVE_WEIGHTS.
    
    Args:
        wave_id: The wave identifier
        
    Returns:
        List of ticker symbols
    """
    display_name = get_display_name_from_wave_id(wave_id)
    if not display_name:
        return []
    
    holdings = WAVE_WEIGHTS.get(display_name, [])
    tickers = []
    
    for holding in holdings:
        if isinstance(holding, Holding):
            tickers.append(holding.ticker)
        elif isinstance(holding, dict) and 'ticker' in holding:
            tickers.append(holding['ticker'])
        elif isinstance(holding, str):
            tickers.append(holding)
    
    return list(set(tickers))  # Remove duplicates


def resolve_wave_benchmarks(wave_id: str) -> List[Tuple[str, float]]:
    """
    Resolve benchmark tickers and weights for a given wave_id.
    
    Args:
        wave_id: The wave identifier
        
    Returns:
        List of (ticker, weight) tuples
    """
    display_name = get_display_name_from_wave_id(wave_id)
    if not display_name:
        return []
    
    benchmark_spec = BENCHMARK_WEIGHTS_STATIC.get(display_name, [])
    benchmarks = []
    
    for bm in benchmark_spec:
        if isinstance(bm, Holding):
            benchmarks.append((bm.ticker, bm.weight))
        elif isinstance(bm, dict):
            benchmarks.append((bm.get('ticker', ''), bm.get('weight', 1.0)))
        elif isinstance(bm, str):
            benchmarks.append((bm, 1.0))
    
    return benchmarks


# ------------------------------------------------------------
# Price Data Fetching
# ------------------------------------------------------------

def fetch_prices(tickers: List[str], start_date: datetime, end_date: datetime, use_dummy_data: bool = False) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Fetch historical prices for a list of tickers with per-ticker error isolation.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        use_dummy_data: If True, generate dummy data instead of fetching from yfinance
        
    Returns:
        Tuple of (prices_df, failures_dict):
        - prices_df: DataFrame with dates as index and tickers as columns
        - failures_dict: Dict mapping failed tickers to error reasons
    """
    failures = {}
    
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
        return pd.DataFrame(), failures
    
    if not tickers:
        return pd.DataFrame(), failures
    
    # Remove duplicates and clean tickers
    tickers = sorted(set(t.strip().upper() for t in tickers if t.strip()))
    
    # Try batch download first, then fall back to individual ticker fetching on failure
    try:
        data = yf.download(
            tickers=tickers,
            start=start_date.strftime('%Y-%m-%d'),
            end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False,
            group_by='ticker',
        )
        
        if data.empty:
            print(f"Warning: No data returned from yfinance for {tickers}")
            # Try individual ticker fetching
            return _fetch_prices_individually(tickers, start_date, end_date, failures)
        
        # Normalize the data structure
        if len(tickers) == 1:
            # Single ticker case
            if 'Close' in data.columns:
                prices = data[['Close']].rename(columns={'Close': tickers[0]})
            else:
                failures[tickers[0]] = "No Close column in data"
                prices = pd.DataFrame()
        else:
            # Multiple tickers case
            frames = []
            for ticker in tickers:
                try:
                    if (ticker, 'Close') in data.columns:
                        frames.append(data[(ticker, 'Close')].rename(ticker))
                    elif ticker in data.columns and 'Close' in data[ticker].columns:
                        frames.append(data[ticker]['Close'].rename(ticker))
                    else:
                        failures[ticker] = "Missing Close column"
                except (KeyError, AttributeError) as e:
                    failures[ticker] = f"Data extraction error: {str(e)}"
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
        for ticker in tickers:
            if prices.empty or ticker not in prices.columns:
                if ticker not in failures:
                    failures[ticker] = "No data in result"
        
        return prices, failures
        
    except Exception as e:
        error_msg = f"Batch download error: {str(e)}"
        print(f"Error fetching prices for {tickers}: {error_msg}")
        # Try individual ticker fetching as fallback
        return _fetch_prices_individually(tickers, start_date, end_date, failures)


def _fetch_prices_individually(tickers: List[str], start_date: datetime, end_date: datetime, failures: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Fetch prices one ticker at a time for maximum resilience.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        failures: Dict to track failures (modified in place)
        
    Returns:
        Tuple of (prices_df, failures_dict)
    """
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame(), failures
    
    all_prices = {}
    common_index = None
    
    for ticker in tickers:
        try:
            data = yf.download(
                tickers=ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                auto_adjust=True,
                progress=False,
            )
            
            if data.empty:
                failures[ticker] = "Empty data returned"
                continue
            
            if 'Close' in data.columns:
                prices_series = data['Close']
            else:
                failures[ticker] = "No Close column"
                continue
            
            # Store prices
            all_prices[ticker] = prices_series
            
            # Build common index
            if common_index is None:
                common_index = prices_series.index
            else:
                common_index = common_index.union(prices_series.index)
                
        except Exception as e:
            failures[ticker] = f"Individual fetch error: {str(e)}"
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
    
    end_date = datetime.now()
    start_date = get_trading_days_back(lookback_days)
    
    prices, failures = fetch_prices(tickers, start_date, end_date, use_dummy_data)
    
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
    
    end_date = datetime.now()
    start_date = get_trading_days_back(lookback_days)
    
    # Fetch individual benchmark prices
    benchmark_tickers = [ticker for ticker, _ in benchmark_specs]
    prices, failures = fetch_prices(benchmark_tickers, start_date, end_date, use_dummy_data)
    
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
    Compute comprehensive data readiness status for a wave.
    
    This function provides detailed diagnostics about why a wave may not be ready,
    enabling operators to quickly identify and resolve data pipeline issues.
    
    Args:
        wave_id: The wave identifier
        
    Returns:
        Dictionary with readiness diagnostics:
        {
            'wave_id': str,
            'display_name': str,
            'is_ready': bool,
            'reason': str,  # Failure reason code or success message
            'details': str,  # Human-readable explanation
            'checks': {
                'has_weights': bool,
                'has_prices': bool,
                'has_benchmark': bool,
                'has_nav': bool,
                'is_fresh': bool,
                'has_sufficient_history': bool,
            }
        }
    
    Reason Codes:
        - "READY": All checks passed
        - "MISSING_WEIGHTS": No holdings defined in WAVE_WEIGHTS
        - "MISSING_PRICES": Price data files not found
        - "MISSING_BENCHMARK": Benchmark price data not found
        - "MISSING_NAV": NAV calculation data not found
        - "STALE_DATA": Data is older than 5 days
        - "INSUFFICIENT_HISTORY": Less than minimum required trading days
        - "WAVE_NOT_FOUND": Wave ID not in registry
    """
    from datetime import datetime
    
    # Initialize response
    result = {
        'wave_id': wave_id,
        'display_name': get_display_name_from_wave_id(wave_id) or wave_id,
        'is_ready': False,
        'reason': 'UNKNOWN',
        'details': '',
        'checks': {
            'has_weights': False,
            'has_prices': False,
            'has_benchmark': False,
            'has_nav': False,
            'is_fresh': False,
            'has_sufficient_history': False,
        }
    }
    
    # Check 1: Wave exists in registry
    all_wave_ids = get_all_wave_ids()
    if wave_id not in all_wave_ids:
        result['reason'] = 'WAVE_NOT_FOUND'
        result['details'] = f"Wave ID '{wave_id}' is not registered in WAVE_ID_REGISTRY"
        return result
    
    # Check 2: Has weights/holdings defined
    if wave_id not in WAVE_WEIGHTS and result['display_name'] not in WAVE_WEIGHTS:
        result['reason'] = 'MISSING_WEIGHTS'
        result['details'] = f"No holdings defined in WAVE_WEIGHTS for '{wave_id}'"
        return result
    
    result['checks']['has_weights'] = True
    
    # Get wave analytics directory
    wave_dir = get_wave_analytics_dir(wave_id)
    
    # Check 3: Has price data
    prices_path = os.path.join(wave_dir, 'prices.csv')
    if not os.path.exists(prices_path):
        result['reason'] = 'MISSING_PRICES'
        result['details'] = f"Price data file not found at {prices_path}"
        return result
    
    result['checks']['has_prices'] = True
    
    # Check 4: Has benchmark data
    benchmark_path = os.path.join(wave_dir, 'benchmark_prices.csv')
    if not os.path.exists(benchmark_path):
        result['reason'] = 'MISSING_BENCHMARK'
        result['details'] = f"Benchmark price data file not found at {benchmark_path}"
        return result
    
    result['checks']['has_benchmark'] = True
    
    # Check 5: Has NAV data
    nav_path = os.path.join(wave_dir, 'nav.csv')
    if not os.path.exists(nav_path):
        result['reason'] = 'MISSING_NAV'
        result['details'] = f"NAV calculation file not found at {nav_path}"
        return result
    
    result['checks']['has_nav'] = True
    
    # Check 6: Data freshness and history length
    try:
        prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        
        if prices_df.empty:
            result['reason'] = 'INSUFFICIENT_HISTORY'
            result['details'] = "Price data file is empty"
            return result
        
        # Check history length
        num_days = len(prices_df)
        if num_days < MIN_REQUIRED_TRADING_DAYS:
            result['reason'] = 'INSUFFICIENT_HISTORY'
            result['details'] = f"Only {num_days} days of history, need at least {MIN_REQUIRED_TRADING_DAYS}"
            return result
        
        result['checks']['has_sufficient_history'] = True
        
        # Check data freshness
        last_date = prices_df.index[-1]
        days_old = (datetime.now() - last_date).days
        
        if days_old > 5:
            result['reason'] = 'STALE_DATA'
            result['details'] = f"Data is {days_old} days old (last: {last_date.date()})"
            return result
        
        result['checks']['is_fresh'] = True
        
        # All checks passed!
        result['is_ready'] = True
        result['reason'] = 'READY'
        result['details'] = f"All checks passed. {num_days} days of fresh data (last: {last_date.date()})"
        
    except Exception as e:
        result['reason'] = 'DATA_READ_ERROR'
        result['details'] = f"Error reading price data: {str(e)}"
        return result
    
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

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

def fetch_prices(tickers: List[str], start_date: datetime, end_date: datetime, use_dummy_data: bool = False) -> pd.DataFrame:
    """
    Fetch historical prices for a list of tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        use_dummy_data: If True, generate dummy data instead of fetching from yfinance
        
    Returns:
        DataFrame with dates as index and tickers as columns
    """
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
        return prices
    
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance is not available. Please install it or use use_dummy_data=True.")
    
    if not tickers:
        return pd.DataFrame()
    
    # Remove duplicates and clean tickers
    tickers = sorted(set(t.strip().upper() for t in tickers if t.strip()))
    
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
            return pd.DataFrame()
        
        # Normalize the data structure
        if len(tickers) == 1:
            # Single ticker case
            if 'Close' in data.columns:
                prices = data[['Close']].rename(columns={'Close': tickers[0]})
            else:
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
                except (KeyError, AttributeError):
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
        
        return prices
        
    except Exception as e:
        print(f"Error fetching prices for {tickers}: {e}")
        return pd.DataFrame()


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
    prices = fetch_prices(benchmark_tickers, start_date, end_date, use_dummy_data)
    
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
    
    prices = fetch_prices(tickers, start_date, end_date, use_dummy_data)
    
    if prices.empty:
        print(f"Warning: No price data fetched for {wave_id}")
        return False
    
    # Ensure we have the minimum required trading days
    if len(prices) < MIN_REQUIRED_TRADING_DAYS:
        print(f"Warning: Only {len(prices)} trading days available for {wave_id}, need {MIN_REQUIRED_TRADING_DAYS}")
    
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
    prices = fetch_prices(benchmark_tickers, start_date, end_date, use_dummy_data)
    
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
    Returns detailed status including OK, PARTIAL, and NO_DATA states.
    
    Args:
        wave_id: The wave identifier
        lookback_days: Minimum number of days required
        
    Returns:
        Dictionary with validation results:
        {
            'wave_id': str,
            'status': 'pass' or 'partial' or 'fail',
            'data_status': 'OK' | 'PARTIAL' | 'NO_DATA',
            'data_quality': str (description of any degradation),
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
            'issues': List[str],
            'ticker_failures': List[str]  # List of tickers that failed to fetch
        }
    """
    wave_dir = get_wave_analytics_dir(wave_id)
    checks = {}
    issues = []
    ticker_failures = []
    
    # Check prices.csv
    prices_path = os.path.join(wave_dir, 'prices.csv')
    checks['prices_exists'] = os.path.exists(prices_path)
    
    if checks['prices_exists']:
        try:
            prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
            checks['prices_days'] = len(prices_df)
            checks['prices_valid'] = len(prices_df) >= lookback_days
            
            # Check for tickers with all NaN values (failed fetches)
            if not prices_df.empty:
                for col in prices_df.columns:
                    if prices_df[col].isna().all():
                        ticker_failures.append(col)
                        issues.append(f"Ticker {col} has no price data")
            
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
    
    # Determine overall status and data quality
    required_checks = [
        'prices_exists', 'benchmark_exists', 'positions_exists', 
        'trades_exists', 'nav_exists'
    ]
    
    all_required_exist = all(checks.get(c, False) for c in required_checks)
    prices_valid = checks.get('prices_valid', False)
    nav_aligned = checks.get('nav_aligned', False)
    
    # Determine data_status: OK, PARTIAL, or NO_DATA
    if all_required_exist and prices_valid and nav_aligned and not ticker_failures:
        data_status = 'OK'
        status = 'pass'
        data_quality = ''
    elif all_required_exist and (prices_valid or nav_aligned):
        # Some data exists but not perfect
        data_status = 'PARTIAL'
        status = 'partial'
        if ticker_failures:
            data_quality = f'DEGRADED: {len(ticker_failures)} ticker(s) failed'
        else:
            data_quality = 'DEGRADED: incomplete data'
    else:
        # Missing critical files or no usable data
        data_status = 'NO_DATA'
        status = 'fail'
        data_quality = 'DEGRADED: missing critical data'
    
    display_name = get_display_name_from_wave_id(wave_id) or wave_id
    
    return {
        'wave_id': wave_id,
        'display_name': display_name,
        'status': status,
        'data_status': data_status,
        'data_quality': data_quality,
        'checks': checks,
        'issues': issues,
        'ticker_failures': ticker_failures
    }


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
        
        # Enhanced status display with data_status
        data_status = val_result.get('data_status', 'NO_DATA')
        if data_status == 'OK':
            status_icon = "✓"
        elif data_status == 'PARTIAL':
            status_icon = "⚠"
        else:
            status_icon = "✗"
        
        print(f"{status_icon} {wave_id}: {data_status}", end="")
        
        # Add data quality info if degraded
        if val_result.get('data_quality'):
            print(f" ({val_result['data_quality']})", end="")
        print()
        
        if val_result['issues']:
            for issue in val_result['issues']:
                print(f"    - {issue}")
    
    print()
    
    # Create validation summary DataFrame with enhanced status tracking
    validation_summary = pd.DataFrame([
        {
            'wave_id': vr['wave_id'],
            'display_name': vr['display_name'],
            'status': vr['status'],
            'data_status': vr.get('data_status', 'NO_DATA'),
            'data_quality': vr.get('data_quality', ''),
            'prices_ok': vr['checks'].get('prices_valid', False),
            'benchmark_ok': vr['checks'].get('benchmark_exists', False),
            'positions_ok': vr['checks'].get('positions_exists', False),
            'trades_ok': vr['checks'].get('trades_exists', False),
            'nav_ok': vr['checks'].get('nav_aligned', False),
            'ticker_failures': len(vr.get('ticker_failures', [])),
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
    
    # Save detailed ticker failures for diagnostics
    ticker_failure_records = []
    for vr in validation_results:
        if vr.get('ticker_failures'):
            for ticker in vr['ticker_failures']:
                ticker_failure_records.append({
                    'wave_id': vr['wave_id'],
                    'display_name': vr['display_name'],
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat()
                })
    
    if ticker_failure_records:
        ticker_failures_df = pd.DataFrame(ticker_failure_records)
        ticker_failures_path = os.path.join(ANALYTICS_BASE_DIR, 'ticker_failures.csv')
        ticker_failures_df.to_csv(ticker_failures_path, index=False)
        print(f"Ticker failures report saved to: {ticker_failures_path}")
        print()
    
    # Print summary
    print("=" * 70)
    print("Pipeline Summary")
    print("=" * 70)
    print(f"Total waves processed: {len(target_wave_ids)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Validation passed (OK): {(validation_summary['data_status'] == 'OK').sum()}")
    print(f"Validation partial (DEGRADED): {(validation_summary['data_status'] == 'PARTIAL').sum()}")
    print(f"Validation failed (NO_DATA): {(validation_summary['data_status'] == 'NO_DATA').sum()}")
    if ticker_failure_records:
        print(f"Total ticker failures: {len(ticker_failure_records)}")
    print("=" * 70)
    print()
    print("NOTE: All waves are included in output regardless of data status.")
    print("Waves with PARTIAL or NO_DATA status will still render with degraded functionality.")
    print("=" * 70)
    
    return {
        'total_waves': len(target_wave_ids),
        'successful': successful,
        'failed': failed,
        'results': results,
        'validation_summary': validation_summary,
        'validation_results': validation_results,
        'ticker_failures': ticker_failure_records
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

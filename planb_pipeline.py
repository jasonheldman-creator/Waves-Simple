"""
planb_pipeline.py

Plan B Data Model and Monitor - Canonical Metrics Pipeline

This module provides a fallback analytics pipeline that is completely decoupled
from live ticker dependencies. It uses pre-computed data files stored in /data/planb/
to generate a canonical metrics snapshot for all 28 waves.

Key Features:
- Always returns 28 rows (one per wave)
- Graceful degradation with status flags (FULL / PARTIAL / UNAVAILABLE)
- No live ticker fetching or blocking operations
- Self-healing with stub file generation

Directory Structure:
    data/planb/
        - nav.csv           (NAV history for all waves)
        - positions.csv     (Current positions snapshot)
        - trades.csv        (Trade history)
        - prices.csv        (Benchmark ticker prices: SPY/QQQ/IWM/etc.)

Usage:
    from planb_pipeline import build_planb_snapshot
    
    # Generate snapshot for all 28 waves
    snapshot_df = build_planb_snapshot(days=365)
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Import from waves_engine for wave registry
try:
    from waves_engine import (
        get_all_wave_ids,
        get_display_name_from_wave_id,
        WAVE_WEIGHTS,
        BENCHMARK_WEIGHTS_STATIC
    )
    WAVES_ENGINE_AVAILABLE = True
except ImportError:
    WAVES_ENGINE_AVAILABLE = False
    get_all_wave_ids = None
    get_display_name_from_wave_id = None
    WAVE_WEIGHTS = {}
    BENCHMARK_WEIGHTS_STATIC = {}

# Constants
PLANB_DATA_DIR = "data/planb"
TRADING_DAYS_PER_YEAR = 252

# Status constants
STATUS_FULL = "FULL"
STATUS_PARTIAL = "PARTIAL"
STATUS_UNAVAILABLE = "UNAVAILABLE"

# File paths
PLANB_NAV_PATH = os.path.join(PLANB_DATA_DIR, "nav.csv")
PLANB_POSITIONS_PATH = os.path.join(PLANB_DATA_DIR, "positions.csv")
PLANB_TRADES_PATH = os.path.join(PLANB_DATA_DIR, "trades.csv")
PLANB_PRICES_PATH = os.path.join(PLANB_DATA_DIR, "prices.csv")


def ensure_planb_directory() -> bool:
    """
    Ensure Plan B data directory exists.
    
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(PLANB_DATA_DIR).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        warnings.warn(f"Failed to create Plan B directory: {e}")
        return False


def generate_stub_nav_file() -> bool:
    """
    Generate stub nav.csv file with correct headers.
    
    Returns:
        True if file was created successfully
    """
    try:
        ensure_planb_directory()
        
        # Create empty DataFrame with correct headers
        stub_df = pd.DataFrame(columns=['date', 'wave_id', 'nav', 'cash', 'holdings_value'])
        stub_df.to_csv(PLANB_NAV_PATH, index=False)
        
        return True
    except Exception as e:
        warnings.warn(f"Failed to create stub nav.csv: {e}")
        return False


def generate_stub_positions_file() -> bool:
    """
    Generate stub positions.csv file with correct headers.
    
    Returns:
        True if file was created successfully
    """
    try:
        ensure_planb_directory()
        
        # Create empty DataFrame with correct headers
        stub_df = pd.DataFrame(columns=['wave_id', 'ticker', 'weight', 'description', 'exposure', 'cash', 'safe_fraction'])
        stub_df.to_csv(PLANB_POSITIONS_PATH, index=False)
        
        return True
    except Exception as e:
        warnings.warn(f"Failed to create stub positions.csv: {e}")
        return False


def generate_stub_trades_file() -> bool:
    """
    Generate stub trades.csv file with correct headers.
    
    Returns:
        True if file was created successfully
    """
    try:
        ensure_planb_directory()
        
        # Create empty DataFrame with correct headers
        stub_df = pd.DataFrame(columns=['date', 'wave_id', 'ticker', 'action', 'shares', 'price', 'value'])
        stub_df.to_csv(PLANB_TRADES_PATH, index=False)
        
        return True
    except Exception as e:
        warnings.warn(f"Failed to create stub trades.csv: {e}")
        return False


def generate_stub_prices_file() -> bool:
    """
    Generate stub prices.csv file with correct headers.
    
    Returns:
        True if file was created successfully
    """
    try:
        ensure_planb_directory()
        
        # Create empty DataFrame with correct headers
        # Include common benchmark tickers as columns
        stub_df = pd.DataFrame(columns=['date', 'SPY', 'QQQ', 'IWM', 'IWV', 'IJH', 'GLD', 'AGG', 'BIL', 'SUB', 'MUB', 'XLE', 'XLI', 'ICLN'])
        stub_df.to_csv(PLANB_PRICES_PATH, index=False)
        
        return True
    except Exception as e:
        warnings.warn(f"Failed to create stub prices.csv: {e}")
        return False


def check_planb_files() -> Dict[str, bool]:
    """
    Check which Plan B files exist.
    
    Returns:
        Dictionary with file existence status
    """
    return {
        'nav': os.path.exists(PLANB_NAV_PATH),
        'positions': os.path.exists(PLANB_POSITIONS_PATH),
        'trades': os.path.exists(PLANB_TRADES_PATH),
        'prices': os.path.exists(PLANB_PRICES_PATH)
    }


def ensure_planb_files() -> Dict[str, Any]:
    """
    Ensure all Plan B files exist, create stubs if missing.
    
    Returns:
        Dictionary with status information
    """
    file_status = check_planb_files()
    created_stubs = []
    
    if not file_status['nav']:
        if generate_stub_nav_file():
            created_stubs.append('nav.csv')
            file_status['nav'] = True
    
    if not file_status['positions']:
        if generate_stub_positions_file():
            created_stubs.append('positions.csv')
            file_status['positions'] = True
    
    if not file_status['trades']:
        if generate_stub_trades_file():
            created_stubs.append('trades.csv')
            file_status['trades'] = True
    
    if not file_status['prices']:
        if generate_stub_prices_file():
            created_stubs.append('prices.csv')
            file_status['prices'] = True
    
    return {
        'file_status': file_status,
        'created_stubs': created_stubs,
        'all_exist': all(file_status.values())
    }


def load_planb_nav(days: int = 365) -> pd.DataFrame:
    """
    Load NAV data from Plan B source.
    
    Args:
        days: Number of days of history to load
        
    Returns:
        DataFrame with columns: date, wave_id, nav, cash, holdings_value
    """
    try:
        if not os.path.exists(PLANB_NAV_PATH):
            return pd.DataFrame(columns=['date', 'wave_id', 'nav', 'cash', 'holdings_value'])
        
        df = pd.read_csv(PLANB_NAV_PATH)
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter to requested time window
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['date'] >= cutoff_date]
        
        return df
    except Exception as e:
        warnings.warn(f"Failed to load nav.csv: {e}")
        return pd.DataFrame(columns=['date', 'wave_id', 'nav', 'cash', 'holdings_value'])


def load_planb_positions() -> pd.DataFrame:
    """
    Load positions data from Plan B source.
    
    Returns:
        DataFrame with columns: wave_id, ticker, weight, description, exposure, cash, safe_fraction
    """
    try:
        if not os.path.exists(PLANB_POSITIONS_PATH):
            return pd.DataFrame(columns=['wave_id', 'ticker', 'weight', 'description', 'exposure', 'cash', 'safe_fraction'])
        
        df = pd.read_csv(PLANB_POSITIONS_PATH)
        return df
    except Exception as e:
        warnings.warn(f"Failed to load positions.csv: {e}")
        return pd.DataFrame(columns=['wave_id', 'ticker', 'weight', 'description', 'exposure', 'cash', 'safe_fraction'])


def load_planb_trades(days: int = 365) -> pd.DataFrame:
    """
    Load trades data from Plan B source.
    
    Args:
        days: Number of days of history to load
        
    Returns:
        DataFrame with columns: date, wave_id, ticker, action, shares, price, value
    """
    try:
        if not os.path.exists(PLANB_TRADES_PATH):
            return pd.DataFrame(columns=['date', 'wave_id', 'ticker', 'action', 'shares', 'price', 'value'])
        
        df = pd.read_csv(PLANB_TRADES_PATH)
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter to requested time window
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['date'] >= cutoff_date]
        
        return df
    except Exception as e:
        warnings.warn(f"Failed to load trades.csv: {e}")
        return pd.DataFrame(columns=['date', 'wave_id', 'ticker', 'action', 'shares', 'price', 'value'])


def load_planb_prices(days: int = 365) -> pd.DataFrame:
    """
    Load benchmark prices data from Plan B source.
    
    Args:
        days: Number of days of history to load
        
    Returns:
        DataFrame with date index and ticker columns
    """
    try:
        if not os.path.exists(PLANB_PRICES_PATH):
            return pd.DataFrame()
        
        df = pd.read_csv(PLANB_PRICES_PATH)
        
        # Convert date column to datetime and set as index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Filter to requested time window
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df.index >= cutoff_date]
        
        return df
    except Exception as e:
        warnings.warn(f"Failed to load prices.csv: {e}")
        return pd.DataFrame()


def calculate_returns(nav_series: pd.Series, periods: List[int]) -> Dict[str, float]:
    """
    Calculate returns for multiple time periods.
    
    Args:
        nav_series: Time series of NAV values (sorted by date)
        periods: List of periods in days (e.g., [1, 30, 60, 365])
        
    Returns:
        Dictionary mapping period to return percentage
    """
    returns = {}
    
    if nav_series is None or len(nav_series) < 2:
        # Return None for all periods if insufficient data
        return {f"{p}d": None for p in periods}
    
    # Get latest NAV value
    latest_nav = nav_series.iloc[-1]
    
    for period in periods:
        try:
            # Find the NAV value closest to 'period' days ago
            if len(nav_series) <= period:
                # Use earliest available data point
                past_nav = nav_series.iloc[0]
            else:
                # Use value approximately 'period' days ago
                past_nav = nav_series.iloc[-period-1] if len(nav_series) > period else nav_series.iloc[0]
            
            # Calculate percentage return
            if past_nav > 0:
                return_pct = ((latest_nav - past_nav) / past_nav) * 100
                returns[f"{period}d"] = return_pct
            else:
                returns[f"{period}d"] = None
        except Exception:
            returns[f"{period}d"] = None
    
    return returns


def calculate_volatility(nav_series: pd.Series, days: int = 365) -> Optional[float]:
    """
    Calculate annualized volatility from NAV series.
    
    Args:
        nav_series: Time series of NAV values
        days: Number of days to use for calculation
        
    Returns:
        Annualized volatility as percentage, or None if insufficient data
    """
    try:
        if nav_series is None or len(nav_series) < 2:
            return None
        
        # Calculate daily returns
        returns = nav_series.pct_change().dropna()
        
        if len(returns) < 2:
            return None
        
        # Annualize volatility
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        
        return annualized_vol
    except Exception:
        return None


def calculate_max_drawdown(nav_series: pd.Series, days: int = 365) -> Optional[float]:
    """
    Calculate maximum drawdown from NAV series.
    
    Args:
        nav_series: Time series of NAV values
        days: Number of days to use for calculation
        
    Returns:
        Maximum drawdown as percentage, or None if insufficient data
    """
    try:
        if nav_series is None or len(nav_series) < 2:
            return None
        
        # Calculate running maximum
        running_max = nav_series.expanding(min_periods=1).max()
        
        # Calculate drawdown from running maximum
        drawdown = ((nav_series - running_max) / running_max) * 100
        
        # Return maximum drawdown (most negative value)
        max_dd = drawdown.min()
        
        return max_dd
    except Exception:
        return None


def calculate_beta(wave_returns: pd.Series, benchmark_returns: pd.Series) -> Optional[float]:
    """
    Calculate beta (rolling regression of wave returns vs benchmark returns).
    
    Args:
        wave_returns: Time series of wave daily returns
        benchmark_returns: Time series of benchmark daily returns
        
    Returns:
        Beta estimate, or None if insufficient data
    """
    try:
        if wave_returns is None or benchmark_returns is None:
            return None
        
        if len(wave_returns) < 30 or len(benchmark_returns) < 30:
            return None
        
        # Align the two series
        aligned = pd.DataFrame({
            'wave': wave_returns,
            'benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < 30:
            return None
        
        # Calculate covariance and variance
        covariance = aligned['wave'].cov(aligned['benchmark'])
        benchmark_variance = aligned['benchmark'].var()
        
        if benchmark_variance > 0:
            beta = covariance / benchmark_variance
            return beta
        else:
            return None
    except Exception:
        return None


def calculate_turnover(trades_df: pd.DataFrame, wave_id: str, nav_df: pd.DataFrame, days: int = 30) -> Optional[float]:
    """
    Calculate turnover rate from trades.
    
    Args:
        trades_df: DataFrame of trades
        wave_id: Wave identifier
        nav_df: DataFrame of NAV values
        days: Number of days for turnover calculation
        
    Returns:
        Turnover rate as percentage, or None if insufficient data
    """
    try:
        if trades_df.empty or nav_df.empty:
            return None
        
        # Filter trades for this wave
        wave_trades = trades_df[trades_df['wave_id'] == wave_id]
        
        if wave_trades.empty:
            return None
        
        # Filter to last N days
        cutoff_date = datetime.now() - timedelta(days=days)
        wave_trades = wave_trades[wave_trades['date'] >= cutoff_date]
        
        if wave_trades.empty:
            return None
        
        # Calculate total trade value (absolute)
        total_trade_value = wave_trades['value'].abs().sum()
        
        # Get average NAV over the period
        wave_nav = nav_df[nav_df['wave_id'] == wave_id]
        wave_nav = wave_nav[wave_nav['date'] >= cutoff_date]
        
        if wave_nav.empty:
            return None
        
        avg_nav = wave_nav['nav'].mean()
        
        if avg_nav > 0:
            turnover_rate = (total_trade_value / avg_nav) * 100
            return turnover_rate
        else:
            return None
    except Exception:
        return None


def generate_alerts(row: Dict[str, Any]) -> List[str]:
    """
    Generate alerts based on metrics.
    
    Args:
        row: Dictionary of metrics for a wave
        
    Returns:
        List of alert messages
    """
    alerts = []
    
    # Beta drift alert
    beta = row.get('beta_est')
    if beta is not None:
        if beta > 1.5:
            alerts.append(f"High beta drift: {beta:.2f}")
        elif beta < 0.5:
            alerts.append(f"Low beta drift: {beta:.2f}")
    
    # Drawdown alert
    maxdd = row.get('maxdd_365d')
    if maxdd is not None and maxdd < -30:
        alerts.append(f"Large drawdown: {maxdd:.1f}%")
    
    # Status alerts
    status = row.get('status')
    if status == STATUS_UNAVAILABLE:
        alerts.append("Data unavailable")
    elif status == STATUS_PARTIAL:
        alerts.append("Partial data")
    
    # Staleness alert (check timestamp)
    timestamp = row.get('timestamp')
    if timestamp:
        try:
            ts = pd.to_datetime(timestamp)
            age_hours = (datetime.now() - ts).total_seconds() / 3600
            if age_hours > 48:
                alerts.append(f"Stale data: {age_hours:.0f}h old")
        except Exception:
            pass
    
    return alerts


def build_planb_snapshot(days: int = 365) -> pd.DataFrame:
    """
    Build canonical Plan B snapshot for all 28 waves.
    
    This function ALWAYS returns 28 rows, one for each wave in the registry.
    If data is missing or computation fails for a wave, the row is populated
    with status=UNAVAILABLE and a descriptive reason.
    
    Args:
        days: Number of days of history to use for calculations (default: 365)
        
    Returns:
        DataFrame with 28 rows and columns:
        - wave_id: Canonical wave identifier
        - display_name: Human-readable wave name
        - mode: Wave mode (Standard/Alpha-Minus-Beta/Private Logic)
        - timestamp: Snapshot timestamp
        - nav_latest: Latest NAV value
        - return_1d, return_30d, return_60d, return_365d: Returns
        - bm_return_1d, bm_return_30d, bm_return_60d, bm_return_365d: Benchmark returns
        - alpha_1d, alpha_30d, alpha_60d, alpha_365d: Alpha (Wave - Benchmark)
        - exposure_pct: Current exposure percentage
        - cash_pct: Current cash percentage
        - beta_est: Beta estimate
        - vol_365d: Annualized volatility
        - maxdd_365d: Maximum drawdown
        - turnover_30d: 30-day turnover rate
        - alerts: List of alerts
        - status: FULL / PARTIAL / UNAVAILABLE
        - reason: Descriptive reason for degraded status
    """
    
    # Ensure Plan B files exist
    file_status_info = ensure_planb_files()
    
    # Load data files
    nav_df = load_planb_nav(days=days)
    positions_df = load_planb_positions()
    trades_df = load_planb_trades(days=days)
    prices_df = load_planb_prices(days=days)
    
    # Get all wave IDs from registry
    if not WAVES_ENGINE_AVAILABLE:
        # Fallback: create empty snapshot
        return pd.DataFrame()
    
    all_wave_ids = get_all_wave_ids()
    
    # Initialize results list
    results = []
    
    # Current timestamp
    snapshot_timestamp = datetime.now()
    
    # Process each wave
    for wave_id in all_wave_ids:
        # Get display name
        display_name = get_display_name_from_wave_id(wave_id) or wave_id
        
        # Initialize row with defaults
        row = {
            'wave_id': wave_id,
            'display_name': display_name,
            'mode': 'Standard',  # Default mode
            'timestamp': snapshot_timestamp,
            'nav_latest': None,
            'return_1d': None,
            'return_30d': None,
            'return_60d': None,
            'return_365d': None,
            'bm_return_1d': None,
            'bm_return_30d': None,
            'bm_return_60d': None,
            'bm_return_365d': None,
            'alpha_1d': None,
            'alpha_30d': None,
            'alpha_60d': None,
            'alpha_365d': None,
            'exposure_pct': 100.0,  # Default
            'cash_pct': 0.0,  # Default
            'beta_est': None,
            'vol_365d': None,
            'maxdd_365d': None,
            'turnover_30d': None,
            'alerts': [],
            'status': STATUS_FULL,
            'reason': ''
        }
        
        # Track degradation reasons
        reasons = []
        
        # Check if we have any data at all
        if nav_df.empty and positions_df.empty and trades_df.empty and prices_df.empty:
            row['status'] = STATUS_UNAVAILABLE
            row['reason'] = "All Plan B data files are empty"
            results.append(row)
            continue
        
        # Calculate NAV-based metrics
        try:
            wave_nav = nav_df[nav_df['wave_id'] == wave_id].copy()
            
            if not wave_nav.empty:
                # Sort by date
                wave_nav = wave_nav.sort_values('date')
                nav_series = wave_nav.set_index('date')['nav']
                
                # Latest NAV
                row['nav_latest'] = nav_series.iloc[-1] if len(nav_series) > 0 else None
                
                # Calculate returns
                returns = calculate_returns(nav_series, [1, 30, 60, 365])
                row['return_1d'] = returns.get('1d')
                row['return_30d'] = returns.get('30d')
                row['return_60d'] = returns.get('60d')
                row['return_365d'] = returns.get('365d')
                
                # Calculate volatility
                row['vol_365d'] = calculate_volatility(nav_series, days=365)
                
                # Calculate max drawdown
                row['maxdd_365d'] = calculate_max_drawdown(nav_series, days=365)
            else:
                reasons.append("NAV data missing")
        except Exception as e:
            reasons.append(f"NAV calculation error: {str(e)[:50]}")
        
        # Calculate benchmark returns
        try:
            # Get benchmark ticker for this wave
            benchmark_ticker = BENCHMARK_WEIGHTS_STATIC.get(display_name, 'SPY')
            
            if not prices_df.empty and benchmark_ticker in prices_df.columns:
                bm_series = prices_df[benchmark_ticker].dropna()
                
                if len(bm_series) > 0:
                    # Calculate benchmark returns
                    bm_returns = calculate_returns(bm_series, [1, 30, 60, 365])
                    row['bm_return_1d'] = bm_returns.get('1d')
                    row['bm_return_30d'] = bm_returns.get('30d')
                    row['bm_return_60d'] = bm_returns.get('60d')
                    row['bm_return_365d'] = bm_returns.get('365d')
                    
                    # Calculate alpha
                    for period in [1, 30, 60, 365]:
                        wave_ret = row.get(f'return_{period}d')
                        bm_ret = row.get(f'bm_return_{period}d')
                        
                        if wave_ret is not None and bm_ret is not None:
                            row[f'alpha_{period}d'] = wave_ret - bm_ret
            else:
                # Use default benchmark (SPY) if available
                if not prices_df.empty and 'SPY' in prices_df.columns:
                    bm_series = prices_df['SPY'].dropna()
                    
                    if len(bm_series) > 0:
                        bm_returns = calculate_returns(bm_series, [1, 30, 60, 365])
                        row['bm_return_1d'] = bm_returns.get('1d')
                        row['bm_return_30d'] = bm_returns.get('30d')
                        row['bm_return_60d'] = bm_returns.get('60d')
                        row['bm_return_365d'] = bm_returns.get('365d')
                        
                        # Calculate alpha
                        for period in [1, 30, 60, 365]:
                            wave_ret = row.get(f'return_{period}d')
                            bm_ret = row.get(f'bm_return_{period}d')
                            
                            if wave_ret is not None and bm_ret is not None:
                                row[f'alpha_{period}d'] = wave_ret - bm_ret
                else:
                    reasons.append("Benchmark price data missing")
        except Exception as e:
            reasons.append(f"Benchmark calculation error: {str(e)[:50]}")
        
        # Calculate beta
        try:
            wave_nav = nav_df[nav_df['wave_id'] == wave_id].copy()
            benchmark_ticker = BENCHMARK_WEIGHTS_STATIC.get(display_name, 'SPY')
            
            if not wave_nav.empty and not prices_df.empty:
                wave_nav = wave_nav.sort_values('date').set_index('date')
                wave_returns = wave_nav['nav'].pct_change().dropna()
                
                if benchmark_ticker in prices_df.columns:
                    bm_returns = prices_df[benchmark_ticker].pct_change().dropna()
                elif 'SPY' in prices_df.columns:
                    bm_returns = prices_df['SPY'].pct_change().dropna()
                else:
                    bm_returns = None
                
                if bm_returns is not None:
                    row['beta_est'] = calculate_beta(wave_returns, bm_returns)
        except Exception as e:
            reasons.append(f"Beta calculation error: {str(e)[:50]}")
        
        # Get exposure and cash from positions
        try:
            wave_positions = positions_df[positions_df['wave_id'] == wave_id]
            
            if not wave_positions.empty:
                # Calculate exposure and cash
                total_exposure = wave_positions['exposure'].sum() if 'exposure' in wave_positions.columns else 1.0
                total_cash = wave_positions['cash'].sum() if 'cash' in wave_positions.columns else 0.0
                
                row['exposure_pct'] = total_exposure * 100
                row['cash_pct'] = total_cash * 100
            else:
                reasons.append("Position data missing")
        except Exception as e:
            reasons.append(f"Position calculation error: {str(e)[:50]}")
        
        # Calculate turnover
        try:
            if not trades_df.empty and not nav_df.empty:
                row['turnover_30d'] = calculate_turnover(trades_df, wave_id, nav_df, days=30)
            else:
                if trades_df.empty:
                    reasons.append("Trade data missing")
        except Exception as e:
            reasons.append(f"Turnover calculation error: {str(e)[:50]}")
        
        # Set status based on available data
        if len(reasons) > 0:
            if row['nav_latest'] is None:
                row['status'] = STATUS_UNAVAILABLE
            else:
                row['status'] = STATUS_PARTIAL
            row['reason'] = "; ".join(reasons)
        
        # Generate alerts
        row['alerts'] = generate_alerts(row)
        
        results.append(row)
    
    # Convert to DataFrame
    snapshot_df = pd.DataFrame(results)
    
    return snapshot_df


def get_planb_diagnostics() -> Dict[str, Any]:
    """
    Get diagnostics for Plan B data pipeline.
    
    Returns:
        Dictionary with diagnostic information
    """
    file_status = check_planb_files()
    
    diagnostics = {
        'timestamp': datetime.now(),
        'files': file_status,
        'all_files_exist': all(file_status.values()),
        'missing_files': [k for k, v in file_status.items() if not v]
    }
    
    # Check file sizes and row counts
    file_info = {}
    
    for file_key, file_path in [
        ('nav', PLANB_NAV_PATH),
        ('positions', PLANB_POSITIONS_PATH),
        ('trades', PLANB_TRADES_PATH),
        ('prices', PLANB_PRICES_PATH)
    ]:
        if os.path.exists(file_path):
            try:
                file_size = os.path.getsize(file_path)
                df = pd.read_csv(file_path)
                row_count = len(df)
                
                file_info[file_key] = {
                    'exists': True,
                    'size_bytes': file_size,
                    'row_count': row_count
                }
            except Exception as e:
                file_info[file_key] = {
                    'exists': True,
                    'error': str(e)
                }
        else:
            file_info[file_key] = {
                'exists': False
            }
    
    diagnostics['file_info'] = file_info
    
    return diagnostics


if __name__ == "__main__":
    # Test the pipeline
    print("Building Plan B snapshot...")
    
    snapshot = build_planb_snapshot(days=365)
    
    print(f"\nSnapshot generated: {len(snapshot)} rows")
    print(f"\nColumns: {list(snapshot.columns)}")
    
    # Show status summary
    status_counts = snapshot['status'].value_counts()
    print(f"\nStatus summary:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Show first few rows
    print(f"\nFirst 3 rows:")
    print(snapshot.head(3))
    
    # Get diagnostics
    print(f"\nDiagnostics:")
    diagnostics = get_planb_diagnostics()
    print(f"  All files exist: {diagnostics['all_files_exist']}")
    print(f"  Missing files: {diagnostics['missing_files']}")

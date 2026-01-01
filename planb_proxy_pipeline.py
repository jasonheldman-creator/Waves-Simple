"""
planb_proxy_pipeline.py

Plan B Proxy Analytics Pipeline

This module provides a parallel analytics system that uses proxy tickers
to deliver consistent analytics for all 28 waves, independent of individual
wave ticker health.

Key Features:
- Fetches daily prices for proxy and benchmark tickers
- Computes returns (1D, 30D, 60D, 365D)
- Performs alpha calculations against benchmarks
- Adds confidence labels (FULL, PARTIAL, UNAVAILABLE)
- MAX_RETRIES=2 for resilience
- Graceful degradation without hanging
- Persists output to site/data/live_proxy_snapshot.csv
- Persists diagnostics to planb_diagnostics_run.json

Usage:
    from planb_proxy_pipeline import build_proxy_snapshot
    
    # Build proxy analytics for all 28 waves
    snapshot_df = build_proxy_snapshot(days=365)
"""

import os
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from threading import Thread
import time

import numpy as np
import pandas as pd
import yfinance as yf

# Import proxy registry validator
try:
    from helpers.proxy_registry_validator import (
        load_proxy_registry,
        validate_proxy_registry,
        get_enabled_proxy_waves
    )
    PROXY_VALIDATOR_AVAILABLE = True
except ImportError:
    PROXY_VALIDATOR_AVAILABLE = False
    load_proxy_registry = None
    validate_proxy_registry = None
    get_enabled_proxy_waves = None

# Constants
MAX_RETRIES = 2  # Total retries for entire build
MAX_RETRIES_PER_TICKER = 1  # Retry per individual ticker
TICKER_TIMEOUT_SECONDS = 15  # Hard timeout per ticker fetch
BUILD_TIMEOUT_SECONDS = 15  # Wall-clock limit for entire build
TRADING_DAYS_PER_YEAR = 252
OUTPUT_SNAPSHOT_PATH = "data/live_proxy_snapshot.csv"  # Moved to stable data/ directory
DIAGNOSTICS_PATH = "data/planb_diagnostics_run.json"  # Moved to stable data/ directory
FRESHNESS_THRESHOLD_MINUTES = 60  # Consider snapshot fresh if less than 60 minutes old
BUILD_LOCK_MINUTES = 2  # Minimum time between build attempts

# Confidence labels
CONFIDENCE_FULL = "FULL"
CONFIDENCE_PARTIAL = "PARTIAL"
CONFIDENCE_UNAVAILABLE = "UNAVAILABLE"


def ensure_output_directory() -> bool:
    """
    Ensure output directory exists.
    
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(os.path.dirname(OUTPUT_SNAPSHOT_PATH)).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        warnings.warn(f"Failed to create output directory: {e}")
        return False


def fetch_ticker_prices(ticker: str, days: int = 365, max_retries: int = MAX_RETRIES_PER_TICKER) -> Optional[pd.DataFrame]:
    """
    Fetch daily prices for a ticker with retry logic.
    
    Note: Per-ticker timeout is handled at the yfinance library level.
    Wall-clock timeout for the entire build is enforced in build_proxy_snapshot.
    
    Args:
        ticker: Ticker symbol
        days: Number of days of history to fetch
        max_retries: Maximum number of retry attempts (default: 1)
        
    Returns:
        DataFrame with price history, or None if failed
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    for attempt in range(max_retries + 1):
        try:
            # Fetch data from yfinance
            # yfinance has built-in timeouts, typically around 30 seconds
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
            
            # Check if we got data
            if df.empty:
                if attempt < max_retries:
                    print(f"  Retry {attempt + 1}/{max_retries} for {ticker} (empty data)")
                    continue
                else:
                    print(f"  Failed to fetch {ticker}: No data returned after {max_retries} retries")
                    return None
            
            # Return only Close prices
            return df[['Close']].copy()
            
        except Exception as e:
            if attempt < max_retries:
                print(f"  Retry {attempt + 1}/{max_retries} for {ticker}: {str(e)}")
            else:
                print(f"  Failed to fetch {ticker} after {max_retries} retries: {str(e)}")
                return None
    
    return None


def compute_returns(prices_df: pd.DataFrame, periods: List[int] = [1, 30, 60, 365]) -> Dict[str, float]:
    """
    Compute returns over specified periods.
    
    Args:
        prices_df: DataFrame with Close prices
        periods: List of periods (in days) to compute returns
        
    Returns:
        Dictionary with returns for each period (e.g., {'1D': 0.05, '30D': 0.12, ...})
    """
    returns = {}
    
    if prices_df is None or prices_df.empty or len(prices_df) < 2:
        # Return NaN for all periods if no data
        for period in periods:
            returns[f'{period}D'] = np.nan
        return returns
    
    # Get latest price
    latest_price = prices_df['Close'].iloc[-1]
    
    for period in periods:
        # Get price from N days ago
        if len(prices_df) > period:
            past_price = prices_df['Close'].iloc[-(period + 1)]
            
            # Calculate return
            if past_price > 0:
                period_return = (latest_price - past_price) / past_price
                returns[f'{period}D'] = period_return
            else:
                returns[f'{period}D'] = np.nan
        else:
            # Not enough data for this period
            returns[f'{period}D'] = np.nan
    
    return returns


def compute_alpha(proxy_returns: Dict, benchmark_returns: Dict) -> Dict[str, float]:
    """
    Compute alpha (excess return) relative to benchmark.
    
    Args:
        proxy_returns: Returns dict for proxy ticker
        benchmark_returns: Returns dict for benchmark ticker
        
    Returns:
        Dictionary with alpha for each period
    """
    alpha = {}
    
    for period_key in proxy_returns.keys():
        proxy_ret = proxy_returns.get(period_key, np.nan)
        benchmark_ret = benchmark_returns.get(period_key, np.nan)
        
        if not np.isnan(proxy_ret) and not np.isnan(benchmark_ret):
            alpha[f'{period_key}_alpha'] = proxy_ret - benchmark_ret
        else:
            alpha[f'{period_key}_alpha'] = np.nan
    
    return alpha


def determine_confidence_label(primary_available: bool, secondary_available: bool, benchmark_available: bool) -> str:
    """
    Determine confidence label based on data availability.
    
    Args:
        primary_available: Whether primary proxy data is available
        secondary_available: Whether secondary proxy data is available
        benchmark_available: Whether benchmark data is available
        
    Returns:
        Confidence label: FULL, PARTIAL, or UNAVAILABLE
    """
    if primary_available and benchmark_available:
        return CONFIDENCE_FULL
    elif secondary_available and benchmark_available:
        return CONFIDENCE_PARTIAL
    else:
        return CONFIDENCE_UNAVAILABLE


def build_proxy_snapshot(
    days: int = 365, 
    enforce_timeout: bool = True,
    session_state: Optional[Dict] = None,
    explicit_button_click: bool = False
) -> pd.DataFrame:
    """
    Build proxy analytics snapshot for all 28 waves.
    STEP 2 & 3: Integrated with SAFE DEMO MODE and Compute Gate
    
    Args:
        days: Number of days of price history to fetch
        enforce_timeout: If True, enforce BUILD_TIMEOUT_SECONDS wall-clock limit
        session_state: Streamlit session_state for SAFE DEMO MODE and compute gate
        explicit_button_click: True if user explicitly clicked a rebuild button
        
    Returns:
        DataFrame with proxy analytics for all waves
    """
    # ========================================================================
    # STEP 2: Check SAFE DEMO MODE - Global Kill Switch
    # ========================================================================
    if session_state is not None and session_state.get("safe_demo_mode", False):
        print("🛡️ SAFE DEMO MODE active - proxy snapshot build suppressed")
        # Load existing snapshot if available
        if os.path.exists(OUTPUT_SNAPSHOT_PATH):
            try:
                print(f"   Loading existing snapshot from {OUTPUT_SNAPSHOT_PATH}")
                return pd.read_csv(OUTPUT_SNAPSHOT_PATH)
            except Exception as e:
                print(f"   Warning: Could not load snapshot: {e}")
        return pd.DataFrame()
    
    # ========================================================================
    # STEP 3: Check Compute Gate
    # ========================================================================
    try:
        from helpers.compute_gate import should_allow_build, mark_build_complete
        
        should_build, reason = should_allow_build(
            snapshot_path=OUTPUT_SNAPSHOT_PATH,
            session_state=session_state,
            build_key="planb_snapshot",
            explicit_button_click=explicit_button_click
        )
        
        if not should_build:
            print(f"⏸️ Plan B snapshot build suppressed: {reason}")
            # Load existing snapshot if available
            if os.path.exists(OUTPUT_SNAPSHOT_PATH):
                try:
                    print(f"   Loading existing snapshot from {OUTPUT_SNAPSHOT_PATH}")
                    return pd.read_csv(OUTPUT_SNAPSHOT_PATH)
                except Exception as e:
                    print(f"   Warning: Could not load snapshot: {e}")
            return pd.DataFrame()
    except ImportError:
        # Compute gate not available - proceed with build
        pass
    
    # Record start time for wall-clock timeout
    build_start_time = datetime.now()
    
    # Validate proxy registry
    if not PROXY_VALIDATOR_AVAILABLE:
        print("⚠️ Proxy validator not available - cannot build snapshot")
        return pd.DataFrame()
    
    validation_result = validate_proxy_registry(strict=False)
    print(validation_result['report'])
    
    if not validation_result['valid'] and not validation_result['degraded_mode']:
        print("❌ Proxy registry validation failed - cannot build snapshot")
        return pd.DataFrame()
    
    # Get enabled waves
    waves = get_enabled_proxy_waves()
    
    if not waves:
        print("⚠️ No enabled waves found in proxy registry")
        return pd.DataFrame()
    
    print(f"\n🔄 Building proxy snapshot for {len(waves)} waves (fetching {days} days of data)...")
    print(f"⏱️  Wall-clock timeout: {BUILD_TIMEOUT_SECONDS}s")
    
    # Track diagnostics
    diagnostics = {
        'timestamp': datetime.now().isoformat(),
        'days': days,
        'total_waves': len(waves),
        'successful_fetches': 0,
        'failed_fetches': 0,
        'timeout_exceeded': False,
        'build_duration_seconds': 0,
        'ticker_failures': [],
        'wave_results': []
    }
    
    # Build snapshot rows
    snapshot_rows = []
    timeout_exceeded = False
    
    for i, wave in enumerate(waves, 1):
        # Check wall-clock timeout
        if enforce_timeout:
            elapsed = (datetime.now() - build_start_time).total_seconds()
            if elapsed > BUILD_TIMEOUT_SECONDS:
                print(f"\n⏱️  ⚠️ WALL-CLOCK TIMEOUT EXCEEDED ({elapsed:.1f}s > {BUILD_TIMEOUT_SECONDS}s)")
                print(f"   Processed {i-1}/{len(waves)} waves. Stopping build and saving diagnostics...")
                timeout_exceeded = True
                diagnostics['timeout_exceeded'] = True
                break
        
        wave_id = wave.get('wave_id', '')
        display_name = wave.get('display_name', '')
        category = wave.get('category', '')
        primary_ticker = wave.get('primary_proxy_ticker', '')
        secondary_ticker = wave.get('secondary_proxy_ticker', '')
        benchmark_ticker = wave.get('benchmark_ticker', '')
        
        print(f"\n[{i}/{len(waves)}] {display_name} ({wave_id})")
        print(f"  Primary: {primary_ticker}, Secondary: {secondary_ticker}, Benchmark: {benchmark_ticker}")
        
        # Fetch prices (only proxy and benchmark tickers, NOT holdings)
        primary_prices = None
        secondary_prices = None
        benchmark_prices = None
        
        if primary_ticker:
            print(f"  Fetching {primary_ticker}...")
            primary_prices = fetch_ticker_prices(primary_ticker, days)
        
        if secondary_ticker:
            print(f"  Fetching {secondary_ticker}...")
            secondary_prices = fetch_ticker_prices(secondary_ticker, days)
        
        if benchmark_ticker:
            print(f"  Fetching {benchmark_ticker}...")
            benchmark_prices = fetch_ticker_prices(benchmark_ticker, days)
        
        # Determine which proxy to use
        proxy_prices = None
        proxy_ticker_used = None
        
        if primary_prices is not None and not primary_prices.empty:
            proxy_prices = primary_prices
            proxy_ticker_used = primary_ticker
            print(f"  ✅ Using primary proxy: {primary_ticker}")
        elif secondary_prices is not None and not secondary_prices.empty:
            proxy_prices = secondary_prices
            proxy_ticker_used = secondary_ticker
            print(f"  ⚠️ Using secondary proxy: {secondary_ticker} (primary unavailable)")
        else:
            print(f"  ❌ No proxy data available")
            diagnostics['failed_fetches'] += 1
            diagnostics['ticker_failures'].append({
                'wave_id': wave_id,
                'primary_ticker': primary_ticker,
                'secondary_ticker': secondary_ticker
            })
        
        # Compute returns
        proxy_returns = compute_returns(proxy_prices) if proxy_prices is not None else {}
        benchmark_returns = compute_returns(benchmark_prices) if benchmark_prices is not None else {}
        
        # Compute alpha
        alpha = compute_alpha(proxy_returns, benchmark_returns)
        
        # Determine confidence label
        confidence = determine_confidence_label(
            primary_available=primary_prices is not None and not primary_prices.empty,
            secondary_available=secondary_prices is not None and not secondary_prices.empty,
            benchmark_available=benchmark_prices is not None and not benchmark_prices.empty
        )
        
        # Build row
        row = {
            'wave_id': wave_id,
            'display_name': display_name,
            'category': category,
            'proxy_ticker': proxy_ticker_used or '',
            'benchmark_ticker': benchmark_ticker,
            'confidence': confidence,
            'return_1D': proxy_returns.get('1D', np.nan),
            'return_30D': proxy_returns.get('30D', np.nan),
            'return_60D': proxy_returns.get('60D', np.nan),
            'return_365D': proxy_returns.get('365D', np.nan),
            'alpha_1D': alpha.get('1D_alpha', np.nan),
            'alpha_30D': alpha.get('30D_alpha', np.nan),
            'alpha_60D': alpha.get('60D_alpha', np.nan),
            'alpha_365D': alpha.get('365D_alpha', np.nan),
            'benchmark_1D': benchmark_returns.get('1D', np.nan),
            'benchmark_30D': benchmark_returns.get('30D', np.nan),
            'benchmark_60D': benchmark_returns.get('60D', np.nan),
            'benchmark_365D': benchmark_returns.get('365D', np.nan)
        }
        
        snapshot_rows.append(row)
        
        # Track diagnostics
        if confidence == CONFIDENCE_FULL:
            diagnostics['successful_fetches'] += 1
        
        diagnostics['wave_results'].append({
            'wave_id': wave_id,
            'confidence': confidence,
            'proxy_ticker_used': proxy_ticker_used
        })
    
    # Create DataFrame
    snapshot_df = pd.DataFrame(snapshot_rows)
    
    # Calculate build duration
    build_duration = (datetime.now() - build_start_time).total_seconds()
    diagnostics['build_duration_seconds'] = build_duration
    
    # Handle timeout case - try to load previous snapshot if available
    if timeout_exceeded and snapshot_df.empty:
        print(f"\n⚠️ Timeout exceeded and no data collected. Attempting to load previous snapshot...")
        try:
            prev_snapshot = load_proxy_snapshot()
            if not prev_snapshot.empty:
                print(f"✅ Loaded previous snapshot with {len(prev_snapshot)} waves")
                snapshot_df = prev_snapshot
                diagnostics['fallback_to_previous'] = True
            else:
                print(f"❌ No previous snapshot available")
                diagnostics['fallback_to_previous'] = False
        except Exception as e:
            print(f"❌ Failed to load previous snapshot: {e}")
            diagnostics['fallback_to_previous'] = False
    
    # Persist to CSV with error handling
    try:
        ensure_output_directory()
        snapshot_df.to_csv(OUTPUT_SNAPSHOT_PATH, index=False)
        print(f"\n✅ Snapshot saved to {OUTPUT_SNAPSHOT_PATH}")
        diagnostics['snapshot_saved'] = True
    except Exception as e:
        print(f"\n⚠️ Failed to save snapshot: {e}")
        print(f"   Attempting to load previous snapshot as fallback...")
        diagnostics['snapshot_saved'] = False
        diagnostics['save_error'] = str(e)
        
        # Try to load previous snapshot
        try:
            prev_snapshot = load_proxy_snapshot()
            if not prev_snapshot.empty:
                print(f"✅ Loaded previous snapshot with {len(prev_snapshot)} waves")
                snapshot_df = prev_snapshot
                diagnostics['fallback_to_previous'] = True
        except Exception as fallback_error:
            print(f"❌ Failed to load previous snapshot: {fallback_error}")
            diagnostics['fallback_to_previous'] = False
    
    # Persist diagnostics with error handling
    try:
        # Ensure data directory exists
        Path(os.path.dirname(DIAGNOSTICS_PATH)).mkdir(parents=True, exist_ok=True)
        with open(DIAGNOSTICS_PATH, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        print(f"✅ Diagnostics saved to {DIAGNOSTICS_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to save diagnostics: {e}")
    
    # Print summary
    print(f"\n📊 SNAPSHOT SUMMARY:")
    print(f"   • Total waves: {len(snapshot_df)}")
    if not snapshot_df.empty:
        print(f"   • FULL confidence: {len(snapshot_df[snapshot_df['confidence'] == CONFIDENCE_FULL])}")
        print(f"   • PARTIAL confidence: {len(snapshot_df[snapshot_df['confidence'] == CONFIDENCE_PARTIAL])}")
        print(f"   • UNAVAILABLE: {len(snapshot_df[snapshot_df['confidence'] == CONFIDENCE_UNAVAILABLE])}")
    print(f"   • Build duration: {build_duration:.1f}s")
    if timeout_exceeded:
        print(f"   • ⚠️ Wall-clock timeout exceeded")
    
    # ========================================================================
    # STEP 3: Mark build as complete in compute gate
    # ========================================================================
    try:
        from helpers.compute_gate import mark_build_complete
        if session_state is not None:
            success = not snapshot_df.empty and not timeout_exceeded
            mark_build_complete(session_state, "planb_snapshot", success=success)
    except ImportError:
        pass
    
    return snapshot_df


def load_proxy_snapshot(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the most recent proxy snapshot from disk.
    
    Args:
        path: Optional custom path to snapshot file
        
    Returns:
        DataFrame with snapshot data, or empty DataFrame if not found
    """
    snapshot_path = path or OUTPUT_SNAPSHOT_PATH
    
    if not os.path.exists(snapshot_path):
        print(f"⚠️ Proxy snapshot not found at {snapshot_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(snapshot_path)
        return df
    except Exception as e:
        print(f"❌ Error loading proxy snapshot: {e}")
        return pd.DataFrame()


def get_snapshot_freshness(path: Optional[str] = None) -> Dict:
    """
    Get freshness information about the proxy snapshot.
    
    Args:
        path: Optional custom path to snapshot file
        
    Returns:
        Dictionary with freshness info
    """
    snapshot_path = path or OUTPUT_SNAPSHOT_PATH
    
    if not os.path.exists(snapshot_path):
        return {
            'exists': False,
            'age_minutes': None,
            'fresh': False,
            'stale': False
        }
    
    try:
        # Get file modification time
        mod_time = datetime.fromtimestamp(os.path.getmtime(snapshot_path))
        age_minutes = (datetime.now() - mod_time).total_seconds() / 60
        
        # Consider fresh if less than FRESHNESS_THRESHOLD_MINUTES old
        fresh = age_minutes < FRESHNESS_THRESHOLD_MINUTES
        stale = age_minutes >= FRESHNESS_THRESHOLD_MINUTES
        
        return {
            'exists': True,
            'modified_at': mod_time.isoformat(),
            'age_minutes': age_minutes,
            'fresh': fresh,
            'stale': stale
        }
    except Exception as e:
        print(f"Error checking snapshot freshness: {e}")
        return {
            'exists': True,
            'age_minutes': None,
            'fresh': False,
            'stale': True
        }


def should_trigger_build(session_state: Optional[Dict] = None) -> Tuple[bool, str]:
    """
    Determine if a snapshot build should be triggered based on conditions.
    
    Build is triggered only if:
    1. Snapshot file does not exist, OR
    2. Snapshot is stale (beyond FRESHNESS_THRESHOLD_MINUTES), OR
    3. User explicitly requested rebuild
    
    Build is suppressed if:
    - A build was attempted within BUILD_LOCK_MINUTES
    
    Args:
        session_state: Optional Streamlit session state dict
        
    Returns:
        Tuple of (should_build: bool, reason: str)
    """
    # Check if snapshot exists and freshness
    freshness = get_snapshot_freshness()
    
    # Check build lock (if session_state provided)
    if session_state is not None:
        last_build_time = session_state.get('planb_last_build_attempt')
        if last_build_time is not None:
            minutes_since_last = (datetime.now() - last_build_time).total_seconds() / 60
            if minutes_since_last < BUILD_LOCK_MINUTES:
                return False, f"Build suppressed: Last attempt {minutes_since_last:.1f}m ago (< {BUILD_LOCK_MINUTES}m lock)"
    
    # Condition 1: Snapshot doesn't exist
    if not freshness['exists']:
        return True, "Snapshot does not exist"
    
    # Condition 2: Snapshot is stale
    if freshness.get('stale', False):
        return True, f"Snapshot is stale ({freshness['age_minutes']:.1f}m old)"
    
    # Otherwise, don't build
    return False, f"Snapshot is fresh ({freshness['age_minutes']:.1f}m old)"


def load_diagnostics() -> Dict:
    """
    Load diagnostics from the most recent build.
    
    Returns:
        Dictionary with diagnostics, or empty dict if not found
    """
    if not os.path.exists(DIAGNOSTICS_PATH):
        return {}
    
    try:
        with open(DIAGNOSTICS_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading diagnostics: {e}")
        return {}

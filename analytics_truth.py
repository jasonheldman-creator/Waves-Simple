"""
analytics_truth.py

CANONICAL TRUTHFRAME - Single Source of Truth for All Wave Analytics

This module provides the TruthFrame: a single, comprehensive DataFrame containing
all analytics data for all 28 waves. Every tab, table, and chart in the application
should consume this TruthFrame without recomputing returns, alphas, or exposures.

Key Features:
- Single function to generate/load TruthFrame: get_truth_frame()
- Contains all 28 waves (never drops rows)
- Supports Safe Mode ON (load from live_snapshot.csv only)
- Supports Safe Mode OFF (build from engine/prices with fallbacks)
- All metrics pre-computed: returns, alphas, exposures, betas, etc.
- Best-effort values with explicit unavailable markers

Required Columns:
- wave_id: Canonical wave identifier (snake_case)
- display_name: Human-readable wave name
- mode: Operating mode (Standard, Alpha-Minus-Beta, etc.)
- readiness_status: Data readiness (full/partial/operational/unavailable)
- coverage_pct: Data coverage percentage
- data_regime_tag: Data regime (LIVE/SANDBOX/HYBRID/UNAVAILABLE)
- return_1d, return_30d, return_60d, return_365d: Wave returns
- alpha_1d, alpha_30d, alpha_60d, alpha_365d: Alpha over benchmark
- benchmark_return_1d, benchmark_return_30d, benchmark_return_60d, benchmark_return_365d
- exposure_pct: Portfolio exposure percentage
- cash_pct: Cash percentage
- beta_real: Realized beta (if available)
- beta_target: Target beta (if available)
- beta_drift: Absolute difference between real and target beta
- turnover_est: Estimated turnover (if available)
- drawdown_60d: Maximum drawdown over 60 days (if available)
- alert_badges: Comma-separated or list of alert badges
- last_snapshot_ts: Timestamp of last snapshot

Usage:
    from analytics_truth import get_truth_frame
    
    # Safe Mode ON: Load from snapshot only
    truth_df = get_truth_frame(safe_mode=True)
    
    # Safe Mode OFF: Build from engine with fallbacks
    truth_df = get_truth_frame(safe_mode=False)
    
    # Filter to specific waves
    sp500_data = truth_df[truth_df['wave_id'] == 'sp500_wave']
    
    # Get all returns for display
    returns_df = truth_df[['wave_id', 'display_name', 'return_1d', 'return_30d', 'return_60d', 'return_365d']]
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

# Import from waves_engine
try:
    from waves_engine import (
        get_all_wave_ids,
        get_display_name_from_wave_id,
        WAVE_ID_REGISTRY,
    )
    WAVES_ENGINE_AVAILABLE = True
except ImportError:
    WAVES_ENGINE_AVAILABLE = False
    get_all_wave_ids = None
    get_display_name_from_wave_id = None
    WAVE_ID_REGISTRY = {}

# Import snapshot_ledger for Safe Mode ON
try:
    from snapshot_ledger import load_snapshot, generate_snapshot, get_snapshot_metadata
    SNAPSHOT_LEDGER_AVAILABLE = True
except ImportError:
    SNAPSHOT_LEDGER_AVAILABLE = False
    load_snapshot = None
    generate_snapshot = None
    get_snapshot_metadata = None

# Constants
SNAPSHOT_FILE = "data/live_snapshot.csv"
WAVE_WEIGHTS_FILE = "wave_weights.csv"


def _derive_expected_waves_from_weights() -> List[str]:
    """
    Derive expected waves from wave_weights.csv.
    
    Returns:
        Sorted list of unique wave names from wave_weights.csv
    """
    if not os.path.exists(WAVE_WEIGHTS_FILE):
        print(f"⚠️ Warning: {WAVE_WEIGHTS_FILE} not found")
        return []
    
    try:
        weights_df = pd.read_csv(WAVE_WEIGHTS_FILE)
        expected_waves = sorted(weights_df['wave'].unique().tolist())
        print(f"✓ Derived {len(expected_waves)} expected waves from {WAVE_WEIGHTS_FILE}")
        return expected_waves
    except Exception as e:
        print(f"✗ Error reading {WAVE_WEIGHTS_FILE}: {e}")
        return []


def _convert_wave_name_to_id(wave_name: str) -> str:
    """
    Convert wave display name to wave_id.
    
    Args:
        wave_name: Display name from wave_weights.csv
        
    Returns:
        wave_id in snake_case format
    """
    try:
        # Try to use waves_engine function if available
        if WAVES_ENGINE_AVAILABLE:
            try:
                from waves_engine import get_wave_id_from_display_name
                return get_wave_id_from_display_name(wave_name)
            except:
                pass
        
        # Fallback: manual conversion
        wave_id = wave_name.lower()
        wave_id = wave_id.replace(' & ', '_')
        wave_id = wave_id.replace('&', '_and_')
        wave_id = wave_id.replace(' ', '_')
        wave_id = wave_id.replace('-', '_')
        wave_id = wave_id.replace('/', '_')
        wave_id = wave_id.replace('__', '_')
        
        return wave_id
    except Exception as e:
        print(f"⚠️ Warning: Failed to convert {wave_name} to wave_id: {e}")
        return wave_name.lower().replace(' ', '_')


def _get_wave_tickers_from_weights(wave_name: str) -> List[str]:
    """
    Get list of tickers for a wave from wave_weights.csv.
    
    Args:
        wave_name: Display name of the wave
        
    Returns:
        List of ticker symbols for the wave
    """
    if not os.path.exists(WAVE_WEIGHTS_FILE):
        return []
    
    try:
        weights_df = pd.read_csv(WAVE_WEIGHTS_FILE)
        tickers = weights_df[weights_df['wave'] == wave_name]['ticker'].tolist()
        return tickers
    except Exception as e:
        print(f"⚠️ Warning: Failed to get tickers for {wave_name}: {e}")
        return []


def _fetch_ticker_data_safe(ticker: str, wave_name: str) -> Optional[pd.Series]:
    """
    Safely fetch data for a single ticker with error handling.
    
    Args:
        ticker: Ticker symbol to fetch
        wave_name: Wave name (for logging)
        
    Returns:
        Price series if successful, None if failed
    """
    try:
        # Try to import and use data fetching capabilities
        try:
            import yfinance as yf
            ticker_data = yf.Ticker(ticker)
            hist = ticker_data.history(period="1y")
            if hist.empty:
                return None
            return hist['Close']
        except ImportError:
            # yfinance not available, return None
            return None
    except Exception as e:
        # Log the failure but don't raise
        print(f"  ⚠️ Failed to fetch {ticker}: {str(e)[:50]}")
        return None


def _compute_wave_metrics_from_tickers(
    wave_name: str,
    wave_id: str,
    successful_tickers: List[str],
    ticker_data: Dict[str, pd.Series],
    failed_tickers: List[str]
) -> Dict[str, Any]:
    """
    Compute wave metrics from successfully fetched ticker data.
    
    Args:
        wave_name: Wave display name
        wave_id: Wave identifier
        successful_tickers: List of tickers that were successfully fetched
        ticker_data: Dictionary mapping ticker to price series
        failed_tickers: List of tickers that failed to fetch
        
    Returns:
        Dictionary with computed metrics
    """
    # Get weights for successful tickers
    try:
        weights_df = pd.read_csv(WAVE_WEIGHTS_FILE)
        wave_weights = weights_df[weights_df['wave'] == wave_name]
        
        # Filter to successful tickers and renormalize weights
        successful_weights = wave_weights[wave_weights['ticker'].isin(successful_tickers)].copy()
        if successful_weights.empty:
            raise ValueError("No successful tickers with weights")
        
        # Renormalize weights to sum to 1.0
        total_weight = successful_weights['weight'].sum()
        successful_weights['weight'] = successful_weights['weight'] / total_weight
        
        # Compute portfolio returns
        # Align all ticker data to common dates
        combined_data = pd.DataFrame({
            ticker: ticker_data[ticker] 
            for ticker in successful_tickers 
            if ticker in ticker_data
        })
        
        if combined_data.empty:
            raise ValueError("No price data available")
        
        # Forward fill missing values
        combined_data = combined_data.fillna(method='ffill')
        
        # Compute daily returns
        returns = combined_data.pct_change()
        
        # Weight the returns
        weighted_returns = pd.Series(0.0, index=returns.index)
        for _, row in successful_weights.iterrows():
            ticker = row['ticker']
            weight = row['weight']
            if ticker in returns.columns:
                weighted_returns += returns[ticker] * weight
        
        # Compute cumulative NAV (starting at 100)
        nav = (1 + weighted_returns).cumprod() * 100
        
        # Compute returns for different timeframes
        def safe_return(nav_series, days):
            if len(nav_series) < max(days + 1, 2):
                return np.nan
            try:
                end_val = float(nav_series.iloc[-1])
                start_idx = max(0, len(nav_series) - days - 1)
                start_val = float(nav_series.iloc[start_idx])
                if start_val <= 0:
                    return np.nan
                return (end_val / start_val) - 1.0
            except:
                return np.nan
        
        return_1d = safe_return(nav, 1)
        return_30d = safe_return(nav, 30)
        return_60d = safe_return(nav, 60)
        return_365d = safe_return(nav, 365)
        
        # Compute coverage score (0-100)
        total_tickers = len(successful_tickers) + len(failed_tickers)
        coverage_score = (len(successful_tickers) / total_tickers * 100) if total_tickers > 0 else 0.0
        
        # Get current NAV
        current_nav = float(nav.iloc[-1]) if len(nav) > 0 else np.nan
        nav_1d_change = float(nav.iloc[-1] - nav.iloc[-2]) if len(nav) >= 2 else np.nan
        
        return {
            'Wave_ID': wave_id,
            'Wave': wave_name,
            'Category': 'Unknown',  # Will be filled if available
            'Mode': 'Standard',
            'Date': datetime.now().strftime("%Y-%m-%d"),
            'NAV': current_nav,
            'NAV_1D_Change': nav_1d_change,
            'Return_1D': return_1d,
            'Return_30D': return_30d,
            'Return_60D': return_60d,
            'Return_365D': return_365d,
            'Benchmark_Return_1D': np.nan,
            'Benchmark_Return_30D': np.nan,
            'Benchmark_Return_60D': np.nan,
            'Benchmark_Return_365D': np.nan,
            'Alpha_1D': np.nan,
            'Alpha_30D': np.nan,
            'Alpha_60D': np.nan,
            'Alpha_365D': np.nan,
            'Exposure': 1.0,
            'CashPercent': 0.0,
            'VIX_Level': np.nan,
            'VIX_Regime': 'unknown',
            'Beta_Real': np.nan,
            'Beta_Target': 1.0,
            'Beta_Drift': np.nan,
            'Turnover_Est': np.nan,
            'MaxDD': np.nan,
            'Flags': f"{len(failed_tickers)} ticker(s) failed" if failed_tickers else "OK",
            'Data_Regime_Tag': 'Operational',
            'Coverage_Score': coverage_score,
            'status': 'OK',
            'missing_tickers': ', '.join(sorted(failed_tickers)) if failed_tickers else '',
            'missing_ticker_count': len(failed_tickers)
        }
        
    except Exception as e:
        print(f"  ✗ Failed to compute metrics: {e}")
        return None


def generate_snapshot_with_full_coverage() -> pd.DataFrame:
    """
    Generate snapshot ensuring exactly one row per expected wave from wave_weights.csv.
    
    This function:
    1. Derives expected waves from wave_weights.csv
    2. For each wave, attempts to fetch all tickers with per-ticker try/except
    3. Tracks failed tickers and logs failures
    4. Computes metrics if at least one ticker succeeds
    5. Adds NO DATA row if all tickers fail
    6. Validates final DataFrame has exactly one row per expected wave
    7. Writes snapshot to data/live_snapshot.csv
    
    Returns:
        DataFrame with exactly one row per expected wave (28 rows)
    """
    print("\n" + "=" * 80)
    print("GENERATING SNAPSHOT WITH FULL COVERAGE")
    print("=" * 80)
    
    # Step 1: Derive expected waves from wave_weights.csv
    expected_waves = _derive_expected_waves_from_weights()
    
    if not expected_waves:
        raise RuntimeError("Failed to derive expected waves from wave_weights.csv")
    
    print(f"Expected waves count: {len(expected_waves)}")
    
    # Step 2: Process each wave
    snapshot_rows = []
    
    for i, wave_name in enumerate(expected_waves, 1):
        print(f"\n[{i}/{len(expected_waves)}] Processing: {wave_name}")
        
        # Convert wave name to wave_id
        wave_id = _convert_wave_name_to_id(wave_name)
        
        # Get tickers for this wave
        tickers = _get_wave_tickers_from_weights(wave_name)
        
        if not tickers:
            print(f"  ⚠️ No tickers found for {wave_name}")
            # Add NO DATA row
            snapshot_rows.append({
                'Wave_ID': wave_id,
                'Wave': wave_name,
                'Category': 'Unknown',
                'Mode': 'Standard',
                'Date': datetime.now().strftime("%Y-%m-%d"),
                'NAV': np.nan,
                'NAV_1D_Change': np.nan,
                'Return_1D': np.nan,
                'Return_30D': np.nan,
                'Return_60D': np.nan,
                'Return_365D': np.nan,
                'Benchmark_Return_1D': np.nan,
                'Benchmark_Return_30D': np.nan,
                'Benchmark_Return_60D': np.nan,
                'Benchmark_Return_365D': np.nan,
                'Alpha_1D': np.nan,
                'Alpha_30D': np.nan,
                'Alpha_60D': np.nan,
                'Alpha_365D': np.nan,
                'Exposure': 1.0,
                'CashPercent': 0.0,
                'VIX_Level': np.nan,
                'VIX_Regime': 'unknown',
                'Beta_Real': np.nan,
                'Beta_Target': 1.0,
                'Beta_Drift': np.nan,
                'Turnover_Est': np.nan,
                'MaxDD': np.nan,
                'Flags': 'No tickers defined',
                'Data_Regime_Tag': 'Unavailable',
                'Coverage_Score': 0.0,
                'status': 'NO DATA',
                'missing_tickers': 'No tickers defined',
                'missing_ticker_count': 0
            })
            continue
        
        print(f"  Attempting to fetch {len(tickers)} tickers...")
        
        # Step 3: Fetch each ticker with per-ticker try/except
        successful_tickers = []
        failed_tickers = []
        ticker_data = {}
        
        for ticker in tickers:
            data = _fetch_ticker_data_safe(ticker, wave_name)
            if data is not None and not data.empty:
                successful_tickers.append(ticker)
                ticker_data[ticker] = data
            else:
                failed_tickers.append(ticker)
        
        print(f"  ✓ Successful: {len(successful_tickers)}, Failed: {len(failed_tickers)}")
        
        # Step 4: Compute metrics or create NO DATA row
        if len(successful_tickers) > 0:
            # At least one ticker succeeded - compute metrics
            metrics = _compute_wave_metrics_from_tickers(
                wave_name, wave_id, successful_tickers, ticker_data, failed_tickers
            )
            
            if metrics is not None:
                # Add successful row
                snapshot_rows.append(metrics)
            else:
                # Computation failed - add NO DATA row
                snapshot_rows.append({
                    'Wave_ID': wave_id,
                    'Wave': wave_name,
                    'Category': 'Unknown',
                    'Mode': 'Standard',
                    'Date': datetime.now().strftime("%Y-%m-%d"),
                    'NAV': np.nan,
                    'NAV_1D_Change': np.nan,
                    'Return_1D': np.nan,
                    'Return_30D': np.nan,
                    'Return_60D': np.nan,
                    'Return_365D': np.nan,
                    'Benchmark_Return_1D': np.nan,
                    'Benchmark_Return_30D': np.nan,
                    'Benchmark_Return_60D': np.nan,
                    'Benchmark_Return_365D': np.nan,
                    'Alpha_1D': np.nan,
                    'Alpha_30D': np.nan,
                    'Alpha_60D': np.nan,
                    'Alpha_365D': np.nan,
                    'Exposure': 1.0,
                    'CashPercent': 0.0,
                    'VIX_Level': np.nan,
                    'VIX_Regime': 'unknown',
                    'Beta_Real': np.nan,
                    'Beta_Target': 1.0,
                    'Beta_Drift': np.nan,
                    'Turnover_Est': np.nan,
                    'MaxDD': np.nan,
                    'Flags': 'Computation failed',
                    'Data_Regime_Tag': 'Unavailable',
                    'Coverage_Score': 0.0,
                    'status': 'NO DATA',
                    'missing_tickers': ', '.join(sorted(failed_tickers)),
                    'missing_ticker_count': len(failed_tickers)
                })
        else:
            # All tickers failed - add NO DATA row
            snapshot_rows.append({
                'Wave_ID': wave_id,
                'Wave': wave_name,
                'Category': 'Unknown',
                'Mode': 'Standard',
                'Date': datetime.now().strftime("%Y-%m-%d"),
                'NAV': np.nan,
                'NAV_1D_Change': np.nan,
                'Return_1D': np.nan,
                'Return_30D': np.nan,
                'Return_60D': np.nan,
                'Return_365D': np.nan,
                'Benchmark_Return_1D': np.nan,
                'Benchmark_Return_30D': np.nan,
                'Benchmark_Return_60D': np.nan,
                'Benchmark_Return_365D': np.nan,
                'Alpha_1D': np.nan,
                'Alpha_30D': np.nan,
                'Alpha_60D': np.nan,
                'Alpha_365D': np.nan,
                'Exposure': 1.0,
                'CashPercent': 0.0,
                'VIX_Level': np.nan,
                'VIX_Regime': 'unknown',
                'Beta_Real': np.nan,
                'Beta_Target': 1.0,
                'Beta_Drift': np.nan,
                'Turnover_Est': np.nan,
                'MaxDD': np.nan,
                'Flags': f'All {len(failed_tickers)} ticker(s) failed',
                'Data_Regime_Tag': 'Unavailable',
                'Coverage_Score': 0.0,
                'status': 'NO DATA',
                'missing_tickers': ', '.join(sorted(failed_tickers)),
                'missing_ticker_count': len(failed_tickers)
            })
    
    # Step 5: Create DataFrame
    df = pd.DataFrame(snapshot_rows)
    
    # Step 6: Hard assertion - validate exactly one row per expected wave
    actual_wave_count = df['Wave'].nunique()
    expected_wave_count = len(expected_waves)
    
    if actual_wave_count != expected_wave_count:
        error_msg = (
            f"VALIDATION FAILED: Expected {expected_wave_count} unique waves "
            f"but got {actual_wave_count} in the snapshot.\n"
            f"Expected waves: {expected_waves}\n"
            f"Actual waves: {sorted(df['Wave'].unique().tolist())}\n"
            f"Missing waves: {set(expected_waves) - set(df['Wave'].unique())}\n"
            f"Extra waves: {set(df['Wave'].unique()) - set(expected_waves)}"
        )
        raise AssertionError(error_msg)
    
    print("\n" + "=" * 80)
    print("SNAPSHOT VALIDATION")
    print("=" * 80)
    print(f"✓ Expected waves: {expected_wave_count}")
    print(f"✓ Actual unique waves in snapshot: {actual_wave_count}")
    print(f"✓ Total rows in snapshot: {len(df)}")
    print(f"✓ Waves with OK status: {(df['status'] == 'OK').sum()}")
    print(f"✓ Waves with NO DATA status: {(df['status'] == 'NO DATA').sum()}")
    
    # Step 7: Write to data/live_snapshot.csv
    try:
        # Ensure data directory exists
        os.makedirs(os.path.dirname(SNAPSHOT_FILE), exist_ok=True)
        df.to_csv(SNAPSHOT_FILE, index=False)
        print(f"✓ Snapshot written to {SNAPSHOT_FILE}")
    except Exception as e:
        print(f"✗ Failed to write snapshot: {e}")
        raise
    
    print("=" * 80)
    
    return df


def _get_all_28_wave_ids() -> List[str]:
    """
    Get all 28 wave IDs, ensuring we always have the complete list.
    
    Returns:
        List of 28 wave IDs
    """
    if WAVES_ENGINE_AVAILABLE and get_all_wave_ids is not None:
        try:
            wave_ids = get_all_wave_ids()
            if len(wave_ids) == 28:
                return wave_ids
            else:
                print(f"Warning: Expected 28 waves, got {len(wave_ids)}")
                return wave_ids
        except Exception as e:
            print(f"Warning: Failed to get wave IDs from engine: {e}")
    
    # Fallback: Use WAVE_ID_REGISTRY directly
    if WAVE_ID_REGISTRY:
        return sorted(list(WAVE_ID_REGISTRY.keys()))
    
    # Ultimate fallback: hardcoded list of 28 waves
    return [
        "sp500_wave",
        "russell_3000_wave",
        "us_megacap_core_wave",
        "ai_cloud_megacap_wave",
        "next_gen_compute_semis_wave",
        "future_energy_ev_wave",
        "ev_infrastructure_wave",
        "us_small_cap_disruptors_wave",
        "us_mid_small_growth_semis_wave",
        "small_cap_growth_wave",
        "small_to_mid_cap_growth_wave",
        "future_power_energy_wave",
        "quantum_computing_wave",
        "clean_transit_infrastructure_wave",
        "income_wave",
        "demas_fund_wave",
        "crypto_l1_growth_wave",
        "crypto_defi_growth_wave",
        "crypto_l2_growth_wave",
        "crypto_ai_growth_wave",
        "crypto_broad_growth_wave",
        "crypto_income_wave",
        "smartsafe_treasury_cash_wave",
        "smartsafe_tax_free_money_market_wave",
        "gold_wave",
        "infinity_multi_asset_growth_wave",
        "vector_treasury_ladder_wave",
        "vector_muni_ladder_wave",
    ]


def _get_display_name(wave_id: str) -> str:
    """
    Get display name for a wave_id, with fallbacks.
    
    Args:
        wave_id: Wave identifier
        
    Returns:
        Display name for the wave
    """
    if WAVES_ENGINE_AVAILABLE and get_display_name_from_wave_id is not None:
        try:
            name = get_display_name_from_wave_id(wave_id)
            if name:
                return name
        except Exception:
            pass
    
    # Fallback: Use WAVE_ID_REGISTRY
    if wave_id in WAVE_ID_REGISTRY:
        return WAVE_ID_REGISTRY[wave_id]
    
    # Ultimate fallback: Format wave_id as title
    return wave_id.replace("_", " ").title()


def _create_empty_truth_frame() -> pd.DataFrame:
    """
    Create an empty TruthFrame with all 28 waves and default values.
    
    This is used when no data is available or as a fallback.
    
    Returns:
        DataFrame with all 28 waves and NaN/default values
    """
    wave_ids = _get_all_28_wave_ids()
    
    rows = []
    for wave_id in wave_ids:
        row = {
            'wave_id': wave_id,
            'display_name': _get_display_name(wave_id),
            'mode': 'Standard',
            'readiness_status': 'unavailable',
            'coverage_pct': 0.0,
            'data_regime_tag': 'UNAVAILABLE',
            'return_1d': np.nan,
            'return_30d': np.nan,
            'return_60d': np.nan,
            'return_365d': np.nan,
            'alpha_1d': np.nan,
            'alpha_30d': np.nan,
            'alpha_60d': np.nan,
            'alpha_365d': np.nan,
            'benchmark_return_1d': np.nan,
            'benchmark_return_30d': np.nan,
            'benchmark_return_60d': np.nan,
            'benchmark_return_365d': np.nan,
            'exposure_pct': np.nan,
            'cash_pct': np.nan,
            'beta_real': np.nan,
            'beta_target': np.nan,
            'beta_drift': np.nan,
            'turnover_est': np.nan,
            'drawdown_60d': np.nan,
            'alert_badges': '',
            'last_snapshot_ts': datetime.now().isoformat(),
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def _map_snapshot_to_truth_frame(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map snapshot_ledger DataFrame to TruthFrame format.
    
    The snapshot_ledger uses different column names, so we need to map them
    to the canonical TruthFrame column names.
    
    Args:
        snapshot_df: DataFrame from snapshot_ledger.load_snapshot()
        
    Returns:
        DataFrame in TruthFrame format
    """
    # Start with empty TruthFrame to ensure all 28 waves
    truth_df = _create_empty_truth_frame()
    
    if snapshot_df is None or snapshot_df.empty:
        return truth_df
    
    # Map snapshot columns to TruthFrame columns
    column_mapping = {
        'Wave_ID': 'wave_id',
        'Wave': 'display_name',
        'Mode': 'mode',
        'Data_Regime_Tag': 'readiness_status',  # Map Data_Regime_Tag to readiness_status
        'Coverage_Score': 'coverage_pct',
        'Return_1D': 'return_1d',
        'Return_30D': 'return_30d',
        'Return_60D': 'return_60d',
        'Return_365D': 'return_365d',
        'Alpha_1D': 'alpha_1d',
        'Alpha_30D': 'alpha_30d',
        'Alpha_60D': 'alpha_60d',
        'Alpha_365D': 'alpha_365d',
        'Benchmark_Return_1D': 'benchmark_return_1d',
        'Benchmark_Return_30D': 'benchmark_return_30d',
        'Benchmark_Return_60D': 'benchmark_return_60d',
        'Benchmark_Return_365D': 'benchmark_return_365d',
        'Exposure': 'exposure_pct',
        'CashPercent': 'cash_pct',
        'Beta_Real': 'beta_real',
        'Beta_Target': 'beta_target',
        'Beta_Drift': 'beta_drift',
        'Turnover_Est': 'turnover_est',
        'MaxDD': 'drawdown_60d',
        'Flags': 'alert_badges',
        'Date': 'last_snapshot_ts',
    }
    
    # Create a copy for mapping
    mapped_df = snapshot_df.copy()
    
    # Rename columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in mapped_df.columns:
            mapped_df.rename(columns={old_col: new_col}, inplace=True)
    
    # Add data_regime_tag based on readiness_status
    # Map the Data_Regime_Tag values to our canonical format
    DATA_REGIME_MAP = {
        'full': 'LIVE',
        'partial': 'HYBRID',
        'operational': 'SANDBOX',
    }
    
    if 'readiness_status' in mapped_df.columns:
        mapped_df['data_regime_tag'] = mapped_df['readiness_status'].apply(
            lambda status: DATA_REGIME_MAP.get(str(status).lower(), 'UNAVAILABLE')
        )
    else:
        mapped_df['data_regime_tag'] = 'UNAVAILABLE'
    
    # Ensure wave_id column exists
    if 'wave_id' not in mapped_df.columns:
        if 'Wave_ID' in snapshot_df.columns:
            mapped_df['wave_id'] = snapshot_df['Wave_ID']
        else:
            # Create wave_id from display_name if available
            if 'display_name' in mapped_df.columns:
                # This is a fallback - ideally snapshot should have Wave_ID
                mapped_df['wave_id'] = mapped_df['display_name'].apply(
                    lambda x: x.lower().replace(' ', '_').replace('&', 'and').replace('-', '_') if pd.notna(x) else 'unknown'
                )
    
    # Ensure display_name exists
    if 'display_name' not in mapped_df.columns:
        if 'wave_id' in mapped_df.columns:
            mapped_df['display_name'] = mapped_df['wave_id'].apply(_get_display_name)
        else:
            mapped_df['display_name'] = 'Unknown Wave'
    
    # Add mode if not present
    if 'mode' not in mapped_df.columns:
        mapped_df['mode'] = 'Standard'
    
    # Select only TruthFrame columns (keep all available, fill missing with NaN)
    truth_columns = list(truth_df.columns)
    
    # Create final truth frame by merging empty template with mapped data
    # This ensures all 28 waves are present even if snapshot is missing some
    if 'wave_id' in mapped_df.columns:
        # Use wave_id as merge key
        truth_df = truth_df.set_index('wave_id')
        mapped_df = mapped_df.set_index('wave_id')
        
        # Update truth_df with available data from mapped_df
        for col in truth_columns:
            if col in mapped_df.columns and col != 'wave_id':
                # Use .loc for explicit DataFrame updates
                truth_df.loc[truth_df.index.isin(mapped_df.index), col] = mapped_df.loc[mapped_df.index.isin(truth_df.index), col]
        
        truth_df = truth_df.reset_index()
    
    # Ensure column order
    truth_df = truth_df[truth_columns]
    
    return truth_df


def get_truth_frame(
    safe_mode: Optional[bool] = None,
    force_refresh: bool = False,
    max_runtime_seconds: int = 300,
    price_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Get the canonical TruthFrame for all 28 waves.
    
    This is the SINGLE SOURCE OF TRUTH for all wave analytics in the application.
    Every tab, table, and chart should call this function to get wave data.
    
    Args:
        safe_mode: If True, load from live_snapshot.csv only (Safe Mode ON).
                  If False, build from engine/prices with fallbacks (Safe Mode OFF).
                  If None, auto-detect from environment or session state.
        force_refresh: Force regeneration of snapshot (Safe Mode OFF only)
        max_runtime_seconds: Maximum time for snapshot generation (Safe Mode OFF only)
        price_df: Optional pre-fetched price DataFrame (Safe Mode OFF only)
        
    Returns:
        DataFrame with complete TruthFrame for all 28 waves
        
    Example:
        # Get TruthFrame in Safe Mode
        truth_df = get_truth_frame(safe_mode=True)
        
        # Display all wave returns
        st.dataframe(truth_df[['display_name', 'return_1d', 'return_30d', 'return_60d']])
        
        # Filter to specific wave
        sp500 = truth_df[truth_df['wave_id'] == 'sp500_wave'].iloc[0]
        st.metric("S&P 500 Wave 1D Return", f"{sp500['return_1d']*100:.2f}%")
    """
    # Auto-detect safe_mode if not specified
    if safe_mode is None:
        # Try to get from session state (Streamlit app)
        try:
            import streamlit as st
            safe_mode = st.session_state.get("safe_mode_enabled", False)
        except ImportError:
            # Try environment variable
            safe_mode = os.environ.get("SAFE_MODE", "False").lower() == "true"
    
    print(f"📊 TruthFrame: Getting canonical data (Safe Mode: {safe_mode})")
    
    # SAFE MODE ON: Load from snapshot only
    if safe_mode:
        if not SNAPSHOT_LEDGER_AVAILABLE or load_snapshot is None:
            print("⚠️ TruthFrame: snapshot_ledger not available, returning empty frame")
            return _create_empty_truth_frame()
        
        try:
            # Load existing snapshot (no generation in Safe Mode)
            snapshot_df = load_snapshot(force_refresh=False)
            
            if snapshot_df is None or snapshot_df.empty:
                print("⚠️ TruthFrame: No snapshot available, returning empty frame")
                return _create_empty_truth_frame()
            
            # Map to TruthFrame format
            truth_df = _map_snapshot_to_truth_frame(snapshot_df)
            print(f"✓ TruthFrame: Loaded {len(truth_df)} waves from snapshot (Safe Mode)")
            return truth_df
            
        except Exception as e:
            print(f"✗ TruthFrame: Failed to load snapshot in Safe Mode: {e}")
            return _create_empty_truth_frame()
    
    # SAFE MODE OFF: Build from engine with fallbacks
    else:
        if not SNAPSHOT_LEDGER_AVAILABLE or generate_snapshot is None or load_snapshot is None:
            print("⚠️ TruthFrame: snapshot_ledger not available, returning empty frame")
            return _create_empty_truth_frame()
        
        try:
            # Try to load existing snapshot first (if not forcing refresh)
            if not force_refresh:
                snapshot_df = load_snapshot(force_refresh=False)
                
                if snapshot_df is not None and not snapshot_df.empty:
                    # Check if snapshot is recent enough
                    if get_snapshot_metadata is not None:
                        metadata = get_snapshot_metadata()
                        if metadata.get("exists") and not metadata.get("is_stale"):
                            truth_df = _map_snapshot_to_truth_frame(snapshot_df)
                            print(f"✓ TruthFrame: Loaded {len(truth_df)} waves from cached snapshot")
                            return truth_df
            
            # Generate new snapshot
            print("⏳ TruthFrame: Generating new snapshot from engine...")
            snapshot_df = generate_snapshot(
                force_refresh=force_refresh,
                max_runtime_seconds=max_runtime_seconds,
                price_df=price_df
            )
            
            if snapshot_df is None or snapshot_df.empty:
                print("⚠️ TruthFrame: Snapshot generation failed, returning empty frame")
                return _create_empty_truth_frame()
            
            # Map to TruthFrame format
            truth_df = _map_snapshot_to_truth_frame(snapshot_df)
            print(f"✓ TruthFrame: Generated {len(truth_df)} waves from engine")
            return truth_df
            
        except Exception as e:
            print(f"✗ TruthFrame: Failed to generate snapshot: {e}")
            # Try to load any existing snapshot as fallback
            try:
                snapshot_df = load_snapshot(force_refresh=False)
                if snapshot_df is not None and not snapshot_df.empty:
                    truth_df = _map_snapshot_to_truth_frame(snapshot_df)
                    print(f"⚠️ TruthFrame: Using stale snapshot as fallback ({len(truth_df)} waves)")
                    return truth_df
            except Exception:
                pass
            
            # Ultimate fallback
            print("⚠️ TruthFrame: All fallbacks failed, returning empty frame")
            return _create_empty_truth_frame()


def get_wave_truth(wave_id: str, truth_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Get truth data for a specific wave.
    
    Args:
        wave_id: Wave identifier
        truth_df: Optional pre-loaded TruthFrame (if None, will load)
        
    Returns:
        Dictionary with wave truth data, or empty dict if wave not found
    """
    if truth_df is None:
        truth_df = get_truth_frame()
    
    wave_data = truth_df[truth_df['wave_id'] == wave_id]
    
    if wave_data.empty:
        return {}
    
    return wave_data.iloc[0].to_dict()


def filter_truth_frame(
    truth_df: pd.DataFrame,
    wave_ids: Optional[List[str]] = None,
    readiness_status: Optional[List[str]] = None,
    data_regime_tag: Optional[List[str]] = None,
    min_coverage_pct: Optional[float] = None
) -> pd.DataFrame:
    """
    Filter TruthFrame by various criteria.
    
    Args:
        truth_df: TruthFrame to filter
        wave_ids: List of wave IDs to include (None = all)
        readiness_status: List of readiness statuses to include (None = all)
        data_regime_tag: List of data regime tags to include (None = all)
        min_coverage_pct: Minimum coverage percentage (None = no filter)
        
    Returns:
        Filtered TruthFrame
    """
    filtered = truth_df.copy()
    
    if wave_ids is not None:
        filtered = filtered[filtered['wave_id'].isin(wave_ids)]
    
    if readiness_status is not None:
        filtered = filtered[filtered['readiness_status'].isin(readiness_status)]
    
    if data_regime_tag is not None:
        filtered = filtered[filtered['data_regime_tag'].isin(data_regime_tag)]
    
    if min_coverage_pct is not None:
        filtered = filtered[filtered['coverage_pct'] >= min_coverage_pct]
    
    return filtered


# Convenience function for backward compatibility with existing code
def load_truth_frame_safe_mode() -> pd.DataFrame:
    """
    Load TruthFrame in Safe Mode (snapshot only).
    
    This is a convenience function for code that explicitly wants Safe Mode behavior.
    
    Returns:
        TruthFrame DataFrame
    """
    return get_truth_frame(safe_mode=True)


def generate_truth_frame_full() -> pd.DataFrame:
    """
    Generate fresh TruthFrame from engine (Safe Mode OFF).
    
    This is a convenience function for code that explicitly wants full generation.
    
    Returns:
        TruthFrame DataFrame
    """
    return get_truth_frame(safe_mode=False, force_refresh=True)

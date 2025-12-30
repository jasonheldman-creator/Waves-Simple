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

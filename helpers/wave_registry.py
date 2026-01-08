"""
Canonical Wave Registry

This module provides access to the canonical wave registry structure.
The registry uses wave_id (not display_name) and includes benchmark recipe metadata.

Key Principles:
- Uses wave_id as primary key
- Includes parsed benchmark recipe metadata (ticker:weight pairs)
- Single source of truth for wave definitions

Usage:
    from helpers.wave_registry import get_wave_registry, get_wave_by_id
    
    # Get full registry
    registry = get_wave_registry()
    
    # Get specific wave
    wave = get_wave_by_id('sp500_wave')
"""

import logging
import os
import pandas as pd
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Path to canonical wave registry CSV
WAVE_REGISTRY_PATH = os.path.join('data', 'wave_registry.csv')


def _parse_benchmark_spec(benchmark_spec: str) -> Dict[str, float]:
    """
    Parse benchmark_spec string into ticker:weight dictionary.
    
    Args:
        benchmark_spec: String like "QQQ:0.6000,IGV:0.4000" or "SPY:1.0000"
        
    Returns:
        Dictionary mapping ticker to weight, e.g., {"QQQ": 0.6, "IGV": 0.4}
    """
    if not benchmark_spec or pd.isna(benchmark_spec):
        return {}
    
    try:
        benchmark_dict = {}
        pairs = benchmark_spec.split(',')
        
        for pair in pairs:
            pair = pair.strip()
            if ':' not in pair:
                continue
            
            ticker, weight = pair.split(':', 1)
            ticker = ticker.strip()
            weight = float(weight.strip())
            benchmark_dict[ticker] = weight
        
        return benchmark_dict
        
    except Exception as e:
        logger.warning(f"Error parsing benchmark_spec '{benchmark_spec}': {e}")
        return {}


def get_wave_registry() -> pd.DataFrame:
    """
    Get the canonical wave registry.
    
    Returns:
        DataFrame with columns:
        - wave_id: Unique identifier for the wave
        - wave_name: Display name
        - mode_default: Default mode (Standard, Private Logic, etc.)
        - benchmark_spec: Raw benchmark specification string
        - benchmark_recipe: Parsed dict of {ticker: weight}
        - holdings_source: Source of holdings (canonical, etc.)
        - category: Wave category (equity_growth, crypto_growth, etc.)
        - active: Whether wave is active (True/False)
        - status: Wave status (ACTIVE/STAGING), defaults to ACTIVE if missing
        - ticker_raw: Raw ticker list
        - ticker_normalized: Normalized ticker list
        - created_at: Creation timestamp
        - updated_at: Last update timestamp
    """
    try:
        if not os.path.exists(WAVE_REGISTRY_PATH):
            logger.warning(f"Wave registry not found: {WAVE_REGISTRY_PATH}")
            return pd.DataFrame()
        
        # Load registry
        registry = pd.read_csv(WAVE_REGISTRY_PATH)
        
        # Add status column if missing, default to ACTIVE
        if 'status' not in registry.columns:
            registry['status'] = 'ACTIVE'
        else:
            # Fill any missing status values with ACTIVE
            registry['status'] = registry['status'].fillna('ACTIVE')
        
        # Parse benchmark_spec into benchmark_recipe
        registry['benchmark_recipe'] = registry['benchmark_spec'].apply(_parse_benchmark_spec)
        
        logger.info(f"Loaded wave registry: {len(registry)} waves")
        return registry
        
    except Exception as e:
        logger.error(f"Error loading wave registry: {e}", exc_info=True)
        return pd.DataFrame()


def get_wave_by_id(wave_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific wave by its wave_id.
    
    Args:
        wave_id: Wave identifier (e.g., 'sp500_wave')
        
    Returns:
        Dictionary with wave information, or None if not found
    """
    registry = get_wave_registry()
    
    if registry.empty:
        return None
    
    wave_rows = registry[registry['wave_id'] == wave_id]
    
    if wave_rows.empty:
        logger.warning(f"Wave not found: {wave_id}")
        return None
    
    # Convert to dict
    wave = wave_rows.iloc[0].to_dict()
    return wave


def get_active_waves() -> pd.DataFrame:
    """
    Get all active waves from the registry.
    
    Returns:
        DataFrame with active waves only (active=True)
    """
    registry = get_wave_registry()
    
    if registry.empty:
        return pd.DataFrame()
    
    active = registry[registry['active']]
    logger.info(f"Found {len(active)} active waves")
    return active


def get_active_status_waves(include_staging: bool = False) -> pd.DataFrame:
    """
    Get waves filtered by status field.
    
    Args:
        include_staging: If True, include both ACTIVE and STAGING waves.
                        If False (default), only return ACTIVE waves.
    
    Returns:
        DataFrame with waves filtered by status
    """
    registry = get_wave_registry()
    
    if registry.empty:
        return pd.DataFrame()
    
    if include_staging:
        # Include both ACTIVE and STAGING
        filtered = registry[registry['status'].isin(['ACTIVE', 'STAGING'])]
        logger.info(f"Found {len(filtered)} waves (ACTIVE + STAGING)")
    else:
        # Only ACTIVE waves
        filtered = registry[registry['status'] == 'ACTIVE']
        logger.info(f"Found {len(filtered)} ACTIVE waves")
    
    return filtered


def check_wave_data_readiness(wave_id: str) -> bool:
    """
    Check if a wave has complete dynamic benchmark and volatility overlay data.
    
    This function checks if the wave has:
    1. Complete benchmark data (valid benchmark_spec)
    2. Volatility overlay data availability (checks wave_history.csv)
    
    Args:
        wave_id: Wave identifier
        
    Returns:
        True if wave has complete data, False otherwise
    """
    try:
        # Get wave info from registry
        wave = get_wave_by_id(wave_id)
        if wave is None:
            logger.warning(f"Wave not found for readiness check: {wave_id}")
            return False
        
        # Check 1: Valid benchmark spec
        benchmark_spec = wave.get('benchmark_spec', '')
        if not benchmark_spec or pd.isna(benchmark_spec):
            logger.debug(f"Wave {wave_id} missing benchmark_spec")
            return False
        
        # Check 2: Wave history data exists
        wave_history_path = os.path.join('wave_history.csv')
        if not os.path.exists(wave_history_path):
            logger.debug(f"Wave history file not found")
            return False
        
        # Check if this wave has entries in wave_history
        try:
            wave_history = pd.read_csv(wave_history_path)
            wave_name = wave.get('wave_name', '')
            
            if 'wave' in wave_history.columns:
                wave_entries = wave_history[wave_history['wave'] == wave_name]
                if len(wave_entries) < 30:  # Require at least 30 days of history
                    logger.debug(f"Wave {wave_id} has insufficient history: {len(wave_entries)} days")
                    return False
                
                # Check for volatility overlay data (vix_level and vix_regime columns)
                if 'vix_level' in wave_history.columns and 'vix_regime' in wave_history.columns:
                    # Check if wave has non-null volatility data
                    wave_entries_with_vix = wave_entries[
                        wave_entries['vix_level'].notna() | 
                        wave_entries['vix_regime'].notna()
                    ]
                    if len(wave_entries_with_vix) > 0:
                        logger.debug(f"Wave {wave_id} has complete data")
                        return True
                    else:
                        logger.debug(f"Wave {wave_id} missing volatility overlay data")
                        return False
            
            logger.debug(f"Wave {wave_id} not found in wave_history")
            return False
            
        except Exception as e:
            logger.warning(f"Error checking wave history for {wave_id}: {e}")
            return False
        
    except Exception as e:
        logger.error(f"Error checking readiness for {wave_id}: {e}", exc_info=True)
        return False


def update_wave_status_based_on_readiness() -> int:
    """
    Update wave status in registry based on data readiness.
    
    Marks waves without complete dynamic benchmark + volatility overlay data as STAGING.
    Waves with complete data are marked as ACTIVE.
    
    Returns:
        Number of waves updated to STAGING status
    """
    try:
        registry = get_wave_registry()
        if registry.empty:
            logger.warning("Cannot update status: registry is empty")
            return 0
        
        updated_count = 0
        
        for idx, row in registry.iterrows():
            wave_id = row['wave_id']
            current_status = row.get('status', 'ACTIVE')
            
            # Check data readiness
            is_ready = check_wave_data_readiness(wave_id)
            
            # Determine new status
            new_status = 'ACTIVE' if is_ready else 'STAGING'
            
            # Update if changed
            if new_status != current_status:
                registry.at[idx, 'status'] = new_status
                updated_count += 1
                logger.info(f"Updated {wave_id}: {current_status} -> {new_status}")
        
        # Save updated registry if any changes
        if updated_count > 0:
            registry.to_csv(WAVE_REGISTRY_PATH, index=False)
            logger.info(f"Updated {updated_count} waves to STAGING status")
        
        return updated_count
        
    except Exception as e:
        logger.error(f"Error updating wave status: {e}", exc_info=True)
        return 0

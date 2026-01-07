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
        DataFrame with active waves only
    """
    registry = get_wave_registry()
    
    if registry.empty:
        return pd.DataFrame()
    
    active = registry[registry['active'] == True]
    logger.info(f"Found {len(active)} active waves")
    return active

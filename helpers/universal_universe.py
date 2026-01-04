"""
Universal Universe Loader

This module provides the SINGLE SOURCE OF TRUTH for all ticker data across the platform.
All ticker references MUST go through this module to ensure consistency.

The universal_universe.csv file is the canonical asset list and this module
enforces its use globally.
"""

import os
import pandas as pd
from typing import List, Dict, Set, Optional, Tuple
from functools import lru_cache
import warnings


# Path to canonical universe file
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIVERSAL_UNIVERSE_PATH = os.path.join(REPO_ROOT, "universal_universe.csv")


class UniverseValidationError(Exception):
    """Raised when universal universe validation fails."""
    pass


@lru_cache(maxsize=1)
def load_universal_universe() -> pd.DataFrame:
    """
    Load the universal universe CSV file.
    
    This function is cached to ensure we only load the file once per session.
    
    Returns:
        DataFrame with columns: ticker, name, asset_class, index_membership, 
                                sector, market_cap_bucket, status, validated, validation_error
    
    Raises:
        UniverseValidationError: If file doesn't exist or is invalid
    """
    if not os.path.exists(UNIVERSAL_UNIVERSE_PATH):
        raise UniverseValidationError(
            f"Universal universe file not found: {UNIVERSAL_UNIVERSE_PATH}\n"
            f"Please run: python build_universal_universe.py"
        )
    
    try:
        df = pd.read_csv(UNIVERSAL_UNIVERSE_PATH)
        
        # Validate required columns
        required_cols = ['ticker', 'asset_class', 'status']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise UniverseValidationError(
                f"Universal universe missing required columns: {missing_cols}"
            )
        
        # Filter to active tickers only
        df = df[df['status'] == 'active'].copy()
        
        return df
        
    except Exception as e:
        if isinstance(e, UniverseValidationError):
            raise
        raise UniverseValidationError(f"Error loading universal universe: {e}")


def get_all_tickers() -> List[str]:
    """
    Get all active tickers from universal universe.
    
    Returns:
        List of ticker symbols (sorted)
    """
    df = load_universal_universe()
    return sorted(df['ticker'].dropna().unique().tolist())


def get_tickers_by_asset_class(asset_class: str) -> List[str]:
    """
    Get tickers filtered by asset class.
    
    Args:
        asset_class: One of 'equity', 'etf', 'crypto', 'fixed_income', 'commodity'
    
    Returns:
        List of ticker symbols
    """
    df = load_universal_universe()
    filtered = df[df['asset_class'] == asset_class]
    return sorted(filtered['ticker'].dropna().unique().tolist())


def get_tickers_by_index(index_name: str) -> List[str]:
    """
    Get tickers that are members of a specific index.
    
    Args:
        index_name: Index identifier (e.g., 'SP500', 'R3000', 'CRYPTO_TOP200')
    
    Returns:
        List of ticker symbols
    """
    df = load_universal_universe()
    
    # Filter rows where index_membership contains the index_name
    mask = df['index_membership'].str.contains(index_name, case=False, na=False)
    filtered = df[mask]
    
    return sorted(filtered['ticker'].dropna().unique().tolist())


def get_ticker_info(ticker: str) -> Optional[Dict[str, any]]:
    """
    Get detailed information about a specific ticker.
    
    Args:
        ticker: Ticker symbol
    
    Returns:
        Dict with ticker info or None if not found
    """
    df = load_universal_universe()
    
    # Normalize ticker for comparison
    ticker_upper = ticker.strip().upper()
    matches = df[df['ticker'].str.upper() == ticker_upper]
    
    if matches.empty:
        return None
    
    # Return first match as dict
    row = matches.iloc[0]
    return row.to_dict()


def validate_ticker(ticker: str) -> Tuple[bool, Optional[str]]:
    """
    Check if ticker exists in universal universe.
    
    Args:
        ticker: Ticker symbol to validate
    
    Returns:
        (is_valid, error_message)
    """
    df = load_universal_universe()
    ticker_upper = ticker.strip().upper()
    
    if ticker_upper in df['ticker'].str.upper().values:
        return True, None
    else:
        return False, f"Ticker '{ticker}' not found in universal universe"


def validate_wave_tickers(wave_tickers: List[str], wave_name: str = "") -> Tuple[List[str], List[str]]:
    """
    Validate a list of tickers against the universal universe.
    
    Args:
        wave_tickers: List of ticker symbols to validate
        wave_name: Name of wave (for logging)
    
    Returns:
        (valid_tickers, invalid_tickers)
    """
    valid = []
    invalid = []
    
    for ticker in wave_tickers:
        is_valid, _ = validate_ticker(ticker)
        if is_valid:
            valid.append(ticker)
        else:
            invalid.append(ticker)
    
    if invalid:
        warning_msg = f"Wave '{wave_name}' has {len(invalid)} invalid tickers: {invalid[:5]}"
        if len(invalid) > 5:
            warning_msg += f" ... and {len(invalid) - 5} more"
        warnings.warn(warning_msg)
    
    return valid, invalid


def get_universe_stats() -> Dict[str, any]:
    """
    Get statistics about the universal universe.
    
    Returns:
        Dict with universe statistics
    """
    df = load_universal_universe()
    
    # Count by asset class
    asset_class_counts = df['asset_class'].value_counts().to_dict()
    
    # Count validated vs not validated
    validation_counts = df['validated'].value_counts().to_dict()
    
    # Count by index membership (approximate - counts tickers in each index)
    index_counts = {}
    for index_name in ['SP500', 'R3000', 'R2000', 'CRYPTO_TOP200', 'INCOME_DEFENSIVE', 'THEMATIC_SECTOR']:
        count = len(get_tickers_by_index(index_name))
        if count > 0:
            index_counts[index_name] = count
    
    return {
        'total_tickers': len(df),
        'asset_classes': asset_class_counts,
        'validation_status': validation_counts,
        'index_membership': index_counts,
        'file_path': UNIVERSAL_UNIVERSE_PATH
    }


def ensure_universe_loaded() -> bool:
    """
    Ensure universal universe can be loaded.
    Used for startup validation.
    
    Returns:
        True if universe loads successfully, False otherwise
    """
    try:
        df = load_universal_universe()
        return len(df) > 0
    except Exception:
        return False


# Graceful degradation helper
def get_ticker_with_fallback(ticker: str, fallback_tickers: Optional[List[str]] = None) -> Optional[str]:
    """
    Try to get ticker from universe, with fallback options.
    
    Args:
        ticker: Primary ticker to try
        fallback_tickers: Optional list of fallback tickers
    
    Returns:
        First valid ticker found, or None
    """
    # Try primary ticker
    is_valid, _ = validate_ticker(ticker)
    if is_valid:
        return ticker
    
    # Try fallbacks
    if fallback_tickers:
        for fallback in fallback_tickers:
            is_valid, _ = validate_ticker(fallback)
            if is_valid:
                return fallback
    
    return None


def get_tickers_for_wave_with_degradation(
    wave_tickers: List[str],
    wave_name: str = ""
) -> Tuple[List[str], Dict[str, str]]:
    """
    Get tickers for a wave with graceful degradation.
    
    This function:
    1. Validates all tickers against universal universe
    2. Returns valid tickers
    3. Logs missing tickers for diagnostics
    4. Does NOT block wave rendering if some tickers are missing
    
    Args:
        wave_tickers: List of ticker symbols from wave definition
        wave_name: Name of wave (for logging)
    
    Returns:
        (valid_tickers, degradation_report)
        - valid_tickers: List of validated tickers
        - degradation_report: Dict mapping invalid tickers to reasons
    """
    valid_tickers, invalid_tickers = validate_wave_tickers(wave_tickers, wave_name)
    
    degradation_report = {}
    for ticker in invalid_tickers:
        degradation_report[ticker] = "not_in_universe"
    
    # Log degradation but don't fail
    if degradation_report:
        warnings.warn(
            f"Wave '{wave_name}' degraded: {len(valid_tickers)}/{len(wave_tickers)} "
            f"tickers available. Missing: {list(degradation_report.keys())[:5]}"
        )
    
    return valid_tickers, degradation_report

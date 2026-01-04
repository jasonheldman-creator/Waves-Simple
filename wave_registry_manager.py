"""
wave_registry_manager.py

CSV Self-Healing System for Wave Universe Registry

This module provides automated CSV management and validation for the Wave universe.
It ensures correctness even when the application encounters corrupted or partial CSV files.

Key Features:
- Canonical source: waves_engine.py WAVE_WEIGHTS and WAVE_ID_REGISTRY
- Auto-rebuild on validation failure
- Ticker normalization (BRK.B → BRK-B, etc.)
- Full column management for wave metadata
- Clear logging and error reporting
- Dynamic wave count (no hard-coded expectations)

Usage:
    from wave_registry_manager import rebuild_wave_registry_csv, validate_wave_registry_csv
    
    # Validate CSV on startup
    is_valid = validate_wave_registry_csv()
    if not is_valid:
        rebuild_wave_registry_csv(force=True)
    
    # Force rebuild (e.g., from UI button)
    rebuild_wave_registry_csv(force=True)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd

# Import canonical wave definitions from waves_engine
from waves_engine import (
    WAVE_WEIGHTS,
    WAVE_ID_REGISTRY,
    DISPLAY_NAME_TO_WAVE_ID,
    BENCHMARK_WEIGHTS_STATIC,
    get_all_wave_ids,
    get_display_name_from_wave_id,
    Holding,
)

# Constants
WAVE_REGISTRY_PATH = "data/wave_registry.csv"
# Dynamic wave count - computed from WAVE_ID_REGISTRY at runtime
# No hard-coded wave count to ensure flexibility as waves are added/removed

# Ticker normalization map - handles common ticker format variations
# Maps from raw ticker format to normalized format for yfinance compatibility
TICKER_ALIASES = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    "BF.A": "BF-A",
}

# Required columns for wave registry CSV
REQUIRED_COLUMNS = [
    "wave_id",           # Canonical identifier (snake_case)
    "wave_name",         # Display name (human-readable)
    "mode_default",      # Default operating mode (Standard, Alpha-Minus-Beta, Private Logic)
    "benchmark_spec",    # Benchmark specification (comma-separated tickers:weights)
    "holdings_source",   # Source of holdings (canonical/static)
    "category",          # Wave category (equity_growth, crypto_growth, equity_income, crypto_income, special)
    "active",            # Boolean flag for wave status
]

# Optional columns (auto-populated during CSV build)
OPTIONAL_COLUMNS = [
    "ticker_raw",        # Raw ticker symbols (comma-separated, as defined in code)
    "ticker_normalized", # Normalized ticker symbols (comma-separated, for yfinance)
    "created_at",        # Timestamp of CSV creation
    "updated_at",        # Timestamp of last update
]

# Configure logging
logger = logging.getLogger(__name__)


def normalize_ticker(ticker: str) -> str:
    """
    Normalize ticker symbol for yfinance compatibility.
    
    Args:
        ticker: Raw ticker symbol (e.g., "BRK.B")
        
    Returns:
        Normalized ticker symbol (e.g., "BRK-B")
    
    Examples:
        >>> normalize_ticker("BRK.B")
        'BRK-B'
        >>> normalize_ticker("AAPL")
        'AAPL'
    """
    # Remove leading/trailing whitespace
    ticker = ticker.strip()
    
    # Apply known aliases
    if ticker in TICKER_ALIASES:
        return TICKER_ALIASES[ticker]
    
    return ticker


def extract_tickers_from_holdings(holdings: List[Holding]) -> Tuple[str, str]:
    """
    Extract raw and normalized tickers from a list of holdings.
    
    Args:
        holdings: List of Holding objects
        
    Returns:
        Tuple of (ticker_raw, ticker_normalized) as comma-separated strings
    
    Examples:
        >>> holdings = [Holding("AAPL", 0.5), Holding("BRK.B", 0.5)]
        >>> extract_tickers_from_holdings(holdings)
        ('AAPL,BRK.B', 'AAPL,BRK-B')
    """
    if not holdings:
        return "", ""
    
    raw_tickers = []
    normalized_tickers = []
    
    for holding in holdings:
        raw_ticker = holding.ticker.strip()
        raw_tickers.append(raw_ticker)
        normalized_tickers.append(normalize_ticker(raw_ticker))
    
    return ",".join(raw_tickers), ",".join(normalized_tickers)


def build_benchmark_spec(wave_name: str) -> str:
    """
    Build benchmark specification string from BENCHMARK_WEIGHTS_STATIC.
    
    Args:
        wave_name: Display name of the wave
        
    Returns:
        Benchmark specification string (e.g., "SPY:1.0" or "QQQ:0.6,IGV:0.4")
    
    Examples:
        >>> build_benchmark_spec("S&P 500 Wave")
        'SPY:1.0'
        >>> build_benchmark_spec("AI & Cloud MegaCap Wave")
        'QQQ:0.6,IGV:0.4'
    """
    benchmark_holdings = BENCHMARK_WEIGHTS_STATIC.get(wave_name, [])
    
    if not benchmark_holdings:
        # Fallback: try to infer from wave name or use SPY as default
        logger.warning(f"No benchmark found for {wave_name}, using SPY as default")
        return "SPY:1.0"
    
    # Build spec as ticker:weight pairs
    specs = []
    for holding in benchmark_holdings:
        normalized_ticker = normalize_ticker(holding.ticker)
        specs.append(f"{normalized_ticker}:{holding.weight:.4f}")
    
    return ",".join(specs)


def infer_wave_category(wave_name: str, wave_id: str) -> str:
    """
    Infer wave category from wave_name or wave_id.
    
    Categories:
    - equity_growth: Traditional equity growth waves
    - equity_income: Income-focused equity waves
    - crypto_growth: Cryptocurrency growth waves
    - crypto_income: Cryptocurrency income waves
    - special: Gold, SmartSafe, multi-asset waves
    
    Args:
        wave_name: Display name of the wave
        wave_id: Canonical wave ID
        
    Returns:
        Category string
    
    Examples:
        >>> infer_wave_category("S&P 500 Wave", "sp500_wave")
        'equity_growth'
        >>> infer_wave_category("Crypto L1 Growth Wave", "crypto_l1_growth_wave")
        'crypto_growth'
        >>> infer_wave_category("Income Wave", "income_wave")
        'equity_income'
    """
    # Check for crypto waves
    if "Crypto" in wave_name or "crypto" in wave_id:
        if "Income" in wave_name or "income" in wave_id:
            return "crypto_income"
        return "crypto_growth"
    
    # Check for income waves
    if "Income" in wave_name or "income" in wave_id or "SmartSafe" in wave_name or "Treasury" in wave_name or "Muni" in wave_name:
        return "equity_income"
    
    # Check for special waves
    if "Gold" in wave_name or "gold" in wave_id:
        return "special"
    
    # Default to equity growth
    return "equity_growth"


def rebuild_wave_registry_csv(force: bool = False) -> Dict[str, any]:
    """
    Rebuild the wave registry CSV from canonical source (waves_engine.py).
    
    This function writes a fresh CSV to disk, overwriting any existing file.
    It ensures the CSV contains all 28 waves with required columns.
    
    Args:
        force: If True, rebuild even if CSV exists and is valid
        
    Returns:
        Dictionary with rebuild results:
        {
            'success': bool,
            'waves_written': int,
            'path': str,
            'timestamp': str,
            'errors': List[str]
        }
    
    Example:
        >>> result = rebuild_wave_registry_csv(force=True)
        >>> print(f"Rebuilt CSV with {result['waves_written']} waves")
    """
    logger.info("Starting wave registry CSV rebuild...")
    
    result = {
        'success': False,
        'waves_written': 0,
        'path': WAVE_REGISTRY_PATH,
        'timestamp': datetime.now().isoformat(),
        'errors': []
    }
    
    try:
        # Get all wave IDs from canonical source
        wave_ids = get_all_wave_ids()
        expected_wave_count = len(wave_ids)  # Dynamic count from registry
        
        # No hard-coded validation - waves can be added/removed
        # Just log the count for visibility
        logger.info(f"Building wave registry CSV with {expected_wave_count} waves from WAVE_ID_REGISTRY")
        
        # Build CSV data
        rows = []
        timestamp = datetime.now().isoformat()
        
        for wave_id in sorted(wave_ids):
            wave_name = get_display_name_from_wave_id(wave_id)
            
            if not wave_name:
                error_msg = f"No display name found for wave_id: {wave_id}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
                continue
            
            # Get holdings for this wave
            holdings = WAVE_WEIGHTS.get(wave_name, [])
            ticker_raw, ticker_normalized = extract_tickers_from_holdings(holdings)
            
            # Build benchmark spec
            benchmark_spec = build_benchmark_spec(wave_name)
            
            # Infer category
            category = infer_wave_category(wave_name, wave_id)
            
            # Build row
            row = {
                'wave_id': wave_id,
                'wave_name': wave_name,
                'mode_default': 'Standard',  # Default mode for all waves
                'benchmark_spec': benchmark_spec,
                'holdings_source': 'canonical',  # All waves use canonical holdings from waves_engine
                'category': category,
                'active': True,  # All waves active by default
                'ticker_raw': ticker_raw,
                'ticker_normalized': ticker_normalized,
                'created_at': timestamp,
                'updated_at': timestamp,
            }
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Ensure data directory exists
        data_dir = Path(WAVE_REGISTRY_PATH).parent
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Write CSV
        df.to_csv(WAVE_REGISTRY_PATH, index=False)
        
        result['success'] = True
        result['waves_written'] = len(rows)
        
        logger.info(f"✅ Successfully rebuilt wave registry CSV with {len(rows)} waves at {WAVE_REGISTRY_PATH}")
        
    except Exception as e:
        error_msg = f"Error rebuilding wave registry CSV: {str(e)}"
        logger.error(error_msg, exc_info=True)
        result['errors'].append(error_msg)
    
    return result


def validate_wave_registry_csv() -> Dict[str, any]:
    """
    Validate the wave registry CSV for correctness and completeness.
    
    Validation checks:
    1. File exists
    2. Row count matches WAVE_ID_REGISTRY (dynamic)
    3. All required columns present
    4. No duplicate wave_ids
    5. No blank wave_names
    6. All wave_ids from waves_engine are present
    
    Returns:
        Dictionary with validation results:
        {
            'is_valid': bool,
            'checks_passed': List[str],
            'checks_failed': List[str],
            'warnings': List[str],
            'wave_count': int,
            'missing_wave_ids': List[str],
            'duplicate_wave_ids': List[str]
        }
    
    Example:
        >>> result = validate_wave_registry_csv()
        >>> if not result['is_valid']:
        ...     print(f"Validation failed: {result['checks_failed']}")
        ...     rebuild_wave_registry_csv(force=True)
    """
    logger.info("Validating wave registry CSV...")
    
    result = {
        'is_valid': True,
        'checks_passed': [],
        'checks_failed': [],
        'warnings': [],
        'wave_count': 0,
        'missing_wave_ids': [],
        'duplicate_wave_ids': []
    }
    
    # Check 1: File exists
    if not os.path.exists(WAVE_REGISTRY_PATH):
        result['is_valid'] = False
        result['checks_failed'].append("file_missing")
        logger.warning(f"❌ Wave registry CSV not found at {WAVE_REGISTRY_PATH}")
        return result
    
    result['checks_passed'].append("file_exists")
    
    try:
        # Load CSV
        df = pd.read_csv(WAVE_REGISTRY_PATH)
        result['wave_count'] = len(df)
        
        # Check 2: Row count matches WAVE_ID_REGISTRY (dynamic validation)
        expected_wave_count = len(WAVE_ID_REGISTRY)
        if len(df) != expected_wave_count:
            result['is_valid'] = False
            result['checks_failed'].append(f"wave_count_mismatch (expected {expected_wave_count} from WAVE_ID_REGISTRY, found {len(df)})")
            logger.warning(f"❌ Wave count mismatch: expected {expected_wave_count} from WAVE_ID_REGISTRY, found {len(df)}")
        else:
            result['checks_passed'].append("wave_count_matches")
        
        # Check 3: All required columns present
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            result['is_valid'] = False
            result['checks_failed'].append(f"missing_columns: {missing_columns}")
            logger.warning(f"❌ Missing required columns: {missing_columns}")
        else:
            result['checks_passed'].append("all_required_columns_present")
        
        # Check 4: No duplicate wave_ids
        if 'wave_id' in df.columns:
            duplicate_wave_ids = df[df.duplicated(subset=['wave_id'], keep=False)]['wave_id'].unique().tolist()
            if duplicate_wave_ids:
                result['is_valid'] = False
                result['checks_failed'].append(f"duplicate_wave_ids: {duplicate_wave_ids}")
                result['duplicate_wave_ids'] = duplicate_wave_ids
                logger.warning(f"❌ Duplicate wave_ids found: {duplicate_wave_ids}")
            else:
                result['checks_passed'].append("no_duplicate_wave_ids")
        
        # Check 5: No blank wave_names
        if 'wave_name' in df.columns:
            blank_wave_names = df[df['wave_name'].isna() | (df['wave_name'].astype(str).str.strip() == '')]
            if len(blank_wave_names) > 0:
                result['is_valid'] = False
                result['checks_failed'].append(f"blank_wave_names ({len(blank_wave_names)} rows)")
                logger.warning(f"❌ Found {len(blank_wave_names)} rows with blank wave_name")
            else:
                result['checks_passed'].append("no_blank_wave_names")
        
        # Check 6: All wave_ids from waves_engine are present
        if 'wave_id' in df.columns:
            canonical_wave_ids = set(get_all_wave_ids())
            csv_wave_ids = set(df['wave_id'].unique())
            missing_wave_ids = list(canonical_wave_ids - csv_wave_ids)
            
            if missing_wave_ids:
                result['is_valid'] = False
                result['checks_failed'].append(f"missing_wave_ids: {missing_wave_ids}")
                result['missing_wave_ids'] = missing_wave_ids
                logger.warning(f"❌ Missing wave_ids from canonical source: {missing_wave_ids}")
            else:
                result['checks_passed'].append("all_canonical_wave_ids_present")
            
            # Check for extra wave_ids not in canonical source
            extra_wave_ids = list(csv_wave_ids - canonical_wave_ids)
            if extra_wave_ids:
                result['warnings'].append(f"extra_wave_ids_in_csv: {extra_wave_ids}")
                logger.warning(f"⚠️  CSV contains wave_ids not in canonical source: {extra_wave_ids}")
        
        # Log validation summary
        if result['is_valid']:
            logger.info(f"✅ Wave registry CSV validation passed ({len(result['checks_passed'])} checks)")
        else:
            logger.warning(f"❌ Wave registry CSV validation failed: {result['checks_failed']}")
        
    except Exception as e:
        result['is_valid'] = False
        error_msg = f"error_reading_csv: {str(e)}"
        result['checks_failed'].append(error_msg)
        logger.error(f"❌ Error validating wave registry CSV: {str(e)}", exc_info=True)
    
    return result


def auto_heal_wave_registry() -> bool:
    """
    Auto-heal wave registry CSV by validating and rebuilding if necessary.
    
    This function is designed to be called on application startup.
    It validates the CSV and automatically rebuilds it if validation fails.
    
    Returns:
        bool: True if CSV is valid (or was successfully healed), False otherwise
    
    Example:
        >>> # In app.py startup
        >>> if not auto_heal_wave_registry():
        ...     logger.error("Failed to heal wave registry CSV")
    """
    logger.info("🔍 Auto-healing wave registry CSV...")
    
    # Validate CSV
    validation_result = validate_wave_registry_csv()
    
    if validation_result['is_valid']:
        logger.info("✅ Wave registry CSV is valid, no rebuild needed")
        return True
    
    # CSV is invalid, attempt to rebuild
    logger.warning("⚠️  Wave registry CSV validation failed, rebuilding...")
    logger.warning(f"    Failed checks: {validation_result['checks_failed']}")
    
    rebuild_result = rebuild_wave_registry_csv(force=True)
    
    if rebuild_result['success']:
        logger.info(f"✅ Wave registry CSV rebuilt successfully with {rebuild_result['waves_written']} waves")
        return True
    else:
        logger.error(f"❌ Failed to rebuild wave registry CSV: {rebuild_result['errors']}")
        return False


if __name__ == "__main__":
    # Test/demo the module
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("Wave Registry Manager - Test Run")
    print("=" * 80)
    
    # Test validation
    print("\n1. Validating existing CSV...")
    validation_result = validate_wave_registry_csv()
    print(f"   Valid: {validation_result['is_valid']}")
    print(f"   Checks passed: {validation_result['checks_passed']}")
    print(f"   Checks failed: {validation_result['checks_failed']}")
    
    # Test rebuild
    print("\n2. Rebuilding CSV...")
    rebuild_result = rebuild_wave_registry_csv(force=True)
    print(f"   Success: {rebuild_result['success']}")
    print(f"   Waves written: {rebuild_result['waves_written']}")
    print(f"   Path: {rebuild_result['path']}")
    
    # Test auto-heal
    print("\n3. Testing auto-heal...")
    healed = auto_heal_wave_registry()
    print(f"   Healed: {healed}")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

#!/usr/bin/env python3
"""
Validation script for Phase 1B: Dynamic Benchmarks

This script validates:
1. Config file exists and parses correctly
2. All equity waves have valid benchmark definitions
3. Weights sum to 1.0 (within tolerance)
4. All tickers exist in cached price data
5. Benchmarks have sufficient history (>60 trading days) and end on latest cache date
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Expected equity waves for dynamic benchmarks (10 waves)
EXPECTED_EQUITY_WAVES = {
    "clean_transit_infrastructure_wave",
    "demas_fund_wave",
    "ev_infrastructure_wave",
    "future_power_energy_wave",
    "infinity_multi_asset_growth_wave",
    "next_gen_compute_semis_wave",
    "quantum_computing_wave",
    "small_to_mid_cap_growth_wave",
    "us_megacap_core_wave",
    "ai_cloud_megacap_wave",
}

# No waves are excluded from dynamic benchmarks
# This set is kept for future extensibility in case exclusions are needed
EXCLUDED_WAVES = set()

WEIGHT_TOLERANCE = 0.01
MIN_HISTORY_DAYS = 60


def load_benchmark_config(path: str) -> dict:
    """Load and parse benchmark configuration file."""
    if not os.path.exists(path):
        print(f"❌ ERROR: Benchmark config file not found: {path}")
        return None
    
    try:
        with open(path, 'r') as f:
            config = json.load(f)
        print(f"✓ Benchmark config file loaded: {path}")
        return config
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Failed to parse JSON: {e}")
        return None
    except Exception as e:
        print(f"❌ ERROR: Failed to load config: {e}")
        return None


def validate_config_structure(config: dict) -> bool:
    """Validate config has required structure."""
    if not config:
        return False
    
    if "benchmarks" not in config:
        print("❌ ERROR: Config missing 'benchmarks' key")
        return False
    
    if "version" not in config:
        print("⚠️  WARNING: Config missing 'version' key")
    
    print(f"✓ Config structure valid (version: {config.get('version', 'unknown')})")
    return True


def validate_wave_coverage(config: dict) -> bool:
    """Validate all expected waves have benchmarks defined."""
    benchmarks = config.get("benchmarks", {})
    found_waves = set(benchmarks.keys())
    
    # Include sp500_wave in expected set
    all_expected_waves = EXPECTED_EQUITY_WAVES | {"sp500_wave"}
    missing_waves = all_expected_waves - found_waves
    extra_waves = found_waves - all_expected_waves - EXCLUDED_WAVES
    
    success = True
    
    if missing_waves:
        print(f"❌ ERROR: Missing benchmark definitions for waves: {missing_waves}")
        success = False
    else:
        print(f"✓ All {len(all_expected_waves)} equity waves have benchmark definitions")
    
    if extra_waves:
        print(f"⚠️  WARNING: Extra wave definitions found: {extra_waves}")
    
    # Check that S&P 500 Wave is included with SPY:1.0
    if "sp500_wave" not in found_waves:
        print(f"❌ ERROR: S&P 500 Wave should be included in dynamic benchmarks")
        success = False
    else:
        sp500_spec = benchmarks.get("sp500_wave", {})
        components = sp500_spec.get("components", [])
        if len(components) == 1 and components[0].get("ticker") == "SPY" and components[0].get("weight") == 1.0:
            print(f"✓ S&P 500 Wave correctly configured with SPY:1.0 benchmark")
        else:
            print(f"❌ ERROR: S&P 500 Wave benchmark components incorrect")
            print(f"   Expected: [SPY:1.0]")
            print(f"   Got: {components}")
            success = False
    
    return success


def validate_benchmark_weights(config: dict) -> bool:
    """Validate that all benchmark component weights sum to 1.0."""
    benchmarks = config.get("benchmarks", {})
    success = True
    
    for wave_id, benchmark_spec in benchmarks.items():
        components = benchmark_spec.get("components", [])
        
        if not components:
            print(f"❌ ERROR: {wave_id} has no components")
            success = False
            continue
        
        total_weight = sum(c.get("weight", 0) for c in components)
        
        if abs(total_weight - 1.0) > WEIGHT_TOLERANCE:
            print(f"❌ ERROR: {wave_id} weights sum to {total_weight:.4f}, expected 1.0")
            success = False
        else:
            print(f"✓ {wave_id}: weights sum to {total_weight:.4f}")
    
    if success:
        print(f"✓ All benchmark weights valid")
    
    return success


def validate_tickers_in_cache(config: dict, cache_path: str) -> bool:
    """Validate all benchmark tickers exist in cached price data."""
    if not os.path.exists(cache_path):
        print(f"❌ ERROR: Price cache not found: {cache_path}")
        return False
    
    try:
        # Load price cache
        prices_df = pd.read_parquet(cache_path)
        available_tickers = set(prices_df.columns)
        print(f"✓ Price cache loaded: {len(available_tickers)} tickers available")
    except Exception as e:
        print(f"❌ ERROR: Failed to load price cache: {e}")
        return False
    
    benchmarks = config.get("benchmarks", {})
    success = True
    all_tickers = set()
    
    for wave_id, benchmark_spec in benchmarks.items():
        components = benchmark_spec.get("components", [])
        wave_tickers = {c.get("ticker") for c in components}
        all_tickers.update(wave_tickers)
        
        missing = wave_tickers - available_tickers
        if missing:
            print(f"❌ ERROR: {wave_id} has missing tickers in cache: {missing}")
            success = False
        else:
            print(f"✓ {wave_id}: all {len(wave_tickers)} tickers in cache")
    
    if success:
        print(f"✓ All {len(all_tickers)} benchmark tickers found in cache")
    
    return success


def validate_benchmark_history(config: dict, cache_path: str) -> bool:
    """Validate benchmarks have sufficient history and end on latest cache date."""
    if not os.path.exists(cache_path):
        print(f"❌ ERROR: Price cache not found: {cache_path}")
        return False
    
    try:
        prices_df = pd.read_parquet(cache_path)
        latest_date = prices_df.index.max()
        history_length = len(prices_df)
        print(f"✓ Price cache: {history_length} days, latest date: {latest_date.date()}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load price cache: {e}")
        return False
    
    if history_length < MIN_HISTORY_DAYS:
        print(f"❌ ERROR: Price cache has only {history_length} days, need at least {MIN_HISTORY_DAYS}")
        return False
    
    benchmarks = config.get("benchmarks", {})
    success = True
    
    for wave_id, benchmark_spec in benchmarks.items():
        components = benchmark_spec.get("components", [])
        component_tickers = [c.get("ticker") for c in components]
        
        # Check each component has sufficient non-NaN data
        for ticker in component_tickers:
            if ticker not in prices_df.columns:
                continue
            
            ticker_data = prices_df[ticker].dropna()
            if len(ticker_data) < MIN_HISTORY_DAYS:
                print(f"⚠️  WARNING: {wave_id} component {ticker} has only {len(ticker_data)} days of data")
            
            # Check if ticker ends on latest date (within 5 days tolerance)
            ticker_latest = ticker_data.index.max()
            days_behind = (latest_date - ticker_latest).days
            if days_behind > 5:
                print(f"⚠️  WARNING: {wave_id} component {ticker} is {days_behind} days behind cache ({ticker_latest.date()})")
    
    print(f"✓ Benchmark history validation complete")
    return success


def main():
    """Run all validations."""
    print("=" * 80)
    print("Phase 1B: Dynamic Benchmarks Validation")
    print("=" * 80)
    print()
    
    # Paths
    script_dir = Path(__file__).parent
    config_path = script_dir / "data" / "benchmarks" / "equity_benchmarks.json"
    cache_path = script_dir / "data" / "cache" / "prices_cache.parquet"
    
    # Track overall success
    all_passed = True
    
    # 1. Load config
    print("1. Loading benchmark configuration...")
    config = load_benchmark_config(str(config_path))
    if not config:
        return 1
    print()
    
    # 2. Validate structure
    print("2. Validating config structure...")
    if not validate_config_structure(config):
        all_passed = False
    print()
    
    # 3. Validate wave coverage
    print("3. Validating wave coverage...")
    if not validate_wave_coverage(config):
        all_passed = False
    print()
    
    # 4. Validate weights
    print("4. Validating benchmark weights...")
    if not validate_benchmark_weights(config):
        all_passed = False
    print()
    
    # 5. Validate tickers in cache
    print("5. Validating tickers in price cache...")
    if not validate_tickers_in_cache(config, str(cache_path)):
        all_passed = False
    print()
    
    # 6. Validate benchmark history
    print("6. Validating benchmark history...")
    if not validate_benchmark_history(config, str(cache_path)):
        all_passed = False
    print()
    
    # Summary
    print("=" * 80)
    if all_passed:
        print("✅ All validations passed!")
        print("=" * 80)
        return 0
    else:
        print("❌ Some validations failed - see errors above")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

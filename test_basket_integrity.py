#!/usr/bin/env python3
"""
Test script for basket integrity validation.

This script verifies that all 28 waves have:
1. Valid weight definitions
2. All tickers in the universal basket
3. Valid benchmark definitions
4. Proper weight sums (allowing SmartSafe gating)

Run with: python test_basket_integrity.py
"""

import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

def test_basket_integrity():
    """Run comprehensive basket integrity tests"""
    print("=" * 80)
    print("BASKET INTEGRITY TEST SUITE")
    print("=" * 80)
    print()
    
    # Import modules
    try:
        from helpers.basket_integrity import validate_basket_integrity
        from waves_engine import get_all_waves_universe
        import pandas as pd
    except ImportError as e:
        print(f"❌ Failed to import required modules: {e}")
        return False
    
    # Test 1: Basket integrity validation
    print("Test 1: Running basket integrity validation...")
    report = validate_basket_integrity()
    
    critical_issues = [i for i in report.issues if i.severity == 'critical']
    warnings = [i for i in report.issues if i.severity == 'warning']
    
    if critical_issues:
        print(f"❌ FAILED: Found {len(critical_issues)} critical issues")
        for issue in critical_issues:
            print(f"   - [{issue.category}] {issue.message}")
        return False
    
    print(f"✓ PASSED: No critical issues ({len(warnings)} warnings)")
    print()
    
    # Test 2: Wave registry
    print("Test 2: Verifying wave registry...")
    universe = get_all_waves_universe()
    
    if universe['count'] != 28:
        print(f"❌ FAILED: Expected 28 waves, got {universe['count']}")
        return False
    
    if universe['source'] != 'wave_registry':
        print(f"❌ FAILED: Expected source 'wave_registry', got '{universe['source']}'")
        return False
    
    print(f"✓ PASSED: Registry contains {universe['count']} waves from {universe['source']}")
    print()
    
    # Test 3: Weight definitions
    print("Test 3: Verifying weight definitions...")
    weights_df = pd.read_csv(repo_root / 'wave_weights.csv')
    waves_with_weights = set(weights_df['wave'].unique())
    expected_waves = set(universe['waves'])
    
    missing_waves = expected_waves - waves_with_weights
    if missing_waves:
        print(f"❌ FAILED: {len(missing_waves)} waves missing from wave_weights.csv")
        for wave in sorted(missing_waves):
            print(f"   - {wave}")
        return False
    
    print(f"✓ PASSED: All {len(expected_waves)} waves have weight definitions")
    print()
    
    # Test 4: Universal basket
    print("Test 4: Verifying universal basket...")
    universe_df = pd.read_csv(repo_root / 'universal_universe.csv')
    universe_tickers = set(universe_df['ticker'].unique())
    weight_tickers = set(weights_df['ticker'].unique())
    
    missing_tickers = weight_tickers - universe_tickers
    if missing_tickers:
        print(f"❌ FAILED: {len(missing_tickers)} tickers missing from universal basket")
        for ticker in sorted(missing_tickers):
            print(f"   - {ticker}")
        return False
    
    print(f"✓ PASSED: All {len(weight_tickers)} weight tickers exist in universal basket")
    print()
    
    # Test 5: Benchmark definitions
    print("Test 5: Verifying benchmark definitions...")
    config_df = pd.read_csv(repo_root / 'wave_config.csv')
    waves_with_config = set(config_df['Wave'].unique())
    
    # Allow for legacy wave names (30 total = 28 + 2 legacy)
    if len(waves_with_config) < 28:
        missing_configs = expected_waves - waves_with_config
        print(f"❌ FAILED: {len(missing_configs)} waves missing from wave_config.csv")
        for wave in sorted(missing_configs):
            print(f"   - {wave}")
        return False
    
    benchmark_tickers = set(config_df['Benchmark'].unique())
    missing_benchmarks = benchmark_tickers - universe_tickers
    if missing_benchmarks:
        print(f"❌ FAILED: {len(missing_benchmarks)} benchmark tickers missing from universal basket")
        for ticker in sorted(missing_benchmarks):
            print(f"   - {ticker}")
        return False
    
    print(f"✓ PASSED: All {len(waves_with_config)} wave configs have valid benchmarks")
    print()
    
    # Test 6: Weight sums
    print("Test 6: Verifying weight sums...")
    weight_sums = weights_df.groupby('wave')['weight'].sum()
    invalid_weights = weight_sums[(weight_sums < 0.0) | (weight_sums > 1.01)]
    
    if len(invalid_weights) > 0:
        print(f"❌ FAILED: {len(invalid_weights)} waves have invalid weight sums")
        for wave, sum_val in invalid_weights.items():
            print(f"   - {wave}: {sum_val:.4f}")
        return False
    
    smartsafe_waves = weight_sums[weight_sums < 0.99]
    print(f"✓ PASSED: All {len(weight_sums)} waves have valid weight sums")
    if len(smartsafe_waves) > 0:
        print(f"   (Note: {len(smartsafe_waves)} waves use SmartSafe gating with weights < 0.99)")
    print()
    
    # Final summary
    print("=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - {universe['count']} waves in registry")
    print(f"  - {len(weight_tickers)} unique tickers in weights")
    print(f"  - {len(universe_tickers)} total tickers in universal basket")
    print(f"  - {len(benchmark_tickers)} unique benchmark tickers")
    print(f"  - {len(waves_with_config)} waves with configuration")
    print()
    
    return True


if __name__ == "__main__":
    success = test_basket_integrity()
    sys.exit(0 if success else 1)

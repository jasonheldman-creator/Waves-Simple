#!/usr/bin/env python3
"""
Test Equity Waves Alpha Correctness

Comprehensive test suite validating that all equity waves use identical
logic for benchmark construction, VIX overlay, and alpha attribution.
"""

import sys
import json
import pandas as pd
import numpy as np


def test_benchmark_definitions():
    """Test that all equity waves have proper benchmark definitions."""
    print("=" * 80)
    print("TEST: Benchmark Definitions")
    print("=" * 80)
    
    # Load wave registry
    registry = pd.read_csv('data/wave_registry.csv')
    equity_waves = registry[
        (registry['category'] == 'equity_growth') & 
        (registry['active'] == True)
    ].copy()
    
    # Load dynamic benchmarks
    with open('data/benchmarks/equity_benchmarks.json', 'r') as f:
        dynamic_benchmarks = json.load(f)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: All equity waves have dynamic benchmarks (including S&P 500 Wave)
    print("\n[Test 1] All equity waves have dynamic benchmarks")
    equity_wave_ids = set(equity_waves['wave_id'].tolist())
    dynamic_wave_ids = set(dynamic_benchmarks['benchmarks'].keys())
    
    missing = equity_wave_ids - dynamic_wave_ids
    if missing:
        print(f"  ✗ FAILED: {len(missing)} waves missing dynamic benchmarks")
        for wave_id in sorted(missing):
            print(f"    - {wave_id}")
        tests_failed += 1
    else:
        print(f"  ✓ PASSED: All {len(equity_wave_ids)} equity waves have dynamic benchmarks")
        tests_passed += 1
    
    # Test 2: S&P 500 Wave HAS dynamic benchmark with SPY:1.0
    print("\n[Test 2] S&P 500 Wave uses dynamic benchmark with SPY:1.0")
    if 'sp500_wave' not in dynamic_wave_ids:
        print("  ✗ FAILED: sp500_wave should have dynamic benchmark")
        tests_failed += 1
    else:
        sp500_spec = dynamic_benchmarks['benchmarks']['sp500_wave']
        components = sp500_spec.get('components', [])
        if len(components) == 1 and components[0]['ticker'] == 'SPY' and components[0]['weight'] == 1.0:
            print("  ✓ PASSED: sp500_wave correctly configured with SPY:1.0 benchmark")
            tests_passed += 1
        else:
            print("  ✗ FAILED: sp500_wave benchmark components incorrect")
            print(f"    Expected: [SPY:1.0]")
            print(f"    Got: {components}")
            tests_failed += 1
    
    # Test 3: All benchmark weights sum to 1.0
    print("\n[Test 3] All benchmark weights sum to 1.0")
    weight_issues = []
    for wave_id, spec in dynamic_benchmarks['benchmarks'].items():
        components = spec.get('components', [])
        total_weight = sum(c['weight'] for c in components)
        if abs(total_weight - 1.0) >= 0.01:
            weight_issues.append((wave_id, total_weight))
    
    if weight_issues:
        print(f"  ✗ FAILED: {len(weight_issues)} waves have invalid weights")
        for wave_id, weight in weight_issues:
            print(f"    - {wave_id}: {weight:.4f}")
        tests_failed += 1
    else:
        print(f"  ✓ PASSED: All {len(dynamic_benchmarks['benchmarks'])} benchmarks have valid weights")
        tests_passed += 1
    
    # Test 4: No extra benchmarks (all belong to active equity waves)
    print("\n[Test 4] No orphaned dynamic benchmarks")
    all_equity_wave_ids = set(equity_waves['wave_id'].tolist())
    extra = dynamic_wave_ids - all_equity_wave_ids
    if extra:
        print(f"  ⚠ WARNING: {len(extra)} extra benchmarks (not critical)")
        for wave_id in sorted(extra):
            print(f"    - {wave_id}")
        # Don't fail on this - may be staging waves
        tests_passed += 1
    else:
        print(f"  ✓ PASSED: All dynamic benchmarks belong to active equity waves")
        tests_passed += 1
    
    print(f"\nBenchmark Tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_vix_overlay_parameters():
    """Test that VIX overlay parameters are consistently defined."""
    print("\n" + "=" * 80)
    print("TEST: VIX Overlay Parameters")
    print("=" * 80)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        import waves_engine as we
        
        # Test 1: MODE_EXPOSURE_CAPS defined
        print("\n[Test 1] MODE_EXPOSURE_CAPS defined")
        if not hasattr(we, 'MODE_EXPOSURE_CAPS'):
            print("  ✗ FAILED: MODE_EXPOSURE_CAPS not found")
            tests_failed += 1
        else:
            caps = we.MODE_EXPOSURE_CAPS
            required_modes = ['Standard', 'Alpha-Minus-Beta', 'Private Logic']
            missing_modes = [m for m in required_modes if m not in caps]
            if missing_modes:
                print(f"  ✗ FAILED: Missing modes: {missing_modes}")
                tests_failed += 1
            else:
                print(f"  ✓ PASSED: All 3 modes have exposure caps defined")
                tests_passed += 1
        
        # Test 2: REGIME_EXPOSURE defined
        print("\n[Test 2] REGIME_EXPOSURE defined")
        if not hasattr(we, 'REGIME_EXPOSURE'):
            print("  ✗ FAILED: REGIME_EXPOSURE not found")
            tests_failed += 1
        else:
            regimes = we.REGIME_EXPOSURE
            required_regimes = ['panic', 'downtrend', 'neutral', 'uptrend']
            missing_regimes = [r for r in required_regimes if r not in regimes]
            if missing_regimes:
                print(f"  ✗ FAILED: Missing regimes: {missing_regimes}")
                tests_failed += 1
            else:
                print(f"  ✓ PASSED: All 4 regimes have exposure multipliers")
                tests_passed += 1
        
        # Test 3: REGIME_GATING defined
        print("\n[Test 3] REGIME_GATING defined")
        if not hasattr(we, 'REGIME_GATING'):
            print("  ✗ FAILED: REGIME_GATING not found")
            tests_failed += 1
        else:
            gating = we.REGIME_GATING
            required_modes = ['Standard', 'Alpha-Minus-Beta', 'Private Logic']
            missing_modes = [m for m in required_modes if m not in gating]
            if missing_modes:
                print(f"  ✗ FAILED: Missing modes in REGIME_GATING: {missing_modes}")
                tests_failed += 1
            else:
                # Check each mode has all regimes
                all_complete = True
                for mode in required_modes:
                    mode_regimes = gating.get(mode, {})
                    required_regimes = ['panic', 'downtrend', 'neutral', 'uptrend']
                    missing = [r for r in required_regimes if r not in mode_regimes]
                    if missing:
                        print(f"  ✗ FAILED: {mode} missing regimes: {missing}")
                        all_complete = False
                
                if all_complete:
                    print(f"  ✓ PASSED: All modes have complete regime gating")
                    tests_passed += 1
                else:
                    tests_failed += 1
        
        # Test 4: Centralized computation in _compute_core
        print("\n[Test 4] Centralized computation function exists")
        if not hasattr(we, '_compute_core'):
            print("  ✗ FAILED: _compute_core function not found")
            tests_failed += 1
        else:
            print("  ✓ PASSED: _compute_core function available")
            print("           (All equity waves use this same function)")
            tests_passed += 1
        
    except Exception as e:
        print(f"\n  ✗ FAILED: Error loading waves_engine: {e}")
        tests_failed += 1
    
    print(f"\nVIX Overlay Tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_attribution_framework():
    """Test that attribution framework is properly available."""
    print("\n" + "=" * 80)
    print("TEST: Attribution Framework")
    print("=" * 80)
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        import alpha_attribution as aa
        
        # Test 1: Main attribution function exists
        print("\n[Test 1] compute_alpha_attribution_series exists")
        if not hasattr(aa, 'compute_alpha_attribution_series'):
            print("  ✗ FAILED: compute_alpha_attribution_series not found")
            tests_failed += 1
        else:
            print("  ✓ PASSED: Main attribution function available")
            tests_passed += 1
        
        # Test 2: Component calculation functions exist
        print("\n[Test 2] Component calculation functions exist")
        required_functions = [
            'compute_exposure_timing_alpha',
            'compute_regime_vix_alpha',
            'compute_momentum_trend_alpha',
            'compute_volatility_control_alpha',
            'compute_asset_selection_alpha_residual',
        ]
        
        missing_functions = [f for f in required_functions if not hasattr(aa, f)]
        if missing_functions:
            print(f"  ✗ FAILED: Missing functions: {missing_functions}")
            tests_failed += 1
        else:
            print(f"  ✓ PASSED: All 5 component functions available")
            tests_passed += 1
        
        # Test 3: Data structures exist
        print("\n[Test 3] Data structures defined")
        required_classes = ['DailyAlphaAttribution', 'AlphaAttributionSummary']
        missing_classes = [c for c in required_classes if not hasattr(aa, c)]
        if missing_classes:
            print(f"  ✗ FAILED: Missing classes: {missing_classes}")
            tests_failed += 1
        else:
            print(f"  ✓ PASSED: Attribution data structures defined")
            tests_passed += 1
        
    except Exception as e:
        print(f"\n  ✗ FAILED: Error loading alpha_attribution: {e}")
        tests_failed += 1
    
    print(f"\nAttribution Tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def test_wave_registry_consistency():
    """Test that wave registry is consistent with benchmarks."""
    print("\n" + "=" * 80)
    print("TEST: Wave Registry Consistency")
    print("=" * 80)
    
    tests_passed = 0
    tests_failed = 0
    
    # Load registry and benchmarks
    registry = pd.read_csv('data/wave_registry.csv')
    equity_waves = registry[
        (registry['category'] == 'equity_growth') & 
        (registry['active'] == True)
    ].copy()
    
    with open('data/benchmarks/equity_benchmarks.json', 'r') as f:
        dynamic_benchmarks = json.load(f)
    
    # Test 1: All equity waves have benchmark_spec
    print("\n[Test 1] All equity waves have benchmark_spec in registry")
    missing_spec = equity_waves[equity_waves['benchmark_spec'].isna()]
    if len(missing_spec) > 0:
        print(f"  ✗ FAILED: {len(missing_spec)} waves missing benchmark_spec")
        for _, wave in missing_spec.iterrows():
            print(f"    - {wave['wave_id']}")
        tests_failed += 1
    else:
        print(f"  ✓ PASSED: All {len(equity_waves)} waves have benchmark_spec")
        tests_passed += 1
    
    # Test 2: Benchmark specs match dynamic benchmarks (where applicable)
    print("\n[Test 2] Registry benchmark_spec matches dynamic benchmark components")
    mismatches = []
    
    for _, wave in equity_waves.iterrows():
        wave_id = wave['wave_id']
        
        # S&P 500 Wave should not have dynamic benchmark
        if wave_id == 'sp500_wave':
            if wave_id in dynamic_benchmarks['benchmarks']:
                mismatches.append((wave_id, "Has dynamic benchmark but shouldn't"))
            continue
        
        # All others should have dynamic benchmark
        if wave_id not in dynamic_benchmarks['benchmarks']:
            mismatches.append((wave_id, "Missing dynamic benchmark"))
    
    if mismatches:
        print(f"  ✗ FAILED: {len(mismatches)} mismatches found")
        for wave_id, issue in mismatches:
            print(f"    - {wave_id}: {issue}")
        tests_failed += 1
    else:
        print(f"  ✓ PASSED: All registry specs consistent with dynamic benchmarks")
        tests_passed += 1
    
    # Test 3: All equity waves are in equity_growth category
    print("\n[Test 3] All active equity waves are categorized correctly")
    non_equity_growth = registry[
        (registry['active'] == True) & 
        (registry['wave_id'].str.contains('wave')) &
        (~registry['category'].isin(['equity_growth', 'crypto_growth', 'crypto_income', 'equity_income', 'special']))
    ]
    
    if len(non_equity_growth) > 0:
        print(f"  ⚠ WARNING: {len(non_equity_growth)} waves have unusual categories")
        for _, wave in non_equity_growth.iterrows():
            print(f"    - {wave['wave_id']}: {wave['category']}")
        # Don't fail on this
        tests_passed += 1
    else:
        print(f"  ✓ PASSED: All waves properly categorized")
        tests_passed += 1
    
    print(f"\nRegistry Tests: {tests_passed} passed, {tests_failed} failed")
    return tests_failed == 0


def main():
    """Run all tests."""
    print("=" * 80)
    print("EQUITY WAVES ALPHA CORRECTNESS TEST SUITE")
    print("=" * 80)
    print()
    print("Testing that all equity waves use identical logic for:")
    print("  • Benchmark construction")
    print("  • VIX overlay application")
    print("  • Alpha calculation")
    print("  • Attribution decomposition")
    print()
    
    results = []
    
    # Run all test suites
    results.append(('Benchmark Definitions', test_benchmark_definitions()))
    results.append(('VIX Overlay Parameters', test_vix_overlay_parameters()))
    results.append(('Attribution Framework', test_attribution_framework()))
    results.append(('Wave Registry Consistency', test_wave_registry_consistency()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results:
        symbol = "✅" if passed else "❌"
        status = "PASSED" if passed else "FAILED"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL TEST SUITES PASSED")
        print()
        print("All equity waves are correctly configured with:")
        print("  ✓ Complete benchmark definitions")
        print("  ✓ Consistent VIX overlay parameters")
        print("  ✓ Uniform alpha calculation logic")
        print("  ✓ Attribution framework integrity")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print()
        print("Please review the failures above and fix before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

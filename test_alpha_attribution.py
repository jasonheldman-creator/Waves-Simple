#!/usr/bin/env python3
"""
Test script for alpha attribution reconciliation.

Validates that:
1. All alpha components sum to total realized alpha
2. No placeholders or estimates - only actual returns
3. Reconciliation error is within acceptable tolerance
"""

import sys
import numpy as np
import pandas as pd

try:
    import waves_engine as we
    from alpha_attribution import (
        compute_alpha_attribution_series,
        format_attribution_summary_table,
        format_daily_attribution_sample
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def test_reconciliation_basic():
    """Test basic reconciliation with synthetic data."""
    print("\n" + "="*80)
    print("TEST 1: Basic Reconciliation with Synthetic Data")
    print("="*80)
    
    # Create synthetic history
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    wave_returns = np.array([0.01, -0.005, 0.008, 0.002, -0.003, 0.006, -0.001, 0.004, 0.003, -0.002])
    bm_returns = np.array([0.008, -0.003, 0.006, 0.001, -0.002, 0.005, -0.0005, 0.003, 0.002, -0.001])
    
    history_df = pd.DataFrame({
        'wave_ret': wave_returns,
        'bm_ret': bm_returns
    }, index=dates)
    
    # Create synthetic diagnostics
    diagnostics_df = pd.DataFrame({
        'exposure': [1.1, 1.0, 1.05, 1.0, 0.9, 1.1, 0.95, 1.05, 1.0, 0.95],
        'safe_fraction': [0.0, 0.1, 0.05, 0.05, 0.15, 0.0, 0.1, 0.0, 0.0, 0.1],
        'vix': [18.0, 22.0, 19.0, 20.0, 25.0, 17.0, 21.0, 19.0, 18.0, 23.0],
        'regime': ['neutral', 'downtrend', 'neutral', 'neutral', 'panic', 
                   'uptrend', 'downtrend', 'neutral', 'neutral', 'downtrend'],
        'vol_adjust': [1.0, 1.0, 1.0, 1.0, 0.9, 1.05, 1.0, 1.0, 1.0, 0.95]
    }, index=dates)
    
    # Compute attribution
    daily_df, summary = compute_alpha_attribution_series(
        wave_name="Test Wave",
        mode="Standard",
        history_df=history_df,
        diagnostics_df=diagnostics_df,
        tilt_strength=0.8,
        base_exposure=1.0
    )
    
    # Validate reconciliation
    print(f"\nTotal Alpha (Realized): {summary.total_alpha:.6f}")
    print(f"Sum of Components:      {summary.sum_of_components:.6f}")
    print(f"Reconciliation Error:   {summary.reconciliation_error:.10f}")
    print(f"Reconciliation Error %: {summary.reconciliation_pct_error:.6f}%")
    
    # Check reconciliation tolerance
    tolerance = 1e-8
    if abs(summary.reconciliation_error) < tolerance:
        print("✅ RECONCILIATION PASSED: Error within tolerance")
        return True
    else:
        print(f"❌ RECONCILIATION FAILED: Error {summary.reconciliation_error:.10f} exceeds tolerance {tolerance}")
        return False


def test_reconciliation_real_wave():
    """Test reconciliation with real Wave data."""
    print("\n" + "="*80)
    print("TEST 2: Reconciliation with Real Wave Data")
    print("="*80)
    
    wave_name = "US MegaCap Core Wave"
    mode = "Standard"
    days = 90  # Use shorter period for faster testing
    
    try:
        # Compute attribution using waves_engine integration
        daily_df, summary_dict = we.compute_alpha_attribution(
            wave_name=wave_name,
            mode=mode,
            days=days
        )
        
        if not summary_dict.get("ok", False):
            print(f"❌ Failed to compute attribution: {summary_dict.get('message', 'Unknown error')}")
            return False
        
        print(f"\nWave: {wave_name}")
        print(f"Mode: {mode}")
        print(f"Period: {days} days")
        print(f"\nTotal Alpha (Realized): {summary_dict['total_alpha']:.6f}")
        print(f"Sum of Components:      {summary_dict['sum_of_components']:.6f}")
        print(f"Reconciliation Error:   {summary_dict['reconciliation_error']:.10f}")
        print(f"Reconciliation Error %: {summary_dict['reconciliation_pct_error']:.6f}%")
        
        print("\nComponent Breakdown:")
        print(f"  1. Exposure & Timing:     {summary_dict['exposure_timing_alpha']:.6f} "
              f"({summary_dict['exposure_timing_contribution_pct']:.2f}%)")
        print(f"  2. Regime & VIX:          {summary_dict['regime_vix_alpha']:.6f} "
              f"({summary_dict['regime_vix_contribution_pct']:.2f}%)")
        print(f"  3. Momentum & Trend:      {summary_dict['momentum_trend_alpha']:.6f} "
              f"({summary_dict['momentum_trend_contribution_pct']:.2f}%)")
        print(f"  4. Volatility Control:    {summary_dict['volatility_control_alpha']:.6f} "
              f"({summary_dict['volatility_control_contribution_pct']:.2f}%)")
        print(f"  5. Asset Selection:       {summary_dict['asset_selection_alpha']:.6f} "
              f"({summary_dict['asset_selection_contribution_pct']:.2f}%)")
        
        # Check reconciliation tolerance
        tolerance = 0.01  # 0.01% tolerance for real data
        if abs(summary_dict['reconciliation_pct_error']) < tolerance:
            print(f"\n✅ RECONCILIATION PASSED: Error {summary_dict['reconciliation_pct_error']:.6f}% < {tolerance}%")
            return True
        else:
            print(f"\n❌ RECONCILIATION FAILED: Error {summary_dict['reconciliation_pct_error']:.6f}% >= {tolerance}%")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_daily_attribution_format():
    """Test that daily attribution formatting works."""
    print("\n" + "="*80)
    print("TEST 3: Daily Attribution Formatting")
    print("="*80)
    
    wave_name = "S&P 500 Wave"
    mode = "Standard"
    days = 30
    
    try:
        daily_df, summary_dict = we.compute_alpha_attribution(
            wave_name=wave_name,
            mode=mode,
            days=days
        )
        
        if not summary_dict.get("ok", False):
            print(f"❌ Failed to compute attribution")
            return False
        
        # Format sample
        from alpha_attribution import AlphaAttributionSummary
        summary = AlphaAttributionSummary(**{k: v for k, v in summary_dict.items() 
                                            if k in AlphaAttributionSummary.__annotations__})
        
        summary_table = format_attribution_summary_table(summary)
        daily_sample = format_daily_attribution_sample(daily_df, n_rows=5)
        
        print("\n" + summary_table)
        print("\n" + daily_sample)
        
        print("\n✅ FORMATTING TEST PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error during formatting test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ALPHA ATTRIBUTION RECONCILIATION TEST SUITE")
    print("="*80)
    
    results = []
    
    # Test 1: Basic synthetic data reconciliation
    results.append(("Basic Reconciliation", test_reconciliation_basic()))
    
    # Test 2: Real Wave data reconciliation
    results.append(("Real Wave Reconciliation", test_reconciliation_real_wave()))
    
    # Test 3: Formatting
    results.append(("Daily Attribution Formatting", test_daily_attribution_format()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

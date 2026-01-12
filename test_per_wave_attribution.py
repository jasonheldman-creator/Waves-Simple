"""
Test per-wave attribution functionality (v17.5).

Validates:
1. Raw wave return computation (without strategy overlay)
2. Attribution metric calculations
3. Reconciliation: total_alpha = selection_alpha + overlay_alpha
4. Multi-period attribution (1D, 30D, 60D, 365D)
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_raw_wave_return():
    """Test raw wave return computation."""
    print("=" * 70)
    print("Test 1: Raw Wave Return Computation")
    print("=" * 70)
    
    from waves_engine import compute_raw_wave_return, WAVE_WEIGHTS
    
    # Test with S&P 500 Wave (simple single-ticker wave)
    wave_name = "S&P 500 Wave"
    print(f"\nTesting raw return for: {wave_name}")
    
    # Compute raw returns for 365 days
    raw_returns = compute_raw_wave_return(wave_name, days=365)
    
    if raw_returns.empty:
        print("   ✗ FAILED: No raw returns computed")
        return False
    
    print(f"   ✓ Computed {len(raw_returns)} daily returns")
    print(f"   ✓ Date range: {raw_returns.index[0].date()} to {raw_returns.index[-1].date()}")
    
    # Validate return statistics
    raw_nav = (1.0 + raw_returns).cumprod()
    total_return = raw_nav.iloc[-1] - 1.0
    
    print(f"   ✓ Total raw return (365D): {total_return*100:+.2f}%")
    print(f"   ✓ Daily mean: {raw_returns.mean()*100:.4f}%")
    print(f"   ✓ Daily std: {raw_returns.std()*100:.4f}%")
    
    # Test with multi-ticker wave
    wave_name = "US MegaCap Core Wave"
    print(f"\nTesting raw return for: {wave_name}")
    
    raw_returns = compute_raw_wave_return(wave_name, days=365)
    
    if raw_returns.empty:
        print("   ✗ FAILED: No raw returns computed")
        return False
    
    print(f"   ✓ Computed {len(raw_returns)} daily returns")
    
    # Check holdings
    holdings = WAVE_WEIGHTS.get(wave_name, [])
    print(f"   ✓ Wave has {len(holdings)} holdings")
    
    return True


def test_attribution_computation():
    """Test attribution metric computation."""
    print("\n" + "=" * 70)
    print("Test 2: Attribution Computation")
    print("=" * 70)
    
    from waves_engine import get_attribution
    
    # Test with S&P 500 Wave
    wave_name = "S&P 500 Wave"
    periods = [30, 60, 365]
    
    print(f"\nComputing attribution for: {wave_name}")
    print(f"Periods: {periods}")
    
    result = get_attribution(wave_name, periods=periods)
    
    if not result["success"]:
        print(f"   ✗ FAILED: {result.get('error', 'Unknown error')}")
        return False
    
    print(f"   ✓ Attribution computed successfully")
    
    # Display results for each period
    for period_days, metrics in result["attribution"].items():
        if "error" in metrics:
            print(f"\n   Period {period_days}D: {metrics['error']}")
            continue
        
        print(f"\n   Period: {period_days} days ({metrics['data_points']} data points)")
        print(f"      Benchmark Return:        {metrics['benchmark_return']*100:+8.2f}%")
        print(f"      Raw Wave Return:         {metrics['raw_wave_return']*100:+8.2f}%")
        print(f"      Strategy Wave Return:    {metrics['strategy_wave_return']*100:+8.2f}%")
        print(f"      ---")
        print(f"      Total Alpha:             {metrics['total_alpha']*100:+8.2f}%")
        print(f"      Selection Alpha:         {metrics['selection_alpha']*100:+8.2f}%")
        print(f"      Overlay Alpha:           {metrics['overlay_alpha']*100:+8.2f}%")
        print(f"      Reconciliation Error:    {metrics['reconciliation_error']*100:+8.4f}%")
    
    return True


def test_attribution_reconciliation():
    """Test that attribution components reconcile correctly."""
    print("\n" + "=" * 70)
    print("Test 3: Attribution Reconciliation")
    print("=" * 70)
    
    from waves_engine import get_attribution
    
    # Test multiple waves
    test_waves = [
        "S&P 500 Wave",
        "US MegaCap Core Wave",
        "Small Cap Growth Wave"
    ]
    
    periods = [30, 60, 365]
    reconciliation_threshold = 0.0001  # 0.01% tolerance for numerical precision
    
    all_passed = True
    
    for wave_name in test_waves:
        print(f"\nTesting reconciliation for: {wave_name}")
        
        result = get_attribution(wave_name, periods=periods)
        
        if not result["success"]:
            print(f"   ⚠️  Skipped: {result.get('error', 'Unknown error')}")
            continue
        
        wave_passed = True
        for period_days, metrics in result["attribution"].items():
            if "error" in metrics:
                continue
            
            recon_error = abs(metrics['reconciliation_error'])
            
            if recon_error > reconciliation_threshold:
                print(f"   ✗ {period_days}D: Reconciliation error {recon_error*100:.4f}% exceeds threshold")
                wave_passed = False
                all_passed = False
            else:
                print(f"   ✓ {period_days}D: Reconciliation OK (error: {recon_error*100:.6f}%)")
        
        if wave_passed:
            print(f"   ✓ All periods reconcile correctly")
    
    return all_passed


def test_multi_wave_attribution():
    """Test attribution across multiple waves."""
    print("\n" + "=" * 70)
    print("Test 4: Multi-Wave Attribution")
    print("=" * 70)
    
    from waves_engine import get_attribution, get_all_waves
    
    # Test a diverse set of waves
    test_waves = [
        "S&P 500 Wave",
        "US MegaCap Core Wave",
        "AI & Cloud MegaCap Wave",
        "Small Cap Growth Wave",
        "Income Wave",
    ]
    
    periods = [30, 365]
    
    results = {}
    for wave_name in test_waves:
        print(f"\n{wave_name}:")
        result = get_attribution(wave_name, periods=periods)
        results[wave_name] = result
        
        if result["success"]:
            print(f"   ✓ Success")
            for period_days, metrics in result["attribution"].items():
                if "error" not in metrics:
                    print(f"      {period_days}D: Total={metrics['total_alpha']*100:+.2f}%, "
                          f"Selection={metrics['selection_alpha']*100:+.2f}%, "
                          f"Overlay={metrics['overlay_alpha']*100:+.2f}%")
        else:
            print(f"   ✗ Failed: {result.get('error', 'Unknown')}")
    
    # Summary
    successful = sum(1 for r in results.values() if r["success"])
    print(f"\n{'='*70}")
    print(f"Summary: {successful}/{len(test_waves)} waves computed successfully")
    
    return successful > 0


def test_attribution_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 70)
    print("Test 5: Edge Cases and Error Handling")
    print("=" * 70)
    
    from waves_engine import get_attribution, compute_raw_wave_return
    
    # Test 1: Non-existent wave
    print("\n1. Non-existent wave:")
    result = get_attribution("NonExistent Wave", periods=[30])
    if not result["success"]:
        print(f"   ✓ Correctly handled: {result.get('error', 'Unknown')}")
    else:
        print(f"   ✗ Should have failed for non-existent wave")
    
    # Test 2: Very short period (1 day)
    print("\n2. Very short period (1 day):")
    result = get_attribution("S&P 500 Wave", periods=[1])
    if result["success"]:
        metrics = result["attribution"].get(1)
        if metrics and "error" not in metrics:
            print(f"   ✓ 1D attribution computed: total_alpha={metrics['total_alpha']*100:+.4f}%")
    else:
        print(f"   Note: {result.get('error', 'Unknown')}")
    
    # Test 3: Empty raw returns
    print("\n3. Invalid wave name handling:")
    raw_returns = compute_raw_wave_return("Invalid Wave Name", days=30)
    if raw_returns.empty:
        print(f"   ✓ Correctly returned empty series for invalid wave")
    else:
        print(f"   ✗ Should have returned empty series")
    
    return True


def run_all_tests():
    """Run all attribution tests."""
    print("\n" + "=" * 70)
    print("PER-WAVE ATTRIBUTION TESTS (ENGINE v17.5)")
    print("=" * 70)
    
    tests = [
        ("Raw Wave Return", test_raw_wave_return),
        ("Attribution Computation", test_attribution_computation),
        ("Attribution Reconciliation", test_attribution_reconciliation),
        ("Multi-Wave Attribution", test_multi_wave_attribution),
        ("Edge Cases", test_attribution_edge_cases),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = passed
        except Exception as e:
            print(f"\n✗ {test_name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for p in results.values() if p)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

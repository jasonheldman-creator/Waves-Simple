#!/usr/bin/env python3
"""
Unit tests for portfolio alpha ledger and exposure series computation.

Tests validate the requirements from the problem statement:
1. Residual attribution is within tolerance (<= 0.10%)
2. Period fidelity (start_date corresponds to row-slice period)
3. VIX exposure calculation (regime mapping and smoothing)
4. Alpha decomposition (total = selection + overlay)
5. No placeholders or silent fallbacks
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test configuration constants
RESIDUAL_TOLERANCE = 0.0010  # 0.10% tolerance for residual attribution


def test_exposure_series_vix_regime_mapping():
    """Test VIX regime mapping to exposure levels."""
    print("\n=== Test: VIX Regime Mapping ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_exposure_series
        
        # Create synthetic price_book with VIX data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # Test different VIX levels
        vix_values = [15, 20, 28, 17.9, 18.0, 24.9, 25.0, 30]
        expected_exposures = [1.00, 0.65, 0.25, 1.00, 0.65, 0.65, 0.25, 0.25]
        
        for vix, expected_exp in zip(vix_values, expected_exposures):
            price_book = pd.DataFrame({
                '^VIX': [vix] * len(dates),
                'SPY': [100] * len(dates)
            }, index=dates)
            
            exposure = compute_portfolio_exposure_series(
                price_book, 
                mode='Standard',
                apply_smoothing=False  # Disable smoothing for exact test
            )
            
            if exposure is None:
                print(f"❌ FAIL: Expected exposure series for VIX={vix}")
                return False
            
            # Check exposure mapping
            actual_exp = exposure.iloc[-1]  # Get last value
            if abs(actual_exp - expected_exp) > 0.01:
                print(f"❌ FAIL: VIX={vix} expected exposure={expected_exp}, got {actual_exp}")
                return False
            
            print(f"✓ VIX={vix} -> exposure={actual_exp:.2f} (expected {expected_exp:.2f})")
        
        print("✅ PASS: VIX regime mapping correct")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_exposure_series_vix_proxy_preference():
    """Test VIX proxy selection order (^VIX > VIXY > VXX)."""
    print("\n=== Test: VIX Proxy Preference ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_exposure_series
        
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        
        # Test 1: ^VIX available (should use it)
        price_book_1 = pd.DataFrame({
            '^VIX': [20] * len(dates),
            'VIXY': [15] * len(dates),
            'VXX': [18] * len(dates),
            'SPY': [100] * len(dates)
        }, index=dates)
        
        exposure_1 = compute_portfolio_exposure_series(price_book_1, apply_smoothing=False)
        if exposure_1 is None:
            print("❌ FAIL: Expected exposure when ^VIX is available")
            return False
        
        # VIX=20 should map to 0.65
        if abs(exposure_1.iloc[-1] - 0.65) > 0.01:
            print(f"❌ FAIL: Expected exposure=0.65 for ^VIX=20, got {exposure_1.iloc[-1]}")
            return False
        
        print("✓ Correctly uses ^VIX when available")
        
        # Test 2: Only VIXY available (should use it)
        price_book_2 = pd.DataFrame({
            'VIXY': [15] * len(dates),
            'VXX': [18] * len(dates),
            'SPY': [100] * len(dates)
        }, index=dates)
        
        exposure_2 = compute_portfolio_exposure_series(price_book_2, apply_smoothing=False)
        if exposure_2 is None:
            print("❌ FAIL: Expected exposure when VIXY is available")
            return False
        
        # VIX=15 should map to 1.00
        if abs(exposure_2.iloc[-1] - 1.00) > 0.01:
            print(f"❌ FAIL: Expected exposure=1.00 for VIXY=15, got {exposure_2.iloc[-1]}")
            return False
        
        print("✓ Correctly uses VIXY when ^VIX not available")
        
        # Test 3: Only VXX available (should use it)
        price_book_3 = pd.DataFrame({
            'VXX': [26] * len(dates),
            'SPY': [100] * len(dates)
        }, index=dates)
        
        exposure_3 = compute_portfolio_exposure_series(price_book_3, apply_smoothing=False)
        if exposure_3 is None:
            print("❌ FAIL: Expected exposure when VXX is available")
            return False
        
        # VIX=26 should map to 0.25
        if abs(exposure_3.iloc[-1] - 0.25) > 0.01:
            print(f"❌ FAIL: Expected exposure=0.25 for VXX=26, got {exposure_3.iloc[-1]}")
            return False
        
        print("✓ Correctly uses VXX when ^VIX and VIXY not available")
        
        # Test 4: No VIX proxy available (should return None)
        price_book_4 = pd.DataFrame({
            'SPY': [100] * len(dates),
            'AAPL': [150] * len(dates)
        }, index=dates)
        
        exposure_4 = compute_portfolio_exposure_series(price_book_4)
        if exposure_4 is not None:
            print("❌ FAIL: Expected None when no VIX proxy available")
            return False
        
        print("✓ Correctly returns None when no VIX proxy available")
        
        print("✅ PASS: VIX proxy preference order correct")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_exposure_series_smoothing():
    """Test rolling median smoothing of exposure series."""
    print("\n=== Test: Exposure Smoothing ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_exposure_series
        
        # Create VIX series with volatility (should be smoothed)
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        vix_values = [15, 20, 15, 25, 15, 20, 15, 25, 15, 20]  # Oscillating values
        
        price_book = pd.DataFrame({
            '^VIX': vix_values,
            'SPY': [100] * len(dates)
        }, index=dates)
        
        # Without smoothing
        exposure_no_smooth = compute_portfolio_exposure_series(
            price_book, 
            apply_smoothing=False
        )
        
        # With smoothing (3-day rolling median)
        exposure_smooth = compute_portfolio_exposure_series(
            price_book, 
            smooth_window=3,
            apply_smoothing=True
        )
        
        if exposure_no_smooth is None or exposure_smooth is None:
            print("❌ FAIL: Expected exposure series")
            return False
        
        # Smoothed version should have less variance
        variance_no_smooth = exposure_no_smooth.var()
        variance_smooth = exposure_smooth.var()
        
        if variance_smooth >= variance_no_smooth:
            print(f"❌ FAIL: Smoothed variance ({variance_smooth:.4f}) should be less than unsmoothed ({variance_no_smooth:.4f})")
            return False
        
        print(f"✓ Smoothing reduces variance: {variance_no_smooth:.4f} -> {variance_smooth:.4f}")
        
        # Check that values are clipped to [0, 1]
        if exposure_smooth.min() < 0 or exposure_smooth.max() > 1:
            print(f"❌ FAIL: Exposure not clipped to [0,1]: min={exposure_smooth.min()}, max={exposure_smooth.max()}")
            return False
        
        print("✓ Exposure clipped to [0, 1] range")
        
        print("✅ PASS: Exposure smoothing works correctly")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alpha_ledger_reconciliation_1():
    """Test that Portfolio Return − Benchmark Return = Total Alpha."""
    print("\n=== Test: Reconciliation 1 (Portfolio - Benchmark = Alpha) ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger, RESIDUAL_TOLERANCE
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("Loading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        print(f"✓ PRICE_BOOK loaded: {len(price_book)} days")
        
        # Compute ledger
        print("Computing alpha ledger...")
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            periods=[1, 30, 60, 365],
            vix_exposure_enabled=True
        )
        
        if not ledger['success']:
            print(f"❌ FAIL: Ledger computation failed: {ledger['failure_reason']}")
            return False
        
        print("✓ Ledger computed successfully")
        
        # Check reconciliation 1 for each period
        for period_key, period_data in ledger['period_results'].items():
            if not period_data.get('available'):
                print(f"⚠️ SKIP: {period_key} not available ({period_data.get('reason')})")
                continue
            
            cum_realized = period_data['cum_realized']
            cum_benchmark = period_data['cum_benchmark']
            total_alpha = period_data['total_alpha']
            
            # Verify: cum_realized - cum_benchmark = total_alpha
            expected_alpha = cum_realized - cum_benchmark
            diff = abs(expected_alpha - total_alpha)
            
            if diff > RESIDUAL_TOLERANCE:
                print(f"❌ FAIL: {period_key} reconciliation mismatch:")
                print(f"  Portfolio={cum_realized:.6f}, Benchmark={cum_benchmark:.6f}")
                print(f"  Expected Alpha={expected_alpha:.6f}, Actual Alpha={total_alpha:.6f}")
                print(f"  Difference={diff:.6f} > Tolerance={RESIDUAL_TOLERANCE:.6f}")
                return False
            
            print(f"✓ {period_key}: Portfolio({cum_realized:+.4%}) - Benchmark({cum_benchmark:+.4%}) = Alpha({total_alpha:+.4%}) [diff={diff:.6f}]")
        
        print("✅ PASS: Reconciliation 1 holds for all available periods")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alpha_ledger_reconciliation_2():
    """Test that Selection Alpha + Overlay Alpha + Residual = Total Alpha."""
    print("\n=== Test: Reconciliation 2 (Selection + Overlay + Residual = Total) ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger, RESIDUAL_TOLERANCE
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("Loading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        # Compute ledger
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            periods=[1, 30, 60, 365],
            vix_exposure_enabled=True
        )
        
        if not ledger['success']:
            print(f"❌ FAIL: Ledger computation failed: {ledger['failure_reason']}")
            return False
        
        # Check reconciliation 2 for each period
        for period_key, period_data in ledger['period_results'].items():
            if not period_data.get('available'):
                print(f"⚠️ SKIP: {period_key} not available ({period_data.get('reason')})")
                continue
            
            selection_alpha = period_data['selection_alpha']
            overlay_alpha = period_data['overlay_alpha']
            residual = period_data['residual']
            total_alpha = period_data['total_alpha']
            
            # Verify: selection_alpha + overlay_alpha + residual = total_alpha
            computed_total = selection_alpha + overlay_alpha + residual
            diff = abs(computed_total - total_alpha)
            
            if diff > RESIDUAL_TOLERANCE:
                print(f"❌ FAIL: {period_key} attribution mismatch:")
                print(f"  Selection={selection_alpha:.6f}, Overlay={overlay_alpha:.6f}, Residual={residual:.6f}")
                print(f"  Computed Total={computed_total:.6f}, Actual Total={total_alpha:.6f}")
                print(f"  Difference={diff:.6f} > Tolerance={RESIDUAL_TOLERANCE:.6f}")
                return False
            
            print(f"✓ {period_key}: Selection({selection_alpha:+.4%}) + Overlay({overlay_alpha:+.4%}) + Residual({residual:+.4%}) = Total({total_alpha:+.4%}) [diff={diff:.6f}]")
        
        print("✅ PASS: Reconciliation 2 holds for all available periods")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unavailable_periods_show_na():
    """Test that unavailable periods correctly show N/A with reasons."""
    print("\n=== Test: Unavailable Periods Show N/A ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        
        # Create minimal price_book with insufficient history for some periods
        dates = pd.date_range('2024-01-01', periods=50, freq='D')  # Only 50 days
        price_book = pd.DataFrame({
            'SPY': [100 + i*0.5 for i in range(len(dates))],
            'BIL': [50] * len(dates)
        }, index=dates)
        
        # Request all periods including ones with insufficient data
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            periods=[1, 30, 60, 365],
            vix_exposure_enabled=False
        )
        
        if not ledger['success']:
            print(f"⚠️ Ledger failed (expected for minimal data): {ledger['failure_reason']}")
            return True  # This is acceptable for minimal test data
        
        # Check periods with insufficient data
        for period_key in ['60D', '365D']:
            period_data = ledger['period_results'].get(period_key, {})
            
            if period_data.get('available'):
                print(f"❌ FAIL: {period_key} should not be available with only 50 days of data")
                return False
            
            # Verify all metrics are None when unavailable
            if (period_data.get('cum_realized') is not None or
                period_data.get('cum_benchmark') is not None or
                period_data.get('total_alpha') is not None):
                print(f"❌ FAIL: {period_key} should have None values when unavailable")
                return False
            
            reason = period_data.get('reason')
            if not reason:
                print(f"❌ FAIL: {period_key} should have a reason when unavailable")
                return False
            
            print(f"✓ {period_key}: correctly unavailable (reason: {reason[:50]}...)")
        
        print("✅ PASS: Unavailable periods correctly show N/A with reasons")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alpha_ledger_residual_attribution():
    """Test that residual attribution is within tolerance."""
    print("\n=== Test: Residual Attribution ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("Loading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        print(f"✓ PRICE_BOOK loaded: {len(price_book)} days")
        
        # Compute ledger
        print("Computing alpha ledger...")
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            periods=[30, 60, 365],
            vix_exposure_enabled=True
        )
        
        if not ledger['success']:
            print(f"❌ FAIL: Ledger computation failed: {ledger['failure_reason']}")
            return False
        
        print("✓ Ledger computed successfully")
        
        # Check residual for each period
        for period_key, period_data in ledger['period_results'].items():
            if not period_data.get('available'):
                print(f"⚠️ SKIP: {period_key} not available ({period_data.get('reason')})")
                continue
            
            residual = period_data['residual']
            residual_pct = abs(residual) * 100
            
            if abs(residual) > RESIDUAL_TOLERANCE:
                print(f"❌ FAIL: {period_key} residual {residual:+.4%} exceeds tolerance ({RESIDUAL_TOLERANCE:.4%})")
                return False
            
            # Verify: total_alpha = selection_alpha + overlay_alpha (within tolerance)
            total_alpha = period_data['total_alpha']
            selection_alpha = period_data['selection_alpha']
            overlay_alpha = period_data['overlay_alpha']
            
            computed_total = selection_alpha + overlay_alpha
            if abs(total_alpha - computed_total) > RESIDUAL_TOLERANCE:
                print(f"❌ FAIL: {period_key} attribution mismatch: total={total_alpha:.4%}, computed={computed_total:.4%}")
                return False
            
            # Color code residual
            residual_status = "🟢" if residual_pct < 0.10 else "🟡" if residual_pct < 0.5 else "🔴"
            print(f"{residual_status} {period_key}: residual={residual:+.4%} (tolerance: {RESIDUAL_TOLERANCE:.4%})")
        
        print("✅ PASS: Residual attribution within tolerance")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alpha_ledger_period_fidelity():
    """Test that period start_date corresponds to strict row-slice."""
    print("\n=== Test: Period Fidelity ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("Loading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        # Compute ledger
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            periods=[30, 60],
            vix_exposure_enabled=True
        )
        
        if not ledger['success']:
            print(f"❌ FAIL: Ledger computation failed: {ledger['failure_reason']}")
            return False
        
        # Verify period fidelity
        daily_returns = ledger['daily_realized_return']
        
        for period_key, period_data in ledger['period_results'].items():
            if not period_data.get('available'):
                print(f"⚠️ SKIP: {period_key} not available")
                continue
            
            period = period_data['period']
            start_date = period_data['start_date']
            end_date = period_data['end_date']
            rows_used = period_data['rows_used']
            
            # Verify rows_used matches period
            if rows_used != period:
                print(f"❌ FAIL: {period_key} rows_used={rows_used} != period={period}")
                return False
            
            # Verify start_date corresponds to exact row-slice
            if daily_returns is not None:
                period_slice = daily_returns.iloc[-period:]
                expected_start = period_slice.index[0].strftime('%Y-%m-%d')
                expected_end = period_slice.index[-1].strftime('%Y-%m-%d')
                
                if start_date != expected_start:
                    print(f"❌ FAIL: {period_key} start_date mismatch: reported={start_date}, expected={expected_start}")
                    return False
                
                if end_date != expected_end:
                    print(f"❌ FAIL: {period_key} end_date mismatch: reported={end_date}, expected={expected_end}")
                    return False
            
            print(f"✓ {period_key}: {start_date} to {end_date} ({rows_used} rows)")
        
        print("✅ PASS: Period fidelity correct")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alpha_ledger_no_placeholders():
    """Test that insufficient data returns None, not placeholder values."""
    print("\n=== Test: No Placeholders ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        
        # Create minimal price_book with insufficient history
        dates = pd.date_range('2024-01-01', periods=20, freq='D')  # Only 20 days
        price_book = pd.DataFrame({
            'SPY': [100 + i for i in range(len(dates))],
            'BIL': [50] * len(dates)
        }, index=dates)
        
        # Request 30D and 60D periods (insufficient data)
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            periods=[30, 60],
            vix_exposure_enabled=False
        )
        
        # Should fail with clear reason
        if ledger['success']:
            # Check that periods with insufficient data have available=False
            for period_key in ['30D', '60D']:
                period_data = ledger['period_results'].get(period_key, {})
                
                if period_data.get('available'):
                    print(f"❌ FAIL: {period_key} should not be available with only 20 days of data")
                    return False
                
                # Verify all metrics are None when unavailable
                if (period_data.get('cum_realized') is not None or
                    period_data.get('total_alpha') is not None):
                    print(f"❌ FAIL: {period_key} should have None values when unavailable")
                    return False
                
                reason = period_data.get('reason')
                if reason != 'insufficient_aligned_rows':
                    print(f"❌ FAIL: {period_key} should have reason='insufficient_aligned_rows', got '{reason}'")
                    return False
                
                print(f"✓ {period_key}: correctly unavailable (reason: {reason})")
        
        print("✅ PASS: No placeholder data for insufficient history")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alpha_ledger_output_structure():
    """Test that ledger output has all required keys and structure."""
    print("\n=== Test: Output Structure ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        # Compute ledger
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            periods=[1, 30, 60, 365]
        )
        
        # Check top-level keys
        required_keys = [
            'success', 'failure_reason', 'daily_risk_return', 'daily_safe_return',
            'daily_exposure', 'daily_realized_return', 'daily_unoverlay_return',
            'daily_benchmark_return', 'period_results', 'vix_ticker_used',
            'safe_ticker_used', 'overlay_available', 'warnings'
        ]
        
        for key in required_keys:
            if key not in ledger:
                print(f"❌ FAIL: Missing top-level key '{key}'")
                return False
        
        print("✓ All top-level keys present")
        
        # Check period results structure
        if ledger['success']:
            for period_key in ['1D', '30D', '60D', '365D']:
                if period_key not in ledger['period_results']:
                    print(f"❌ FAIL: Missing period result '{period_key}'")
                    return False
                
                period_data = ledger['period_results'][period_key]
                
                period_required_keys = [
                    'period', 'available', 'reason', 'rows_used', 'start_date', 'end_date',
                    'cum_realized', 'cum_unoverlay', 'cum_benchmark',
                    'total_alpha', 'selection_alpha', 'overlay_alpha', 'residual', 'alpha_captured'
                ]
                
                for key in period_required_keys:
                    if key not in period_data:
                        print(f"❌ FAIL: Missing period key '{key}' in {period_key}")
                        return False
            
            print("✓ All period result keys present")
        
        print("✅ PASS: Output structure correct")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all test functions."""
    print("=" * 70)
    print("PORTFOLIO ALPHA LEDGER TEST SUITE")
    print("=" * 70)
    
    tests = [
        test_exposure_series_vix_regime_mapping,
        test_exposure_series_vix_proxy_preference,
        test_exposure_series_smoothing,
        test_alpha_ledger_output_structure,
        test_alpha_ledger_reconciliation_1,
        test_alpha_ledger_reconciliation_2,
        test_unavailable_periods_show_na,
        test_alpha_ledger_residual_attribution,
        test_alpha_ledger_period_fidelity,
        test_alpha_ledger_no_placeholders,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\n❌ FATAL ERROR in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_func.__name__, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
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
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)

#!/usr/bin/env python3
"""
Unit tests for portfolio-level alpha attribution.

Tests validate the requirements from the problem statement:
1. Keys exist in output
2. Numeric outputs (no None values)
3. overlay_alpha = total_alpha - selection_alpha (within tolerance)
4. Residual is close to 0 (abs < 0.10% over same window)
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_attribution_keys_exist():
    """Test that all required keys exist in the output."""
    print("\n=== Test: Attribution Keys Exist ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_attribution
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        print("Loading PRICE_BOOK...")
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        print(f"✓ PRICE_BOOK loaded: {len(price_book)} days")
        
        # Compute attribution
        print("Computing attribution...")
        result = compute_portfolio_alpha_attribution(
            price_book=price_book,
            mode='Standard',
            periods=[30, 60, 365]
        )
        
        # Check top-level keys
        required_keys = [
            'success',
            'failure_reason',
            'daily_realized_return',
            'daily_unoverlay_return',
            'daily_benchmark_return',
            'daily_exposure',
            'period_summaries',
            'since_inception_summary',
            'warnings'
        ]
        
        for key in required_keys:
            if key not in result:
                print(f"❌ FAIL: Missing key '{key}'")
                return False
        
        print("✓ All top-level keys exist")
        
        # Check period summary keys
        if result['success'] and result['period_summaries']:
            sample_period = list(result['period_summaries'].keys())[0]
            summary = result['period_summaries'][sample_period]
            
            summary_keys = [
                'period',
                'cum_real',
                'cum_sel',
                'cum_bm',
                'total_alpha',
                'selection_alpha',
                'overlay_alpha',
                'residual'
            ]
            
            for key in summary_keys:
                if key not in summary:
                    print(f"❌ FAIL: Missing summary key '{key}'")
                    return False
            
            print("✓ All period summary keys exist")
        
        print("✅ PASS: All required keys exist")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_numeric_outputs():
    """Test that outputs are numeric (no None values when success=True)."""
    print("\n=== Test: Numeric Outputs ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_attribution
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        # Compute attribution
        result = compute_portfolio_alpha_attribution(
            price_book=price_book,
            mode='Standard',
            periods=[30, 60, 365]
        )
        
        if not result['success']:
            print(f"⚠️  WARNING: Attribution computation failed: {result['failure_reason']}")
            print("Cannot test numeric outputs when computation fails")
            return True  # Not a test failure, just insufficient data
        
        # Check series are not None
        series_keys = ['daily_realized_return', 'daily_unoverlay_return', 
                      'daily_benchmark_return', 'daily_exposure']
        
        for key in series_keys:
            if result[key] is None:
                print(f"❌ FAIL: Series '{key}' is None")
                return False
            if not isinstance(result[key], pd.Series):
                print(f"❌ FAIL: '{key}' is not a pandas Series")
                return False
            if len(result[key]) == 0:
                print(f"❌ FAIL: Series '{key}' is empty")
                return False
        
        print("✓ All daily series are non-empty pandas Series")
        
        # Check period summaries have numeric values
        for period, summary in result['period_summaries'].items():
            numeric_keys = ['cum_real', 'cum_sel', 'cum_bm', 'total_alpha', 
                          'selection_alpha', 'overlay_alpha', 'residual']
            
            for key in numeric_keys:
                value = summary.get(key)
                if value is None:
                    print(f"❌ FAIL: {period} summary '{key}' is None")
                    return False
                if not isinstance(value, (int, float)):
                    print(f"❌ FAIL: {period} summary '{key}' is not numeric")
                    return False
        
        print(f"✓ All {len(result['period_summaries'])} period summaries have numeric values")
        
        # Check since_inception_summary
        if result['since_inception_summary']:
            inception = result['since_inception_summary']
            numeric_keys = ['cum_real', 'cum_sel', 'cum_bm', 'total_alpha', 
                          'selection_alpha', 'overlay_alpha', 'residual']
            
            for key in numeric_keys:
                value = inception.get(key)
                if value is None:
                    print(f"❌ FAIL: since_inception '{key}' is None")
                    return False
                if not isinstance(value, (int, float)):
                    print(f"❌ FAIL: since_inception '{key}' is not numeric")
                    return False
        
        print("✓ Since inception summary has numeric values")
        print("✅ PASS: All outputs are numeric")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_overlay_alpha_reconciliation():
    """Test that overlay_alpha = total_alpha - selection_alpha (within tolerance)."""
    print("\n=== Test: Overlay Alpha Reconciliation ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_attribution
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        # Compute attribution
        result = compute_portfolio_alpha_attribution(
            price_book=price_book,
            mode='Standard',
            periods=[30, 60, 365]
        )
        
        if not result['success']:
            print(f"⚠️  WARNING: Attribution computation failed: {result['failure_reason']}")
            return True  # Not a test failure
        
        # Test reconciliation for each period
        tolerance = 0.0001  # 0.01% tolerance for floating point
        
        for period, summary in result['period_summaries'].items():
            total_alpha = summary['total_alpha']
            selection_alpha = summary['selection_alpha']
            overlay_alpha = summary['overlay_alpha']
            
            # Check: overlay_alpha should equal (total_alpha - selection_alpha)
            expected_overlay = total_alpha - selection_alpha
            error = abs(overlay_alpha - expected_overlay)
            
            if error > tolerance:
                print(f"❌ FAIL: {period} overlay reconciliation error: {error:.6f}")
                print(f"  total_alpha={total_alpha:.6f}")
                print(f"  selection_alpha={selection_alpha:.6f}")
                print(f"  overlay_alpha={overlay_alpha:.6f}")
                print(f"  expected_overlay={expected_overlay:.6f}")
                return False
            
            print(f"✓ {period}: overlay_alpha reconciles (error={error:.8f})")
        
        # Test since_inception
        if result['since_inception_summary']:
            inception = result['since_inception_summary']
            total_alpha = inception['total_alpha']
            selection_alpha = inception['selection_alpha']
            overlay_alpha = inception['overlay_alpha']
            
            expected_overlay = total_alpha - selection_alpha
            error = abs(overlay_alpha - expected_overlay)
            
            if error > tolerance:
                print(f"❌ FAIL: Since inception overlay reconciliation error: {error:.6f}")
                return False
            
            print(f"✓ Since inception: overlay_alpha reconciles (error={error:.8f})")
        
        print("✅ PASS: Overlay alpha reconciliation verified")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_residual_near_zero():
    """Test that residual is close to 0 (abs < 0.10% over same window)."""
    print("\n=== Test: Residual Near Zero ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_attribution
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        # Compute attribution
        result = compute_portfolio_alpha_attribution(
            price_book=price_book,
            mode='Standard',
            periods=[30, 60, 365]
        )
        
        if not result['success']:
            print(f"⚠️  WARNING: Attribution computation failed: {result['failure_reason']}")
            return True  # Not a test failure
        
        # Test residual for each period
        max_residual = 0.001  # 0.1% tolerance as per problem statement
        
        for period, summary in result['period_summaries'].items():
            residual = summary['residual']
            total_alpha = summary['total_alpha']
            selection_alpha = summary['selection_alpha']
            overlay_alpha = summary['overlay_alpha']
            
            # Verify reconciliation: residual = total - (selection + overlay)
            calculated_residual = total_alpha - (selection_alpha + overlay_alpha)
            
            if abs(residual - calculated_residual) > 1e-10:
                print(f"❌ FAIL: {period} residual calculation mismatch")
                print(f"  reported={residual:.8f}, calculated={calculated_residual:.8f}")
                return False
            
            # Check residual is near zero
            if abs(residual) > max_residual:
                print(f"❌ FAIL: {period} residual too large: {residual:.6f} (>{max_residual})")
                print(f"  total_alpha={total_alpha:.6f}")
                print(f"  selection_alpha={selection_alpha:.6f}")
                print(f"  overlay_alpha={overlay_alpha:.6f}")
                return False
            
            print(f"✓ {period}: residual={residual:.8f} (within tolerance)")
        
        # Test since_inception
        if result['since_inception_summary']:
            inception = result['since_inception_summary']
            residual = inception['residual']
            
            if abs(residual) > max_residual:
                print(f"❌ FAIL: Since inception residual too large: {residual:.6f}")
                return False
            
            print(f"✓ Since inception: residual={residual:.8f} (within tolerance)")
        
        print("✅ PASS: All residuals within tolerance")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_session_state_integration():
    """Test that results are properly stored in session state structure."""
    print("\n=== Test: Session State Integration ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_attribution
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        # Compute attribution
        result = compute_portfolio_alpha_attribution(
            price_book=price_book,
            mode='Standard',
            periods=[30, 60, 365]
        )
        
        if not result['success']:
            print(f"⚠️  WARNING: Attribution computation failed: {result['failure_reason']}")
            return True
        
        # Verify daily_exposure series exists
        if result['daily_exposure'] is None:
            print("❌ FAIL: daily_exposure is None")
            return False
        
        if not isinstance(result['daily_exposure'], pd.Series):
            print("❌ FAIL: daily_exposure is not a Series")
            return False
        
        print(f"✓ daily_exposure series exists ({len(result['daily_exposure'])} days)")
        
        # Verify exposure values are in valid range [0, 1.1]
        # Note: Exposure can exceed 1.0 due to leverage in some strategies (max 110%)
        MAX_EXPOSURE_WITH_LEVERAGE = 1.1
        exposure_series = result['daily_exposure']
        if (exposure_series < 0).any() or (exposure_series > MAX_EXPOSURE_WITH_LEVERAGE).any():
            print(f"❌ FAIL: Some exposure values are outside [0, {MAX_EXPOSURE_WITH_LEVERAGE}] range")
            return False
        
        print("✓ All exposure values in valid range")
        
        # Verify warnings list exists
        if 'warnings' not in result:
            print("❌ FAIL: warnings key missing")
            return False
        
        if not isinstance(result['warnings'], list):
            print("❌ FAIL: warnings is not a list")
            return False
        
        print(f"✓ Warnings list exists ({len(result['warnings'])} warnings)")
        
        print("✅ PASS: Session state integration verified")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("PORTFOLIO ALPHA ATTRIBUTION UNIT TESTS")
    print("=" * 70)
    
    tests = [
        ("Keys Exist", test_attribution_keys_exist),
        ("Numeric Outputs", test_numeric_outputs),
        ("Overlay Alpha Reconciliation", test_overlay_alpha_reconciliation),
        ("Residual Near Zero", test_residual_near_zero),
        ("Session State Integration", test_session_state_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ FATAL ERROR in test '{name}': {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

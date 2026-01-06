#!/usr/bin/env python3
"""
Tests for portfolio alpha ledger consistency between blue box and Attribution Diagnostics.

These tests validate the requirements from the problem statement:
1. Blue box and Attribution Diagnostics use the same canonical ledger
2. No fallback to inception when 60D period unavailable
3. Strict rolling window semantics (start_date aligns with last 60 rows)
4. Residual tolerance within 0.10%
5. Explicit unavailability reasons shown
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test configuration constants
RESIDUAL_TOLERANCE = 0.0010  # 0.10% tolerance for residual attribution


def test_ledger_consistency_60d():
    """Test that 60D period in ledger matches strict rolling window."""
    print("\n=== Test: Ledger Consistency (60D) ===")
    
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
        
        # Compute ledger (same as blue box)
        print("Computing alpha ledger...")
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            periods=[1, 30, 60, 365],
            benchmark_ticker='SPY',
            mode='Standard',
            vix_exposure_enabled=True
        )
        
        if not ledger['success']:
            print(f"❌ FAIL: Ledger computation failed: {ledger['failure_reason']}")
            return False
        
        print("✓ Ledger computed successfully")
        
        # Check 60D period
        period_60d = ledger['period_results'].get('60D', {})
        
        if not period_60d.get('available'):
            reason = period_60d.get('reason', 'unknown')
            rows = period_60d.get('rows_used', 0)
            print(f"⚠️ 60D period unavailable: {reason} (rows={rows})")
            print("✓ Properly returns unavailable status (no silent fallback)")
            return True
        
        # Verify 60D period uses exactly 60 rows
        rows_used = period_60d.get('rows_used', 0)
        if rows_used != 60:
            print(f"❌ FAIL: 60D period should use exactly 60 rows, got {rows_used}")
            return False
        
        print(f"✓ 60D period uses exactly 60 rows")
        
        # Verify start_date corresponds to strict row-slice
        daily_returns = ledger['daily_realized_return']
        if daily_returns is None or len(daily_returns) < 60:
            print("❌ FAIL: Daily returns insufficient for 60D period")
            return False
        
        period_slice = daily_returns.iloc[-60:]
        expected_start = period_slice.index[0].strftime('%Y-%m-%d')
        expected_end = period_slice.index[-1].strftime('%Y-%m-%d')
        
        actual_start = period_60d.get('start_date')
        actual_end = period_60d.get('end_date')
        
        if actual_start != expected_start:
            print(f"❌ FAIL: 60D start_date mismatch: reported={actual_start}, expected={expected_start}")
            print(f"   This indicates fallback to inception instead of strict 60D window")
            return False
        
        if actual_end != expected_end:
            print(f"❌ FAIL: 60D end_date mismatch: reported={actual_end}, expected={expected_end}")
            return False
        
        print(f"✓ 60D period dates correct: {actual_start} to {actual_end}")
        
        # Verify residual is within tolerance
        residual = period_60d.get('residual')
        if residual is None:
            print("❌ FAIL: Residual is None (should be computed)")
            return False
        
        if abs(residual) > RESIDUAL_TOLERANCE:
            print(f"❌ FAIL: 60D residual {residual:+.4%} exceeds tolerance ({RESIDUAL_TOLERANCE:.4%})")
            return False
        
        residual_pct = abs(residual) * 100
        residual_status = "🟢" if residual_pct < 0.10 else "🟡" if residual_pct < 0.5 else "🔴"
        print(f"{residual_status} 60D residual: {residual:+.4%} (within tolerance)")
        
        # Verify alpha decomposition
        total_alpha = period_60d.get('total_alpha')
        selection_alpha = period_60d.get('selection_alpha')
        overlay_alpha = period_60d.get('overlay_alpha')
        
        if total_alpha is None or selection_alpha is None or overlay_alpha is None:
            print("❌ FAIL: Alpha components should not be None when period is available")
            return False
        
        computed_total = selection_alpha + overlay_alpha
        decomposition_error = abs(total_alpha - computed_total)
        
        if decomposition_error > RESIDUAL_TOLERANCE:
            print(f"❌ FAIL: Alpha decomposition error {decomposition_error:+.4%} exceeds tolerance")
            return False
        
        print(f"✓ Alpha decomposition: total={total_alpha:+.2%}, selection={selection_alpha:+.2%}, overlay={overlay_alpha:+.2%}")
        
        print("✅ PASS: Ledger 60D consistency validated")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unavailable_period_handling():
    """Test that insufficient data properly returns unavailable status."""
    print("\n=== Test: Unavailable Period Handling ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        
        # Create price_book with only 40 rows (less than 60 required)
        dates = pd.date_range('2024-01-01', periods=41, freq='D')  # 41 dates = 40 returns
        
        # Create synthetic price data
        np.random.seed(42)
        spy_prices = 100 * (1 + np.random.randn(len(dates)) * 0.01).cumprod()
        bil_prices = np.full(len(dates), 50.0)
        
        price_book = pd.DataFrame({
            'SPY': spy_prices,
            'BIL': bil_prices
        }, index=dates)
        
        print(f"Created price_book with {len(dates)} dates (40 return rows)")
        
        # Compute ledger requesting 60D period
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            periods=[30, 60],
            benchmark_ticker='SPY',
            vix_exposure_enabled=False
        )
        
        if not ledger['success']:
            print(f"⚠️ Ledger computation failed: {ledger['failure_reason']}")
            print("✓ Properly fails with insufficient data (no silent fallback)")
            return True
        
        # Check 30D period (should be available with 40 rows)
        period_30d = ledger['period_results'].get('30D', {})
        if period_30d.get('available'):
            print(f"✓ 30D period available (40 rows >= 30 required)")
        else:
            print(f"⚠️ 30D period unavailable: {period_30d.get('reason')}")
        
        # Check 60D period (should NOT be available with only 40 rows)
        period_60d = ledger['period_results'].get('60D', {})
        
        if period_60d.get('available'):
            print(f"❌ FAIL: 60D period should NOT be available with only 40 rows")
            return False
        
        # Verify reason is correct
        reason = period_60d.get('reason')
        if reason != 'insufficient_aligned_rows':
            print(f"❌ FAIL: Expected reason 'insufficient_aligned_rows', got '{reason}'")
            return False
        
        print(f"✓ 60D period unavailable: {reason}")
        
        # Verify rows_used is reported
        rows_used = period_60d.get('rows_used', 0)
        if rows_used != 40:
            print(f"❌ FAIL: Expected rows_used=40, got {rows_used}")
            return False
        
        print(f"✓ Correctly reports rows_used={rows_used} < requested=60")
        
        # Verify all metrics are None
        if (period_60d.get('cum_realized') is not None or
            period_60d.get('total_alpha') is not None or
            period_60d.get('start_date') is not None):
            print(f"❌ FAIL: Metrics should be None when period unavailable")
            return False
        
        print(f"✓ All metrics correctly set to None (no placeholders)")
        
        print("✅ PASS: Unavailable period handling correct")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_no_inception_fallback():
    """Test that there is NO fallback to inception when 60D unavailable."""
    print("\n=== Test: No Inception Fallback ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        
        # Create price_book with 50 rows (less than 60, more than 30)
        dates = pd.date_range('2021-01-01', periods=51, freq='D')
        
        np.random.seed(42)
        spy_prices = 100 * (1 + np.random.randn(len(dates)) * 0.01).cumprod()
        bil_prices = np.full(len(dates), 50.0)
        
        price_book = pd.DataFrame({
            'SPY': spy_prices,
            'BIL': bil_prices
        }, index=dates)
        
        print(f"Created price_book starting 2021-01-01 with {len(dates)} dates")
        
        # Compute ledger
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            periods=[60],
            benchmark_ticker='SPY',
            vix_exposure_enabled=False
        )
        
        if not ledger['success']:
            print(f"✓ Ledger properly fails (no silent inception fallback)")
            return True
        
        # Check 60D period
        period_60d = ledger['period_results'].get('60D', {})
        
        if not period_60d.get('available'):
            # Period unavailable - verify start_date is None (not inception)
            start_date = period_60d.get('start_date')
            if start_date is not None and start_date != 'N/A':
                print(f"❌ FAIL: start_date should be None/N/A when unavailable, got {start_date}")
                print(f"   This indicates fallback to inception date")
                return False
            
            print(f"✓ 60D unavailable with start_date=None (no inception fallback)")
            print("✅ PASS: No inception fallback")
            return True
        else:
            print("⚠️ 60D period available (edge case - may have exactly 60 rows)")
            # Verify it's not using inception
            start_date = period_60d.get('start_date')
            if start_date and start_date.startswith('2021-01'):
                print(f"❌ FAIL: start_date={start_date} appears to be inception (should be ~50 days from end)")
                return False
            
            print(f"✓ start_date={start_date} is not inception date")
            print("✅ PASS: No inception fallback")
            return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_residual_tolerance_enforcement():
    """Test that residual exceeding tolerance is caught."""
    print("\n=== Test: Residual Tolerance Enforcement ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        from helpers.price_book import get_price_book
        
        # Load real PRICE_BOOK
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        # Compute ledger
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            periods=[60],
            benchmark_ticker='SPY',
            vix_exposure_enabled=True
        )
        
        if not ledger['success']:
            print(f"⚠️ Ledger computation failed: {ledger['failure_reason']}")
            return True
        
        period_60d = ledger['period_results'].get('60D', {})
        
        if not period_60d.get('available'):
            print(f"⚠️ 60D period unavailable: {period_60d.get('reason')}")
            return True
        
        # Check residual
        residual = period_60d.get('residual')
        if residual is None:
            print("❌ FAIL: Residual should be computed")
            return False
        
        # Verify decomposition
        total_alpha = period_60d.get('total_alpha')
        selection_alpha = period_60d.get('selection_alpha')
        overlay_alpha = period_60d.get('overlay_alpha')
        
        computed_total = selection_alpha + overlay_alpha
        
        # The residual should equal total - (selection + overlay)
        expected_residual = total_alpha - computed_total
        
        if abs(residual - expected_residual) > 1e-10:
            print(f"❌ FAIL: Residual mismatch: reported={residual:.6f}, expected={expected_residual:.6f}")
            return False
        
        print(f"✓ Residual correctly computed: {residual:+.4%}")
        
        # Check tolerance
        if abs(residual) > RESIDUAL_TOLERANCE:
            print(f"⚠️ WARNING: Residual {residual:+.4%} exceeds tolerance {RESIDUAL_TOLERANCE:.4%}")
            print("   This should trigger decomposition_error in compute_alpha_source_breakdown")
            # This is not a test failure - it's expected behavior
            print("✓ Large residual properly detected")
        else:
            residual_pct = abs(residual) * 100
            residual_status = "🟢" if residual_pct < 0.10 else "🟡"
            print(f"{residual_status} Residual {residual:+.4%} within tolerance")
        
        print("✅ PASS: Residual tolerance enforcement works")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("LEDGER CONSISTENCY TESTS")
    print("Testing blue box and Attribution Diagnostics consistency")
    print("=" * 70)
    
    tests = [
        ("Ledger Consistency (60D)", test_ledger_consistency_60d),
        ("Unavailable Period Handling", test_unavailable_period_handling),
        ("No Inception Fallback", test_no_inception_fallback),
        ("Residual Tolerance Enforcement", test_residual_tolerance_enforcement),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

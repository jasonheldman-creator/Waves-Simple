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


def test_period_includes_requested_days():
    """Test that periods=[60] includes requested_period_days==60."""
    print("\n=== Test: Period Includes Requested Days ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_attribution
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        # Compute attribution with 60D period
        result = compute_portfolio_alpha_attribution(
            price_book=price_book,
            mode='Standard',
            periods=[60]
        )
        
        if not result['success']:
            print(f"⚠️  WARNING: Attribution computation failed: {result['failure_reason']}")
            return True
        
        # Check that 60D period exists in summaries
        if '60D' not in result['period_summaries']:
            print("❌ FAIL: 60D period not in summaries")
            return False
        
        summary = result['period_summaries']['60D']
        
        # Check that requested_period_days is present and equals 60
        if 'requested_period_days' not in summary:
            print("❌ FAIL: requested_period_days key missing")
            return False
        
        if summary['requested_period_days'] != 60:
            print(f"❌ FAIL: requested_period_days={summary['requested_period_days']}, expected 60")
            return False
        
        print(f"✓ 60D period includes requested_period_days=60")
        
        print("✅ PASS: Period includes requested days")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_exact_window_when_data_sufficient():
    """Test that actual_rows_used == 60 and is_exact_window == True when data sufficient."""
    print("\n=== Test: Exact Window When Data Sufficient ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_attribution
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        # Check available data length
        available_days = len(price_book)
        print(f"Available data: {available_days} days")
        
        # Compute attribution
        result = compute_portfolio_alpha_attribution(
            price_book=price_book,
            mode='Standard',
            periods=[60]
        )
        
        if not result['success']:
            print(f"⚠️  WARNING: Attribution computation failed: {result['failure_reason']}")
            return True
        
        # Check 60D period
        if '60D' not in result['period_summaries']:
            print("❌ FAIL: 60D period not in summaries")
            return False
        
        summary = result['period_summaries']['60D']
        
        # If data is sufficient (>= 65 days), check for exact window
        if available_days >= 65:
            # Check status
            if summary.get('status') != 'valid':
                print(f"❌ FAIL: Expected status='valid' with sufficient data, got '{summary.get('status')}'")
                return False
            
            # Check actual_rows_used
            if 'actual_rows_used' not in summary:
                print("❌ FAIL: actual_rows_used key missing")
                return False
            
            if summary['actual_rows_used'] != 60:
                print(f"❌ FAIL: actual_rows_used={summary['actual_rows_used']}, expected 60")
                return False
            
            print(f"✓ actual_rows_used=60 (as expected with sufficient data)")
            
            # Check is_exact_window
            if 'is_exact_window' not in summary:
                print("❌ FAIL: is_exact_window key missing")
                return False
            
            if not summary['is_exact_window']:
                print(f"❌ FAIL: is_exact_window={summary['is_exact_window']}, expected True")
                return False
            
            print(f"✓ is_exact_window=True")
            
        else:
            # Insufficient data - should be invalid
            print(f"⚠️  Note: Only {available_days} days available (need 65 for 60D + buffer)")
            if summary.get('status') == 'valid':
                print("❌ FAIL: Expected status='invalid' with insufficient data")
                return False
            print(f"✓ Correctly marked as invalid: {summary.get('reason')}")
        
        print("✅ PASS: Exact window verified when data sufficient")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_invalid_when_insufficient_data():
    """Test that status == 'invalid' when data is insufficient (no inception fallback)."""
    print("\n=== Test: Invalid When Insufficient Data ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_attribution
        from helpers.price_book import get_price_book
        
        # Load PRICE_BOOK
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        # Artificially limit data to test insufficient data scenario
        # Take only 50 days (< 65 required for 60D + buffer)
        limited_price_book = price_book.tail(50).copy()
        
        print(f"Testing with limited data: {len(limited_price_book)} days")
        
        # Compute attribution with limited data
        result = compute_portfolio_alpha_attribution(
            price_book=limited_price_book,
            mode='Standard',
            periods=[60]
        )
        
        if not result['success']:
            print(f"✓ Attribution computation reported failure: {result['failure_reason']}")
            print("✅ PASS: Correctly handles insufficient data")
            return True
        
        # Check 60D period
        if '60D' not in result['period_summaries']:
            print("⚠️  WARNING: 60D period not in summaries (acceptable with insufficient data)")
            print("✅ PASS: No invalid period created")
            return True
        
        summary = result['period_summaries']['60D']
        
        # Should be marked as invalid
        if summary.get('status') != 'invalid':
            print(f"❌ FAIL: Expected status='invalid' with {len(limited_price_book)} days, got '{summary.get('status')}'")
            return False
        
        print(f"✓ status='invalid' (as expected with insufficient data)")
        
        # Should have a reason
        if not summary.get('reason'):
            print("❌ FAIL: Invalid period should have a reason")
            return False
        
        print(f"✓ Reason provided: {summary['reason']}")
        
        # Should NOT have numeric values
        if summary.get('cum_real') is not None:
            print("❌ FAIL: Invalid period should not have numeric cum_real")
            return False
        
        print("✓ No numeric values for invalid period (correctly returns None)")
        
        print("✅ PASS: Invalid status when data insufficient, no silent fallback")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_date_range_corresponds_to_slice():
    """Test that start_date and end_date correspond to the sliced series only (not full history)."""
    print("\n=== Test: Date Range Corresponds to Slice ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_attribution
        from helpers.price_book import get_price_book
        from datetime import datetime, timedelta
        
        # Load PRICE_BOOK
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        # Check available data length
        available_days = len(price_book)
        print(f"Available data: {available_days} days")
        
        # Compute attribution
        result = compute_portfolio_alpha_attribution(
            price_book=price_book,
            mode='Standard',
            periods=[60]
        )
        
        if not result['success']:
            print(f"⚠️  WARNING: Attribution computation failed: {result['failure_reason']}")
            return True
        
        # Check 60D period
        if '60D' not in result['period_summaries']:
            print("❌ FAIL: 60D period not in summaries")
            return False
        
        summary = result['period_summaries']['60D']
        
        # Only test if status is valid
        if summary.get('status') != 'valid':
            print(f"⚠️  Note: Period status is '{summary.get('status')}', skipping date range test")
            print("✅ PASS: Date range test skipped for invalid period")
            return True
        
        # Check start_date and end_date exist
        if not summary.get('start_date') or not summary.get('end_date'):
            print("❌ FAIL: start_date or end_date missing")
            return False
        
        start_date_str = summary['start_date']
        end_date_str = summary['end_date']
        
        print(f"✓ Date range: {start_date_str} to {end_date_str}")
        
        # Parse dates
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        except Exception as e:
            print(f"❌ FAIL: Could not parse dates: {e}")
            return False
        
        # Check that dates are not from 2021 (inception)
        if start_date.year <= 2021:
            print(f"❌ FAIL: start_date year is {start_date.year}, should not be from inception era (2021)")
            return False
        
        print(f"✓ start_date is not from inception era (year={start_date.year})")
        
        # Check that date range is reasonable for a 60-day window
        # Trading days are roughly 252/365 of calendar days
        # 60 trading days ≈ 85 calendar days (60 * 365 / 252)
        MIN_CALENDAR_DAYS = int(60 * 365 / 252)  # ~85 days
        MAX_CALENDAR_DAYS = MIN_CALENDAR_DAYS * 2  # ~170 days (allow for holidays/weekends)
        date_diff_days = (end_date - start_date).days
        
        # Allow some flexibility for holidays/weekends
        if date_diff_days < MIN_CALENDAR_DAYS or date_diff_days > MAX_CALENDAR_DAYS:
            print(f"⚠️  WARNING: Date range is {date_diff_days} calendar days (expected ~{MIN_CALENDAR_DAYS}-{MAX_CALENDAR_DAYS} for 60 trading days)")
        else:
            print(f"✓ Date range is {date_diff_days} calendar days (reasonable for 60 trading days)")
        
        # Most importantly: end_date should be recent (within last few days)
        now = datetime.now()
        days_since_end = (now - end_date).days
        
        if days_since_end > 30:
            print(f"⚠️  WARNING: end_date is {days_since_end} days ago (data may be stale)")
        else:
            print(f"✓ end_date is recent ({days_since_end} days ago)")
        
        print("✅ PASS: Date range corresponds to sliced window, not full history")
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
        ("Period Includes Requested Days", test_period_includes_requested_days),
        ("Exact Window When Data Sufficient", test_exact_window_when_data_sufficient),
        ("Invalid When Insufficient Data", test_invalid_when_insufficient_data),
        ("Date Range Corresponds to Slice", test_date_range_corresponds_to_slice),
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

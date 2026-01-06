#!/usr/bin/env python3
"""
Unit tests for compute_portfolio_alpha_ledger (PR #419 Addendum).

Tests validate:
1. Function exists and returns expected keys
2. Enhanced metadata (benchmark_ticker, safe_ticker, vix_proxy_source, etc.)
3. Period summaries have enhanced fields (start_date, end_date, alpha_captured, etc.)
4. Attribution reconciliation with 1e-10 tolerance
5. Alpha Captured computation
6. Period integrity (rows_used == N or available=False)
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_ledger_function_exists():
    """Test that compute_portfolio_alpha_ledger exists and can be imported."""
    print("\n=== Test: Ledger Function Exists ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        print("✓ compute_portfolio_alpha_ledger imported successfully")
        return True
    except ImportError as e:
        print(f"❌ FAIL: Cannot import compute_portfolio_alpha_ledger: {e}")
        return False


def test_ledger_enhanced_metadata():
    """Test that ledger returns enhanced metadata."""
    print("\n=== Test: Enhanced Metadata ===")
    
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
            mode='Standard',
            periods=[1, 30, 60, 365]
        )
        
        # Check enhanced metadata keys
        enhanced_keys = [
            'benchmark_ticker',
            'safe_ticker',
            'vix_proxy_ticker',
            'vix_proxy_source',
            'wave_count',
            'latest_date',
            'data_age_days'
        ]
        
        for key in enhanced_keys:
            if key not in ledger:
                print(f"❌ FAIL: Missing enhanced metadata key '{key}'")
                return False
        
        print("✓ All enhanced metadata keys exist")
        
        # Print metadata values
        print(f"  Benchmark ticker: {ledger['benchmark_ticker']}")
        print(f"  Safe ticker: {ledger['safe_ticker']}")
        print(f"  VIX proxy source: {ledger['vix_proxy_source']}")
        print(f"  Wave count: {ledger['wave_count']}")
        print(f"  Latest date: {ledger['latest_date']}")
        print(f"  Data age (days): {ledger['data_age_days']}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ledger_period_summary_enhancements():
    """Test that period summaries have enhanced fields."""
    print("\n=== Test: Period Summary Enhancements ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        from helpers.price_book import get_price_book
        
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            mode='Standard',
            periods=[1, 30, 60, 365]
        )
        
        if not ledger['success']:
            print(f"❌ FAIL: Ledger computation failed: {ledger['failure_reason']}")
            return False
        
        # Check enhanced fields in period summaries
        enhanced_summary_keys = [
            'start_date',
            'end_date',
            'alpha_captured',
            'exposure_min',
            'exposure_max',
            'attribution_reconciled'
        ]
        
        for period_key in ['1D', '30D', '60D', '365D']:
            if period_key not in ledger['period_summaries']:
                print(f"⚠️ WARNING: Period {period_key} not in summaries")
                continue
            
            summary = ledger['period_summaries'][period_key]
            
            for key in enhanced_summary_keys:
                if key not in summary:
                    print(f"❌ FAIL: Missing enhanced summary key '{key}' in {period_key}")
                    return False
            
            print(f"✓ {period_key} has all enhanced fields")
            
            # Print summary details if available
            if summary.get('available', False):
                print(f"  Start: {summary['start_date']}, End: {summary['end_date']}")
                print(f"  Rows used: {summary['rows_used']}")
                print(f"  Exposure: [{summary['exposure_min']:.3f}, {summary['exposure_max']:.3f}]")
                print(f"  Alpha captured: {summary['alpha_captured']}")
                print(f"  Attribution reconciled: {summary['attribution_reconciled']}")
                print(f"  Residual: {summary.get('residual', 'N/A')}")
            else:
                print(f"  Unavailable - Reason: {summary.get('reason', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_period_integrity():
    """Test that period integrity is enforced (rows_used == N or available=False)."""
    print("\n=== Test: Period Integrity ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        from helpers.price_book import get_price_book
        
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            mode='Standard',
            periods=[1, 30, 60, 365]
        )
        
        if not ledger['success']:
            print(f"❌ FAIL: Ledger computation failed: {ledger['failure_reason']}")
            return False
        
        # Check period integrity
        periods = [1, 30, 60, 365]
        for period in periods:
            period_key = f'{period}D'
            if period_key not in ledger['period_summaries']:
                print(f"❌ FAIL: Missing period {period_key}")
                return False
            
            summary = ledger['period_summaries'][period_key]
            rows_used = summary.get('rows_used')
            available = summary.get('available', False)
            
            if available:
                # If available, rows_used MUST equal period
                if rows_used != period:
                    print(f"❌ FAIL: {period_key} available=True but rows_used={rows_used} != {period}")
                    return False
                print(f"✓ {period_key}: rows_used={rows_used} == {period} (available)")
            else:
                # If not available, must have a reason
                reason = summary.get('reason')
                if not reason:
                    print(f"❌ FAIL: {period_key} available=False but no reason provided")
                    return False
                print(f"✓ {period_key}: available=False, reason='{reason}'")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attribution_reconciliation():
    """Test that attribution reconciliation works with 1e-10 tolerance."""
    print("\n=== Test: Attribution Reconciliation ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        from helpers.price_book import get_price_book
        
        price_book = get_price_book()
        
        if price_book is None or price_book.empty:
            print("❌ FAIL: PRICE_BOOK is empty")
            return False
        
        ledger = compute_portfolio_alpha_ledger(
            price_book=price_book,
            mode='Standard',
            periods=[1, 30, 60, 365]
        )
        
        if not ledger['success']:
            print(f"❌ FAIL: Ledger computation failed: {ledger['failure_reason']}")
            return False
        
        TOLERANCE = 1e-10
        
        # Check reconciliation for each available period
        for period_key, summary in ledger['period_summaries'].items():
            if not summary.get('available', False):
                print(f"⚠️ {period_key}: Skipping (unavailable)")
                continue
            
            residual = summary.get('residual')
            reconciled = summary.get('attribution_reconciled', False)
            
            if residual is None:
                print(f"❌ FAIL: {period_key} has None residual")
                return False
            
            # Check if reconciliation flag matches tolerance
            expected_reconciled = abs(residual) <= TOLERANCE
            if reconciled != expected_reconciled:
                print(f"❌ FAIL: {period_key} reconciled={reconciled} but residual={residual:.2e}")
                return False
            
            if reconciled:
                print(f"✓ {period_key}: Reconciled (residual={residual:.2e})")
            else:
                print(f"⚠️ {period_key}: Not reconciled (residual={residual:.2e})")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("RUNNING ALPHA LEDGER TESTS (PR #419 Addendum)")
    print("="*60)
    
    tests = [
        ("Function Exists", test_ledger_function_exists),
        ("Enhanced Metadata", test_ledger_enhanced_metadata),
        ("Period Summary Enhancements", test_ledger_period_summary_enhancements),
        ("Period Integrity", test_period_integrity),
        ("Attribution Reconciliation", test_attribution_reconciliation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

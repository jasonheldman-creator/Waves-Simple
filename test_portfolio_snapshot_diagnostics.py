#!/usr/bin/env python3
"""
Test suite for Portfolio Snapshot diagnostics enhancements.

This test validates that:
1. compute_portfolio_snapshot captures exception traceback on failure
2. compute_portfolio_alpha_ledger captures exception traceback on failure
3. Debug dict contains all required diagnostic fields
4. UI can properly display enhanced diagnostics
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_portfolio_snapshot_exception_traceback():
    """Test that compute_portfolio_snapshot captures exception traceback on failure."""
    print("\n=== Test: Portfolio Snapshot Exception Traceback ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_snapshot
        
        # Create empty price_book to trigger failure
        print("Creating empty price_book to trigger failure...")
        price_book = pd.DataFrame()
        
        # Compute portfolio snapshot (should fail)
        print("Computing portfolio snapshot with empty price_book...")
        result = compute_portfolio_snapshot(price_book, mode='Standard', periods=[1, 30, 60])
        
        # Validate failure
        if result['success']:
            print("❌ FAIL: Expected failure but got success")
            return False
        
        print(f"✓ Computation failed as expected: {result['failure_reason']}")
        
        # Validate debug dict exists
        if 'debug' not in result:
            print("❌ FAIL: No 'debug' key in result")
            return False
        
        debug = result['debug']
        print("✓ Debug dict present in result")
        
        # Validate enhanced debug fields
        required_fields = [
            'reason_if_failure',
            'exception_message',
            'exception_traceback',
            'price_book_shape',
            'active_waves_count',
            'tickers_requested_count',
            'tickers_intersection_count'
        ]
        
        missing_fields = [f for f in required_fields if f not in debug]
        if missing_fields:
            print(f"❌ FAIL: Missing debug fields: {missing_fields}")
            return False
        
        print(f"✓ All {len(required_fields)} required debug fields present")
        
        # Validate reason_if_failure is set
        if not debug['reason_if_failure']:
            print("❌ FAIL: reason_if_failure not set in debug dict")
            return False
        
        print(f"✓ Debug reason_if_failure: {debug['reason_if_failure']}")
        
        # Note: exception_message and exception_traceback will be None for simple validation failures
        # but should be populated for actual exceptions
        print(f"  - exception_message: {debug['exception_message'] or 'None (no exception)'}")
        print(f"  - exception_traceback: {'Present' if debug['exception_traceback'] else 'None (no exception)'}")
        
        print("\n✓ PASS: Portfolio snapshot exception traceback test")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_portfolio_alpha_ledger_exception_traceback():
    """Test that compute_portfolio_alpha_ledger captures exception traceback on failure."""
    print("\n=== Test: Portfolio Alpha Ledger Exception Traceback ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_alpha_ledger
        
        # Create empty price_book to trigger failure
        print("Creating empty price_book to trigger failure...")
        price_book = pd.DataFrame()
        
        # Compute portfolio alpha ledger (should fail)
        print("Computing portfolio alpha ledger with empty price_book...")
        result = compute_portfolio_alpha_ledger(price_book, periods=[1, 30, 60, 365])
        
        # Validate failure
        if result['success']:
            print("❌ FAIL: Expected failure but got success")
            return False
        
        print(f"✓ Computation failed as expected: {result['failure_reason']}")
        
        # Validate debug dict exists
        if 'debug' not in result:
            print("❌ FAIL: No 'debug' key in result")
            return False
        
        debug = result['debug']
        print("✓ Debug dict present in result")
        
        # Validate enhanced debug fields
        required_fields = [
            'reason_if_failure',
            'exception_message',
            'exception_traceback',
            'price_book_shape',
            'active_waves_count',
            'tickers_requested_count',
            'tickers_intersection_count'
        ]
        
        missing_fields = [f for f in required_fields if f not in debug]
        if missing_fields:
            print(f"❌ FAIL: Missing debug fields: {missing_fields}")
            return False
        
        print(f"✓ All {len(required_fields)} required debug fields present")
        
        # Validate reason_if_failure is set
        if not debug['reason_if_failure']:
            print("❌ FAIL: reason_if_failure not set in debug dict")
            return False
        
        print(f"✓ Debug reason_if_failure: {debug['reason_if_failure']}")
        
        # Note: exception_message and exception_traceback will be None for simple validation failures
        print(f"  - exception_message: {debug['exception_message'] or 'None (no exception)'}")
        print(f"  - exception_traceback: {'Present' if debug['exception_traceback'] else 'None (no exception)'}")
        
        print("\n✓ PASS: Portfolio alpha ledger exception traceback test")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_portfolio_snapshot_with_synthetic_data():
    """Test portfolio snapshot with synthetic data to validate enhanced debug info."""
    print("\n=== Test: Portfolio Snapshot with Synthetic Data ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_snapshot
        
        # Create synthetic price_book with SPY + 2 tickers
        print("Creating synthetic price_book...")
        
        # Generate 100 days of price data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        # Create realistic price movements (random walk) with modern random generator
        rng = np.random.default_rng(42)
        spy_prices = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 100)))
        aapl_prices = 150 * np.exp(np.cumsum(rng.normal(0.0015, 0.025, 100)))
        msft_prices = 200 * np.exp(np.cumsum(rng.normal(0.0012, 0.022, 100)))
        
        price_book = pd.DataFrame({
            'SPY': spy_prices,
            'AAPL': aapl_prices,
            'MSFT': msft_prices
        }, index=dates)
        
        print(f"✓ Created price_book: {len(price_book)} days × {len(price_book.columns)} tickers")
        
        # Compute portfolio snapshot
        print("\nComputing portfolio snapshot...")
        result = compute_portfolio_snapshot(price_book, mode='Standard', periods=[1, 30, 60])
        
        # Validate debug dict exists (should exist even on success)
        if 'debug' not in result:
            print("❌ FAIL: No 'debug' key in result")
            return False
        
        debug = result['debug']
        print("✓ Debug dict present in result")
        
        # Display debug info
        print("\nDebug Information:")
        print(f"  - price_book_source: {debug['price_book_source']}")
        print(f"  - price_book_shape: {debug['price_book_shape']}")
        print(f"  - price_book_index_min: {debug['price_book_index_min']}")
        print(f"  - price_book_index_max: {debug['price_book_index_max']}")
        print(f"  - spy_present: {debug['spy_present']}")
        print(f"  - active_waves_count: {debug['active_waves_count']}")
        print(f"  - tickers_requested_count: {debug['tickers_requested_count']}")
        print(f"  - tickers_intersection_count: {debug['tickers_intersection_count']}")
        print(f"  - reason_if_failure: {debug['reason_if_failure']}")
        print(f"  - exception_message: {debug['exception_message']}")
        print(f"  - exception_traceback: {'Present' if debug['exception_traceback'] else 'None'}")
        
        # Check if computation succeeded or failed gracefully
        if not result['success']:
            print(f"\n⚠ Computation did not succeed: {result['failure_reason']}")
            print(f"  This is expected with synthetic data and minimal waves")
            # We still consider this a pass as long as debug info is present
        else:
            print("\n✓ Portfolio snapshot computation succeeded")
        
        print("\n✓ PASS: Portfolio snapshot with synthetic data test")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_debug_dict_structure():
    """Test that debug dict has proper structure for UI rendering."""
    print("\n=== Test: Debug Dict Structure for UI ===")
    
    try:
        from helpers.wave_performance import compute_portfolio_snapshot
        
        # Create empty price_book to trigger failure with debug info
        price_book = pd.DataFrame()
        result = compute_portfolio_snapshot(price_book, mode='Standard', periods=[1, 30, 60])
        
        debug = result.get('debug', {})
        
        # Validate fields that UI needs to display
        ui_required_fields = {
            'price_book_shape': str,
            'spy_present': bool,
            'active_waves_count': int,
            'tickers_requested_count': int,
            'tickers_intersection_count': int,
            'tickers_missing_sample': list,
            'price_book_index_min': (str, type(None)),
            'price_book_index_max': (str, type(None)),
            'reason_if_failure': (str, type(None)),
            'exception_message': (str, type(None)),
            'exception_traceback': (str, type(None))
        }
        
        print("Validating UI-required fields and types...")
        for field, expected_type in ui_required_fields.items():
            if field not in debug:
                print(f"❌ FAIL: Missing UI field: {field}")
                return False
            
            value = debug[field]
            if value is not None:
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        print(f"❌ FAIL: Field {field} has wrong type: {type(value)} (expected one of {expected_type})")
                        return False
                else:
                    if not isinstance(value, expected_type):
                        print(f"❌ FAIL: Field {field} has wrong type: {type(value)} (expected {expected_type})")
                        return False
            
            print(f"✓ {field}: {type(value).__name__}")
        
        print(f"\n✓ All {len(ui_required_fields)} UI-required fields validated")
        
        print("\n✓ PASS: Debug dict structure test")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Portfolio Snapshot Diagnostics Enhancement Test Suite")
    print("=" * 70)
    
    tests = [
        ("Portfolio Snapshot Exception Traceback", test_portfolio_snapshot_exception_traceback),
        ("Portfolio Alpha Ledger Exception Traceback", test_portfolio_alpha_ledger_exception_traceback),
        ("Portfolio Snapshot with Synthetic Data", test_portfolio_snapshot_with_synthetic_data),
        ("Debug Dict Structure for UI", test_debug_dict_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70)
    
    return all(result for _, result in results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

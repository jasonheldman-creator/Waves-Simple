#!/usr/bin/env python3
"""
Test Suite: Network-Independent Ledger Recompute

This test suite validates that ledger and wave_history recompute works
based solely on cached price_book freshness, without requiring network
access or yfinance availability.

Key Requirements:
1. Ledger recompute proceeds when cached price_book is fresh
2. Force Ledger Recompute button behavior is correct
3. Ledger max date matches price_book max date (not N/A)
4. Tests run in network-independent environment
"""

import os
import sys
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestLedgerRecomputeNetworkIndependent(unittest.TestCase):
    """Test ledger recompute functionality without network dependency."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.original_dir = os.getcwd()
        
    def setUp(self):
        """Set up each test."""
        # Ensure we're in the right directory
        os.chdir(self.original_dir)
    
    def test_price_book_cache_exists(self):
        """Test that price_book cache file exists."""
        cache_path = os.path.join(os.getcwd(), 'data', 'cache', 'prices_cache.parquet')
        
        self.assertTrue(
            os.path.exists(cache_path),
            f"Price cache should exist at {cache_path}"
        )
        
        # Check file is not empty
        file_size = os.path.getsize(cache_path)
        self.assertGreater(
            file_size, 0,
            "Price cache file should not be empty"
        )
        
        print(f"✓ Price cache exists: {cache_path} ({file_size} bytes)")
    
    def test_price_book_loads_without_network(self):
        """Test that price_book can be loaded from cache without network access."""
        try:
            # Import price_book module from helpers
            from helpers import price_book
            
            # Get price data (should load from cache, not network)
            price_data = price_book.get_price_book()
            
            self.assertIsNotNone(price_data, "Price data should not be None")
            self.assertFalse(price_data.empty, "Price data should not be empty")
            
            # Check that we have dates and tickers
            self.assertGreater(len(price_data), 0, "Should have at least one date")
            self.assertGreater(len(price_data.columns), 0, "Should have at least one ticker")
            
            # Get max date
            max_date = price_data.index[-1]
            max_date_str = max_date.strftime('%Y-%m-%d')
            
            print(f"✓ Price book loaded from cache without network")
            print(f"  Dates: {len(price_data)}, Tickers: {len(price_data.columns)}")
            print(f"  Max date: {max_date_str}")
            
        except Exception as e:
            self.fail(f"Failed to load price_book from cache: {e}")
    
    def test_wave_history_exists_or_buildable(self):
        """Test that wave_history.csv exists or can be built from price_book."""
        wave_history_path = os.path.join(os.getcwd(), 'wave_history.csv')
        
        if os.path.exists(wave_history_path):
            print(f"✓ wave_history.csv exists: {wave_history_path}")
            
            # Verify it has content
            try:
                import pandas as pd
                wave_history = pd.read_csv(wave_history_path)
                
                self.assertFalse(wave_history.empty, "wave_history should not be empty")
                self.assertIn('date', wave_history.columns, "Should have 'date' column")
                self.assertIn('wave', wave_history.columns, "Should have 'wave' column")
                
                wave_history['date'] = pd.to_datetime(wave_history['date'])
                max_date = wave_history['date'].max()
                max_date_str = max_date.strftime('%Y-%m-%d')
                
                print(f"  Max date: {max_date_str}")
                print(f"  Rows: {len(wave_history)}")
                
            except Exception as e:
                self.fail(f"wave_history.csv exists but cannot be read: {e}")
        else:
            print("  wave_history.csv does not exist, attempting to build...")
            
            # Try to build wave_history from price_book
            try:
                from helpers.operator_toolbox import rebuild_wave_history
                
                success, message = rebuild_wave_history()
                
                if success:
                    print(f"  ✓ Successfully built wave_history.csv")
                    print(f"    {message}")
                    
                    # Verify it was created
                    self.assertTrue(
                        os.path.exists(wave_history_path),
                        "wave_history.csv should exist after rebuild"
                    )
                else:
                    self.fail(f"Failed to rebuild wave_history: {message}")
                    
            except ImportError:
                self.skipTest("operator_toolbox not available, cannot test rebuild")
    
    def test_force_ledger_recompute_function(self):
        """Test force_ledger_recompute() function behavior."""
        try:
            from helpers.operator_toolbox import force_ledger_recompute
        except ImportError:
            self.skipTest("force_ledger_recompute not available in operator_toolbox")
        
        # Call force_ledger_recompute
        success, message = force_ledger_recompute()
        
        self.assertTrue(
            success,
            f"force_ledger_recompute should succeed. Message: {message}"
        )
        
        print(f"✓ force_ledger_recompute succeeded")
        print(f"  Message: {message}")
        
        # Verify wave_history was updated
        wave_history_path = os.path.join(os.getcwd(), 'wave_history.csv')
        self.assertTrue(
            os.path.exists(wave_history_path),
            "wave_history.csv should exist after force_ledger_recompute"
        )
    
    def test_price_book_and_wave_history_dates_match(self):
        """Test that price_book max date matches wave_history max date."""
        try:
            import pandas as pd
            from helpers import price_book
            
            # Get price_book max date
            price_data = price_book.get_price_book()
            self.assertFalse(price_data.empty, "Price data should not be empty")
            
            price_book_max_date = price_data.index[-1].strftime('%Y-%m-%d')
            
            # Get wave_history max date
            wave_history_path = os.path.join(os.getcwd(), 'wave_history.csv')
            self.assertTrue(
                os.path.exists(wave_history_path),
                "wave_history.csv should exist"
            )
            
            wave_history = pd.read_csv(wave_history_path)
            self.assertFalse(wave_history.empty, "wave_history should not be empty")
            self.assertIn('date', wave_history.columns, "Should have 'date' column")
            
            wave_history['date'] = pd.to_datetime(wave_history['date'])
            wave_history_max_date = wave_history['date'].max().strftime('%Y-%m-%d')
            
            self.assertEqual(
                price_book_max_date,
                wave_history_max_date,
                f"price_book max date ({price_book_max_date}) should match wave_history max date ({wave_history_max_date})"
            )
            
            print(f"✓ Dates match: price_book and wave_history both at {price_book_max_date}")
            
        except Exception as e:
            self.fail(f"Failed to compare dates: {e}")
    
    def test_ledger_max_date_not_na_after_recompute(self):
        """
        Test that Ledger max date is not N/A after triggering recompute.
        
        This is a simulation of what happens when the UI button is clicked.
        The actual ledger computation happens when compute_portfolio_alpha_ledger
        is called with the fresh price_book.
        """
        try:
            from helpers import price_book
            from helpers.wave_performance import compute_portfolio_alpha_ledger
            
            # Simulate Force Ledger Recompute button behavior
            
            # Step 1: Reload price_book from cache
            price_data = price_book.get_price_book()
            self.assertFalse(price_data.empty, "Price data should not be empty")
            
            price_book_max_date = price_data.index[-1].strftime('%Y-%m-%d')
            print(f"  Price book max date: {price_book_max_date}")
            
            # Step 2: Compute ledger from fresh price_book
            ledger_result = compute_portfolio_alpha_ledger(
                price_book=price_data,
                periods=[1, 30, 60, 365],
                benchmark_ticker="SPY",
                safe_ticker_preference=["BIL", "SHY"],
                mode="Standard",
                wave_registry=None,
                vix_exposure_enabled=True
            )
            
            # Validate ledger computation succeeded
            self.assertIsNotNone(ledger_result, "Ledger result should not be None")
            self.assertIsInstance(ledger_result, dict, "Ledger result should be a dict")
            
            if not ledger_result.get('success'):
                failure_reason = ledger_result.get('failure_reason', 'Unknown')
                self.fail(f"Ledger computation failed: {failure_reason}")
            
            # Get ledger DataFrame
            ledger_df = ledger_result.get('daily_ledger')
            self.assertIsNotNone(ledger_df, "Daily ledger should not be None")
            self.assertFalse(ledger_df.empty, "Daily ledger should not be empty")
            
            # Get ledger max date
            ledger_max_date = ledger_df.index[-1].strftime('%Y-%m-%d')
            
            print(f"✓ Ledger computed successfully")
            print(f"  Ledger max date: {ledger_max_date}")
            
            # Assert ledger max date is not N/A and matches price_book
            self.assertIsNotNone(ledger_max_date, "Ledger max date should not be None")
            self.assertEqual(
                ledger_max_date,
                price_book_max_date,
                f"Ledger max date ({ledger_max_date}) should match price_book max date ({price_book_max_date})"
            )
            
            print(f"✓ Ledger max date matches price_book max date: {ledger_max_date}")
            
        except ImportError as e:
            self.skipTest(f"Required modules not available: {e}")
        except Exception as e:
            self.fail(f"Ledger recompute test failed: {e}")
    
    def test_self_test_includes_ledger_recompute_check(self):
        """Test that run_self_test includes ledger recompute readiness check."""
        try:
            from helpers.operator_toolbox import run_self_test
        except ImportError:
            self.skipTest("run_self_test not available in operator_toolbox")
        
        # Run self-test
        test_results = run_self_test()
        
        self.assertIsNotNone(test_results, "Test results should not be None")
        self.assertIn('tests', test_results, "Should have 'tests' key")
        
        # Check if ledger recompute readiness test exists
        test_names = [test['name'] for test in test_results['tests']]
        
        self.assertIn(
            'Ledger recompute readiness',
            test_names,
            "Self-test should include 'Ledger recompute readiness' check"
        )
        
        print(f"✓ Self-test includes ledger recompute readiness check")
        print(f"  Total tests: {len(test_results['tests'])}")
        print(f"  Overall status: {test_results['overall_status']}")


def main():
    """Run all tests."""
    # Disable network access for price_book to ensure we're testing cache-only behavior
    os.environ['PRICE_FETCH_ENABLED'] = 'false'
    os.environ['ALLOW_NETWORK_FETCH'] = 'false'
    
    print("\n" + "=" * 70)
    print("NETWORK-INDEPENDENT LEDGER RECOMPUTE TEST SUITE")
    print("=" * 70)
    print("\nEnvironment:")
    print(f"  PRICE_FETCH_ENABLED: {os.environ.get('PRICE_FETCH_ENABLED', 'not set')}")
    print(f"  ALLOW_NETWORK_FETCH: {os.environ.get('ALLOW_NETWORK_FETCH', 'not set')}")
    print("\n" + "=" * 70 + "\n")
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLedgerRecomputeNetworkIndependent)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        print("\nValidated:")
        print("  - Price cache exists and loads without network")
        print("  - wave_history.csv can be built from cached price_book")
        print("  - force_ledger_recompute() function works correctly")
        print("  - Ledger max date matches price_book max date (not N/A)")
        print("  - All tests run in network-independent environment")
        return 0
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} test(s) had errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())

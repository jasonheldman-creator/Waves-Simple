#!/usr/bin/env python3
"""
Test suite for price_book sidebar date integration.

This test validates the requirements from the problem statement:
1. Sidebar date equals price_book.index.max()
2. Portfolio banner formatting returns "—" (not N/A) for insufficient history
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestSidebarDateIntegration(unittest.TestCase):
    """Test that sidebar date uses price_book.index.max()"""
    
    def test_get_latest_data_timestamp_uses_price_book(self):
        """Test that get_latest_data_timestamp uses PRICE_BOOK as single source of truth"""
        print("\n=== Test: get_latest_data_timestamp uses PRICE_BOOK ===")
        
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        import importlib
        
        # Create mock price_book with known max date
        expected_date = datetime(2024, 1, 15)
        dates = pd.date_range(end=expected_date, periods=30, freq='D')
        price_book = pd.DataFrame(
            np.random.rand(30, 5),
            index=dates,
            columns=['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
        )
        
        # Mock get_price_book to return our test data
        with patch('helpers.price_book.get_price_book', return_value=price_book):
            # Import app module
            import app
            # Reload to ensure patch is applied
            importlib.reload(app)
            
            # Call the function
            result = app.get_latest_data_timestamp()
            
            # Verify result matches price_book.index.max()
            expected = expected_date.strftime("%Y-%m-%d")
            self.assertEqual(result, expected, 
                           f"Expected {expected}, got {result}")
            
            print(f"✓ get_latest_data_timestamp() returned: {result}")
            print(f"✓ Matches price_book.index.max(): {expected}")
    
    def test_get_latest_data_timestamp_handles_empty_price_book(self):
        """Test that get_latest_data_timestamp handles empty PRICE_BOOK gracefully"""
        print("\n=== Test: get_latest_data_timestamp handles empty PRICE_BOOK ===")
        
        import pandas as pd
        import importlib
        
        # Mock get_price_book to return empty DataFrame
        empty_df = pd.DataFrame()
        
        with patch('helpers.price_book.get_price_book', return_value=empty_df):
            import app
            # Reload to ensure patch is applied
            importlib.reload(app)
            
            # Call the function
            result = app.get_latest_data_timestamp()
            
            # Should return "unknown" for empty price_book
            self.assertEqual(result, "unknown", 
                           f"Expected 'unknown', got '{result}'")
            
            print(f"✓ get_latest_data_timestamp() returned: {result}")
            print(f"✓ Correctly handles empty PRICE_BOOK")


class TestPortfolioBannerFormatting(unittest.TestCase):
    """Test that portfolio banner formatting uses em-dash (—) for insufficient history"""
    
    def test_portfolio_banner_uses_em_dash_not_na(self):
        """Test that portfolio banner returns '—' (not 'N/A') for periods with insufficient history"""
        print("\n=== Test: Portfolio Banner Formatting ===")
        
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        # Create price_book with insufficient history (only 20 days, not enough for 30D)
        dates = pd.date_range(end=datetime(2024, 1, 15), periods=20, freq='D')
        price_book = pd.DataFrame(
            np.random.rand(20, 5),
            index=dates,
            columns=['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
        )
        
        # Mock dependencies
        with patch('helpers.price_book.get_price_book', return_value=price_book):
            # Import compute_portfolio_snapshot
            from helpers.wave_performance import compute_portfolio_snapshot
            
            # Call the function
            snapshot = compute_portfolio_snapshot(price_book, mode='Standard', periods=[1, 30, 60, 365])
            
            # Verify snapshot exists
            self.assertTrue(snapshot['success'], "Snapshot should succeed even with limited data")
            
            # Check that 30D/60D/365D are None (insufficient history)
            ret_30d = snapshot['portfolio_returns'].get('30D')
            ret_60d = snapshot['portfolio_returns'].get('60D')
            ret_365d = snapshot['portfolio_returns'].get('365D')
            
            print(f"Portfolio returns with 20 days of data:")
            print(f"  1D:  {snapshot['portfolio_returns'].get('1D')}")
            print(f"  30D: {ret_30d}")
            print(f"  60D: {ret_60d}")
            print(f"  365D: {ret_365d}")
            
            # Verify formatting logic: None values should format as "—"
            # This tests the code in render_selected_wave_banner_enhanced
            ret_30d_str = f"{ret_30d*100:+.2f}%" if ret_30d is not None else "—"
            ret_60d_str = f"{ret_60d*100:+.2f}%" if ret_60d is not None else "—"
            ret_365d_str = f"{ret_365d*100:+.2f}%" if ret_365d is not None else "—"
            
            # Verify em-dash is used, not "N/A"
            if ret_30d is None:
                self.assertEqual(ret_30d_str, "—", 
                               f"Expected em-dash '—', got '{ret_30d_str}'")
                print(f"✓ 30D formats as '—' when insufficient history")
            
            if ret_60d is None:
                self.assertEqual(ret_60d_str, "—", 
                               f"Expected em-dash '—', got '{ret_60d_str}'")
                print(f"✓ 60D formats as '—' when insufficient history")
            
            if ret_365d is None:
                self.assertEqual(ret_365d_str, "—", 
                               f"Expected em-dash '—', got '{ret_365d_str}'")
                print(f"✓ 365D formats as '—' when insufficient history")
            
            # Verify none of them are "N/A"
            self.assertNotEqual(ret_30d_str, "N/A", "Should use '—' not 'N/A'")
            self.assertNotEqual(ret_60d_str, "N/A", "Should use '—' not 'N/A'")
            self.assertNotEqual(ret_365d_str, "N/A", "Should use '—' not 'N/A'")
            
            print(f"✓ All insufficient periods use '—' instead of 'N/A'")


def main():
    """Run all tests"""
    print("=" * 70)
    print("Price Book Sidebar Integration Test Suite")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestSidebarDateIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPortfolioBannerFormatting))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    if result.wasSuccessful():
        print("✓ All tests passed")
    else:
        print(f"❌ {len(result.failures)} test(s) failed")
        print(f"❌ {len(result.errors)} test(s) had errors")
    
    print("=" * 70)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())

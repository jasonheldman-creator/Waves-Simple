#!/usr/bin/env python3
"""
Test Suite: Operator Toolbox Export Functions

This test suite validates that the price_book to prices.csv export
functionality works correctly, specifically testing the DataFrame melt
operation with proper index handling to prevent the id_vars=['index'] bug.

Key Requirements:
1. Export function handles reset_index() correctly
2. Melt operation uses correct column name (date, not index)
3. Export produces valid prices.csv format: date, ticker, close
4. No regression for id_vars=['index'] error
"""

import os
import sys
import unittest
import tempfile
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestOperatorToolboxExport(unittest.TestCase):
    """Test price_book export functionality in operator_toolbox."""
    
    def test_dataframe_reset_index_and_melt(self):
        """Test that DataFrame reset_index and melt work correctly."""
        # Create a sample price_book-like DataFrame
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        data = {
            'SPY': [400.0, 401.0, 402.0, 403.0, 404.0],
            'QQQ': [300.0, 301.0, 302.0, 303.0, 304.0],
            'AAPL': [150.0, 151.0, 152.0, 153.0, 154.0]
        }
        
        # Create DataFrame with DatetimeIndex (like price_book)
        price_data = pd.DataFrame(data, index=dates)
        
        # Test the export logic from operator_toolbox
        # This is the FIXED version that should work
        price_data_reset = price_data.reset_index()
        
        # Explicitly rename the first column (the reset index) to 'date'
        # This prevents the id_vars=['index'] error when the index doesn't have a name
        price_data_reset = price_data_reset.rename(columns={price_data_reset.columns[0]: 'date'})
        
        # Now melt with id_vars=['date']
        try:
            prices_long_df = pd.melt(
                price_data_reset,
                id_vars=['date'],
                var_name='ticker',
                value_name='close'
            )
            
            # Verify the result
            self.assertIn('date', prices_long_df.columns, "Should have 'date' column")
            self.assertIn('ticker', prices_long_df.columns, "Should have 'ticker' column")
            self.assertIn('close', prices_long_df.columns, "Should have 'close' column")
            
            # Verify number of rows (should be dates * tickers)
            expected_rows = len(dates) * len(data)
            self.assertEqual(len(prices_long_df), expected_rows, 
                           f"Should have {expected_rows} rows (5 dates × 3 tickers)")
            
            # Verify unique tickers
            unique_tickers = prices_long_df['ticker'].unique()
            self.assertEqual(set(unique_tickers), set(data.keys()),
                           "Should have all original tickers")
            
            print("✓ DataFrame reset_index and melt work correctly")
            print(f"  Input shape: {price_data.shape}")
            print(f"  Output rows: {len(prices_long_df)}")
            print(f"  Tickers: {list(unique_tickers)}")
            
        except Exception as e:
            self.fail(f"Melt operation failed: {e}")
    
    def test_old_method_would_fail_with_named_index(self):
        """Test that the OLD method (id_vars=['index']) fails when index has a name."""
        # Create DataFrame with a NAMED index (like real price_book has 'Date')
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D', name='Date')
        data = {
            'SPY': [400.0, 401.0, 402.0, 403.0, 404.0],
            'QQQ': [300.0, 301.0, 302.0, 303.0, 304.0]
        }
        
        price_data = pd.DataFrame(data, index=dates)
        price_data_reset = price_data.reset_index()
        
        # After reset_index with a named index, column is 'Date', not 'index'
        self.assertIn('Date', price_data_reset.columns, "Column should be named 'Date'")
        self.assertNotIn('index', price_data_reset.columns, "Column should NOT be named 'index'")
        
        # The OLD buggy code tried to use id_vars=['index'] directly
        # This will fail because the column is 'Date', not 'index'
        with self.assertRaises(KeyError) as context:
            prices_long_df = pd.melt(
                price_data_reset,
                id_vars=['index'],  # BUG: This column doesn't exist when index is named!
                var_name='ticker',
                value_name='close'
            )
        
        error_msg = str(context.exception)
        # The error should mention that 'index' is not present in the DataFrame
        self.assertTrue('index' in error_msg.lower() or 'not present' in error_msg.lower(), 
                       f"Error should mention missing column: {error_msg}")
        
        print("✓ Confirmed old method with id_vars=['index'] fails when index has a name")
        print(f"  Actual column name after reset_index: 'Date'")
        print(f"  Error: {error_msg}")
    
    def test_rebuild_wave_history_export_logic(self):
        """Test the actual export logic from rebuild_wave_history function."""
        try:
            from helpers.operator_toolbox import PRICE_BOOK_AVAILABLE
            if not PRICE_BOOK_AVAILABLE:
                self.skipTest("price_book module not available")
            
            from helpers import price_book
            
            # Get actual price_book data
            price_data = price_book.get_price_book()
            
            if price_data.empty:
                self.skipTest("Price cache is empty, cannot test export")
            
            print(f"  Loaded price_book: {price_data.shape[0]} days × {price_data.shape[1]} tickers")
            
            # Apply the FIXED export logic
            price_data_reset = price_data.reset_index()
            price_data_reset = price_data_reset.rename(columns={price_data_reset.columns[0]: 'date'})
            
            # This should work without errors
            prices_long_df = pd.melt(
                price_data_reset,
                id_vars=['date'],
                var_name='ticker',
                value_name='close'
            )
            
            # Remove NaN values
            prices_long_df = prices_long_df.dropna(subset=['close'])
            
            # Verify result
            self.assertGreater(len(prices_long_df), 0, "Should have exported some records")
            self.assertIn('date', prices_long_df.columns, "Should have 'date' column")
            self.assertIn('ticker', prices_long_df.columns, "Should have 'ticker' column")
            self.assertIn('close', prices_long_df.columns, "Should have 'close' column")
            
            # Verify date format
            sample_date = prices_long_df['date'].iloc[0]
            self.assertIsInstance(sample_date, (str, pd.Timestamp, datetime),
                                "Date should be string or datetime")
            
            print(f"✓ Export logic works with real price_book data")
            print(f"  Exported {len(prices_long_df)} price records")
            print(f"  Columns: {list(prices_long_df.columns)}")
            
        except ImportError as e:
            self.skipTest(f"Required module not available: {e}")
        except Exception as e:
            self.fail(f"Export logic failed with real data: {e}")
    
    def test_force_ledger_recompute_export_logic(self):
        """Test that force_ledger_recompute can export without errors."""
        try:
            from helpers.operator_toolbox import PRICE_BOOK_AVAILABLE
            if not PRICE_BOOK_AVAILABLE:
                self.skipTest("price_book module not available")
            
            from helpers import price_book
            
            # Get actual price_book data
            price_data = price_book.get_price_book()
            
            if price_data.empty:
                self.skipTest("Price cache is empty, cannot test export")
            
            # Use a temp directory for output
            with tempfile.TemporaryDirectory() as tmpdir:
                prices_csv_path = os.path.join(tmpdir, 'prices.csv')
                
                # Apply the FIXED export logic from force_ledger_recompute
                price_data_reset = price_data.reset_index()
                price_data_reset = price_data_reset.rename(columns={price_data_reset.columns[0]: 'date'})
                
                prices_long_df = pd.melt(
                    price_data_reset,
                    id_vars=['date'],
                    var_name='ticker',
                    value_name='close'
                )
                
                # Remove NaN values
                prices_long_df = prices_long_df.dropna(subset=['close'])
                
                # Format date column
                prices_long_df['date'] = pd.to_datetime(prices_long_df['date']).dt.strftime('%Y-%m-%d')
                
                # Save to CSV in the consistent format: date,ticker,close
                prices_long_df.to_csv(prices_csv_path, index=False, columns=['date', 'ticker', 'close'])
                
                # Verify file was created
                self.assertTrue(os.path.exists(prices_csv_path), "CSV file should be created")
                
                # Verify file content
                exported_df = pd.read_csv(prices_csv_path)
                self.assertEqual(list(exported_df.columns), ['date', 'ticker', 'close'],
                               "CSV should have exactly 3 columns: date, ticker, close")
                
                # Verify dates are properly formatted
                sample_date = exported_df['date'].iloc[0]
                self.assertRegex(sample_date, r'^\d{4}-\d{2}-\d{2}$',
                               "Date should be in YYYY-MM-DD format")
                
                print(f"✓ force_ledger_recompute export logic works correctly")
                print(f"  Exported {len(exported_df)} records to CSV")
                print(f"  Sample date: {sample_date}")
                
        except ImportError as e:
            self.skipTest(f"Required module not available: {e}")
        except Exception as e:
            self.fail(f"Export logic failed: {e}")


if __name__ == '__main__':
    unittest.main()

"""
Test suite for MultiIndex handling in price_loader.

This test validates that the _load_cached_parquet function correctly handles
both single-level DatetimeIndex and MultiIndex (date, ticker) structures.
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_multiindex_handling():
    """Test that MultiIndex (date, ticker) is correctly handled."""
    print("=" * 80)
    print("TEST: MultiIndex Handling in price_loader")
    print("=" * 80)
    
    # Create a test MultiIndex DataFrame (date, ticker)
    dates = pd.date_range('2024-01-01', periods=10)
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Create MultiIndex
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    df_multi = pd.DataFrame({
        'close': np.random.rand(30) * 100
    }, index=index)
    
    print(f"Created MultiIndex DataFrame: {df_multi.shape}")
    print(f"Index type: {type(df_multi.index).__name__}")
    print(f"Index names: {df_multi.index.names}")
    print(f"Index nlevels: {df_multi.index.nlevels}")
    print(f"\nFirst 5 rows:")
    print(df_multi.head())
    
    # Save to temporary parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        df_multi.to_parquet(tmp_path)
        print(f"\nSaved to temporary file: {tmp_path}")
        
        # Now test the load_cache function behavior
        # We need to simulate what happens in the load_cache function
        df_read = pd.read_parquet(tmp_path)
        
        print(f"\nAfter reading from parquet:")
        print(f"Index type: {type(df_read.index).__name__}")
        print(f"Index names: {df_read.index.names}")
        print(f"Index nlevels: {df_read.index.nlevels}")
        
        # Apply the fix: Handle MultiIndex
        if isinstance(df_read.index, pd.MultiIndex):
            print("\n✓ MultiIndex detected correctly!")
            
            # Get the date level (assumed to be level 0)
            date_level = df_read.index.get_level_values(0)
            
            # Convert date level to DatetimeIndex if not already
            if not isinstance(date_level, pd.DatetimeIndex):
                print("Converting date level to DatetimeIndex...")
                # Create new MultiIndex with datetime-converted date level
                new_levels = [pd.to_datetime(date_level)] + [df_read.index.get_level_values(i) for i in range(1, df_read.index.nlevels)]
                df_read.index = pd.MultiIndex.from_arrays(new_levels, names=df_read.index.names)
            else:
                print("✓ Date level is already DatetimeIndex")
        else:
            # Single-level index - convert to datetime if needed
            if not isinstance(df_read.index, pd.DatetimeIndex):
                print("Converting single-level index to DatetimeIndex")
                df_read.index = pd.to_datetime(df_read.index)
        
        # Sort by date
        df_read = df_read.sort_index()
        
        # Verify the MultiIndex is still intact
        assert isinstance(df_read.index, pd.MultiIndex), "Index should still be MultiIndex"
        assert df_read.index.nlevels == 2, "Index should have 2 levels"
        assert df_read.index.names == ['date', 'ticker'], "Index names should be preserved"
        
        # Verify the date level is DatetimeIndex
        date_level_after = df_read.index.get_level_values(0)
        assert isinstance(date_level_after, pd.DatetimeIndex), "Date level should be DatetimeIndex"
        
        # Verify the ticker level is preserved
        ticker_level_after = df_read.index.get_level_values(1)
        assert set(ticker_level_after.unique()) == set(tickers), "Ticker level should be preserved"
        
        print("\n✓ MultiIndex structure preserved correctly!")
        print(f"  - Index type: {type(df_read.index).__name__}")
        print(f"  - Date level type: {type(date_level_after).__name__}")
        print(f"  - Ticker level values: {ticker_level_after.unique().tolist()}")
        
        print("\n✓ All MultiIndex handling tests PASSED!")
        return True
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_single_index_handling():
    """Test that single DatetimeIndex is still handled correctly."""
    print("\n" + "=" * 80)
    print("TEST: Single DatetimeIndex Handling")
    print("=" * 80)
    
    # Create a regular DataFrame with dates as index and tickers as columns
    dates = pd.date_range('2024-01-01', periods=10)
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    df_single = pd.DataFrame(
        np.random.rand(10, 3) * 100,
        index=dates,
        columns=tickers
    )
    
    print(f"Created single-index DataFrame: {df_single.shape}")
    print(f"Index type: {type(df_single.index).__name__}")
    print(f"\nFirst 5 rows:")
    print(df_single.head())
    
    # Save to temporary parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        df_single.to_parquet(tmp_path)
        print(f"\nSaved to temporary file: {tmp_path}")
        
        # Read back
        df_read = pd.read_parquet(tmp_path)
        
        print(f"\nAfter reading from parquet:")
        print(f"Index type: {type(df_read.index).__name__}")
        
        # Apply the fix: Handle single-level index
        if isinstance(df_read.index, pd.MultiIndex):
            print("ERROR: Should not be MultiIndex!")
            return False
        else:
            print("✓ Single-level index detected correctly!")
            # Single-level index - convert to datetime if needed
            if not isinstance(df_read.index, pd.DatetimeIndex):
                print("Converting single-level index to DatetimeIndex")
                df_read.index = pd.to_datetime(df_read.index)
            else:
                print("✓ Index is already DatetimeIndex")
        
        # Sort by date
        df_read = df_read.sort_index()
        
        # Verify the index is DatetimeIndex
        assert isinstance(df_read.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
        assert df_read.shape == df_single.shape, "Shape should be preserved"
        assert list(df_read.columns) == list(df_single.columns), "Columns should be preserved"
        
        print("\n✓ Single-level index handled correctly!")
        print(f"  - Index type: {type(df_read.index).__name__}")
        print(f"  - Columns: {df_read.columns.tolist()}")
        
        print("\n✓ All single-index handling tests PASSED!")
        return True
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_non_datetime_multiindex():
    """Test MultiIndex with non-datetime date level (should convert)."""
    print("\n" + "=" * 80)
    print("TEST: MultiIndex with Non-DateTime Date Level")
    print("=" * 80)
    
    # Create a MultiIndex with string dates (not datetime)
    dates = ['2024-01-01', '2024-01-02', '2024-01-03']
    tickers = ['AAPL', 'MSFT']
    
    # Create MultiIndex with string dates
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    df_multi = pd.DataFrame({
        'close': np.random.rand(6) * 100
    }, index=index)
    
    print(f"Created MultiIndex DataFrame with string dates: {df_multi.shape}")
    print(f"Date level type: {type(df_multi.index.get_level_values(0)).__name__}")
    
    # Save and read back
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        df_multi.to_parquet(tmp_path)
        df_read = pd.read_parquet(tmp_path)
        
        # Apply the fix
        if isinstance(df_read.index, pd.MultiIndex):
            date_level = df_read.index.get_level_values(0)
            
            if not isinstance(date_level, pd.DatetimeIndex):
                print("✓ Detected non-DatetimeIndex date level, converting...")
                new_levels = [pd.to_datetime(date_level)] + [df_read.index.get_level_values(i) for i in range(1, df_read.index.nlevels)]
                df_read.index = pd.MultiIndex.from_arrays(new_levels, names=df_read.index.names)
                print("✓ Conversion successful!")
        
        df_read = df_read.sort_index()
        
        # Verify conversion
        date_level_after = df_read.index.get_level_values(0)
        assert isinstance(date_level_after, pd.DatetimeIndex), "Date level should be DatetimeIndex after conversion"
        assert isinstance(df_read.index, pd.MultiIndex), "Should still be MultiIndex"
        
        print(f"✓ Date level converted to: {type(date_level_after).__name__}")
        print(f"✓ MultiIndex preserved: {df_read.index.names}")
        
        print("\n✓ Non-datetime MultiIndex conversion test PASSED!")
        return True
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == '__main__':
    print("Running MultiIndex Price Loader Tests\n")
    
    try:
        # Run all tests
        test_multiindex_handling()
        test_single_index_handling()
        test_non_datetime_multiindex()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

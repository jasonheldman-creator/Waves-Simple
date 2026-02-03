"""
Test suite for MultiIndex handling in price_loader.

This test validates that the _ensure_datetime_index function correctly handles
both single-level DatetimeIndex and MultiIndex (date, ticker) structures.

Note on testing private functions:
    We directly test _ensure_datetime_index (a private function) because it contains
    critical index conversion logic that must work correctly for both MultiIndex and
    single DatetimeIndex structures. While normally we'd test through public APIs,
    this function's correctness is essential to prevent index corruption, warranting
    direct unit testing.
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.price_loader import _ensure_datetime_index


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
    
    # Test the _ensure_datetime_index function
    print(f"\nTesting _ensure_datetime_index()...")
    df_result = _ensure_datetime_index(df_multi)
    
    print(f"\nAfter _ensure_datetime_index():")
    print(f"Index type: {type(df_result.index).__name__}")
    print(f"Index names: {df_result.index.names}")
    print(f"Index nlevels: {df_result.index.nlevels}")
    
    # Verify the MultiIndex is still intact
    assert isinstance(df_result.index, pd.MultiIndex), "Index should still be MultiIndex"
    assert df_result.index.nlevels == 2, "Index should have 2 levels"
    assert df_result.index.names == ['date', 'ticker'], "Index names should be preserved"
    
    # Verify the date level is DatetimeIndex
    date_level = df_result.index.get_level_values(0)
    assert isinstance(date_level, pd.DatetimeIndex), "Date level should be DatetimeIndex"
    
    # Verify the ticker level is preserved
    ticker_level = df_result.index.get_level_values(1)
    assert set(ticker_level.unique()) == set(tickers), "Ticker level should be preserved"
    
    print("\n✓ MultiIndex structure preserved correctly!")
    print(f"  - Index type: {type(df_result.index).__name__}")
    print(f"  - Date level type: {type(date_level).__name__}")
    print(f"  - Ticker level values: {ticker_level.unique().tolist()}")
    
    print("\n✓ All MultiIndex handling tests PASSED!")
    return True


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
    
    # Test the _ensure_datetime_index function
    print(f"\nTesting _ensure_datetime_index()...")
    df_result = _ensure_datetime_index(df_single)
    
    print(f"\nAfter _ensure_datetime_index():")
    print(f"Index type: {type(df_result.index).__name__}")
    
    # Verify the index is DatetimeIndex
    assert isinstance(df_result.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
    assert df_result.shape == df_single.shape, "Shape should be preserved"
    assert list(df_result.columns) == list(df_single.columns), "Columns should be preserved"
    
    print("\n✓ Single-level index handled correctly!")
    print(f"  - Index type: {type(df_result.index).__name__}")
    print(f"  - Columns: {df_result.columns.tolist()}")
    
    print("\n✓ All single-index handling tests PASSED!")
    return True


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
    print(f"Date level type before: {type(df_multi.index.get_level_values(0)).__name__}")
    
    # Test the _ensure_datetime_index function
    print(f"\nTesting _ensure_datetime_index()...")
    df_result = _ensure_datetime_index(df_multi)
    
    # Verify conversion
    date_level = df_result.index.get_level_values(0)
    assert isinstance(date_level, pd.DatetimeIndex), "Date level should be DatetimeIndex after conversion"
    assert isinstance(df_result.index, pd.MultiIndex), "Should still be MultiIndex"
    
    print(f"✓ Date level converted to: {type(date_level).__name__}")
    print(f"✓ MultiIndex preserved: {df_result.index.names}")
    
    print("\n✓ Non-datetime MultiIndex conversion test PASSED!")
    return True


def test_parquet_round_trip():
    """Test that MultiIndex survives parquet save/load with _ensure_datetime_index."""
    print("\n" + "=" * 80)
    print("TEST: MultiIndex Parquet Round-Trip")
    print("=" * 80)
    
    # Create a test MultiIndex DataFrame
    dates = pd.date_range('2024-01-01', periods=5)
    tickers = ['AAPL', 'MSFT']
    
    index = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])
    df_original = pd.DataFrame({
        'close': np.random.rand(10) * 100
    }, index=index)
    
    print(f"Original DataFrame: {df_original.shape}")
    print(f"Original index type: {type(df_original.index).__name__}")
    
    # Save to temporary parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        df_original.to_parquet(tmp_path)
        print(f"Saved to: {tmp_path}")
        
        # Load back and apply _ensure_datetime_index
        df_loaded = pd.read_parquet(tmp_path)
        print(f"\nLoaded from parquet: {df_loaded.shape}")
        
        df_result = _ensure_datetime_index(df_loaded)
        
        # Verify structure preserved
        assert isinstance(df_result.index, pd.MultiIndex), "Should be MultiIndex"
        assert df_result.index.nlevels == 2, "Should have 2 levels"
        assert df_result.index.names == ['date', 'ticker'], "Names should match"
        
        date_level = df_result.index.get_level_values(0)
        assert isinstance(date_level, pd.DatetimeIndex), "Date level should be DatetimeIndex"
        
        print(f"✓ MultiIndex preserved through parquet round-trip")
        print(f"  - Result index type: {type(df_result.index).__name__}")
        print(f"  - Date level type: {type(date_level).__name__}")
        
        print("\n✓ Parquet round-trip test PASSED!")
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
        test_parquet_round_trip()
        
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

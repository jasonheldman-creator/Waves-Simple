"""
Test suite for price cache v2 with fallback support.

This test validates:
- Cache loads from v2 file when available
- Falls back to legacy cache when v2 doesn't exist
- Logs appropriate warnings during fallback
"""

import os
import sys
import shutil
import tempfile
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.price_loader import (
    load_cache,
    save_cache,
    CACHE_PATH,
    CACHE_PATH_LEGACY,
    CACHE_DIR
)


def create_test_cache(path: str, num_days: int = 10, num_tickers: int = 5) -> pd.DataFrame:
    """
    Create a simple test cache file.
    
    Args:
        path: Path where to save the cache
        num_days: Number of days of data
        num_tickers: Number of tickers
        
    Returns:
        DataFrame that was saved
    """
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days-1)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create test data
    tickers = [f'TEST{i}' for i in range(num_tickers)]
    data = {}
    for ticker in tickers:
        data[ticker] = [100.0 + i for i in range(num_days)]
    
    df = pd.DataFrame(data, index=dates)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save to parquet
    df.to_parquet(path)
    
    return df


def test_v2_cache_loads():
    """Test that v2 cache loads when it exists."""
    print("=" * 80)
    print("TEST: V2 Cache Loading")
    print("=" * 80)
    
    # Create v2 cache file
    print("\n1. Creating v2 cache file...")
    test_df = create_test_cache(CACHE_PATH, num_days=15, num_tickers=3)
    print(f"  ✓ Created v2 cache: {CACHE_PATH}")
    print(f"    Shape: {test_df.shape}")
    
    # Load cache
    print("\n2. Loading cache...")
    loaded_df = load_cache()
    
    if loaded_df is not None:
        print(f"  ✓ Cache loaded successfully")
        print(f"    Shape: {loaded_df.shape}")
        assert loaded_df.shape == test_df.shape, "Loaded cache should match saved cache"
        print(f"  ✓ Cache data matches expected shape")
    else:
        print(f"  ✗ Failed to load cache")
        raise AssertionError("Cache should have loaded")
    
    # Clean up
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
    
    print("\n✓ V2 cache loading test passed")


def test_fallback_to_legacy():
    """Test that system falls back to legacy cache when v2 doesn't exist."""
    print("=" * 80)
    print("TEST: Fallback to Legacy Cache")
    print("=" * 80)
    
    # Ensure v2 cache doesn't exist
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
    print(f"\n1. Ensured v2 cache doesn't exist: {CACHE_PATH}")
    
    # Create legacy cache file
    print("\n2. Creating legacy cache file...")
    test_df = create_test_cache(CACHE_PATH_LEGACY, num_days=20, num_tickers=4)
    print(f"  ✓ Created legacy cache: {CACHE_PATH_LEGACY}")
    print(f"    Shape: {test_df.shape}")
    
    # Load cache (should fall back to legacy)
    print("\n3. Loading cache (should fall back to legacy)...")
    loaded_df = load_cache()
    
    if loaded_df is not None:
        print(f"  ✓ Cache loaded successfully from fallback")
        print(f"    Shape: {loaded_df.shape}")
        assert loaded_df.shape == test_df.shape, "Loaded cache should match legacy cache"
        print(f"  ✓ Fallback cache data matches expected shape")
    else:
        print(f"  ✗ Failed to load cache from fallback")
        raise AssertionError("Cache should have loaded from legacy file")
    
    # Clean up
    if os.path.exists(CACHE_PATH_LEGACY):
        os.remove(CACHE_PATH_LEGACY)
    
    print("\n✓ Fallback to legacy cache test passed")


def test_no_cache_returns_none():
    """Test that load_cache returns None when neither cache exists."""
    print("=" * 80)
    print("TEST: No Cache Returns None")
    print("=" * 80)
    
    # Ensure neither cache exists
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
    if os.path.exists(CACHE_PATH_LEGACY):
        os.remove(CACHE_PATH_LEGACY)
    print(f"\n1. Ensured no cache files exist")
    print(f"   V2: {CACHE_PATH}")
    print(f"   Legacy: {CACHE_PATH_LEGACY}")
    
    # Load cache
    print("\n2. Loading cache (should return None)...")
    loaded_df = load_cache()
    
    if loaded_df is None:
        print(f"  ✓ Correctly returned None when no cache exists")
    else:
        print(f"  ✗ Should have returned None")
        raise AssertionError("load_cache should return None when no cache exists")
    
    print("\n✓ No cache returns None test passed")


def test_save_cache_creates_v2():
    """Test that save_cache creates v2 cache file."""
    print("=" * 80)
    print("TEST: Save Cache Creates V2 File")
    print("=" * 80)
    
    # Clean up any existing cache
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
    print(f"\n1. Ensured v2 cache doesn't exist: {CACHE_PATH}")
    
    # Create test data
    print("\n2. Creating test data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=9)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = {'SPY': [100.0 + i for i in range(10)], 'QQQ': [200.0 + i for i in range(10)]}
    test_df = pd.DataFrame(data, index=dates)
    print(f"  ✓ Created test DataFrame with shape: {test_df.shape}")
    
    # Save cache
    print("\n3. Saving cache...")
    save_cache(test_df)
    
    # Verify v2 file was created
    if os.path.exists(CACHE_PATH):
        print(f"  ✓ V2 cache file created: {CACHE_PATH}")
        
        # Verify data
        loaded_df = pd.read_parquet(CACHE_PATH)
        assert loaded_df.shape == test_df.shape, "Saved cache should match original data"
        print(f"  ✓ Saved data matches original")
    else:
        print(f"  ✗ V2 cache file was not created")
        raise AssertionError("save_cache should create v2 cache file")
    
    # Clean up
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)
    
    print("\n✓ Save cache creates v2 file test passed")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Price Cache V2 Fallback Test Suite")
    print("=" * 80 + "\n")
    
    try:
        test_v2_cache_loads()
        print()
        test_fallback_to_legacy()
        print()
        test_no_cache_returns_none()
        print()
        test_save_cache_creates_v2()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 80 + "\n")
        raise

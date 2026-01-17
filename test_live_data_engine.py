"""
Test script for live data engine functionality.
"""

import os
os.environ['LIVE_DATA_ENABLED'] = 'false'  # Use cache fallback for testing

from helpers.live_data_engine import build_live_price_book, get_required_tickers_from_cache

def test_live_data_engine():
    """Test the live data engine with cache fallback."""
    print("=" * 70)
    print("Testing Live Data Engine")
    print("=" * 70)
    
    # Get tickers from cache
    print("\n1. Getting tickers from cache...")
    tickers = get_required_tickers_from_cache()
    print(f"   Found {len(tickers)} tickers: {tickers[:5]}..." if len(tickers) > 5 else f"   Found {len(tickers)} tickers: {tickers}")
    
    # Build PRICE_BOOK from live data (will use cache fallback since LIVE_DATA_ENABLED=false)
    print("\n2. Building PRICE_BOOK from live data engine...")
    price_book, metadata = build_live_price_book(tickers=tickers, use_cache_fallback=True)
    
    # Display metadata
    print("\n3. Metadata:")
    for key, value in metadata.items():
        if key != 'failed_tickers':
            print(f"   {key}: {value}")
    
    # Display PRICE_BOOK info
    print("\n4. PRICE_BOOK Info:")
    if not price_book.empty:
        print(f"   Shape: {price_book.shape[0]} rows × {price_book.shape[1]} columns")
        print(f"   Date range: {price_book.index[0]} to {price_book.index[-1]}")
        print(f"   First 5 tickers: {price_book.columns[:5].tolist()}")
        print(f"   Memory ID: {hex(id(price_book))}")
    else:
        print("   PRICE_BOOK is empty!")
    
    # Test second call to verify new object is created (no caching)
    print("\n5. Testing second call (should create new object)...")
    price_book2, metadata2 = build_live_price_book(tickers=tickers, use_cache_fallback=True)
    print(f"   First call memory ID:  {metadata['memory_id']}")
    print(f"   Second call memory ID: {metadata2['memory_id']}")
    print(f"   Different objects: {metadata['memory_id'] != metadata2['memory_id']}")
    
    print("\n" + "=" * 70)
    print("✓ Live Data Engine Test Complete")
    print("=" * 70)

if __name__ == '__main__':
    test_live_data_engine()

"""
Test script for live Yahoo Finance data fetching.
"""

import os
os.environ['LIVE_DATA_ENABLED'] = 'true'  # Enable live data fetching

from helpers.live_data_engine import build_live_price_book

def test_live_yahoo_fetch():
    """Test live data fetching from Yahoo Finance with a small set of tickers."""
    print("=" * 70)
    print("Testing Live Yahoo Finance Data Fetch")
    print("=" * 70)
    
    # Test with a small set of common tickers
    test_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA']
    
    print(f"\n1. Fetching live data for {len(test_tickers)} tickers: {test_tickers}")
    print("   This may take 10-30 seconds...")
    
    # Build PRICE_BOOK from live Yahoo Finance
    price_book, metadata = build_live_price_book(tickers=test_tickers, use_cache_fallback=True, period="1y")
    
    # Display metadata
    print("\n2. Fetch Results:")
    print(f"   Data Source: {metadata.get('data_source', 'Unknown')}")
    print(f"   Success: {metadata.get('success', False)}")
    print(f"   Tickers Requested: {metadata.get('tickers_requested', 0)}")
    print(f"   Tickers Fetched: {metadata.get('tickers_fetched', 0)}")
    print(f"   Tickers Failed: {metadata.get('tickers_failed', 0)}")
    print(f"   Timestamp UTC: {metadata.get('timestamp_utc', 'N/A')}")
    print(f"   Render UTC: {metadata.get('render_utc', 'N/A')}")
    print(f"   Memory ID: {metadata.get('memory_id', 'N/A')}")
    
    # Display PRICE_BOOK info
    print("\n3. PRICE_BOOK Info:")
    if not price_book.empty:
        print(f"   Shape: {price_book.shape[0]} rows × {price_book.shape[1]} columns")
        print(f"   Date range: {price_book.index[0]} to {price_book.index[-1]}")
        print(f"   Tickers: {price_book.columns.tolist()}")
        print(f"   Latest prices:")
        for ticker in price_book.columns:
            latest_price = price_book[ticker].iloc[-1]
            print(f"      {ticker}: ${latest_price:.2f}")
    else:
        print("   PRICE_BOOK is empty!")
        if 'failed_tickers' in metadata:
            print(f"   Failed tickers: {metadata['failed_tickers']}")
    
    print("\n" + "=" * 70)
    print("✓ Live Yahoo Finance Test Complete")
    print("=" * 70)

if __name__ == '__main__':
    test_live_yahoo_fetch()

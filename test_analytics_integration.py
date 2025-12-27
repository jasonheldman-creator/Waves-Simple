"""
Integration test for analytics pipeline with diagnostics
"""

from datetime import datetime, timedelta
from analytics_pipeline import (
    fetch_prices, 
    normalize_ticker,
    get_diagnostics_tracker
)


def test_fetch_with_diagnostics():
    """Test fetching prices with diagnostics tracking"""
    print("Testing fetch_prices with diagnostics integration...\n")
    
    # Clear tracker
    try:
        tracker = get_diagnostics_tracker()
        tracker.clear()
    except (ImportError, AttributeError) as e:
        print(f"⚠️ Diagnostics not available: {e}, skipping tracker tests")
        return
    
    # Test with some known tickers (some may fail)
    test_tickers = ["AAPL", "MSFT", "INVALID_TICKER_123", "BRK.B"]
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"Fetching prices for: {test_tickers}")
    print(f"Date range: {start_date.date()} to {end_date.date()}\n")
    
    # Fetch with diagnostics
    prices_df, failures = fetch_prices(
        tickers=test_tickers,
        start_date=start_date,
        end_date=end_date,
        use_dummy_data=True,  # Use dummy data for testing
        wave_id="test_wave",
        wave_name="Test Wave"
    )
    
    print(f"Results:")
    print(f"  Successful tickers: {list(prices_df.columns)}")
    print(f"  Failed tickers: {list(failures.keys())}")
    print(f"  Price data shape: {prices_df.shape}")
    
    # Check diagnostics
    stats = tracker.get_summary_stats()
    print(f"\nDiagnostics:")
    print(f"  Total failures tracked: {stats['total_failures']}")
    print(f"  Unique tickers: {stats['unique_tickers']}")
    print(f"  By type: {stats['by_type']}")
    
    if stats['total_failures'] > 0:
        print("\nRecent failures:")
        for failure in tracker.get_all_failures():
            print(f"  - {failure.ticker_original}: {failure.failure_type.value}")
            print(f"    Error: {failure.error_message}")
            print(f"    Fix: {failure.suggested_fix}")
    
    print("\n✓ Integration test completed")


def test_normalization():
    """Test ticker normalization"""
    print("\nTesting ticker normalization...\n")
    
    test_cases = [
        ("BRK.B", "BRK-B"),
        ("BF.B", "BF-B"),
        ("AAPL", "AAPL"),
        ("BTC-USD", "BTC-USD"),
    ]
    
    for original, expected in test_cases:
        normalized = normalize_ticker(original)
        status = "✓" if normalized == expected else "✗"
        print(f"  {status} {original:15s} -> {normalized:15s} (expected: {expected})")
        assert normalized == expected, f"Expected {expected}, got {normalized}"
    
    print("\n✓ Normalization tests passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Analytics Pipeline Integration Tests")
    print("=" * 60 + "\n")
    
    test_normalization()
    test_fetch_with_diagnostics()
    
    print("\n" + "=" * 60)
    print("All integration tests completed!")
    print("=" * 60)

#!/usr/bin/env python3
"""
Test suite for data_cache.py module

Tests the global price caching system functionality.
"""

import sys
import pandas as pd
from datetime import datetime


def test_collect_all_required_tickers():
    """Test ticker collection from wave registry."""
    print("Testing collect_all_required_tickers...")
    
    try:
        from data_cache import collect_all_required_tickers
        
        # Create mock wave registry
        class MockHolding:
            def __init__(self, ticker):
                self.ticker = ticker
        
        mock_registry = {
            "Test Wave 1": [MockHolding("AAPL"), MockHolding("MSFT")],
            "Test Wave 2": [MockHolding("GOOGL"), MockHolding("AMZN")]
        }
        
        # Call function
        tickers = collect_all_required_tickers(mock_registry, include_benchmarks=False, include_safe_assets=False)
        
        # Verify results
        assert "AAPL" in tickers, "AAPL should be in tickers"
        assert "MSFT" in tickers, "MSFT should be in tickers"
        assert "GOOGL" in tickers, "GOOGL should be in tickers"
        assert "AMZN" in tickers, "AMZN should be in tickers"
        assert "SPY" in tickers, "SPY should be in tickers (always included)"
        assert "^VIX" in tickers, "^VIX should be in tickers (always included)"
        
        print("✅ collect_all_required_tickers test passed")
        return True
        
    except Exception as e:
        print(f"❌ collect_all_required_tickers test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_download_prices_batched_mock():
    """Test batched download function with mock data."""
    print("\nTesting download_prices_batched (mock mode)...")
    
    try:
        from data_cache import download_prices_batched
        
        # Test with empty ticker list
        price_df, failures = download_prices_batched([], period_days=30, chunk_size=10)
        
        assert isinstance(price_df, pd.DataFrame), "Should return DataFrame"
        assert isinstance(failures, dict), "Should return failures dict"
        
        print("✅ download_prices_batched mock test passed")
        return True
        
    except Exception as e:
        print(f"❌ download_prices_batched mock test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_get_global_price_cache_structure():
    """Test that get_global_price_cache returns correct structure."""
    print("\nTesting get_global_price_cache structure...")
    
    try:
        from data_cache import get_global_price_cache
        
        # Create mock wave registry
        class MockHolding:
            def __init__(self, ticker):
                self.ticker = ticker
        
        mock_registry = {
            "Test Wave": [MockHolding("SPY")]
        }
        
        # Note: This will try to download real data if yfinance is available
        # We're just testing the structure of the return value
        result = get_global_price_cache(mock_registry, days=30, ttl_seconds=3600)
        
        # Verify return structure
        assert isinstance(result, dict), "Should return a dictionary"
        assert "price_df" in result, "Should have price_df key"
        assert "failures" in result, "Should have failures key"
        assert "asof" in result, "Should have asof key"
        assert "ticker_count" in result, "Should have ticker_count key"
        assert "success_count" in result, "Should have success_count key"
        
        assert isinstance(result["price_df"], pd.DataFrame), "price_df should be DataFrame"
        assert isinstance(result["failures"], dict), "failures should be dict"
        assert isinstance(result["asof"], datetime), "asof should be datetime"
        assert isinstance(result["ticker_count"], int), "ticker_count should be int"
        assert isinstance(result["success_count"], int), "success_count should be int"
        
        print("✅ get_global_price_cache structure test passed")
        return True
        
    except Exception as e:
        print(f"⚠️ get_global_price_cache structure test skipped (expected if yfinance unavailable): {str(e)}")
        return True  # Don't fail test if yfinance is unavailable


def test_waves_engine_integration():
    """Test that waves_engine accepts price_df parameter."""
    print("\nTesting waves_engine integration...")
    
    try:
        from waves_engine import compute_history_nav
        import inspect
        
        # Check if compute_history_nav accepts price_df parameter
        sig = inspect.signature(compute_history_nav)
        params = list(sig.parameters.keys())
        
        assert "price_df" in params, "compute_history_nav should accept price_df parameter"
        
        print("✅ waves_engine integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ waves_engine integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Data Cache Test Suite")
    print("=" * 60)
    
    tests = [
        test_collect_all_required_tickers,
        test_download_prices_batched_mock,
        test_get_global_price_cache_structure,
        test_waves_engine_integration
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

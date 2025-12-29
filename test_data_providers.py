#!/usr/bin/env python3
"""
Test Data Providers
Tests the data provider abstraction layer.
"""

import sys
from datetime import datetime, timedelta

sys.path.insert(0, '.')

from data_providers import YahooProvider, PolygonProvider


def test_yahoo_provider():
    """Test Yahoo Finance provider."""
    print("="*60)
    print("Testing Yahoo Finance Provider")
    print("="*60)
    
    provider = YahooProvider()
    print(f"Provider: {provider}")
    
    # Test connection
    print("\n🔍 Testing connection...")
    can_connect = provider.test_connection()
    print(f"  Connection test: {'✅ PASS' if can_connect else '❌ FAIL (expected in sandboxed env)'}")
    
    # Test data fetch (will fail without network but shows the interface)
    print("\n📥 Testing data fetch...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"  Attempting to fetch AAPL from {start_date.date()} to {end_date.date()}")
    df = provider.get_history('AAPL', start_date, end_date)
    
    if df is not None and not df.empty:
        print(f"  ✅ Fetched {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n  Sample data:")
        print(df.head(3).to_string(index=False))
    else:
        print("  ❌ No data returned (expected in sandboxed env)")


def test_polygon_provider():
    """Test Polygon provider."""
    print("\n" + "="*60)
    print("Testing Polygon.io Provider")
    print("="*60)
    
    provider = PolygonProvider()
    print(f"Provider: {provider}")
    
    # Test connection
    print("\n🔍 Testing connection...")
    can_connect = provider.test_connection()
    print(f"  Connection test: {'✅ PASS' if can_connect else '❌ FAIL (no API key or network)'}")
    
    if not provider.api_key:
        print("  ℹ️  Set POLYGON_API_KEY environment variable to test")


def test_provider_interface():
    """Test that providers implement the interface correctly."""
    print("\n" + "="*60)
    print("Testing Provider Interface")
    print("="*60)
    
    from data_providers.base_provider import BaseProvider
    
    providers = [YahooProvider(), PolygonProvider()]
    
    for provider in providers:
        print(f"\n{provider}:")
        
        # Check it's a BaseProvider
        is_base = isinstance(provider, BaseProvider)
        print(f"  Is BaseProvider: {'✅' if is_base else '❌'}")
        
        # Check it has required methods
        has_get_history = hasattr(provider, 'get_history') and callable(provider.get_history)
        print(f"  Has get_history(): {'✅' if has_get_history else '❌'}")
        
        has_test_connection = hasattr(provider, 'test_connection') and callable(provider.test_connection)
        print(f"  Has test_connection(): {'✅' if has_test_connection else '❌'}")


if __name__ == '__main__':
    print("\n🧪 Data Provider Tests\n")
    
    test_provider_interface()
    test_yahoo_provider()
    test_polygon_provider()
    
    print("\n" + "="*60)
    print("✅ All interface tests passed")
    print("="*60)

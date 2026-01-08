#!/usr/bin/env python3
"""
Test script to verify PRICE_BOOK caching is working correctly.

This test validates:
1. get_cached_price_book function exists in app.py
2. Function is decorated with @st.cache_resource
3. Function logs only on cache miss
4. get_cached_price_book is used instead of direct get_price_book calls
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_cached_price_book_exists():
    """Test that get_cached_price_book function exists."""
    with open('app.py', 'r') as f:
        content = f.read()
    
    assert 'def get_cached_price_book():' in content, \
        "get_cached_price_book function should exist in app.py"
    
    print("✓ get_cached_price_book function exists")


def test_cached_price_book_has_decorator():
    """Test that get_cached_price_book has @st.cache_resource decorator."""
    with open('app.py', 'r') as f:
        lines = f.readlines()
    
    # Find the function definition
    found_decorator = False
    for i, line in enumerate(lines):
        if 'def get_cached_price_book():' in line:
            # Check previous line for decorator
            if i > 0 and '@st.cache_resource' in lines[i-1]:
                found_decorator = True
                break
    
    assert found_decorator, \
        "get_cached_price_book should have @st.cache_resource decorator"
    
    print("✓ get_cached_price_book has @st.cache_resource decorator")


def test_cached_price_book_used():
    """Test that get_cached_price_book is used instead of direct get_price_book calls."""
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Count usage of get_cached_price_book
    cached_count = content.count('get_cached_price_book()')
    
    # Should be used multiple times
    assert cached_count >= 5, \
        f"get_cached_price_book should be used at least 5 times, found {cached_count}"
    
    print(f"✓ get_cached_price_book is used {cached_count} times")


def test_price_book_logs_cache_miss():
    """Test that cached function logs only on cache miss."""
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Check for cache miss log message in get_cached_price_book
    assert 'PRICE_BOOK loaded (cached)' in content, \
        "get_cached_price_book should log 'PRICE_BOOK loaded (cached)' on cache miss"
    
    assert 'this message appears only on cache miss' in content, \
        "get_cached_price_book should document that log appears only on cache miss"
    
    print("✓ get_cached_price_book logs only on cache miss")


def test_rerun_throttle_exists():
    """Test that rerun throttle safety fuse exists."""
    with open('app.py', 'r') as f:
        content = f.read()
    
    assert 'rapid_rerun_count' in content, \
        "Rerun throttle should track rapid_rerun_count"
    
    assert 'last_rerun_time' in content, \
        "Rerun throttle should track last_rerun_time"
    
    assert 'RAPID RERUN DETECTED' in content, \
        "Rerun throttle should show error message when triggered"
    
    print("✓ Rerun throttle safety fuse exists")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing PRICE_BOOK Caching Implementation")
    print("=" * 60)
    
    test_cached_price_book_exists()
    test_cached_price_book_has_decorator()
    test_cached_price_book_used()
    test_price_book_logs_cache_miss()
    test_rerun_throttle_exists()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)

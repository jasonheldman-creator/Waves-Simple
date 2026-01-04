"""
Test suite for ticker normalization helper.

Tests the normalize_ticker function to ensure it handles:
1. None values
2. Whitespace trimming
3. Case conversion (uppercase)
4. Various dash/hyphen Unicode characters
5. Dots to hyphens conversion
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from the module file to avoid helpers/__init__.py dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ticker_normalize",
    os.path.join(os.path.dirname(__file__), "helpers", "ticker_normalize.py")
)
ticker_normalize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ticker_normalize)
normalize_ticker = ticker_normalize.normalize_ticker


def test_none_handling():
    """Test that None is converted to empty string."""
    result = normalize_ticker(None)
    assert result == "", f"Expected empty string for None, got '{result}'"
    print("✓ None handling test passed")


def test_whitespace_stripping():
    """Test that whitespace is stripped."""
    test_cases = [
        ("  AAPL  ", "AAPL"),
        (" MSFT", "MSFT"),
        ("GOOGL ", "GOOGL"),
        ("\tTSLA\t", "TSLA"),
    ]
    
    for input_val, expected in test_cases:
        result = normalize_ticker(input_val)
        assert result == expected, f"Expected '{expected}' for '{input_val}', got '{result}'"
    
    print("✓ Whitespace stripping test passed")


def test_uppercase_conversion():
    """Test that lowercase is converted to uppercase."""
    test_cases = [
        ("aapl", "AAPL"),
        ("msft", "MSFT"),
        ("GoOgL", "GOOGL"),
        ("TsLa", "TSLA"),
    ]
    
    for input_val, expected in test_cases:
        result = normalize_ticker(input_val)
        assert result == expected, f"Expected '{expected}' for '{input_val}', got '{result}'"
    
    print("✓ Uppercase conversion test passed")


def test_dot_to_hyphen():
    """Test that dots are replaced with hyphens."""
    test_cases = [
        ("BRK.B", "BRK-B"),
        ("BF.B", "BF-B"),
        ("A.B.C", "A-B-C"),
    ]
    
    for input_val, expected in test_cases:
        result = normalize_ticker(input_val)
        assert result == expected, f"Expected '{expected}' for '{input_val}', got '{result}'"
    
    print("✓ Dot to hyphen conversion test passed")


def test_unicode_dash_normalization():
    """Test that various Unicode dash characters are normalized to standard hyphen."""
    # Test en-dash (U+2013)
    result = normalize_ticker("BRK–B")
    assert result == "BRK-B", f"Expected 'BRK-B' for en-dash, got '{result}'"
    
    # Test em-dash (U+2014)
    result = normalize_ticker("BRK—B")
    assert result == "BRK-B", f"Expected 'BRK-B' for em-dash, got '{result}'"
    
    # Test hyphen (U+2010)
    result = normalize_ticker("BRK‐B")
    assert result == "BRK-B", f"Expected 'BRK-B' for hyphen, got '{result}'"
    
    # Test minus sign (U+2212)
    result = normalize_ticker("BRK−B")
    assert result == "BRK-B", f"Expected 'BRK-B' for minus sign, got '{result}'"
    
    print("✓ Unicode dash normalization test passed")


def test_combined_normalization():
    """Test combination of multiple normalization rules."""
    test_cases = [
        ("  brk.b  ", "BRK-B"),
        (" brk–b ", "BRK-B"),  # en-dash
        ("Brk.B", "BRK-B"),
        ("  BF.b  ", "BF-B"),
    ]
    
    for input_val, expected in test_cases:
        result = normalize_ticker(input_val)
        assert result == expected, f"Expected '{expected}' for '{input_val}', got '{result}'"
    
    print("✓ Combined normalization test passed")


def test_idempotency():
    """Test that normalizing already normalized tickers doesn't change them."""
    test_cases = ["AAPL", "MSFT", "BRK-B", "GOOGL"]
    
    for ticker in test_cases:
        result = normalize_ticker(ticker)
        assert result == ticker, f"Expected '{ticker}' to remain unchanged, got '{result}'"
        # Test double normalization
        result2 = normalize_ticker(result)
        assert result2 == ticker, f"Expected idempotency for '{ticker}', got '{result2}'"
    
    print("✓ Idempotency test passed")


def run_all_tests():
    """Run all ticker normalization tests."""
    print("="*70)
    print("Running ticker normalization tests...")
    print("="*70)
    
    try:
        test_none_handling()
        test_whitespace_stripping()
        test_uppercase_conversion()
        test_dot_to_hyphen()
        test_unicode_dash_normalization()
        test_combined_normalization()
        test_idempotency()
        
        print("="*70)
        print("✓ All ticker normalization tests passed!")
        print("="*70)
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        print("="*70)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

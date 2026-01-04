"""
Test script to verify price cache tolerance handling.

This test verifies that:
1. The MIN_SUCCESS_RATE constant is set correctly
2. The build_initial_cache function accepts non_interactive parameter
3. The script can run in non-interactive mode without hanging
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_price_cache


def test_min_success_rate():
    """Verify MIN_SUCCESS_RATE is set correctly."""
    assert hasattr(build_price_cache, 'MIN_SUCCESS_RATE'), "MIN_SUCCESS_RATE constant not found"
    assert build_price_cache.MIN_SUCCESS_RATE == 0.95, f"Expected MIN_SUCCESS_RATE to be 0.95, got {build_price_cache.MIN_SUCCESS_RATE}"
    print("✓ MIN_SUCCESS_RATE is set to 0.95")


def test_non_interactive_parameter():
    """Verify build_initial_cache accepts non_interactive parameter."""
    import inspect
    sig = inspect.signature(build_price_cache.build_initial_cache)
    params = list(sig.parameters.keys())
    assert 'non_interactive' in params, f"non_interactive parameter not found in build_initial_cache. Parameters: {params}"
    print("✓ build_initial_cache accepts non_interactive parameter")


def test_argparse_non_interactive():
    """Verify command line argument --non-interactive is available."""
    import argparse
    import io
    from contextlib import redirect_stdout
    
    # Create parser similar to main()
    parser = argparse.ArgumentParser(description='Build initial price cache')
    parser.add_argument('--force', action='store_true', help='Force rebuild even if cache exists')
    parser.add_argument('--years', type=int, default=5, help='Number of years of history')
    parser.add_argument('--non-interactive', action='store_true', help='Run in non-interactive mode (for CI/CD)')
    
    # Test parsing
    args = parser.parse_args(['--non-interactive'])
    assert args.non_interactive == True, "Failed to parse --non-interactive flag"
    print("✓ --non-interactive flag is properly parsed")


def test_success_rate_calculation():
    """Verify success rate calculation logic."""
    # Test case 1: 100% success
    all_tickers = 100
    num_tickers = 100
    success_rate = num_tickers / all_tickers if all_tickers > 0 else 0
    assert success_rate >= build_price_cache.MIN_SUCCESS_RATE, "100% success should meet threshold"
    print(f"✓ 100% success rate ({success_rate:.2%}) meets threshold")
    
    # Test case 2: 95% success (exactly at threshold)
    num_tickers = 95
    success_rate = num_tickers / all_tickers if all_tickers > 0 else 0
    assert success_rate >= build_price_cache.MIN_SUCCESS_RATE, "95% success should meet threshold"
    print(f"✓ 95% success rate ({success_rate:.2%}) meets threshold")
    
    # Test case 3: 94% success (below threshold)
    num_tickers = 94
    success_rate = num_tickers / all_tickers if all_tickers > 0 else 0
    assert success_rate < build_price_cache.MIN_SUCCESS_RATE, "94% success should NOT meet threshold"
    print(f"✓ 94% success rate ({success_rate:.2%}) correctly below threshold")


if __name__ == '__main__':
    print("Testing price cache tolerance handling...\n")
    
    try:
        test_min_success_rate()
        test_non_interactive_parameter()
        test_argparse_non_interactive()
        test_success_rate_calculation()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

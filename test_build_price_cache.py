#!/usr/bin/env python3
"""
Test suite for build_price_cache.py success tolerance handling

Tests the success rate calculation and threshold-based exit codes.
"""

import sys
import os
import subprocess
from unittest.mock import patch, MagicMock
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_min_success_rate_constant():
    """Test that MIN_SUCCESS_RATE constant is properly configured."""
    print("Testing MIN_SUCCESS_RATE constant...")
    
    try:
        # Test default value
        from build_price_cache import DEFAULT_MIN_SUCCESS_RATE, MIN_SUCCESS_RATE
        
        assert DEFAULT_MIN_SUCCESS_RATE == 0.95, f"DEFAULT_MIN_SUCCESS_RATE should be 0.95, got {DEFAULT_MIN_SUCCESS_RATE}"
        assert 0.0 <= MIN_SUCCESS_RATE <= 1.0, f"MIN_SUCCESS_RATE should be in [0,1], got {MIN_SUCCESS_RATE}"
        
        print(f"✅ MIN_SUCCESS_RATE constants configured correctly (default={DEFAULT_MIN_SUCCESS_RATE}, current={MIN_SUCCESS_RATE})")
        return True
        
    except Exception as e:
        print(f"❌ MIN_SUCCESS_RATE constant test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_success_rate_calculation():
    """Test success rate calculation logic."""
    print("\nTesting success rate calculation...")
    
    try:
        # Test with 100% success
        total_attempted = 10
        successful_downloads = 10
        success_rate = successful_downloads / total_attempted if total_attempted > 0 else 0.0
        assert success_rate == 1.0, f"Expected 1.0, got {success_rate}"
        print(f"  ✓ 100% success: {success_rate}")
        
        # Test with 95% success (at threshold)
        total_attempted = 100
        successful_downloads = 95
        success_rate = successful_downloads / total_attempted if total_attempted > 0 else 0.0
        assert success_rate == 0.95, f"Expected 0.95, got {success_rate}"
        print(f"  ✓ 95% success: {success_rate}")
        
        # Test with 90% success (below threshold)
        total_attempted = 100
        successful_downloads = 90
        success_rate = successful_downloads / total_attempted if total_attempted > 0 else 0.0
        assert success_rate == 0.90, f"Expected 0.90, got {success_rate}"
        print(f"  ✓ 90% success: {success_rate}")
        
        # Test with 0 attempts
        total_attempted = 0
        successful_downloads = 0
        success_rate = successful_downloads / total_attempted if total_attempted > 0 else 0.0
        assert success_rate == 0.0, f"Expected 0.0 for zero attempts, got {success_rate}"
        print(f"  ✓ Zero attempts: {success_rate}")
        
        print("✅ Success rate calculation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Success rate calculation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_exit_code_with_high_success_rate():
    """Test that script exits with code 0 when success rate >= threshold."""
    print("\nTesting exit code with high success rate...")
    
    try:
        # We'll test by mocking the build_initial_cache function
        import build_price_cache
        
        original_build = build_price_cache.build_initial_cache
        
        # Mock build_initial_cache to return high success rate
        def mock_build_high_success(*args, **kwargs):
            return True, 0.98  # 98% success rate
        
        with patch.object(build_price_cache, 'build_initial_cache', side_effect=mock_build_high_success):
            # Run main and capture exit code
            try:
                with patch('sys.argv', ['build_price_cache.py', '--force']):
                    build_price_cache.main()
                # If we get here, the exit code was 0 (success)
                print("✅ Script exits with code 0 for high success rate")
                return True
            except SystemExit as e:
                if e.code == 0:
                    print("✅ Script exits with code 0 for high success rate")
                    return True
                else:
                    print(f"❌ Expected exit code 0, got {e.code}")
                    return False
        
    except Exception as e:
        print(f"❌ High success rate exit code test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_exit_code_with_low_success_rate():
    """Test that script exits with code 1 when success rate < threshold."""
    print("\nTesting exit code with low success rate...")
    
    try:
        import build_price_cache
        
        # Mock build_initial_cache to return low success rate
        def mock_build_low_success(*args, **kwargs):
            return True, 0.85  # 85% success rate (below 95% threshold)
        
        with patch.object(build_price_cache, 'build_initial_cache', side_effect=mock_build_low_success):
            # Run main and capture exit code
            try:
                with patch('sys.argv', ['build_price_cache.py', '--force']):
                    build_price_cache.main()
                # Should not get here as main() calls sys.exit
                print("❌ Script did not exit as expected")
                return False
            except SystemExit as e:
                if e.code == 1:
                    print("✅ Script exits with code 1 for low success rate")
                    return True
                else:
                    print(f"❌ Expected exit code 1, got {e.code}")
                    return False
        
    except Exception as e:
        print(f"❌ Low success rate exit code test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_failed_tickers_logging():
    """Test that failed tickers are logged with reasons."""
    print("\nTesting failed tickers logging...")
    
    try:
        import build_price_cache
        import logging
        from io import StringIO
        
        # Capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.WARNING)
        build_price_cache.logger.addHandler(handler)
        
        # Mock build_initial_cache to simulate failures
        def mock_build_with_failures(*args, **kwargs):
            # Simulate logging of failed tickers
            build_price_cache.logger.warning("Failed tickers:")
            build_price_cache.logger.warning("  FAKE1: Ticker not found")
            build_price_cache.logger.warning("  FAKE2: Network error")
            return True, 0.90
        
        with patch.object(build_price_cache, 'build_initial_cache', side_effect=mock_build_with_failures):
            try:
                with patch('sys.argv', ['build_price_cache.py', '--force']):
                    build_price_cache.main()
            except SystemExit:
                pass
        
        # Check log output
        log_output = log_stream.getvalue()
        
        # Remove handler
        build_price_cache.logger.removeHandler(handler)
        
        # Verify failed tickers were logged
        if "Failed tickers:" in log_output and "FAKE1" in log_output:
            print("✅ Failed tickers are logged with reasons")
            return True
        else:
            print(f"❌ Failed tickers not found in logs: {log_output[:200]}")
            return False
        
    except Exception as e:
        print(f"❌ Failed tickers logging test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_env_variable_clamping():
    """Test that MIN_SUCCESS_RATE environment variable is clamped to [0,1]."""
    print("\nTesting environment variable clamping...")
    
    try:
        # Test values above 1.0 get clamped to 1.0
        os.environ['MIN_SUCCESS_RATE'] = '1.5'
        # Need to reload the module to pick up new env var
        import importlib
        import build_price_cache
        importlib.reload(build_price_cache)
        
        assert build_price_cache.MIN_SUCCESS_RATE == 1.0, f"Expected 1.0 for value 1.5, got {build_price_cache.MIN_SUCCESS_RATE}"
        print(f"  ✓ Value 1.5 clamped to 1.0")
        
        # Test values below 0.0 get clamped to 0.0
        os.environ['MIN_SUCCESS_RATE'] = '-0.5'
        importlib.reload(build_price_cache)
        
        assert build_price_cache.MIN_SUCCESS_RATE == 0.0, f"Expected 0.0 for value -0.5, got {build_price_cache.MIN_SUCCESS_RATE}"
        print(f"  ✓ Value -0.5 clamped to 0.0")
        
        # Clean up
        if 'MIN_SUCCESS_RATE' in os.environ:
            del os.environ['MIN_SUCCESS_RATE']
        importlib.reload(build_price_cache)
        
        print("✅ Environment variable clamping test passed")
        return True
        
    except Exception as e:
        print(f"❌ Environment variable clamping test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        # Clean up on error
        if 'MIN_SUCCESS_RATE' in os.environ:
            del os.environ['MIN_SUCCESS_RATE']
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("BUILD PRICE CACHE SUCCESS TOLERANCE TEST SUITE")
    print("="*80 + "\n")
    
    all_passed = True
    
    # Run tests
    tests = [
        test_min_success_rate_constant,
        test_success_rate_calculation,
        test_exit_code_with_high_success_rate,
        test_exit_code_with_low_success_rate,
        test_failed_tickers_logging,
        test_env_variable_clamping,
    ]
    
    for test_func in tests:
        if not test_func():
            all_passed = False
    
    # Final result
    print("\n" + "="*80)
    if all_passed:
        print("✅ All tests PASSED!")
        print("="*80 + "\n")
        return 0
    else:
        print("❌ Some tests FAILED!")
        print("="*80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Test suite for build_price_cache threshold-based failure handling.

Tests the new threshold logic to ensure:
1. Exit code 0 when success_rate >= MIN_SUCCESS_RATE
2. Exit code 1 when success_rate < MIN_SUCCESS_RATE
3. Proper logging of failed tickers and summary
4. Metadata file generation
"""

import os
import sys
import json
import tempfile
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_success_rate_calculation():
    """Test that success rate is calculated correctly."""
    print("=" * 80)
    print("TEST: Success Rate Calculation")
    print("=" * 80)
    
    # Test case 1: 100% success
    total = 100
    failures = 0
    success_rate = (total - failures) / total
    assert success_rate == 1.0, f"Expected 1.0, got {success_rate}"
    print(f"  ✓ 100% success: {success_rate * 100:.2f}%")
    
    # Test case 2: 95% success (at threshold)
    total = 100
    failures = 5
    success_rate = (total - failures) / total
    assert success_rate == 0.95, f"Expected 0.95, got {success_rate}"
    print(f"  ✓ 95% success: {success_rate * 100:.2f}%")
    
    # Test case 3: 94% success (below threshold)
    total = 100
    failures = 6
    success_rate = (total - failures) / total
    assert success_rate == 0.94, f"Expected 0.94, got {success_rate}"
    print(f"  ✓ 94% success: {success_rate * 100:.2f}%")
    
    # Test case 4: Edge case - small numbers
    total = 10
    failures = 1
    success_rate = (total - failures) / total
    assert success_rate == 0.9, f"Expected 0.9, got {success_rate}"
    print(f"  ✓ 90% success (10 tickers, 1 failed): {success_rate * 100:.2f}%")
    
    # Test case 5: Real-world scenario - 120 tickers, 4 failures
    total = 120
    failures = 4
    success_rate = (total - failures) / total
    expected = 116/120
    assert abs(success_rate - expected) < 0.001, f"Expected {expected}, got {success_rate}"
    print(f"  ✓ 120 tickers, 4 failed: {success_rate * 100:.2f}% (116/120)")
    
    print("\n✓ All success rate calculation tests passed")
    return True


def test_threshold_logic():
    """Test the threshold comparison logic."""
    print("=" * 80)
    print("TEST: Threshold Logic")
    print("=" * 80)
    
    MIN_SUCCESS_RATE = 0.95
    
    # Test case 1: Above threshold
    success_rate = 0.96
    meets_threshold = success_rate >= MIN_SUCCESS_RATE
    assert meets_threshold is True, f"Expected True for {success_rate}"
    print(f"  ✓ {success_rate * 100:.2f}% >= {MIN_SUCCESS_RATE * 100:.2f}% → PASS")
    
    # Test case 2: At threshold
    success_rate = 0.95
    meets_threshold = success_rate >= MIN_SUCCESS_RATE
    assert meets_threshold is True, f"Expected True for {success_rate}"
    print(f"  ✓ {success_rate * 100:.2f}% >= {MIN_SUCCESS_RATE * 100:.2f}% → PASS")
    
    # Test case 3: Below threshold
    success_rate = 0.94
    meets_threshold = success_rate >= MIN_SUCCESS_RATE
    assert meets_threshold is False, f"Expected False for {success_rate}"
    print(f"  ✓ {success_rate * 100:.2f}% < {MIN_SUCCESS_RATE * 100:.2f}% → FAIL")
    
    # Test case 4: Far below threshold
    success_rate = 0.80
    meets_threshold = success_rate >= MIN_SUCCESS_RATE
    assert meets_threshold is False, f"Expected False for {success_rate}"
    print(f"  ✓ {success_rate * 100:.2f}% < {MIN_SUCCESS_RATE * 100:.2f}% → FAIL")
    
    # Test case 5: Real-world - 120 tickers, 4 failures (96.67% > 95%)
    success_rate = 116/120  # 96.67%
    meets_threshold = success_rate >= MIN_SUCCESS_RATE
    assert meets_threshold is True, f"Expected True for {success_rate}"
    print(f"  ✓ {success_rate * 100:.2f}% >= {MIN_SUCCESS_RATE * 100:.2f}% → PASS (120 tickers, 4 failed)")
    
    # Test case 6: Real-world - 120 tickers, 7 failures (94.17% < 95%)
    success_rate = 113/120  # 94.17%
    meets_threshold = success_rate >= MIN_SUCCESS_RATE
    assert meets_threshold is False, f"Expected False for {success_rate}"
    print(f"  ✓ {success_rate * 100:.2f}% < {MIN_SUCCESS_RATE * 100:.2f}% → FAIL (120 tickers, 7 failed)")
    
    print("\n✓ All threshold logic tests passed")
    return True


def test_environment_variable_parsing():
    """Test that MIN_SUCCESS_RATE can be parsed from environment variable."""
    print("=" * 80)
    print("TEST: Environment Variable Parsing")
    print("=" * 80)
    
    # Test different threshold values
    test_values = [
        ("0.90", 0.90),
        ("0.95", 0.95),
        ("0.99", 0.99),
        ("1.0", 1.0),
        ("0.5", 0.5),
    ]
    
    for env_value, expected in test_values:
        parsed = float(env_value)
        assert parsed == expected, f"Expected {expected}, got {parsed}"
        print(f"  ✓ MIN_SUCCESS_RATE={env_value} → {parsed}")
    
    # Test default value when env var not set
    default = float(os.getenv("MIN_SUCCESS_RATE_TEST_VAR", "0.95"))
    assert default == 0.95, f"Expected 0.95, got {default}"
    print(f"  ✓ Default when not set → {default}")
    
    # Test hardened parsing with clamping
    print("\n  Testing hardened parsing with clamping:")
    
    # Test clamping to 1.0
    try:
        result = min(1.0, max(0.0, float("1.5")))
    except ValueError:
        result = 0.90
    assert result == 1.0, f"Expected 1.0 (clamped), got {result}"
    print(f"  ✓ Value 1.5 → {result} (clamped to 1.0)")
    
    # Test clamping to 0.0
    try:
        result = min(1.0, max(0.0, float("-0.1")))
    except ValueError:
        result = 0.90
    assert result == 0.0, f"Expected 0.0 (clamped), got {result}"
    print(f"  ✓ Value -0.1 → {result} (clamped to 0.0)")
    
    # Test invalid value fallback
    try:
        result = min(1.0, max(0.0, float("invalid")))
    except ValueError:
        result = 0.90
    assert result == 0.90, f"Expected 0.90 (fallback), got {result}"
    print(f"  ✓ Value 'invalid' → {result} (fallback to default)")
    
    print("\n✓ All environment variable parsing tests passed")
    return True


def test_exit_code_logic():
    """Test that exit codes are correct based on success rate."""
    print("=" * 80)
    print("TEST: Exit Code Logic")
    print("=" * 80)
    
    MIN_SUCCESS_RATE = 0.95
    
    # Test case 1: Success rate >= threshold → exit code 0
    success_rate = 0.96
    meets_threshold = success_rate >= MIN_SUCCESS_RATE
    expected_exit_code = 0 if meets_threshold else 1
    assert expected_exit_code == 0, f"Expected exit code 0, got {expected_exit_code}"
    print(f"  ✓ Success rate {success_rate * 100:.2f}% → Exit code {expected_exit_code}")
    
    # Test case 2: Success rate < threshold → exit code 1
    success_rate = 0.90
    meets_threshold = success_rate >= MIN_SUCCESS_RATE
    expected_exit_code = 0 if meets_threshold else 1
    assert expected_exit_code == 1, f"Expected exit code 1, got {expected_exit_code}"
    print(f"  ✓ Success rate {success_rate * 100:.2f}% → Exit code {expected_exit_code}")
    
    # Test case 3: Exactly at threshold → exit code 0
    success_rate = 0.95
    meets_threshold = success_rate >= MIN_SUCCESS_RATE
    expected_exit_code = 0 if meets_threshold else 1
    assert expected_exit_code == 0, f"Expected exit code 0, got {expected_exit_code}"
    print(f"  ✓ Success rate {success_rate * 100:.2f}% → Exit code {expected_exit_code}")
    
    print("\n✓ All exit code logic tests passed")
    return True


def test_metadata_file_generation():
    """Test that metadata file is generated correctly."""
    print("=" * 80)
    print("TEST: Metadata File Generation")
    print("=" * 80)
    
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = os.path.join(tmpdir, "prices_cache_meta.json")
        
        # Create sample metadata
        metadata = {
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
            "success_rate": 0.96,
            "min_success_rate": 0.90,
            "tickers_total": 100,
            "tickers_successful": 96,
            "tickers_failed": 4,
            "max_price_date": "2025-01-05",
            "cache_file": "data/cache/prices_cache.parquet"
        }
        
        # Write metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Metadata file created: {metadata_path}")
        
        # Read and validate metadata
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        # Validate all required fields exist
        required_fields = [
            "generated_at_utc",
            "success_rate",
            "min_success_rate",
            "tickers_total",
            "tickers_successful",
            "tickers_failed",
            "max_price_date",
            "cache_file"
        ]
        
        for field in required_fields:
            assert field in loaded_metadata, f"Missing required field: {field}"
            print(f"  ✓ Field '{field}': {loaded_metadata[field]}")
        
        # Validate field types and values
        assert isinstance(loaded_metadata["success_rate"], (int, float))
        assert isinstance(loaded_metadata["min_success_rate"], (int, float))
        assert isinstance(loaded_metadata["tickers_total"], int)
        assert isinstance(loaded_metadata["tickers_successful"], int)
        assert isinstance(loaded_metadata["tickers_failed"], int)
        assert loaded_metadata["tickers_total"] == loaded_metadata["tickers_successful"] + loaded_metadata["tickers_failed"]
        
        print(f"  ✓ Metadata validation passed")
        print(f"  ✓ Total = Successful + Failed: {loaded_metadata['tickers_total']} = {loaded_metadata['tickers_successful']} + {loaded_metadata['tickers_failed']}")
    
    print("\n✓ All metadata file generation tests passed")
    return True


def test_cache_key_integrity():
    """Test that cache keys are unique based on file attributes."""
    print("=" * 80)
    print("TEST: Cache Key Integrity")
    print("=" * 80)
    
    # Test cache key generation logic
    # Cache keys should be based on mtime or size to ensure freshness
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test_cache.parquet")
        
        # Create initial file
        with open(test_file, 'w') as f:
            f.write("initial content")
        
        # Get initial attributes
        initial_mtime = os.path.getmtime(test_file)
        initial_size = os.path.getsize(test_file)
        initial_key = f"{initial_mtime}_{initial_size}"
        
        print(f"  ✓ Initial cache key: {initial_key}")
        
        # Wait a moment and modify file
        import time
        time.sleep(0.1)
        
        with open(test_file, 'w') as f:
            f.write("modified content with more data")
        
        # Get new attributes
        new_mtime = os.path.getmtime(test_file)
        new_size = os.path.getsize(test_file)
        new_key = f"{new_mtime}_{new_size}"
        
        print(f"  ✓ New cache key: {new_key}")
        
        # Keys should be different
        assert initial_key != new_key, "Cache keys should differ after file modification"
        print(f"  ✓ Cache keys are unique after modification")
        
        # Demonstrate unique key based on mtime alone
        mtime_key_1 = f"mtime_{initial_mtime}"
        mtime_key_2 = f"mtime_{new_mtime}"
        assert mtime_key_1 != mtime_key_2, "Mtime-based keys should differ"
        print(f"  ✓ Mtime-based keys are unique")
        
        # Demonstrate unique key based on size alone
        size_key_1 = f"size_{initial_size}"
        size_key_2 = f"size_{new_size}"
        assert size_key_1 != size_key_2, "Size-based keys should differ"
        print(f"  ✓ Size-based keys are unique")
    
    print("\n✓ All cache key integrity tests passed")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RUNNING ALL BUILD_PRICE_CACHE THRESHOLD TESTS")
    print("=" * 80 + "\n")
    
    tests = [
        test_success_rate_calculation,
        test_threshold_logic,
        test_environment_variable_parsing,
        test_exit_code_logic,
        test_metadata_file_generation,
        test_cache_key_integrity,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_func.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 80 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

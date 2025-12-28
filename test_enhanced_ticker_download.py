"""
Test script for enhanced ticker download functionality with retry logic and diagnostics
"""

import os
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Test the helper functions
def test_normalize_ticker():
    """Test ticker normalization"""
    print("Testing ticker normalization...")
    
    # Import the function
    from waves_engine import _normalize_ticker
    
    test_cases = [
        ("BRK.B", "BRK-B"),
        ("brk.b", "BRK-B"),
        ("AAPL", "AAPL"),
        ("  MSFT  ", "MSFT"),
        ("BF.B", "BF-B"),
    ]
    
    for original, expected in test_cases:
        result = _normalize_ticker(original)
        assert result == expected, f"Expected {expected}, got {result}"
        print(f"  {original:15s} -> {result:15s} ✓")
    
    print("✓ Ticker normalization tests passed\n")


def test_retry_logic():
    """Test retry with backoff"""
    print("Testing retry logic...")
    
    from waves_engine import _retry_with_backoff
    
    # Test successful function
    call_count = [0]
    def successful_func():
        call_count[0] += 1
        return "success"
    
    result = _retry_with_backoff(successful_func, max_retries=3, initial_delay=0.1)
    assert result == "success", "Expected 'success'"
    assert call_count[0] == 1, f"Expected 1 call, got {call_count[0]}"
    print(f"  Successful function called {call_count[0]} time(s) ✓")
    
    # Test function that succeeds on second try
    call_count = [0]
    def retry_once_func():
        call_count[0] += 1
        if call_count[0] < 2:
            raise Exception("Temporary failure")
        return "success after retry"
    
    result = _retry_with_backoff(retry_once_func, max_retries=3, initial_delay=0.1)
    assert result == "success after retry", "Expected 'success after retry'"
    assert call_count[0] == 2, f"Expected 2 calls, got {call_count[0]}"
    print(f"  Retry function called {call_count[0]} time(s) ✓")
    
    # Test function that always fails
    call_count = [0]
    def always_fail():
        call_count[0] += 1
        raise Exception("Permanent failure")
    
    try:
        _retry_with_backoff(always_fail, max_retries=3, initial_delay=0.1)
        assert False, "Should have raised exception"
    except Exception as e:
        assert str(e) == "Permanent failure"
        assert call_count[0] == 3, f"Expected 3 calls, got {call_count[0]}"
        print(f"  Failing function called {call_count[0]} time(s) ✓")
    
    print("✓ Retry logic tests passed\n")


def test_diagnostics_logging():
    """Test JSON diagnostics logging"""
    print("Testing diagnostics logging...")
    
    from waves_engine import _log_diagnostics_to_json
    
    # Clean up any existing test logs
    test_log_dir = "logs/diagnostics"
    if os.path.exists(test_log_dir):
        shutil.rmtree(test_log_dir)
    
    # Test logging
    failures = {
        "INVALID": "Empty data returned",
        "BRK.B": "No Close column",
        "TEST": "Download error: rate limit exceeded"
    }
    
    _log_diagnostics_to_json(failures, wave_id="test_wave", wave_name="Test Wave")
    
    # Verify log file was created
    date_str = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(test_log_dir, f"failed_tickers_{date_str}.json")
    assert os.path.exists(log_file), f"Log file should exist at {log_file}"
    print(f"  Log file created: {log_file} ✓")
    
    # Verify log content
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    assert len(log_data) == 1, "Should have one log entry"
    entry = log_data[0]
    assert entry['wave_id'] == "test_wave", "Wave ID should match"
    assert entry['wave_name'] == "Test Wave", "Wave name should match"
    assert len(entry['failures']) == 3, "Should have 3 failures"
    print(f"  Log entry contains {len(entry['failures'])} failures ✓")
    
    # Verify failure details
    failure_tickers = [f['ticker_original'] for f in entry['failures']]
    assert set(failure_tickers) == set(failures.keys()), "Should log all failed tickers"
    print(f"  All failed tickers logged ✓")
    
    # Test appending to existing log
    more_failures = {"ANOTHER": "Network timeout"}
    _log_diagnostics_to_json(more_failures, wave_id="test_wave_2", wave_name="Test Wave 2")
    
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    assert len(log_data) == 2, "Should have two log entries now"
    print(f"  Log appending works ✓")
    
    # Cleanup
    shutil.rmtree(test_log_dir)
    
    print("✓ Diagnostics logging tests passed\n")


def test_download_history_signature():
    """Test that _download_history has the correct signature"""
    print("Testing _download_history signature...")
    
    import inspect
    from waves_engine import _download_history, _download_history_individually
    
    # Check _download_history signature
    sig = inspect.signature(_download_history)
    params = list(sig.parameters.keys())
    assert 'tickers' in params, "Should have 'tickers' parameter"
    assert 'days' in params, "Should have 'days' parameter"
    assert 'wave_id' in params, "Should have 'wave_id' parameter"
    assert 'wave_name' in params, "Should have 'wave_name' parameter"
    print(f"  _download_history has correct parameters: {params} ✓")
    
    # Check _download_history_individually signature
    sig = inspect.signature(_download_history_individually)
    params = list(sig.parameters.keys())
    assert 'tickers' in params, "Should have 'tickers' parameter"
    assert 'start' in params, "Should have 'start' parameter"
    assert 'end' in params, "Should have 'end' parameter"
    assert 'wave_id' in params, "Should have 'wave_id' parameter"
    assert 'wave_name' in params, "Should have 'wave_name' parameter"
    print(f"  _download_history_individually has correct parameters: {params} ✓")
    
    print("✓ Signature tests passed\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced Ticker Download Tests")
    print("=" * 70 + "\n")
    
    test_normalize_ticker()
    test_retry_logic()
    test_diagnostics_logging()
    test_download_history_signature()
    
    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)

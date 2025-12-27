"""
Test script for ticker diagnostics module
"""

from helpers.ticker_diagnostics import (
    FailedTickerReport, 
    FailureType, 
    categorize_error,
    get_diagnostics_tracker
)
from datetime import datetime


def test_error_categorization():
    """Test error categorization logic"""
    print("Testing error categorization...")
    
    test_cases = [
        ("rate limit exceeded", "AAPL", FailureType.RATE_LIMIT),
        ("429 too many requests", "TSLA", FailureType.RATE_LIMIT),
        ("connection timeout", "MSFT", FailureType.NETWORK_TIMEOUT),
        ("empty data returned", "BRK.B", FailureType.SYMBOL_NEEDS_NORMALIZATION),
        ("no data returned", "GOOGL", FailureType.PROVIDER_EMPTY),
        ("insufficient history", "NVDA", FailureType.INSUFFICIENT_HISTORY),
    ]
    
    for error_msg, ticker, expected_type in test_cases:
        failure_type, suggestion = categorize_error(error_msg, ticker)
        print(f"  {error_msg[:30]:30s} -> {failure_type.value:30s} ✓")
        assert failure_type == expected_type, f"Expected {expected_type}, got {failure_type}"
    
    print("✓ Error categorization tests passed\n")


def test_tracker():
    """Test diagnostics tracker"""
    print("Testing diagnostics tracker...")
    
    tracker = get_diagnostics_tracker()
    tracker.clear()
    
    # Add some test failures
    report1 = FailedTickerReport(
        ticker_original="BRK.B",
        ticker_normalized="BRK-B",
        wave_id="sp500_wave",
        wave_name="S&P 500 Wave",
        failure_type=FailureType.SYMBOL_NEEDS_NORMALIZATION,
        error_message="Empty data returned",
        suggested_fix="Normalize ticker: BRK.B -> BRK-B"
    )
    tracker.record_failure(report1)
    
    report2 = FailedTickerReport(
        ticker_original="INVALID",
        ticker_normalized="INVALID",
        wave_id="growth_wave",
        wave_name="Growth Wave",
        failure_type=FailureType.PROVIDER_EMPTY,
        error_message="No data returned - ticker may be delisted",
        suggested_fix="Verify ticker validity"
    )
    tracker.record_failure(report2)
    
    # Test statistics
    stats = tracker.get_summary_stats()
    print(f"  Total failures: {stats['total_failures']}")
    print(f"  Unique tickers: {stats['unique_tickers']}")
    print(f"  Fatal count: {stats['fatal_count']}")
    print(f"  By type: {stats['by_type']}")
    
    assert stats['total_failures'] == 2, "Expected 2 failures"
    assert stats['unique_tickers'] == 2, "Expected 2 unique tickers"
    
    # Test CSV export
    csv_path = tracker.export_to_csv("test_report.csv")
    print(f"  CSV exported to: {csv_path}")
    
    # Verify CSV exists and has content
    import os
    assert os.path.exists(csv_path), "CSV file should exist"
    
    with open(csv_path, 'r') as f:
        content = f.read()
        assert 'BRK.B' in content, "CSV should contain BRK.B"
        assert 'INVALID' in content, "CSV should contain INVALID"
    
    print("✓ Tracker tests passed\n")
    
    # Cleanup
    tracker.clear()


def test_failure_report():
    """Test FailedTickerReport dataclass"""
    print("Testing FailedTickerReport...")
    
    report = FailedTickerReport(
        ticker_original="AAPL",
        ticker_normalized="AAPL",
        wave_id="sp500_wave",
        wave_name="S&P 500 Wave",
        failure_type=FailureType.RATE_LIMIT,
        error_message="Rate limit exceeded",
        suggested_fix="Wait and retry with exponential backoff"
    )
    
    # Test to_dict conversion
    report_dict = report.to_dict()
    assert report_dict['ticker_original'] == "AAPL"
    assert report_dict['failure_type'] == "RATE_LIMIT"
    assert report_dict['wave_id'] == "sp500_wave"
    
    print(f"  Report created: {report_dict['ticker_original']} - {report_dict['failure_type']}")
    print("✓ FailedTickerReport tests passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Ticker Diagnostics Module Tests")
    print("=" * 60 + "\n")
    
    test_error_categorization()
    test_failure_report()
    test_tracker()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

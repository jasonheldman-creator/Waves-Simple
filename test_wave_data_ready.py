"""
Test suite for Wave Data Ready pipeline.

Tests the compute_data_ready_status function and related diagnostics.
"""

import os
import sys
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics_pipeline import compute_data_ready_status
from waves_engine import get_all_wave_ids


def test_compute_data_ready_status_all_waves():
    """Test that compute_data_ready_status works for all 28 waves."""
    print("=" * 80)
    print("TEST: compute_data_ready_status for all 28 waves")
    print("=" * 80)
    
    wave_ids = get_all_wave_ids()
    assert len(wave_ids) == 28, f"Expected 28 waves, got {len(wave_ids)}"
    print(f"✓ Found all 28 waves in registry")
    
    # Track results
    ready_waves = []
    missing_prices = []
    missing_benchmark = []
    missing_nav = []
    stale_data = []
    insufficient_history = []
    other_issues = []
    
    for wave_id in wave_ids:
        result = compute_data_ready_status(wave_id)
        
        # Validate result structure (enhanced for new fields)
        assert 'wave_id' in result, f"Missing wave_id in result for {wave_id}"
        assert 'display_name' in result, f"Missing display_name in result for {wave_id}"
        assert 'is_ready' in result, f"Missing is_ready in result for {wave_id}"
        assert 'reason' in result, f"Missing reason in result for {wave_id}"
        assert 'reason_codes' in result, f"Missing reason_codes in result for {wave_id}"
        assert 'details' in result, f"Missing details in result for {wave_id}"
        assert 'checks' in result, f"Missing checks in result for {wave_id}"
        assert 'missing_tickers' in result, f"Missing missing_tickers in result for {wave_id}"
        assert 'missing_benchmark_tickers' in result, f"Missing missing_benchmark_tickers in result for {wave_id}"
        assert 'missing_dates' in result, f"Missing missing_dates in result for {wave_id}"
        assert 'history_window_used' in result, f"Missing history_window_used in result for {wave_id}"
        assert 'source_used' in result, f"Missing source_used in result for {wave_id}"
        assert 'exception' in result, f"Missing exception in result for {wave_id}"
        
        # Validate types
        assert isinstance(result['reason_codes'], list), f"reason_codes should be a list for {wave_id}"
        assert isinstance(result['missing_tickers'], list), f"missing_tickers should be a list for {wave_id}"
        assert isinstance(result['missing_benchmark_tickers'], list), f"missing_benchmark_tickers should be a list for {wave_id}"
        assert isinstance(result['missing_dates'], dict), f"missing_dates should be a dict for {wave_id}"
        assert isinstance(result['history_window_used'], dict), f"history_window_used should be a dict for {wave_id}"
        
        # Categorize results
        if result['is_ready']:
            ready_waves.append(wave_id)
        else:
            reason = result['reason']
            if reason == 'MISSING_PRICES' or 'MISSING_PRICE' in result['reason_codes']:
                missing_prices.append(wave_id)
            elif reason == 'MISSING_BENCHMARK':
                missing_benchmark.append(wave_id)
            elif reason == 'MISSING_NAV':
                missing_nav.append(wave_id)
            elif reason == 'STALE_DATA':
                stale_data.append(wave_id)
            elif reason == 'INSUFFICIENT_HISTORY':
                insufficient_history.append(wave_id)
            else:
                other_issues.append((wave_id, reason))
    
    # Print summary
    print(f"\n✓ Successfully tested all {len(wave_ids)} waves")
    print("\nREADINESS SUMMARY:")
    print(f"  Ready: {len(ready_waves)} waves")
    print(f"  Missing Prices: {len(missing_prices)} waves")
    print(f"  Missing Benchmark: {len(missing_benchmark)} waves")
    print(f"  Missing NAV: {len(missing_nav)} waves")
    print(f"  Stale Data: {len(stale_data)} waves")
    print(f"  Insufficient History: {len(insufficient_history)} waves")
    print(f"  Other Issues: {len(other_issues)} waves")
    
    # Show ready waves
    if ready_waves:
        print(f"\nREADY WAVES ({len(ready_waves)}):")
        for wave_id in ready_waves:
            result = compute_data_ready_status(wave_id)
            print(f"  ✓ {wave_id}: {result['details']}")
    
    # Show sample of not-ready waves with enhanced diagnostics
    if missing_prices:
        print(f"\nSAMPLE MISSING PRICES ({min(3, len(missing_prices))} of {len(missing_prices)}):")
        for wave_id in missing_prices[:3]:
            result = compute_data_ready_status(wave_id)
            print(f"  ✗ {wave_id}: {result['details']}")
            print(f"     Reason codes: {result['reason_codes']}")
            if result['missing_tickers']:
                print(f"     Missing tickers: {result['missing_tickers'][:5]}{'...' if len(result['missing_tickers']) > 5 else ''}")
            print(f"     Source: {result['source_used']}")
    
    print("\n✓ All tests passed!")
    return True


def test_reason_codes():
    """Test that reason codes are consistent and meaningful."""
    print("\n" + "=" * 80)
    print("TEST: Reason code validation")
    print("=" * 80)
    
    expected_reason_codes = [
        'READY',
        'MISSING_WEIGHTS',
        'MISSING_PRICES',
        'MISSING_PRICE',  # Individual ticker missing
        'MISSING_BENCHMARK',
        'MISSING_NAV',
        'STALE_DATA',
        'INSUFFICIENT_HISTORY',
        'WAVE_NOT_FOUND',
        'DATA_READ_ERROR',
        'UNSUPPORTED_TICKER',
        'DELISTED_TICKER',
        'API_FAILURE',
        'NAN_SERIES'
    ]
    
    wave_ids = get_all_wave_ids()
    observed_reasons = set()
    
    for wave_id in wave_ids:
        result = compute_data_ready_status(wave_id)
        observed_reasons.add(result['reason'])
        # Also check reason_codes list
        for code in result['reason_codes']:
            observed_reasons.add(code)
    
    print(f"Expected reason codes: {sorted(expected_reason_codes)}")
    print(f"Observed reason codes: {sorted(observed_reasons)}")
    
    # All observed reasons should be in expected list
    unexpected = observed_reasons - set(expected_reason_codes)
    if unexpected:
        print(f"⚠ Warning: Unexpected reason codes: {unexpected}")
    else:
        print("✓ All reason codes are valid")
    
    return True


def test_checks_structure():
    """Test that checks dictionary has expected structure."""
    print("\n" + "=" * 80)
    print("TEST: Checks structure validation")
    print("=" * 80)
    
    expected_checks = [
        'has_weights',
        'has_prices',
        'has_benchmark',
        'has_nav',
        'is_fresh',
        'has_sufficient_history'
    ]
    
    wave_ids = get_all_wave_ids()
    
    for wave_id in wave_ids[:3]:  # Test first 3 waves
        result = compute_data_ready_status(wave_id)
        checks = result['checks']
        
        for check_name in expected_checks:
            assert check_name in checks, f"Missing check '{check_name}' for {wave_id}"
            assert isinstance(checks[check_name], bool), f"Check '{check_name}' should be bool for {wave_id}"
    
    print(f"✓ Checks structure is valid for all tested waves")
    return True


def test_ready_waves_have_all_checks():
    """Test that ready waves have all checks passed."""
    print("\n" + "=" * 80)
    print("TEST: Ready waves validation")
    print("=" * 80)
    
    wave_ids = get_all_wave_ids()
    ready_waves_count = 0
    
    for wave_id in wave_ids:
        result = compute_data_ready_status(wave_id)
        
        if result['is_ready']:
            ready_waves_count += 1
            checks = result['checks']
            
            # All checks should be True for ready waves
            assert checks['has_weights'], f"Ready wave {wave_id} missing weights"
            assert checks['has_prices'], f"Ready wave {wave_id} missing prices"
            assert checks['has_benchmark'], f"Ready wave {wave_id} missing benchmark"
            assert checks['has_nav'], f"Ready wave {wave_id} missing NAV"
            assert checks['is_fresh'], f"Ready wave {wave_id} has stale data"
            assert checks['has_sufficient_history'], f"Ready wave {wave_id} has insufficient history"
    
    print(f"✓ All {ready_waves_count} ready waves have all checks passed")
    return True


def test_diagnostic_report_generation():
    """Test that diagnostic report generation functions work."""
    print("\n" + "=" * 80)
    print("TEST: Diagnostic report generation")
    print("=" * 80)
    
    from analytics_pipeline import generate_readiness_report_dataframe, generate_readiness_report_json
    
    # Test DataFrame generation
    df = generate_readiness_report_dataframe()
    assert not df.empty, "DataFrame should not be empty"
    assert 'wave_id' in df.columns, "DataFrame should have wave_id column"
    assert 'is_ready' in df.columns, "DataFrame should have is_ready column"
    assert 'reason_codes' in df.columns, "DataFrame should have reason_codes column"
    assert 'missing_tickers' in df.columns, "DataFrame should have missing_tickers column"
    assert 'source_used' in df.columns, "DataFrame should have source_used column"
    
    print(f"✓ Generated DataFrame with {len(df)} rows")
    print(f"  Columns: {', '.join(df.columns)}")
    
    # Test JSON generation
    json_report = generate_readiness_report_json()
    assert 'summary' in json_report, "JSON report should have summary"
    assert 'waves' in json_report, "JSON report should have waves"
    assert 'total_waves' in json_report['summary'], "Summary should have total_waves"
    assert 'ready_count' in json_report['summary'], "Summary should have ready_count"
    
    print(f"✓ Generated JSON report")
    print(f"  Total waves: {json_report['summary']['total_waves']}")
    print(f"  Ready: {json_report['summary']['ready_count']}")
    print(f"  Degraded: {json_report['summary']['degraded_count']}")
    print(f"  Missing: {json_report['summary']['missing_count']}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("WAVE DATA READY PIPELINE TEST SUITE")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Run all tests
        test_compute_data_ready_status_all_waves()
        test_reason_codes()
        test_checks_structure()
        test_ready_waves_have_all_checks()
        test_diagnostic_report_generation()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

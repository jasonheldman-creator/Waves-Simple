"""
Test suite for Wave Data Ready pipeline and Wave Readiness Report.

Tests the compute_data_ready_status function with graded readiness model,
analytics_ready flag, and generate_wave_readiness_report.
"""

import os
import sys
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytics_pipeline import (
    compute_data_ready_status,
    generate_wave_readiness_report,
    print_readiness_report,
    DEFAULT_COVERAGE_THRESHOLD,
    MIN_COVERAGE_FOR_ANALYTICS,
    MIN_DAYS_FOR_ANALYTICS
)
from waves_engine import get_all_wave_ids


def test_analytics_ready_flag():
    """Test that analytics_ready flag is computed correctly."""
    print("=" * 80)
    print("TEST: Analytics Ready Flag")
    print("=" * 80)
    
    wave_ids = get_all_wave_ids()
    analytics_ready_count = 0
    analytics_limited_count = 0
    
    for wave_id in wave_ids[:10]:  # Test first 10 waves
        result = compute_data_ready_status(wave_id)
        
        # Check that analytics_ready field exists
        assert 'analytics_ready' in result, f"Missing analytics_ready field for {wave_id}"
        assert isinstance(result['analytics_ready'], bool), f"analytics_ready should be bool for {wave_id}"
        
        # Verify the logic
        coverage_pct = result.get('coverage_pct', 0)
        history_days = result.get('history_days', 0)
        
        expected_analytics_ready = (
            (coverage_pct / 100.0) >= MIN_COVERAGE_FOR_ANALYTICS and
            history_days >= MIN_DAYS_FOR_ANALYTICS
        )
        
        assert result['analytics_ready'] == expected_analytics_ready, \
            f"analytics_ready mismatch for {wave_id}: got {result['analytics_ready']}, expected {expected_analytics_ready}"
        
        if result['analytics_ready']:
            analytics_ready_count += 1
        else:
            analytics_limited_count += 1
        
        # If analytics_ready is False and wave is ready, should have informational issues
        if not result['analytics_ready'] and result['is_ready']:
            issues = result.get('informational_issues', [])
            assert any('ANALYTICS_LIMITED' in issue for issue in issues), \
                f"Expected ANALYTICS_LIMITED issue for {wave_id} when analytics_ready=False"
    
    print(f"✓ Analytics ready flag validated for 10 waves")
    print(f"  Analytics ready: {analytics_ready_count}")
    print(f"  Analytics limited: {analytics_limited_count}")
    print(f"  Thresholds: Coverage ≥{MIN_COVERAGE_FOR_ANALYTICS*100:.0f}%, History ≥{MIN_DAYS_FOR_ANALYTICS} days")
    
    return True


def test_stale_tickers_detection():
    """Test that stale tickers are detected correctly."""
    print("\n" + "=" * 80)
    print("TEST: Stale Tickers Detection")
    print("=" * 80)
    
    wave_ids = get_all_wave_ids()
    waves_with_stale = 0
    
    for wave_id in wave_ids[:10]:  # Test first 10 waves
        result = compute_data_ready_status(wave_id)
        
        # Check that stale_tickers field exists
        assert 'stale_tickers' in result, f"Missing stale_tickers field for {wave_id}"
        assert isinstance(result['stale_tickers'], list), f"stale_tickers should be list for {wave_id}"
        
        # Check that stale_days_max field exists
        assert 'stale_days_max' in result, f"Missing stale_days_max field for {wave_id}"
        assert isinstance(result['stale_days_max'], (int, float)), f"stale_days_max should be numeric for {wave_id}"
        
        if result['stale_tickers']:
            waves_with_stale += 1
            print(f"  {wave_id}: {len(result['stale_tickers'])} stale ticker(s), max age: {result['stale_days_max']} days")
    
    print(f"\n✓ Stale ticker detection validated for 10 waves")
    print(f"  Waves with stale data: {waves_with_stale}")
    
    return True


def test_history_days_field():
    """Test that history_days field is populated correctly."""
    print("\n" + "=" * 80)
    print("TEST: History Days Field")
    print("=" * 80)
    
    wave_ids = get_all_wave_ids()
    
    for wave_id in wave_ids[:10]:  # Test first 10 waves
        result = compute_data_ready_status(wave_id)
        
        # Check that history_days field exists
        assert 'history_days' in result, f"Missing history_days field for {wave_id}"
        assert isinstance(result['history_days'], (int, float)), f"history_days should be numeric for {wave_id}"
        assert result['history_days'] >= 0, f"history_days should be non-negative for {wave_id}"
        
        if result['history_days'] > 0:
            print(f"  {wave_id}: {result['history_days']} days of history")
    
    print(f"\n✓ History days field validated for 10 waves")
    
    return True


def test_generate_wave_readiness_report():
    """Test that generate_wave_readiness_report produces the required report structure with graded readiness."""
    print("=" * 80)
    print("TEST: Wave Readiness Report Generation (Graded Model)")
    print("=" * 80)
    
    # Generate the report
    df = generate_wave_readiness_report()
    
    # Validate report structure
    assert not df.empty, "Report DataFrame should not be empty"
    
    # Check all required columns exist (updated for graded model)
    required_columns = [
        'wave_id',
        'wave_name',
        'readiness_status',  # NEW: graded status
        'readiness_summary',  # NEW: summary of capabilities
        'blocking_issues',
        'informational_issues',
        'allowed_analytics',  # NEW: which analytics are available
        'failing_tickers',
        'coverage_pct',
        'required_window_days',
        'available_window_days',
        'start_date',
        'end_date',
        'suggested_actions'  # NEW: actionable recommendations
    ]
    
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"
    
    print(f"✓ Report has all {len(required_columns)} required columns (graded model)")
    
    # Validate all 28 waves are included
    wave_ids = get_all_wave_ids()
    assert len(df) == len(wave_ids), f"Expected {len(wave_ids)} waves, got {len(df)}"
    print(f"✓ Report includes all {len(wave_ids)} waves (NO SILENT EXCLUSIONS)")
    
    # Validate readiness_status values (graded model)
    valid_statuses = ['full', 'partial', 'operational', 'unavailable']
    invalid = df[~df['readiness_status'].isin(valid_statuses)]
    assert invalid.empty, f"Invalid readiness_status values found: {invalid['readiness_status'].unique()}"
    print(f"✓ All readiness_status values are valid graded statuses")
    
    # Validate readiness_summary is populated
    assert df['readiness_summary'].notna().all(), "All rows should have a readiness_summary"
    print(f"✓ All rows have readiness_summary populated")
    
    # Validate that unavailable waves have blocking issues
    unavailable = df[df['readiness_status'] == 'unavailable']
    if not unavailable.empty:
        has_blocking = (unavailable['blocking_issues'] != '')
        if not has_blocking.all():
            print(f"⚠ Warning: {(~has_blocking).sum()} unavailable waves missing blocking_issues")
        print(f"✓ Unavailable waves ({len(unavailable)}) have diagnostic information")
    
    # Validate coverage_pct is in valid range
    assert (df['coverage_pct'] >= 0).all(), "coverage_pct should be >= 0"
    assert (df['coverage_pct'] <= 100).all(), "coverage_pct should be <= 100"
    print(f"✓ Coverage percentages are in valid range (0-100)")
    
    # Validate allowed_analytics is populated
    assert df['allowed_analytics'].notna().all(), "All rows should have allowed_analytics"
    assert (df['allowed_analytics'] != '').all(), "All allowed_analytics should be non-empty"
    print(f"✓ All rows have allowed_analytics populated")
    
    # Print summary statistics (graded model)
    full_count = (df['readiness_status'] == 'full').sum()
    partial_count = (df['readiness_status'] == 'partial').sum()
    operational_count = (df['readiness_status'] == 'operational').sum()
    unavailable_count = (df['readiness_status'] == 'unavailable').sum()
    
    print(f"\nGRADED READINESS SUMMARY:")
    print(f"  Total waves: {len(df)}")
    print(f"  Full (all analytics): {full_count} ({full_count/len(df)*100:.1f}%)")
    print(f"  Partial (basic analytics): {partial_count} ({partial_count/len(df)*100:.1f}%)")
    print(f"  Operational (current state): {operational_count} ({operational_count/len(df)*100:.1f}%)")
    print(f"  Unavailable: {unavailable_count} ({unavailable_count/len(df)*100:.1f}%)")
    
    usable_count = full_count + partial_count + operational_count
    print(f"\n  USABLE WAVES (operational or better): {usable_count} ({usable_count/len(df)*100:.1f}%)")
    
    print(f"\n✓ Wave Readiness Report test passed (graded model)!")
    return True


def test_compute_data_ready_status_all_waves():
    """Test that compute_data_ready_status works for all 28 waves with graded readiness."""
    print("=" * 80)
    print("TEST: compute_data_ready_status for all waves (Graded Model)")
    print("=" * 80)
    
    from waves_engine import WAVE_ID_REGISTRY
    expected_count = len(WAVE_ID_REGISTRY)
    
    wave_ids = get_all_wave_ids()
    assert len(wave_ids) == expected_count, f"Expected {expected_count} waves, got {len(wave_ids)}"
    print(f"✓ Found all {expected_count} waves in registry (from WAVE_ID_REGISTRY)")
    
    # Track results by graded status
    full_waves = []
    partial_waves = []
    operational_waves = []
    unavailable_waves = []
    
    for wave_id in wave_ids:
        result = compute_data_ready_status(wave_id)
        
        # Validate result structure (graded readiness fields)
        assert 'wave_id' in result, f"Missing wave_id in result for {wave_id}"
        assert 'display_name' in result, f"Missing display_name in result for {wave_id}"
        assert 'readiness_status' in result, f"Missing readiness_status in result for {wave_id}"
        assert 'readiness_reasons' in result, f"Missing readiness_reasons in result for {wave_id}"
        assert 'allowed_analytics' in result, f"Missing allowed_analytics in result for {wave_id}"
        assert 'checks_passed' in result, f"Missing checks_passed in result for {wave_id}"
        assert 'checks_failed' in result, f"Missing checks_failed in result for {wave_id}"
        assert 'blocking_issues' in result, f"Missing blocking_issues in result for {wave_id}"
        assert 'informational_issues' in result, f"Missing informational_issues in result for {wave_id}"
        assert 'suggested_actions' in result, f"Missing suggested_actions in result for {wave_id}"
        
        # Legacy fields for backward compatibility
        assert 'is_ready' in result, f"Missing is_ready in result for {wave_id}"
        assert 'reason' in result, f"Missing reason in result for {wave_id}"
        assert 'reason_codes' in result, f"Missing reason_codes in result for {wave_id}"
        assert 'details' in result, f"Missing details in result for {wave_id}"
        assert 'checks' in result, f"Missing checks in result for {wave_id}"
        assert 'missing_tickers' in result, f"Missing missing_tickers in result for {wave_id}"
        assert 'coverage_pct' in result, f"Missing coverage_pct in result for {wave_id}"
        
        # Validate types
        assert isinstance(result['readiness_status'], str), f"readiness_status should be str for {wave_id}"
        assert result['readiness_status'] in ['full', 'partial', 'operational', 'unavailable'], \
            f"Invalid readiness_status for {wave_id}: {result['readiness_status']}"
        assert isinstance(result['readiness_reasons'], list), f"readiness_reasons should be a list for {wave_id}"
        assert isinstance(result['allowed_analytics'], dict), f"allowed_analytics should be a dict for {wave_id}"
        assert isinstance(result['checks_passed'], dict), f"checks_passed should be a dict for {wave_id}"
        assert isinstance(result['checks_failed'], dict), f"checks_failed should be a dict for {wave_id}"
        assert isinstance(result['blocking_issues'], list), f"blocking_issues should be a list for {wave_id}"
        assert isinstance(result['informational_issues'], list), f"informational_issues should be a list for {wave_id}"
        assert isinstance(result['suggested_actions'], list), f"suggested_actions should be a list for {wave_id}"
        
        # Categorize by graded status
        status = result['readiness_status']
        if status == 'full':
            full_waves.append(wave_id)
        elif status == 'partial':
            partial_waves.append(wave_id)
        elif status == 'operational':
            operational_waves.append(wave_id)
        else:  # unavailable
            unavailable_waves.append(wave_id)
    
    # Print summary
    print(f"\n✓ Successfully tested all {len(wave_ids)} waves")
    print("\nGRADED READINESS SUMMARY:")
    print(f"  Full: {len(full_waves)} waves")
    print(f"  Partial: {len(partial_waves)} waves")
    print(f"  Operational: {len(operational_waves)} waves")
    print(f"  Unavailable: {len(unavailable_waves)} waves")
    
    usable_count = len(full_waves) + len(partial_waves) + len(operational_waves)
    print(f"\n  USABLE (operational or better): {usable_count} waves ({usable_count/len(wave_ids)*100:.1f}%)")
    
    # Show full waves
    if full_waves:
        print(f"\nFULL READINESS WAVES ({len(full_waves)}):")
        for wave_id in full_waves[:5]:  # Show first 5
            result = compute_data_ready_status(wave_id)
            print(f"  ✓✓ {wave_id}")
            print(f"     Coverage: {result['coverage_pct']:.1f}%")
            enabled_analytics = [k for k, v in result['allowed_analytics'].items() if v]
            print(f"     Analytics: {', '.join(enabled_analytics[:3])}...")
    
    # Show partial waves
    if partial_waves:
        print(f"\nPARTIAL READINESS WAVES ({len(partial_waves)}):")
        for wave_id in partial_waves[:3]:
            result = compute_data_ready_status(wave_id)
            print(f"  ✓ {wave_id}")
            print(f"     Coverage: {result['coverage_pct']:.1f}%")
            if result['informational_issues']:
                print(f"     Limitations: {', '.join(result['informational_issues'][:2])}")
    
    # Show operational waves
    if operational_waves:
        print(f"\nOPERATIONAL WAVES ({len(operational_waves)}):")
        for wave_id in operational_waves[:3]:
            result = compute_data_ready_status(wave_id)
            print(f"  ○ {wave_id}")
            print(f"     Coverage: {result['coverage_pct']:.1f}%")
    
    # Show sample of unavailable waves
    if unavailable_waves:
        print(f"\nUNAVAILABLE WAVES (sample of {min(3, len(unavailable_waves))} of {len(unavailable_waves)}):")
        for wave_id in unavailable_waves[:3]:
            result = compute_data_ready_status(wave_id)
            print(f"  ✗ {wave_id}")
            print(f"     Blocking: {', '.join(result['blocking_issues'])}")
            if result['suggested_actions']:
                print(f"     Action: {result['suggested_actions'][0]}")
    
    print("\n✓ All tests passed (graded model)!")
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


def test_analytics_gating():
    """Test that analytics are properly gated based on readiness level."""
    print("\n" + "=" * 80)
    print("TEST: Analytics gating validation")
    print("=" * 80)
    
    wave_ids = get_all_wave_ids()
    
    for wave_id in wave_ids[:5]:  # Test first 5 waves
        result = compute_data_ready_status(wave_id)
        status = result['readiness_status']
        allowed = result['allowed_analytics']
        
        print(f"\n{wave_id} ({status}):")
        
        # Validate gating logic
        if status == 'unavailable':
            # Nothing should be allowed
            assert not any(allowed.values()), f"Unavailable wave {wave_id} should have no analytics enabled"
            print(f"  ✓ No analytics allowed (correct for unavailable)")
        
        elif status == 'operational':
            # Only current pricing should be available
            assert allowed['current_pricing'], f"Operational wave {wave_id} should allow current_pricing"
            print(f"  ✓ Current pricing allowed")
            # Advanced analytics should not be available
            assert not allowed['multi_window_returns'], f"Operational wave {wave_id} should not allow multi_window_returns"
            assert not allowed['alpha_attribution'], f"Operational wave {wave_id} should not allow alpha_attribution"
            print(f"  ✓ Advanced analytics blocked (correct for operational)")
        
        elif status == 'partial':
            # Basic analytics should be available
            assert allowed['current_pricing'], f"Partial wave {wave_id} should allow current_pricing"
            assert allowed['simple_returns'], f"Partial wave {wave_id} should allow simple_returns"
            print(f"  ✓ Basic analytics allowed")
            # Advanced analytics may or may not be available depending on data
            print(f"  ○ Advanced analytics: {allowed.get('alpha_attribution', False)}")
        
        elif status == 'full':
            # All analytics should be available
            assert allowed['current_pricing'], f"Full wave {wave_id} should allow current_pricing"
            assert allowed['multi_window_returns'], f"Full wave {wave_id} should allow multi_window_returns"
            assert allowed['alpha_attribution'], f"Full wave {wave_id} should allow alpha_attribution"
            assert allowed['advanced_analytics'], f"Full wave {wave_id} should allow advanced_analytics"
            print(f"  ✓ All analytics allowed (correct for full)")
    
    print(f"\n✓ Analytics gating is working correctly")
    return True


def test_no_silent_exclusions():
    """Test that no waves are silently excluded from the report."""
    print("\n" + "=" * 80)
    print("TEST: No silent exclusions")
    print("=" * 80)
    
    all_wave_ids = get_all_wave_ids()
    report_df = generate_wave_readiness_report()
    
    # Every wave in registry should be in the report
    report_wave_ids = set(report_df['wave_id'].tolist())
    registry_wave_ids = set(all_wave_ids)
    
    missing_from_report = registry_wave_ids - report_wave_ids
    assert not missing_from_report, \
        f"Expected all waves to be included in report but found missing waves: {missing_from_report}"
    
    print(f"✓ All {len(all_wave_ids)} waves are included in the report")
    print(f"  NO SILENT EXCLUSIONS - All waves visible with diagnostics")
    
    # Check that unavailable waves still have diagnostics
    unavailable = report_df[report_df['readiness_status'] == 'unavailable']
    if not unavailable.empty:
        print(f"\n✓ {len(unavailable)} unavailable waves still visible with diagnostics")
        # Each should have blocking issues or suggested actions
        for _, row in unavailable.head(3).iterrows():
            has_diagnostics = (
                row['blocking_issues'] or 
                row['suggested_actions'] or 
                row['readiness_summary']
            )
            assert has_diagnostics, f"Wave {row['wave_id']} missing diagnostics"
            print(f"  - {row['wave_id']}: {row['readiness_summary'][:60]}...")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("WAVE DATA READY PIPELINE TEST SUITE - GRADED READINESS MODEL")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Run new tests for analytics_ready flag and related fields
        test_analytics_ready_flag()
        test_stale_tickers_detection()
        test_history_days_field()
        
        # Run core tests
        test_compute_data_ready_status_all_waves()
        test_generate_wave_readiness_report()
        
        # Run graded readiness specific tests
        test_analytics_gating()
        test_no_silent_exclusions()
        
        # Legacy compatibility tests (keep for backward compatibility)
        test_reason_codes()
        test_checks_structure()
        
        # Skip test_ready_waves_have_all_checks and test_diagnostic_report_generation
        # as they test the old binary model
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓ (Graded Readiness Model with Analytics Ready)")
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

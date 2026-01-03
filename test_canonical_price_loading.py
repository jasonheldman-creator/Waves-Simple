"""
Test suite for Canonical Price Loading Mechanism

Tests the generate_all_wave_prices.py script and its integration with
the analytics pipeline, including SmartSafe exemptions and readiness
diagnostics.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_all_wave_prices import (
    generate_prices_for_wave,
    generate_all_prices,
)
from analytics_pipeline import (
    compute_data_ready_status,
    get_wave_analytics_dir,
)
from waves_engine import (
    is_smartsafe_cash_wave,
    get_all_wave_ids,
)


def test_smartsafe_exemption():
    """Test that SmartSafe cash waves are properly exempted."""
    print("=" * 80)
    print("TEST: SmartSafe Cash Wave Exemption")
    print("=" * 80)
    
    smartsafe_waves = [
        'smartsafe_treasury_cash_wave',
        'smartsafe_tax_free_money_market_wave'
    ]
    
    for wave_id in smartsafe_waves:
        # Verify is_smartsafe_cash_wave function
        assert is_smartsafe_cash_wave(wave_id), \
            f"Wave {wave_id} should be identified as SmartSafe"
        
        # Test generation skips SmartSafe waves
        result = generate_prices_for_wave(wave_id, use_dummy_data=True)
        
        assert result['is_smartsafe'] == True, \
            f"Result should mark {wave_id} as SmartSafe"
        assert result['skipped'] == True, \
            f"Generation should skip {wave_id}"
        assert 'SmartSafe' in result['skip_reason'], \
            f"Skip reason should mention SmartSafe for {wave_id}"
        
        # Test readiness check passes without prices.csv
        status = compute_data_ready_status(wave_id)
        assert status['readiness_status'] == 'full', \
            f"SmartSafe wave {wave_id} should have 'full' readiness"
        assert status['is_ready'] == True, \
            f"SmartSafe wave {wave_id} should be ready"
        assert 'SMARTSAFE_CASH' in status.get('reason_codes', []), \
            f"SmartSafe wave {wave_id} should have SMARTSAFE_CASH reason code"
    
    print(f"✓ SmartSafe exemption working for {len(smartsafe_waves)} waves")
    print(f"  - Generation properly skips SmartSafe waves")
    print(f"  - Readiness checks pass without price files")
    return True


def test_non_smartsafe_requires_prices():
    """Test that non-SmartSafe waves require prices.csv."""
    print("\n" + "=" * 80)
    print("TEST: Non-SmartSafe Waves Require Prices")
    print("=" * 80)
    
    # Test with a known non-SmartSafe wave
    test_wave = 'sp500_wave'
    
    assert not is_smartsafe_cash_wave(test_wave), \
        f"Wave {test_wave} should NOT be SmartSafe"
    
    # Check that readiness requires prices.csv
    status = compute_data_ready_status(test_wave)
    
    # If prices.csv doesn't exist, should be unavailable
    wave_dir = get_wave_analytics_dir(test_wave)
    prices_path = os.path.join(wave_dir, 'prices.csv')
    
    if not os.path.exists(prices_path):
        assert status['readiness_status'] == 'unavailable', \
            f"Wave {test_wave} should be unavailable without prices.csv"
        assert 'MISSING_PRICES' in status.get('reason_codes', []), \
            f"Wave {test_wave} should have MISSING_PRICES reason code"
    else:
        # If prices exist, should be at least operational
        assert status['readiness_status'] in ['operational', 'partial', 'full'], \
            f"Wave {test_wave} with prices.csv should be at least operational"
    
    print(f"✓ Non-SmartSafe wave {test_wave} properly requires prices.csv")
    return True


def test_price_generation_with_dummy_data():
    """Test price generation with dummy data."""
    print("\n" + "=" * 80)
    print("TEST: Price Generation with Dummy Data")
    print("=" * 80)
    
    # Pick a wave that's not SmartSafe
    test_wave = 'gold_wave'
    
    # Generate prices with dummy data
    result = generate_prices_for_wave(
        test_wave,
        lookback_days=14,
        use_dummy_data=True,
        skip_existing=False
    )
    
    # Verify result structure
    assert 'wave_id' in result
    assert 'success' in result
    assert 'files_generated' in result
    assert 'errors' in result
    
    # Should succeed with dummy data
    assert result['success'] == True, \
        f"Dummy data generation should succeed for {test_wave}"
    
    # Should generate prices.csv
    assert 'prices.csv' in result['files_generated'], \
        f"Should generate prices.csv for {test_wave}"
    
    # Verify files exist
    wave_dir = get_wave_analytics_dir(test_wave)
    prices_path = os.path.join(wave_dir, 'prices.csv')
    assert os.path.exists(prices_path), \
        f"prices.csv should exist at {prices_path}"
    
    # Verify readiness improves
    status = compute_data_ready_status(test_wave)
    assert status['readiness_status'] != 'unavailable', \
        f"Wave {test_wave} should not be unavailable after price generation"
    
    print(f"✓ Price generation with dummy data works for {test_wave}")
    print(f"  Files generated: {', '.join(result['files_generated'])}")
    print(f"  Readiness status: {status['readiness_status']}")
    return True


def test_skip_existing_functionality():
    """Test skip_existing parameter."""
    print("\n" + "=" * 80)
    print("TEST: Skip Existing Functionality")
    print("=" * 80)
    
    # Use a wave that already has prices.csv
    test_wave = 'gold_wave'
    
    # Ensure it has prices.csv
    wave_dir = get_wave_analytics_dir(test_wave)
    prices_path = os.path.join(wave_dir, 'prices.csv')
    
    if not os.path.exists(prices_path):
        # Generate it first
        generate_prices_for_wave(test_wave, use_dummy_data=True)
    
    # Test with skip_existing=True
    result = generate_prices_for_wave(
        test_wave,
        use_dummy_data=True,
        skip_existing=True
    )
    
    assert result['skipped'] == True, \
        f"Should skip {test_wave} when skip_existing=True"
    assert 'already exists' in result['skip_reason'], \
        f"Skip reason should mention 'already exists' for {test_wave}"
    
    # Test with skip_existing=False
    result = generate_prices_for_wave(
        test_wave,
        use_dummy_data=True,
        skip_existing=False
    )
    
    assert result['skipped'] == False or result['is_smartsafe'], \
        f"Should not skip {test_wave} when skip_existing=False"
    
    print(f"✓ Skip existing functionality works correctly")
    return True


def test_all_waves_coverage():
    """Test that all waves are properly categorized."""
    print("\n" + "=" * 80)
    print("TEST: All Waves Coverage")
    print("=" * 80)
    
    all_waves = get_all_wave_ids()
    smartsafe_count = 0
    non_smartsafe_count = 0
    
    for wave_id in all_waves:
        if is_smartsafe_cash_wave(wave_id):
            smartsafe_count += 1
            
            # SmartSafe waves should always be ready
            status = compute_data_ready_status(wave_id)
            assert status['readiness_status'] == 'full', \
                f"SmartSafe wave {wave_id} should be 'full' ready"
        else:
            non_smartsafe_count += 1
    
    # Verify total count
    assert len(all_waves) == smartsafe_count + non_smartsafe_count, \
        "All waves should be categorized as either SmartSafe or non-SmartSafe"
    
    # Known values from the system
    assert smartsafe_count == 2, \
        "Should have exactly 2 SmartSafe cash waves"
    assert non_smartsafe_count == 26, \
        "Should have 26 non-SmartSafe waves"
    
    print(f"✓ All {len(all_waves)} waves properly categorized")
    print(f"  SmartSafe: {smartsafe_count}")
    print(f"  Non-SmartSafe: {non_smartsafe_count}")
    return True


def test_readiness_levels():
    """Test that readiness levels are properly computed."""
    print("\n" + "=" * 80)
    print("TEST: Readiness Levels")
    print("=" * 80)
    
    # Test a wave with prices
    test_wave = 'gold_wave'
    
    # Ensure it has prices
    result = generate_prices_for_wave(test_wave, use_dummy_data=True)
    
    status = compute_data_ready_status(test_wave)
    
    # Should have a valid readiness status
    assert status['readiness_status'] in ['full', 'partial', 'operational', 'unavailable'], \
        f"Wave {test_wave} should have valid readiness status"
    
    # Should have coverage info
    assert 'coverage_pct' in status, \
        "Status should include coverage_pct"
    assert 'history_days' in status, \
        "Status should include history_days"
    
    # Should have allowed_analytics
    assert 'allowed_analytics' in status, \
        "Status should include allowed_analytics"
    
    print(f"✓ Readiness levels properly computed for {test_wave}")
    print(f"  Status: {status['readiness_status']}")
    print(f"  Coverage: {status['coverage_pct']:.1f}%")
    print(f"  History: {status['history_days']} days")
    return True


def test_batch_generation():
    """Test batch generation for multiple waves."""
    print("\n" + "=" * 80)
    print("TEST: Batch Generation")
    print("=" * 80)
    
    # Test with a small set of waves
    test_waves = ['gold_wave', 'sp500_wave', 'smartsafe_treasury_cash_wave']
    
    summary = generate_all_prices(
        wave_ids=test_waves,
        lookback_days=14,
        use_dummy_data=True,
        skip_existing=False
    )
    
    # Verify summary structure
    assert 'total_waves' in summary
    assert 'successful' in summary
    assert 'failed' in summary
    assert 'skipped_smartsafe' in summary
    assert 'results' in summary
    
    # Verify counts
    assert summary['total_waves'] == len(test_waves), \
        "Should process all requested waves"
    assert summary['skipped_smartsafe'] == 1, \
        "Should skip 1 SmartSafe wave"
    assert summary['successful'] + summary['failed'] + summary['skipped_smartsafe'] == summary['total_waves'], \
        "All waves should be accounted for"
    
    print(f"✓ Batch generation works correctly")
    print(f"  Total: {summary['total_waves']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Skipped (SmartSafe): {summary['skipped_smartsafe']}")
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 80)
    print("CANONICAL PRICE LOADING MECHANISM - TEST SUITE")
    print("=" * 80)
    print()
    
    tests = [
        ("SmartSafe Exemption", test_smartsafe_exemption),
        ("Non-SmartSafe Requires Prices", test_non_smartsafe_requires_prices),
        ("Price Generation with Dummy Data", test_price_generation_with_dummy_data),
        ("Skip Existing Functionality", test_skip_existing_functionality),
        ("All Waves Coverage", test_all_waves_coverage),
        ("Readiness Levels", test_readiness_levels),
        ("Batch Generation", test_batch_generation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n✗ FAILED: {test_name}")
            print(f"  Error: {str(e)}")
        except Exception as e:
            failed += 1
            print(f"\n✗ ERROR: {test_name}")
            print(f"  Exception: {str(e)}")
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 80)
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED")
        return True
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

"""
Test suite for graceful degradation and data resilience changes.

Tests the following key features:
1. Ticker failure tracking in waves_engine.py
2. Live snapshot generation and loading
3. Broken tickers report
4. Wave weights completeness
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_download_history_returns_failures():
    """Test that _download_history returns both DataFrame and failures dict."""
    print("=" * 70)
    print("TEST: _download_history returns failures dict")
    print("=" * 70)
    
    from waves_engine import _download_history
    
    # Test with dummy tickers (some valid, some invalid)
    tickers = ["SPY", "INVALID_TICKER_XYZ123"]
    
    try:
        result = _download_history(tickers, days=30)
        
        # Check return type
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return (DataFrame, dict)"
        
        prices_df, failures = result
        
        assert isinstance(prices_df, pd.DataFrame), "First element should be DataFrame"
        assert isinstance(failures, dict), "Second element should be dict"
        
        print(f"✓ Returns correct types")
        print(f"  Prices DataFrame shape: {prices_df.shape}")
        print(f"  Failures dict keys: {len(failures)}")
        
        # Note: In test environment without real data, both might be empty
        print("✓ Test passed")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_live_snapshot_fallback():
    """Test that load_live_snapshot provides fallback when file doesn't exist."""
    print("\n" + "=" * 70)
    print("TEST: Live snapshot fallback mechanism")
    print("=" * 70)
    
    from analytics_pipeline import load_live_snapshot
    
    try:
        # Test with non-existent file
        df = load_live_snapshot(path='nonexistent_snapshot.csv', fallback=True)
        
        assert not df.empty, "Fallback should return non-empty DataFrame"
        assert len(df) == 28, f"Should have 28 waves, got {len(df)}"
        
        required_cols = ['wave_id', 'wave_name', 'readiness_status', 'data_regime']
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
        
        print(f"✓ Fallback works correctly")
        print(f"  Waves: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_broken_tickers_report():
    """Test that broken tickers report generates correctly."""
    print("\n" + "=" * 70)
    print("TEST: Broken tickers report generation")
    print("=" * 70)
    
    from analytics_pipeline import get_broken_tickers_report
    
    try:
        report = get_broken_tickers_report()
        
        # Check structure
        assert isinstance(report, dict), "Should return dict"
        
        required_keys = [
            'total_broken',
            'broken_by_wave',
            'ticker_failure_counts',
            'most_common_failures',
            'total_waves_with_failures'
        ]
        
        for key in required_keys:
            assert key in report, f"Missing required key: {key}"
        
        print(f"✓ Report structure correct")
        print(f"  Total broken tickers: {report['total_broken']}")
        print(f"  Waves with failures: {report['total_waves_with_failures']}")
        
        if report['most_common_failures']:
            top_ticker, count = report['most_common_failures'][0]
            print(f"  Most common failure: {top_ticker} ({count} waves)")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_wave_weights_completeness():
    """Test that wave_weights.csv has all 28 waves."""
    print("\n" + "=" * 70)
    print("TEST: Wave weights CSV completeness")
    print("=" * 70)
    
    from waves_engine import WAVE_WEIGHTS, WAVE_ID_REGISTRY
    import pandas as pd
    
    try:
        # Check engine has 28 waves
        assert len(WAVE_WEIGHTS) == 28, f"WAVE_WEIGHTS should have 28 waves, got {len(WAVE_WEIGHTS)}"
        assert len(WAVE_ID_REGISTRY) == 28, f"WAVE_ID_REGISTRY should have 28 waves, got {len(WAVE_ID_REGISTRY)}"
        
        print(f"✓ Engine has 28 waves")
        
        # Check CSV has 28 waves
        csv_path = 'wave_weights.csv'
        assert os.path.exists(csv_path), f"wave_weights.csv not found"
        
        df = pd.read_csv(csv_path)
        unique_waves = df['wave'].nunique()
        
        assert unique_waves == 28, f"CSV should have 28 waves, got {unique_waves}"
        
        print(f"✓ wave_weights.csv has 28 waves")
        print(f"  Total rows: {len(df)}")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_graded_readiness_levels():
    """Test that compute_data_ready_status returns graded readiness."""
    print("\n" + "=" * 70)
    print("TEST: Graded readiness status levels")
    print("=" * 70)
    
    from analytics_pipeline import compute_data_ready_status
    
    try:
        # Pick a test wave
        wave_id = 'sp500_wave'
        
        status = compute_data_ready_status(wave_id)
        
        # Check structure
        assert isinstance(status, dict), "Should return dict"
        assert 'readiness_status' in status, "Should have readiness_status"
        
        # Check it's one of the graded levels
        valid_levels = ['full', 'partial', 'operational', 'unavailable']
        actual_status = status['readiness_status']
        assert actual_status in valid_levels, f"Status '{actual_status}' not in {valid_levels}"
        
        print(f"✓ Graded readiness works")
        print(f"  Wave: {wave_id}")
        print(f"  Status: {actual_status}")
        print(f"  Coverage: {status.get('coverage_pct', 0):.1f}%")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_compute_core_handles_failures():
    """Test that _compute_core handles missing tickers gracefully."""
    print("\n" + "=" * 70)
    print("TEST: _compute_core graceful failure handling")
    print("=" * 70)
    
    from waves_engine import compute_history_nav
    
    try:
        # Try to compute NAV for a wave (should not crash even if data is missing)
        result = compute_history_nav("S&P 500 Wave", mode="Standard", days=30)
        
        # Should return a DataFrame, even if empty
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        
        expected_cols = ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]
        for col in expected_cols:
            assert col in result.columns, f"Missing expected column: {col}"
        
        print(f"✓ compute_history_nav handles failures gracefully")
        print(f"  Result shape: {result.shape}")
        print(f"  Empty: {result.empty}")
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("GRACEFUL DEGRADATION & DATA RESILIENCE TEST SUITE")
    print("=" * 70)
    print()
    
    tests = [
        test_download_history_returns_failures,
        test_live_snapshot_fallback,
        test_broken_tickers_report,
        test_wave_weights_completeness,
        test_graded_readiness_levels,
        test_compute_core_handles_failures,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ Test {test_func.__name__} crashed: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

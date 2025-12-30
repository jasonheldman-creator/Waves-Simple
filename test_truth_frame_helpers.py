"""
test_truth_frame_helpers.py

Unit tests for truth_frame_helpers.py
"""

import pandas as pd
import numpy as np
from truth_frame_helpers import (
    get_wave_metric,
    get_wave_returns,
    get_wave_alphas,
    format_return_display,
    format_alpha_display,
    get_wave_summary,
    get_top_performers,
    get_readiness_summary,
    format_readiness_badge,
)


def create_test_truthframe():
    """Create a test TruthFrame with sample data"""
    return pd.DataFrame([
        {
            'wave_id': 'sp500_wave',
            'display_name': 'S&P 500 Wave',
            'return_1d': 0.01,
            'return_30d': 0.05,
            'return_60d': 0.10,
            'return_365d': 0.20,
            'alpha_1d': 0.005,
            'alpha_30d': 0.02,
            'alpha_60d': 0.04,
            'alpha_365d': 0.08,
            'readiness_status': 'full',
            'coverage_pct': 100.0,
        },
        {
            'wave_id': 'income_wave',
            'display_name': 'Income Wave',
            'return_1d': 0.002,
            'return_30d': 0.015,
            'return_60d': 0.03,
            'return_365d': 0.06,
            'alpha_1d': 0.001,
            'alpha_30d': 0.005,
            'alpha_60d': 0.01,
            'alpha_365d': 0.02,
            'readiness_status': 'partial',
            'coverage_pct': 75.0,
        },
        {
            'wave_id': 'gold_wave',
            'display_name': 'Gold Wave',
            'return_1d': np.nan,
            'return_30d': np.nan,
            'return_60d': np.nan,
            'return_365d': np.nan,
            'alpha_1d': np.nan,
            'alpha_30d': np.nan,
            'alpha_60d': np.nan,
            'alpha_365d': np.nan,
            'readiness_status': 'unavailable',
            'coverage_pct': 0.0,
        }
    ])


def test_get_wave_metric():
    """Test getting individual metrics"""
    truth_df = create_test_truthframe()
    
    # Test valid metric
    return_1d = get_wave_metric(truth_df, 'sp500_wave', 'return_1d')
    assert return_1d == 0.01, f"Expected 0.01, got {return_1d}"
    print("✓ get_wave_metric: Valid metric")
    
    # Test missing wave
    result = get_wave_metric(truth_df, 'nonexistent_wave', 'return_1d', default=0.0)
    assert result == 0.0, f"Expected 0.0, got {result}"
    print("✓ get_wave_metric: Missing wave returns default")
    
    # Test NaN value
    result = get_wave_metric(truth_df, 'gold_wave', 'return_1d', default=-999)
    assert result == -999, f"Expected -999, got {result}"
    print("✓ get_wave_metric: NaN returns default")


def test_get_wave_returns():
    """Test getting all returns for a wave"""
    truth_df = create_test_truthframe()
    
    returns = get_wave_returns(truth_df, 'sp500_wave')
    
    assert returns['1d'] == 0.01
    assert returns['30d'] == 0.05
    assert returns['60d'] == 0.10
    assert returns['365d'] == 0.20
    
    print("✓ get_wave_returns: All timeframes returned")


def test_get_wave_alphas():
    """Test getting all alphas for a wave"""
    truth_df = create_test_truthframe()
    
    alphas = get_wave_alphas(truth_df, 'income_wave')
    
    assert alphas['1d'] == 0.001
    assert alphas['30d'] == 0.005
    assert alphas['60d'] == 0.01
    assert alphas['365d'] == 0.02
    
    print("✓ get_wave_alphas: All timeframes returned")


def test_format_return_display():
    """Test formatting returns for display"""
    
    # Positive return
    result = format_return_display(0.0123)
    assert result == "+1.23%", f"Expected '+1.23%', got '{result}'"
    print("✓ format_return_display: Positive value")
    
    # Negative return
    result = format_return_display(-0.0056)
    assert result == "-0.56%", f"Expected '-0.56%', got '{result}'"
    print("✓ format_return_display: Negative value")
    
    # NaN return
    result = format_return_display(np.nan)
    assert result == "N/A", f"Expected 'N/A', got '{result}'"
    print("✓ format_return_display: NaN value")
    
    # None return
    result = format_return_display(None)
    assert result == "N/A", f"Expected 'N/A', got '{result}'"
    print("✓ format_return_display: None value")


def test_get_wave_summary():
    """Test getting complete wave summary"""
    truth_df = create_test_truthframe()
    
    summary = get_wave_summary(truth_df, 'sp500_wave')
    
    assert summary['wave_id'] == 'sp500_wave'
    assert summary['display_name'] == 'S&P 500 Wave'
    assert summary['return_1d'] == 0.01
    assert summary['readiness_status'] == 'full'
    
    print("✓ get_wave_summary: Complete summary returned")


def test_get_top_performers():
    """Test getting top performers"""
    truth_df = create_test_truthframe()
    
    # Get top performers by 30d return
    top = get_top_performers(truth_df, metric='return_30d', n=2, ascending=False)
    
    assert len(top) == 2
    assert top.iloc[0]['wave_id'] == 'sp500_wave'  # Highest return
    assert top.iloc[1]['wave_id'] == 'income_wave'  # Second highest
    
    print("✓ get_top_performers: Correct sorting and filtering")


def test_get_readiness_summary():
    """Test readiness summary"""
    truth_df = create_test_truthframe()
    
    summary = get_readiness_summary(truth_df)
    
    assert summary['total'] == 3
    assert summary['full'] == 1
    assert summary['partial'] == 1
    assert summary['unavailable'] == 1
    
    print("✓ get_readiness_summary: Correct counts")


def test_format_readiness_badge():
    """Test readiness badge formatting"""
    
    badge = format_readiness_badge('full', 100.0)
    assert '🟢' in badge
    assert 'Full' in badge
    assert '100%' in badge
    print("✓ format_readiness_badge: Full status")
    
    badge = format_readiness_badge('partial', 75.0)
    assert '🟡' in badge
    assert 'Partial' in badge
    print("✓ format_readiness_badge: Partial status")
    
    badge = format_readiness_badge('unavailable')
    assert '🔴' in badge
    assert 'Unavailable' in badge
    print("✓ format_readiness_badge: Unavailable status")


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("TRUTHFRAME HELPERS UNIT TESTS")
    print("=" * 80)
    
    tests = [
        ("Get Wave Metric", test_get_wave_metric),
        ("Get Wave Returns", test_get_wave_returns),
        ("Get Wave Alphas", test_get_wave_alphas),
        ("Format Return Display", test_format_return_display),
        ("Get Wave Summary", test_get_wave_summary),
        ("Get Top Performers", test_get_top_performers),
        ("Get Readiness Summary", test_get_readiness_summary),
        ("Format Readiness Badge", test_format_readiness_badge),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"Test: {test_name}")
        print(f"{'='*80}")
        try:
            test_func()
            print(f"✓ PASSED: {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {test_name}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {test_name}")
            print(f"  Exception: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

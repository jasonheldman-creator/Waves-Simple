"""
Test suite for daily ledger reconciliation in compute_portfolio_alpha_ledger.

This test validates that the reconciliation logic correctly:
1. Verifies realized_return - benchmark_return == alpha_total
2. Verifies alpha_selection + alpha_overlay + alpha_residual == alpha_total
3. Marks periods as unavailable when tolerance (0.10%) is exceeded
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.wave_performance import compute_portfolio_alpha_ledger, RESIDUAL_TOLERANCE
# Monkey-patch get_all_waves_universe for testing
import helpers.wave_performance as wave_perf_module


@dataclass
class Holding:
    """Test holding class matching waves_engine.Holding."""
    ticker: str
    weight: float
    name: str = None


def mock_get_all_waves_universe():
    """Mock universe function for testing."""
    # Return the test wave names
    return {
        'waves': ['Test Wave 1', 'Test Wave 2', 'Test Wave'],
        'wave_ids': ['test_wave_1', 'test_wave_2', 'test_wave'],
        'count': 3,
        'source': 'test',
        'version': '1.0'
    }


def create_test_price_book():
    """Create a simple test price book with known values."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Create price data that varies smoothly
    price_book = pd.DataFrame({
        'SPY': [400 + i * 0.5 for i in range(100)],  # Benchmark
        '^VIX': [20 + np.sin(i * 0.1) * 5 for i in range(100)],  # VIX varies 15-25
        'BIL': [90 + i * 0.01 for i in range(100)],  # Safe asset
        'AAPL': [150 + i * 0.7 for i in range(100)],  # Sample ticker
        'MSFT': [300 + i * 0.6 for i in range(100)],  # Sample ticker
    }, index=dates)
    
    return price_book


def test_ledger_reconciliation_basic():
    """Test that ledger is created with correct columns and reconciliation passes."""
    # Patch get_all_waves_universe for this test
    original_func = wave_perf_module.get_all_waves_universe
    wave_perf_module.get_all_waves_universe = mock_get_all_waves_universe
    
    try:
        price_book = create_test_price_book()
        
        # Use a smaller wave registry for testing
        test_wave_registry = {
            'Test Wave 1': [
                Holding(ticker='AAPL', weight=1.0, name='Apple')
            ],
            'Test Wave 2': [
                Holding(ticker='MSFT', weight=1.0, name='Microsoft')
            ]
        }
    
    result = compute_portfolio_alpha_ledger(
        price_book,
        periods=[30],
        wave_registry=test_wave_registry,
        vix_exposure_enabled=True
    )
    
    # Check that computation succeeded
    assert result['success'] is True, f"Computation failed: {result.get('failure_reason')}"
    
    # Check that ledger exists
    assert result['daily_ledger'] is not None, "Daily ledger not created"
    ledger = result['daily_ledger']
    
    # Check required columns
    required_columns = [
        'risk_return', 'safe_return', 'benchmark_return', 'exposure',
        'realized_return', 'alpha_total', 'alpha_selection', 'alpha_overlay', 'alpha_residual'
    ]
    for col in required_columns:
        assert col in ledger.columns, f"Missing required column: {col}"
    
    # Check reconciliation passed
    assert result['reconciliation_passed'] is True, "Reconciliation should pass with valid data"
    
    # Check reconciliation diffs are small
    assert result['reconciliation_1_max_diff'] is not None
    assert result['reconciliation_2_max_diff'] is not None
    assert result['reconciliation_1_max_diff'] < RESIDUAL_TOLERANCE, \
        f"Reconciliation 1 diff {result['reconciliation_1_max_diff']:.8f} exceeds tolerance {RESIDUAL_TOLERANCE:.8f}"
    assert result['reconciliation_2_max_diff'] < RESIDUAL_TOLERANCE, \
        f"Reconciliation 2 diff {result['reconciliation_2_max_diff']:.8f} exceeds tolerance {RESIDUAL_TOLERANCE:.8f}"
    
    print(f"✓ Ledger created with {len(ledger)} rows")
    print(f"✓ Reconciliation 1 max diff: {result['reconciliation_1_max_diff']:.8f}")
    print(f"✓ Reconciliation 2 max diff: {result['reconciliation_2_max_diff']:.8f}")
    
    finally:
        # Restore original function
        wave_perf_module.get_all_waves_universe = original_func


def test_ledger_reconciliation_formulas():
    """Test that reconciliation formulas are correctly implemented."""
    original_func = wave_perf_module.get_all_waves_universe
    wave_perf_module.get_all_waves_universe = mock_get_all_waves_universe
    
    try:
        price_book = create_test_price_book()
        
        test_wave_registry = {
            'Test Wave': [
                Holding(ticker='AAPL', weight=1.0, name='Apple')
            ]
        }
        
        result = compute_portfolio_alpha_ledger(
            price_book,
            periods=[30],
            wave_registry=test_wave_registry,
            vix_exposure_enabled=False  # Disable VIX to simplify
        )
        
        assert result['success'] is True
        ledger = result['daily_ledger']
        
        # Manually verify reconciliation 1: realized_return - benchmark_return == alpha_total
        check_1 = np.abs(
            (ledger['realized_return'] - ledger['benchmark_return']) - ledger['alpha_total']
        )
        assert (check_1 < 1e-10).all(), "Reconciliation 1 formula incorrect"
        
        # Manually verify reconciliation 2: alpha_selection + alpha_overlay + alpha_residual == alpha_total
        check_2 = np.abs(
            (ledger['alpha_selection'] + ledger['alpha_overlay'] + ledger['alpha_residual']) - 
            ledger['alpha_total']
        )
        assert (check_2 < 1e-10).all(), "Reconciliation 2 formula incorrect"
        
        print(f"✓ Reconciliation formulas verified")
    
    finally:
        wave_perf_module.get_all_waves_universe = original_func


def test_ledger_alpha_components():
    """Test that alpha components are correctly computed."""
    price_book = create_test_price_book()
    
    test_wave_registry = {
        'Test Wave': [
            Holding(ticker='AAPL', weight=1.0, name='Apple')
        ]
    }
    
    result = compute_portfolio_alpha_ledger(
        price_book,
        periods=[30],
        wave_registry=test_wave_registry,
        vix_exposure_enabled=True
    )
    
    assert result['success'] is True
    ledger = result['daily_ledger']
    
    # Check alpha_total computation
    expected_alpha_total = ledger['realized_return'] - ledger['benchmark_return']
    assert np.allclose(ledger['alpha_total'], expected_alpha_total), \
        "alpha_total should equal realized_return - benchmark_return"
    
    # Check alpha_selection computation
    expected_alpha_selection = ledger['risk_return'] - ledger['benchmark_return']
    assert np.allclose(ledger['alpha_selection'], expected_alpha_selection), \
        "alpha_selection should equal risk_return - benchmark_return"
    
    # Check alpha_overlay computation
    expected_alpha_overlay = ledger['realized_return'] - ledger['risk_return']
    assert np.allclose(ledger['alpha_overlay'], expected_alpha_overlay), \
        "alpha_overlay should equal realized_return - risk_return"
    
    # Check alpha_residual computation
    expected_alpha_residual = (
        ledger['alpha_total'] - 
        (ledger['alpha_selection'] + ledger['alpha_overlay'])
    )
    assert np.allclose(ledger['alpha_residual'], expected_alpha_residual), \
        "alpha_residual should equal alpha_total - (alpha_selection + alpha_overlay)"
    
    print(f"✓ Alpha components verified")


def test_ledger_with_vix_overlay():
    """Test that ledger correctly handles VIX overlay exposure."""
    price_book = create_test_price_book()
    
    test_wave_registry = {
        'Test Wave': [
            Holding(ticker='AAPL', weight=1.0, name='Apple')
        ]
    }
    
    result = compute_portfolio_alpha_ledger(
        price_book,
        periods=[30],
        wave_registry=test_wave_registry,
        vix_exposure_enabled=True
    )
    
    assert result['success'] is True
    assert result['overlay_available'] is True, "VIX overlay should be available"
    assert result['vix_ticker_used'] == '^VIX', "Should use ^VIX ticker"
    
    ledger = result['daily_ledger']
    
    # Check that exposure varies (not all 1.0)
    assert ledger['exposure'].min() < 1.0, "Exposure should vary with VIX"
    assert ledger['exposure'].max() <= 1.0, "Exposure should not exceed 1.0"
    assert ledger['exposure'].min() >= 0.0, "Exposure should not be negative"
    
    # Check that realized_return reflects exposure
    # realized_return = exposure * risk_return + (1 - exposure) * safe_return
    expected_realized = (
        ledger['exposure'] * ledger['risk_return'] + 
        (1 - ledger['exposure']) * ledger['safe_return']
    )
    assert np.allclose(ledger['realized_return'], expected_realized, atol=1e-10), \
        "realized_return should match exposure formula"
    
    print(f"✓ VIX overlay correctly applied")
    print(f"  Exposure range: [{ledger['exposure'].min():.3f}, {ledger['exposure'].max():.3f}]")


def test_ledger_without_vix_overlay():
    """Test that ledger works when VIX overlay is disabled."""
    price_book = create_test_price_book()
    
    test_wave_registry = {
        'Test Wave': [
            Holding(ticker='AAPL', weight=1.0, name='Apple')
        ]
    }
    
    result = compute_portfolio_alpha_ledger(
        price_book,
        periods=[30],
        wave_registry=test_wave_registry,
        vix_exposure_enabled=False
    )
    
    assert result['success'] is True
    assert result['overlay_available'] is False, "VIX overlay should not be available"
    
    ledger = result['daily_ledger']
    
    # Check that exposure is constant at 1.0
    assert (ledger['exposure'] == 1.0).all(), "Exposure should be 1.0 when overlay disabled"
    
    # Check that realized_return equals risk_return (no overlay)
    assert np.allclose(ledger['realized_return'], ledger['risk_return']), \
        "realized_return should equal risk_return when exposure is 1.0"
    
    # Check that alpha_overlay is zero (no overlay effect)
    assert np.allclose(ledger['alpha_overlay'], 0.0, atol=1e-10), \
        "alpha_overlay should be zero when exposure is 1.0"
    
    print(f"✓ Ledger works correctly without VIX overlay")


if __name__ == '__main__':
    print("=" * 70)
    print("Testing Daily Ledger Reconciliation")
    print("=" * 70)
    
    tests = [
        ("Basic ledger creation and reconciliation", test_ledger_reconciliation_basic),
        ("Reconciliation formulas", test_ledger_reconciliation_formulas),
        ("Alpha components computation", test_ledger_alpha_components),
        ("VIX overlay integration", test_ledger_with_vix_overlay),
        ("No VIX overlay", test_ledger_without_vix_overlay),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}...")
            test_func()
            print(f"  ✓ PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed > 0:
        sys.exit(1)

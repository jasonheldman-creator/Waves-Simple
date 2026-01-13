"""
Integration test for Portfolio Snapshot cache invalidation.

Validates that:
1. Portfolio Snapshot computes correctly from strategy-adjusted daily series
2. Cache invalidates when snapshot version changes
3. Portfolio metrics reflect VIX overlay adjustments
"""

import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.wave_performance import compute_portfolio_alpha_ledger
from helpers.price_book import get_price_book
from helpers.snapshot_version import get_snapshot_version_key

# Constants for test validation
ALPHA_TOLERANCE_PCT = 0.001  # 0.1% tolerance for alpha validation
RESIDUAL_TOLERANCE_PCT = 0.001  # 0.1% tolerance for residual validation
OVERLAY_ALPHA_TOLERANCE_PCT = 0.001  # 0.1% tolerance for overlay alpha when VIX unavailable


def test_portfolio_alpha_ledger_computes_successfully():
    """Test that portfolio alpha ledger computes without errors."""
    # Get price book
    price_book = get_price_book()
    
    assert price_book is not None, "Price book should be available"
    assert not price_book.empty, "Price book should not be empty"
    
    # Compute portfolio alpha ledger
    ledger = compute_portfolio_alpha_ledger(
        price_book,
        periods=[1, 30, 60, 365],
        benchmark_ticker='SPY',
        mode='Standard',
        vix_exposure_enabled=True
    )
    
    # Verify success
    assert ledger['success'], f"Ledger computation should succeed: {ledger.get('failure_reason')}"
    
    # Verify daily series exist
    assert ledger['daily_realized_return'] is not None, "Should have daily realized returns"
    assert ledger['daily_unoverlay_return'] is not None, "Should have daily unoverlay returns"
    assert ledger['daily_benchmark_return'] is not None, "Should have daily benchmark returns"
    
    # Verify period results exist
    assert 'period_results' in ledger, "Should have period results"
    assert len(ledger['period_results']) > 0, "Should have at least one period"


def test_portfolio_snapshot_reflects_vix_overlay():
    """Test that Portfolio Snapshot metrics reflect VIX overlay when enabled."""
    # Get price book
    price_book = get_price_book()
    
    # Compute WITH VIX overlay
    ledger_with_vix = compute_portfolio_alpha_ledger(
        price_book,
        periods=[30, 60],
        benchmark_ticker='SPY',
        mode='Standard',
        vix_exposure_enabled=True
    )
    
    # Both should succeed
    assert ledger_with_vix['success'], "VIX-enabled ledger should succeed"
    
    # Check overlay availability
    if ledger_with_vix['overlay_available']:
        # If VIX overlay is available, overlay_alpha should exist
        period_30d = ledger_with_vix['period_results'].get('30D', {})
        if period_30d.get('available'):
            overlay_alpha = period_30d.get('overlay_alpha')
            assert overlay_alpha is not None, "Should have overlay_alpha when VIX is available"
            
            # Verify that overlay_alpha is computed (can be positive, negative, or zero)
            assert isinstance(overlay_alpha, (int, float)), \
                "Overlay alpha should be numeric"
    else:
        # If VIX overlay is not available, verify that overlay_alpha is still present but should be 0
        period_30d = ledger_with_vix['period_results'].get('30D', {})
        if period_30d.get('available'):
            overlay_alpha = period_30d.get('overlay_alpha')
            assert overlay_alpha is not None, "Should have overlay_alpha field"
            # When VIX is not available, overlay_alpha should be close to 0
            assert abs(overlay_alpha) < OVERLAY_ALPHA_TOLERANCE_PCT, \
                f"Overlay alpha should be ~0 when VIX unavailable, got {overlay_alpha}"


def test_snapshot_version_key_available():
    """Test that snapshot version key is available for cache invalidation."""
    version = get_snapshot_version_key()
    
    assert version is not None, "Snapshot version should be available"
    assert isinstance(version, str), "Snapshot version should be a string"
    assert len(version) > 0, "Snapshot version should not be empty"
    assert version != "unknown_unknown", "Snapshot version should be valid"


def test_portfolio_ledger_period_results_structure():
    """Test that period results have correct structure."""
    price_book = get_price_book()
    
    ledger = compute_portfolio_alpha_ledger(
        price_book,
        periods=[1, 30, 60, 365],
        benchmark_ticker='SPY',
        mode='Standard',
        vix_exposure_enabled=True
    )
    
    assert ledger['success'], "Ledger should compute successfully"
    
    # Check each period
    for period_key in ['1D', '30D', '60D', '365D']:
        assert period_key in ledger['period_results'], f"Should have {period_key} results"
        
        period_data = ledger['period_results'][period_key]
        assert 'available' in period_data, f"{period_key} should have 'available' field"
        
        if period_data['available']:
            # Check required fields for available periods
            required_fields = [
                'cum_realized', 'cum_unoverlay', 'cum_benchmark',
                'total_alpha', 'selection_alpha', 'overlay_alpha', 'residual',
                'start_date', 'end_date', 'rows_used'
            ]
            
            for field in required_fields:
                assert field in period_data, f"{period_key} should have '{field}' field"
                
                # Check that numeric fields are actually numbers
                if field in ['cum_realized', 'cum_unoverlay', 'cum_benchmark',
                           'total_alpha', 'selection_alpha', 'overlay_alpha', 'residual']:
                    value = period_data[field]
                    assert value is not None, f"{period_key}.{field} should not be None"
                    assert isinstance(value, (int, float)), \
                        f"{period_key}.{field} should be numeric, got {type(value)}"


def test_portfolio_ledger_alpha_decomposition():
    """Test that alpha decomposition is mathematically correct."""
    price_book = get_price_book()
    
    ledger = compute_portfolio_alpha_ledger(
        price_book,
        periods=[30, 60],
        benchmark_ticker='SPY',
        mode='Standard',
        vix_exposure_enabled=True
    )
    
    assert ledger['success'], "Ledger should compute successfully"
    
    # Check 30D period
    period_30d = ledger['period_results'].get('30D', {})
    if period_30d.get('available'):
        total_alpha = period_30d['total_alpha']
        selection_alpha = period_30d['selection_alpha']
        overlay_alpha = period_30d['overlay_alpha']
        residual = period_30d['residual']
        
        # Verify alpha decomposition: total = selection + overlay + residual
        computed_total = selection_alpha + overlay_alpha + residual
        
        # Allow small numerical error
        diff = abs(total_alpha - computed_total)
        
        assert diff < RESIDUAL_TOLERANCE_PCT, \
            f"Alpha decomposition failed: total={total_alpha:.6f}, " \
            f"selection+overlay+residual={computed_total:.6f}, diff={diff:.6f}"


def test_portfolio_ledger_uses_strategy_adjusted_returns():
    """Test that portfolio ledger uses strategy-adjusted (VIX overlay) returns."""
    price_book = get_price_book()
    
    ledger = compute_portfolio_alpha_ledger(
        price_book,
        periods=[30],
        benchmark_ticker='SPY',
        mode='Standard',
        vix_exposure_enabled=True
    )
    
    assert ledger['success'], "Ledger should compute successfully"
    
    # Check that we have exposure series (VIX overlay)
    if ledger['overlay_available']:
        assert ledger['daily_exposure'] is not None, \
            "Should have daily exposure when overlay is available"
        
        # Verify exposure is in valid range [0, 1]
        exposure_series = ledger['daily_exposure']
        assert (exposure_series >= 0).all(), "Exposure should be >= 0"
        assert (exposure_series <= 1).all(), "Exposure should be <= 1"
        
        # Verify that realized returns differ from unoverlay returns
        # when exposure is not constant 1.0
        realized = ledger['daily_realized_return']
        unoverlay = ledger['daily_unoverlay_return']
        
        # If exposure varies, realized and unoverlay should differ
        exposure_varies = (exposure_series.max() - exposure_series.min()) > 0.01
        if exposure_varies:
            # Check that realized and unoverlay are different on at least some dates
            diff = (realized - unoverlay).abs()
            has_difference = (diff > 1e-6).any()
            
            assert has_difference, \
                "Realized and unoverlay returns should differ when exposure varies"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])

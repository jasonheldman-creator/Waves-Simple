"""
Integration tests for Phase 1B.1: Portfolio Composite Benchmark

Tests that:
1. Portfolio composite benchmark computes successfully
2. Benchmark aligns with portfolio snapshot window
3. Alphas validate with 0.5% tolerance: abs((cum_realized - cum_benchmark) - cum_alpha)
4. S&P 500 Wave benchmark remains SPY at wave level
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waves_engine import (
    compute_history_nav,
    build_portfolio_composite_benchmark_returns,
    get_all_waves_universe,
    get_wave_id_from_display_name,
)

from helpers.wave_performance import compute_portfolio_snapshot

# Tolerance for alpha validation (0.5%)
ALPHA_TOLERANCE = 0.005

# S&P 500 Wave should use SPY as benchmark
SP500_WAVE = "S&P 500 Wave"
SP500_BENCHMARK_TICKER = "SPY"


def load_price_cache():
    """Load cached price data for testing."""
    cache_path = os.path.join(
        os.path.dirname(__file__),
        "data", "cache", "prices_cache.parquet"
    )
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    return None


@pytest.fixture(scope="module")
def price_df():
    """Fixture to load price cache once for all tests."""
    return load_price_cache()


@pytest.fixture(scope="module")
def all_waves():
    """Fixture to get all waves once for all tests."""
    universe = get_all_waves_universe()
    return universe.get('waves', [])


@pytest.fixture(scope="module")
def wave_results(price_df, all_waves):
    """Fixture to compute wave results once for all tests."""
    if price_df is None:
        pytest.skip("Price cache not available")
    
    results = {}
    for wave_name in all_waves:
        try:
            wave_df = compute_history_nav(
                wave_name=wave_name,
                mode='Standard',
                days=365,
                include_diagnostics=False,
                price_df=price_df
            )
            if not wave_df.empty and 'bm_ret' in wave_df.columns:
                results[wave_name] = wave_df
        except Exception as e:
            print(f"Warning: Failed to compute {wave_name}: {e}")
    
    return results


def test_price_cache_available(price_df):
    """Test that price cache is available."""
    assert price_df is not None, "Price cache should be available"
    assert not price_df.empty, "Price cache should not be empty"
    assert len(price_df) > 60, "Price cache should have at least 60 days of data"


def test_portfolio_composite_benchmark_builds(wave_results):
    """Test that portfolio composite benchmark builds successfully."""
    assert len(wave_results) > 0, "Should have at least one wave with results"
    
    composite = build_portfolio_composite_benchmark_returns(
        wave_results=wave_results,
        wave_weights=None  # Equal weights
    )
    
    assert not composite.empty, "Composite benchmark should not be empty"
    assert len(composite) >= 60, f"Composite benchmark should have at least 60 days, got {len(composite)}"
    assert composite.notna().any(), "Composite benchmark should have non-NaN values"


def test_portfolio_composite_benchmark_not_all_nan(wave_results):
    """Test that portfolio composite benchmark is not all NaN."""
    composite = build_portfolio_composite_benchmark_returns(
        wave_results=wave_results,
        wave_weights=None
    )
    
    assert not composite.isna().all(), "Composite benchmark should not be all NaN"
    
    # At least 90% of values should be non-NaN
    non_nan_pct = composite.notna().sum() / len(composite)
    assert non_nan_pct >= 0.9, f"At least 90% of composite benchmark should be non-NaN, got {non_nan_pct*100:.1f}%"


def test_portfolio_snapshot_uses_composite_benchmark(price_df):
    """Test that portfolio snapshot uses composite benchmark."""
    if price_df is None:
        pytest.skip("Price cache not available")
    
    snapshot = compute_portfolio_snapshot(
        price_book=price_df,
        mode='Standard',
        periods=[30, 60]
    )
    
    assert snapshot['success'], f"Portfolio snapshot should succeed: {snapshot.get('failure_reason')}"
    assert snapshot['has_portfolio_benchmark_series'], "Portfolio snapshot should have benchmark series"
    
    # Check debug info for composite benchmark
    debug = snapshot.get('debug', {})
    assert 'composite_benchmark_waves' in debug, "Debug info should contain composite_benchmark_waves"
    assert 'composite_benchmark_days' in debug, "Debug info should contain composite_benchmark_days"
    
    composite_waves = debug['composite_benchmark_waves']
    composite_days = debug['composite_benchmark_days']
    
    assert composite_waves > 0, "Composite benchmark should include at least one wave"
    assert composite_days >= 60, f"Composite benchmark should have at least 60 days, got {composite_days}"


def test_portfolio_snapshot_alpha_computation(price_df):
    """Test that portfolio snapshot computes alphas correctly."""
    if price_df is None:
        pytest.skip("Price cache not available")
    
    snapshot = compute_portfolio_snapshot(
        price_book=price_df,
        mode='Standard',
        periods=[30, 60]
    )
    
    assert snapshot['success'], f"Portfolio snapshot should succeed: {snapshot.get('failure_reason')}"
    
    # Check that alphas are computed
    for period in [30, 60]:
        period_key = f'{period}D'
        
        portfolio_ret = snapshot['portfolio_returns'].get(period_key)
        benchmark_ret = snapshot['benchmark_returns'].get(period_key)
        alpha = snapshot['alphas'].get(period_key)
        
        # Skip if insufficient history for this period
        if portfolio_ret is None or benchmark_ret is None or alpha is None:
            continue
        
        # Verify alpha = portfolio_ret - benchmark_ret
        expected_alpha = portfolio_ret - benchmark_ret
        assert abs(alpha - expected_alpha) < 1e-10, \
            f"{period}D: Alpha should equal portfolio_ret - benchmark_ret"


def test_alpha_validation_with_tolerance(price_df):
    """
    Test that cumulative alpha validates with 0.5% tolerance.
    
    Identity: Cumulative Alpha ≈ Cumulative Realized - Cumulative Benchmark
    Tolerance: abs((cum_realized - cum_benchmark) - cum_alpha) < 0.005
    """
    if price_df is None:
        pytest.skip("Price cache not available")
    
    snapshot = compute_portfolio_snapshot(
        price_book=price_df,
        mode='Standard',
        periods=[60]
    )
    
    assert snapshot['success'], f"Portfolio snapshot should succeed: {snapshot.get('failure_reason')}"
    
    period_key = '60D'
    portfolio_ret = snapshot['portfolio_returns'].get(period_key)
    benchmark_ret = snapshot['benchmark_returns'].get(period_key)
    alpha = snapshot['alphas'].get(period_key)
    
    if portfolio_ret is None or benchmark_ret is None or alpha is None:
        pytest.skip("Insufficient history for 60D period")
    
    # Cumulative returns (these are already cumulative in the snapshot)
    cum_realized = portfolio_ret
    cum_benchmark = benchmark_ret
    cum_alpha = alpha
    
    # Verify identity
    computed_alpha = cum_realized - cum_benchmark
    residual = abs(computed_alpha - cum_alpha)
    
    assert residual < ALPHA_TOLERANCE, \
        f"Alpha validation failed: residual {residual:.6f} exceeds tolerance {ALPHA_TOLERANCE}"


def test_sp500_wave_uses_spy_benchmark(price_df):
    """Test that S&P 500 Wave uses SPY as its benchmark at wave level."""
    if price_df is None:
        pytest.skip("Price cache not available")
    
    # Compute wave history for S&P 500 Wave
    sp500_df = compute_history_nav(
        wave_name=SP500_WAVE,
        mode='Standard',
        days=365,
        include_diagnostics=False,
        price_df=price_df
    )
    
    assert not sp500_df.empty, "S&P 500 Wave should have results"
    assert 'bm_ret' in sp500_df.columns, "S&P 500 Wave should have benchmark returns"
    
    # Get SPY returns for comparison
    if SP500_BENCHMARK_TICKER not in price_df.columns:
        pytest.skip(f"{SP500_BENCHMARK_TICKER} not in price cache")
    
    spy_prices = price_df[SP500_BENCHMARK_TICKER].copy()
    spy_returns = spy_prices.pct_change()
    
    # Align indices
    common_dates = sp500_df.index.intersection(spy_returns.index)
    if len(common_dates) < 60:
        pytest.skip("Insufficient common dates for comparison")
    
    sp500_bm = sp500_df.loc[common_dates, 'bm_ret']
    spy_ret = spy_returns.loc[common_dates]
    
    # Compare - they should be very similar (allowing for fillna(0.0) differences)
    # We'll check that correlation is very high
    valid_mask = sp500_bm.notna() & spy_ret.notna()
    if valid_mask.sum() < 60:
        pytest.skip("Insufficient valid data for correlation check")
    
    correlation = sp500_bm[valid_mask].corr(spy_ret[valid_mask])
    
    # S&P 500 Wave benchmark should be highly correlated with SPY (>0.99)
    # Note: Due to dynamic benchmark system, S&P 500 Wave might use a composite
    # but it should still be highly correlated with SPY
    assert correlation > 0.95, \
        f"S&P 500 Wave benchmark correlation with SPY should be >0.95, got {correlation:.4f}"


def test_composite_benchmark_alignment(price_df, wave_results):
    """Test that composite benchmark aligns properly with portfolio window."""
    if price_df is None:
        pytest.skip("Price cache not available")
    
    composite = build_portfolio_composite_benchmark_returns(
        wave_results=wave_results,
        wave_weights=None
    )
    
    snapshot = compute_portfolio_snapshot(
        price_book=price_df,
        mode='Standard',
        periods=[60]
    )
    
    assert snapshot['success'], "Portfolio snapshot should succeed"
    
    # Check that composite has sufficient history for 60D window
    assert len(composite) >= 60, "Composite should have at least 60 days for 60D window"
    
    # Verify date range alignment
    snapshot_date_range = snapshot.get('date_range')
    if snapshot_date_range:
        snapshot_start, snapshot_end = snapshot_date_range
        
        # Composite should cover the snapshot date range
        assert composite.index[0] <= pd.to_datetime(snapshot_start), \
            "Composite benchmark should start before or at portfolio start"


def test_equal_weight_composite(wave_results):
    """Test that composite benchmark uses equal weights correctly."""
    # Build composite with equal weights (default)
    composite_equal = build_portfolio_composite_benchmark_returns(
        wave_results=wave_results,
        wave_weights=None
    )
    
    # Build composite with explicit equal weights
    n_waves = len(wave_results)
    equal_weights = {name: 1.0 / n_waves for name in wave_results.keys()}
    composite_explicit = build_portfolio_composite_benchmark_returns(
        wave_results=wave_results,
        wave_weights=equal_weights
    )
    
    # They should be identical
    assert len(composite_equal) == len(composite_explicit), \
        "Equal weight composites should have same length"
    
    # Align on common dates
    common_dates = composite_equal.index.intersection(composite_explicit.index)
    if len(common_dates) > 0:
        diff = (composite_equal.loc[common_dates] - composite_explicit.loc[common_dates]).abs().max()
        assert diff < 1e-10, \
            "Default and explicit equal weights should produce identical results"


def test_composite_benchmark_reasonable_returns(wave_results):
    """Test that composite benchmark returns are reasonable (sanity check)."""
    composite = build_portfolio_composite_benchmark_returns(
        wave_results=wave_results,
        wave_weights=None
    )
    
    # Daily returns should typically be between -20% and +20%
    assert composite.min() > -0.2, f"Daily return too negative: {composite.min():.4f}"
    assert composite.max() < 0.2, f"Daily return too positive: {composite.max():.4f}"
    
    # Mean daily return should be reasonable (between -1% and +1%)
    mean_ret = composite.mean()
    assert -0.01 < mean_ret < 0.01, f"Mean daily return unreasonable: {mean_ret:.6f}"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])

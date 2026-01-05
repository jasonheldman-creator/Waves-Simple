#!/usr/bin/env python3
"""
Test suite for beta computation functionality.

Tests the compute_beta and compute_wave_beta functions added to helpers/wave_performance.py.
"""

import pandas as pd
import numpy as np
from helpers.wave_performance import compute_beta, compute_wave_beta


def test_compute_beta_basic():
    """Test basic beta computation with synthetic data."""
    print("Testing compute_beta with synthetic data...")
    
    # Create synthetic return series
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Benchmark returns (market)
    benchmark_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
    
    # Wave returns - correlated with benchmark (beta ~ 1.2)
    wave_returns = benchmark_returns * 1.2 + pd.Series(np.random.normal(0, 0.01, 100), index=dates)
    
    # Compute beta
    result = compute_beta(wave_returns, benchmark_returns, min_n=60)
    
    # Verify result structure
    assert result['success'] == True, "Beta computation should succeed"
    assert result['beta'] is not None, "Beta should not be None"
    assert result['n_observations'] == 100, f"Expected 100 observations, got {result['n_observations']}"
    assert result['correlation'] is not None, "Correlation should not be None"
    assert result['r_squared'] is not None, "R-squared should not be None"
    
    # Verify beta is approximately 1.2 (with some tolerance due to noise)
    assert 0.8 < result['beta'] < 1.6, f"Beta should be around 1.2, got {result['beta']}"
    
    print(f"✓ Beta computation test passed! Beta={result['beta']:.3f}, R²={result['r_squared']:.3f}")


def test_compute_beta_insufficient_data():
    """Test beta computation with insufficient data."""
    print("\nTesting compute_beta with insufficient data...")
    
    # Create short return series (less than min_n)
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    wave_returns = pd.Series(np.random.normal(0.001, 0.02, 30), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.001, 0.02, 30), index=dates)
    
    # Compute beta with min_n=60
    result = compute_beta(wave_returns, benchmark_returns, min_n=60)
    
    # Verify failure
    assert result['success'] == False, "Beta computation should fail with insufficient data"
    assert result['beta'] is None, "Beta should be None when computation fails"
    assert 'Insufficient data' in result['failure_reason'], f"Unexpected failure reason: {result['failure_reason']}"
    assert result['n_observations'] == 30, f"Expected 30 observations, got {result['n_observations']}"
    
    print(f"✓ Insufficient data test passed! Failure reason: {result['failure_reason']}")


def test_compute_beta_with_nans():
    """Test beta computation with NaN values in data."""
    print("\nTesting compute_beta with NaN values...")
    
    # Create return series with some NaN values
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    wave_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
    
    # Introduce some NaN values
    wave_returns.iloc[10:15] = np.nan
    benchmark_returns.iloc[20:25] = np.nan
    
    # Compute beta
    result = compute_beta(wave_returns, benchmark_returns, min_n=60)
    
    # Verify result - should succeed with fewer observations
    assert result['success'] == True, "Beta computation should succeed after dropping NaNs"
    assert result['beta'] is not None, "Beta should not be None"
    # We should have 100 - 5 (wave NaNs) - 5 (benchmark NaNs) = 90 observations
    assert result['n_observations'] == 90, f"Expected 90 observations after dropping NaNs, got {result['n_observations']}"
    
    print(f"✓ NaN handling test passed! Beta={result['beta']:.3f} with {result['n_observations']} observations")


def test_compute_beta_alignment():
    """Test beta computation with misaligned time series."""
    print("\nTesting compute_beta with misaligned time series...")
    
    # Create return series with different date ranges
    dates_wave = pd.date_range('2024-01-01', periods=100, freq='D')
    dates_benchmark = pd.date_range('2024-01-11', periods=100, freq='D')  # Starts 10 days later
    
    wave_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates_wave)
    benchmark_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates_benchmark)
    
    # Compute beta
    result = compute_beta(wave_returns, benchmark_returns, min_n=60)
    
    # Verify result - should succeed with overlapping dates only
    assert result['success'] == True, "Beta computation should succeed with aligned overlap"
    # Overlap should be from 2024-01-11 to 2024-04-09 (90 days)
    assert result['n_observations'] == 90, f"Expected 90 overlapping observations, got {result['n_observations']}"
    assert result['beta'] is not None, "Beta should not be None"
    
    print(f"✓ Alignment test passed! Beta={result['beta']:.3f} with {result['n_observations']} overlapping observations")


def test_compute_beta_zero_variance():
    """Test beta computation with zero variance benchmark."""
    print("\nTesting compute_beta with zero variance benchmark...")
    
    # Create return series where benchmark has zero variance
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    wave_returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)
    benchmark_returns = pd.Series(np.ones(100) * 0.001, index=dates)  # Constant returns
    
    # Compute beta
    result = compute_beta(wave_returns, benchmark_returns, min_n=60)
    
    # Verify failure
    assert result['success'] == False, "Beta computation should fail with zero variance benchmark"
    assert 'variance is zero' in result['failure_reason'].lower(), f"Unexpected failure reason: {result['failure_reason']}"
    
    print(f"✓ Zero variance test passed! Failure reason: {result['failure_reason']}")


def run_all_tests():
    """Run all beta computation tests."""
    print("=" * 60)
    print("Running Beta Computation Tests")
    print("=" * 60)
    
    test_compute_beta_basic()
    test_compute_beta_insufficient_data()
    test_compute_beta_with_nans()
    test_compute_beta_alignment()
    test_compute_beta_zero_variance()
    
    print("\n" + "=" * 60)
    print("All Beta Computation Tests Passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

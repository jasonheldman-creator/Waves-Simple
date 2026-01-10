"""
Integration tests for Phase 1B: Dynamic Benchmarks

Tests that:
1. All waves load dynamic benchmarks successfully
2. Benchmark series compute without error
3. Benchmark series align with wave return series (same last date)
4. No benchmarks are all-NaN
5. S&P 500 Wave remains static and unaffected
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add parent directory to path to import waves_engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waves_engine import (
    compute_history_nav,
    load_dynamic_benchmark_specs,
    build_benchmark_series_from_components,
    get_wave_id_from_display_name,
)

# The 10 equity waves that should have dynamic benchmarks
DYNAMIC_BENCHMARK_WAVES = [
    "Clean Transit-Infrastructure Wave",
    "Demas Fund Wave",
    "EV & Infrastructure Wave",
    "Future Power & Energy Wave",
    "Infinity Multi-Asset Growth Wave",
    "Next-Gen Compute & Semis Wave",
    "Quantum Computing Wave",
    "Small to Mid Cap Growth Wave",
    "US MegaCap Core Wave",
    "AI & Cloud MegaCap Wave",
]

# S&P 500 Wave should remain static
SP500_WAVE = "S&P 500 Wave"


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
def benchmark_specs():
    """Fixture to load benchmark specs once for all tests."""
    return load_dynamic_benchmark_specs()


def test_benchmark_specs_loaded(benchmark_specs):
    """Test that benchmark specs file loads successfully."""
    assert benchmark_specs is not None, "Benchmark specs should load"
    assert "benchmarks" in benchmark_specs, "Specs should have benchmarks key"
    assert "version" in benchmark_specs, "Specs should have version"
    benchmark_count = len(benchmark_specs["benchmarks"])
    assert benchmark_count > 0, f"Expected at least 1 dynamic benchmark, found {benchmark_count}"


def test_sp500_excluded_from_dynamic_benchmarks(benchmark_specs):
    """Test that S&P 500 Wave is excluded from dynamic benchmarks."""
    wave_id = get_wave_id_from_display_name(SP500_WAVE)
    assert wave_id == "sp500_wave", "S&P 500 Wave should have correct wave_id"
    assert wave_id not in benchmark_specs["benchmarks"], \
        "S&P 500 Wave should not be in dynamic benchmarks"


@pytest.mark.parametrize("wave_name", DYNAMIC_BENCHMARK_WAVES)
def test_dynamic_benchmark_defined(wave_name, benchmark_specs):
    """Test that each equity wave has a dynamic benchmark defined."""
    wave_id = get_wave_id_from_display_name(wave_name)
    assert wave_id in benchmark_specs["benchmarks"], \
        f"{wave_name} ({wave_id}) should have dynamic benchmark"
    
    benchmark = benchmark_specs["benchmarks"][wave_id]
    assert "components" in benchmark, f"{wave_name} benchmark should have components"
    assert len(benchmark["components"]) > 0, f"{wave_name} should have at least one component"
    
    # Validate component structure
    for comp in benchmark["components"]:
        assert "ticker" in comp, "Component should have ticker"
        assert "weight" in comp, "Component should have weight"
        assert 0 <= comp["weight"] <= 1, "Component weight should be between 0 and 1"


@pytest.mark.parametrize("wave_name", DYNAMIC_BENCHMARK_WAVES)
def test_wave_computes_with_dynamic_benchmark(wave_name, price_df):
    """Test that wave computes successfully with dynamic benchmark."""
    if price_df is None:
        pytest.skip("Price cache not available")
    
    # Compute wave with price cache
    result = compute_history_nav(
        wave_name=wave_name,
        mode="Standard",
        days=90,
        include_diagnostics=False,
        price_df=price_df
    )
    
    assert not result.empty, f"{wave_name} should produce non-empty result"
    assert "wave_nav" in result.columns, "Result should have wave_nav column"
    assert "bm_nav" in result.columns, "Result should have bm_nav column"
    assert "wave_ret" in result.columns, "Result should have wave_ret column"
    assert "bm_ret" in result.columns, "Result should have bm_ret column"


@pytest.mark.parametrize("wave_name", DYNAMIC_BENCHMARK_WAVES)
def test_dynamic_benchmark_series_not_all_nan(wave_name, price_df):
    """Test that dynamic benchmark series is not all NaN."""
    if price_df is None:
        pytest.skip("Price cache not available")
    
    result = compute_history_nav(
        wave_name=wave_name,
        mode="Standard",
        days=90,
        include_diagnostics=False,
        price_df=price_df
    )
    
    assert not result.empty, f"{wave_name} should produce non-empty result"
    bm_ret = result["bm_ret"]
    
    # Check that benchmark returns are not all NaN
    non_nan_count = bm_ret.notna().sum()
    assert non_nan_count > 0, f"{wave_name} benchmark should have non-NaN returns"
    
    # Most returns should be non-NaN (at least 80%)
    pct_non_nan = non_nan_count / len(bm_ret) * 100
    assert pct_non_nan > 80, \
        f"{wave_name} benchmark should have >80% non-NaN returns, got {pct_non_nan:.1f}%"


@pytest.mark.parametrize("wave_name", DYNAMIC_BENCHMARK_WAVES)
def test_benchmark_series_alignment(wave_name, price_df):
    """Test that benchmark series aligns with wave return series."""
    if price_df is None:
        pytest.skip("Price cache not available")
    
    result = compute_history_nav(
        wave_name=wave_name,
        mode="Standard",
        days=90,
        include_diagnostics=False,
        price_df=price_df
    )
    
    assert not result.empty, f"{wave_name} should produce non-empty result"
    
    # Check that wave and benchmark series have same index
    assert len(result.index) > 0, "Result should have dates"
    
    wave_last_date = result["wave_ret"].dropna().index[-1] if result["wave_ret"].notna().any() else None
    bm_last_date = result["bm_ret"].dropna().index[-1] if result["bm_ret"].notna().any() else None
    
    if wave_last_date and bm_last_date:
        # Allow small difference (within 5 business days)
        date_diff = abs((wave_last_date - bm_last_date).days)
        assert date_diff <= 7, \
            f"{wave_name}: wave and benchmark last dates should align (diff: {date_diff} days)"


@pytest.mark.parametrize("wave_name", DYNAMIC_BENCHMARK_WAVES)
def test_dynamic_benchmark_diagnostics(wave_name, price_df):
    """Test that dynamic benchmark diagnostics are included in result."""
    if price_df is None:
        pytest.skip("Price cache not available")
    
    result = compute_history_nav(
        wave_name=wave_name,
        mode="Standard",
        days=90,
        include_diagnostics=False,
        price_df=price_df
    )
    
    assert "coverage" in result.attrs, "Result should have coverage attrs"
    coverage = result.attrs["coverage"]
    
    assert "dynamic_benchmark" in coverage, "Coverage should have dynamic_benchmark info"
    dynamic_info = coverage["dynamic_benchmark"]
    
    # For equity waves with dynamic benchmarks, enabled should be True
    wave_id = get_wave_id_from_display_name(wave_name)
    if wave_id != "sp500_wave":
        assert dynamic_info.get("enabled") is True, \
            f"{wave_name} should have dynamic benchmark enabled"
        assert "benchmark_name" in dynamic_info, "Should have benchmark name"
        assert "version" in dynamic_info, "Should have version"
        assert "components" in dynamic_info, "Should have components"
        
        # Check component availability
        components = dynamic_info["components"]
        assert len(components) > 0, "Should have components"
        for comp in components:
            assert "ticker" in comp, "Component should have ticker"
            assert "weight" in comp, "Component should have weight"
            assert "available" in comp, "Component should have availability status"


def test_sp500_wave_uses_static_benchmark(price_df):
    """Test that S&P 500 Wave uses static SPY benchmark (not dynamic)."""
    if price_df is None:
        pytest.skip("Price cache not available")
    
    result = compute_history_nav(
        wave_name=SP500_WAVE,
        mode="Standard",
        days=90,
        include_diagnostics=False,
        price_df=price_df
    )
    
    assert not result.empty, "S&P 500 Wave should produce non-empty result"
    assert "coverage" in result.attrs, "Result should have coverage attrs"
    
    coverage = result.attrs["coverage"]
    assert "dynamic_benchmark" in coverage, "Coverage should have dynamic_benchmark info"
    
    dynamic_info = coverage["dynamic_benchmark"]
    assert dynamic_info.get("enabled") is False, \
        "S&P 500 Wave should NOT use dynamic benchmark"
    assert "sp500" in dynamic_info.get("reason", "").lower() or \
           "excluded" in dynamic_info.get("reason", "").lower(), \
        "Reason should indicate S&P 500 exclusion"


def test_build_benchmark_series_from_components(price_df, benchmark_specs):
    """Test the build_benchmark_series_from_components function."""
    if price_df is None:
        pytest.skip("Price cache not available")
    
    # Get a sample benchmark
    wave_id = get_wave_id_from_display_name("AI & Cloud MegaCap Wave")
    components = benchmark_specs["benchmarks"][wave_id]["components"]
    
    # Build benchmark series
    benchmark_series = build_benchmark_series_from_components(price_df, components)
    
    assert not benchmark_series.empty, "Benchmark series should not be empty"
    assert len(benchmark_series) > 0, "Should have data points"
    
    # Check that series is not all NaN
    assert benchmark_series.notna().sum() > 0, "Should have non-NaN values"
    
    # Check that series is aligned with price_df
    assert len(benchmark_series) == len(price_df), "Series should match price_df length"


def test_benchmark_weights_sum_to_one(benchmark_specs):
    """Test that all benchmark component weights sum to 1.0."""
    tolerance = 0.01
    
    for wave_id, benchmark in benchmark_specs["benchmarks"].items():
        components = benchmark["components"]
        total_weight = sum(c["weight"] for c in components)
        
        assert abs(total_weight - 1.0) < tolerance, \
            f"{wave_id}: weights sum to {total_weight}, expected 1.0"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

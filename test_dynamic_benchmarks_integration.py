"""
Integration tests for Phase 1B: Dynamic Benchmarks

Tests that:
1. Dynamic benchmark specs load successfully
2. Specs structure is valid (keys, components, weights)
3. Benchmarks can be constructed from components (smoke test)
4. No benchmark definitions are empty / malformed
5. S&P 500 Wave remains static and excluded from dynamic benchmarks
"""

import os
import sys
import pytest
import pandas as pd

# Add repo root to path so we can import waves_engine when tests run from repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from waves_engine import (
    load_dynamic_benchmark_specs,
    build_benchmark_series_from_components,
    get_wave_id_from_display_name,
)

SP500_WAVE = "S&P 500 Wave"


def _load_price_cache():
    """
    Load cached price data for testing.
    Note: In CI, parquet should exist if cache workflow has run; if not, we skip series-build tests.
    """
    cache_path = os.path.join(REPO_ROOT, "data", "cache", "prices_cache.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    return None


@pytest.fixture(scope="module")
def price_df():
    return _load_price_cache()


@pytest.fixture(scope="module")
def benchmark_specs():
    return load_dynamic_benchmark_specs()


def test_benchmark_specs_loaded(benchmark_specs):
    """Specs must load and contain the expected top-level keys."""
    assert benchmark_specs is not None, "Benchmark specs should load"
    assert isinstance(benchmark_specs, dict), "Benchmark specs should be a dict"
    assert "benchmarks" in benchmark_specs, "Specs should have 'benchmarks' key"
    assert "version" in benchmark_specs, "Specs should have 'version' key"

    # No hard-coded count: repo can add/remove equity waves over time.
    benchmark_count = len(benchmark_specs["benchmarks"])
    assert benchmark_count > 0, f"Expected at least 1 dynamic benchmark, found {benchmark_count}"


def test_sp500_excluded_from_dynamic_benchmarks(benchmark_specs):
    """S&P 500 Wave should remain static and not appear in dynamic benchmarks."""
    wave_id = get_wave_id_from_display_name(SP500_WAVE)
    assert wave_id == "sp500_wave", "S&P 500 Wave should map to wave_id 'sp500_wave'"
    assert wave_id not in benchmark_specs["benchmarks"], "S&P 500 Wave must be excluded from dynamic benchmarks"


def test_all_benchmarks_have_valid_structure(benchmark_specs):
    """
    Every benchmark entry must have:
      - components: non-empty list
      - each component: ticker (str), weight (float in [0,1])
      - weights sum ~ 1.0 (tolerance)
    """
    benchmarks = benchmark_specs["benchmarks"]
    assert isinstance(benchmarks, dict), "'benchmarks' should be a dict of wave_id -> benchmark spec"

    tolerance = 0.01

    for wave_id, bench in benchmarks.items():
        assert isinstance(wave_id, str) and wave_id, "wave_id keys must be non-empty strings"
        assert isinstance(bench, dict), f"{wave_id}: benchmark spec must be a dict"
        assert "components" in bench, f"{wave_id}: benchmark spec missing 'components'"
        components = bench["components"]
        assert isinstance(components, list), f"{wave_id}: components must be a list"
        assert len(components) > 0, f"{wave_id}: components must be non-empty"

        total_weight = 0.0
        for comp in components:
            assert isinstance(comp, dict), f"{wave_id}: each component must be a dict"
            assert "ticker" in comp, f"{wave_id}: component missing 'ticker'"
            assert "weight" in comp, f"{wave_id}: component missing 'weight'"

            ticker = comp["ticker"]
            weight = comp["weight"]

            assert isinstance(ticker, str) and ticker.strip(), f"{wave_id}: ticker must be non-empty string"
            assert isinstance(weight, (int, float)), f"{wave_id}: weight must be numeric"
            assert 0.0 <= float(weight) <= 1.0, f"{wave_id}: weight must be in [0,1], got {weight}"

            total_weight += float(weight)

        assert abs(total_weight - 1.0) < tolerance, (
            f"{wave_id}: component weights must sum to 1.0 (+/- {tolerance}); got {total_weight}"
        )


def test_can_build_benchmark_series_from_components_smoke(price_df, benchmark_specs):
    """
    Smoke test: if we have a price cache, we should be able to build a benchmark series
    for at least one benchmark definition without errors and with non-NaN values.
    """
    if price_df is None:
        pytest.skip("Price cache not available in this environment")

    benchmarks = benchmark_specs["benchmarks"]
    # pick the first benchmark in the dict for a smoke test
    sample_wave_id = next(iter(benchmarks.keys()))
    components = benchmarks[sample_wave_id]["components"]

    series = build_benchmark_series_from_components(price_df, components)

    assert series is not None, "Benchmark series should not be None"
    assert len(series) > 0, "Benchmark series should have data points"
    assert series.notna().sum() > 0, "Benchmark series should have at least some non-NaN values"
    assert len(series) == len(price_df), "Benchmark series should align to price_df index length"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
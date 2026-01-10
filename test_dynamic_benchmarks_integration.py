"""
Integration tests for Phase 1B: Dynamic Benchmarks

Tests that:
1. Dynamic benchmark specs load successfully
2. Benchmark series compute without error (for every defined benchmark)
3. Benchmark series align with the price cache index (same dates/length)
4. No benchmark series is all-NaN
5. S&P 500 Wave remains static/excluded from dynamic benchmarks
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# -----------------------------
# Repo path helpers
# -----------------------------

def _find_repo_root(start_dir: str, max_up: int = 6) -> str:
    """
    Walk upward until we find data/cache/prices_cache.parquet (repo root indicator).
    Falls back to start_dir if not found.
    """
    cur = os.path.abspath(start_dir)
    for _ in range(max_up):
        candidate = os.path.join(cur, "data", "cache", "prices_cache.parquet")
        if os.path.exists(candidate):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_dir)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = _find_repo_root(THIS_DIR)

# Ensure we can import waves_engine from repo root
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


from waves_engine import (  # noqa: E402
    compute_history_nav,
    load_dynamic_benchmark_specs,
    build_benchmark_series_from_components,
    get_wave_id_from_display_name,
)


# -----------------------------
# Data loaders
# -----------------------------

def _load_price_cache() -> pd.DataFrame | None:
    cache_path = os.path.join(REPO_ROOT, "data", "cache", "prices_cache.parquet")
    if not os.path.exists(cache_path):
        return None
    return pd.read_parquet(cache_path)


@pytest.fixture(scope="module")
def price_df():
    return _load_price_cache()


@pytest.fixture(scope="module")
def benchmark_specs():
    return load_dynamic_benchmark_specs()


# -----------------------------
# Core tests
# -----------------------------

def test_benchmark_specs_loaded(benchmark_specs):
    assert benchmark_specs is not None, "Benchmark specs should load"
    assert isinstance(benchmark_specs, dict), "Benchmark specs must be a dict"
    assert "benchmarks" in benchmark_specs, "Specs should have 'benchmarks' key"
    assert "version" in benchmark_specs, "Specs should have 'version' key"

    # No hard-coded counts. Registry may evolve (10 -> 14 -> etc).
    benchmark_count = len(benchmark_specs["benchmarks"])
    assert benchmark_count > 0, f"Expected at least 1 dynamic benchmark, found {benchmark_count}"
    # If you still want a minimum expectation for Phase 1B, keep it flexible:
    assert benchmark_count >= 10, f"Expected at least 10 dynamic benchmarks for Phase 1B, found {benchmark_count}"


def test_sp500_excluded_from_dynamic_benchmarks(benchmark_specs):
    sp500_wave_id = get_wave_id_from_display_name("S&P 500 Wave")
    assert sp500_wave_id == "sp500_wave", "S&P 500 Wave should map to wave_id 'sp500_wave'"
    assert sp500_wave_id not in benchmark_specs["benchmarks"], "S&P 500 Wave must NOT be in dynamic benchmarks"


def _validate_components_structure(components, wave_id: str):
    assert isinstance(components, list), f"{wave_id}: components must be a list"
    assert len(components) > 0, f"{wave_id}: components must be non-empty"

    total_weight = 0.0
    for i, comp in enumerate(components):
        assert isinstance(comp, dict), f"{wave_id}: component[{i}] must be a dict"
        assert "ticker" in comp, f"{wave_id}: component[{i}] missing 'ticker'"
        assert "weight" in comp, f"{wave_id}: component[{i}] missing 'weight'"

        ticker = comp["ticker"]
        weight = comp["weight"]

        assert isinstance(ticker, str) and ticker.strip(), f"{wave_id}: component[{i}] invalid ticker"
        assert isinstance(weight, (int, float)), f"{wave_id}: component[{i}] weight must be numeric"
        assert 0.0 <= float(weight) <= 1.0, f"{wave_id}: component[{i}] weight out of range: {weight}"
        total_weight += float(weight)

    # Weights should sum to ~1.0
    assert abs(total_weight - 1.0) <= 0.01, f"{wave_id}: weights sum to {total_weight:.6f}, expected ~1.0"


def test_all_dynamic_benchmarks_have_valid_components(benchmark_specs):
    for wave_id, spec in benchmark_specs["benchmarks"].items():
        assert isinstance(spec, dict), f"{wave_id}: benchmark spec must be a dict"
        assert "components" in spec, f"{wave_id}: benchmark spec missing 'components'"
        _validate_components_structure(spec["components"], wave_id)


def test_build_benchmark_series_for_all_defined_benchmarks(price_df, benchmark_specs):
    if price_df is None:
        pytest.skip("Price cache not available (data/cache/prices_cache.parquet missing)")

    assert isinstance(price_df, pd.DataFrame) and not price_df.empty, "Price cache must be a non-empty DataFrame"

    # For each benchmark, ensure we can build a non-empty, non-all-NaN series aligned to price_df.
    for wave_id, spec in benchmark_specs["benchmarks"].items():
        components = spec["components"]

        series = build_benchmark_series_from_components(price_df, components)

        assert isinstance(series, (pd.Series, pd.DataFrame)), f"{wave_id}: benchmark output must be Series/DataFrame"
        if isinstance(series, pd.DataFrame):
            # If engine ever returns a 1-col df, convert to series
            assert series.shape[1] == 1, f"{wave_id}: expected 1-column DataFrame from benchmark builder"
            series = series.iloc[:, 0]

        assert len(series) == len(price_df), f"{wave_id}: benchmark series length must match price_df length"
        assert series.index.equals(price_df.index), f"{wave_id}: benchmark series index must match price_df index"
        assert series.notna().sum() > 0, f"{wave_id}: benchmark series is all-NaN"

        # Light sanity: returns should not be all zeros unless truly flat (rare)
        # (don’t make this too strict)
        values = series.dropna().values
        assert np.isfinite(values).all(), f"{wave_id}: benchmark series contains non-finite values"


def test_smoke_compute_history_nav_equity_wave(price_df):
    """
    Smoke test: one known equity wave should compute nav & benchmark without error.
    (Keeps runtime reasonable.)
    """
    if price_df is None:
        pytest.skip("Price cache not available (data/cache/prices_cache.parquet missing)")

    wave_name = "AI & Cloud MegaCap Wave"
    result = compute_history_nav(
        wave_name=wave_name,
        mode="Standard",
        days=90,
        include_diagnostics=False,
        price_df=price_df,
    )

    assert isinstance(result, pd.DataFrame) and not result.empty, f"{wave_name}: expected non-empty DataFrame"
    for col in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
        assert col in result.columns, f"{wave_name}: missing expected column '{col}'"
    assert result["bm_ret"].notna().sum() > 0, f"{wave_name}: bm_ret should not be all-NaN"


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))

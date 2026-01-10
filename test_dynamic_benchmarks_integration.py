"""
Integration tests for Phase 1B: Dynamic Benchmarks

Tests that:
1. All waves load dynamic benchmarks successfully
2. Benchmark series compute without error
3. Benchmark series align with wave return series (same last date)
4. No benchmarks are all-NaN
5. S&P 500 Wave remains static and unaffected
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pytest
import pandas as pd

# --------------------------------------------------------------------------------------
# Path setup (works whether this file lives in ./tests/ or repo root)
# --------------------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name == "tests" else THIS_FILE.parent

# Ensure repo root is importable (waves_engine.py typically lives at repo root)
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from waves_engine import (  # noqa: E402
    compute_history_nav,
    load_dynamic_benchmark_specs,
    build_benchmark_series_from_components,
    get_wave_id_from_display_name,
)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

SP500_WAVE = "S&P 500 Wave"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _load_wave_registry() -> Dict[str, Any]:
    """
    Load wave registry for wave_id <-> display_name mapping.

    Try a couple common locations:
      - config/wave_registry.json
      - config/waves_registry.json (fallback, in case naming differs)
    """
    candidates = [
        REPO_ROOT / "config" / "wave_registry.json",
        REPO_ROOT / "config" / "waves_registry.json",
    ]
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _wave_id_to_display_name_map(registry: Dict[str, Any]) -> Dict[str, str]:
    """
    Build a wave_id -> display_name mapping from the registry.
    Supports common registry shapes:
      - {"waves": [{...}, {...}]}
      - [{"wave_id":..., "display_name":...}, ...]
      - {"<wave_id>": {"display_name": ...}, ...}
    """
    mapping: Dict[str, str] = {}

    if not registry:
        return mapping

    # Case 1: {"waves": [ ... ]}
    if isinstance(registry, dict) and isinstance(registry.get("waves"), list):
        for w in registry["waves"]:
            if isinstance(w, dict):
                wid = w.get("wave_id") or w.get("id")
                dn = w.get("display_name") or w.get("name")
                if wid and dn:
                    mapping[str(wid)] = str(dn)
        return mapping

    # Case 2: list of dicts
    if isinstance(registry, list):
        for w in registry:
            if isinstance(w, dict):
                wid = w.get("wave_id") or w.get("id")
                dn = w.get("display_name") or w.get("name")
                if wid and dn:
                    mapping[str(wid)] = str(dn)
        return mapping

    # Case 3: dict keyed by wave_id
    if isinstance(registry, dict):
        for k, v in registry.items():
            if isinstance(v, dict):
                dn = v.get("display_name") or v.get("name")
                if dn:
                    mapping[str(k)] = str(dn)

    return mapping


def _load_dynamic_benchmark_waves_display_names() -> List[str]:
    """
    Derive the list of *display names* for waves that have dynamic benchmarks,
    from the benchmark specs registry + wave registry.

    This avoids hardcoding counts or a manual map.
    """
    specs = load_dynamic_benchmark_specs()
    if not specs or "benchmarks" not in specs or not isinstance(specs["benchmarks"], dict):
        return []

    registry = _load_wave_registry()
    id_to_name = _wave_id_to_display_name_map(registry)

    wave_ids = list(specs["benchmarks"].keys())
    display_names: List[str] = []
    for wid in wave_ids:
        display_names.append(id_to_name.get(wid, wid))  # fallback to wid if registry missing

    return display_names


def load_price_cache() -> Optional[pd.DataFrame]:
    """
    Load cached price data for testing.

    Primary path:
      data/cache/prices_cache.parquet
    """
    cache_path = REPO_ROOT / "data" / "cache" / "prices_cache.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return None


# Dynamically derived list
DYNAMIC_BENCHMARK_WAVES = _load_dynamic_benchmark_waves_display_names()


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def price_df() -> Optional[pd.DataFrame]:
    return load_price_cache()


@pytest.fixture(scope="module")
def benchmark_specs() -> Dict[str, Any]:
    return load_dynamic_benchmark_specs() or {}


@pytest.fixture(scope="module")
def wave_registry() -> Dict[str, Any]:
    return _load_wave_registry()


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_benchmark_specs_loaded(benchmark_specs: Dict[str, Any], wave_registry: Dict[str, Any]) -> None:
    """Test that benchmark specs file loads successfully and has expected top-level keys."""
    assert benchmark_specs is not None, "Benchmark specs should load"
    assert "benchmarks" in benchmark_specs, "Specs should have 'benchmarks' key"
    assert "version" in benchmark_specs, "Specs should have 'version' key"

    benchmark_count = len(benchmark_specs["benchmarks"])
    assert benchmark_count > 0, f"Should have at least one dynamic benchmark, found {benchmark_count}"

    # Registry isn't strictly required for the core behavior, but if present, it should be parseable
    assert isinstance(wave_registry, (dict, list)), "Wave registry should be dict or list if present"


def test_sp500_excluded_from_dynamic_benchmarks(benchmark_specs: Dict[str, Any]) -> None:
    """Test that S&P 500 Wave is excluded from dynamic benchmarks."""
    wave_id = get_wave_id_from_display_name(SP500_WAVE)
    assert wave_id == "sp500_wave", "S&P 500 Wave should have wave_id 'sp500_wave'"
    assert wave_id not in benchmark_specs["benchmarks"], "S&P 500 Wave should not be in dynamic benchmarks"


@pytest.mark.parametrize("wave_name", DYNAMIC_BENCHMARK_WAVES)
def test_dynamic_benchmark_defined(wave_name: str, benchmark_specs: Dict[str, Any]) -> None:
    """Test that each dynamic-benchmark wave has a benchmark defined with valid component structure."""
    wave_id = get_wave_id_from_display_name(wave_name)
    assert wave_id in benchmark_specs["benchmarks"], f"{wave_name} ({wave_id}) should have a dynamic benchmark"

    benchmark = benchmark_specs["benchmarks"][wave_id]
    assert "components" in benchmark, f"{wave_name} benchmark should have components"
    assert isinstance(benchmark["components"], list), f"{wave_name} components should be a list"
    assert len(benchmark["components"]) > 0, f"{wave_name} should have at least one component"

    for comp in benchmark["components"]:
        assert "ticker" in comp, "Component should have ticker"
        assert "weight" in comp, "Component should have weight"
        assert 0 <= float(comp["weight"]) <= 1, "Component weight should be between 0 and 1"


@pytest.mark.parametrize("wave_name", DYNAMIC_BENCHMARK_WAVES)
def test_wave_computes_with_dynamic_benchmark(wave_name: str, price_df: Optional[pd.DataFrame]) -> None:
    """Test that wave computes successfully with dynamic benchmark."""
    if price_df is None:
        pytest.skip("Price cache not available")

    result = compute_history_nav(
        wave_name=wave_name,
        mode="Standard",
        days=90,
        include_diagnostics=False,
        price_df=price_df,
    )

    assert result is not None and not result.empty, f"{wave_name} should produce non-empty result"
    for col in ["wave_nav", "bm_nav", "wave_ret", "bm_ret"]:
        assert col in result.columns, f"Result should have '{col}' column"


@pytest.mark.parametrize("wave_name", DYNAMIC_BENCHMARK_WAVES)
def test_dynamic_benchmark_series_not_all_nan(wave_name: str, price_df: Optional[pd.DataFrame]) -> None:
    """Test that dynamic benchmark return series is not all NaN."""
    if price_df is None:
        pytest.skip("Price cache not available")

    result = compute_history_nav(
        wave_name=wave_name,
        mode="Standard",
        days=90,
        include_diagnostics=False,
        price_df=price_df,
    )

    assert not result.empty, f"{wave_name} should produce non-empty result"
    bm_ret = result["bm_ret"]

    non_nan_count = int(bm_ret.notna().sum())
    assert non_nan_count > 0, f"{wave_name} benchmark should have non-NaN returns"

    pct_non_nan = non_nan_count / len(bm_ret) * 100
    assert pct_non_nan > 80, f"{wave_name} benchmark should have >80% non-NaN returns, got {pct_non_nan:.1f}%"


@pytest.mark.parametrize("wave_name", DYNAMIC_BENCHMARK_WAVES)
def test_benchmark_series_alignment(wave_name: str, price_df: Optional[pd.DataFrame]) -> None:
    """Test that benchmark series aligns with wave return series (last available date close)."""
    if price_df is None:
        pytest.skip("Price cache not available")

    result = compute_history_nav(
        wave_name=wave_name,
        mode="Standard",
        days=90,
        include_diagnostics=False,
        price_df=price_df,
    )

    assert not result.empty, f"{wave_name} should produce non-empty result"
    assert len(result.index) > 0, "Result should have dates"

    wave_last_date = result["wave_ret"].dropna().index[-1] if result["wave_ret"].notna().any() else None
    bm_last_date = result["bm_ret"].dropna().index[-1] if result["bm_ret"].notna().any() else None

    if wave_last_date is not None and bm_last_date is not None:
        date_diff = abs((wave_last_date - bm_last_date).days)
        assert date_diff <= 7, f"{wave_name}: last dates should align (diff: {date_diff} days)"


@pytest.mark.parametrize("wave_name", DYNAMIC_BENCHMARK_WAVES)
def test_dynamic_benchmark_diagnostics(wave_name: str, price_df: Optional[pd.DataFrame]) -> None:
    """Test that dynamic benchmark diagnostics are present and show enabled=True for dynamic-benchmark waves."""
    if price_df is None:
        pytest.skip("Price cache not available")

    result = compute_history_nav(
        wave_name=wave_name,
        mode="Standard",
        days=90,
        include_diagnostics=False,
        price_df=price_df,
    )

    assert "coverage" in result.attrs, "Result should have coverage attrs"
    coverage = result.attrs["coverage"]

    assert "dynamic_benchmark" in coverage, "Coverage should include dynamic_benchmark info"
    dynamic_info = coverage["dynamic_benchmark"]

    wave_id = get_wave_id_from_display_name(wave_name)
    assert wave_id != "sp500_wave", "Parametrized dynamic benchmark wave list should not include SP500"

    assert dynamic_info.get("enabled") is True, f"{wave_name} should have dynamic benchmark enabled"
    for k in ["benchmark_name", "version", "components"]:
        assert k in dynamic_info, f"{wave_name} dynamic_benchmark should include '{k}'"

    components = dynamic_info["components"]
    assert isinstance(components, list) and len(components) > 0, "Should have components in diagnostics"
    for comp in components:
        assert "ticker" in comp
        assert "weight" in comp
        assert "available" in comp


def test_sp500_wave_uses_static_benchmark(price_df: Optional[pd.DataFrame]) -> None:
    """Test that S&P 500 Wave uses static SPY benchmark (not dynamic)."""
    if price_df is None:
        pytest.skip("Price cache not available")

    result = compute_history_nav(
        wave_name=SP500_WAVE,
        mode="Standard",
        days=90,
        include_diagnostics=False,
        price_df=price_df,
    )

    assert not result.empty, "S&P 500 Wave should produce non-empty result"
    assert "coverage" in result.attrs, "Result should have coverage attrs"

    coverage = result.attrs["coverage"]
    assert "dynamic_benchmark" in coverage, "Coverage should include dynamic_benchmark info"

    dynamic_info = coverage["dynamic_benchmark"]
    assert dynamic_info.get("enabled") is False, "S&P 500 Wave should NOT use dynamic benchmark"

    reason = (dynamic_info.get("reason") or "").lower()
    assert ("sp500" in reason) or ("excluded" in reason), "Reason should indicate S&P 500 exclusion"


def test_build_benchmark_series_from_components(price_df: Optional[pd.DataFrame], benchmark_specs: Dict[str, Any]) -> None:
    """Test build_benchmark_series_from_components produces a non-empty, aligned series for a sample benchmark."""
    if price_df is None:
        pytest.skip("Price cache not available")

    # Choose a stable sample: first benchmark in the registry
    benchmarks = benchmark_specs.get("benchmarks", {})
    assert isinstance(benchmarks, dict) and len(benchmarks) > 0, "Benchmarks should be a non-empty dict"

    sample_wave_id = next(iter(benchmarks.keys()))
    components = benchmarks[sample_wave_id]["components"]

    benchmark_series = build_benchmark_series_from_components(price_df, components)

    assert benchmark_series is not None and not benchmark_series.empty, "Benchmark series should not be empty"
    assert benchmark_series.notna().sum() > 0, "Benchmark series should have non-NaN values"
    assert len(benchmark_series) == len(price_df), "Series should match price_df length"


def test_benchmark_weights_sum_to_one(benchmark_specs: Dict[str, Any]) -> None:
    """Test all dynamic benchmark component weights sum to 1.0 (within tolerance)."""
    tolerance = 0.01
    benchmarks = benchmark_specs.get("benchmarks", {})
    assert isinstance(benchmarks, dict), "benchmarks should be a dict"

    for wave_id, benchmark in benchmarks.items():
        components = benchmark.get("components", [])
        total_weight = sum(float(c["weight"]) for c in components)
        assert abs(total_weight - 1.0) < tolerance, f"{wave_id}: weights sum to {total_weight}, expected 1.0"


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([str(THIS_FILE), "-v"]))
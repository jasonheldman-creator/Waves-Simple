"""
Integration test for 9 Equity Waves

This test verifies that the complete system works correctly with the 9 equity waves:
1. Ticker discovery collects all wave tickers
2. Wave registry is valid
3. Positions files exist and are valid
4. Benchmarks are properly defined
5. Wave engine can load all waves
"""

import os
import sys
import pytest
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waves_engine import (
    get_all_wave_ids,
    WAVE_WEIGHTS,
    WAVE_ID_REGISTRY,
    DISPLAY_NAME_TO_WAVE_ID
)
from analytics_pipeline import resolve_wave_tickers, resolve_wave_benchmarks

# Import wave_registry directly to avoid streamlit dependency in helpers/__init__.py
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "wave_registry",
    os.path.join(os.path.dirname(__file__), "helpers", "wave_registry.py")
)
_wave_registry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wave_registry)
get_wave_registry = _wave_registry.get_wave_registry
get_active_waves = _wave_registry.get_active_waves


# The 9 equity waves to test
EQUITY_WAVES = [
    "clean_transit_infrastructure_wave",
    "demas_fund_wave",
    "ev_infrastructure_wave",
    "future_power_energy_wave",
    "infinity_multi_asset_growth_wave",
    "next_gen_compute_semis_wave",
    "quantum_computing_wave",
    "small_to_mid_cap_growth_wave",
    "us_megacap_core_wave",
]

EXPECTED_DISPLAY_NAMES = {
    "clean_transit_infrastructure_wave": "Clean Transit-Infrastructure Wave",
    "demas_fund_wave": "Demas Fund Wave",
    "ev_infrastructure_wave": "EV & Infrastructure Wave",
    "future_power_energy_wave": "Future Power & Energy Wave",
    "infinity_multi_asset_growth_wave": "Infinity Multi-Asset Growth Wave",
    "next_gen_compute_semis_wave": "Next-Gen Compute & Semis Wave",
    "quantum_computing_wave": "Quantum Computing Wave",
    "small_to_mid_cap_growth_wave": "Small to Mid Cap Growth Wave",
    "us_megacap_core_wave": "US MegaCap Core Wave",
}


def test_all_equity_waves_in_registry():
    """Test that all 9 equity waves are present in the wave ID registry."""
    for wave_id in EQUITY_WAVES:
        assert wave_id in WAVE_ID_REGISTRY, f"Wave {wave_id} not found in WAVE_ID_REGISTRY"
        
        # Check display name matches
        display_name = WAVE_ID_REGISTRY[wave_id]
        expected_name = EXPECTED_DISPLAY_NAMES[wave_id]
        assert display_name == expected_name, (
            f"Wave {wave_id}: expected display_name '{expected_name}', "
            f"got '{display_name}'"
        )


def test_all_equity_waves_have_weights():
    """Test that all 9 equity waves have weights defined in WAVE_WEIGHTS."""
    for wave_id in EQUITY_WAVES:
        display_name = WAVE_ID_REGISTRY[wave_id]
        assert display_name in WAVE_WEIGHTS, (
            f"Wave {wave_id} ({display_name}) not found in WAVE_WEIGHTS"
        )
        
        # Check that weights are not empty
        holdings = WAVE_WEIGHTS[display_name]
        assert len(holdings) > 0, (
            f"Wave {wave_id} ({display_name}) has no holdings in WAVE_WEIGHTS"
        )


def test_ticker_discovery():
    """Test that ticker discovery works for all 9 equity waves."""
    for wave_id in EQUITY_WAVES:
        tickers = resolve_wave_tickers(wave_id)
        
        assert len(tickers) > 0, f"Wave {wave_id} has no tickers"
        assert all(isinstance(t, str) for t in tickers), (
            f"Wave {wave_id} has non-string tickers"
        )
        
        # Check no duplicates
        assert len(tickers) == len(set(tickers)), (
            f"Wave {wave_id} has duplicate tickers"
        )


def test_benchmark_definitions():
    """Test that all 9 equity waves have benchmark definitions."""
    for wave_id in EQUITY_WAVES:
        benchmarks = resolve_wave_benchmarks(wave_id)
        
        assert len(benchmarks) > 0, f"Wave {wave_id} has no benchmarks"
        
        # Check benchmark format: list of (ticker, weight) tuples
        for ticker, weight in benchmarks:
            assert isinstance(ticker, str), (
                f"Wave {wave_id} has non-string benchmark ticker: {ticker}"
            )
            assert isinstance(weight, (int, float)), (
                f"Wave {wave_id} has non-numeric benchmark weight: {weight}"
            )
            assert 0 <= weight <= 1, (
                f"Wave {wave_id} has invalid benchmark weight: {weight}"
            )
        
        # Check benchmark weights sum to ~1.0
        total_weight = sum(w for _, w in benchmarks)
        assert abs(total_weight - 1.0) < 0.01, (
            f"Wave {wave_id} benchmark weights sum to {total_weight}, expected 1.0"
        )


def test_wave_registry_csv():
    """Test that all 9 equity waves are in the CSV registry."""
    registry = get_wave_registry()
    assert not registry.empty, "Wave registry is empty"
    
    for wave_id in EQUITY_WAVES:
        wave_rows = registry[registry['wave_id'] == wave_id]
        assert not wave_rows.empty, f"Wave {wave_id} not found in wave_registry.csv"
        
        wave = wave_rows.iloc[0]
        
        # Check active status
        assert wave['active'] == True, f"Wave {wave_id} is not active"
        
        # Check category is equity
        category = str(wave['category']).lower()
        assert 'equity' in category, (
            f"Wave {wave_id} has non-equity category: {wave['category']}"
        )
        
        # Check benchmark_spec exists
        assert not pd.isna(wave['benchmark_spec']), (
            f"Wave {wave_id} missing benchmark_spec"
        )


def test_positions_files_exist():
    """Test that all 9 equity waves have positions.csv files."""
    for wave_id in EQUITY_WAVES:
        positions_path = f"data/waves/{wave_id}/positions.csv"
        assert os.path.exists(positions_path), (
            f"Positions file not found for {wave_id}: {positions_path}"
        )
        
        # Try to load positions file
        df = pd.read_csv(positions_path)
        assert 'ticker' in df.columns, (
            f"Wave {wave_id} positions file missing 'ticker' column"
        )
        assert 'weight' in df.columns, (
            f"Wave {wave_id} positions file missing 'weight' column"
        )
        
        # Check weights sum to 1.0
        total_weight = df['weight'].sum()
        assert abs(total_weight - 1.0) < 0.01, (
            f"Wave {wave_id} weights sum to {total_weight}, expected 1.0"
        )


def test_ticker_collection_complete():
    """Test that we can collect all tickers from all 9 equity waves."""
    all_tickers = set()
    
    for wave_id in EQUITY_WAVES:
        tickers = resolve_wave_tickers(wave_id)
        all_tickers.update(tickers)
    
    # Should have at least 50 unique tickers across all waves
    assert len(all_tickers) >= 50, (
        f"Expected at least 50 unique tickers, found {len(all_tickers)}"
    )
    
    print(f"\n✓ Total unique tickers across 9 equity waves: {len(all_tickers)}")


def test_get_all_wave_ids_includes_equity_waves():
    """Test that get_all_wave_ids() returns all 9 equity waves."""
    all_wave_ids = get_all_wave_ids()
    
    for wave_id in EQUITY_WAVES:
        assert wave_id in all_wave_ids, (
            f"Wave {wave_id} not returned by get_all_wave_ids()"
        )


def test_active_waves_includes_equity_waves():
    """Test that get_active_waves() includes all 9 equity waves."""
    active_waves = get_active_waves()
    assert not active_waves.empty, "No active waves found"
    
    active_wave_ids = set(active_waves['wave_id'].tolist())
    
    for wave_id in EQUITY_WAVES:
        assert wave_id in active_wave_ids, (
            f"Wave {wave_id} not in active waves"
        )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

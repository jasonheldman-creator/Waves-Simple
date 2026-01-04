"""
Test ROUND 7 Features Implementation

This test validates all 6 phases of ROUND 7 features:
1. Canonical Wave Universe (dynamic wave count)
2. Dual-Source Metrics Resolution
3. Snapshot Auto-Build and Staleness Management
4. Circuit Breaker Enforcement
5. Wave Monitor Tab (UI component - tested via app)
6. Enhanced Diagnostics (UI component - tested via app)
"""

def test_phase_1_wave_universe():
    """Test Phase 1: Canonical Wave Universe"""
    from waves_engine import get_all_waves_universe, WAVE_ID_REGISTRY
    
    universe = get_all_waves_universe()
    
    # Dynamic expected count from registry
    expected_count = len(WAVE_ID_REGISTRY)
    
    # Verify universe structure
    assert 'waves' in universe, "Universe missing 'waves' key"
    assert 'wave_ids' in universe, "Universe missing 'wave_ids' key"
    assert 'count' in universe, "Universe missing 'count' key"
    
    # Verify count matches registry (dynamic)
    assert universe['count'] == expected_count, f"Expected {expected_count} waves, got {universe['count']}"
    assert len(universe['waves']) == expected_count, f"Expected {expected_count} display names, got {len(universe['waves'])}"
    assert len(universe['wave_ids']) == expected_count, f"Expected {expected_count} wave_ids, got {len(universe['wave_ids'])}"
    
    print(f"✅ Phase 1: Wave Universe validated ({expected_count} waves from WAVE_ID_REGISTRY)")


def test_phase_2_metrics_resolution():
    """Test Phase 2: Dual-Source Metrics Resolution"""
    from analytics_pipeline import resolve_wave_metrics
    
    # Test with valid wave_id
    metrics = resolve_wave_metrics('sp500_wave', mode='Standard')
    
    # Verify all required fields exist
    required_fields = [
        'source', 'wave_id', 'display_name', 'mode',
        'return_1d', 'return_30d', 'return_60d', 'return_365d',
        'benchmark_return_1d', 'benchmark_return_30d', 'benchmark_return_60d', 'benchmark_return_365d',
        'alpha_1d', 'alpha_30d', 'alpha_60d', 'alpha_365d',
        'exposure_pct', 'cash_pct',
        'readiness_status', 'coverage_pct', 'coverage_score',
        'has_prices', 'has_benchmark', 'has_nav',
        'alerts', 'degradation_cause'
    ]
    
    for field in required_fields:
        assert field in metrics, f"Metrics missing required field: {field}"
    
    # Verify source is one of expected values
    assert metrics['source'] in ['pipeline', 'snapshot', 'degraded'], \
        f"Invalid source: {metrics['source']}"
    
    print(f"✅ Phase 2: Metrics Resolution validated (source: {metrics['source']})")


def test_phase_3_snapshot_management():
    """Test Phase 3: Snapshot Auto-Build and Staleness Management"""
    from analytics_pipeline import ensure_live_snapshot_exists
    import os
    
    # Test snapshot check
    status = ensure_live_snapshot_exists(
        path='data/live_snapshot.csv',
        max_age_minutes=15,
        force_rebuild=False
    )
    
    # Verify status structure
    assert 'exists' in status, "Status missing 'exists' key"
    assert 'fresh' in status, "Status missing 'fresh' key"
    assert 'age_minutes' in status, "Status missing 'age_minutes' key"
    assert 'rebuilt' in status, "Status missing 'rebuilt' key"
    
    # Verify snapshot file exists
    assert os.path.exists('data/live_snapshot.csv'), "Snapshot file does not exist"
    
    print(f"✅ Phase 3: Snapshot Management validated (exists={status['exists']}, fresh={status['fresh']})")


def test_phase_4_circuit_breaker_constants():
    """Test Phase 4: Circuit Breaker Constants"""
    from analytics_pipeline import (
        MAX_BATCH_RETRIES,
        MAX_TICKER_RETRIES,
        MAX_TOTAL_TICKERS_PER_RUN,
        MAX_WAVE_COMPUTE_SECONDS
    )
    
    # Verify constants are set correctly
    assert MAX_BATCH_RETRIES == 2, f"Expected MAX_BATCH_RETRIES=2, got {MAX_BATCH_RETRIES}"
    assert MAX_TICKER_RETRIES == 1, f"Expected MAX_TICKER_RETRIES=1, got {MAX_TICKER_RETRIES}"
    assert MAX_TOTAL_TICKERS_PER_RUN == 250, f"Expected MAX_TOTAL_TICKERS_PER_RUN=250, got {MAX_TOTAL_TICKERS_PER_RUN}"
    assert MAX_WAVE_COMPUTE_SECONDS == 8, f"Expected MAX_WAVE_COMPUTE_SECONDS=8, got {MAX_WAVE_COMPUTE_SECONDS}"
    
    print("✅ Phase 4: Circuit Breaker Constants validated")


def test_phase_5_wave_monitor_function():
    """Test Phase 5: Wave Monitor Tab Function Exists"""
    # Check function exists in app.py (without importing which requires Streamlit)
    import os
    
    assert os.path.exists('app.py'), "app.py not found"
    
    with open('app.py', 'r') as f:
        app_content = f.read()
    
    assert 'def render_wave_monitor_tab():' in app_content, \
        "render_wave_monitor_tab function not found in app.py"
    
    assert 'Wave Monitor' in app_content, \
        "Wave Monitor tab not referenced in app.py"
    
    print("✅ Phase 5: Wave Monitor Tab function exists in app.py")


def test_all_phases():
    """Run all phase tests"""
    print("\n" + "=" * 70)
    print("ROUND 7 Features Validation")
    print("=" * 70 + "\n")
    
    try:
        test_phase_1_wave_universe()
        test_phase_2_metrics_resolution()
        test_phase_3_snapshot_management()
        test_phase_4_circuit_breaker_constants()
        test_phase_5_wave_monitor_function()
        
        print("\n" + "=" * 70)
        print("✅ ALL ROUND 7 PHASES VALIDATED SUCCESSFULLY")
        print("=" * 70 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    test_all_phases()

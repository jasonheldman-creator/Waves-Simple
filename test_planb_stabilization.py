"""
test_planb_stabilization.py

Test script for Plan B Round 1 Stabilization features:
- Circuit breakers
- Build locks
- Timeouts
- Snapshot-first rendering
- Safe Mode
"""

import os
import sys
import time
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planb_proxy_pipeline import (
    MAX_RETRIES_PER_TICKER,
    TICKER_TIMEOUT_SECONDS,
    BUILD_TIMEOUT_SECONDS,
    BUILD_LOCK_MINUTES,
    OUTPUT_SNAPSHOT_PATH,
    DIAGNOSTICS_PATH,
    should_trigger_build,
    get_snapshot_freshness,
    load_proxy_snapshot,
    load_diagnostics
)


def test_constants():
    """Test that all constants are set correctly."""
    print("=" * 60)
    print("TEST 1: Circuit Breaker Constants")
    print("=" * 60)
    
    assert MAX_RETRIES_PER_TICKER == 1, f"Expected MAX_RETRIES_PER_TICKER=1, got {MAX_RETRIES_PER_TICKER}"
    print(f"✅ MAX_RETRIES_PER_TICKER = {MAX_RETRIES_PER_TICKER}")
    
    assert TICKER_TIMEOUT_SECONDS == 15, f"Expected TICKER_TIMEOUT_SECONDS=15, got {TICKER_TIMEOUT_SECONDS}"
    print(f"✅ TICKER_TIMEOUT_SECONDS = {TICKER_TIMEOUT_SECONDS}")
    
    assert BUILD_TIMEOUT_SECONDS == 15, f"Expected BUILD_TIMEOUT_SECONDS=15, got {BUILD_TIMEOUT_SECONDS}"
    print(f"✅ BUILD_TIMEOUT_SECONDS = {BUILD_TIMEOUT_SECONDS}")
    
    assert BUILD_LOCK_MINUTES == 2, f"Expected BUILD_LOCK_MINUTES=2, got {BUILD_LOCK_MINUTES}"
    print(f"✅ BUILD_LOCK_MINUTES = {BUILD_LOCK_MINUTES}")
    
    assert OUTPUT_SNAPSHOT_PATH == "data/live_proxy_snapshot.csv", f"Expected data/ path, got {OUTPUT_SNAPSHOT_PATH}"
    print(f"✅ OUTPUT_SNAPSHOT_PATH = {OUTPUT_SNAPSHOT_PATH}")
    
    assert DIAGNOSTICS_PATH == "data/planb_diagnostics_run.json", f"Expected data/ path, got {DIAGNOSTICS_PATH}"
    print(f"✅ DIAGNOSTICS_PATH = {DIAGNOSTICS_PATH}")
    
    print()


def test_build_lock():
    """Test build lock mechanism."""
    print("=" * 60)
    print("TEST 2: Build Lock Mechanism")
    print("=" * 60)
    
    # Test with no session state (first run)
    should_build, reason = should_trigger_build()
    print(f"No session state:")
    print(f"  Should build: {should_build}")
    print(f"  Reason: {reason}")
    
    # Test with recent build attempt (within lock period)
    mock_session = {'planb_last_build_attempt': datetime.now()}
    should_build, reason = should_trigger_build(mock_session)
    assert not should_build, "Build should be suppressed within lock period"
    assert "Build suppressed" in reason, f"Expected suppression message, got: {reason}"
    print(f"\nRecent build (0m ago):")
    print(f"  Should build: {should_build}")
    print(f"  Reason: {reason}")
    print(f"✅ Build correctly suppressed")
    
    # Test with old build attempt (outside lock period)
    old_time = datetime.now() - timedelta(minutes=3)
    mock_session = {'planb_last_build_attempt': old_time}
    should_build, reason = should_trigger_build(mock_session)
    print(f"\nOld build (3m ago):")
    print(f"  Should build: {should_build}")
    print(f"  Reason: {reason}")
    
    print()


def test_snapshot_first_rendering():
    """Test snapshot-first rendering."""
    print("=" * 60)
    print("TEST 3: Snapshot-First Rendering")
    print("=" * 60)
    
    # Check if snapshot exists
    freshness = get_snapshot_freshness()
    print(f"Snapshot exists: {freshness['exists']}")
    
    if freshness['exists']:
        print(f"  Age: {freshness['age_minutes']:.1f} minutes")
        print(f"  Fresh: {freshness['fresh']}")
        print(f"  Stale: {freshness.get('stale', False)}")
        
        # Load snapshot
        snapshot = load_proxy_snapshot()
        assert not snapshot.empty, "Snapshot should not be empty"
        assert len(snapshot) == 28, f"Expected 28 waves, got {len(snapshot)}"
        print(f"✅ Loaded snapshot with {len(snapshot)} waves")
        
        # Check required columns
        required_cols = ['wave_id', 'display_name', 'category', 'proxy_ticker', 'confidence']
        for col in required_cols:
            assert col in snapshot.columns, f"Missing required column: {col}"
        print(f"✅ All required columns present")
        
        # Check confidence distribution
        confidence_counts = snapshot['confidence'].value_counts()
        print(f"\nConfidence distribution:")
        for conf, count in confidence_counts.items():
            print(f"  {conf}: {count}")
    else:
        print("⚠️  No snapshot found (expected for first run)")
        print("   Snapshot-first rendering will show empty state")
    
    print()


def test_diagnostics():
    """Test diagnostics loading."""
    print("=" * 60)
    print("TEST 4: Diagnostics")
    print("=" * 60)
    
    diagnostics = load_diagnostics()
    
    if diagnostics:
        print(f"Diagnostics loaded:")
        print(f"  Timestamp: {diagnostics.get('timestamp', 'N/A')}")
        print(f"  Total waves: {diagnostics.get('total_waves', 'N/A')}")
        print(f"  Successful fetches: {diagnostics.get('successful_fetches', 'N/A')}")
        print(f"  Failed fetches: {diagnostics.get('failed_fetches', 'N/A')}")
        print(f"  Timeout exceeded: {diagnostics.get('timeout_exceeded', False)}")
        print(f"  Build duration: {diagnostics.get('build_duration_seconds', 'N/A')}s")
        print(f"✅ Diagnostics loaded successfully")
    else:
        print("⚠️  No diagnostics file found (expected for first run)")
    
    print()


def test_file_paths():
    """Test that file paths are correct."""
    print("=" * 60)
    print("TEST 5: File Paths")
    print("=" * 60)
    
    # Check that paths use data/ directory
    assert OUTPUT_SNAPSHOT_PATH.startswith("data/"), "Snapshot should be in data/ directory"
    print(f"✅ Snapshot path: {OUTPUT_SNAPSHOT_PATH}")
    
    assert DIAGNOSTICS_PATH.startswith("data/"), "Diagnostics should be in data/ directory"
    print(f"✅ Diagnostics path: {DIAGNOSTICS_PATH}")
    
    # Check data directory exists
    data_dir = os.path.dirname(OUTPUT_SNAPSHOT_PATH)
    if os.path.exists(data_dir):
        print(f"✅ Data directory exists: {data_dir}")
    else:
        print(f"⚠️  Data directory will be created on first build: {data_dir}")
    
    print()


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PLAN B STABILIZATION TESTS")
    print("=" * 60)
    print()
    
    try:
        test_constants()
        test_build_lock()
        test_snapshot_first_rendering()
        test_diagnostics()
        test_file_paths()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print()
        
        return 0
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"❌ TEST FAILED: {str(e)}")
        print("=" * 60)
        print()
        return 1
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ ERROR: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        print()
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

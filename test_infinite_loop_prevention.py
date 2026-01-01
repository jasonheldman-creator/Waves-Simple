"""
Test Infinite Loop Prevention Mechanisms

This test validates that the changes made to prevent infinite reruns are working correctly.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_auto_refresh_config_defaults():
    """Test that auto-refresh is disabled by default."""
    from auto_refresh_config import DEFAULT_AUTO_REFRESH_ENABLED
    
    assert DEFAULT_AUTO_REFRESH_ENABLED == False, \
        "Auto-refresh should be disabled by default"
    print("✅ Auto-refresh is disabled by default")


def test_compute_gate_no_auto_rebuild():
    """Test that compute gate prevents automatic rebuilds on stale snapshots."""
    from helpers.compute_gate import should_allow_build
    
    # Mock session state with safe mode ON (default)
    mock_session_state = {
        "safe_mode_no_fetch": True,
        "run_id": 1
    }
    
    # Test that auto-build is blocked when safe mode is ON
    should_build, reason = should_allow_build(
        snapshot_path="data/nonexistent.csv",
        session_state=mock_session_state,
        build_key="test",
        explicit_button_click=False
    )
    
    assert should_build == False, \
        f"Auto-build should be blocked when safe mode is ON, but got: {reason}"
    assert "Safe Mode active" in reason, \
        f"Reason should mention Safe Mode, got: {reason}"
    print(f"✅ Auto-rebuild blocked in safe mode: {reason}")
    
    # Test that explicit button click overrides safe mode
    should_build, reason = should_allow_build(
        snapshot_path="data/nonexistent.csv",
        session_state=mock_session_state,
        build_key="test",
        explicit_button_click=True
    )
    
    assert should_build == True, \
        f"Explicit button click should override safe mode, but got: {reason}"
    print(f"✅ Explicit button click allowed: {reason}")


def test_one_run_only_latch():
    """Test that ONE RUN ONLY latch is implemented in compute gate."""
    from helpers.compute_gate import should_allow_build
    
    # Mock session state with ONE RUN ONLY block active
    mock_session_state = {
        "safe_mode_no_fetch": False,
        "one_run_only_block": True,
        "run_id": 2
    }
    
    # Test that build is blocked by ONE RUN ONLY latch
    should_build, reason = should_allow_build(
        snapshot_path="data/nonexistent.csv",
        session_state=mock_session_state,
        build_key="test",
        explicit_button_click=False
    )
    
    assert should_build == False, \
        f"Build should be blocked by ONE RUN ONLY latch, but got: {reason}"
    assert "ONE RUN ONLY latch active" in reason, \
        f"Reason should mention ONE RUN ONLY latch, got: {reason}"
    print(f"✅ ONE RUN ONLY latch working: {reason}")


def test_stale_snapshot_no_auto_rebuild():
    """Test that stale snapshots don't trigger automatic rebuilds."""
    from helpers.compute_gate import should_allow_build
    import tempfile
    import os
    from datetime import datetime, timedelta
    
    # Create a temporary file with an old timestamp
    # Using delete=False because we need to modify the timestamp
    temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', text=True)
    
    try:
        # Write data to the file
        with os.fdopen(temp_fd, 'w') as f:
            f.write("test,data\n")
        
        # Set the file's modification time to 2 hours ago (stale)
        two_hours_ago = (datetime.now() - timedelta(hours=2)).timestamp()
        os.utime(temp_path, (two_hours_ago, two_hours_ago))
        
        # Mock session state
        mock_session_state = {
            "safe_mode_no_fetch": False,
            "run_id": 1
        }
        
        # Test that stale snapshot doesn't trigger auto-rebuild
        should_build, reason = should_allow_build(
            snapshot_path=temp_path,
            session_state=mock_session_state,
            build_key="test",
            explicit_button_click=False
        )
        
        assert should_build == False, \
            f"Stale snapshot should NOT trigger auto-rebuild, but got: {reason}"
        assert "stale" in reason.lower(), \
            f"Reason should mention staleness, got: {reason}"
        print(f"✅ Stale snapshot does not trigger auto-rebuild: {reason}")
        
        # Verify that snapshot_stale flag is set in session state
        assert mock_session_state.get("test_snapshot_stale") == True, \
            "Stale flag should be set in session state"
        print("✅ Stale flag set in session state")
        
    finally:
        # Cleanup - remove temp file
        try:
            os.unlink(temp_path)
        except OSError:
            pass  # Ignore errors if file was already deleted


def test_missing_snapshot_no_auto_rebuild():
    """Test that missing snapshots don't trigger automatic rebuilds."""
    from helpers.compute_gate import should_allow_build
    
    # Mock session state
    mock_session_state = {
        "safe_mode_no_fetch": False,
        "run_id": 1
    }
    
    # Test that missing snapshot doesn't trigger auto-rebuild
    should_build, reason = should_allow_build(
        snapshot_path="/tmp/definitely_does_not_exist_12345.csv",
        session_state=mock_session_state,
        build_key="test",
        explicit_button_click=False
    )
    
    assert should_build == False, \
        f"Missing snapshot should NOT trigger auto-rebuild, but got: {reason}"
    assert "missing" in reason.lower(), \
        f"Reason should mention missing snapshot, got: {reason}"
    print(f"✅ Missing snapshot does not trigger auto-rebuild: {reason}")


def test_check_stale_snapshot():
    """Test the check_stale_snapshot helper function."""
    from helpers.compute_gate import check_stale_snapshot
    import tempfile
    import os
    from datetime import datetime, timedelta
    
    # Create a temporary file with an old timestamp
    # Using delete=False because we need to modify the timestamp
    temp_fd, temp_path = tempfile.mkstemp(suffix='.csv', text=True)
    
    try:
        # Write data to the file
        with os.fdopen(temp_fd, 'w') as f:
            f.write("test,data\n")
        
        # Set the file's modification time to 2 hours ago (stale)
        two_hours_ago = (datetime.now() - timedelta(hours=2)).timestamp()
        os.utime(temp_path, (two_hours_ago, two_hours_ago))
        
        # Mock session state
        mock_session_state = {}
        
        # Check if snapshot is stale
        is_stale, age_minutes = check_stale_snapshot(
            temp_path,
            mock_session_state,
            build_key="test"
        )
        
        assert is_stale == True, \
            "2-hour old snapshot should be marked as stale"
        assert age_minutes > 60, \
            f"Age should be > 60 minutes, got {age_minutes}"
        print(f"✅ Stale snapshot detected correctly: {age_minutes:.1f} minutes old")
        
        # Verify session state was updated
        assert mock_session_state.get("test_snapshot_stale") == True
        assert mock_session_state.get("test_snapshot_age_minutes") == age_minutes
        print("✅ Session state updated with stale info")
        
    finally:
        # Cleanup - remove temp file
        try:
            os.unlink(temp_path)
        except OSError:
            pass  # Ignore errors if file was already deleted


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Infinite Loop Prevention Mechanisms")
    print("=" * 60)
    print()
    
    tests = [
        test_auto_refresh_config_defaults,
        test_compute_gate_no_auto_rebuild,
        test_one_run_only_latch,
        test_stale_snapshot_no_auto_rebuild,
        test_missing_snapshot_no_auto_rebuild,
        test_check_stale_snapshot,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"Running {test.__name__}...")
        try:
            test()
            passed += 1
            print()
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            print()
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
